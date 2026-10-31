import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import copy
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.dataset import UITVSFCDataset, Collator
from src.sampler import ClassAwareBatchSampler
from src.model import PhoBertSentimentClassifier
from src.losses import compute_class_weights
from src.metrics import compute_all_metrics
from src.calibration import search_optimal_neutral_bias, apply_neutral_bias


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_model(model, dataloader, device):
    """
    Evaluate model, returning loss, metrics dict, logits numpy array, and predictions.
    """
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )

            loss = outputs["loss"]
            logits = outputs["logits"]

            total_loss += loss.item()
            num_batches += 1

            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.argmax(all_logits, axis=-1)

    avg_loss = total_loss / max(1, num_batches)
    metrics = compute_all_metrics(all_labels, all_preds)
    metrics["loss"] = float(avg_loss)

    return avg_loss, metrics, all_logits, all_preds, all_labels


def run_training_experiment(
    config,
    train_dataset,
    val_dataset,
    test_dataset,
    tokenizer,
    device,
    save_checkpoint=True,
    output_dir="outputs",
):
    """
    Execute a single training run for a given configuration and seed.
    """
    seed = config.get("seed", 42)
    set_seed(seed)

    exp_name = config.get("name", "experiment")
    batch_size = config.get("batch_size", 32)
    max_length = config.get("max_length", 96)
    lr = float(config.get("lr", 2e-5))
    weight_decay = float(config.get("weight_decay", 0.01))
    epochs = config.get("epochs", 5)
    loss_type = config.get("loss_type", "ce")
    use_sampler = config.get("use_sampler", False)
    lambda_contrast = float(config.get("lambda_contrastive", 0.1))
    margin = float(config.get("margin", 0.2))
    use_val_calibration = config.get("use_val_calibration", False)
    model_path = config.get("model_path", "vinai/phobert-base")

    collator = Collator(tokenizer=tokenizer, max_length=max_length)

    # Setup train DataLoader
    if use_sampler:
        train_labels = train_dataset.get_labels()
        batch_sampler = ClassAwareBatchSampler(
            labels=train_labels,
            batch_size=batch_size,
            seed=seed,
            shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collator
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator
        )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    # Class weights for B2 & Class counts for Balanced Softmax
    train_labels_all = train_dataset.get_labels()
    class_counts = torch.bincount(torch.tensor(train_labels_all, dtype=torch.long), minlength=3).tolist()
    class_weights = None
    if loss_type in ["weighted_ce", "weighted_supcon"]:
        class_weights = compute_class_weights(train_labels_all, num_classes=3).to(device)

    # Model
    gamma = float(config.get("gamma", 0.5))
    model = PhoBertSentimentClassifier(
        model_name_or_path=model_path,
        num_classes=3,
        loss_type=loss_type,
        class_weights=class_weights,
        class_counts=class_counts,
        lambda_contrastive=lambda_contrast,
        margin=margin,
        gamma=gamma,
    ).to(device)

    # Optimizer & Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, foreach=False)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Training state tracking
    best_val_macro = -1.0
    best_model_state = None
    best_epoch = -1
    history = []

    print(f"\n[{exp_name} | Seed {seed}] Starting training: {epochs} epochs, loss_type={loss_type}, sampler={use_sampler}")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        if use_sampler and hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        train_ce_loss = 0.0
        train_contrast_loss = 0.0
        n_train_batches = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                loss = outputs["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            if outputs["ce_loss"] is not None:
                train_ce_loss += outputs["ce_loss"].item()
            if outputs["contrast_loss"] is not None:
                train_contrast_loss += outputs["contrast_loss"].item()
            n_train_batches += 1

        avg_train_loss = train_loss / max(1, n_train_batches)
        avg_ce = train_ce_loss / max(1, n_train_batches)
        avg_cnt = train_contrast_loss / max(1, n_train_batches)

        # Validation at epoch end
        val_loss, val_metrics, val_logits, val_preds, val_labels = evaluate_model(model, val_loader, device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_ce_loss": avg_ce,
            "train_contrast_loss": avg_cnt,
            "val_loss": val_loss,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_neutral_f1": val_metrics["neutral"]["f1"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(epoch_record)

        print(
            f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Macro-F1: {val_metrics['macro_f1']:.4f} | "
            f"Val Neu-F1: {val_metrics['neutral']['f1']:.4f}"
        )

        # Checkpoint selection strictly based on Validation Macro-F1
        if val_metrics["macro_f1"] > best_val_macro:
            best_val_macro = val_metrics["macro_f1"]
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    elapsed_time = time.time() - t_start
    print(f"[{exp_name} | Seed {seed}] Best Validation Macro-F1: {best_val_macro:.4f} at Epoch {best_epoch} (Time: {elapsed_time:.1f}s)")

    # Restore best checkpoint
    model.load_state_dict(best_model_state)

    # Re-evaluate best model on Validation
    _, best_val_metrics, best_val_logits, best_val_preds, best_val_labels = evaluate_model(model, val_loader, device)

    # Validation Calibration (if applicable: B3, B7)
    optimal_bias = 0.0
    val_calib_history = None
    if use_val_calibration:
        optimal_bias, calib_score, val_calib_history = search_optimal_neutral_bias(
            best_val_logits,
            best_val_labels,
            b_min=-1.5,
            b_max=1.5,
            step=0.05
        )
        print(f"[{exp_name} | Seed {seed}] Optimal Neutral Bias found on Validation: {optimal_bias:.2f} (Val Macro-F1: {best_val_macro:.4f} -> {calib_score:.4f})")
        # Update val metrics post-calibration
        calib_val_logits = apply_neutral_bias(best_val_logits, optimal_bias)
        calib_val_preds = np.argmax(calib_val_logits, axis=-1)
        best_val_metrics = compute_all_metrics(best_val_labels, calib_val_preds)
        best_val_metrics["optimal_neutral_bias"] = optimal_bias

    # FINAL EVALUATION ON TEST SET (STRICTLY ONCE)
    test_loss, test_metrics, test_logits, test_preds, test_labels = evaluate_model(model, test_loader, device)

    # If calibrated, apply optimal validation bias to test logits
    if use_val_calibration and optimal_bias != 0.0:
        calib_test_logits = apply_neutral_bias(test_logits, optimal_bias)
        calib_test_preds = np.argmax(calib_test_logits, axis=-1)
        test_metrics = compute_all_metrics(test_labels, calib_test_preds)
        test_metrics["applied_neutral_bias"] = optimal_bias
        test_preds = calib_test_preds

    print(
        f"[{exp_name} | Seed {seed}] FINAL TEST RESULTS: Macro-F1 = {test_metrics['macro_f1']:.4f} | "
        f"Neu-F1 = {test_metrics['neutral']['f1']:.4f} | Neu-P = {test_metrics['neutral']['precision']:.4f} | "
        f"Neu-R = {test_metrics['neutral']['recall']:.4f} | Acc = {test_metrics['accuracy']:.4f}"
    )

    # Save artifacts
    exp_output_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_output_dir, exist_ok=True)

    result_data = {
        "config": config,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "training_time_seconds": elapsed_time,
        "history": history,
        "val_calibration_history": val_calib_history,
        "applied_neutral_bias": optimal_bias,
    }

    result_json_path = os.path.join(exp_output_dir, f"result_seed_{seed}.json")
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    # Save predictions
    pred_path = os.path.join(exp_output_dir, f"preds_seed_{seed}.npz")
    np.savez_compressed(
        pred_path,
        test_labels=test_labels,
        test_preds=test_preds,
        test_logits=test_logits,
        val_labels=best_val_labels,
        val_logits=best_val_logits,
    )

    return result_data
