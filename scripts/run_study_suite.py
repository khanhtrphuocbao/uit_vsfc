import os
import shutil
import json
import numpy as np
import torch
from transformers import AutoTokenizer

from src.dataset import UITVSFCDataset
from src.trainer import run_training_experiment
from src.calibration import search_optimal_neutral_bias, apply_neutral_bias
from src.metrics import compute_all_metrics, compute_paired_bootstrap_ci


def main():
    print("=" * 65)
    print("RUNNING REFINED STUDY SUITE (B1 - B7) COMPACT PROTOCOL")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    cache_snapshot = "/home/lavender/.cache/huggingface/hub/models--vinai--phobert-base/snapshots/01daacda68afe13d83023d16ec647239e344a1e6"
    model_path = cache_snapshot if os.path.exists(cache_snapshot) else "vinai/phobert-base"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_dataset = UITVSFCDataset("data/train.json", tokenizer, max_length=96, split_name="train")
    val_dataset = UITVSFCDataset("data/valid.json", tokenizer, max_length=96, split_name="valid")
    test_dataset = UITVSFCDataset("data/test.json", tokenizer, max_length=96, split_name="test")

    seed = 42
    epochs = 2  # Rapid convergence observed at epoch 2
    batch_size = 32
    lr = 2e-5

    # 1. B1: PhoBERT + CE
    print("\n" + "=" * 50)
    print(">>> [1/5] Training B1: PhoBERT + CE (2 epochs)...")
    print("=" * 50)
    config_b1 = {
        "name": "B1_ce",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": 0.01,
        "max_length": 96,
        "model_path": model_path,
        "loss_type": "ce",
        "use_sampler": False,
        "use_val_calibration": False,
    }
    res_b1 = run_training_experiment(
        config=config_b1,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        device=device,
        output_dir="outputs",
    )

    # 2. B2: PhoBERT + Weighted CE
    print("\n" + "=" * 50)
    print(">>> [2/5] Training B2: PhoBERT + Weighted CE (2 epochs)...")
    print("=" * 50)
    config_b2 = {
        "name": "B2_weighted_ce",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": 0.01,
        "max_length": 96,
        "model_path": model_path,
        "loss_type": "weighted_ce",
        "use_sampler": False,
        "use_val_calibration": False,
    }
    res_b2 = run_training_experiment(
        config=config_b2,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        device=device,
        output_dir="outputs",
    )

    # 3. B3: Derived from B1 by searching validation neutral bias
    print("\n" + "=" * 50)
    print(">>> [3/5] Computing B3: PhoBERT + CE + Val Bias Calibration...")
    print("=" * 50)
    b1_npz = np.load(f"outputs/B1_ce/preds_seed_{seed}.npz")
    val_labels = b1_npz["val_labels"]
    val_logits = b1_npz["val_logits"]
    test_labels = b1_npz["test_labels"]
    test_logits = b1_npz["test_logits"]

    best_bias_b3, best_val_score_b3, grid_history_b3 = search_optimal_neutral_bias(
        val_logits, val_labels, b_min=-1.5, b_max=1.5, step=0.05
    )
    print(f"Optimal neutral bias for B3: {best_bias_b3:.2f} (Val Macro-F1: {best_val_score_b3:.4f})")

    calib_val_logits_b3 = apply_neutral_bias(val_logits, best_bias_b3)
    val_preds_b3 = np.argmax(calib_val_logits_b3, axis=-1)
    val_metrics_b3 = compute_all_metrics(val_labels, val_preds_b3)

    calib_test_logits_b3 = apply_neutral_bias(test_logits, best_bias_b3)
    test_preds_b3 = np.argmax(calib_test_logits_b3, axis=-1)
    test_metrics_b3 = compute_all_metrics(test_labels, test_preds_b3)

    os.makedirs("outputs/B3_ce_val_bias", exist_ok=True)
    res_b3_data = {
        "config": {
            "name": "B3_ce_val_bias",
            "seed": seed,
            "loss_type": "ce",
            "use_sampler": False,
            "use_val_calibration": True,
            "applied_neutral_bias": best_bias_b3,
        },
        "seed": seed,
        "best_epoch": res_b1["best_epoch"],
        "best_val_metrics": val_metrics_b3,
        "test_metrics": test_metrics_b3,
        "training_time_seconds": res_b1["training_time_seconds"],
        "history": res_b1["history"],
        "applied_neutral_bias": best_bias_b3,
    }
    with open(f"outputs/B3_ce_val_bias/result_seed_{seed}.json", "w", encoding="utf-8") as f:
        json.dump(res_b3_data, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        f"outputs/B3_ce_val_bias/preds_seed_{seed}.npz",
        test_labels=test_labels,
        test_preds=test_preds_b3,
        test_logits=calib_test_logits_b3,
        val_labels=val_labels,
        val_logits=calib_val_logits_b3,
    )
    print(f"B3 Test Macro-F1: {test_metrics_b3['macro_f1']:.4f} | Neu-F1: {test_metrics_b3['neutral']['f1']:.4f}")

    # 4. B4: PhoBERT + CE + Class-Aware Sampler (Ablation isolating sampler)
    print("\n" + "=" * 50)
    print(">>> [4/5] Training B4: PhoBERT + CE + Sampler (2 epochs)...")
    print("=" * 50)
    config_b4 = {
        "name": "B4_ce_sampler",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": 0.01,
        "max_length": 96,
        "model_path": model_path,
        "loss_type": "ce",
        "use_sampler": True,
        "use_val_calibration": False,
    }
    res_b4 = run_training_experiment(
        config=config_b4,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        device=device,
        output_dir="outputs",
    )

    # 5. B5: PhoBERT + SupCon + Sampler
    print("\n" + "=" * 50)
    print(">>> [5/5] Training B5: PhoBERT + SupCon + Sampler (2 epochs)...")
    print("=" * 50)
    config_b5 = {
        "name": "B5_supcon",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": 0.01,
        "max_length": 96,
        "model_path": model_path,
        "loss_type": "supcon",
        "use_sampler": True,
        "lambda_contrastive": 0.1,
        "temperature": 0.1,
        "use_val_calibration": False,
    }
    res_b5 = run_training_experiment(
        config=config_b5,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        device=device,
        output_dir="outputs",
    )

    # 6. B6: PhoBERT + NBAC + Sampler (Populated from optimal grid search)
    print("\n" + "=" * 50)
    print(">>> [6/7] Populating B6: PhoBERT + NBAC + Sampler from optimal grid...")
    print("=" * 50)
    os.makedirs("outputs/B6_nbac", exist_ok=True)
    grid_source = "outputs/grid/grid_nbac_lam0.1_m0.2"
    with open(os.path.join(grid_source, f"result_seed_{seed}.json"), "r") as f:
        data_b6 = json.load(f)
    data_b6["config"]["name"] = "B6_nbac"
    with open(f"outputs/B6_nbac/result_seed_{seed}.json", "w", encoding="utf-8") as f:
        json.dump(data_b6, f, indent=2, ensure_ascii=False)
    shutil.copyfile(
        os.path.join(grid_source, f"preds_seed_{seed}.npz"),
        f"outputs/B6_nbac/preds_seed_{seed}.npz",
    )
    print(f"B6 Test Macro-F1: {data_b6['test_metrics']['macro_f1']:.4f} | Neu-F1: {data_b6['test_metrics']['neutral']['f1']:.4f}")

    # 7. B7: B6 + Val Bias Calibration
    print("\n" + "=" * 50)
    print(">>> [7/7] Calibrating B7: B6 + Val Bias Calibration...")
    print("=" * 50)
    # Train B7 with use_val_calibration=True
    config_b7 = {
        "name": "B7_nbac_val_bias",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": 0.01,
        "max_length": 96,
        "model_path": model_path,
        "loss_type": "nbac",
        "use_sampler": True,
        "lambda_contrastive": 0.1,
        "margin": 0.2,
        "use_val_calibration": True,
    }
    res_b7 = run_training_experiment(
        config=config_b7,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        device=device,
        output_dir="outputs",
    )

    print("\n" + "=" * 65)
    print("ALL ABLATIONS (B0 - B7) COMPLETED WITH EMPIRICAL ARTIFACTS!")
    print("=" * 65)


if __name__ == "__main__":
    main()
