import os
import sys
import gc
import time
import json

# Ensure project root is in sys.path whether run from root or scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from transformers import AutoTokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from src.dataset import UITVSFCDataset
from src.trainer import run_training_experiment
from src.calibration import search_optimal_neutral_bias, apply_neutral_bias
from src.metrics import compute_all_metrics, compute_paired_bootstrap_ci


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    print("=" * 70)
    print("STANDARDIZED BENCHMARK SUITE: 3 EPOCHS FOR ALL CONFIGURATIONS (B1 - B7)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**2):.0f} MB)")

    cache_snapshot = "/home/lavender/.cache/huggingface/hub/models--vinai--phobert-base/snapshots/01daacda68afe13d83023d16ec647239e344a1e6"
    model_path = cache_snapshot if os.path.exists(cache_snapshot) else "vinai/phobert-base"

    print("Loading tokenizer and datasets...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_dataset = UITVSFCDataset("data/train.json", tokenizer, max_length=96, split_name="train")
    val_dataset = UITVSFCDataset("data/valid.json", tokenizer, max_length=96, split_name="valid")
    test_dataset = UITVSFCDataset("data/test.json", tokenizer, max_length=96, split_name="test")

    seed = 42
    epochs = 3
    batch_size = 32
    lr = 2e-5
    weight_decay = 0.01
    output_dir = "outputs"

    suite_start_time = time.time()

    # -------------------------------------------------------------
    # 1. B1: PhoBERT + Cross-Entropy (Random Sampler)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(">>> [1/7] Training B1: PhoBERT + CE (3 epochs)...")
    print("=" * 60)
    cleanup_cuda()
    config_b1 = {
        "name": "B1_ce",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
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
        output_dir=output_dir,
    )

    # -------------------------------------------------------------
    # 2. B2: PhoBERT + Class-Weighted Cross-Entropy
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(">>> [2/7] Training B2: PhoBERT + Weighted CE (3 epochs)...")
    print("=" * 60)
    cleanup_cuda()
    config_b2 = {
        "name": "B2_weighted_ce",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
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
        output_dir=output_dir,
    )

    # -------------------------------------------------------------
    # 3. B3: B1 + Validation Logit/Bias Calibration
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(">>> [3/7] Calibrating B3: PhoBERT + CE + Val Bias Calibration...")
    print("=" * 60)
    b1_npz = np.load(f"{output_dir}/B1_ce/preds_seed_{seed}.npz")
    val_labels = b1_npz["val_labels"]
    val_logits = b1_npz["val_logits"]
    test_labels = b1_npz["test_labels"]
    test_logits = b1_npz["test_logits"]

    best_bias_b3, best_val_score_b3, grid_history_b3 = search_optimal_neutral_bias(
        val_logits, val_labels, b_min=-1.5, b_max=1.5, step=0.05
    )
    print(f"Optimal neutral bias for B3: {best_bias_b3:+.2f} (Val Macro-F1: {best_val_score_b3:.4f})")

    calib_val_logits_b3 = apply_neutral_bias(val_logits, best_bias_b3)
    val_preds_b3 = np.argmax(calib_val_logits_b3, axis=-1)
    val_metrics_b3 = compute_all_metrics(val_labels, val_preds_b3)
    val_metrics_b3["optimal_neutral_bias"] = best_bias_b3

    calib_test_logits_b3 = apply_neutral_bias(test_logits, best_bias_b3)
    test_preds_b3 = np.argmax(calib_test_logits_b3, axis=-1)
    test_metrics_b3 = compute_all_metrics(test_labels, test_preds_b3)
    test_metrics_b3["applied_neutral_bias"] = best_bias_b3

    os.makedirs(f"{output_dir}/B3_ce_val_bias", exist_ok=True)
    res_b3_data = {
        "config": {
            "name": "B3_ce_val_bias",
            "seed": seed,
            "epochs": epochs,
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
        "val_calibration_history": grid_history_b3,
        "applied_neutral_bias": best_bias_b3,
    }
    with open(f"{output_dir}/B3_ce_val_bias/result_seed_{seed}.json", "w", encoding="utf-8") as f:
        json.dump(res_b3_data, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        f"{output_dir}/B3_ce_val_bias/preds_seed_{seed}.npz",
        test_labels=test_labels,
        test_preds=test_preds_b3,
        test_logits=calib_test_logits_b3,
        val_labels=val_labels,
        val_logits=calib_val_logits_b3,
    )
    print(f"B3 Final Test Results: Macro-F1 = {test_metrics_b3['macro_f1']:.4f} | Neu-F1 = {test_metrics_b3['neutral']['f1']:.4f}")

    # -------------------------------------------------------------
    # 4. B4: PhoBERT + CE + Class-Aware Batch Sampler
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(">>> [4/7] Training B4: PhoBERT + CE + Sampler (3 epochs)...")
    print("=" * 60)
    cleanup_cuda()
    config_b4 = {
        "name": "B4_ce_sampler",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
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
        output_dir=output_dir,
    )

    # -------------------------------------------------------------
    # 5. B5: PhoBERT + SupCon + Class-Aware Batch Sampler
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(">>> [5/7] Training B5: PhoBERT + SupCon + Sampler (3 epochs)...")
    print("=" * 60)
    cleanup_cuda()
    config_b5 = {
        "name": "B5_supcon",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
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
        output_dir=output_dir,
    )

    # -------------------------------------------------------------
    # 6. B6: PhoBERT + Neutral Boundary Loss (NBAC) + Sampler
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(">>> [6/7] Training B6: PhoBERT + NBAC + Sampler (3 epochs)...")
    print("=" * 60)
    cleanup_cuda()
    config_b6 = {
        "name": "B6_nbac",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "max_length": 96,
        "model_path": model_path,
        "loss_type": "nbac",
        "use_sampler": True,
        "lambda_contrastive": 0.1,
        "margin": 0.2,
        "positive_mode": "hardest",
        "use_val_calibration": False,
    }
    res_b6 = run_training_experiment(
        config=config_b6,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        device=device,
        output_dir=output_dir,
    )

    # -------------------------------------------------------------
    # 7. B7: B6 + Validation Logit/Bias Calibration
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(">>> [7/7] Calibrating B7: B6 (NBAC) + Val Bias Calibration...")
    print("=" * 60)
    b6_npz = np.load(f"{output_dir}/B6_nbac/preds_seed_{seed}.npz")
    val_labels_6 = b6_npz["val_labels"]
    val_logits_6 = b6_npz["val_logits"]
    test_labels_6 = b6_npz["test_labels"]
    test_logits_6 = b6_npz["test_logits"]

    best_bias_b7, best_val_score_b7, grid_history_b7 = search_optimal_neutral_bias(
        val_logits_6, val_labels_6, b_min=-1.5, b_max=1.5, step=0.05
    )
    print(f"Optimal neutral bias for B7: {best_bias_b7:+.2f} (Val Macro-F1: {best_val_score_b7:.4f})")

    calib_val_logits_b7 = apply_neutral_bias(val_logits_6, best_bias_b7)
    val_preds_b7 = np.argmax(calib_val_logits_b7, axis=-1)
    val_metrics_b7 = compute_all_metrics(val_labels_6, val_preds_b7)
    val_metrics_b7["optimal_neutral_bias"] = best_bias_b7

    calib_test_logits_b7 = apply_neutral_bias(test_logits_6, best_bias_b7)
    test_preds_b7 = np.argmax(calib_test_logits_b7, axis=-1)
    test_metrics_b7 = compute_all_metrics(test_labels_6, test_preds_b7)
    test_metrics_b7["applied_neutral_bias"] = best_bias_b7

    os.makedirs(f"{output_dir}/B7_nbac_val_bias", exist_ok=True)
    res_b7_data = {
        "config": {
            "name": "B7_nbac_val_bias",
            "seed": seed,
            "epochs": epochs,
            "loss_type": "nbac",
            "use_sampler": True,
            "lambda_contrastive": 0.1,
            "margin": 0.2,
            "use_val_calibration": True,
            "applied_neutral_bias": best_bias_b7,
        },
        "seed": seed,
        "best_epoch": res_b6["best_epoch"],
        "best_val_metrics": val_metrics_b7,
        "test_metrics": test_metrics_b7,
        "training_time_seconds": res_b6["training_time_seconds"],
        "history": res_b6["history"],
        "val_calibration_history": grid_history_b7,
        "applied_neutral_bias": best_bias_b7,
    }
    with open(f"{output_dir}/B7_nbac_val_bias/result_seed_{seed}.json", "w", encoding="utf-8") as f:
        json.dump(res_b7_data, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        f"{output_dir}/B7_nbac_val_bias/preds_seed_{seed}.npz",
        test_labels=test_labels_6,
        test_preds=test_preds_b7,
        test_logits=calib_test_logits_b7,
        val_labels=val_labels_6,
        val_logits=calib_val_logits_b7,
    )
    print(f"B7 Final Test Results: Macro-F1 = {test_metrics_b7['macro_f1']:.4f} | Neu-F1 = {test_metrics_b7['neutral']['f1']:.4f}")

    total_time = time.time() - suite_start_time
    print("\n" + "=" * 70)
    print(f"STANDARDIZED 3-EPOCH SUITE FINISHED IN {total_time/60:.2f} MINUTES!")
    print("=" * 70)


if __name__ == "__main__":
    main()
