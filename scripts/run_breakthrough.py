import os
import sys
import gc
import time
import json

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


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    print("=" * 70)
    print("BREAKTHROUGH EXPERIMENT SUITE: WEIGHTED-SUPCON & BALANCED-SUPCON (B8 - B10)")
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
    output_dir = "outputs"

    suite_start = time.time()

    # -------------------------------------------------------------
    # 1. B8: PhoBERT + Balanced Softmax (Smoothed gamma=0.5) + SupCon + Sampler
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(">>> [1/3] Training B8: PhoBERT + Balanced-SupCon (Smoothed gamma=0.5)...")
    print("=" * 60)
    cleanup()
    config_b8 = {
        "name": "B8_balanced_supcon",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": 0.01,
        "max_length": 96,
        "model_path": model_path,
        "loss_type": "balanced_supcon",
        "use_sampler": True,
        "lambda_contrastive": 0.1,
        "temperature": 0.1,
        "gamma": 0.5,
        "use_val_calibration": False,
    }
    res_b8 = run_training_experiment(
        config=config_b8,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        device=device,
        output_dir=output_dir,
    )
    print(f"B8 Test Macro-F1: {res_b8['test_metrics']['macro_f1']:.4f} | Neu-F1: {res_b8['test_metrics']['neutral']['f1']:.4f}")

    # -------------------------------------------------------------
    # 2. B9: PhoBERT + Weighted-Cross-Entropy + Supervised Contrastive (SupCon) + Sampler
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(">>> [2/3] Training B9: PhoBERT + Weighted-SupCon (Cost-Sensitive + Contrastive)...")
    print("=" * 60)
    cleanup()
    config_b9 = {
        "name": "B9_weighted_supcon",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": 0.01,
        "max_length": 96,
        "model_path": model_path,
        "loss_type": "weighted_supcon",
        "use_sampler": True,
        "lambda_contrastive": 0.1,
        "temperature": 0.1,
        "use_val_calibration": False,
    }
    res_b9 = run_training_experiment(
        config=config_b9,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        device=device,
        output_dir=output_dir,
    )
    print(f"B9 Test Macro-F1: {res_b9['test_metrics']['macro_f1']:.4f} | Neu-F1: {res_b9['test_metrics']['neutral']['f1']:.4f}")

    # -------------------------------------------------------------
    # 3. B10: B9 + Validation Logit Calibration
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(">>> [3/3] Calibrating B10: B9 + Validation Logit Calibration...")
    print("=" * 60)
    b9_npz = np.load(f"{output_dir}/B9_weighted_supcon/preds_seed_{seed}.npz")
    val_labels_9 = b9_npz["val_labels"]
    val_logits_9 = b9_npz["val_logits"]
    test_labels_9 = b9_npz["test_labels"]
    test_logits_9 = b9_npz["test_logits"]

    best_bias_10, best_val_score_10, grid_history_10 = search_optimal_neutral_bias(
        val_logits_9, val_labels_9, b_min=-1.5, b_max=1.5, step=0.05
    )
    print(f"Optimal neutral bias for B10: {best_bias_10:+.2f} (Val Macro-F1: {best_val_score_10:.4f})")

    calib_val_logits_10 = apply_neutral_bias(val_logits_9, best_bias_10)
    val_preds_10 = np.argmax(calib_val_logits_10, axis=-1)
    val_metrics_10 = compute_all_metrics(val_labels_9, val_preds_10)
    val_metrics_10["optimal_neutral_bias"] = best_bias_10

    calib_test_logits_10 = apply_neutral_bias(test_logits_9, best_bias_10)
    test_preds_10 = np.argmax(calib_test_logits_10, axis=-1)
    test_metrics_10 = compute_all_metrics(test_labels_9, test_preds_10)
    test_metrics_10["applied_neutral_bias"] = best_bias_10

    os.makedirs(f"{output_dir}/B10_weighted_supcon_calib", exist_ok=True)
    res_b10_data = {
        "config": {
            "name": "B10_weighted_supcon_calib",
            "seed": seed,
            "epochs": epochs,
            "loss_type": "weighted_supcon",
            "use_sampler": True,
            "lambda_contrastive": 0.1,
            "temperature": 0.1,
            "use_val_calibration": True,
            "applied_neutral_bias": best_bias_10,
        },
        "seed": seed,
        "best_epoch": res_b9["best_epoch"],
        "best_val_metrics": val_metrics_10,
        "test_metrics": test_metrics_10,
        "training_time_seconds": res_b9["training_time_seconds"],
        "history": res_b9["history"],
        "val_calibration_history": grid_history_10,
        "applied_neutral_bias": best_bias_10,
    }
    with open(f"{output_dir}/B10_weighted_supcon_calib/result_seed_{seed}.json", "w", encoding="utf-8") as f:
        json.dump(res_b10_data, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        f"{output_dir}/B10_weighted_supcon_calib/preds_seed_{seed}.npz",
        test_labels=test_labels_9,
        test_preds=test_preds_10,
        test_logits=calib_test_logits_10,
        val_labels=val_labels_9,
        val_logits=calib_val_logits_10,
    )
    print(f"B10 Test Results: Macro-F1 = {test_metrics_10['macro_f1']:.4f} | Neu-F1 = {test_metrics_10['neutral']['f1']:.4f}")

    total_time = time.time() - suite_start
    print("\n" + "=" * 70)
    print(f"BREAKTHROUGH EXPERIMENT COMPLETED IN {total_time/60:.2f} MINUTES!")
    print("=" * 70)


if __name__ == "__main__":
    main()
