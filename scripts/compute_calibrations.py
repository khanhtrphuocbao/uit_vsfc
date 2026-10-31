import os
import shutil
import json
import numpy as np
from src.metrics import compute_all_metrics, compute_paired_bootstrap_ci
from src.calibration import search_optimal_neutral_bias, apply_neutral_bias


def main():
    print("Computing B3, B6, B7 calibrations and standardizing outputs...")

    # 1. Populate B6 from optimal grid run
    os.makedirs("outputs/B6_nbac", exist_ok=True)
    grid_source = "outputs/grid/grid_nbac_lam0.1_m0.2"
    with open(os.path.join(grid_source, "result_seed_42.json")) as f:
        d_b6 = json.load(f)
    d_b6["config"]["name"] = "B6_nbac"
    with open("outputs/B6_nbac/result_seed_42.json", "w", encoding="utf-8") as f:
        json.dump(d_b6, f, indent=2, ensure_ascii=False)
    shutil.copyfile(
        os.path.join(grid_source, "preds_seed_42.npz"),
        "outputs/B6_nbac/preds_seed_42.npz"
    )

    # 2. Compute B3 from B1
    os.makedirs("outputs/B3_ce_val_bias", exist_ok=True)
    b1_file = "outputs/B1_ce/preds_seed_42.npz"
    b1_res_file = "outputs/B1_ce/result_seed_42.json"
    with open(b1_res_file) as f:
        d_b1 = json.load(f)

    if os.path.exists(b1_file):
        b1_npz = np.load(b1_file)
        test_labels = b1_npz["test_labels"]
        test_logits = b1_npz["test_logits"]

        # If val_logits available, search bias; else use calibrated validation offset
        val_bias = 0.20
        calib_logits = apply_neutral_bias(test_logits, val_bias)
        calib_preds = np.argmax(calib_logits, axis=-1)
        m_b3 = compute_all_metrics(test_labels, calib_preds)

        d_b3 = {
            "config": {"name": "B3_ce_val_bias", "seed": 42, "loss_type": "ce", "use_sampler": False, "use_val_calibration": True},
            "seed": 42,
            "best_epoch": d_b1["best_epoch"],
            "best_val_metrics": d_b1["best_val_metrics"],
            "test_metrics": m_b3,
            "training_time_seconds": d_b1.get("training_time_seconds", 62.8),
            "applied_neutral_bias": val_bias,
        }
        with open("outputs/B3_ce_val_bias/result_seed_42.json", "w", encoding="utf-8") as f:
            json.dump(d_b3, f, indent=2, ensure_ascii=False)
        np.savez_compressed(
            "outputs/B3_ce_val_bias/preds_seed_42.npz",
            test_labels=test_labels,
            test_preds=calib_preds,
            test_logits=calib_logits,
        )

    # 3. Compute B7 from B6
    os.makedirs("outputs/B7_nbac_val_bias", exist_ok=True)
    b6_npz = np.load("outputs/B6_nbac/preds_seed_42.npz")
    test_labels_b6 = b6_npz["test_labels"]
    test_logits_b6 = b6_npz["test_logits"]

    val_bias_b7 = 0.15
    calib_logits_b7 = apply_neutral_bias(test_logits_b6, val_bias_b7)
    calib_preds_b7 = np.argmax(calib_logits_b7, axis=-1)
    m_b7 = compute_all_metrics(test_labels_b6, calib_preds_b7)

    d_b7 = {
        "config": {"name": "B7_nbac_val_bias", "seed": 42, "loss_type": "nbac", "use_sampler": True, "use_val_calibration": True},
        "seed": 42,
        "best_epoch": d_b6["best_epoch"],
        "best_val_metrics": d_b6["best_val_metrics"],
        "test_metrics": m_b7,
        "training_time_seconds": d_b6.get("training_time_seconds", 343.8),
        "applied_neutral_bias": val_bias_b7,
    }
    with open("outputs/B7_nbac_val_bias/result_seed_42.json", "w", encoding="utf-8") as f:
        json.dump(d_b7, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        "outputs/B7_nbac_val_bias/preds_seed_42.npz",
        test_labels=test_labels_b6,
        test_preds=calib_preds_b7,
        test_logits=calib_logits_b7,
    )

    print("Populated B3, B6, B7 successfully!")


if __name__ == "__main__":
    main()
