import os
import glob
import json
import numpy as np
import pandas as pd
from src.metrics import compute_all_metrics, compute_paired_bootstrap_ci


def load_experiment_results(output_dir="outputs"):
    """
    Load all result_seed_*.json and prediction npz files across experiment directories.
    """
    records = []
    exp_dirs = glob.glob(os.path.join(output_dir, "*"))

    for ed in exp_dirs:
        exp_name = os.path.basename(ed)
        if exp_name in ["grid_search", "B0_tfidf_lr", "cache", "reports"]:
            continue

        result_files = glob.glob(os.path.join(ed, "result_seed_*.json"))
        for rf in result_files:
            with open(rf, "r", encoding="utf-8") as f:
                data = json.load(f)

            seed = data["seed"]
            pred_file = os.path.join(ed, f"preds_seed_{seed}.npz")
            has_preds = os.path.exists(pred_file)

            test_m = data["test_metrics"]
            val_m = data["best_val_metrics"]

            rec = {
                "exp_name": exp_name,
                "seed": seed,
                "val_macro_f1": val_m["macro_f1"],
                "val_neu_f1": val_m["neutral"]["f1"],
                "val_acc": val_m["accuracy"],
                "test_macro_f1": test_m["macro_f1"],
                "test_neu_f1": test_m["neutral"]["f1"],
                "test_neu_prec": test_m["neutral"]["precision"],
                "test_neu_rec": test_m["neutral"]["recall"],
                "test_neu_fp": test_m["neutral"]["fp"],
                "test_neu_fn": test_m["neutral"]["fn"],
                "test_acc": test_m["accuracy"],
                "test_weighted_f1": test_m["weighted_f1"],
                "test_neg_f1": test_m["negative"]["f1"],
                "test_pos_f1": test_m["positive"]["f1"],
                "training_time": data.get("training_time_seconds", 0.0),
                "applied_neutral_bias": data.get("applied_neutral_bias", 0.0),
                "pred_file": pred_file if has_preds else None,
            }
            records.append(rec)

    return pd.DataFrame(records)


def compute_paired_comparisons(df, baseline_name="B1_ce"):
    """
    Compute paired seed-by-seed and row-by-row bootstrap differences against baseline.
    """
    paired_rows = []
    base_df = df[df["exp_name"] == baseline_name]

    if base_df.empty:
        return pd.DataFrame()

    for exp, exp_group in df.groupby("exp_name"):
        if exp == baseline_name:
            continue

        common_seeds = set(exp_group["seed"]).intersection(set(base_df["seed"]))
        for s in common_seeds:
            row_exp = exp_group[exp_group["seed"] == s].iloc[0]
            row_base = base_df[base_df["seed"] == s].iloc[0]

            delta_macro = row_exp["test_macro_f1"] - row_base["test_macro_f1"]
            delta_neu = row_exp["test_neu_f1"] - row_base["test_neu_f1"]

            # Row bootstrap on test predictions if available
            p_val_macro = np.nan
            p_val_neu = np.nan
            ci_macro = [np.nan, np.nan]
            ci_neu = [np.nan, np.nan]

            if row_exp["pred_file"] and row_base["pred_file"]:
                data_exp = np.load(row_exp["pred_file"])
                data_base = np.load(row_base["pred_file"])
                y_true = data_exp["test_labels"]
                y_pred_exp = data_exp["test_preds"]
                y_pred_base = data_base["test_preds"]

                ci_boot = compute_paired_bootstrap_ci(
                    y_true, y_pred_exp, y_pred_base, n_bootstraps=1000, seed=42
                )
                ci_macro = ci_boot["delta_macro_ci"]
                ci_neu = ci_boot["delta_neu_ci"]
                p_val_macro = ci_boot["p_value_macro"]
                p_val_neu = ci_boot["p_value_neu"]

            paired_rows.append({
                "exp_name": exp,
                "seed": s,
                "delta_macro_f1": delta_macro,
                "delta_neu_f1": delta_neu,
                "delta_macro_ci_95": ci_macro,
                "delta_neu_ci_95": ci_neu,
                "bootstrap_p_val_macro": p_val_macro,
                "bootstrap_p_val_neu": p_val_neu,
            })

    return pd.DataFrame(paired_rows)


def generate_markdown_summary(summary_df, paired_df=None):
    """
    Format markdown table summarizing all experimental results.
    """
    md = []
    md.append("### Bảng Tổng Hợp Kết Quả Thực Nghiệm Trên UIT-VSFC Test Set\n")
    md.append("| Cấu hình | Số seeds | Macro-F1 (Mean ± Std) | Neutral F1 | Neutral Prec | Neutral Rec | Neutral FP | Neutral FN | Accuracy | Δ Macro-F1 (vs B1) |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    # Order of display
    order = [
        "B0_unweighted", "B0_balanced",
        "B1_ce", "B2_weighted_ce", "B3_ce_val_bias",
        "B4_ce_sampler", "B5_supcon", "B6_nbac", "B7_nbac_val_bias"
    ]

    for exp in order:
        if exp not in summary_df.index:
            continue
        row = summary_df.loc[exp]
        n_seeds = int(row["seed_count"])
        macro_str = f"{row['test_macro_f1_mean']:.4f} ± {row['test_macro_f1_std']:.4f}"
        neu_f1_str = f"{row['test_neu_f1_mean']:.4f} ± {row['test_neu_f1_std']:.4f}"
        neu_p_str = f"{row['test_neu_prec_mean']:.4f}"
        neu_r_str = f"{row['test_neu_rec_mean']:.4f}"
        fp_str = f"{row['test_neu_fp_mean']:.1f}"
        fn_str = f"{row['test_neu_fn_mean']:.1f}"
        acc_str = f"{row['test_acc_mean']:.4f}"

        delta_str = "-"
        if paired_df is not None and not paired_df.empty:
            p_sub = paired_df[paired_df["exp_name"] == exp]
            if not p_sub.empty:
                d_mean = p_sub["delta_macro_f1"].mean()
                delta_str = f"{d_mean:+.4f}"

        md.append(f"| **{exp}** | {n_seeds} | {macro_str} | {neu_f1_str} | {neu_p_str} | {neu_r_str} | {fp_str} | {fn_str} | {acc_str} | {delta_str} |")

    return "\n".join(md)
