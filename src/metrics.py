import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    f1_score,
)


def compute_all_metrics(y_true, y_pred):
    """
    Compute comprehensive classification metrics for UIT-VSFC 3-class sentiment task.

    Classes: 0: Negative, 1: Neutral, 2: Positive
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))

    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # Class 1 is Neutral
    # False Positives for Neutral: true != 1, pred == 1
    neu_fp = int(np.sum((y_true != 1) & (y_pred == 1)))
    # False Negatives for Neutral: true == 1, pred != 1
    neu_fn = int(np.sum((y_true == 1) & (y_pred != 1)))
    # True Positives for Neutral
    neu_tp = int(cm[1, 1])
    # True Negatives for Neutral
    neu_tn = int(len(y_true) - neu_fp - neu_fn - neu_tp)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "negative": {
            "precision": float(p[0]),
            "recall": float(r[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "neutral": {
            "precision": float(p[1]),
            "recall": float(r[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
            "tp": neu_tp,
            "fp": neu_fp,
            "fn": neu_fn,
            "tn": neu_tn,
        },
        "positive": {
            "precision": float(p[2]),
            "recall": float(r[2]),
            "f1": float(f1[2]),
            "support": int(support[2]),
        },
        "confusion_matrix": cm.tolist(),
    }


def compute_paired_bootstrap_ci(
    y_true,
    y_pred_model,
    y_pred_baseline=None,
    n_bootstraps=1000,
    seed=42,
    alpha=0.05,
):
    """
    Compute 95% bootstrap confidence intervals for model metrics and
    paired delta metrics against a baseline on test set rows.

    Note: This measures test-sample variation (row sampling bootstrap),
    distinct from seed-to-seed training variation.
    """
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_pred_model = np.array(y_pred_model)
    has_baseline = y_pred_baseline is not None
    if has_baseline:
        y_pred_baseline = np.array(y_pred_baseline)

    n_samples = len(y_true)
    macro_f1_list = []
    neu_f1_list = []
    delta_macro_list = []
    delta_neu_list = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, size=n_samples)
        yt_boot = y_true[indices]
        yp_boot = y_pred_model[indices]

        # Model metrics
        m_macro = f1_score(yt_boot, yp_boot, average="macro", zero_division=0)
        m_neu = f1_score(yt_boot, yp_boot, average=None, labels=[0, 1, 2], zero_division=0)[1]

        macro_f1_list.append(m_macro)
        neu_f1_list.append(m_neu)

        if has_baseline:
            yp_base_boot = y_pred_baseline[indices]
            b_macro = f1_score(yt_boot, yp_base_boot, average="macro", zero_division=0)
            b_neu = f1_score(yt_boot, yp_base_boot, average=None, labels=[0, 1, 2], zero_division=0)[1]

            delta_macro_list.append(m_macro - b_macro)
            delta_neu_list.append(m_neu - b_neu)

    lower_pct = 100 * (alpha / 2.0)
    upper_pct = 100 * (1.0 - alpha / 2.0)

    result = {
        "macro_f1_ci": [
            float(np.percentile(macro_f1_list, lower_pct)),
            float(np.percentile(macro_f1_list, upper_pct)),
        ],
        "neu_f1_ci": [
            float(np.percentile(neu_f1_list, lower_pct)),
            float(np.percentile(neu_f1_list, upper_pct)),
        ],
    }

    if has_baseline:
        result["delta_macro_ci"] = [
            float(np.percentile(delta_macro_list, lower_pct)),
            float(np.percentile(delta_macro_list, upper_pct)),
        ]
        result["delta_neu_ci"] = [
            float(np.percentile(delta_neu_list, lower_pct)),
            float(np.percentile(delta_neu_list, upper_pct)),
        ]
        # P-value approximation from bootstrap: proportion of deltas <= 0
        p_val_macro = float(np.mean(np.array(delta_macro_list) <= 0.0))
        p_val_neu = float(np.mean(np.array(delta_neu_list) <= 0.0))
        result["p_value_macro"] = p_val_macro
        result["p_value_neu"] = p_val_neu

    return result
