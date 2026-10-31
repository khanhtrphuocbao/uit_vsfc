import numpy as np
from src.metrics import compute_all_metrics, compute_paired_bootstrap_ci


def test_metrics_computation():
    # 0: Neg, 1: Neu, 2: Pos
    y_true = [0, 0, 1, 1, 2, 2, 0, 1]
    y_pred = [0, 1, 1, 0, 2, 2, 0, 1]

    metrics = compute_all_metrics(y_true, y_pred)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0

    # Neutral class analysis
    neu = metrics["neutral"]
    # y_true neutrals are at idx 2, 3, 7 (total 3)
    # y_pred for neutrals: idx 1 is False Positive (true is 0, pred is 1)
    # idx 2 is True Positive (true 1, pred 1)
    # idx 3 is False Negative (true 1, pred 0)
    # idx 7 is True Positive (true 1, pred 1)
    assert neu["support"] == 3
    assert neu["tp"] == 2
    assert neu["fp"] == 1
    assert neu["fn"] == 1


def test_paired_bootstrap_ci():
    y_true = np.random.choice([0, 1, 2], size=100, p=[0.45, 0.1, 0.45])
    y_pred_m = y_true.copy()
    # Introduce a few errors
    y_pred_m[0:10] = (y_pred_m[0:10] + 1) % 3
    y_pred_b = y_true.copy()
    y_pred_b[0:20] = (y_pred_b[0:20] + 1) % 3

    ci_res = compute_paired_bootstrap_ci(
        y_true, y_pred_m, y_pred_b, n_bootstraps=100, seed=42
    )

    assert "macro_f1_ci" in ci_res
    assert "delta_macro_ci" in ci_res
    assert ci_res["macro_f1_ci"][0] <= ci_res["macro_f1_ci"][1]
