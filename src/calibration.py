import numpy as np
from sklearn.metrics import f1_score


def search_optimal_neutral_bias(
    val_logits,
    val_labels,
    b_min=-2.0,
    b_max=2.0,
    step=0.05,
    metric="macro_f1",
):
    """
    Search for the optimal Neutral logit bias strictly on the validation set.

    For each candidate bias b in [b_min, b_max]:
        adjusted_logits[:, 1] = val_logits[:, 1] + b
        preds = argmax(adjusted_logits, axis=-1)
        score = Macro-F1(val_labels, preds)

    Returns:
        best_bias (float): The optimal bias that maximizes the validation metric.
        best_score (float): The metric score achieved on validation.
        grid_results (list): Detailed list of (bias, macro_f1, neutral_f1) across the search.
    """
    val_logits = np.array(val_logits, copy=True)
    val_labels = np.array(val_labels)

    grid = np.arange(b_min, b_max + 1e-6, step)
    best_bias = 0.0
    best_score = -1.0
    grid_results = []

    for b in grid:
        b = round(float(b), 4)
        adj_logits = val_logits.copy()
        adj_logits[:, 1] += b
        preds = np.argmax(adj_logits, axis=1)

        macro = float(f1_score(val_labels, preds, average="macro", zero_division=0))
        neu_f1 = float(f1_score(val_labels, preds, average=None, labels=[0, 1, 2], zero_division=0)[1])

        grid_results.append({
            "bias": b,
            "macro_f1": macro,
            "neutral_f1": neu_f1,
        })

        if macro > best_score:
            best_score = macro
            best_bias = b

    return best_bias, best_score, grid_results


def apply_neutral_bias(logits, bias):
    """
    Apply scalar bias shift to Neutral class (index 1).
    """
    adj = np.array(logits, copy=True)
    adj[:, 1] += bias
    return adj
