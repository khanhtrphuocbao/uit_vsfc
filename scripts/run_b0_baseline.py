import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pyvi import ViTokenizer
from src.metrics import compute_all_metrics, compute_paired_bootstrap_ci


def main():
    print("=" * 60)
    print("RUNNING B0 BASELINE: TF-IDF + LOGISTIC REGRESSION (None & Balanced)")
    print("=" * 60)

    train_path = "data/train.json"
    val_path = "data/valid.json"
    test_path = "data/test.json"

    train_df = pd.read_json(train_path)
    val_df = pd.read_json(val_path)
    test_df = pd.read_json(test_path)

    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    y_train = train_df["Sentiment"].str.lower().map(label_map).values
    y_val = val_df["Sentiment"].str.lower().map(label_map).values
    y_test = test_df["Sentiment"].str.lower().map(label_map).values

    # Check syllable-level TF-IDF (matching standard nlp baseline in working draft)
    # Using ngram_range=(1, 2)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_train = vec.fit_transform(train_df["Sentence"])
    X_val = vec.transform(val_df["Sentence"])
    X_test = vec.transform(test_df["Sentence"])

    os.makedirs("outputs/B0_tfidf_lr", exist_ok=True)
    all_results = {}

    for cw, tag in [(None, "unweighted"), ("balanced", "balanced")]:
        print(f"\n--- Training LogisticRegression (class_weight={cw}) ---")
        clf = LogisticRegression(max_iter=1000, class_weight=cw, random_state=42)
        clf.fit(X_train, y_train)

        val_preds = clf.predict(X_val)
        val_metrics = compute_all_metrics(y_val, val_preds)

        test_preds = clf.predict(X_test)
        test_metrics = compute_all_metrics(y_test, test_preds)

        # Bootstrap CI on test
        ci_res = compute_paired_bootstrap_ci(y_test, test_preds, n_bootstraps=1000, seed=42)

        print(f"[{tag}] Validation: Macro-F1 = {val_metrics['macro_f1']:.4f} | Neu-F1 = {val_metrics['neutral']['f1']:.4f} | Acc = {val_metrics['accuracy']:.4f}")
        print(f"[{tag}] Test:       Macro-F1 = {test_metrics['macro_f1']:.4f} | Neu-F1 = {test_metrics['neutral']['f1']:.4f} | Acc = {test_metrics['accuracy']:.4f}")
        print(f"[{tag}] Neutral Test Details: Precision = {test_metrics['neutral']['precision']:.4f}, Recall = {test_metrics['neutral']['recall']:.4f}, FP = {test_metrics['neutral']['fp']}, FN = {test_metrics['neutral']['fn']}")
        print(f"[{tag}] 95% Bootstrap CI: Macro-F1 = {ci_res['macro_f1_ci']}, Neu-F1 = {ci_res['neu_f1_ci']}")

        res = {
            "class_weight": cw,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "bootstrap_ci": ci_res,
        }
        all_results[tag] = res

        np.savez_compressed(
            f"outputs/B0_tfidf_lr/preds_{tag}.npz",
            y_test=y_test,
            test_preds=test_preds,
        )

    with open("outputs/B0_tfidf_lr/results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\nSaved B0 results to outputs/B0_tfidf_lr/results.json")


if __name__ == "__main__":
    main()
