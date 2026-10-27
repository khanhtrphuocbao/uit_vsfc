#!/usr/bin/env python3
"""UIT-VSFC: label scarcity and class imbalance (UIT NLP release).

Default source:
    from datasets import load_dataset
    ds = load_dataset("uitnlp/vietnamese_students_feedback")

If modern datasets rejects the legacy loading script on the main branch,
retry the SAME repository's Hugging Face auto-converted Parquet branch.
No other dataset or mirror is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATASET_ID = "uitnlp/vietnamese_students_feedback"
PARQUET_REVISION = "refs/convert/parquet"
EXPECTED_ROWS = {"train": 11426, "validation": 1583, "test": 3166}
SPLITS = tuple(EXPECTED_ROWS)
LABELS = ("negative", "neutral", "positive")
TOPICS = ("lecturer", "training_program", "facility", "others")
DEFAULT_FRACTIONS = "0.05,0.10,0.25,0.50,1.0"
DEFAULT_SEEDS = "42,52,62"


def load_source(local_dataset_dir: Path | None = None) -> tuple[Any, dict]:
    """Load the UIT NLP release without using third-party mirrors.

    When --local-dataset-dir is used, it must be a prior save_to_disk() copy
    of the same dataset. Content hashes and schema are checked afterward.
    """
    if local_dataset_dir is not None:
        from datasets import load_from_disk

        ds = load_from_disk(str(local_dataset_dir))
        return ds, {
            "dataset_id": DATASET_ID,
            "loading_method": "local_save_to_disk",
            "local_dataset_dir": str(local_dataset_dir.resolve()),
            "revision": "user-supplied local copy; verify content SHA256",
        }

    from datasets import load_dataset

    try:
        # Exact official source requested by the user.
        ds = load_dataset("uitnlp/vietnamese_students_feedback")
        method, revision = "load_dataset_main", "main"
    except RuntimeError as exc:
        message = str(exc).lower()
        if "dataset scripts are no longer supported" not in message:
            raise
        print(
            "Official repository uses a legacy loading script; "
            "retrying its own auto-converted Parquet revision.",
            file=sys.stderr,
        )
        ds = load_dataset(DATASET_ID, revision=PARQUET_REVISION)
        method, revision = "load_dataset_official_parquet", PARQUET_REVISION
    return ds, {"dataset_id": DATASET_ID, "loading_method": method, "revision": revision}


def label_value(value: Any, names: tuple[str, ...], field: str, split: str, row: int) -> str:
    """Decode using documented UIT class IDs, never using guessed mapping."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{split}[{row}].{field}: boolean is not a valid label")
    if isinstance(value, (int, np.integer)):
        idx = int(value)
        if 0 <= idx < len(names):
            return names[idx]
    elif isinstance(value, str):
        val = value.strip().lower()
        if val in names:
            return val
        if val.isascii() and val.isdecimal():
            idx = int(val)
            if 0 <= idx < len(names):
                return names[idx]
    raise ValueError(f"{split}[{row}].{field}: invalid class {value!r}; expected {list(names)}")


def normalize_split(split: str, raw: Any) -> pd.DataFrame:
    required = {"sentence", "sentiment", "topic"}
    columns = set(getattr(raw, "column_names", []))
    if not required.issubset(columns):
        raise ValueError(f"{split}: expected columns {sorted(required)}; found {sorted(columns)}")
    features = getattr(raw, "features", {})
    for col, names in (("sentiment", LABELS), ("topic", TOPICS)):
        feature = features.get(col) if hasattr(features, "get") else None
        declared_names = getattr(feature, "names", None)
        if declared_names is not None and tuple(declared_names) != names:
            raise ValueError(
                f"{split}.{col}: unexpected ClassLabel order {declared_names}; "
                f"expected {names}; refusing to guess label IDs"
            )

    raw_df = raw.to_pandas()
    if not required.issubset(raw_df.columns):
        raise ValueError(f"{split}: missing fields after conversion to DataFrame")
    sentences = raw_df["sentence"].tolist()
    if any(not isinstance(s, str) or not s.strip() for s in sentences):
        raise ValueError(f"{split}: null, non-string or empty sentence detected")
    labels = [label_value(v, LABELS, "sentiment", split, i)
              for i, v in enumerate(raw_df["sentiment"].tolist())]
    topics = [label_value(v, TOPICS, "topic", split, i)
              for i, v in enumerate(raw_df["topic"].tolist())]
    # Preserve sentence bytes and source row ordering: no cleaning/normalizing.
    return pd.DataFrame({
        "row_index": np.arange(len(sentences), dtype=int),
        "sentence": sentences,
        "label": labels,
        "topic": topics,
    })


def prepare(source: Any) -> dict[str, pd.DataFrame]:
    try:
        actual = set(source.keys())
    except AttributeError as exc:
        raise ValueError("Expected a DatasetDict containing train, validation and test") from exc
    missing = set(SPLITS) - actual
    if missing:
        raise ValueError(f"Missing required source splits: {sorted(missing)}")
    data = {split: normalize_split(split, source[split]) for split in SPLITS}
    for split, df in data.items():
        counts = Counter(df["label"])
        if set(counts) != set(LABELS):
            raise ValueError(f"{split}: missing sentiment classes: {set(LABELS) - set(counts)}")
    return data


def content_sha256(df: pd.DataFrame) -> str:
    """Canonical raw-content hash over ordered (sentence, sentiment, topic)."""
    h = hashlib.sha256()
    for text, sentiment, topic in zip(df["sentence"], df["label"], df["topic"]):
        line = json.dumps([text, sentiment, topic], ensure_ascii=False, separators=(",", ":"))
        h.update(line.encode("utf-8") + b"\n")
    return h.hexdigest()


def audit_text_key(text: str) -> str:
    """Normalized only for leakage/duplicate audit, NEVER for fitting."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def audit(data: dict[str, pd.DataFrame], source: Any) -> dict:
    report: dict[str, Any] = {"dataset_id": DATASET_ID, "splits": {}, "cross_split_overlap": {}}
    keys = {}
    for split, df in data.items():
        normalized = df["sentence"].map(audit_text_key)
        keys[split] = set(normalized)
        distinct_labels = pd.DataFrame({"key": normalized, "label": df["label"]})
        conflicts = distinct_labels.groupby("key")["label"].nunique().gt(1).sum()
        report["splits"][split] = {
            "rows": len(df),
            "matches_published_size": len(df) == EXPECTED_ROWS[split],
            "expected_published_rows": EXPECTED_ROWS[split],
            "sentiment_counts": {c: int((df["label"] == c).sum()) for c in LABELS},
            "topic_counts": {c: int((df["topic"] == c).sum()) for c in TOPICS},
            "duplicate_text_rows": int(normalized.duplicated().sum()),
            "texts_with_conflicting_sentiment": int(conflicts),
            "ordered_content_sha256": content_sha256(df),
            "huggingface_fingerprint": getattr(source[split], "_fingerprint", None),
        }
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            common = keys[left] & keys[right]
            report["cross_split_overlap"][f"{left}_vs_{right}"] = len(common)
    return report


def parse_fractions(text: str) -> list[float]:
    try:
        vals = sorted({float(v.strip()) for v in text.split(",")})
    except ValueError as exc:
        raise ValueError("--fractions must be comma-separated numbers") from exc
    if not vals or any(not 0 < v <= 1 for v in vals):
        raise ValueError("Training fractions must be within (0, 1]")
    return vals


def parse_seeds(text: str) -> list[int]:
    try:
        vals = [int(v.strip()) for v in text.split(",")]
    except ValueError as exc:
        raise ValueError("--seeds must be comma-separated integers") from exc
    if not vals or len(set(vals)) != len(vals):
        raise ValueError("Provide at least one seed with no duplicates")
    return vals


def sample_indices(y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if fraction == 1:
        return np.arange(len(y), dtype=int)
    selected, _ = train_test_split(
        np.arange(len(y)), train_size=fraction, stratify=y, random_state=seed
    )
    return np.sort(selected)


def build_pipeline(balanced: bool, seed: int) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents=None,
            sublinear_tf=True,
            min_df=1,
            max_features=45000,
        )),
        ("classifier", LogisticRegression(
            C=1.0,
            max_iter=800,
            random_state=seed,
            class_weight="balanced" if balanced else None,
        )),
    ])


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(LABELS), zero_division=0
    )
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f.mean()),
        "weighted_f1": float(np.average(f, weights=support)),
    }
    for i, cls in enumerate(LABELS):
        result[f"{cls}_precision"] = float(p[i])
        result[f"{cls}_recall"] = float(r[i])
        result[f"{cls}_f1"] = float(f[i])
        result[f"{cls}_support"] = int(support[i])
    return result


def make_summary(frame: pd.DataFrame) -> pd.DataFrame:
    group = ["train_fraction", "class_weight"]
    numerical = ["accuracy", "macro_f1", "weighted_f1", "neutral_precision",
                 "neutral_recall", "neutral_f1", "negative_f1", "positive_f1"]
    summary = frame.groupby(group)[numerical].agg(["mean", lambda x: x.std(ddof=0)])
    summary.columns = [f"{name}_{'std' if stat == '<lambda_0>' else stat}"
                       for name, stat in summary.columns]
    return summary.reset_index()


def fit_and_evaluate(
    data: dict[str, pd.DataFrame], fractions: list[float], seeds: list[int],
    output_dir: Path, include_test: bool,
) -> dict:
    x_train = data["train"]["sentence"].to_numpy()
    y_train = data["train"]["label"].to_numpy()
    x_valid = data["validation"]["sentence"].to_numpy()
    y_valid = data["validation"]["label"].to_numpy()
    x_test = data["test"]["sentence"].to_numpy() if include_test else None
    y_test = data["test"]["label"].to_numpy() if include_test else None
    valid_metrics: list[dict] = []
    test_metrics: list[dict] = []
    valid_confusions: list[dict] = []
    test_confusions: list[dict] = []
    test_predictions: list[dict] = []
    samples: list[dict] = []
    endpoints = {min(fractions), max(fractions)}

    for seed in seeds:
        for fraction in fractions:
            indices = sample_indices(y_train, fraction, seed)
            counts = Counter(y_train[indices])
            if set(counts) != set(LABELS):
                raise ValueError(f"train fraction={fraction}, seed={seed} excludes a class")
            samples.append({
                "seed": seed, "train_fraction": fraction,
                "n_train": len(indices),
                "class_counts": {c: int(counts[c]) for c in LABELS},
                "row_indices_sha256": hashlib.sha256(
                    indices.astype("<i8").tobytes()).hexdigest(),
            })
            for balanced in (False, True):
                model = build_pipeline(balanced, seed)
                model.fit(x_train[indices], y_train[indices])
                config = {
                    "seed": seed, "train_fraction": fraction, "n_train": len(indices),
                    "class_weight": "balanced" if balanced else "none",
                }
                pred = model.predict(x_valid)
                valid_metrics.append({**config, **metrics(y_valid, pred)})
                matrix = confusion_matrix(y_valid, pred, labels=list(LABELS))
                for i, actual in enumerate(LABELS):
                    for j, predicted in enumerate(LABELS):
                        valid_confusions.append({**config, "actual": actual,
                                                 "predicted": predicted,
                                                 "count": int(matrix[i, j])})
                # Test is opt-in and restricted to predeclared low/full data
                # endpoints. No test-based model selection is performed.
                if include_test and fraction in endpoints:
                    assert x_test is not None and y_test is not None
                    pred_test = model.predict(x_test)
                    test_metrics.append({**config, **metrics(y_test, pred_test)})
                    cm = confusion_matrix(y_test, pred_test, labels=list(LABELS))
                    for i, actual in enumerate(LABELS):
                        for j, predicted in enumerate(LABELS):
                            test_confusions.append({**config, "actual": actual,
                                                    "predicted": predicted,
                                                    "count": int(cm[i, j])})
                    for row, (actual, predicted) in enumerate(zip(y_test, pred_test)):
                        test_predictions.append({**config, "row_index": row,
                                                 "true_label": str(actual),
                                                 "predicted_label": str(predicted)})
            print(f"Completed seed={seed}, fraction={fraction}, n={len(indices)}", flush=True)

    pd.DataFrame(valid_metrics).to_csv(output_dir / "validation_results.csv", index=False)
    make_summary(pd.DataFrame(valid_metrics)).to_csv(
        output_dir / "validation_summary.csv", index=False
    )
    pd.DataFrame(valid_confusions).to_csv(output_dir / "validation_confusions.csv", index=False)
    (output_dir / "train_sample_manifest.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if include_test:
        pd.DataFrame(test_metrics).to_csv(output_dir / "test_results.csv", index=False)
        make_summary(pd.DataFrame(test_metrics)).to_csv(
            output_dir / "test_summary.csv", index=False
        )
        pd.DataFrame(test_confusions).to_csv(output_dir / "test_confusions.csv", index=False)
        pd.DataFrame(test_predictions).to_csv(output_dir / "test_predictions.csv", index=False)
    return {
        "validation_model_fits": len(valid_metrics),
        "test_evaluations": len(test_metrics),
        "test_evaluated": include_test,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/uit_vsfc_uitnlp"))
    parser.add_argument("--local-dataset-dir", type=Path, default=None,
                        help="Optional DatasetDict save_to_disk folder from official UIT dataset")
    parser.add_argument("--fractions", default=DEFAULT_FRACTIONS)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--include-test", action="store_true",
                        help="Evaluate predeclared min/full train endpoints on test (opt-in)")
    parser.add_argument("--strict-source-counts", action="store_true",
                        help="Fail if train/validation/test counts differ from UIT dataset card")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly replace prior outputs produced by this pilot")
    args = parser.parse_args()
    try:
        fractions = parse_fractions(args.fractions)
        seeds = parse_seeds(args.seeds)
    except ValueError as exc:
        parser.error(str(exc))

    source, provenance = load_source(args.local_dataset_dir)
    data = prepare(source)
    review = audit(data, source)
    mismatched = [s for s in SPLITS if not review["splits"][s]["matches_published_size"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists at {manifest_path}. Use a new --output-dir, "
            "or pass --overwrite to explicitly replace this pilot's outputs."
        )
    if args.overwrite:
        known_outputs = (
            "manifest.json", "train_sample_manifest.json", "validation_results.csv",
            "validation_summary.csv", "validation_confusions.csv", "test_results.csv",
            "test_summary.csv", "test_confusions.csv", "test_predictions.csv",
        )
        for filename in known_outputs:
            (args.output_dir / filename).unlink(missing_ok=True)
    manifest: dict[str, Any] = {
        "source": provenance,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "protocol": {
            "task": "sentiment (3-way); topic is audit-only",
            "text_field": "sentence (unchanged)",
            "sentiment_id_mapping": {str(i): name for i, name in enumerate(LABELS)},
            "topic_id_mapping": {str(i): name for i, name in enumerate(TOPICS)},
            "vectorizer": "word TF-IDF, token_pattern=(?u)\\b\\w+\\b, 1-2 grams",
            "model": "LogisticRegression(C=1.0, max_iter=800)",
            "comparison": "class_weight=None vs balanced",
            "fractions": fractions, "seeds": seeds,
            "split_policy": "published splits unchanged; no silent de-duplication",
            "test_policy": "default: validation only; --include-test: both weights at min/max fractions",
        },
        "audit": review,
        "status": "AUDIT_ONLY" if args.audit_only else "STARTED",
    }
    if args.strict_source_counts and mismatched:
        manifest["status"] = "BLOCKED_SIZE_MISMATCH"
        manifest["size_mismatch_splits"] = mismatched
    # Save an audit manifest before fitting, so interrupted runs retain evidence.
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(review, indent=2, ensure_ascii=False))
    if manifest["status"] == "BLOCKED_SIZE_MISMATCH":
        raise ValueError(f"Unexpected split sizes: {mismatched}; see manifest audit")
    if not args.audit_only:
        manifest["runs"] = fit_and_evaluate(data, fractions, seeds, args.output_dir,
                                            include_test=args.include_test)
        manifest["status"] = "COMPLETED"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    print(f"Output folder: {args.output_dir}")


if __name__ == "__main__":
    main()
