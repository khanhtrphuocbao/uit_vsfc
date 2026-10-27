"""Synthetic offline unit tests: no network, no actual UIT data included."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_experiment as pilot


class DummyLabelFeature:
    def __init__(self, names):
        self.names = names


class FakeSplit:
    def __init__(self, records, names=None):
        self._df = pd.DataFrame.from_records(records)
        self.column_names = list(self._df.columns)
        self.features = {
            "sentiment": DummyLabelFeature(names or list(pilot.LABELS)),
            "topic": DummyLabelFeature(list(pilot.TOPICS)),
        }
        self._fingerprint = "synthetic"

    def to_pandas(self):
        return self._df.copy()


def example_source():
    source = {}
    # Distinct strings per row and split, deliberately imbalanced classes.
    for split, size in (("train", 150), ("validation", 60), ("test", 60)):
        values = [0] * (size // 2) + [1] * (size // 6)
        values += [2] * (size - len(values))
        records = [
            {"sentence": f"{split} example {i} teacher knowledge topic {label}",
             "sentiment": label, "topic": i % 4}
            for i, label in enumerate(values)
        ]
        source[split] = FakeSplit(records)
    return source


class SourceTests(unittest.TestCase):
    def test_requested_call_and_fallback_to_same_repo(self):
        calls = []
        sample = example_source()

        def simulated_load(name, **kwargs):
            calls.append((name, kwargs))
            if not kwargs:
                raise RuntimeError(
                    "Dataset scripts are no longer supported, but found "
                    "vietnamese_students_feedback.py"
                )
            return sample

        with patch.dict(sys.modules, {"datasets": types.SimpleNamespace(load_dataset=simulated_load)}):
            data, info = pilot.load_source()
        self.assertEqual(data, sample)
        self.assertEqual(info["revision"], pilot.PARQUET_REVISION)
        self.assertEqual(calls, [
            ("uitnlp/vietnamese_students_feedback", {}),
            ("uitnlp/vietnamese_students_feedback", {"revision": "refs/convert/parquet"}),
        ])

    def test_other_errors_do_not_fallback(self):
        calls = []

        def broken_load(*args, **kwargs):
            calls.append((args, kwargs))
            raise RuntimeError("Permission denied")

        with patch.dict(sys.modules, {"datasets": types.SimpleNamespace(load_dataset=broken_load)}):
            with self.assertRaisesRegex(RuntimeError, "Permission denied"):
                pilot.load_source()
        self.assertEqual(len(calls), 1)

    def test_expected_label_mapping(self):
        data = pilot.prepare(example_source())
        self.assertEqual(set(data["train"]["label"]), set(pilot.LABELS))
        self.assertEqual(set(data["train"]["topic"]), set(pilot.TOPICS))
        self.assertEqual(data["train"].iloc[0]["sentence"],
                         "train example 0 teacher knowledge topic 0")
        self.assertEqual(data["train"].iloc[0]["label"], "negative")

    def test_wrong_classlabel_order_is_rejected(self):
        bad = example_source()
        bad["train"].features["sentiment"] = DummyLabelFeature(
            ["positive", "negative", "neutral"]
        )
        with self.assertRaisesRegex(ValueError, "unexpected ClassLabel order"):
            pilot.prepare(bad)

    def test_unsupported_label_is_rejected(self):
        bad = example_source()
        bad["train"]._df.loc[0, "sentiment"] = 9
        with self.assertRaisesRegex(ValueError, "invalid class"):
            pilot.prepare(bad)

    def test_audit_detects_duplicated_and_overlapping_text(self):
        source = example_source()
        first = source["train"]._df.loc[0, "sentence"]
        source["validation"]._df.loc[1, "sentence"] = first.upper()
        source["train"]._df.loc[1, "sentence"] = first
        data = pilot.prepare(source)
        report = pilot.audit(data, source)
        self.assertEqual(report["cross_split_overlap"]["train_vs_validation"], 1)
        self.assertEqual(report["splits"]["train"]["duplicate_text_rows"], 1)
        self.assertFalse(report["splits"]["train"]["matches_published_size"])
        self.assertEqual(report["splits"]["test"]["rows"], 60)


class ExperimentTests(unittest.TestCase):
    def test_validation_only_and_opt_in_test(self):
        source = example_source()
        data = pilot.prepare(source)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            r = pilot.fit_and_evaluate(data, [0.2, 1.0], [42], out, False)
            self.assertEqual(r["validation_model_fits"], 4)
            self.assertEqual(r["test_evaluations"], 0)
            self.assertTrue((out / "validation_results.csv").exists())
            self.assertFalse((out / "test_results.csv").exists())
            summary = pd.read_csv(out / "validation_summary.csv")
            self.assertEqual(len(summary), 4)
            for name in ["macro_f1_mean", "macro_f1_std", "neutral_f1_mean"]:
                self.assertIn(name, summary.columns)
            manifest = json.loads((out / "train_sample_manifest.json").read_text())
            self.assertEqual(len(manifest), 2)
            self.assertEqual(manifest[1]["n_train"], 150)

            other = out / "testmode"
            other.mkdir()
            r2 = pilot.fit_and_evaluate(data, [0.2, 1.0], [42], other, True)
            self.assertEqual(r2["test_evaluations"], 4)
            preds = pd.read_csv(other / "test_predictions.csv")
            self.assertEqual(len(preds), 4 * 60)
            self.assertTrue(preds["row_index"].isin(range(60)).all())
            self.assertNotIn("sentence", preds.columns)

    def test_fractions_and_reproducibility(self):
        train = pilot.prepare(example_source())["train"]
        y = train["label"].to_numpy()
        a = pilot.sample_indices(y, 0.2, 42)
        b = pilot.sample_indices(y, 0.2, 42)
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(len(a), 30)
        self.assertEqual(set(y[a]), set(pilot.LABELS))
        self.assertEqual(pilot.parse_fractions("1.0,0.05,0.1"), [0.05, 0.1, 1.0])
        with self.assertRaises(ValueError):
            pilot.parse_fractions("0,1")


if __name__ == "__main__":
    unittest.main()
