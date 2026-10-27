# UIT-VSFC Low-Data Pilot (UIT NLP release, v2)

This pilot examines **training-data scarcity and class imbalance** for 3-class Vietnamese student-feedback sentiment classification. It is a coursework-level empirical comparison, not a new normalization algorithm or a claim of a previously unsolved research gap.

## Dataset and the requested loader

**The official `uitnlp/vietnamese_students_feedback` dataset is the only source.** No `ura-hcmut/UIT-VSFC` mirror is used.

The main loader follows the exact user-requested command:

```python
from datasets import load_dataset

ds = load_dataset("uitnlp/vietnamese_students_feedback")
```

**Compatibility note:** the UIT NLP repository contains a legacy Python loading script. Modern versions of `datasets` may reject these scripts with `Dataset scripts are no longer supported`. For that *specific* error, this pilot transparently retries the *same repository* using Hugging Face's auto-converted Parquet revision:

```python
ds = load_dataset(
    "uitnlp/vietnamese_students_feedback",
    revision="refs/convert/parquet",
)
```

If loading fails for another reason (no Internet, access issue, missing file), it does **not** silently use another dataset. The chosen loading method, split fingerprints and ordered-content SHA256 hashes are saved in `manifest.json`.

### Columns and fixed UIT mappings

| Source field | Meaning | Use in this pilot |
| --- | --- | --- |
| `sentence` | Original Vietnamese student feedback | Model input; no manual text repair |
| `sentiment` | `0=negative`, `1=neutral`, `2=positive` | Target label |
| `topic` | `0=lecturer`, `1=training_program`, `2=facility`, `3=others` | Audit only; **not** an input feature or prediction target |

Published split sizes: **train 11,426**, **validation 1,583**, **test 3,166** (total **16,175**). Exact splits and label IDs are checked against the UIT NLP dataset card. A mismatch is reported, not silently altered; add `--strict-source-counts` to make a size mismatch fatal.

## Research questions

- **RQ1:** How does training-data scarcity affect 3-way sentiment classification, especially the rare neutral class?
- **RQ2:** Does `class_weight="balanced"` improve neutral recall/F1, and how does it change precision and the majority-class performance?

The two comparisons use the same features and classifier, differing only in class weighting.

## Installation (Python 3.10+ recommended)

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the venv using `.venv\\Scripts\\activate`.

## Commands

Run dataset audit first (no model fitting):

```sh
python run_experiment.py --audit-only --strict-source-counts \
  --output-dir outputs/dataset_audit
```

Small validation smoke test (2 models x 1 fraction x 1 seed):

```sh
python run_experiment.py --fractions 0.05 --seeds 42 \
  --output-dir outputs/smoke
```

Full validation experiment (five fractions, three seeds, two class-weight strategies):

```sh
python run_experiment.py --strict-source-counts \
  --output-dir outputs/uit_vsfc_uitnlp
```

**Only after validation review**, explicitly unlock the held-out test. The code evaluates test only at the predeclared minimum and maximum training fractions and both class-weight settings; do not change the plan based on test results.

```sh
python run_experiment.py --strict-source-counts --include-test \
  --output-dir outputs/final_eval
```

Optional offline reload (after obtaining a trusted source copy):

```python
from datasets import load_dataset

ds = load_dataset("uitnlp/vietnamese_students_feedback",
                  revision="refs/convert/parquet")
ds.save_to_disk("data/uitnlp_feedback")
```

```sh
python run_experiment.py --local-dataset-dir data/uitnlp_feedback --audit-only
```

## Experimental controls

- Unchanged source train/validation/test. **Never** sample evaluation items from train.
- Training subsets: `5%, 10%, 25%, 50%, 100%`, stratified by `sentiment` within the original train split, seeds `42,52,62`.
- TF-IDF word/syllable surface tokens, 1- and 2-grams; no claim of linguistic Vietnamese word segmentation.
- LogisticRegression with `C=1.0`, `max_iter=800`, default weights vs `class_weight="balanced"`.
- **Primary metric:** Macro-F1 on validation. Also report neutral precision, recall, F1, all other class F1, accuracy and weighted F1.
- Inspect confusion counts and overlapping/contradictory text statistics. Audit matches use normalized text *for audit only*, not for model input. Preserve raw sentences.
- Do not tune model settings, tokenization or thresholds against held-out test results.

## Output files

Every run writes `manifest.json`: dataset ID, loading path, feature schema, published-size checks, class distributions, source fingerprints/content hashes, model settings and run status.

Validation runs write `train_sample_manifest.json`, `validation_results.csv`, `validation_summary.csv`, `validation_confusions.csv`. With `--include-test`, it additionally writes `test_results.csv`, `test_summary.csv`, `test_confusions.csv`, `test_predictions.csv` (without the raw text). An audit-only run writes the manifest only. To prevent stale results, a non-empty pilot output with an existing `manifest.json` is not overwritten unless `--overwrite` is specified; using a new `--output-dir` is preferred.

## Limitations and responsible interpretation

- This experiment studies **sentiment** only, although UIT-VSFC also has topic annotations.
- TF-IDF uses surface tokens under spaces. It is not a Vietnamese word segmentation benchmark.
- Topic is never used to predict sentiment: that would introduce a separate explanatory variable.
- Class weighting does not change the number of source examples; it changes the classifier's loss.
- Duplicate/overlapping text and conflicting labels are *reported* rather than silently removed. Document potential leakage before drawing broad conclusions.
- Dataset creators list the license as **unknown**. Confirm usage and redistribution terms with the authors if necessary. This archive contains **no dataset rows** and **no invented scores**.
- This is an empirical class-imbalance experiment; it does not demonstrate a novel method or generalize to all Vietnamese writing styles.

## References

- UIT NLP source: https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback
- Source dataset card: https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/blob/main/README.md
- HF converted Parquet branch: https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/tree/refs%2Fconvert%2Fparquet
- Nguyen et al. (2018), UIT-VSFC, KSE: https://doi.org/10.1109/KSE.2018.8573337
- scikit-learn leakage guidance: https://scikit-learn.org/stable/common_pitfalls
# uit_vsfc
