# Pilot verification status

Last verified: 2026-10-26.

## Tests completed here

- `python -m py_compile run_experiment.py` : PASS
- `python -m unittest discover -s tests -v` : 8/8 PASS
- Synthetic train/validation/test splits (not UIT source data): PASS
- Direct loader call and exact fallback on unsupported dataset scripts, using a stub: PASS
- ClassLabel numeric ID mapping, rejection of invalid labels and orders: PASS
- Duplicate/overlap checks, stratified sampling and output schema: PASS
- Validation-only default and opt-in test behavior: PASS

## Tests NOT completed here

- Real Hugging Face dataset download: NOT RUN. The runner has no external DNS/PyPI access.
- Real dataset split sizes, frequencies, loss convergence and actual Macro-F1: NOT MEASURED.
- End-to-end `load_dataset()` against a live Hugging Face server: NOT VERIFIED.

Re-run first with `python run_experiment.py --audit-only --strict-source-counts --output-dir outputs/dataset_audit` on a machine with Internet access and the requirements installed. This package contains no actual UIT-VSFC rows and makes no claim about achieved accuracy/F1.

## Extended validation (2026-10-27)

The new runner is validation-only. It verifies the audited ordered-content hashes, split counts, label counts and recorded Python/NumPy/pandas/scikit-learn versions before fitting. Its `evaluate` function receives only the train and validation frames. The dataset's test split is read solely for the existing content audit; it is never passed to a model. The existing CLI is unchanged.

```sh
.venv/bin/python -m unittest discover -s tests -v
# Audit baseline first (writes audit evidence):
.venv/bin/python scripts/verify_baseline.py
# Use fresh directories; the runner refuses to overwrite a prior run:
.venv/bin/python extended_validation.py --smoke --output-dir /tmp/uit-vsfc-smoke-new
.venv/bin/python extended_validation.py
.venv/bin/python scripts/build_extended_report.py
# Plotting requires matplotlib, separately installed; set MPLCONFIGDIR if needed:
python scripts/plot_extended.py
```

This run used a copy of the existing official Hugging Face cache in `/private/tmp/uit-vsfc-hf`, with `HF_HOME=/private/tmp/uit-vsfc-hf HF_HUB_OFFLINE=1` and `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. The copy is a local cache, not evidence of the live revision; content hashes were verified. The baseline manifest does not record the `datasets` version, so equality of that package version cannot be established retrospectively. Extended summaries use sample standard deviation (`ddof=1`); the old baseline summary uses population standard deviation (`ddof=0`). The full-data row has only one independent sample and no standard deviation in extended feature summaries. The report generator verifies result shapes, sample hashes, matching baseline scores for the previously observed seeds, confusion matrices, and absence of test result/prediction files. It reads existing result CSVs and does not fit models.

`final_test_plan.json` is a proposed plan only, not authorization. Do not run the legacy `--include-test` command or any test prediction until the user explicitly approves the frozen plan.

## Frozen final-test runner

`run_final_test.py` is separate from the pilot's `--include-test` path. It reads the six configurations from the frozen plan, requires its exact file SHA-256 and checks the existing saved train indices, dataset hashes, source schema, model parameters and recorded library versions before fitting. The output directory must not already exist. The runner writes six sets of 3,166 predictions, metrics, confusion matrices, declared comparisons and one paired bootstrap for A2 − A1. It records the plan's offline-provenance limitation. An interrupted run has `test_evaluated: null` because test evaluation may have started; do not assume it is safe to repeat without inspection.

This read-only preflight does **not** load any dataset or run models:

```sh
.venv/bin/python run_final_test.py --check
```

After the user explicitly approves execution of the plan, the runner requires both the approval flag and the exact approved plan hash. The following is a future command, **not approval**:

```sh
.venv/bin/python run_final_test.py --approve-final-test \
  --plan-sha256 8ffa49ee313d7d30413007cc82ddc374ced219db9d714c197bb359b30d5b2a89
```

The plan hash pins the file bytes: any revision requires a new review and a corresponding runner update. The runner verifies all dataset hashes at execution, after approval and before model fitting. The recorded `datasets` version remains unavailable for the historical baseline, and the plan's source revision note remains applicable.
