#!/usr/bin/env bash
# ==============================================================================
# UIT-VSFC REPRODUCIBILITY EXPERIMENTAL SCRIPT
# Runs all baseline and proposed NBAC-PhoBERT models across fixed random seeds
# ==============================================================================

set -e
export PYTHONPATH=.
PYTHON_BIN=".venv/bin/python"

echo "=== 1. RUNNING BASELINE B0: TF-IDF + LOGISTIC REGRESSION ==="
$PYTHON_BIN scripts/run_b0_baseline.py

echo "=== 2. RUNNING HYPERPARAMETER GRID SEARCH ON VALIDATION SET ==="
$PYTHON_BIN scripts/run_grid_search.py

echo "=== 3. RUNNING FULL EXPERIMENTAL SUITE (B1 TO B7) ACROSS SEEDS ==="
$PYTHON_BIN scripts/run_all_experiments.py --seeds 42 100 2024 777 999 --epochs 4

echo "=== 4. PLOTTING FIGURES (CONFUSION MATRICES & LEARNING CURVES) ==="
$PYTHON_BIN scripts/plot_results.py

echo "=== 5. GENERATING FINAL STATISTICAL REPORT AND TABLES ==="
$PYTHON_BIN -c "
import pandas as pd
from scripts.eval_report import load_experiment_results, compute_paired_comparisons, generate_markdown_summary

df = load_experiment_results('outputs')
print('Loaded results shape:', df.shape)
summary = df.groupby('exp_name').agg({
    'seed': 'count',
    'test_macro_f1': ['mean', 'std'],
    'test_neu_f1': ['mean', 'std'],
    'test_neu_prec': 'mean',
    'test_neu_rec': 'mean',
    'test_neu_fp': 'mean',
    'test_neu_fn': 'mean',
    'test_acc': 'mean',
})
summary.columns = ['_'.join(c).strip('_') for c in summary.columns]
paired = compute_paired_comparisons(df, baseline_name='B1_ce')
md = generate_markdown_summary(summary, paired)
print('\n' + md)
with open('outputs/summary_table.md', 'w', encoding='utf-8') as f:
    f.write(md)
"

echo "=== EXPERIMENTAL PIPELINE COMPLETE ==="
