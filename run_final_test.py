#!/usr/bin/env python3
"""Execute only the frozen UIT-VSFC final-test plan, after explicit approval."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix
import run_experiment as base

ROOT = Path(__file__).resolve().parent
PLAN_PATH = ROOT / 'outputs/extended_validation/final_test_plan.json'
SAMPLES_PATH = ROOT / 'outputs/extended_validation/sample_manifest.json'
BASELINE_PATH = ROOT / 'outputs/main_validation/manifest.json'
BASELINE_SAMPLES_PATH = ROOT / 'outputs/main_validation/train_sample_manifest.json'
FROZEN_PLAN_SHA256 = '8ffa49ee313d7d30413007cc82ddc374ced219db9d714c197bb359b30d5b2a89'
EXPECTED = {'A1': (1.0, 'word', 'none'), 'A2': (1.0, 'word', 'balanced'),
            'B1': (1.0, 'char', 'none'), 'B2': (1.0, 'char', 'balanced'),
            'C1': (0.1, 'word', 'none'), 'C2': (0.1, 'word', 'balanced')}


def read_plan(path=PLAN_PATH):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FROZEN_PLAN_SHA256:
        raise ValueError('Plan SHA-256 differs from frozen plan; stop and review.')
    plan = json.loads(raw)
    validate_plan(plan)
    return plan


def validate_plan(plan):
    if plan.get('status') != 'AWAITING_USER_APPROVAL_NOT_EXECUTED':
        raise ValueError('Unexpected plan status')
    configs = plan.get('configurations', [])
    by_id = {c['id']: c for c in configs}
    if len(configs) != 6 or set(by_id) != set(EXPECTED):
        raise ValueError('Exactly six distinct fixed configurations required')
    for ident, (fraction, feature, weight) in EXPECTED.items():
        c = by_id[ident]
        if (c['train_fraction'], c['feature'], c['class_weight'], c['seed'], c['n_train']) != (
                fraction, feature, weight, 42, 1142 if fraction == .1 else 11426):
            raise ValueError(f'Unexpected configuration {ident}')
    if plan['dataset'] != base.DATASET_ID or plan['test_split'] != {'name':'test','n_rows':3166}:
        raise ValueError('Unexpected source or test size')
    if plan['primary_comparison']['configurations'] != ['A1','A2'] or plan['primary_comparison']['difference'] != 'A2 minus A1':
        raise ValueError('Unexpected primary comparison')
    bootstrap = plan['paired_bootstrap']
    if (bootstrap['samples'], bootstrap['seed'], bootstrap['interval'], bootstrap['zero_division']) != (10000, 2026, '95% percentile', 0):
        raise ValueError('Unexpected primary bootstrap')


def digest_indices(indices):
    return hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest()


def load_indices(plan, path=SAMPLES_PATH):
    samples = json.loads(path.read_text())
    baseline = json.loads(BASELINE_SAMPLES_PATH.read_text())
    indices = {}
    for fraction in (.1, 1.0):
        matches = [x for x in samples if x.get('phase') == 'feature_comparison' and x.get('train_fraction') == fraction and x.get('seed') == 42]
        originals = [x for x in baseline if x['train_fraction'] == fraction and x['seed'] == 42]
        if len(matches) != 1 or len(originals) != 1:
            raise ValueError('Expected exactly one saved sample and baseline record')
        record, original = matches[0], originals[0]
        vals = record['indices']
        if not isinstance(vals, list) or any(type(x) is not int for x in vals):
            raise ValueError('Invalid train indices')
        expected_size = 1142 if fraction == .1 else 11426
        if len(vals) != expected_size or len(set(vals)) != expected_size or vals != sorted(vals) or any(i < 0 or i >= 11426 for i in vals):
            raise ValueError('Train indices have wrong size, order, duplicates or range')
        if fraction == 1.0 and vals != list(range(11426)):
            raise ValueError('Full sample is not the complete train split')
        h = digest_indices(vals)
        configs = [c for c in plan['configurations'] if c['train_fraction'] == fraction]
        if h != record['indices_sha256'] or h != original['row_indices_sha256'] or original['n_train'] != expected_size or any(c['train_indices_sha256'] != h for c in configs):
            raise ValueError('Train sample SHA-256 does not match plan and baseline')
        if record['class_counts'] != original['class_counts']:
            raise ValueError('Sample class counts differ from baseline')
        indices[fraction] = np.array(vals, dtype=int)
    if not set(indices[.1]).issubset(set(indices[1.0])):
        raise ValueError('Low-data sample is not a subset of full train')
    return indices


def json_params(params):
    return json.loads(json.dumps(params, default=lambda x: x.__module__ + '.' + x.__qualname__ if isinstance(x, type) else str(x)))


def make_pipeline(config):
    model = base.build_pipeline(config['class_weight'] == 'balanced', config['seed'])
    if config['feature'] == 'char':
        model['tfidf'].set_params(analyzer='char', ngram_range=(3, 5), token_pattern=None)
    return model


def check_pipelines(plan):
    params = plan['parameters']
    recorded = json.loads((ROOT / 'outputs/extended_validation/run_manifest.json').read_text())
    recorded_classifier = dict(recorded['classifier_parameters'])
    recorded_classifier.pop('class_weight')
    if params['word_vectorizer'] != recorded['word_parameters'] or params['classifier'] != recorded_classifier or params['character_override'] != recorded['character_override']:
        raise ValueError('Plan parameters differ from the recorded validation pipeline')
    if params['model_variants'] != {'none': {'class_weight': None}, 'balanced': {'class_weight': 'balanced'}}:
        raise ValueError('Unexpected class weights')
    expected_vector = params['word_vectorizer']
    for config in plan['configurations']:
        model = make_pipeline(config)
        actual_vector = json_params(model['tfidf'].get_params())
        vector = dict(expected_vector)
        if config['feature'] == 'char':
            vector.update(params['character_override'])
        if actual_vector != vector:
            raise ValueError(f"TF-IDF parameters differ for {config['id']}")
        actual_classifier = json_params(model['classifier'].get_params())
        classifier = dict(params['classifier'])
        classifier.update(params['model_variants'][config['class_weight']])
        if actual_classifier != classifier:
            raise ValueError(f"Classifier parameters differ for {config['id']}")


def check_environment():
    baseline = json.loads(BASELINE_PATH.read_text())
    current = {'python':platform.python_version(), 'numpy':np.__version__, 'pandas':pd.__version__, 'sklearn':sklearn.__version__}
    if current != baseline['versions']:
        raise ValueError(f'Environment differs from baseline: {current} != {baseline["versions"]}')
    return current


def verify_source(data, plan):
    baseline = json.loads(BASELINE_PATH.read_text())
    for split in base.SPLITS:
        if split not in data or len(data[split]) != base.EXPECTED_ROWS[split]:
            raise ValueError(f'Source split missing or unexpected size: {split}')
        actual = base.content_sha256(data[split])
        if actual != plan['source_content_sha256'][split] or actual != baseline['audit']['splits'][split]['ordered_content_sha256']:
            raise ValueError(f'Source content SHA-256 mismatch: {split}')
        counts = {c: int((data[split].label == c).sum()) for c in base.LABELS}
        if counts != baseline['audit']['splits'][split]['sentiment_counts']:
            raise ValueError(f'Source label counts mismatch: {split}')
    return {s: base.content_sha256(data[s]) for s in base.SPLITS}


def verify_sample_counts(train, indices):
    baseline = json.loads(BASELINE_SAMPLES_PATH.read_text())
    for fraction, rows in indices.items():
        original = next(x for x in baseline if x['train_fraction'] == fraction and x['seed'] == 42)
        counts = {c: int((train.label.iloc[rows] == c).sum()) for c in base.LABELS}
        if counts != original['class_counts']:
            raise ValueError('Sample class counts differ from baseline')


def run_models(train, evaluation, configs, indices):
    results, predictions, confusions = [], [], []
    labels = evaluation.label.to_numpy()
    for config in configs:
        ident = config['id']
        model = make_pipeline(config)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            subset = indices[config['train_fraction']]
            model.fit(train.sentence.iloc[subset].to_numpy(), train.label.iloc[subset].to_numpy())
        pred = model.predict(evaluation.sentence.to_numpy())
        if len(pred) != len(evaluation):
            raise ValueError('Wrong number of predictions')
        results.append({'config_id':ident, **config, **base.metrics(labels, pred)})
        predictions.extend({'config_id':ident, 'row_index':int(i), 'true_label':str(y), 'predicted_label':str(p)} for i,y,p in zip(evaluation.row_index, labels, pred))
        cm = confusion_matrix(labels, pred, labels=list(base.LABELS))
        confusions.extend({'config_id':ident,'actual':a,'predicted':b,'count':int(cm[i,j])} for i,a in enumerate(base.LABELS) for j,b in enumerate(base.LABELS))
        print('Completed', ident, flush=True)
    return pd.DataFrame(results), pd.DataFrame(predictions), pd.DataFrame(confusions)


def paired_bootstrap(y_true, pred_none, pred_balanced, samples, seed):
    true = np.asarray(y_true); none = np.asarray(pred_none); balanced = np.asarray(pred_balanced)
    if not len(true) or len(true) != len(none) or len(true) != len(balanced):
        raise ValueError('Bootstrap arrays must be equally sized and nonempty')
    def macro(y,p):
        scores=[]
        for label in base.LABELS:
            tp=np.sum((y==label)&(p==label)); fp=np.sum((y!=label)&(p==label)); fn=np.sum((y==label)&(p!=label))
            den=2*tp+fp+fn; scores.append(2*tp/den if den else 0.)
        return float(np.mean(scores))
    rng=np.random.default_rng(seed); delta=np.empty(samples)
    for i in range(samples):
        sample=rng.integers(0,len(true),size=len(true))
        delta[i]=macro(true[sample],balanced[sample])-macro(true[sample],none[sample])
    return {'comparison':'A2 minus A1','metric':'macro_f1','samples':samples,'seed':seed,'interval':'95% percentile','point_estimate':macro(true,balanced)-macro(true,none),'lower':float(np.percentile(delta,2.5)),'upper':float(np.percentile(delta,97.5))}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate frozen plan and saved samples without loading dataset or fitting')
    parser.add_argument('--approve-final-test', action='store_true', help='Execute after explicit user approval')
    parser.add_argument('--plan-sha256', help='The SHA-256 of the explicitly approved frozen plan')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/final_test')
    args=parser.parse_args(argv)
    if args.check and (args.approve_final_test or args.plan_sha256):
        parser.error('--check cannot be combined with approval')
    if not args.check and (not args.approve_final_test or args.plan_sha256 != FROZEN_PLAN_SHA256):
        parser.error('Execution requires --approve-final-test and the frozen --plan-sha256; obtain explicit user approval first')
    plan=read_plan(); indices=load_indices(plan); versions=check_environment(); check_pipelines(plan)
    if args.check:
        print(f'Preflight PASS: frozen plan {FROZEN_PLAN_SHA256}, six configurations, stored indices, environment and pipeline parameters. Dataset not loaded; hashes require verification at execution.')
        return
    if args.output_dir.exists():
        raise FileExistsError('Output path exists; choose a new path')
    source,provenance=base.load_source()
    data=base.prepare(source)
    hashes=verify_source(data,plan); verify_sample_counts(data['train'],indices)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    manifest={'status':'STARTED','plan_sha256':FROZEN_PLAN_SHA256,'source':provenance,'source_revision_note':plan['source_revision_note'],'content_sha256':hashes,'versions':versions,'approval_flag':True,'test_evaluated':None,'configurations':plan['configurations']}
    manifest_path=args.output_dir/'manifest.json'
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    results,predictions,confusions=run_models(data['train'],data['test'],plan['configurations'],indices)
    counts=predictions.groupby('config_id').size()
    if len(results)!=6 or set(counts.index)!=set(EXPECTED) or not (counts==3166).all():
        raise ValueError('Unexpected number of test results')
    lookup=results.set_index('config_id'); metrics=['macro_f1','neutral_f1','neutral_precision','neutral_recall','accuracy']
    comparisons=[]
    for name, left, right, role in [('A2-A1','A2','A1','primary'),('A1-C1','A1','C1','secondary'),('A2-C2','A2','C2','secondary'),('C2-C1','C2','C1','secondary'),('B2-B1','B2','B1','exploratory'),('B1-A1','B1','A1','exploratory'),('B2-A2','B2','A2','exploratory')]:
        comparisons.append({'comparison':name,'role':role,**{m:float(lookup.loc[left,m]-lookup.loc[right,m]) for m in metrics}})
    a1=predictions[predictions.config_id=='A1'].sort_values('row_index'); a2=predictions[predictions.config_id=='A2'].sort_values('row_index')
    if not np.array_equal(a1.row_index,a2.row_index) or not np.array_equal(a1.true_label,a2.true_label):
        raise ValueError('A1 and A2 predictions are not paired')
    bs=plan['paired_bootstrap']; boot=paired_bootstrap(a1.true_label,a1.predicted_label,a2.predicted_label,bs['samples'],bs['seed'])
    results.to_csv(args.output_dir/'test_results.csv',index=False); predictions.to_csv(args.output_dir/'test_predictions.csv',index=False); confusions.to_csv(args.output_dir/'test_confusions.csv',index=False); pd.DataFrame(comparisons).to_csv(args.output_dir/'test_comparisons.csv',index=False)
    (args.output_dir/'primary_paired_bootstrap.json').write_text(json.dumps(boot,indent=2)+'\n')
    (args.output_dir/'report.md').write_text('# Final test\n\nPrimary: A2 − A1, macro-F1. Secondary and exploratory comparisons are reported separately in test_comparisons.csv.\n\nSource provenance: '+plan['source_revision_note']+'. Content hashes were checked against the plan and baseline.\n')
    manifest.update(status='COMPLETED',test_evaluated=True,predictions_per_configuration=3166,bootstrap_comparison='A2 minus A1')
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('Completed final test:',args.output_dir)

if __name__ == '__main__':
    main()
