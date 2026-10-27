#!/usr/bin/env python3
"""Verify frozen validation predictions and prepare a blind qualitative sample."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import os
import numpy as np
import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
import run_experiment as base
OUT = ROOT / 'outputs/extended_validation/neutral_blind_review'
SOURCE_ID = base.DATASET_ID
VALIDATION_SHA256 = 'b0e7ff0e1da48b3c1f4f4923a0d39e5136bb3def7863b22cd662e35bebcc0ae7'
VALIDATION_FINGERPRINT = '422c5b0a41e3786e'
PREDICTIONS = ROOT/'outputs/extended_validation/error_analysis/validation_predictions.csv'
BASELINE_RESULTS = ROOT/'outputs/main_validation/validation_results.csv'
BASELINE_CONFUSIONS = ROOT/'outputs/main_validation/validation_confusions.csv'
BASELINE_MANIFEST = ROOT/'outputs/main_validation/manifest.json'
PREDICTION_RUN = ROOT/'outputs/extended_validation/run_manifest.json'
SEED = 42

def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def load_official_validation():
    try:
        raw=load_dataset(SOURCE_ID, split='validation')
        method='load_dataset_main_validation_split'
    except RuntimeError as exc:
        if 'dataset scripts are no longer supported' not in str(exc).lower(): raise
        raw=load_dataset(SOURCE_ID, revision=base.PARQUET_REVISION, split='validation')
        method='load_dataset_official_parquet_validation_split'
    frame=base.normalize_split('validation',raw)
    fingerprint=getattr(raw,'_fingerprint',None)
    if len(frame)!=1583: raise ValueError(f'Expected 1,583 validation rows, got {len(frame)}')
    digest=base.content_sha256(frame)
    if digest!=VALIDATION_SHA256: raise ValueError(f'Validation ordered content hash mismatch: {digest}')
    if fingerprint!=VALIDATION_FINGERPRINT: raise ValueError(f'Validation fingerprint mismatch: {fingerprint}')
    return frame, {'dataset_id':SOURCE_ID,'split':'validation','loading_method':method,'fingerprint':fingerprint,'ordered_content_sha256':digest,'rows':len(frame)}

def metric_close(observed, expected):
    return set(observed)==set(expected) and all(np.isclose(float(observed[k]),float(expected[k]),rtol=0,atol=1e-12) for k in expected)

def verify_predictions(validation):
    preds=pd.read_csv(PREDICTIONS,keep_default_na=False)
    need={'row_index','sentence','true_label','pred_none','pred_balanced'}
    if not need.issubset(preds): raise ValueError(f'Missing prediction columns: {sorted(need-set(preds))}')
    if len(preds)!=1583 or preds.row_index.tolist()!=list(range(1583)):
        raise ValueError('Predictions are not exactly aligned to validation rows 0..1582')
    if not np.array_equal(preds.sentence.to_numpy(dtype=str),validation.sentence.to_numpy(dtype=str)):
        raise ValueError('Saved prediction sentence alignment differs from official validation')
    if not np.array_equal(preds.true_label.to_numpy(dtype=str),validation.label.to_numpy(dtype=str)):
        raise ValueError('Saved gold labels differ from official validation')
    run=json.loads(PREDICTION_RUN.read_text())
    expected_hash=VALIDATION_SHA256
    if run.get('audit_hashes',{}).get('validation')!=expected_hash:
        raise ValueError('Prediction run validation content hash differs')
    if run.get('test_evaluated') is not False:
        raise ValueError('Validation prediction run manifest does not affirm validation-only execution')
    baseline=json.loads(BASELINE_MANIFEST.read_text())
    if baseline['audit']['splits']['validation']['ordered_content_sha256']!=expected_hash:
        raise ValueError('Baseline validation hash differs')
    results=pd.read_csv(BASELINE_RESULTS)
    conf=pd.read_csv(BASELINE_CONFUSIONS)
    summaries={}
    expected_pairs={'none':'none','balanced':'balanced'}
    for weight in expected_pairs:
        label='pred_'+weight
        y=validation.label.to_numpy(); p=preds[label].to_numpy()
        got=base.metrics(y,p)
        row=results[(results.seed==42)&(results.train_fraction==1.0)&(results.class_weight==weight)]
        if len(row)!=1: raise ValueError(f'Expected one full-train seed-42 baseline result for {weight}')
        reference=row.iloc[0].to_dict()
        if not metric_close(got,{k:reference[k] for k in got}):
            raise ValueError(f'Metrics differ from baseline for {weight}')
        mat=base.confusion_matrix(y,p,labels=list(base.LABELS))
        baseline_cm=conf[(conf.seed==42)&(conf.train_fraction==1.0)&(conf.class_weight==weight)]
        expected={(r.actual,r.predicted):int(r['count']) for _,r in baseline_cm.iterrows()}
        actual={(a,b):int(mat[i,j]) for i,a in enumerate(base.LABELS) for j,b in enumerate(base.LABELS)}
        if actual!=expected: raise ValueError(f'Confusion matrix differs from baseline for {weight}')
        summaries[weight]=got
    pred_none=preds.pred_none.to_numpy(); pred_balanced=preds.pred_balanced.to_numpy(); gold=validation.label.to_numpy()
    groups=np.select([(pred_none==gold)&(pred_balanced==gold),(pred_none!=gold)&(pred_balanced==gold),(pred_none!=gold)&(pred_balanced!=gold)],['both_correct','rescued','both_wrong'],default='harmed')
    check=pd.DataFrame({'gold':gold,'error_group':groups})
    counts={f'{label}_{group}':int(((check.gold==label)&(check.error_group==group)).sum()) for label in base.LABELS for group in ['both_correct','rescued','both_wrong','harmed']}
    if counts['neutral_both_correct']!=10 or counts['neutral_rescued']!=31 or counts['neutral_both_wrong']!=32:
        raise ValueError(f'Neutral error transition discrepancy: {counts}')
    if counts['negative_harmed']+counts['positive_harmed']!=66:
        raise ValueError(f'Negative/Positive harmed discrepancy: {counts}')
    return preds, groups, {'n_rows':len(preds),'row_indices_sha256':hashlib.sha256(np.arange(len(preds),dtype='<i8').tobytes()).hexdigest(),'metrics':summaries,'error_group_counts':counts,'confirmed_transitions':{'neutral_both_correct':10,'neutral_rescued':31,'neutral_both_wrong':32,'negative_positive_harmed':66}}

def build_sample(validation,predictions,groups,provenance,verification):
    if OUT.exists(): raise FileExistsError(f'{OUT} already exists; refusing to overwrite review material')
    frame=pd.DataFrame({'validation_row_index':np.arange(len(validation)), 'sentence':validation.sentence.to_numpy(), 'gold_label':validation.label.to_numpy(), 'prediction_none':predictions.pred_none.to_numpy(), 'prediction_balanced':predictions.pred_balanced.to_numpy(), 'error_group':groups})
    neutral=np.flatnonzero(frame.gold_label.to_numpy()=='neutral')
    harmed=np.flatnonzero((frame.gold_label.isin(['negative','positive'])) & (frame.error_group=='harmed'))
    both_correct=np.flatnonzero((frame.gold_label.isin(['negative','positive'])) & (frame.error_group=='both_correct'))
    if len(neutral)!=73 or len(harmed)!=66 or len(both_correct)<20: raise ValueError('Candidate pool count mismatch')
    rng=np.random.RandomState(SEED)
    selected_harmed=np.sort(rng.choice(harmed,size=30,replace=False))
    selected_correct=np.sort(rng.choice(both_correct,size=20,replace=False))
    selected=np.concatenate([neutral,selected_harmed,selected_correct])
    if len(np.unique(selected))!=123: raise ValueError('Selected review sample has duplicate source rows')
    order=rng.permutation(len(selected)); selected=selected[order]
    # Stable pseudonymous IDs; no source row number or label is encoded in the ID.
    ids=[hashlib.sha256(f'UIT-VSFC-blind-review-v1|{SEED}|{int(i)}'.encode()).hexdigest()[:16].upper() for i in selected]
    blind=pd.DataFrame({'review_id':ids,'sentence':frame.sentence.iloc[selected].to_numpy()})
    key=pd.DataFrame({'review_id':ids,'validation_row_index':selected,'gold_label':frame.gold_label.iloc[selected].to_numpy(),'prediction_none':frame.prediction_none.iloc[selected].to_numpy(),'prediction_balanced':frame.prediction_balanced.iloc[selected].to_numpy(),'error_group':frame.error_group.iloc[selected].to_numpy()})
    # Shuffle annotation key independently, so file ordering does not reveal presentation order.
    key=key.iloc[rng.permutation(len(key))].reset_index(drop=True)
    OUT.mkdir(parents=True,exist_ok=False)
    blind.to_csv(OUT/'blind_review.csv',index=False,encoding='utf-8')
    key.to_csv(OUT/'annotation_key_private.csv',index=False,encoding='utf-8')
    os.chmod(OUT/'annotation_key_private.csv',0o600)
    counts={'total':len(selected),'neutral_all':int((key.gold_label=='neutral').sum()),'negative_positive_harmed_sampled':int((key.error_group=='harmed').sum()),'negative_positive_both_correct_sampled':int(((key.error_group=='both_correct') & (key.gold_label!='neutral')).sum()),'gold_label':{str(k):int(v) for k,v in key.gold_label.value_counts().sort_index().items()},'error_group':{str(k):int(v) for k,v in key.error_group.value_counts().sort_index().items()}}
    manifest={'status':'BLIND_REVIEW_PACKAGE_READY','dataset':provenance,'source_revision_note':json.loads(PREDICTION_RUN.read_text()).get('offline_cache_note',''), 'prediction_source':str(PREDICTIONS.relative_to(ROOT)),'prediction_source_sha256':sha_file(PREDICTIONS),'baseline_results_sha256':sha_file(BASELINE_RESULTS),'baseline_confusions_sha256':sha_file(BASELINE_CONFUSIONS),'validation_verification':verification,'selection':{'seed':SEED,'unit':'validation row','logic':['include all validation rows whose gold label is neutral','sample 30 without replacement from negative/positive rows harmed by Balanced','sample 20 without replacement from negative/positive rows correctly classified by both models','assign deterministic pseudonymous IDs, then shuffle blind presentation order'],'candidate_pool_counts':{'neutral':len(neutral),'negative_positive_harmed':len(harmed),'negative_positive_both_correct':len(both_correct)},'selected_counts':counts,'id_scheme':'first 16 uppercase hex characters of SHA-256 over fixed namespace, seed and validation row index'},'files':{'blind_review.csv':{'sha256':sha_file(OUT/'blind_review.csv'),'rows':len(blind),'columns':list(blind.columns)},'annotation_key_private.csv':{'sha256':sha_file(OUT/'annotation_key_private.csv'),'rows':len(key),'columns':list(key.columns),'permissions':'0600'}}}
    (OUT/'selection_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return manifest

def main():
    validation,provenance=load_official_validation()
    predictions,groups,verification=verify_predictions(validation)
    manifest=build_sample(validation,predictions,groups,provenance,verification)
    print(json.dumps({'status':manifest['status'],'dataset':manifest['dataset'],'verification':verification,'selection':manifest['selection']['selected_counts'],'files':manifest['files'],'output_dir':str(OUT)},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
