#!/usr/bin/env python3
"""Validation-only experiments; never fit or predict on the held-out split."""
import argparse, hashlib, json, platform
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix
import run_experiment as base

ROOT=Path('outputs/extended_validation')
SEEDS=[42,52,62,72,82,92,102,112,122,132]
def digest(idx): return hashlib.sha256(np.asarray(idx,dtype='<i8').tobytes()).hexdigest()
def ratio_sample(y,neutral,seed):
    rng=np.random.RandomState(seed)
    pools={c:rng.permutation(np.flatnonzero(y==c)) for c in base.LABELS}
    remain=1142-neutral
    neg=round(remain*len(pools['negative'])/(len(pools['negative'])+len(pools['positive'])))
    return np.sort(np.concatenate([pools['negative'][:neg],pools['neutral'][:neutral],pools['positive'][:remain-neg]]))
def evaluate(train,validation,indices,seed,weight,feature):
    idx=np.asarray(indices,dtype=int)
    if len(set(idx))!=len(idx) or (idx<0).any() or (idx>=len(train)).any(): raise ValueError('invalid train indices')
    model=base.build_pipeline(weight=='balanced',seed)
    if feature=='char': model['tfidf'].set_params(analyzer='char',ngram_range=(3,5),token_pattern=None)
    elif feature!='word': raise ValueError('unknown feature')
    model.fit(train.sentence.iloc[idx].to_numpy(),train.label.iloc[idx].to_numpy())
    pred=model.predict(validation.sentence.to_numpy())
    result={'seed':seed,'class_weight':weight,'feature':feature,'n_train':len(idx),'indices_sha256':digest(idx),**base.metrics(validation.label.to_numpy(),pred)}
    return result,pred,model

def savejson(path,obj): path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=lambda value: value.__module__+'.'+value.__qualname__ if isinstance(value,type) else str(value)),encoding='utf8')
def record_sample(phase,seed,indices,train,**extra):
    y=train.label.to_numpy(); return {'phase':phase,'seed':seed,**extra,'split':'train','indices':indices.tolist(),'indices_sha256':digest(indices),'class_counts':{c:int(sum(y[indices]==c)) for c in base.LABELS}}
def conf_rows(result,y,pred):
    mat=confusion_matrix(y,pred,labels=list(base.LABELS)); return [{**{k:v for k,v in result.items() if k in ('seed','condition','neutral_count','class_weight','feature','n_train','indices_sha256')},'actual':a,'predicted':b,'count':int(mat[i,j])} for i,a in enumerate(base.LABELS) for j,b in enumerate(base.LABELS)]
def summarize(df,groups):
    metrics=['accuracy','macro_f1','weighted_f1','neutral_precision','neutral_recall','neutral_f1','negative_f1','positive_f1']
    out=df.groupby(groups,dropna=False)[metrics].agg(['mean','std','min','max','count']); out.columns=['_'.join(c) for c in out.columns]; return out.reset_index()
def paired(df,groups):
    cols=['accuracy','macro_f1','neutral_f1','neutral_precision','neutral_recall']
    x=df.pivot(index=groups+['seed'],columns='class_weight',values=cols)
    return pd.DataFrame({m:x[(m,'balanced')]-x[(m,'none')] for m in cols}).reset_index()
def load_verified():
    baseline=json.loads(Path('outputs/main_validation/manifest.json').read_text()); source,provenance=base.load_source(); data=base.prepare(source); audit=base.audit(data,source)
    actual={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__}
    if actual!=baseline['versions']: raise ValueError('environment differs from audit')
    for s in base.SPLITS:
        if audit['splits'][s]['ordered_content_sha256']!=baseline['audit']['splits'][s]['ordered_content_sha256'] or audit['splits'][s]['rows']!=base.EXPECTED_ROWS[s] or audit['splits'][s]['sentiment_counts']!=baseline['audit']['splits'][s]['sentiment_counts']: raise ValueError('source differs from audit: '+s)
    if json.loads((ROOT/'audit/baseline_verification.json').read_text())['status']!='PASS': raise ValueError('baseline verification not passed')
    return data['train'],data['validation'],audit,provenance

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true'); parser.add_argument('--output-dir',type=Path,default=ROOT); args=parser.parse_args()
    out=args.output_dir
    if (out/'run_manifest.json').exists(): raise FileExistsError('Output already exists; use a fresh output directory')
    train,val,audit,provenance=load_verified(); y=train.label.to_numpy(); vy=val.label.to_numpy(); manifests=[]
    for d in ('neutral_ratio','error_analysis','seed_stability','feature_comparison'): (out/d).mkdir(parents=True,exist_ok=True)
    savejson(out/'run_manifest.json',{'status':'STARTED','test_evaluated':False,'source':provenance,'offline_cache_note':'cache from official repository; main revision not verified offline','audit_hashes':{s:audit['splits'][s]['ordered_content_sha256'] for s in base.SPLITS},'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},'word_parameters':base.build_pipeline(False,42)['tfidf'].get_params(),'classifier_parameters':base.build_pipeline(False,42)['classifier'].get_params(),'character_override':{'analyzer':'char','ngram_range':[3,5],'token_pattern':None}})
    ratio=[]; cm=[]; rm=[]
    for seed in ([42] if args.smoke else SEEDS[:3]):
        for n in ([46] if args.smoke else [46,114,228]):
            idx=ratio_sample(y,n,seed); sample=record_sample('neutral_ratio',seed,idx,train,neutral_count=n); rm.append(sample); manifests.append(sample)
            for weight in ('none','balanced'):
                result,pred,_=evaluate(train,val,idx,seed,weight,'word'); result.update(condition={46:'original_like',114:'neutral_10',228:'neutral_20'}[n],neutral_count=n); ratio.append(result); cm+=conf_rows(result,vy,pred)
            print('ratio',seed,n,flush=True)
    rd=pd.DataFrame(ratio); rd.to_csv(out/'neutral_ratio/results.csv',index=False); summarize(rd,['condition','neutral_count','class_weight']).to_csv(out/'neutral_ratio/summary.csv',index=False); pd.DataFrame(cm).to_csv(out/'neutral_ratio/confusions.csv',index=False); savejson(out/'neutral_ratio/sample_manifest.json',rm)
    full=np.arange(len(train)); full_models={}; predictions={}
    for weight in ('none','balanced'):
        result,pred,model=evaluate(train,val,full,42,weight,'word'); full_models[weight]=(result,pred); predictions[weight]=pred
        if weight=='none': frame=pd.DataFrame({'row_index':val.row_index,'sentence':val.sentence,'true_label':vy})
        frame['pred_'+weight]=pred
        probs=model.predict_proba(val.sentence.to_numpy())
        for i,c in enumerate(model.classes_): frame[f'proba_{weight}_{c}']=probs[:,i]
        print('full word',weight,flush=True)
    frame['transition']=np.select([(frame.pred_none!=frame.true_label)&(frame.pred_balanced==frame.true_label),(frame.pred_none==frame.true_label)&(frame.pred_balanced!=frame.true_label),(frame.pred_none!=frame.true_label)&(frame.pred_balanced!=frame.true_label)],['rescued','harmed','both_wrong'],default='both_correct')
    frame.to_csv(out/'error_analysis/validation_predictions.csv',index=False); frame.groupby(['transition','true_label','pred_none','pred_balanced']).size().reset_index(name='count').to_csv(out/'error_analysis/prediction_transitions.csv',index=False)
    # Representative deterministic sample across label and transition, without asserting annotation errors.
    picked=[]
    for _,group in frame.groupby(['transition','true_label'],sort=True): picked.extend(group.index[:5].tolist())
    chosen=frame.loc[sorted(set(picked))].copy(); chosen['researcher_notes']=''; chosen.to_csv(out/'error_analysis/review_candidates.csv',index=False)
    stability=[]
    for frac in ([.05] if args.smoke else [.05,.10]):
        for seed in ([42] if args.smoke else SEEDS):
            idx=base.sample_indices(y,frac,seed); manifests.append(record_sample('seed_stability',seed,idx,train,train_fraction=frac))
            for weight in ('none','balanced'):
                result,_,_=evaluate(train,val,idx,seed,weight,'word'); result['train_fraction']=frac; stability.append(result)
            print('stability',frac,seed,flush=True)
    sd=pd.DataFrame(stability); sd.to_csv(out/'seed_stability/results.csv',index=False); summarize(sd,['train_fraction','class_weight']).to_csv(out/'seed_stability/summary.csv',index=False); paired(sd,['train_fraction']).to_csv(out/'seed_stability/paired_differences.csv',index=False)
    feature=[]
    for frac,seed in ([(.1,42)] if args.smoke else [(0.1,42),(0.1,52),(0.1,62),(1.0,42)]):
        idx=base.sample_indices(y,frac,seed); manifests.append(record_sample('feature_comparison',seed,idx,train,train_fraction=frac))
        for method in ('word','char'):
            for weight in ('none','balanced'):
                if frac==1 and method=='word': result=full_models[weight][0].copy()
                else: result,_,_=evaluate(train,val,idx,seed,weight,method)
                result['train_fraction']=frac; feature.append(result); print('feature',frac,seed,method,weight,flush=True)
    fd=pd.DataFrame(feature); fd.to_csv(out/'feature_comparison/results.csv',index=False); summarize(fd,['train_fraction','feature','class_weight']).to_csv(out/'feature_comparison/summary.csv',index=False)
    savejson(out/'sample_manifest.json',manifests)
    manifest=json.loads((out/'run_manifest.json').read_text()); manifest.update(status='COMPLETED',validation_fits={'neutral_ratio':len(rd),'error_analysis':2,'seed_stability':len(sd),'feature_comparison':len(fd)-2 if not args.smoke else len(fd)},test_evaluated=False,smoke=args.smoke); savejson(out/'run_manifest.json',manifest)
if __name__=='__main__': main()
