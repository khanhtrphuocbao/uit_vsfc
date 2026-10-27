import os,sys,json,hashlib,platform
from pathlib import Path
from collections import Counter
import numpy as np,pandas as pd,sklearn
sys.path.insert(0,os.getcwd())
import run_experiment as r
out=Path('outputs/extended_validation/audit'); out.mkdir(parents=True,exist_ok=True)
b=Path('outputs/main_validation'); m=json.loads((b/'manifest.json').read_text()); a=json.loads(Path('outputs/dataset_audit/manifest.json').read_text()); samples=json.loads((b/'train_sample_manifest.json').read_text()); res=pd.read_csv(b/'validation_results.csv'); summ=pd.read_csv(b/'validation_summary.csv'); conf=pd.read_csv(b/'validation_confusions.csv')
checks={}; details={}
checks['environment_match']={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__}==m['versions']
source,prov=r.load_source(); data=r.prepare(source); audit=r.audit(data,source)
checks['source_hash_schema_counts_match']=all(audit['splits'][s]['ordered_content_sha256']==m['audit']['splits'][s]['ordered_content_sha256']==a['audit']['splits'][s]['ordered_content_sha256'] and audit['splits'][s]['rows']==r.EXPECTED_ROWS[s] and audit['splits'][s]['sentiment_counts']==m['audit']['splits'][s]['sentiment_counts'] for s in r.SPLITS)
checks['30_complete_unique_configs']=len(res)==30 and set(res.train_fraction)=={.05,.1,.25,.5,1.} and set(res.seed)=={42,52,62} and set(res.class_weight)=={'none','balanced'} and len(res.groupby(['seed','train_fraction','class_weight']))==30
checks['no_test_evaluation']=m.get('runs',{}).get('test_evaluated') is False and m.get('runs',{}).get('test_evaluations')==0 and not list(b.glob('test_*.csv'))
y=data['train'].label.to_numpy(); sample_ok=True
for s in samples:
 idx=r.sample_indices(y,s['train_fraction'],s['seed']); digest=hashlib.sha256(idx.astype('<i8').tobytes()).hexdigest(); sample_ok &= digest==s['row_indices_sha256'] and len(idx)==s['n_train'] and dict(Counter(y[idx]))==s['class_counts']; group=res[(res.seed==s['seed']) & (res.train_fraction==s['train_fraction'])]; sample_ok &= len(group)==2 and set(group.class_weight)=={'none','balanced'} and (group.n_train==len(idx)).all()
checks['samples_reconstructed_paired']=len(samples)==15 and bool(sample_ok)
computed=r.make_summary(res); checks['summary_matches']=list(computed.columns)==list(summ.columns) and np.allclose(computed.select_dtypes('number'),summ.select_dtypes('number'),atol=1e-12,rtol=1e-12) and computed.class_weight.tolist()==summ.class_weight.tolist()
conf_ok=True
for _,row in res.iterrows():
 c=conf[(conf.seed==row.seed)&(conf.train_fraction==row.train_fraction)&(conf.class_weight==row.class_weight)]; mat=c.pivot(index='actual',columns='predicted',values='count').reindex(index=r.LABELS,columns=r.LABELS); conf_ok &= len(c)==9 and mat.to_numpy().sum()==len(data['validation']); actual=[]; pred=[]
 for i,cl in enumerate(r.LABELS):
  for j,pl in enumerate(r.LABELS): actual += [cl]*int(mat.iloc[i,j]); pred += [pl]*int(mat.iloc[i,j])
 metrics=r.metrics(np.array(actual),np.array(pred)); conf_ok &= all(np.isclose(row[k],v,atol=1e-12) for k,v in metrics.items())
checks['confusions_match_all_metrics']=bool(conf_ok) and len(conf)==270
full=res[res.train_fraction==1.0]; details['full_train_unique_sample_hashes']=len({s['row_indices_sha256'] for s in samples if s['train_fraction']==1}); details['full_train_metrics_equal_by_weight']={w: bool(g.drop(columns=['seed']).nunique().max()==1) for w,g in full.groupby('class_weight')}
checks['full_train_same_indices']=details['full_train_unique_sample_hashes']==1
keys=data['train'].sentence.map(r.audit_text_key); tmp=data['train'].assign(key=keys); conflicts=tmp.groupby('key').label.nunique(); selected=tmp[tmp.key.isin(conflicts[conflicts>1].index)][['row_index','sentence','label','topic']].rename(columns={'label':'sentiment'}); selected.to_csv(out/'conflicting_labels.csv',index=False)
checks['one_conflicting_text']=len(conflicts[conflicts>1])==1
result={'status':'PASS' if all(checks.values()) else 'BLOCKED','checks':checks,'details':details,'source':prov,'cache_note':'offline cache copied from Hugging Face official dataset cache; main loader cache identity is not proof of main revision','source_hashes':{s:audit['splits'][s]['ordered_content_sha256'] for s in r.SPLITS},'versions':m['versions'],'baseline_status':m['status'],'test_evaluated':False}
(out/'baseline_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
(out/'audit_notes.md').write_text('# Kiểm tra baseline\n\nTrạng thái: '+result['status']+'. Nội dung được nạp từ cache của repository chính thức; hash nội dung khớp cả hai manifest. Không xác nhận được revision main từ chế độ offline.\n\nHai nhãn khác nhau cho cùng văn bản chuẩn hóa phục vụ audit: xem conflicting_labels.csv. Đây là khả năng mơ hồ hoặc bất nhất, không đủ chứng cứ kết luận nhãn nào sai; giữ nguyên cả hai dòng.\n\nỞ 100%, ba seed có cùng indices; đây không phải ba mẫu train độc lập. '+json.dumps(details,ensure_ascii=False)+'\n\nCác kiểm tra: '+json.dumps(checks,ensure_ascii=False)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2)); print(selected.to_string(index=False))
