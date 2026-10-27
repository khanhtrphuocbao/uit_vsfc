from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('outputs/extended_validation'); dest=p/'figures'; dest.mkdir(exist_ok=True)
r=pd.read_csv(p/'neutral_ratio/summary.csv'); s=pd.read_csv(p/'seed_stability/results.csv'); f=pd.read_csv(p/'feature_comparison/summary.csv')
for metric,name in [('macro_f1','macro_f1_by_neutral_ratio'),('neutral_f1','neutral_f1_by_neutral_ratio')]:
 fig,ax=plt.subplots(figsize=(7,4))
 for w in ['none','balanced']:
  d=r[r.class_weight==w].sort_values('neutral_count'); ax.plot(d.neutral_count/1142*100,d[metric+'_mean'],marker='o',label=w)
 ax.set(xlabel='Neutral trong train (%)',ylabel=metric,title='Validation: thành phần train cố định 1.142'); ax.legend(title='class_weight'); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(dest/(name+'.png'),dpi=160); plt.close(fig)
fig,ax=plt.subplots(figsize=(7,5))
for w in ['none','balanced']:
 d=r[r.class_weight==w].sort_values('neutral_count'); ax.plot(d.neutral_recall_mean,d.neutral_precision_mean,marker='o',label=w)
 for _,z in d.iterrows(): ax.annotate(f"{z.neutral_count/1142:.0%}",(z.neutral_recall_mean,z.neutral_precision_mean),xytext=(4,4),textcoords='offset points')
ax.set(xlabel='Neutral recall',ylabel='Neutral precision',title='Trung bình 3 seed theo thành phần train',xlim=(0,1),ylim=(0,1)); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(dest/'neutral_precision_recall.png',dpi=160); plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
for ax,frac in zip(axes,[.05,.1]):
 for w in ['none','balanced']:
  d=s[(s.train_fraction==frac)&(s.class_weight==w)].sort_values('seed'); ax.plot(d.seed,d.neutral_f1,marker='o',label=w)
 ax.set(title=f'Train {frac:.0%}',xlabel='Seed',ylabel='Neutral F1',ylim=(0,.4)); ax.grid(alpha=.25)
axes[1].legend(); fig.tight_layout(); fig.savefig(dest/'neutral_f1_seed_variation.png',dpi=160); plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(9,4),sharey=True)
for ax,frac in zip(axes,[.1,1.]):
 d=f[f.train_fraction==frac]; labels=[]; vals=[]
 for feature in ['word','char']:
  for weight in ['none','balanced']:
   labels.append(feature+'\n'+weight); vals.append(d[(d.feature==feature)&(d.class_weight==weight)].macro_f1_mean.iloc[0])
 ax.bar(labels,vals,color=['#437cb4','#ee9a43']*2); ax.set(title=f'Train {frac:.0%}',ylabel='Macro-F1',ylim=(0,1)); ax.grid(axis='y',alpha=.2)
fig.tight_layout(); fig.savefig(dest/'word_vs_character.png',dpi=160); plt.close(fig)
