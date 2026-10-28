"""Read-only data/prediction diagnostics for the KuMo strategy discussion.

No model training, input mutations, or submission generation.
"""
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path('/home/neva/Lomba/LOGIKA-UI')
OUT = Path('/tmp/logika_kumo_strategy_20261027.json')
EVAL = {'target', 'application_id', 'customer_id', 'application_datetime', 'decision_datetime', 'cutoff'}

def jsave(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

def ks(y, p):
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)

def metrics(y, p):
    return {'ks': ks(y,p), 'auc': float(roc_auc_score(y,p)),
            'mean_positive': float(p[y == 1].mean()), 'mean_negative': float(p[y == 0].mean())}

train = pl.read_csv(ROOT/'data/raw/train_applications.csv', try_parse_dates=True)
test = pl.read_csv(ROOT/'data/raw/test_applications.csv', try_parse_dates=True)
tr = pl.read_parquet(ROOT/'data/processed/train_features_v5.parquet')
te = pl.read_parquet(ROOT/'data/processed/test_features_v5.parquet')
folds = pl.read_parquet(ROOT/'data/processed/cv_folds_70k.parquet')
assert tr['application_id'].to_list() == train['application_id'].to_list()
assert te['application_id'].to_list() == test['application_id'].to_list()
assert folds['application_id'].to_list() == train['application_id'].to_list()
y = train['target'].to_numpy()
fold = folds['fold_seed42'].to_numpy()

def population(df):
    repay = df['cov_has_repayment'].to_numpy() == 1
    prev = df['cov_has_prev_apps'].to_numpy() == 1
    account = df['cov_has_account'].to_numpy() == 1
    return {'full': repay & account, 'thin': ~prev & ~account,
            'prev_no_repay': prev & ~repay}

pops, test_pops = population(tr), population(te)
assert np.all(sum(pops.values()) == 1)
assert np.all(sum(test_pops.values()) == 1)
result = {'sources': {'train':str(ROOT/'data/raw/train_applications.csv'),
                      'test':str(ROOT/'data/raw/test_applications.csv'),
                      'features':'data/processed/*_features_v5.parquet',
                      'friend':'/home/neva/Downloads/ens_q4cbf_full.csv'}}
result['data'] = {'train_rows':train.height, 'test_rows':test.height,
    'positive':int(y.sum()), 'negative':int((y == 0).sum()), 'target_rate':float(y.mean()),
    'train_unique_customers':train['customer_id'].n_unique(),
    'test_unique_customers':test['customer_id'].n_unique(),
    'customer_overlap':len(set(train['customer_id']) & set(test['customer_id'])),
    'train_decision_range':[str(train['decision_datetime'].min()),str(train['decision_datetime'].max())],
    'test_decision_range':[str(test['decision_datetime'].min()),str(test['decision_datetime'].max())],
    'term_values':train['term_months'].unique().sort().to_list(),
    'rate_counts':train.group_by('annual_interest_rate').agg(pl.len().alias('n'), pl.col('target').mean().alias('default_rate')).sort('annual_interest_rate').to_dicts(),
    'amount_grid_100k_fraction':float((train['loan_amount'] % 100000 == 0).mean())}
r = train['annual_interest_rate'].to_numpy()/12
expected_inst=train['loan_amount'].to_numpy()*r/(1-(1+r)**(-train['term_months'].to_numpy()))
result['data']['amortization_abs_error_quantiles']=np.quantile(np.abs(expected_inst-train['proposed_monthly_installment'].to_numpy()),[0,.5,.99,1]).tolist()
result['data']['monthly']=train.with_columns(pl.col('decision_datetime').dt.strftime('%Y-%m').alias('month')).group_by('month').agg(pl.len().alias('n'),pl.col('target').sum().alias('positive'),pl.col('target').mean().alias('default_rate')).sort('month').to_dicts()
result['populations']={name:{'train_n':int(mask.sum()),'positive':int(y[mask].sum()),'train_share':float(mask.mean()),
 'default_rate':float(y[mask].mean()),'test_n':int(test_pops[name].sum()),
 'previous_approved_nonzero':int((tr.filter(pl.Series(mask))['prev_approved_count'].fill_null(0)>0).sum())} for name,mask in pops.items()}

profile=[]
numeric=[]
hash_groups={}
for c,dtype in tr.schema.items():
    if c in EVAL: continue
    entry={'feature':c,'null_fraction':float(tr[c].null_count()/len(tr))}
    if dtype.is_numeric():
        a=tr[c].cast(pl.Float64).to_numpy()
        finite=np.isfinite(a)
        values=a[finite]
        entry.update({'nonfinite_fraction':float((~finite).mean()), 'n_unique':len(np.unique(values))})
        if len(values): entry.update({'min':float(values.min()),'max':float(values.max())})
        numeric.append(c)
        canonical=np.array(a,copy=True); canonical[~finite]=-9.99999999e99; canonical[canonical==0]=0
        digest=hashlib.blake2b(canonical.tobytes(),digest_size=16).hexdigest()
        hash_groups.setdefault(digest,[]).append(c)
    else: entry['n_unique']=tr[c].n_unique()
    profile.append(entry)
result['features']={'train_shape':tr.shape,'test_shape':te.shape,'predictors':len(profile),
 'numeric':len(numeric),'categorical':[c for c in tr.columns if c not in EVAL and tr[c].dtype == pl.String],
 'constant':[p['feature'] for p in profile if p['n_unique']<=1],
 'sparse_over_90pct':[p['feature'] for p in profile if p.get('nonfinite_fraction',p['null_fraction'])>.9],
 'exact_duplicate_groups':[v for v in hash_groups.values() if len(v)>1]}

friend=pl.read_csv('/home/neva/Downloads/ens_q4cbf_full.csv')
assert friend['application_id'].n_unique()==friend.height
friend_aligned=test.select('application_id').join(friend,on='application_id',how='left',validate='1:1')
p=friend_aligned['default_probability'].to_numpy()
assert np.all(np.isfinite(p))
n=len(p)
result['friend']={'rows':friend.height,'unique_ids':friend['application_id'].n_unique(),
 'missing_test_ids':int(friend_aligned['default_probability'].null_count()),
 'extra_ids':friend.join(test.select('application_id'),on='application_id',how='anti').height,
 'same_order_as_test':friend['application_id'].to_list()==test['application_id'].to_list(),
 'min':float(p.min()),'max':float(p.max()),'mean':float(p.mean()),'std':float(p.std()),
 'n_unique':len(np.unique(p)), 'quantiles':dict(zip(['0','1','5','10','25','50','75','90','95','99','100'],np.quantile(p,[0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]).tolist())),
 'rank_grid_r_over_n_max_error':float(np.max(np.abs(p-rankdata(p)/n))),
 'rank_grid_centered_max_error':float(np.max(np.abs(p-(rankdata(p)-.5)/n))),
 'top10_values_repeated':friend.group_by('default_probability').len().sort('len',descending=True).head(10).to_dicts(),
 'by_population':{name:{'n':int(mask.sum()),'mean':float(p[mask].mean()),'median':float(np.median(p[mask])),
 'top10pct_fraction':float((p[mask]>=np.quantile(p,.9)).mean())} for name,mask in test_pops.items()}}

prediction_correlations=[]
test_scores={}
paths=[*sorted((ROOT/'preds').glob('test*.parquet')),*sorted((ROOT/'submissions').glob('*.csv')),
       ROOT/'HPC/anchors/ens_q4cb.csv',Path('/home/neva/Downloads/Datasets_Data/ens_q4cb.csv'),Path('/home/neva/Downloads/Datasets_Data/ens_q4cf.csv')]
for path in paths:
    df=pl.read_parquet(path) if path.suffix=='.parquet' else pl.read_csv(path)
    aligned=test.select('application_id').join(df,on='application_id',how='left',validate='1:1')
    if aligned.height!=n: continue
    for c,dtype in df.schema.items():
        if c in EVAL or not dtype.is_numeric(): continue
        q=aligned[c].cast(pl.Float64).to_numpy()
        if not np.isfinite(q).all():continue
        name=f'{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path.name}:{c}'
        test_scores[name]=q
        item={'source':name,'pearson':float(np.corrcoef(p,q)[0,1]),'spearman':float(spearmanr(p,q).statistic),
              'mean_abs_rank_delta':float(np.mean(np.abs(rankdata(p)-rankdata(q)))/n),
              'max_abs_value_delta':float(np.max(np.abs(p-q))),
              'by_population':{name:float(spearmanr(p[mask],q[mask]).statistic) for name,mask in test_pops.items()}}
        prediction_correlations.append(item)
result['friend']['prediction_correlations']=sorted(prediction_correlations,key=lambda e:-e['spearman'])

oof_metrics=[]
for path in sorted((ROOT/'preds').glob('oof*.parquet')):
    df=pl.read_parquet(path)
    aligned=train.select('application_id').join(df,on='application_id',how='left',validate='1:1')
    for c,dtype in df.schema.items():
        if c in EVAL or not dtype.is_numeric():continue
        q=aligned[c].cast(pl.Float64).to_numpy()
        if not np.isfinite(q).all():continue
        item={'source':f'{path.name}:{c}',**metrics(y,q),
            'fold_ks':[ks(y[fold==f],q[fold==f]) for f in range(5)],
            'by_population':{name:metrics(y[mask],q[mask]) for name,mask in pops.items()}}
        oof_metrics.append(item)
result['oof_artifact_metrics']=sorted(oof_metrics,key=lambda e:-e['ks'])

rng=np.random.default_rng(20261027)
random_ks=[]
for _ in range(200):
    shuffled=rng.permutation(y)
    separation=np.cumsum(shuffled)/y.sum()-np.cumsum(1-shuffled)/(len(y)-y.sum())
    random_ks.append(float(np.max(np.abs(separation))))
result['random_baseline']={'repeats':len(random_ks),'ks_quantiles_2p5_50_97p5':np.quantile(random_ks,[.025,.5,.975]).tolist()}

univariate=[]
shift=[]
for c in numeric:
    a=tr[c].cast(pl.Float64).to_numpy(); b=te[c].cast(pl.Float64).to_numpy()
    finite=np.isfinite(a); finite_b=np.isfinite(b)
    if not finite.any() or np.unique(a[finite]).size<=1:continue
    median=float(np.median(a[finite])); a=np.where(finite,a,median); b=np.where(finite_b,b,median)
    item={'feature':c,**metrics(y,a),
          'by_population':{name:metrics(y[mask],a[mask]) for name,mask in pops.items()}}
    univariate.append(item)
    if finite_b.any():
        shift.append({'feature':c,'two_sample_ks':float(ks_2samp(a,b).statistic),
            'train_null':float((~finite).mean()),'test_null':float((~finite_b).mean()),
            'train_median':float(np.median(a)),'test_median':float(np.median(b))})
result['univariate']=sorted(univariate,key=lambda e:-e['ks'])
result['train_test_feature_shifts']=sorted(shift,key=lambda e:-e['two_sample_ks'])
friend_feature_corr=[]
for c in numeric:
    a=te[c].cast(pl.Float64).to_numpy(); m=np.isfinite(a)
    if not m.any() or len(np.unique(a[m]))<=1:continue
    a=np.where(m,a,float(np.median(a[m])))
    friend_feature_corr.append({'feature':c,'spearman':float(spearmanr(p,a).statistic)})
result['friend']['feature_correlations']=sorted(friend_feature_corr,key=lambda e:-abs(e['spearman']))

bins={}
for c in ['ecom_spending_30d','ecom_orders_30d','ecom_recency_days','repay_max_max_dpd','bureau_score_mean','declared_monthly_income','annual_interest_rate','acct_mean_balance_all']:
    a=tr[c].cast(pl.Float64).to_numpy();m=np.isfinite(a)
    boundaries=np.unique(np.quantile(a[m],[0,.25,.5,.75,1]))
    idx=np.searchsorted(boundaries[1:-1],a,side='right')
    entries=[]
    for bidx in range(len(boundaries)-1):
        mask=m&(idx==bidx)
        if mask.any():entries.append({'lower':float(boundaries[bidx]),'upper':float(boundaries[bidx+1]),'n':int(mask.sum()),'default_rate':float(y[mask].mean())})
    if (~m).any():entries.append({'bin':'missing','n':int((~m).sum()),'default_rate':float(y[~m].mean())})
    bins[c]=entries
result['binned_signals']=bins

OUT.write_text(json.dumps(result,indent=2,default=jsave))
print('DATA',json.dumps(result['data'],default=jsave))
print('POPULATIONS',json.dumps(result['populations']))
print('FEATURES',json.dumps(result['features'],default=jsave))
print('CSV',{k:v for k,v in result['friend'].items() if k not in ['prediction_correlations','feature_correlations','top10_values_repeated']})
print('PREDICTION_CORRELATIONS',json.dumps(result['friend']['prediction_correlations'][:12],default=jsave))
print('OOF_METRICS',json.dumps([{'source':e['source'],'ks':e['ks'],'auc':e['auc'],'fold_ks':e['fold_ks'],'population_ks':{k:v['ks'] for k,v in e['by_population'].items()}} for e in result['oof_artifact_metrics']],default=jsave))
print('UNIVARIATE_TOP',json.dumps([{k:v for k,v in e.items() if k!='by_population'} for e in result['univariate'][:20]],default=jsave))
print('SHIFT_TOP',json.dumps(result['train_test_feature_shifts'][:12],default=jsave))
print('FRIEND_FEATURE_TOP',json.dumps(result['friend']['feature_correlations'][:15],default=jsave))
print('RANDOM_BASELINE',result['random_baseline'])
print('OUTPUT',OUT)
