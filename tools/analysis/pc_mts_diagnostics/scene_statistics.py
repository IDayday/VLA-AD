"""Paired scene statistics with transparent missing-subset denominators."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon,rankdata
from common import *

def summary_and_tests(df,group,out,prefix,replicates=3000):
    columns=[c for c in df.select_dtypes(include=np.number).columns if c not in ['group']]
    categories=list(dict.fromkeys(df[group]));tokens=sorted(df.token.unique());rng=np.random.default_rng(20260913)
    indices=rng.integers(0,len(tokens),(replicates,len(tokens)))
    summary=[];matrices={}
    for cat in categories:
        part=df[df[group]==cat].set_index('token').reindex(tokens);matrix=part[columns].to_numpy(float);matrices[cat]=matrix
        for j,col in enumerate(columns):
            v=matrix[:,j];finite=np.isfinite(v);n=int(finite.sum())
            if not n:summary.append(dict(group=cat,metric=col,mean=np.nan,median=np.nan,ci_low=np.nan,ci_high=np.nan,n_scenes=0));continue
            samples=v[indices];counts=np.isfinite(samples).sum(1);boots=np.nansum(samples,axis=1)/np.maximum(counts,1);boots=boots[counts>0]
            summary.append(dict(group=cat,metric=col,mean=float(np.nanmean(v)),median=float(np.nanmedian(v)),ci_low=float(np.quantile(boots,.025)),ci_high=float(np.quantile(boots,.975)),n_scenes=n))
    pairs=[('mts_8692','official_il'),('mts_8751','official_il'),('grpo_9041','official_il'),('apr_9145','official_il'),('apr_9145','grpo_9041')] if group=='checkpoint' else [(a,b) for i,a in enumerate(categories) for b in categories[:i]]
    tests=[]
    for a,b in pairs:
        for j,col in enumerate(columns):
            delta=matrices[a][:,j]-matrices[b][:,j];finite=np.isfinite(delta);d=delta[finite]
            if not len(d):continue
            samples=delta[indices];counts=np.isfinite(samples).sum(1);boots=np.nansum(samples,axis=1)/np.maximum(counts,1);boots=boots[counts>0]
            nz=d[np.abs(d)>1e-12]
            p=float(wilcoxon(nz,method='approx').pvalue) if len(nz)>1 else 1.
            ranks=rankdata(abs(nz));effect=float((ranks[nz>0].sum()-ranks[nz<0].sum())/ranks.sum()) if len(nz) else 0.
            tests.append(dict(a=a,b=b,metric=col,mean_difference=float(d.mean()),median_difference=float(np.median(d)),ci_low=float(np.quantile(boots,.025)),ci_high=float(np.quantile(boots,.975)),wilcoxon_p=p,rank_biserial_effect=effect,n_scenes=len(d),fraction_a_gt_b=float((d>0).mean())))
    pd.DataFrame(summary).to_csv(out/(prefix+'_summary.csv'),index=False);pd.DataFrame(tests).to_csv(out/(prefix+'_paired_tests.csv'),index=False)
    return pd.DataFrame(summary)
