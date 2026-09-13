"""Scene-equal summaries, prespecified paired inference and honest denominators."""
from common_mass import *
import pandas as pd
from analyze_B import mean,median

def bootstrap(values,key,clusters=None):
    a=np.asarray(values,float);valid=np.isfinite(a);a=a[valid]
    if clusters is not None:clusters=np.asarray(clusters)[valid]
    if not len(a):return dict(n=0,mean_difference=np.nan,median_difference=np.nan,ci_low=np.nan,ci_high=np.nan,scene_win_fraction=np.nan,p_value=np.nan,cluster_count=0,bootstrap_replicates=3000)
    rng=np.random.default_rng(random_seed(str(key),'bootstrap',0));means=[];medians=[]
    if clusters is not None:
        keys=sorted(set(clusters));sums=np.array([a[clusters==k].sum() for k in keys]);counts=np.array([(clusters==k).sum() for k in keys])
    for start in range(0,3000,100):
        if clusters is None:
            v=a[rng.integers(len(a),size=(100,len(a)))];means.extend(v.mean(1));medians.extend(np.median(v,axis=1))
        else:
            ind=rng.integers(len(keys),size=(100,len(keys)));means.extend(sums[ind].sum(1)/counts[ind].sum(1))
    # Centered bootstrap estimates the null sampling distribution; paired scene
    # resampling is preserved for all members of the fixed primary test family.
    p=(np.sum(np.abs(np.asarray(means)-a.mean())>=abs(a.mean()))+1)/3001
    return dict(n=len(a),mean_difference=float(a.mean()),median_difference=float(np.median(a)),ci_low=float(np.quantile(means,.025)),ci_high=float(np.quantile(means,.975)),median_ci_low=float(np.quantile(medians,.025)) if medians else np.nan,median_ci_high=float(np.quantile(medians,.975)) if medians else np.nan,scene_win_fraction=float((a>1e-10).mean()),p_value=p,bootstrap_replicates=3000,cluster_count=len(set(clusters)) if clusters is not None else len(a))

def summarize_methods(df):
    keys=['variant','pool_type','method'];exclude=set(keys+['token','log','marginal_coverage'])
    numeric=[k for k in df.columns if k not in exclude and pd.api.types.is_numeric_dtype(df[k])];rows=[]
    for key,g in df.groupby(keys,sort=False):
        row=dict(zip(keys,key));row.update(full_scene_denominator=1000,observed_scene_count=g.token.nunique(),nonempty_scene_count=int((g.unique_parent_count>0).sum()),candidate_denominator=int(g.unique_parent_count.sum()),local_gain_candidate_denominator=int(g.local_gain_defined_candidate_count.sum()),local_gain_scene_denominator=int(g.local_gain_defined_scene_count.sum()))
        for k in numeric:row[k]=mean(g[k])
        v=next(v for v in read(OUT/'manifests/selection_frozen.json')['variants'] if v['name']==key[0]);row.update(epsilon=v['epsilon'],pmin=v['pmin'],gt_geometry_radius=v['radius'])
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    m=OUT/'metrics';pools=pd.read_parquet(m/'all_variant_pool_scene_metrics.parquet');selected=pd.read_parquet(m/'selected_pools_4x1000x16.parquet');raw=pd.read_parquet(m/'candidate_metrics.parquet');paired=pd.read_parquet(m/'source_quality_matched.parquet')
    summaries=summarize_methods(pools);summaries[summaries.variant=='primary'].to_csv(m/'method_summary.csv',index=False);summaries[summaries.variant=='no_il_native'].to_csv(m/'no_il_native_control.csv',index=False);summaries[~summaries.variant.isin(['primary','no_il_native'])].to_csv(m/'sensitivity.csv',index=False)
    comparisons=[];primary_metrics=cfg()['analysis']['primary_endpoints'];other_metrics=['conservative_feasible_fraction','mean_candidate_p_B_quality','QualityMatchedPoolMass','external_source_fraction','near_duplicate_representatives_0_10m','fallback_fraction']
    for variant in ['primary','no_il_native']:
        for pt in ['operational','strict']:
            subset=pools[(pools.variant==variant)&(pools.pool_type==pt)];pc=subset[subset.method=='grpo_mass_pc'].set_index('token')
            for method in METHODS[:3]:
                b=subset[subset.method==method].set_index('token');joint=pc.join(b,lsuffix='_pc',rsuffix='_base',how='outer');assert len(joint)==1000
                for k in primary_metrics+other_metrics:
                    a=joint[k+'_pc']-joint[k+'_base'];row=dict(variant=variant,pool_type=pt,comparator=method,metric=k,full_scene_denominator=1000,pc_defined_scenes=int(joint[k+'_pc'].notna().sum()),baseline_defined_scenes=int(joint[k+'_base'].notna().sum()),**bootstrap(a,f'{variant}:{pt}:{method}:{k}'))
                    log=bootstrap(a,f'log:{variant}:{pt}:{method}:{k}',joint.log_pc);row.update(log_cluster_ci_low=log['ci_low'],log_cluster_ci_high=log['ci_high'],log_cluster_count=log['cluster_count']);comparisons.append(row)
    comparisons=pd.DataFrame(comparisons);family=(comparisons.variant=='primary')&(comparisons.pool_type=='operational')&comparisons.metric.isin(primary_metrics);order=comparisons.loc[family].sort_values('p_value').index;running=0.
    for j,i in enumerate(order):running=max(running,min(1,comparisons.loc[i,'p_value']*(len(order)-j)));comparisons.loc[i,'Holm_p_primary_family12']=running
    comparisons.to_csv(m/'paired_comparisons.csv',index=False)
    matchresults=[]
    for pt,other in [(pt,m) for pt in ['operational','strict'] for m in METHODS[:3]]:
        group=paired[(paired.pool_type==pt)&(paired.comparator==other)]
        for metric in ['p_B','p_B_safe','p_B_quality','candidate_local_gain']:
            g=group[group.local_gain_pair_valid] if metric=='candidate_local_gain' else group
            values=g['delta_'+metric];pair=bootstrap(values,f'pair:{pt}:{other}:{metric}');cluster=bootstrap(values,f'pair_scene:{pt}:{other}:{metric}',g.token)
            scene=g.groupby('token')['delta_'+metric].mean();scene_equal=bootstrap(scene,f'match_scene_equal:{pt}:{other}:{metric}')
            matchresults.append(dict(pool_type=pt,comparator=other,metric=metric,status='DESCRIPTIVE_MATCHED' if len(g) else 'UNTESTED_COMMON_SUPPORT_UNAVAILABLE',matched_pair_count=len(g),matched_scene_count=g.token.nunique(),all_matched_pair_count=len(group),PDMS_gap_mean=mean(g.PDMS_gap_points),PDMS_gap_max=float(g.PDMS_gap_points.max()) if len(g) else np.nan,mean_difference=pair['mean_difference'],median_difference=pair['median_difference'],pair_ci_low=pair['ci_low'],pair_ci_high=pair['ci_high'],scene_cluster_ci_low=cluster['ci_low'],scene_cluster_ci_high=cluster['ci_high'],scene_equal_mean=scene_equal['mean_difference'],scene_equal_ci_low=scene_equal['ci_low'],scene_equal_ci_high=scene_equal['ci_high'],scene_win_fraction=scene_equal['scene_win_fraction']))
    if matchresults:pd.DataFrame(matchresults).to_csv(m/'source_quality_matched.csv',index=False)
    else:pd.DataFrame(columns=['pool_type','comparator','metric','matched_pair_count','matched_scene_count','scene_equal_mean','scene_equal_ci_low','scene_equal_ci_high']).to_csv(m/'source_quality_matched.csv',index=False)
    paired.groupby(['pool_type','comparator','exact_source']).agg(matched_pair_count=('token','size'),matched_scene_count=('token','nunique'),score_gap_mean=('PDMS_gap_points','mean'),score_gap_median=('PDMS_gap_points','median'),score_gap_max=('PDMS_gap_points','max'),p_B_delta_mean=('delta_p_B','mean')).reset_index().to_csv(m/'matched_source_breakdown.csv',index=False)
    source_rows=[]
    for source in sorted(raw.source.unique()):
        g=raw[raw.source==source];scene=g.groupby('token');row=dict(source=source,unique_candidates=len(g),covered_scenes=g.token.nunique(),scene_denominator=1000,mean_unique_per_covered_scene=len(g)/g.token.nunique(),raw_mean_PDMS=mean(scene.PDMS.mean()),raw_median_PDMS=median(g.PDMS),raw_mean_p_A=mean(scene.p_A.mean()),raw_mean_p_B=mean(scene.p_B.mean()),raw_hard_safe_fraction=mean(scene.hard_safe.mean()),raw_conservative_feasible_fraction=mean(scene.conservative_feasible.mean()),raw_local_gain_primary=mean(g[g.hit_count_B>=10].groupby('token').candidate_local_gain.mean()))
        for value in ['PDMS','p_A','p_B']:
            for q in [.1,.25,.5,.75,.9]:row[f'{value}_P{int(q*100)}']=float(g[value].quantile(q))
        for method in METHODS:
            h=selected[(selected.method==method)&selected.valid_mask.fillna(False)&(selected.source==source)];local=h[h.hit_count_B>=10]
            row.update({method+'_selected_count':len(h),method+'_selected_scene_count':h.token.nunique(),method+'_selected_fraction':len(h)/16000,method+'_PDMS':mean(h.groupby('token').PDMS.mean()),method+'_p_B':mean(h.groupby('token').p_B.mean()),method+'_local_gain':mean(local.groupby('token').candidate_local_gain.mean()),method+'_strict_fraction':mean(h.strict_qualified),method+'_fallback_fraction':mean(~h.strict_qualified.astype(bool))})
        source_rows.append(row)
    pre={s:0 for s in ['GT','GT_structured','ddv2','drivor','IL_native_C']};counts=[];matchcoverage=[];pareto=[]
    for scene in scenes():
        t=scene['token'];r=read(OUT/'cache/raw'/f'{t}.json')['metadata'];counts.append(dict(token=t,**r['source_counts_pre_dedup'],raw_generation_count=r['generation_count'],raw_unique_count=r['unique_count'],emergency_batches=r['emergency_batches']))
        for s,n in r['source_counts_pre_dedup'].items():pre[s]+=n
        s=read(OUT/'cache/selection'/f'{t}.json')
        for pt,c in s['matched_pairs'].items():
            for other,cov in c.items():matchcoverage.append(dict(token=t,pool_type=pt,comparator=other,**{k:v for k,v in cov.items() if k!='pairs'},matched_pair_count=len(cov['pairs'])))
        pp=s['variants']['primary'];a=set(pp['score']['strict_indices']);b=set(pp['pareto']['strict_indices']);au=pp['pareto']['pareto_audit'];pareto.append(dict(token=t,score_pareto_jaccard=len(a&b)/len(a|b) if a|b else np.nan,identical_set=a==b,front_count=au['front_count'],front_sizes=json.dumps(au['front_sizes']),EP_variance=au['objective_variances'][0],TTC_variance=au['objective_variances'][1],DDC_variance=au['objective_variances'][2]))
    for r in source_rows:r['generated_pre_dedup']=pre.get(r['source'],0)
    pd.DataFrame(source_rows).to_csv(m/'raw_source_summary.csv',index=False);pd.DataFrame(counts).to_csv(m/'raw_scene_counts.csv',index=False);pd.DataFrame(matchcoverage).to_csv(m/'matching_coverage.csv',index=False);pd.DataFrame(pareto).to_csv(m/'pareto_audit.csv',index=False)
    types=selected[selected.valid_mask.fillna(False)].groupby(['method','token','descriptive_type']).size().rename('count').reset_index();types.to_csv(m/'expert_gain_types.csv',index=False)
    save(OUT/'audits/statistical_analysis.json',dict(bootstrap_replicates=3000,primary_family_size=12,primary_pool='operational',scene_equal=True,local_gain_nonpaired_descriptive_only=True,matched_pairs_not_using_B=True,completed_at=utc()))
    print(summaries[(summaries.variant=='primary')&(summaries.pool_type=='operational')][['method','candidate_PDMS','mean_candidate_p_B','PoolMass','strict_16_scene_fraction']].to_string(index=False))
if __name__=='__main__':main()
