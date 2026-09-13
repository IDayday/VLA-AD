"""Independent real-data checks and supplementary paired tables; no reselection."""
from common_mass import *
import pandas as pd
from summarize import bootstrap

def main():
    m=OUT/'metrics';raw=pd.read_parquet(m/'candidate_metrics.parquet')
    frozen=read(OUT/'manifests/selection_frozen.json');assert frozen['selection_hashes']
    checks=[]
    smoke=read(OUT/'manifests/sampler_identity.json')['checks']
    for item in smoke:
        token=item['token'];r=np.load(OUT/'cache/raw'/f'{token}.npz')['trajectories']
        b=np.load(OUT/'cache/rollouts/B'/f'{token}.npz')['trajectories']
        bs=np.load(OUT/'cache/scores/rollouts/B'/f'{token}.npz')['scores']
        cs=np.load(OUT/'cache/scores/raw'/f'{token}.npz')['scores']
        table=raw[raw.token==token].set_index('raw_index').sort_index()
        maxerr=0.
        # Independent per-candidate loop, explicitly XY norm then time mean.
        for i,c in enumerate(r):
            ade=np.sqrt(((b[:,:,:2].astype(float)-c[None,:,:2])**2).sum(axis=2)).mean(axis=1)
            hit=ade<=.5;safe=np.all(bs[:,[0,1,3,5]]>=1-1e-8,axis=1)
            quality=bs[:,6]*100>=cs[i,6]*100-1-1e-10
            row=table.loc[i]
            assert int(hit.sum())==row.hit_count_B
            for key,value in [('p_B',hit.sum()/1024),('p_B_safe',(hit&safe).sum()/1024),('p_B_quality',(hit&safe&quality).sum()/1024)]:
                maxerr=max(maxerr,abs(row[key]-value));assert abs(row[key]-value)<1e-12
            if hit.any():assert abs(row.local_mean_PDMS-bs[hit,6].mean()*100)<1e-9
            else:assert np.isnan(row.local_mean_PDMS) and np.isnan(row.candidate_local_gain)
        checks.append(dict(token=token,candidates=len(r),B_samples=len(b),probability_max_abs=maxerr))
    same=0;strict_empty=0
    for scene in scenes():
        path=OUT/'cache/selection'/f'{scene["token"]}.json'
        assert sha(path)==frozen['selection_hashes'][scene['token']]
        p=read(path)['variants']['primary']
        same+=p['grpo_mass_pc']['operational_indices']==p['score']['operational_indices']
        strict_empty+=len(p['grpo_mass_pc']['strict_indices'])==0
    diagnostics=dict(raw_candidates=len(raw),raw_scenes=raw.token.nunique(),pc_score_identical_ordered_selections=same,pc_empty_strict_scenes=strict_empty,max_hits_A=int(raw.hit_count_A.max()),max_hits_B=int(raw.hit_count_B.max()),raw_A_mass_qualified=int((raw.hit_count_A>=31).sum()),raw_B_at_least_10_hits=int((raw.hit_count_B>=10).sum()),raw_zero_hit_A_fraction=float((raw.hit_count_A==0).mean()),raw_zero_hit_B_fraction=float((raw.hit_count_B==0).mean()),nearest_ADE_B_quantiles={str(q):float(raw.nearest_ADE_B.quantile(q)) for q in [0,.25,.5,.75,.9,.95,.99,1]})
    save(OUT/'audits/independent_real_data_review.json',dict(status='PASS',completed_at=utc(),checks=checks,diagnostics=diagnostics,selection_manifest_hash=sha(OUT/'manifests/selection_frozen.json')))
    # Every frozen sensitivity, all baselines, both pool types: descriptive only.
    # The primary 12-test family is unchanged and remains in paired_comparisons.
    pools=pd.read_parquet(m/'all_variant_pool_scene_metrics.parquet');rows=[]
    metrics=cfg()['analysis']['primary_endpoints']+['conservative_feasible_fraction','QualityMatchedPoolMass']
    for variant in [v['name'] for v in frozen['variants'] if v['name'] not in ['primary','no_il_native']]:
        for pt in ['operational','strict']:
            p=pools[(pools.variant==variant)&(pools.pool_type==pt)]
            pc=p[p.method=='grpo_mass_pc'].set_index('token')
            for method in METHODS[:3]:
                other=p[p.method==method].set_index('token');joint=pc.join(other,lsuffix='_pc',rsuffix='_other')
                for metric in metrics:
                    values=joint[metric+'_pc']-joint[metric+'_other']
                    rows.append(dict(variant=variant,pool_type=pt,comparator=method,metric=metric,interpretation='PRESPECIFIED_SENSITIVITY_DESCRIPTIVE',**bootstrap(values,f'sensitivity:{variant}:{pt}:{method}:{metric}')))
    pd.DataFrame(rows).to_csv(m/'sensitivity_paired_comparisons.csv',index=False)
    print(json.dumps(diagnostics,indent=2),flush=True)

if __name__=='__main__':main()
