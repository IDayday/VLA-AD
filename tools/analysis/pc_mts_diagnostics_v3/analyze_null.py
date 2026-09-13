from common_v3 import *
def main():
    raw=pd.read_parquet(OUT/'metrics/A_raw_candidates.parquet');rows=[];holdout=set(tokens('holdout'))
    for token,g in raw.groupby('token',sort=False):
        a=np.load(OUT/'cache/null_support'/f'{token}.npz');domain=g.unique.to_numpy()&(g.q_holdout.to_numpy()>=95)&(g.PDMS.to_numpy()>=g.PDMS.quantile(.75));prom=domain&(a['q_holdout']<95)
        eligible=prom&g.hard_safe.to_numpy()&(g.PDMS.to_numpy()>=g.reference_PDMS.to_numpy()-1e-8)
        rows.append(dict(token=token,split='holdout' if token in holdout else 'train',old_high_quality_far=int(domain.sum()),null_promoted_count=int(prom.sum()),null_promotion_rate=float(prom.sum()/domain.sum()) if domain.any() else np.nan,null_newly_eligible_rate=float(eligible.sum()/domain.sum()) if domain.any() else np.nan))
    null=pd.DataFrame(rows);csv(null,'G_unchanged_policy_null.csv');d=pd.read_csv(OUT/'metrics/G_scene_promotion.csv').merge(null[['token','null_promotion_rate','null_newly_eligible_rate']],on='token',validate='many_to_one');stats=[]
    for scope in ['Full-1000','holdout-300']:
        n=null if scope=='Full-1000' else null[null.split=='holdout'];stats.append(dict(scope=scope,run='unchanged_IL',metric='null_promotion_rate',**bootstrap(n.null_promotion_rate,'G_null_'+scope)))
        stats.append(dict(scope=scope,run='unchanged_IL',metric='null_newly_eligible_rate',**bootstrap(n.null_newly_eligible_rate,'G_null_eligible_'+scope)))
        for run,g in d.groupby('run'):
            if scope=='holdout-300':g=g[g.split=='holdout']
            stats.append(dict(scope=scope,run=run,metric='promotion_minus_null',**bootstrap(g.frontier_promotion_rate-g.null_promotion_rate,'G_null_delta_'+scope+run)))
            stats.append(dict(scope=scope,run=run,metric='newly_eligible_minus_null',**bootstrap(g.newly_eligible_PC_rate-g.null_newly_eligible_rate,'G_null_eligible_delta_'+scope+run)))
    csv(pd.DataFrame(stats),'G_promotion_vs_null.csv');stage_audit('G_null',scenes=1000,R=128,Q=128,unchanged_checkpoint=v1.models()[0]['sha256'],predeclared_additive_control=read(OUT/'manifests/calibration_control_frozen.json'),primary_G_unchanged=True)
    print(pd.DataFrame(stats).query("scope=='holdout-300'")[['run','metric','mean','ci_low','ci_high']].to_string(index=False),flush=True)
if __name__=='__main__':main()
