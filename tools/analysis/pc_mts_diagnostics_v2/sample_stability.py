"""User-requested 64-rollout primary comparison; 16/32 subsampling sensitivity only."""
import concurrent.futures
from v2_common import *
import pandas as pd
REPEATS=100

def work(scene):
    token=scene['token'];rng=np.random.default_rng(seed(token,'v2_sample_count_sensitivity'));indices=np.stack([rng.permutation(64) for _ in range(REPEATS)]);rows=[];scene_rows=[];ref=float(np.load(V1/'evaluator/rollouts/official_il'/f'{token}.npz')['reference'][0,6])*100
    for model in MODELS:
        t=np.load(V1/'rollouts'/model/f'{token}.npz')['trajectories'];s=np.load(V1/'evaluator/rollouts'/model/f'{token}.npz')['trajectories'];assert t.shape==(64,8,3);d=distance(t,t);f=feasible(s);positive=f&(s[:,6]*100>ref+1)
        for n in [16,32,64]:
            ix=indices[:,:n];xy=t[ix][...,:2];spread=np.sqrt(xy.var(1,ddof=1).sum(-1)).mean(-1);ade=d[ix[:,:,None],ix[:,None,:]].sum((1,2))/(n*(n-1));hit=positive[ix].reshape(REPEATS,n//8,8).any(-1).mean(-1)
            for k in range(REPEATS):rows.append(dict(checkpoint=model,n_rollouts=n,repeat=k,pairwise_ADE=ade[k],spread_auc=spread[k],hit8_1=hit[k]))
            scene_rows.append(dict(token=token,checkpoint=model,n_rollouts=n,mean_pairwise_ADE=float(ade.mean()),std_pairwise_ADE=float(ade.std(ddof=1)),pairwise_ADE_p025=float(np.quantile(ade,.025)),pairwise_ADE_p975=float(np.quantile(ade,.975)),mean_spread_auc=float(spread.mean()),std_spread_auc=float(spread.std(ddof=1)),mean_hit8_1=float(hit.mean()),std_hit8_1=float(hit.std(ddof=1))))
    return rows,scene_rows

def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=96) as ex:result=list(ex.map(work,scenes()['scenes'],chunksize=1))
    records=pd.DataFrame(sum([r[0] for r in result],[]));by_repeat=records.groupby(['checkpoint','n_rollouts','repeat']).mean(numeric_only=True).reset_index();by_repeat.to_csv(OUT/'metrics/sample_count_global_repeats.csv',index=False);scene=pd.DataFrame(sum([r[1] for r in result],[]));scene.to_parquet(OUT/'metrics/sample_count_scene.parquet',index=False)
    summary=[]
    for (model,n),g in by_repeat.groupby(['checkpoint','n_rollouts']):
        for metric in ['pairwise_ADE','spread_auc','hit8_1']:summary.append(dict(checkpoint=model,n_rollouts=n,metric=metric,mean=g[metric].mean(),subsample_p025=g[metric].quantile(.025),subsample_p975=g[metric].quantile(.975)))
    pd.DataFrame(summary).to_csv(OUT/'metrics/sample_count_summary.csv',index=False)
    comparisons=[]
    for n in [16,32,64]:
        for metric in ['pairwise_ADE','spread_auc','hit8_1']:
            g=by_repeat[by_repeat.n_rollouts==n].pivot(index='repeat',columns='checkpoint',values=metric)
            for model in MODELS[1:]:
                diff=g[model]-g.official_il;comparisons.append(dict(n_rollouts=n,metric=metric,model=model,delta_to_IL_mean=diff.mean(),subsample_p025=diff.quantile(.025),subsample_p975=diff.quantile(.975),fraction_repeats_model_greater_IL=(diff>0).mean()))
    pd.DataFrame(comparisons).to_csv(OUT/'metrics/sample_count_ranking_stability.csv',index=False)
    # Separate uncertainty across the sampled scenes, not conditional subsampling intervals.
    dist=pd.read_csv(V1/'metrics/policy_distribution.csv');ready=pd.read_csv(V1/'metrics/grpo_readiness.csv');d=dist.merge(ready[['token','checkpoint','hit8_1']],on=['token','checkpoint']);ci=[];rng=np.random.default_rng(seed('all','v2_64_scene_bootstrap'))
    for metric in ['pairwise_ade','spread_auc','hit8_1']:
        pivot=d.pivot(index='token',columns='checkpoint',values=metric)
        for model in MODELS[1:]:
            x=(pivot[model]-pivot.official_il).to_numpy();bs=np.concatenate([x[rng.integers(0,len(x),size=(100,len(x)))].mean(1) for _ in range(30)]);ci.append(dict(model=model,metric=metric,n_rollouts=64,n_scenes=len(x),delta_to_IL=float(x.mean()),CI95_low=float(np.quantile(bs,.025)),CI95_high=float(np.quantile(bs,.975))))
    pd.DataFrame(ci).to_csv(OUT/'metrics/primary64_paired_scene_CI.csv',index=False)
    save(OUT/'manifests/sample_count_protocol.json',dict(identity=ident(),user_steering='Official GT-IL policy distribution must use 64 trajectories per scene',primary='All five checkpoints use the immutable V1 1000 x64 rollouts',sensitivity='100 common-index permutations, without-replacement prefixes16/32/64; no new model inference',subsample_intervals='Conditional sensitivity to cached-rollout choice; not population confidence intervals',scene_CI='Separate paired-scene bootstrap,3000 replicates',historical_chain='300 scenes x32 paired rollouts is a separate longitudinal analysis under the originally specified chain protocol',candidate_pool='16 selected trajectories per method remains a selection-rule diagnostic; not an estimate of each neural policy support'))

if __name__=='__main__':main()
