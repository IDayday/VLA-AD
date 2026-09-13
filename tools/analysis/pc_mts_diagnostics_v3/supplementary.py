"""Report log dependence and actual target source weights, with all primary results retained."""
from common_v3 import *
from analyze_c import cluster_bootstrap
def main():
    mapping={r['token']:r['log'] for r in scenes()};comparisons=[];snapshot_comparisons=[]
    for tag,final in [('E',200),('F',100)]:
        all_steps=pd.read_csv(OUT/'metrics'/f'{tag}_seed_averaged_scene.csv');d=all_steps[all_steps.step==final]
        # Report all frozen snapshots, including early=10 in F. No new cutoff
        # or best-checkpoint choice is introduced by this descriptive table.
        for step,frame in all_steps.groupby('step'):
            for metric in ['PDMS','feasible_rate','hard_failure','EP','TTC','zero_score_rate']:
                t=frame.pivot(index='token',columns='method',values=metric)
                for method in CFG['training']['methods']:
                    if method=='conditional_pc':continue
                    snapshot_comparisons.append(dict(stage=tag,step=step,method=method,metric=metric,
                        **bootstrap(t.conditional_pc-t[method],f'S_snapshot_{tag}_{step}_{method}_{metric}')))
        for metric in ['PDMS','feasible_rate','center_shift','CRN_policy_change_ADE','IL_retention_loss','Hit8']:
            t=d.pivot(index='token',columns='method',values=metric)
            for method in CFG['training']['methods']:
                if method=='conditional_pc':continue
                delta=t.conditional_pc-t[method];r=cluster_bootstrap(delta.to_numpy(),[mapping[k] for k in t.index],f'S_log_{tag}_{method}_{metric}');comparisons.append(dict(stage=tag,method=method,metric=metric,mean=delta.mean(),log_clusters=r['cluster_scenes'],log_cluster_ci_low=r['cluster_ci_low'],log_cluster_ci_high=r['cluster_ci_high']))
    csv(pd.DataFrame(comparisons),'S_log_cluster_final_comparisons.csv')
    csv(pd.DataFrame(snapshot_comparisons),'S_all_fixed_snapshot_paired.csv')
    raw=pd.read_parquet(OUT/'metrics/A_raw_candidates.parquet');source={(r.token,int(r.raw_index)):r.exact_source for r in raw.itertuples(index=False)};weighted=[]
    for method in CFG['training']['methods']:
        for sd in CFG['training']['seeds']:
            ledger=read(OUT/'cache/training_ledgers'/f'sft_{method}_{sd}.json')['records'];totals={}
            for row in ledger:
                for i,w in zip(row['parents'],row['weights']):
                    s=source[(row['token'],i)];totals[s]=totals.get(s,0)+w
            for s,total in totals.items():weighted.append(dict(method=method,seed=sd,source=s,total_candidate_weight=total,candidate_weight_fraction=total/len(ledger),GT_retention_coefficient=.25))
    csv(pd.DataFrame(weighted),'S_actual_training_source_weights.csv')
    support=[]
    for run in [f'{l}_seed{s}' for s in CFG['progressive']['seeds'] for l in ['sft_final','grpo_middle','grpo_final']]:
        for token,g in raw.groupby('token',sort=False):
            oldcore=g.q_holdout.to_numpy()<50;a=np.load(OUT/'cache/progressive'/run/f'{token}.npz');q=a['q_holdout'];R=a['R'];oldR=np.load(V2/'il_banks'/f'{token}.npz')['R'];spread=np.sqrt(R[...,:2].var(0,ddof=1).sum(-1)).mean();oldspread=np.sqrt(oldR[...,:2].var(0,ddof=1).sum(-1)).mean()
            support.append(dict(run=run,token=token,split='holdout' if token in set(tokens('holdout')) else 'train',old_core_count=int(oldcore.sum()),old_core_retained_fraction=float((q[oldcore]<95).mean()) if oldcore.any() else np.nan,current_R_spread=float(spread),R_spread_ratio=float(spread/max(oldspread,1e-12)),R_center_shift=float(distance(R.mean(0)[None],oldR.mean(0)[None])[0,0])))
    csv(pd.DataFrame(support),'S_progressive_support_retention.csv')
    stage_audit('supplementary',new_thresholds_or_scene_filters=False,definitions='Log-cluster uncertainty; observed training source weights; old Core<50 retention under new q<95, using the already frozen region definitions. All primary Full-1000 and holdout-300 results retained.')
if __name__=='__main__':main()
