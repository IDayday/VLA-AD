"""Descriptive gate funnel for frozen PC-MTS eligibility; no threshold search."""
import argparse,concurrent.futures
import pandas as pd
from common import *
from select_candidate_pools import pareto_ranks

def audit(task):
    scene,cfg,out=task;token=scene['token']
    raw=np.load(out/'raw_candidates'/f'{token}.npz')['trajectories'];il=np.load(out/'rollouts/official_il'/f'{token}.npz')['trajectories']
    scores=np.load(out/'evaluator/raw_candidates'/f'{token}.npz')['trajectories'];ref=float(np.load(out/'evaluator/rollouts/official_il'/f'{token}.npz')['reference'][0,6])
    loc=feasible(np.load(out/'evaluator/selection_local/common'/f'{token}.npz')['trajectories']).mean(1)
    q=policy_position(raw,il,cfg['knn'])['q_policy'];rank=pareto_ranks(scores)
    quality=np.isfinite(scores).all(1)&feasible(scores)&(scores[:,6]>=ref-1e-10)&(rank<=cfg['pareto_qualified_max_front'])
    finalq=quality&(q>=40)&(q<99);final=finalq&(loc>=.5)
    reason='accepted_parent_exists' if final.any() else 'no_base_quality_candidate' if not quality.any() else 'none_in_relaxed_policy_interval' if not finalq.any() else 'none_pass_local_feasibility'
    return dict(token=token,command=scene['command'],reason=reason,quality_count=int(quality.sum()),raw_primary_boundary_count=int(((q>=50)&(q<95)).sum()),quality_primary_boundary_count=int((quality&(q>=50)&(q<95)).sum()),quality_relaxed_interval_count=int(finalq.sum()),final_eligible_count=int(final.sum()),il_pairwise_ade=pair_mean(il),il_reference_pdms=ref*100)

def main(args):
    cfg=config(args.config);sc=manifest(args.manifest);out=ROOT/cfg['output_dir'];pending=pending_pc_tokens(out,cfg,sc)
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(cfg['cpu_workers'],len(sc['scenes']))) as ex:rows=list(ex.map(audit,[(r,cfg,out) for r in sc['scenes']],chunksize=1))
    assert {r['token'] for r in rows if r['final_eligible_count']==0}==pending
    frame=pd.DataFrame(rows);frame.to_csv(out/'metrics/pc_eligibility_funnel.csv',index=False)
    summary=dict(identity=identity(cfg,sc),frozen_thresholds_unchanged=True,reason_counts=frame.reason.value_counts().to_dict(),command_reason_counts=frame.groupby(['command','reason']).size().reset_index(name='count').to_dict('records'),group_means=frame.assign(parent_available=frame.final_eligible_count>0).groupby('parent_available').mean(numeric_only=True).reset_index().to_dict('records'))
    save(out/'manifests/pc_eligibility_funnel.json',summary);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--manifest');main(p.parse_args())
