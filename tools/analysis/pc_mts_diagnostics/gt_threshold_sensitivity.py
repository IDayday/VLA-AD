"""Fixed 80/90/95 percentile checks; never optimize the primary threshold."""
import argparse
import pandas as pd
from common import *
from select_candidate_pools import calibrate,pareto_ranks,greedy

def main(args):
    cfg=config(args.config);sc=manifest(args.manifest);out=ROOT/cfg['output_dir'];thresholds=calibrate(out,sc);rows=[]
    for r in sc['scenes']:
        token=r['token'];a=np.load(out/'raw_candidates'/f'{token}.npz')['trajectories'];s=np.load(out/'evaluator/raw_candidates'/f'{token}.npz')['trajectories'];il=np.load(out/'rollouts/official_il'/f'{token}.npz')['trajectories'];q=policy_position(a,il,cfg['knn'])['q_policy'];dg=distance(a,np.asarray(r['gt'])[None])[:,0];ranks=pareto_ranks(s)
        for quantile,threshold in zip([.8,.9,.95],thresholds[r['command']]):
            selected=greedy(np.flatnonzero(dg<=threshold),s,a,cfg['pool_size'],cfg,ranks);idx=np.asarray(selected,dtype=int);n=len(idx)
            rows.append(dict(token=token,command=r['command'],quantile=quantile,threshold_m=threshold,count=n,short_count=cfg['pool_size']-n,candidate_pdms=float(s[idx,6].mean()*100) if n else np.nan,d_GT=float(dg[idx].mean()) if n else np.nan,q_policy=float(q[idx].mean()) if n else np.nan,core_fraction=float((q[idx]<50).mean()) if n else np.nan,boundary_fraction=float(((q[idx]>=50)&(q[idx]<95)).mean()) if n else np.nan,ood_fraction=float((q[idx]>=95).mean()) if n else np.nan,pool_diversity=pair_mean(a[idx])))
    df=pd.DataFrame(rows);(out/'metrics').mkdir(exist_ok=True);df.to_csv(out/'metrics/gt_threshold_sensitivity.csv',index=False);df.groupby('quantile').mean(numeric_only=True).to_csv(out/'metrics/gt_threshold_sensitivity_summary.csv')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--manifest');main(p.parse_args())
