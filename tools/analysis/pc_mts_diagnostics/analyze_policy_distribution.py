"""Scene-level geometric distribution summaries, separate from PDM quality."""
import argparse
import pandas as pd
from common import *

def geometry(trajs):
    xy=np.asarray(trajs,dtype=np.float64)[...,:2];n=len(xy)
    d=distance(xy,xy);medoid=int(d.sum(1).argmin());rad=d[medoid]
    spread=np.sqrt(xy.var(axis=0,ddof=1).sum(-1))
    end=np.linalg.norm(xy[:,-1]-xy[medoid,-1],axis=1)
    cov=np.cov(xy.reshape(n,-1),rowvar=False);e=np.linalg.eigvalsh(cov).clip(0)
    p=e[e>1e-16]/e.sum() if e.sum()>1e-16 else np.array([])
    result=dict(pairwise_ade=float(d[np.triu_indices(n,1)].mean()),pairwise_fde=float(np.linalg.norm(xy[:,-1,None]-xy[None,:,-1],axis=-1)[np.triu_indices(n,1)].mean()),spread_auc=float(spread.mean()),r50=float(np.quantile(rad,.5)),r90=float(np.quantile(rad,.9)),endpoint_cov_trace=float(xy[:,-1].var(axis=0,ddof=1).sum()),endpoint_spread=float(spread[-1]),endpoint_r90=float(np.quantile(end,.9)),effective_rank=float(np.exp(-(p*np.log(p)).sum())) if len(p) else 0.)
    result.update({f'spread_t{i+1}':float(v) for i,v in enumerate(spread)})
    return result,trajs[medoid]

def main(args):
    cfg=config(args.config);sc=manifest(args.manifest);out=ROOT/cfg['output_dir'];rows=[]
    for r in sc['scenes']:
        token=r['token'];il=np.load(out/'rollouts/official_il'/f'{token}.npz')['trajectories'];ilg,ilm=geometry(il)
        for name in MODELS:
            trajs=np.load(out/'rollouts'/name/f'{token}.npz')['trajectories'];g,medoid=geometry(trajs)
            scores=np.load(out/'evaluator/rollouts'/name/f'{token}.npz')['trajectories'];ref=float(np.load(out/'evaluator/rollouts/official_il'/f'{token}.npz')['reference'][0,6])
            feas=feasible(scores);hq=feas&(scores[:,6]>=ref);positive=feas&(scores[:,6]>ref)
            rows.append(dict(token=token,log=r['log'],command=r['command'],checkpoint=name,**g,center_shift_from_il=float(distance(medoid[None],ilm[None])[0,0]),spread_ratio_il=g['spread_auc']/max(ilg['spread_auc'],1e-12),D_all=g['pairwise_ade'],D_feasible=pair_mean(trajs[feas]),D_high_quality=pair_mean(trajs[hq]),D_positive=pair_mean(trajs[positive]),feasible_count=int(feas.sum()),positive_count=int(positive.sum())))
    dest=out/'metrics';dest.mkdir(exist_ok=True);frame=pd.DataFrame(rows);frame.to_csv(dest/'policy_distribution.csv',index=False);frame.to_parquet(dest/'policy_distribution.parquet',index=False)
    print('Distribution scene rows:',len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--manifest');main(p.parse_args())
