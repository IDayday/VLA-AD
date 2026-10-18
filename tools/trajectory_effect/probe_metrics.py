"""Held-out real-update prediction metrics and paired clustered comparisons."""
from __future__ import annotations
import argparse
import numpy as np
from .common import *
from .probe_runner import merge_eval,run_dir
from .statistics import log_weights,interval


def average_precision(y,score,weights=None):
    """Non-interpolated precision-recall integral, with score ties grouped."""
    y=np.asarray(y,dtype=np.float64).ravel();score=np.asarray(score).ravel()
    w=np.ones(len(y)) if weights is None else np.asarray(weights,dtype=float).ravel()
    keep=w>0;y=y[keep];score=score[keep];w=w[keep]
    order=np.argsort(-score,kind='stable');y=y[order];score=score[order];w=w[order]
    positive=np.sum(y*w)
    if positive==0:return None
    ends=np.r_[np.flatnonzero(np.diff(score)),len(score)-1]
    tp=np.cumsum(y*w)[ends];all_weight=np.cumsum(w)[ends]
    return float(np.sum(np.diff(np.r_[0.,tp/positive])*(tp/all_weight)))


def metrics(pred,target):
    error=pred-target
    mean_p=pred.mean(-1);mean_t=target.mean(-1);drop=(mean_t<-.02);risk=-mean_p
    truth=np.where(mean_t<-.02,0,np.where(mean_t>.02,2,1))
    guessed=np.where(mean_p<-.02,0,np.where(mean_p>.02,2,1))
    # Review quota applies independently to each seed/family/checkpoint.
    # Input shape [train seed,scene,family,checkpoint,inference seed].
    recalls=[];n=pred.shape[1];k=max(1,int(np.ceil(.1*n)))
    for s in range(pred.shape[0]):
        for f in range(pred.shape[2]):
            for t in range(pred.shape[3]):
                idx=np.argsort(-risk[s,:,f,t],kind='stable')[:k];actual=drop[s,:,f,t]
                if actual.sum():recalls.append(float(actual[idx].sum()/actual.sum()))
    severe=mean_t<-.1
    return dict(MAE=float(abs(error).mean()),RMSE=float(np.sqrt((error**2).mean())),
        degradation_PR_AUC_AP=average_precision(drop,risk),
        degradation_prevalence=float(drop.mean()),top10pct_scene_recall=float(np.mean(recalls)) if recalls else None,
        confusion_labels=['material_drop','near_zero','material_improvement'],
        confusion_matrix=np.bincount(truth.ravel()*3+guessed.ravel(),minlength=9).reshape(3,3).tolist(),
        severe_scene_count_per_seed=int(severe[0].sum()),
        severe_RMSE=float(np.sqrt((error[severe]**2).mean())) if severe.any() else None,
        severe_miss_rate=float((mean_p[severe]>=-.02).mean()) if severe.any() else None)


def compare_bootstrap(preds,target,logs,reps,seed):
    rng=np.random.default_rng(seed);names=list(preds);pairs=[('JQ','FQ'),('JE','FE'),('JE','JQ'),('FE','FQ'),('JE','T'),('JE','ZERO')]
    point={a:np.sqrt(np.mean((p-target)**2)) for a,p in preds.items()}
    boot={a:[] for a in names};maes={a:[] for a in names};differences={a+'-vs-'+b:[] for a,b in pairs}
    ns,n,nf,nt,nz=next(iter(preds.values())).shape
    for _ in range(reps):
        si=rng.integers(0,ns,ns);fi=rng.integers(0,nf,nf);zi=rng.integers(0,nz,nz);w=log_weights(logs,rng)
        sampled_target=target[si][:,:,fi][...,zi]
        values={}
        for arm in names:
            error=preds[arm][si][:,:,fi][...,zi]-sampled_target
            per_scene=(error**2).mean((0,2,3,4));values[arm]=np.sqrt(np.dot(per_scene,w)/w.sum());boot[arm].append(values[arm])
            maes[arm].append(np.dot(abs(error).mean((0,2,3,4)),w)/w.sum())
        for a,b in pairs:differences[a+'-vs-'+b].append((values[b]-values[a])/values[b])
    intervals={a:dict(RMSE=interval(point[a],boot[a]),MAE=interval(abs(preds[a]-target).mean(),maes[a])) for a in names}
    comparisons={a+'-vs-'+b:interval((point[b]-point[a])/point[b],differences[a+'-vs-'+b]) for a,b in pairs}
    return intervals,comparisons


def analyze(args):
    cfg,identity=protocol();preds={};names={}
    for arm in cfg['probes']['arms']:
        runs=[f'{args.prefix}_{arm}_{seed}' for seed in cfg['seeds']['training']];values=[]
        for name in runs:
            rows,pred,y=merge_eval(name,args.split)
            if 'reference_rows' not in locals():reference_rows=rows;target=y
            elif [r['token'] for r in rows]!=[r['token'] for r in reference_rows] or not np.array_equal(y,target):
                raise ValueError('Unpaired evaluations or label mismatch')
            values.append(pred)
        preds[arm]=np.stack(values);names[arm]=runs
    target=np.broadcast_to(target[None],preds['JE'].shape)
    preds['ZERO']=np.zeros_like(target);logs=np.array([r['log'] for r in reference_rows])
    intervals,comparisons=compare_bootstrap(preds,target,logs,cfg['statistics']['bootstrap_replicates'],cfg['seeds']['bootstrap'])
    values={arm:dict(**metrics(pred,target),intervals=intervals[arm]) for arm,pred in preds.items()}
    # Gate uses the independently locked audit only after tuning is finished.
    generalizes=comparisons['JE-vs-ZERO']['lower95']>0 and comparisons['JE-vs-T']['lower95']>0
    no_material_effect=comparisons['JE-vs-JQ']['upper95']<cfg['statistics']['h2_material_relative_rmse_reduction']
    narrow=(comparisons['JE-vs-JQ']['upper95']-comparisons['JE-vs-JQ']['lower95'])<=.2
    gate='STOP_NO_MATERIAL_EFFECT' if no_material_effect and narrow else ('PASS' if generalizes else 'INCONCLUSIVE_GENERALIZATION')
    result=dict(status='ANALYZED',protocol_sha256=identity,split=args.split,prefix=args.prefix,
        arms=values,comparisons= comparisons,names=names,gate_D=gate,
        primary_unit='Unseen target logs x unseen source update families; checkpoints kept within family cluster',
        heldout_families=reference_rows[0]['families'],target_logs=len(set(logs)),training_seeds=cfg['seeds']['training'],
        confirmation_width_trigger=any(v['upper95']-v['lower95']>.2 for k,v in comparisons.items() if k in ('JQ-vs-FQ','JE-vs-FE','JE-vs-JQ')),
        interpretation='Difference supervision uses the same absolute labels; LoRA updates28 language Q/V projections, not the entire VLM.')
    atomic_json(ROOT/'analysis'/f'probes_{args.prefix}_{args.split}.json',result)
    print(json.dumps(dict(gate_D=gate,comparisons=comparisons,RMSE={k:v['RMSE'] for k,v in values.items()}),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',default='formal');p.add_argument('--split',default='D_audit')
    analyze(p.parse_args())


if __name__=='__main__':main()
