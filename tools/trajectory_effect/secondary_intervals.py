"""Log/family/train/inference bootstrap intervals for detection metrics."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
from .common import *
from .probe_runner import merge_eval
from .probe_metrics import average_precision,metrics
from .statistics import log_weights,interval


def one(arm):
    cfg,_=protocol();values=[]
    for seed in cfg['seeds']['training']:
        rows,p,y=merge_eval(f'formal_{arm}_{seed}','D_audit');values.append(p)
    pred=np.stack(values);target=np.broadcast_to(y[None],pred.shape);logs=np.array([r['log'] for r in rows])
    rng=np.random.default_rng(cfg['seeds']['bootstrap']);draws={k:[] for k in ['degradation_PR_AUC_AP','top10pct_scene_recall','severe_RMSE','severe_miss_rate']}
    for _ in range(cfg['statistics']['bootstrap_replicates']):
        si=rng.integers(0,3,3);fi=rng.integers(0,pred.shape[2],pred.shape[2]);zi=rng.integers(0,5,5);w=log_weights(logs,rng)
        p=pred[si][:,:,fi][...,zi];t=target[si][:,:,fi][...,zi];pm=p.mean(-1);tm=t.mean(-1)
        drop=tm<-.02;weights=np.broadcast_to(w[None,:,None,None],drop.shape)
        ap=average_precision(drop,-pm,weights);draws['degradation_PR_AUC_AP'].append(ap if ap is not None else np.nan)
        # Exact top10% on the resampled population, retaining log multiplicities.
        order=np.argsort(pm.transpose(0,2,3,1),axis=-1,kind='stable')
        ws=w[order];labels=np.take_along_axis(drop.transpose(0,2,3,1),order,axis=-1)
        before=np.cumsum(ws,axis=-1)-ws;quota=np.ceil(.1*w.sum())
        selected=np.minimum(ws,np.maximum(quota-before,0));positive=(labels*ws).sum(-1)
        recall=np.divide((labels*selected).sum(-1),positive,out=np.full(positive.shape,np.nan),where=positive>0)
        draws['top10pct_scene_recall'].append(float(np.nanmean(recall)))
        severe=tm<-.1;sw=weights*severe
        if sw.sum():
            draws['severe_RMSE'].append(float(np.sqrt((((p-t)**2).mean(-1)*sw).sum()/sw.sum())))
            draws['severe_miss_rate'].append(float(((pm>=-.02)*sw).sum()/sw.sum()))
        else:draws['severe_RMSE'].append(np.nan);draws['severe_miss_rate'].append(np.nan)
    point=metrics(pred,target)
    return arm,{k:dict(estimate=point[k],lower95=float(np.nanquantile(v,.025)),upper95=float(np.nanquantile(v,.975)),
        valid_replicates=int(np.isfinite(v).sum())) for k,v in draws.items()}


def main():
    cfg,identity=protocol()
    with ProcessPoolExecutor(5,mp_context=mp.get_context('spawn')) as pool:result=dict(pool.map(one,cfg['probes']['arms']))
    atomic_json(ROOT/'analysis/probe_secondary_intervals.json',dict(protocol_sha256=identity,arms=result,
        unit='Paired whole logs, source-update families, training seeds and inference seeds',replicates=2000))


if __name__=='__main__':main()
