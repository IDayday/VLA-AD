"""Paired log/family/seed clustered summaries; no trajectory-pair pseudoreplication."""
from __future__ import annotations
import argparse
import numpy as np
from .common import *
from .bank import baseline_score_path
from .updates import families,targets,score_path,directory


def estimate(delta,weights=None):
    # delta: [...,scene]; first average inference draws before invoking this.
    weights=np.ones(delta.shape[-1]) if weights is None else weights
    return np.stack([np.sum(delta*weights,axis=-1)/weights.sum(),
        np.sum(np.maximum(delta,0)*weights,axis=-1)/weights.sum(),
        np.sum(np.maximum(-delta,0)*weights,axis=-1)/weights.sum(),
        np.sum((delta<-.02)*weights,axis=-1)/weights.sum()],axis=-1)


def interval(point,draws):
    low,high=np.quantile(draws,[.025,.975],axis=0)
    return dict(estimate=np.asarray(point).tolist(),lower95=np.asarray(low).tolist(),upper95=np.asarray(high).tolist())


def log_weights(logs,rng):
    unique,inverse=np.unique(logs,return_inverse=True)
    counts=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))
    return counts[inverse]


def paired_policy_summary(delta,logs,replicates=2000,seed=20261019):
    # [family_or_training_seed,checkpoint,scene,inference_seed]
    delta=np.asarray(delta);rng=np.random.default_rng(seed);point=estimate(delta.mean(-1)).mean(0);draws=[]
    for _ in range(replicates):
        fi=rng.integers(0,len(delta),len(delta));zi=rng.integers(0,delta.shape[-1],delta.shape[-1])
        d=delta[fi][...,zi].mean(-1);w=log_weights(logs,rng)
        draws.append(estimate(d,w).mean(0))
    return interval(point,np.array(draws))


def collect_h1(args):
    cfg,identity=protocol();result=[];all_audit=[];audits=[];rows_out=[]
    for f in families():
        if not (directory(f['id'])/'train_complete.json').exists() or not (directory(f['id'])/'score_complete.json').exists():
            raise RuntimeError(f'Incomplete real update: {f["id"]}')
        rows=targets(f);logs=np.array([r['log_name'] for r in rows]);old=np.stack([np.load(baseline_score_path(r['token']))['scores'][:5] for r in rows])
        new=np.stack([np.stack([np.load(score_path(f['id'],step,r['token']))['scores'] for r in rows]) for step in (1,4)])
        delta=new[...,-1]-old[None,...,-1]
        summary=paired_policy_summary(delta[None],logs,cfg['statistics']['bootstrap_replicates'])
        for j,step in enumerate((1,4)):
            d=delta[j].mean(-1);train=read(directory(f['id'])/'train_complete.json')
            result.append(dict(family=f['id'],split=f['split'],source_kind=f['source_kind'],step=step,
                metrics={k:dict(estimate=summary['estimate'][j][q],lower95=summary['lower95'][j][q],upper95=summary['upper95'][j][q])
                    for q,k in enumerate(['G','M_plus','M_minus','material_drop_fraction'])},
                material_improvements=int((d>.02).sum()),material_drops=int((d<-.02).sum()),
                drop_logs=len(set(logs[d<-.02])),severe_drops=int((d<-.1).sum()),
                single_output_drop_probability=float((delta[j]<-.02).mean()),
                positive_to_zero=int(((old[...,-1]>1e-12)&(new[j,...,-1]<=1e-12)).sum()),
                high_to_nearzero=int(((old[...,-1]>=.9)&(new[j,...,-1]<=.01)).sum()),
                new_component_violations={FIELDS[k]:int(((old[...,k]>=1-1e-6)&(new[j,...,k]<1-1e-6)).sum()) for k in [0,1,3,4,5]},
                parameter_change=train['changed']))
            rows_out.extend(dict(token=r['token'],log=r['log_name'],family=f['id'],step=step,
                old_mean=float(old[i,:,-1].mean()),new_mean=float(new[j,i,:,-1].mean()),delta_J=float(d[i]),
                paired_deltas=delta[j,i].tolist()) for i,r in enumerate(rows))
        if f['split']=='audit':all_audit.append(delta);audits.append(f)
    rows=targets(audits[0]);logs=np.array([r['log_name'] for r in rows]);combined=paired_policy_summary(np.stack(all_audit),logs,cfg['statistics']['bootstrap_replicates'])
    full=np.stack([np.load(baseline_score_path(r['token']))['scores'] for r in rows])
    noise=full[:,5:10,-1]-full[:,:5,-1]
    noise_summary=paired_policy_summary(noise[None,None],logs,cfg['statistics']['bootstrap_replicates'])
    heldout=[r for r in result if r['split']=='audit' and r['step']==4]
    signal_families=[r for r in heldout if r['parameter_change'] and r['material_drops']>=20 and r['drop_logs']>=4]
    # This is a stage-admission gate, not a final causal/statistical H1 claim.
    gate=len(signal_families)>=2
    atomic_json(ROOT/'analysis/h1_scene_changes.json',rows_out)
    atomic_json(ROOT/'analysis/h1.json',dict(protocol_sha256=identity,status='ANALYZED',families=result,
        columns=['G','M_plus','M_minus','material_drop_fraction'],audit_combined=combined,
        independent_noise_self_comparison=noise_summary,gate_B='PASS' if gate else 'INSUFFICIENT_EVENTS',
        gate_families=[r['family'] for r in signal_families],audit_family_count=len(audits),
        caution='Five inference draws do not certify individual scene degradation. Family endpoints are correlated.'))
    print(json.dumps(dict(gate_B=gate,audit_combined=combined,noise=noise_summary,heldout=heldout),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['h1']);args=p.parse_args();collect_h1(args)


if __name__=='__main__':main()
