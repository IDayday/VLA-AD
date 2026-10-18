"""Matched initialization, training/calibration descriptions, image-dependence check."""
from __future__ import annotations
import numpy as np
from .common import *
from .probe_runner import merge_eval
from .probe_metrics import metrics
from .statistics import log_weights,interval


def main():
    cfg,identity=protocol();initial=[];descriptive={}
    for seed in cfg['seeds']['training']:
        runs={a:read(ROOT/'probes'/f'formal_{a}_{seed}'/'complete.json') for a in cfg['probes']['arms']}
        hashes={a:r['initial_head_sha256'] for a,r in runs.items()}
        assert len(set(hashes.values()))==1
        loss={a:read(ROOT/'probes'/f'formal_{a}_{seed}'/'progress.json')[0] for a in ['FQ','FE','JQ','JE']}
        for key in ['L_abs','L_diff']:assert max(v[key] for v in loss.values())-min(v[key] for v in loss.values())==0
        assert len({runs[a]['config']['fit_library_manifest_sha256'] for a in runs})==1
        initial.append(dict(seed=seed,head_sha256=next(iter(hashes.values())),first_losses=loss,
            VLM_updates={a:dict(max_gradient_norm=runs[a]['max_vlm_gradient_norm'],changed_parameters=len(runs[a]['changed_vlm_parameters'])) for a in ['JQ','JE']}))
    atomic_json(ROOT/'analysis/probe_identity_audit.json',dict(status='PASS',protocol_sha256=identity,seeds=initial,
        minibatch_note='Same4 scenes/candidates/pairs per step. World2 uses2 accumulated microbatches/rank; world4 uses1. Both separately normalize absolute and difference losses by the global sample counts.'))
    for split in ['D_fit','D_cal']:
        descriptive[split]={}
        for arm in cfg['probes']['arms']:
            ps=[]
            for seed in cfg['seeds']['training']:rows,p,t=merge_eval(f'formal_{arm}_{seed}',split);ps.append(p)
            pred=np.stack(ps);truth=np.broadcast_to(t[None],pred.shape)
            descriptive[split][arm]=metrics(pred,truth)
    atomic_json(ROOT/'analysis/fit_cal_diagnostics.json',dict(protocol_sha256=identity,point_estimates=descriptive,
        use='Descriptive diagnostics only; no new hyperparameter or checkpoint selection'))
    normal=[];shuffled=[]
    for seed in cfg['seeds']['training']:
        rows,p,t=merge_eval(f'formal_JE_{seed}','D_audit');other,shuf,t2=merge_eval(f'formal_JE_{seed}','D_audit_shuffled')
        assert np.array_equal(t,t2) and [r['token'] for r in rows]==[r['token'] for r in other]
        normal.append(p);shuffled.append(shuf)
    normal=np.stack(normal);shuffled=np.stack(shuffled);truth=np.broadcast_to(t[None],normal.shape);logs=np.array([r['log'] for r in rows])
    base_rmse=float(np.sqrt(np.mean((normal-truth)**2)));shuffle_rmse=float(np.sqrt(np.mean((shuffled-truth)**2)))
    rng=np.random.default_rng(cfg['seeds']['bootstrap']);draws=[]
    for _ in range(2000):
        si=rng.integers(0,3,3);fi=rng.integers(0,normal.shape[2],normal.shape[2]);zi=rng.integers(0,5,5);w=log_weights(logs,rng)
        target=truth[si][:,:,fi][...,zi]
        errors=[(v[si][:,:,fi][...,zi]-target)**2 for v in [normal,shuffled]]
        rmse=[np.sqrt(np.dot(e.mean((0,2,3,4)),w)/w.sum()) for e in errors]
        draws.append(rmse[1]-rmse[0])
    result=interval(shuffle_rmse-base_rmse,draws);manifest=read(ROOT/'manifests/D_audit_observation_shuffle.json')
    interpretation=('分组置换图像后误差增加，支持对场景图像的输入依赖；不证明某交通因素的因果作用' if result['lower95']>0 else
                    '当前分组置换结果没有稳定显示误差增加；这不能单独证明模型完全忽略图像，也不证明因果因素已被识别')
    atomic_json(ROOT/'analysis/observation_shuffle.json',dict(protocol_sha256=identity,normal_RMSE=base_rmse,shuffled_RMSE=shuffle_rmse,
        RMSE_increase=result,moved_images=manifest['moved'],scenes=manifest['total'],training_seeds=[42,43,44],
        interpretation=interpretation,grouping=manifest['grouping'],labels_used_for_grouping=False))


if __name__=='__main__':main()
