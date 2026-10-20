"""Audit batched Monte Carlo suffixes against native B1 continuations.

This is numerical validation, not method selection. The already locked score
tolerance is used without relaxation; discrete score components must agree.
The known failed full-inference batching case is retained as an engineering
counterexample. No score from this audit constructs a training target.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import time

import numpy as np

from scripts.process_improvement.common import *


def main():
    import torch
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head
    from navsim.agents.recogdrive.process_improvement.rng_plan import RNGKey,candidate_noise
    from navsim.agents.recogdrive.process_improvement.suffix_rollout import NativeAdapter
    from navsim.agents.recogdrive.process_improvement.frozen_suffix import FrozenSuffixAdapter
    from scripts.opd.scoring import Scoring,load_cache
    configure();cfg=config();started=time.time();records=[]
    reference=build_head(cfg['teacher_init'],frozen=True)
    native=NativeAdapter(reference);batched=FrozenSuffixAdapter(reference)
    paths=dict(IL=cfg['student_init'],RL=cfg['teacher_init'],OLD_OPD=HISTORY/'runs/train_E3_42/latest.pt')
    rows=membership(diagnostic_plan()['train_split'])[:2]
    known=next(r for r in membership('dev') if r['token']=='a02f3e19c1ad5991')
    rows=rows+[known]
    scorer=Scoring();indices=[0,1,15,16,31,63,95,127]
    for source,path in paths.items():
        head=build_head(path,frozen=True);rollin_adapter=NativeAdapter(head)
        for row in rows:
            tokens,inputs=conditioning(observation(row))
            key=RNGKey(STUDY,42,0,0,row['token'],0,-1)
            rollin=rollin_adapter.sample_rollin(tokens,inputs,key.plan('rollin',device='cuda'))
            with torch.no_grad():context=native.prepare_conditioning(tokens,inputs)
            cache=load_cache(row['metric_cache_path'])
            for index in range(5):
                sk=key.at_step(index)
                with torch.no_grad():stats=native.transition_stats(rollin.states[:,index],index,context)
                candidates=native.sample_next(stats,candidate_noise(sk,16,device='cuda'))
                noise=torch.cat([sk.plan('fit_suffix',r,device='cuda').steps for r in range(8)])
                starts=candidates[:,None].expand(-1,8,-1,-1).reshape(128,8,3)
                noises=noise[None].expand(16,-1,-1,-1,-1).reshape(128,5,8,3)
                vector,vs=batched.rollout_suffix(context,starts,index+1,noises,True)
                individual=[];states=[]
                for j in indices:
                    trajectory,ss=native.rollout_suffix(context,starts[j:j+1],index+1,noises[j:j+1],True)
                    individual.append(trajectory);states.append(ss)
                scalar=torch.cat(individual);single_states=torch.cat(states)
                cpu_started=time.process_time()
                a=scorer.batch(cache,scalar.cpu().numpy())['scores']
                b=scorer.batch(cache,vector[indices].cpu().numpy())['scores']
                record=dict(source=source,token=row['token'],index=index,batch=128,checked_rows=indices,
                    role='known_full_inference_batching_failure' if row['token']==known['token'] else 'fixed_D_fit_parent_hash',
                    latent_max_error=float((single_states-vs[indices]).abs().max()) if single_states.numel() else 0.,
                    trajectory_max_error=float((scalar-vector[indices]).abs().max()),
                    score_max_error=float(np.abs(a-b).max()),
                    PDMS_max_error=float(np.abs(a[:,-1]-b[:,-1]).max()),
                    cpu_scoring_seconds=time.process_time()-cpu_started,
                    discrete_equal=bool(np.array_equal(a[:,[0,1,3,4,5]],b[:,[0,1,3,4,5]])))
                records.append(record);print('SUFFIX_B1_BATCH_AUDIT',record,flush=True)
        del head,rollin_adapter;torch.cuda.empty_cache()
    valid=all(r['PDMS_max_error']<=1e-6 and r['discrete_equal'] for r in records)
    result=dict(status='PASS' if valid else 'FAIL',records=records,
        score_tolerance=1e-6,PDMS_max_error=max(r['PDMS_max_error'] for r in records),
        trajectory_max_error=max(r['trajectory_max_error'] for r in records),
        scalar_continuations=len(records)*len(indices),seconds=time.time()-started,
        physical_scores=2*len(records)*len(indices),logical_scores=2*len(records)*len(indices),
        reference_simulations=2*len(records),cpu_scoring_seconds=sum(r['cpu_scoring_seconds'] for r in records),
        interpretation='Same B1 conditioning, state and raw noise; only suffix branch execution batch differs. This gate does not admit full-inference batching.',
        checkpoint_sha256={k:sha(v) for k,v in paths.items()},owner=identity())
    write(ROOT/'tests/suffix_scalar_batch.json',result)
    if not valid:raise AssertionError('Batched diagnostic continuation differs beyond locked native scoring tolerance')
    print('SUFFIX_SCALAR_BATCH_GATE_PASS',result['PDMS_max_error'],flush=True)


if __name__=='__main__':main()
