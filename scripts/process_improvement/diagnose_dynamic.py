"""Small diagnostic teaching chain; never a deployed student performance claim."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import time
from scripts.process_improvement.common import *


def main():
    import torch
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head
    from navsim.agents.recogdrive.process_improvement.rng_plan import RNGKey
    from navsim.agents.recogdrive.process_improvement.suffix_rollout import NativeAdapter
    from navsim.agents.recogdrive.process_improvement.projection_guard import TargetBuilder
    from navsim.agents.recogdrive.process_improvement.budget_counter import CanonicalScorer
    from scripts.opd.scoring import load_cache
    p=argparse.ArgumentParser();p.add_argument('--split',default='diagnostic_train');p.add_argument('--limit',type=int,default=16)
    a=p.parse_args();configure();cfg=config();head=build_head(cfg['teacher_init'],frozen=True);adapter=NativeAdapter(head)
    scorer=CanonicalScorer();builder=TargetBuilder(adapter,scorer,cfg);started=time.time();records=[]
    for row in membership(a.split)[:a.limit]:
        tokens,inputs=conditioning(observation(row));cache=load_cache(row['metric_cache_path'])
        key=RNGKey(STUDY,42,0,0,row['token'],0,-1);noise=key.plan('rollin',device='cuda')
        baseline=adapter.sample_rollin(tokens,inputs,noise)
        with torch.no_grad():context=adapter.prepare_conditioning(tokens,inputs)
        state=noise.initial;states=[state.clone()];targets=[]
        for index in range(5):
            target=builder.build(state,context,index,key.at_step(index),cache,trace=True)
            with torch.no_grad():stats=adapter.transition_stats(state,index,context)
            state=adapter.sample_next(stats,noise.steps[:,index],mean=target.target)
            states.append(state.clone());targets.append(target)
        trajectory=adapter.finalize(state)
        values=scorer.score(row['token'],cache,torch.cat([baseline.trajectory,trajectory]).cpu().numpy())
        record=dict(token=row['token'],driver_record=row['driver_record'],reference=values[0].tolist(),
                    dynamic_teaching=values[1].tolist(),gain=float(values[1,-1]-values[0,-1]),
                    accepted=[v.accepted for v in targets],budget=scorer.counter.state_dict())
        records.append(record)
        save(ROOT/'diagnostics/dynamic_teaching/traces'/f"{row['token']}.pt",dict(row=row,states=torch.stack(states,1).cpu(),
             targets=targets,trajectory=trajectory.cpu(),baseline_trajectory=baseline.trajectory.cpu(),record=record))
        print('DYNAMIC_TEACHING',record,flush=True)
    write(ROOT/'diagnostics/dynamic_teaching/results.json',dict(status='COMPLETE',records=records,
          seconds=time.time()-started,budget=scorer.counter.state_dict(),owner=identity(),
          scope='Targets recomputed at every newly visited state; searched diagnostic teacher, not a native deployed student'))


if __name__=='__main__':main()
