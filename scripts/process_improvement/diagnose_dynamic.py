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
    from navsim.agents.recogdrive.process_improvement.compute_ledger import ComputeLedger
    from scripts.opd.scoring import load_cache
    p=argparse.ArgumentParser();p.add_argument('--split');p.add_argument('--limit',type=int,default=16)
    p.add_argument('--name',default='dynamic_parent_teaching');p.add_argument('--config');p.add_argument('--checkpoint')
    a=p.parse_args();configure();plan=diagnostic_plan();cfg=config(a.config or SOURCE/plan['config'])
    cfg.update(candidate_count=16,fit_replicates=8,guard_replicates=8)
    split=a.split or plan['train_split']
    if split not in [plan['train_split'],plan['confirm_split']]:raise ValueError('Only locked diagnostic train/confirm data')
    dest=ROOT/'diagnostics'/a.name
    if (dest/'results.json').exists() or (dest/'configuration.json').exists():
        raise FileExistsError('Preserve existing dynamic teaching diagnostic; explicit new name required after a failure')
    rows=membership(split)[:a.limit]
    head=build_head(cfg['teacher_init'],frozen=True);adapter=NativeAdapter(head)
    student=build_head(a.checkpoint,frozen=True) if a.checkpoint else None
    sa=NativeAdapter(student) if student is not None else None
    compute=ComputeLedger(reference=head,student=student)
    scorer=CanonicalScorer(workers=4,metric_paths={r['token']:r['metric_cache_path'] for r in rows})
    builder=TargetBuilder(adapter,scorer,cfg);started=time.time();records=[]
    write(dest/'configuration.json',dict(split=split,manifest_sha256=sha(ROOT/'manifests'/f'{split}.json'),
          scene_tokens=[r['token'] for r in rows],config=cfg,rollin_seed=42,
          teacher_sha256=sha(cfg['teacher_init']),student_sha256=sha(a.checkpoint) if student is not None else None,
          script_sha256=sha(__file__),owner=identity(),
          scope='Small fixed teaching-program diagnostic, targets recomputed on every newly visited state, all5 positions; no training or deployment search'))
    for row in rows:
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
        trajectories=[baseline.trajectory,trajectory]
        if sa is not None:trajectories.append(sa.sample_rollin(tokens,inputs,noise).trajectory)
        values=scorer.score(row['token'],cache,torch.cat(trajectories).cpu().numpy())
        record=dict(token=row['token'],driver_record=row['driver_record'],reference=values[0].tolist(),
                    dynamic_teaching=values[1].tolist(),gain=float(values[1,-1]-values[0,-1]),
                    accepted=[v.accepted for v in targets],budget=scorer.counter.state_dict(),
                    frozen_student=values[2].tolist() if sa is not None else None)
        records.append(record)
        save(dest/'traces'/f"{row['token']}.pt",dict(row=row,states=torch.stack(states,1).cpu(),
             targets=targets,trajectory=trajectory.cpu(),baseline_trajectory=baseline.trajectory.cpu(),record=record))
        print('DYNAMIC_TEACHING',record,flush=True)
    write(dest/'results.json',dict(status='COMPLETE',records=records,
          seconds=time.time()-started,budget=scorer.counter.state_dict(),compute=compute.state_dict(),owner=identity(),
          scope='Targets recomputed at every newly visited state; searched diagnostic teacher, not a native deployed student'))
    scorer.close();compute.close()


if __name__=='__main__':main()
