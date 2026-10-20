"""Explicitly qualified batching of independent native single-trajectory queries.

This does not batch candidate alternatives or select trajectories. Each scene
keeps its own initial/transition noises. Formal use requires full E0 score and
discrete-component parity against the completed scalar inference evaluation.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import time
import numpy as np
from scripts.process_improvement.common import *


def main():
    import torch
    from transformers.feature_extraction_utils import BatchFeature
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head
    from navsim.agents.recogdrive.opd.inference_cache import native_inference_cache
    from navsim.agents.recogdrive.process_improvement.contracts import NoisePlan
    from navsim.agents.recogdrive.process_improvement.rng_plan import RNGKey
    from navsim.agents.recogdrive.process_improvement.suffix_rollout import NativeAdapter
    from navsim.agents.recogdrive.process_improvement.budget_counter import CanonicalScorer
    from scripts.opd.scoring import load_cache
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--checkpoint')
    p.add_argument('--baseline',choices=['IL','RL']);p.add_argument('--split',default='dev')
    p.add_argument('--rank',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--limit',type=int)
    p.add_argument('--seeds',nargs='+',type=int);p.add_argument('--screening',action='store_true')
    p.add_argument('--qualification',action='store_true');a=p.parse_args();configure();cfg=config()
    if not a.qualification and read(ROOT/'tests/batched_evaluation.json')['status']!='PASS':
        raise ValueError('Independent-observation inference batching is not qualified')
    if a.qualification and (not a.name.startswith('qualification_') or a.split!='dev'):
        raise ValueError('Qualification must compare real dev E0 outputs in a separate namespace')
    if a.split=='navtest' and not (ROOT/'selection/final.lock.json').exists():raise ValueError('Final configuration not locked')
    if bool(a.checkpoint)==bool(a.baseline):raise ValueError('Specify checkpoint xor baseline')
    path=a.checkpoint or cfg['student_init' if a.baseline=='IL' else 'teacher_init']
    head=build_head(path,frozen=True);adapter=NativeAdapter(head);rows=membership(a.split)
    if a.limit:rows=rows[:a.limit]
    rows=rows[a.rank::a.shards];seeds=a.seeds or cfg['screening_seeds' if a.screening else 'evaluation_seeds']
    dest=ROOT/'evaluation'/a.split/a.name
    spec=dict(checkpoint_path=path,checkpoint_sha256=sha(path),population_sha256=sha(ROOT/'manifests'/f'{a.split}.json'),
              evaluation_seeds=seeds,screening=a.screening,contract=CONTRACT,sampling='native stochastic DDIM5 single trajectory',
              precision='float32',total_limit=a.limit)
    if (dest/f'config_{a.rank}.json').exists():assert read(dest/f'config_{a.rank}.json')['spec']==spec
    write(dest/f'config_{a.rank}.json',dict(spec=spec,owner=identity(),rank=a.rank,shards=a.shards,batch_size=a.batch_size))
    remaining=[r for r in rows if not (dest/'scenes'/f"{r['token']}.json").exists()]
    scorer=CanonicalScorer();started=time.time();errors=[]
    with ThreadPoolExecutor(4) as loader:
        for begin in range(0,len(remaining),a.batch_size):
            batch=remaining[begin:begin+a.batch_size]
            moved=[conditioning(v) for v in loader.map(observation,batch)]
            tokens=torch.cat([v[0] for v in moved]);inputs=BatchFeature({k:torch.cat([v[1][k] for v in moved]) for k in moved[0][1]})
            actions=[]
            with native_inference_cache(head):
                for seed in seeds:
                    plans=[RNGKey(STUDY,seed,0,0,r['token'],0,-1).plan('final_eval',device='cuda') for r in batch]
                    plan=NoisePlan(torch.cat([v.initial for v in plans]),torch.cat([v.steps for v in plans]))
                    actions.append(adapter.sample_rollin(tokens,inputs,plan).trajectory.cpu().numpy())
            actions=np.stack(actions,1)
            for row,trajectories in zip(batch,actions):
                scores=scorer.score(row['token'],load_cache(row['metric_cache_path']),trajectories)
                if a.qualification:
                    old=np.load(ROOT/'evaluation/dev'/('B0_'+a.baseline)/'scenes'/f"{row['token']}.npz")
                    error=float(np.abs(trajectories-old['trajectories']).max());score_error=float(np.abs(scores-old['scores']).max())
                    discrete=np.array_equal(scores[:,[0,1,3,4,5]],old['scores'][:,[0,1,3,4,5]])
                    # FP32 batch GEMM can differ by rounding; 0.1mm is locked
                    # before measuring. Canonical PDMS tolerance stays 1e-6.
                    assert error<=1e-4,(row['token'],error)
                    assert score_error<=1e-6 and discrete,(row['token'],score_error,discrete)
                    errors.append(dict(token=row['token'],trajectory_max_error=error,score_max_error=score_error,discrete_exact=discrete))
                output=dest/'scenes'/f"{row['token']}.npz";output.parent.mkdir(parents=True,exist_ok=True)
                temp=output.with_name(output.name+f'.{os.getpid()}.tmp.npz')
                np.savez_compressed(temp,trajectories=trajectories,scores=scores,seeds=seeds);temp.replace(output)
                write(output.with_suffix('.json'),dict(token=row['token'],log_name=row['log_name'],driver_record=row['driver_record'],
                      evaluation_identity=digest(spec),prediction_sha256=sha(output)))
            print('BATCH_EVAL',a.name,a.rank,min(begin+a.batch_size,len(remaining)),len(remaining),time.time()-started,flush=True)
    write(dest/f'complete_{a.rank}.json',dict(status='COMPLETE',scenes=len(rows),seconds=time.time()-started,budget=scorer.counter.state_dict(),owner=identity()))
    if a.qualification:
        write(ROOT/'tests'/('batch_inference_'+a.baseline+'.json'),dict(status='PASS',records=errors,batch_size=a.batch_size,
              scene_count=len(rows),inference_seeds=len(seeds),trajectory_tolerance=1e-4,score_tolerance=1e-6,owner=identity()))


if __name__=='__main__':main()
