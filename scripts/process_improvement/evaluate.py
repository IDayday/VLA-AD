"""Native single-trajectory deployment evaluation, paired semantic noise per scene."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import time
from contextlib import nullcontext
import numpy as np
from scripts.process_improvement.common import *


def main():
    import torch
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head
    from navsim.agents.recogdrive.process_improvement.rng_plan import RNGKey
    from navsim.agents.recogdrive.process_improvement.suffix_rollout import NativeAdapter
    from navsim.agents.recogdrive.process_improvement.budget_counter import CanonicalScorer
    from scripts.opd.scoring import load_cache
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--checkpoint')
    p.add_argument('--baseline',choices=['IL','RL','COPY']);p.add_argument('--split',default='dev')
    p.add_argument('--rank',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--limit',type=int);p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--screening',action='store_true');p.add_argument('--native-cache',action='store_true')
    p.add_argument('--cuda-graph',action='store_true',help='Qualified exact native B1 CUDA scheduling only')
    a=p.parse_args();configure();cfg=config()
    if a.native_cache and read(ROOT/'tests/eval_native_cache.json')['status']!='PASS':
        raise ValueError('Native inference memoization has not passed actual E0 output parity')
    if a.cuda_graph:
        gate=read(ROOT/'tests/cuda_graph_inference.json')
        if gate['status']!='PASS' or gate['scenes_per_baseline']!=1696 or not gate['trajectory_exact']:
            raise ValueError('Full E0 single-scene CUDA graph parity has not passed')
        if any(sha(SOURCE/name)!=value for name,value in gate['source_hashes'].items()):
            raise ValueError('Qualified inference implementation changed')
        if a.native_cache:raise ValueError('Choose one explicitly qualified scheduling implementation')
    if bool(a.checkpoint)==bool(a.baseline):raise ValueError('Specify exactly one checkpoint or baseline')
    if a.split=='navtest' and not (ROOT/'selection/final.lock.json').exists():
        raise ValueError('Navtest requires a precommitted final configuration lock')
    path=a.checkpoint or cfg['student_init' if a.baseline=='IL' else 'teacher_init']
    head=build_head(path,frozen=True);adapter=NativeAdapter(head)
    rows=membership(a.split)
    if a.limit:rows=rows[:a.limit]
    rows=rows[a.rank::a.shards]
    seeds=a.seeds or cfg['screening_seeds' if a.screening else 'evaluation_seeds']
    dest=ROOT/'evaluation'/a.split/a.name
    spec=dict(checkpoint_path=path,checkpoint_sha256=sha(path),population_sha256=sha(ROOT/'manifests'/f'{a.split}.json'),
              evaluation_seeds=seeds,screening=a.screening,contract=CONTRACT,sampling='native stochastic DDIM5 single trajectory',
              precision='float32',total_limit=a.limit)
    if a.split=='navtest':
        from navsim.agents.recogdrive.process_improvement.final_evaluation import validate_final_evaluation
        final_path=SOURCE/'configs/process_improvement/final_eval_lock.json'
        relative=str(final_path.relative_to(SOURCE))
        committed=subprocess.check_output(['git','show','HEAD:'+relative],cwd=SOURCE)
        if committed!=final_path.read_bytes() or read(final_path)!=read(ROOT/'selection/final.lock.json'):
            raise ValueError('Final configuration must be identical to the committed branch lock')
        validate_final_evaluation(read(final_path), name=a.name,
            checkpoint_sha256=spec['checkpoint_sha256'], seeds=seeds, screening=a.screening,
            limit=a.limit, population_sha256=spec['population_sha256'],
            protocol_sha256=sha(ROOT/'protocol.yaml'))
    if (dest/f'config_{a.rank}.json').exists():
        assert read(dest/f'config_{a.rank}.json')['spec']==spec
    write(dest/f'config_{a.rank}.json',dict(spec=spec,owner=identity(),rank=a.rank,shards=a.shards))
    scorer=CanonicalScorer();started=time.time();graph=None
    for index,row in enumerate(rows):
        output=dest/'scenes'/f"{row['token']}.npz"
        meta=output.with_suffix('.json')
        if output.exists() and meta.exists():
            previous=read(meta)
            assert previous['evaluation_identity']==digest(spec)
            assert previous['prediction_sha256']==sha(output),'Existing prediction payload changed'
            continue
        tokens,inputs=conditioning(observation(row));predictions=[]
        from navsim.agents.recogdrive.opd.inference_cache import native_inference_cache
        if a.cuda_graph:
            from navsim.agents.recogdrive.process_improvement.cuda_inference import FrozenNativeGraph
            if graph is None:graph=FrozenNativeGraph(head,tokens,inputs)
            graph.prepare(tokens,inputs)
        with native_inference_cache(head) if a.native_cache else nullcontext():
            for seed in seeds:
                key=RNGKey(STUDY,seed,0,0,row['token'],0,-1)
                plan=key.plan('final_eval',device='cuda')
                trajectory=graph.sample(plan)[0] if graph is not None else adapter.sample_rollin(tokens,inputs,plan).trajectory[0]
                predictions.append(trajectory.cpu().numpy())
        predictions=np.stack(predictions)
        scores=scorer.score(row['token'],load_cache(row['metric_cache_path']),predictions)
        output.parent.mkdir(parents=True,exist_ok=True)
        temp=output.with_name(output.name+f'.{os.getpid()}.tmp.npz')
        np.savez_compressed(temp,trajectories=predictions,scores=scores,seeds=seeds)
        temp.replace(output)
        write(meta,dict(token=row['token'],log_name=row['log_name'],driver_record=row['driver_record'],
                        evaluation_identity=digest(spec),prediction_sha256=sha(output)))
        if (index+1)%64==0:print('EVAL',a.name,a.rank,index+1,len(rows),time.time()-started,flush=True)
    write(dest/f'complete_{a.rank}.json',dict(status='COMPLETE',scenes=len(rows),seconds=time.time()-started,
          budget=scorer.counter.state_dict(),owner=identity(),cuda_graph=a.cuda_graph,native_cache=a.native_cache))


if __name__=='__main__':main()
