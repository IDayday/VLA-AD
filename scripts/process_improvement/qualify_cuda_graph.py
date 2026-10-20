"""Real full-E0 native graph parity, no score tolerance relaxation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import time
import argparse
import numpy as np
from scripts.process_improvement.common import *


def aggregate(shards):
    records=[];receipts=[]
    for rank in range(shards):
        part=read(ROOT/'tests'/f'cuda_graph_inference_shard_{rank}.json')
        assert part['status']=='PASS' and part['rank']==rank and part['shards']==shards
        receipts.append(part);records.extend(part['records'])
    expected={(label,r['token']) for label in ['IL','RL'] for r in membership('dev')}
    assert len(records)==len(expected) and {(r['baseline'],r['token']) for r in records}==expected
    assert max(r['trajectory_max_error'] for r in records)==0.
    assert max(r['score_max_error'] for r in records)<=1e-6
    hashes={name:sha(SOURCE/name) for name in [
        'navsim/agents/recogdrive/process_improvement/cuda_inference.py',
        'navsim/agents/recogdrive/process_improvement/rotary_cache.py']}
    for rank in range(shards):
        child=read(ROOT/'jobs'/f'qualify_cuda_graph_shard_{rank}'/'child_0.json')
        assert all(child['source_hashes'][name]==value for name,value in hashes.items())
    result=dict(status='PASS',scenes_per_baseline=len(membership('dev')),inference_seeds=5,
                trajectory_exact=True,score_tolerance=1e-6,source_hashes=hashes,
                records=records,shard_receipts=receipts)
    write(ROOT/'tests/cuda_graph_inference.json',result)
    print('FULL_E0_CUDA_GRAPH_PARITY_PASS',len(records),flush=True)


def main():
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head
    from navsim.agents.recogdrive.process_improvement.cuda_inference import FrozenNativeGraph
    from navsim.agents.recogdrive.process_improvement.rng_plan import RNGKey
    from scripts.opd.scoring import Scoring,load_cache
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--aggregate',action='store_true')
    a=p.parse_args()
    if a.aggregate:return aggregate(a.shards)
    configure();cfg=config();records=[];started=time.time()
    for label,path in [('IL',cfg['student_init']),('RL',cfg['teacher_init'])]:
        head=build_head(path,frozen=True);graph=None;scorer=Scoring()
        for index,row in enumerate(membership('dev')[a.rank::a.shards]):
            tokens,inputs=conditioning(observation(row))
            if graph is None:graph=FrozenNativeGraph(head,tokens,inputs)
            graph.prepare(tokens,inputs);actions=[]
            for seed in cfg['evaluation_seeds']:
                key=RNGKey(STUDY,seed,0,0,row['token'],0,-1)
                actions.append(graph.sample(key.plan('final_eval',device='cuda'))[0].cpu().numpy())
            actions=np.stack(actions)
            old=np.load(ROOT/'evaluation/dev'/('B0_'+label)/'scenes'/f"{row['token']}.npz")
            error=float(np.abs(actions-old['trajectories']).max())
            assert np.array_equal(actions,old['trajectories']),(row['token'],error)
            scores=scorer.batch(load_cache(row['metric_cache_path']),actions)['scores']
            assert np.max(np.abs(scores-old['scores']))<=1e-6
            records.append(dict(baseline=label,token=row['token'],trajectory_max_error=error,score_max_error=float(np.abs(scores-old['scores']).max())))
            if (index+1)%128==0:print('CUDA_GRAPH_PARITY',label,index+1,time.time()-started,flush=True)
        del graph,head
    target=ROOT/'tests/cuda_graph_inference.json' if a.shards==1 else ROOT/'tests'/f'cuda_graph_inference_shard_{a.rank}.json'
    write(target,dict(status='PASS',records=records,rank=a.rank,shards=a.shards,scenes_per_baseline=len(membership('dev')[a.rank::a.shards]),inference_seeds=5,
          trajectory_exact=True,score_tolerance=1e-6,seconds=time.time()-started,owner=identity()))
    print('CUDA_GRAPH_NATIVE_PASS',len(records),time.time()-started,flush=True)


if __name__=='__main__':main()
