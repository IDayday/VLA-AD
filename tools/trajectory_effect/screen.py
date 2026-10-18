"""Independent sampling seeds define the stable-high baseline subset."""
from __future__ import annotations
import argparse
import numpy as np
from .common import *
from .bank import load_observation,score_record
from .updates import wait_file


def trajectory(token):return ROOT/'baseline/high_screen/trajectories'/f'{token}.npz'
def scores(token):return ROOT/'baseline/high_screen/scores'/f'{token}.npz'


def extract(args):
    from .runtime import load_agent,action_input,predict
    cfg,identity=protocol();agent=load_agent(online=False);start=time.perf_counter()
    for row in scene_rows('D_audit')[args.rank::args.shards]:
        token=row['token'];data=load_observation(token);inputs=action_input(data['features'])
        value=np.stack([predict(agent.action_head,data['tokens'].cuda(),inputs,token,s) for s in cfg['seeds']['high_screen']])
        atomic_npz(trajectory(token),dict(token=token,protocol_sha256=identity,seeds=cfg['seeds']['high_screen']),trajectories=value)
    atomic_json(ROOT/'baseline/high_screen'/f'extract_{args.rank}.json',dict(status='COMPLETE',seconds=time.perf_counter()-start,owner=runtime_identity()))


def score(args):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    _,identity=protocol();rows=scene_rows('D_audit')
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        for i in range(0,len(rows),args.workers*4):
            batch=rows[i:i+args.workers*4]
            for r in batch:wait_file(trajectory(r['token']))
            list(pool.map(score_record,[(r,str(trajectory(r['token'])),str(scores(r['token'])),identity) for r in batch]))
    stable=[r['token'] for r in rows if (np.load(scores(r['token']))['scores'][:,-1]>=.9).all()]
    atomic_json(ROOT/'baseline/high_screen/complete.json',dict(status='COMPLETE',protocol_sha256=identity,
        stable_tokens=stable,stable_count=len(stable),scenes=len(rows),definition='All five independent screening scores>=0.90; evaluation1101..1105'))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['extract','score']);p.add_argument('--rank',type=int,default=0)
    p.add_argument('--shards',type=int,default=1);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();globals()[a.action](a)


if __name__=='__main__':main()
