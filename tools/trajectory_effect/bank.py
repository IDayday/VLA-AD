"""Resumable native observation/policy banks, with score-only CPU workers."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time
import traceback
import numpy as np
from .common import *


def observation_path(token): return ROOT/'observations'/f'{token}.pt'
def baseline_path(token): return ROOT/'baseline/trajectories'/f'{token}.npz'
def baseline_score_path(token): return ROOT/'baseline/scores'/f'{token}.npz'


def load_observation(token):
    import torch
    value=torch.load(observation_path(token),map_location='cpu',weights_only=False)
    if value['token']!=token:raise ValueError('Observation/token mismatch')
    value['tokens']=value['tokens'].float()
    return value


def extract(args):
    import torch
    from .runtime import load_agent,features_for,encode,action_input,predict
    cfg,identity=protocol();agent=load_agent();start=time.perf_counter()
    rows=scene_rows(args.split)[args.rank::args.shards]
    record=runtime_identity();record.update(split=args.split,rank=args.rank,shards=args.shards,protocol_sha256=identity)
    atomic_json(ROOT/'runtime'/f'extract_{args.split}_{args.rank}.json',record)
    for i,row in enumerate(rows):
        token=row['token'];dest=baseline_path(token)
        if dest.exists():
            if json.loads(str(np.load(dest)['metadata']))['protocol_sha256']!=identity:
                raise ValueError('Conflicting baseline bank')
            if not observation_path(token).exists():raise ValueError('Missing observation paired with existing trajectory')
            continue
        pilot=ROOT/'pilot/observations'/f'{token}.pt'
        if pilot.exists():
            value=torch.load(pilot,map_location='cpu',weights_only=False)
            features=value['features'];tokens=value['tokens'].cuda()
        else:
            features=features_for(agent,row);tokens=encode(agent,features)
        # Native VLM produces BF16, then casts to FP32 for the head. Storage is
        # lossless relative to those native tokens, verified before every write.
        stored=tokens.detach().cpu().bfloat16()
        if not torch.equal(stored.float(),tokens.detach().cpu()):
            raise ValueError('Native tokens cannot be stored losslessly as BF16')
        atomic_torch(observation_path(token),dict(token=token,features=features,tokens=stored,
            protocol_sha256=identity,input_contract='native current image and ego history, no target'))
        inputs=action_input(features)
        seeds=cfg['seeds']['inference']+(cfg['seeds']['independent_noise'] if args.split=='D_audit' else [])
        trajectories=np.stack([predict(agent.action_head,tokens,inputs,token,s) for s in seeds])
        atomic_npz(dest,dict(token=token,log=row['log_name'],split=args.split,seeds=seeds,
            protocol_sha256=identity,checkpoint_sha256=cfg['primary_start']['sha256']),trajectories=trajectories)
        if (i+1)%16==0:
            atomic_json(ROOT/'runtime'/f'extract_{args.split}_{args.rank}_progress.json',dict(done=i+1,total=len(rows),seconds=time.perf_counter()-start))
            print('EXTRACT',args.split,args.rank,i+1,len(rows),round(time.perf_counter()-start,1),flush=True)
    atomic_json(ROOT/'runtime'/f'extract_{args.split}_{args.rank}_complete.json',dict(status='COMPLETE',scenes=len(rows),
        seconds=time.perf_counter()-start,protocol_sha256=identity,peak_gpu_bytes=torch.cuda.max_memory_allocated(),owner=record))


def score_record(task):
    row,source,dest,identity=task
    from .scoring import Scoring,load_cache
    if Path(dest).exists():
        meta=json.loads(str(np.load(dest)['metadata']))
        if meta['input_sha256']!=sha(source) or meta['protocol_sha256']!=identity:
            raise ValueError('Scored input identity changed')
        return dict(token=row['token'],cached=True)
    bank=np.load(source);trajectory=bank['trajectories']
    scorer=Scoring();result=scorer.batch(load_cache(row['metric_cache_path']),trajectory.reshape(-1,8,3))
    shape=trajectory.shape[:-2]
    result={k:(v.reshape(*shape,*v.shape[1:]) if isinstance(v,np.ndarray) and len(v)==trajectory.reshape(-1,8,3).shape[0] else v)
            for k,v in result.items()}
    atomic_npz(dest,dict(token=row['token'],log=row['log_name'],protocol_sha256=identity,
        input_sha256=sha(source),score_fields=FIELDS,raw_units='[0,1]',source_metadata=json.loads(str(bank['metadata']))),**result)
    return dict(token=row['token'],cached=False)


def score(args):
    require_host();_,identity=protocol();rows=scene_rows(args.split);start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        remaining={r['token']:r for r in rows}; pending={};complete=0
        while remaining or pending:
            for token,row in list(remaining.items()):
                if len(pending)>=args.workers*3:break
                source=baseline_path(token)
                if source.exists():
                    future=pool.submit(score_record,(row,str(source),str(baseline_score_path(token)),identity))
                    pending[future]=token;del remaining[token]
            done=[f for f in pending if f.done()]
            for f in done:
                f.result();del pending[f];complete+=1
            atomic_json(ROOT/'runtime'/f'score_{args.split}_progress.json',dict(done=complete,total=len(rows),
                waiting_for_predictions=len(remaining),pending=len(pending),seconds=time.perf_counter()-start))
            if remaining and not pending and not args.watch:
                raise RuntimeError(f'Missing {len(remaining)} predictions; do not mark complete')
            if remaining or pending:time.sleep(1)
    atomic_json(ROOT/'runtime'/f'score_{args.split}_complete.json',dict(status='COMPLETE',scenes=len(rows),
        protocol_sha256=identity,seconds=time.perf_counter()-start,workers=args.workers))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=['extract','score'])
    p.add_argument('--split',choices=['D_fit','D_cal','D_audit'],required=True)
    p.add_argument('--shards',type=int,default=1);p.add_argument('--rank',type=int,default=0)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--watch',action='store_true')
    args=p.parse_args()
    try:globals()[args.action](args)
    except Exception:
        atomic_json(ROOT/'runtime'/f'{args.action}_{args.split}_{args.rank}_failure.json',
                    dict(status='FAILED',traceback=traceback.format_exc(),owner=runtime_identity()))
        raise


if __name__=='__main__':main()
