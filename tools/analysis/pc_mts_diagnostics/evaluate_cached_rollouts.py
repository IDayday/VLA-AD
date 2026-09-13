"""Multiprocess CPU scoring through the existing scalar-equivalent NAVSIM evaluator."""
from __future__ import annotations
import argparse, concurrent.futures, dataclasses, lzma, pickle, sys, time
from common import *

STATE={}
def init_worker():
    sys.path.insert(0,str(CODE))
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer,PDMScorerConfig
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    sampling=TrajectorySampling(num_poses=40,interval_length=.1)
    STATE.update(sampling=sampling,simulator=PDMSimulator(sampling),scorer=PDMScorer(sampling,PDMScorerConfig()))

def score_arrays(cache,arrays,batch=128):
    from navsim.evaluate.pdm_score_batch import pdm_score_batch_same_cache
    shape=arrays.shape[:-2];flat=np.asarray(arrays,dtype=np.float64).reshape(-1,8,3);rows=[]
    assert np.isfinite(flat).all()
    for start in range(0,len(flat),batch):
        result=pdm_score_batch_same_cache(cache,flat[start:start+batch],STATE['sampling'],STATE['simulator'],STATE['scorer'],use_exact_array_conversion=True)
        rows += [[float(dataclasses.asdict(r)[f]) for f in FIELDS] for r in result]
    a=np.array(rows);assert np.isfinite(a).all() and np.all((a>=-1e-8)&(a<=1+1e-8))
    return a.reshape(*shape,len(FIELDS))

def work(task):
    source,dest,scene,ident=task;start=time.time()
    payload=np.load(source,allow_pickle=False);meta=dict(ident,input_sha256=sha(source),token=scene['token'],score_fields=FIELDS)
    if valid_npz(dest,meta):return dict(cached=True,trajectories=0,seconds=0)
    with lzma.open(scene['metric_cache_path'],'rb') as f:cache=pickle.load(f)
    output={}
    for key in payload.files:
        if key=='metadata':continue
        a=payload[key]
        if a.ndim>=3 and a.shape[-2:]==(8,3):output[key]=score_arrays(cache,a)
    assert output,'No trajectories to score'
    save_npz(dest,meta,**output)
    return dict(cached=False,trajectories=sum(v.size//7 for v in output.values()),seconds=time.time()-start)

def parity(scene,trajs):
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.common.dataclasses import Trajectory
    with lzma.open(scene['metric_cache_path'],'rb') as f:cache=pickle.load(f)
    batch=score_arrays(cache,trajs);single=[]
    for t in trajs:
        v=pdm_score(cache,Trajectory(np.asarray(t,dtype=np.float64)),STATE['sampling'],STATE['simulator'],STATE['scorer'])
        single.append([dataclasses.asdict(v)[f] for f in FIELDS])
    err=np.abs(batch-np.asarray(single));assert err.max()<=1e-8,(scene['token'],err.max())
    return dict(token=scene['token'],max_abs_error=float(err.max()),trajectories=len(trajs),fields=FIELDS)

def main(args):
    cfg=config(args.config);sc=manifest(args.manifest);out=ROOT/cfg['output_dir'];ident=identity(cfg,sc)
    pending=pending_pc_tokens(out,cfg,sc) if args.allow_pending else set()
    if args.allow_pending:assert args.stage in ['heldout','candidate_pools']
    tasks=[]
    for r in sc['scenes']:
        token=r['token']
        paths=list((out/'rollouts').glob(f'*/{token}.npz')) if args.stage=='rollouts' else list((out/args.stage).glob(f'*/{token}.npz'))
        if args.stage=='raw_candidates':paths=[out/'raw_candidates'/f'{token}.npz']
        if args.stage in ['heldout','candidate_pools']:paths=[p for p in paths if p.parent.name in args.methods]
        expected=5 if args.stage=='rollouts' else 1 if args.stage in ['raw_candidates','selection_local'] else len(args.methods)-int('pc_mts' in args.methods and token in pending)
        assert len(paths)==expected, (args.stage,token,len(paths),expected)
        for p in paths:tasks.append((p,out/'evaluator'/args.stage/p.relative_to(out/args.stage),r,ident))
    if args.validate:
        init_worker();checks=[]
        for r in sc['scenes'][:4]:
            a=np.load(out/'rollouts/official_il'/f"{r['token']}.npz")['trajectories'][:4]
            checks.append(parity(r,a))
        save(out/'manifests/evaluator_parity.json',checks)
    start=time.time();results=[]
    actual_workers=min(args.workers,len(tasks))
    with concurrent.futures.ProcessPoolExecutor(max_workers=actual_workers,initializer=init_worker) as ex:
        for i,r in enumerate(ex.map(work,tasks,chunksize=1)):
            results.append(r)
            if i%100==0:print(json.dumps(dict(stage=args.stage,done=i+1,total=len(tasks),seconds=time.time()-start)),flush=True)
    wall=time.time()-start;count=sum(r['trajectories'] for r in results)
    suffix='_pending_'+'_'.join(args.methods) if args.allow_pending else ''
    save(out/'manifests'/f'benchmark_cpu_{args.stage}{suffix}.json',dict(workers=actual_workers,requested_workers=args.workers,wall_seconds=wall,trajectories_scored=count,trajectory_per_second=count/max(wall,1e-9),scene_per_minute=len({t[2]['token'] for t in tasks})/max(wall/60,1e-9),completed_files=len(results),cached_files=sum(r['cached'] for r in results),pending_pc_scene_count=len(pending),full_scene_count=len(sc['scenes'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--manifest');p.add_argument('--stage',choices=['rollouts','raw_candidates','selection_local','heldout','candidate_pools'],default='rollouts');p.add_argument('--workers',type=int,default=96);p.add_argument('--validate',action='store_true');p.add_argument('--allow-pending',action='store_true');p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS);main(p.parse_args())
