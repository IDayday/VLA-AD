"""Resumable real NAVSIM scoring, sharded by host without cache collisions."""
from common_r import *
import argparse,concurrent.futures
def init():
    from scoring import init as original_init
    original_init()
def task(item):
    path,scene=item
    from scoring import cache,score_arrays,STATE,SCORERS
    dest=OUT/'cache/scores'/path.relative_to(OUT/'cache/rollouts');meta=dict(identity(),input_sha256=sha(path))
    if v3.valid(dest,meta):return str(dest)
    STATE['scorer']=SCORERS['evaluation'];scores=score_arrays(cache(scene),np.load(path)['trajectories'])
    npz(dest,meta,scores=scores);return str(dest)
def completed_inputs(root,lookup):
    # Atomic writers intentionally expose *.tmp.npz while serializing. Only
    # exact manifest token names denote committed inference results.
    return sorted(p for p in Path(root).glob('*/*/*.npz') if p.stem in lookup)
def main(args):
    lookup={r['token']:r for r in scenes()};done=0
    with concurrent.futures.ProcessPoolExecutor(args.workers,initializer=init) as pool:
        while True:
            allpaths=completed_inputs(OUT/'cache/rollouts',lookup)
            paths=[p for p in allpaths if int(digest(str(p.relative_to(OUT)))[:8],16)%args.shards==args.shard]
            pending=[p for p in paths if not (OUT/'cache/scores'/p.relative_to(OUT/'cache/rollouts')).exists()]
            for i,_ in enumerate(pool.map(task,[(p,lookup[p.stem]) for p in pending],chunksize=1)):
                done+=1
                if done%200==0:print('Scoring',args.shard,done,'this pass',i+1,len(pending),flush=True)
            if not args.watch or len(allpaths)==16800:break
            time.sleep(10)
    save(OUT/'manifests'/f'score_shard{args.shard}.json',dict(identity(),files=len(paths),shard=args.shard,shards=args.shards,host=socket.gethostname()))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--workers',type=int,default=64);p.add_argument('--watch',action='store_true');main(p.parse_args())
