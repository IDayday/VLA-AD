"""Persistent CPU NAVSIM scoring. Missing/errors never become zero rewards."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
os.environ.setdefault('NUPLAN_MAPS_ROOT','/mnt/navsim/maps')
os.environ.setdefault('NUPLAN_MAP_VERSION','nuplan-maps-v1.0')
from common_support import *
import argparse,concurrent.futures,lzma,pickle,traceback
import evaluate_cached_rollouts as evaluator

def init():evaluator.init_worker()
def work(source,scene):
    try:
        source=Path(source);dest=OUT/'cache/scores'/source.relative_to(OUT/'cache')
        meta=dict(protocol_hash=identity(),input_sha256=sha(source),token=scene['token'])
        if v1.valid_npz(dest,meta):return dict(status='CACHED',source=str(source))
        with lzma.open(scene['metric_cache_path'],'rb') as f:cache=pickle.load(f)
        with np.load(source) as bank:
            result={k:evaluator.score_arrays(cache,bank[k],batch=128) for k in bank.files if k!='metadata'}
        npz(dest,meta,**result)
        return dict(status='OK',source=str(source))
    except Exception:return dict(status='ERROR',source=str(source),traceback=traceback.format_exc())

def main(args):
    rows={s['token']:s for s in scenes()};submitted=set();pending={};results=[];start=time.time();idle=0
    with concurrent.futures.ProcessPoolExecutor(args.workers,initializer=init) as pool:
        while True:
            paths=[p for p in list((OUT/'cache/rollouts').glob('*/*.npz'))+list((OUT/'cache/teachers').glob('*.npz')) if p.stem in rows]
            for p in paths:
                if p in submitted:continue
                if len(pending)>=args.workers*3:break
                dest=OUT/'cache/scores'/p.relative_to(OUT/'cache')
                if dest.exists() and v1.valid_npz(dest,dict(protocol_hash=identity(),input_sha256=sha(p),token=p.stem)):
                    submitted.add(p);continue
                f=pool.submit(work,str(p),rows[p.stem]);pending[f]=p;submitted.add(p)
            done,_=concurrent.futures.wait(pending,timeout=2,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                p=pending.pop(f);r=f.result();results.append(r)
                if r['status']=='ERROR':print(json.dumps(r),flush=True)
                elif len(results)%50==0:print('SCORE',len(results),'queued',len(pending),'seconds',round(time.time()-start),flush=True)
            save(OUT/'audits/scoring_progress.json',dict(workers=args.workers,completed_this_run=len(results),submitted=len(submitted),pending=len(pending),seconds=time.time()-start,errors=[r for r in results if r['status']=='ERROR']))
            if not pending and len(submitted)==len(paths):
                if not args.watch or (OUT/'audits/gpu_sampling_complete.json').exists():break
                time.sleep(5)
    csv('scoring_errors.csv',[r for r in results if r['status']=='ERROR'])

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=96);a.add_argument('--watch',action='store_true');main(a.parse_args())
