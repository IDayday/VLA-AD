"""Persistent CPU NAVSIM workers score each shared trajectory bank once."""
from common_mass import *
import argparse, concurrent.futures, multiprocessing, time, traceback

def init():
    from scoring import init as initialize
    initialize()

def task(scene,source):
    from scoring import online
    path=Path(source);relative=path.relative_to(OUT/'cache');dest=OUT/'cache/scores'/relative
    meta=dict(token=scene['token'],input_sha256=sha(path),metric_cache_sha256=sha(scene['metric_cache_path']),score_fields=FIELDS,PDMS_scale=1)
    if valid(dest,meta):return dict(path=str(path),status='CACHED')
    z=np.load(path);a=z['trajectories'];errors=[]
    for attempt in range(3):
        try:
            standard,reward=online(scene,a)
            assert standard.shape==(len(a),7) and np.isfinite(standard).all()
            npz(dest,meta,scores=standard,train_reward=reward[:,6],score_status=np.full(len(a),'OK'))
            return dict(path=str(path),status='OK',count=len(a))
        except Exception as e:errors.append(dict(error=repr(e),traceback=traceback.format_exc()))
    save(dest.with_suffix('.ERROR.json'),dict(metadata=meta,errors=errors,status='ERROR'))
    return dict(path=str(path),status='ERROR',errors=errors)

def main(workers,watch):
    from build_raw import build
    rows=scenes();lookup={s['token']:s for s in rows};seen=set();pending={};results=[];start=time.time()
    with concurrent.futures.ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn'),initializer=init) as pool:
        while True:
            for s in rows:build(s)
            paths=[p for p in list((OUT/'cache/raw').glob('*.npz'))+list((OUT/'cache/rollouts').glob('*/*.npz')) if p.stem in lookup]
            for p in paths:
                if p in seen:continue
                dest=OUT/'cache/scores'/p.relative_to(OUT/'cache')
                if valid(dest,dict(input_sha256=sha(p))):seen.add(p);continue
                if len(pending)>=workers*2:break
                pending[pool.submit(task,lookup[p.stem],str(p))]=p;seen.add(p)
            completed=[f for f in pending if f.done()]
            for f in completed:
                r=f.result();results.append(r);del pending[f]
                if r['status']=='ERROR':print(r,flush=True)
            save(OUT/'manifests/scoring_progress.json',dict(input_files=len(paths),submitted_or_cached=len(seen),pending=len(pending),finished_new=len(results),errors=sum(r['status']=='ERROR' for r in results),seconds=time.time()-start))
            if completed:print('scoring',len(seen)-len(pending),'/4000','pending',len(pending),'seconds',round(time.time()-start),flush=True)
            if not pending and ((not watch) or len(seen)==4000):break
            time.sleep(5)
    save(OUT/'audits/scoring_run.json',dict(workers=workers,blas_threads=1,results=results,files=len(seen),seconds=time.time()-start))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=96);p.add_argument('--watch',action='store_true');a=p.parse_args();main(a.workers,a.watch)
