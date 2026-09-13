"""V2 destinations, existing exact NAVSIM simulator/scorer; 96 CPU processes."""
import argparse,concurrent.futures,time,lzma,pickle
from v2_common import *
from evaluate_cached_rollouts import init_worker,score_arrays,parity

def work(task):
    scene,source,dest=task;meta=dict(ident(),token=scene['token'],input_sha256=sha(source),fields=FIELDS)
    if valid(dest,meta):return dict(cached=True,count=0)
    with lzma.open(scene['metric_cache_path'],'rb') as f:cache=pickle.load(f)
    values={};data=np.load(source)
    for key in data.files:
        a=data[key]
        if a.ndim>=3 and a.shape[-2:]==(8,3):values[key]=score_arrays(cache,a)
    assert values;npz(dest,meta,**values);return dict(cached=False,count=sum(v.size//7 for v in values.values()))

def main(a):
    rows=scenes()['scenes']
    if a.smoke:rows=[r for r in rows if r['token'] in set(subset('smoke'))]
    tasks=[]
    for r in rows:
        token=r['token'];is_heldout=a.stage.startswith('heldout');paths=list((OUT/a.stage).glob(f'*/{token}.npz')) if is_heldout else [OUT/a.stage/f'{token}.npz']
        assert len(paths)==(4 if is_heldout else 1)
        tasks.extend((r,p,OUT/'evaluator'/a.stage/p.relative_to(OUT/a.stage)) for p in paths)
    if a.validate:
        init_worker();checks=[]
        for r in rows[:4]:checks.append(parity(r,np.load(OUT/'raw_candidates'/f"{r['token']}.npz")['trajectories'][[0,32,64,128]]))
        save(OUT/'manifests/scorer_parity.json',checks)
    start=time.time();results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(96,len(tasks)),initializer=init_worker) as ex:
        for i,r in enumerate(ex.map(work,tasks,chunksize=1)):
            results.append(r)
            if i%100==0:print(a.stage,i+1,len(tasks),time.time()-start,flush=True)
    save(OUT/'manifests'/f'score_{a.stage}_{"smoke" if a.smoke else "full"}.json',dict(files=len(results),new_trajectories=sum(r['count'] for r in results),cached_files=sum(r['cached'] for r in results),workers=min(96,len(tasks)),wall_seconds=time.time()-start))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['raw_candidates','selection_local','heldout','heldout_seeded']);p.add_argument('--smoke',action='store_true');p.add_argument('--validate',action='store_true');main(p.parse_args())
