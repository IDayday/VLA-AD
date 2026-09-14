"""True NAVSIM scores of geometric interventions and the actual training targets."""
import argparse, concurrent.futures, traceback
from common_fd import *
from evaluate_cached_rollouts import init_worker, score_arrays, parity

def task(scene):
    token=scene['token']; dest=OUT/'cache/scored'/f'{token}.npz'
    bank={}; scores={}; inputs={}
    for m in MODELS:
        t,s,meta=load_model_scene(m,token)
        bank[m]=t;scores[m]=s
        inputs[m]=sha(v1.OUT/'rollouts'/m/f'{token}.npz')
    teachers={}
    for m in ARCHIVES:
        p,r,idx,t=teacher_record(m,token)
        teachers['teacher__'+m]=t
        inputs['archive__'+m]=sha(p)
    ident=dict(protocol_sha256=config_identity(),token=token,input_hashes=inputs,metric_cache_sha256=sha(scene['metric_cache_path']))
    if v1.valid_npz(dest,ident):return dict(token=token,cached=True,count=0)
    arrays=make_counterfactuals(bank); arrays.update(teachers)
    with lzma.open(scene['metric_cache_path'],'rb') as f:cache=pickle.load(f)
    shapes={k:len(v) for k,v in arrays.items()}
    all_scores=score_arrays(cache,np.concatenate(list(arrays.values())),batch=128)
    result={}; begin=0
    for k,n in shapes.items():
        result[k]=all_scores[begin:begin+n];begin+=n
    v1.save_npz(dest,ident,**result)
    return dict(token=token,cached=False,count=begin)

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=96);p.add_argument('--smoke-only',action='store_true');args=p.parse_args()
    config_identity();init_worker();checks=[]
    for scene in SCENES[:8]:
        t,s,_=load_model_scene('official_il',scene['token'])
        checks.append(parity(scene,t[:4]))
        with lzma.open(scene['metric_cache_path'],'rb') as f:cache=pickle.load(f)
        delta=np.max(np.abs(score_arrays(cache,t[:4])-s[:4]));assert delta<1e-8
        checks[-1]['old_cache_max_abs_error']=float(delta)
    save(OUT/'audits/evaluator_parity.json',checks)
    if args.smoke_only:return
    start=time.time();rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker) as ex:
        jobs={ex.submit(task,s):s['token'] for s in SCENES}
        for f in concurrent.futures.as_completed(jobs):
            try:rows.append(f.result())
            except Exception:
                save(OUT/'audits/scoring_error.json',dict(token=jobs[f],traceback=traceback.format_exc()))
                raise
            if len(rows)%25==0:print(json.dumps(dict(done=len(rows),total=len(SCENES),seconds=time.time()-start)),flush=True)
    save(OUT/'audits/scoring_completion.json',dict(workers=args.workers,seconds=time.time()-start,scene_count=len(rows),new_trajectories=sum(x['count'] for x in rows),rows=rows))

if __name__=='__main__':main()
