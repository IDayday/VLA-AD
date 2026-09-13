"""Frozen V2 selection. Denoising and held-out local results are never read."""
import argparse,concurrent.futures,lzma,pickle
from v2_common import *
from compatibility import fit,position
from perturbations import perturb,bank
from select_candidate_pools import pareto_ranks,greedy
from evaluate_cached_rollouts import init_worker,score_arrays

def choices(traj,s,q,local,dg,thresholds,ref,dcr):
    rank=pareto_ranks(s);valid=np.isfinite(s).all(1);idx=np.flatnonzero(valid);n=16;cfg=v1.config()
    chosen={'score':(sorted(idx,key=lambda i:(-s[i,6],i))[:n],[0]*n),'pareto':(greedy(idx,s,traj,n,cfg,rank),[0]*n)}
    gt=[];gtlevel=0
    for gtlevel,threshold in enumerate(thresholds):
        gt=greedy(np.flatnonzero(valid&(dg<=threshold)),s,traj,n,cfg,rank)
        if len(gt)==n:break
    levels=[gtlevel]*len(gt)
    if len(gt)<n:
        rest=sorted((i for i in idx if i not in gt),key=lambda i:(rank[i],dg[i],-s[i,6],i))[:n-len(gt)];gt.extend(rest);levels.extend([3]*len(rest))
    chosen['gt_distance']=(gt,levels)
    quality=valid&feasible(s)&(rank>=1)&(rank<=2)&(s[:,6]>=ref-.01-1e-10)
    primary=quality&(q>=50)&(q<95)&(local>=.75);pc=[];levels=[]
    for level,gate in [(1,primary),(2,quality&(q>=50)&(q<95)&(local>=.5))]:
        fresh=greedy([i for i in np.flatnonzero(gate) if i not in pc],s,traj,n-len(pc),cfg,suppress=True,existing=pc);pc.extend(fresh);levels.extend([level]*len(fresh))
    core=sorted((i for i in np.flatnonzero(quality&(q<50)&(local>=.5)) if i not in pc),key=lambda i:(-q[i],-s[i,6],i))[:n-len(pc)];pc.extend(core);levels.extend([3]*len(core));qualified_count=len(pc)
    if not pc:
        compatible=np.flatnonzero(valid&(q<95));parent=sorted(compatible,key=lambda i:(-int(feasible(s[i:i+1])[0]),-int(q[i]>=50),-s[i,6],-q[i],i))
        if parent:pc=[int(parent[0])];levels=[6]
        else:
            native=[i for i in range(32,64) if valid[i]];assert native;pc=[min(native,key=lambda i:(dcr[i],-s[i,6],i))];levels=[7]
    chosen['pc_mts']=(pc,levels)
    return chosen,dict(primary_eligible_count=int(primary.sum()),qualified_raw_count=qualified_count,zero_qualified_parent=qualified_count==0),rank

def work(scene):
    token=scene['token'];rawpath=OUT/'raw_candidates'/f'{token}.npz';raw=np.load(rawpath);traj=raw['trajectories'];s=np.load(OUT/'evaluator/raw_candidates'/f'{token}.npz')['trajectories'];local=feasible(np.load(OUT/'evaluator/selection_local'/f'{token}.npz')['trajectories']).mean(1);pos=np.load(OUT/'positions'/f'{token}.npz');dg=distance(traj,np.asarray(scene['gt'])[None])[:,0];thresholds=read(V1/'manifests/gt_distance_thresholds.json')['thresholds'][scene['command']];ref=float(np.load(V1/'evaluator/rollouts/official_il'/f'{token}.npz')['reference'][0,6]);allchoices,audit,ranks=choices(traj,s,pos['q_holdout'],local,dg,thresholds,ref,pos['policy_distance_knn'])
    il=np.load(OUT/'il_banks'/f'{token}.npz');cal=fit(il['R'],il['Q']);cache=None
    for method,(idx,levels) in allchoices.items():
        dest=OUT/'candidate_pools'/method/f'{token}.npz';meta=dict(ident(),token=token,method=method,raw_sha256=sha(rawpath),audit=audit if method=='pc_mts' else {},V1_GT_thresholds=thresholds if method=='gt_distance' else None)
        if valid(dest,meta):continue
        ts=[traj[i] for i in idx];ss=[s[i] for i in idx];sources=[str(raw['source'][i]) for i in idx];ids=[str(raw['candidate_id'][i]) for i in idx];alphas=[raw['bridge_alpha'][i] for i in idx];parents=['']*len(idx);fills=[False]*len(idx);accepted=len(idx);qualified=[method!='pc_mts' or not audit['zero_qualified_parent']]*len(idx)
        if len(ts)<16:
            with lzma.open(scene['metric_cache_path'],'rb') as f:cache=pickle.load(f)
        while len(ts)<16:
            j=len(ts);parent=j%accepted;new=None;value=None
            for attempt in range(4):
                trial=perturb(ts[parent],.01/(attempt+1),'spline',1 if j%2 else -1,seed(token,'v2_tiny_fill',j*4+attempt));value=score_arrays(cache,trial[None])[0];p=position(trial[None],cal)
                # Fillers retain compatibility; strict parents additionally retain feasibility and quality.
                okay=p['q_holdout'][0]<95 and np.isfinite(value).all()
                if qualified[parent]:okay=okay and bool(feasible(value[None])[0]) and value[6]>=ref-.01-1e-10
                if okay:new=trial;break
            exact=new is None
            if exact:new=ts[parent].copy();value=score_arrays(cache,new[None])[0]
            ts.append(new);ss.append(value);sources.append(sources[parent]);ids.append(f'pc_mts_{"duplicate" if exact else "tiny"}_{j:02d}');alphas.append(alphas[parent]);parents.append(ids[parent]);fills.append(True);levels.append(5 if exact else 4);idx.append(-1);qualified.append(qualified[parent])
        ts=np.stack(ts);ss=np.stack(ss);pp=position(ts,cal);ood=(pp['q_holdout']>=95)&np.asarray(fills)
        assert ts.shape==(16,8,3) and np.isfinite(ts).all() and np.isfinite(ss).all()
        npz(dest,meta,trajectories=ts,scores=ss,source=np.asarray(sources),candidate_id=np.asarray(ids),raw_index=np.asarray(idx),fallback_level=np.asarray(levels),is_fill=np.asarray(fills),fill_parent_id=np.asarray(parents),qualified=np.asarray(qualified),ood_filler=ood,bridge_alpha=np.asarray(alphas),**pp)
    return dict(token=token,**audit)

def heldout(scene):
    token=scene['token']
    for method in METHODS:
        src=OUT/'candidate_pools'/method/f'{token}.npz';dest=OUT/'heldout'/method/f'{token}.npz';meta=dict(ident(),token=token,method=method,pool_sha256=sha(src),stream='v2_heldout_robustness')
        if not valid(dest,meta):npz(dest,meta,trajectories=np.stack([bank(t,token,'v2_heldout_robustness') for t in np.load(src)['trajectories']]))
    return token

def main(a):
    rows=scenes()['scenes'];tag='smoke' if a.smoke else 'full'
    assert (OUT/'manifests'/f'raw_audit_{tag}.json').exists(),'Reservoir must be audited before selection'
    if a.smoke:rows=[r for r in rows if r['token'] in set(subset('smoke'))]
    fn=heldout if a.heldout else work
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(96,len(rows)),initializer=init_worker if not a.heldout else None) as ex:result=list(ex.map(fn,rows,chunksize=1))
    save(OUT/'manifests'/f'{"heldout_generation" if a.heldout else "selection"}_{tag}.json',dict(identity=ident(),rows=result,scene_count=len(rows)))
    print('Completed',fn.__name__,len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--heldout',action='store_true');main(p.parse_args())
