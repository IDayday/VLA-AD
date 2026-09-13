"""Frozen M1-M4 rules on one common reservoir; audit every fallback."""
import argparse, sys, concurrent.futures, lzma, pickle
from common import *

COVERAGE_POLICY=ROOT/'configs/pc_mts_diagnostics/coverage_fallback.yaml'

def coverage_select(scores,trajs,local,rank,ref,cfg):
    """Authorized coverage extension. Original support percentiles stay unchanged."""
    valid=np.isfinite(scores).all(1);safe=valid&feasible(scores)
    gates=[safe&(rank<=cfg['pareto_qualified_max_front'])&(scores[:,6]>=ref-1e-10)&(local>=.5),safe&(local>=.5),safe,valid&(~hard_failure(scores))]
    selected=[];levels=[]
    for level,gate in enumerate(gates,6):
        fresh=greedy([i for i in np.flatnonzero(gate) if i not in selected],scores,trajs,cfg['pool_size']-len(selected),cfg,suppress=True,existing=selected)
        selected.extend(fresh);levels.extend([level]*len(fresh))
        if len(selected)==cfg['pool_size']:break
    if not selected:raise RuntimeError('No NC=DAC=1 raw candidate: coverage extension cannot supply an accepted parent')
    return selected,levels

def pareto_ranks(scores):
    sys.path.insert(0,str(CODE))
    from navsim.agents.recogdrive.pareto_support.pareto_archive import pareto_front_mask,PARETO_COMPONENT_KEYS
    assert list(PARETO_COMPONENT_KEYS)==config()['pareto_components']
    values=scores[:,[FIELDS.index(k) for k in PARETO_COMPONENT_KEYS]]
    remain=np.isfinite(scores).all(1);rank=np.full(len(scores),-1);n=1
    while remain.any():
        front=pareto_front_mask(values,remain,eps=0.);assert front.any()
        rank[front]=n;remain[front]=False;n+=1
    return rank

def greedy(indices,scores,trajs,count,cfg,rank=None,suppress=False,existing=None):
    remaining=list(map(int,indices));chosen=[];dist=distance(trajs,trajs);tolerance=cfg['score_tie_tolerance_points']/100
    while remaining and len(chosen)<count:
        context=list(existing or [])+chosen
        available=remaining
        if suppress and context:
            different=[i for i in available if dist[i,context].min()>=cfg['redundancy_threshold_m']]
            if different:available=different
        if rank is not None:available=[i for i in available if rank[i]==min(rank[j] for j in available)]
        best=max(scores[i,6] for i in available);ties=[i for i in available if scores[i,6]>=best-tolerance]
        index=max(ties,key=lambda i:(float(dist[i,context].min()) if context else 0.,scores[i,6],-i))
        chosen.append(index);remaining.remove(index)
    return chosen

def choose(scores,trajs,q,local,gt_dist,thresholds,ref,cfg):
    n=cfg['pool_size'];rank=pareto_ranks(scores);valid=np.isfinite(scores).all(1)
    idx=np.flatnonzero(valid)
    result={'score':dict(indices=sorted(idx,key=lambda i:(-scores[i,6],i))[:n],fallback_level=0),
            'pareto':dict(indices=greedy(idx,scores,trajs,n,cfg,rank),fallback_level=0)}
    chosen=[];gt_level=0
    for level,threshold in enumerate(thresholds):
        chosen=greedy(np.flatnonzero(valid&(gt_dist<=threshold)),scores,trajs,n,cfg,rank)
        gt_level=level
        if len(chosen)>=n:break
    result['gt_distance']=dict(indices=chosen,fallback_level=gt_level,threshold_m=float(thresholds[gt_level]))
    quality=valid&feasible(scores)&(scores[:,6]>=ref-1e-10)&(rank<=cfg['pareto_qualified_max_front'])
    primary=quality&(q>=50)&(q<95)&(local>=.75)
    pc=[];levels=[]
    for level,(lo,hi,lf) in enumerate([(50,95,.75),(50,95,.5),(40,95,.5),(40,99,.5)]):
        eligible=quality&(q>=lo)&(q<hi)&(local>=lf)
        fresh=greedy([i for i in np.flatnonzero(eligible) if i not in pc],scores,trajs,n-len(pc),cfg,suppress=True,existing=pc)
        pc.extend(fresh);levels.extend([level]*len(fresh))
        if len(pc)>=n:break
    result['pc_mts']=dict(indices=pc,fallback_levels=levels,fallback_level=max(levels,default=3),primary_eligible_count=int(primary.sum()),final_eligible_count=len(pc),no_parent=len(pc)==0)
    return result,rank

def calibrate(out,sc):
    values={}
    for r in sc['scenes']:
        a=np.load(out/'rollouts/official_il'/f"{r['token']}.npz")['trajectories']
        values.setdefault(r['command'],[]).extend(distance(a,np.asarray(r['gt'])[None])[:,0])
    return {c:np.quantile(v,[.8,.9,.95]).tolist() for c,v in values.items()}

def finalize_scene(task):
    from build_candidate_reservoir import smooth_perturbation
    from evaluate_cached_rollouts import init_worker,score_arrays
    out,scene,cfg,ident,thresholds,empty_policy,methods,defer_empty=task;token=scene['token']
    raw_path=out/'raw_candidates'/f'{token}.npz';raw=np.load(raw_path);trajs=raw['trajectories'];scores=np.load(out/'evaluator/raw_candidates'/f'{token}.npz')['trajectories']
    local=feasible(np.load(out/'evaluator/selection_local/common'/f'{token}.npz')['trajectories']).mean(1)
    il=np.load(out/'rollouts/official_il'/f'{token}.npz')['trajectories'];pos=policy_position(trajs,il,cfg['knn']);dg=distance(trajs,np.asarray(scene['gt'])[None])[:,0]
    ref=float(np.load(out/'evaluator/rollouts/official_il'/f'{token}.npz')['reference'][0,6])
    choices,ranks=choose(scores,trajs,pos['q_policy'],local,dg,thresholds[scene['command']],ref,cfg)
    if 'pc_mts' in methods and choices['pc_mts']['no_parent'] and empty_policy=='error' and not defer_empty:raise RuntimeError(f'{token}: no accepted PC-MTS parent after all allowed fallback levels')
    cache=None;records=[]
    for method,choice in choices.items():
        if method not in methods:continue
        if method=='pc_mts' and choice['no_parent'] and defer_empty:
            assert not (out/'candidate_pools'/method/f'{token}.npz').exists()
            continue
        dest=out/'candidate_pools'/method/f'{token}.npz';meta=dict(ident,token=token,method=method,raw_sha256=sha(raw_path),empty_policy=empty_policy if method=='pc_mts' and not choice['indices'] else 'error',selection_status='complete',raw_initial_selected_count=len(choice['indices']))
        if method=='pc_mts' and not choice['indices'] and empty_policy=='coverage':
            meta.update(coverage_policy_hash=sha(COVERAGE_POLICY),coverage_policy_path=str(COVERAGE_POLICY),primary_pc_failure=True,selection_status='coverage_fallback')
        if valid_npz(dest,{k:v for k,v in meta.items() if k!='selection_status'}):continue
        selected=list(map(int,choice['indices']));chosen=[trajs[i] for i in selected];scored=[scores[i] for i in selected]
        ids=[str(raw['candidate_id'][i]) for i in selected];sources=[str(raw['source'][i]) for i in selected]
        fallback=list(choice.get('fallback_levels',[choice['fallback_level']]*len(selected)))
        fills=[False]*len(selected);parents=['']*len(selected)
        if not selected and empty_policy=='coverage':
            selected,fallback=coverage_select(scores,trajs,local,ranks,ref,cfg)
            chosen=[trajs[i] for i in selected];scored=[scores[i] for i in selected];ids=[str(raw['candidate_id'][i]) for i in selected];sources=[str(raw['source'][i]) for i in selected];fills=[False]*len(selected);parents=['']*len(selected)
        if not selected:
            if empty_policy=='missing':
                meta['selection_status']='no_qualified_parent';meta['primary_pc_failure']=True
                save_npz(dest,meta,trajectories=np.empty((0,8,3)),scores=np.empty((0,7)),source=np.asarray([],dtype='U1'),candidate_id=np.asarray([],dtype='U1'),is_fill=np.empty(0,bool),fill_parent_id=np.asarray([],dtype='U1'),fallback_level=np.empty(0,int),raw_index=np.empty(0,int));continue
            feasible_ids=np.flatnonzero(feasible(scores))
            if not len(feasible_ids):raise RuntimeError(f'{token}: emergency fallback requested but no feasible raw candidate')
            selected=[int(feasible_ids[np.argmax(scores[feasible_ids,6])])];index=selected[0]
            chosen=[trajs[index]];scored=[scores[index]];ids=[str(raw['candidate_id'][index])];sources=[str(raw['source'][index])];fallback=[6];fills=[True];parents=[ids[0]];meta['primary_pc_failure']=True
        if len(chosen)<cfg['pool_size']:
            init_worker()
            with lzma.open(scene['metric_cache_path'],'rb') as f:cache=pickle.load(f)
        accepted_parent_count=len(chosen)
        while len(chosen)<cfg['pool_size']:
            j=len(chosen);parent=j%accepted_parent_count;original=chosen[parent];new=None;new_scores=None
            if empty_policy=='coverage' and meta.get('primary_pc_failure'):
                new=original.copy();new_scores=score_arrays(cache,new[None])[0]
                chosen.append(new);scored.append(new_scores);ids.append(f'{method}_coverage_duplicate_{j:02d}');sources.append('exact_duplicate');fills.append(True);parents.append(ids[parent]);fallback.append(10);selected.append(-1);continue
            for trial in range(4):
                perturb=smooth_perturbation(original,.01/(trial+1),seed_for(token,method+'_filler',j*4+trial))
                value=score_arrays(cache,perturb[None])[0]
                quality=bool(feasible(value[None])[0]) and value[6]>=ref-1e-10
                if method=='gt_distance':quality &= distance(perturb[None],np.asarray(scene['gt'])[None])[0,0]<=choice['threshold_m']
                if quality:new=perturb;new_scores=value;break
            exact=new is None
            if exact:
                new=original.copy();new_scores=score_arrays(cache,new[None])[0]
            chosen.append(new);scored.append(new_scores);ids.append(f'{method}_fill_{j:02d}');sources.append('exact_duplicate' if exact else 'tiny_smooth_fill');fills.append(True);parents.append(ids[parent]);fallback.append(6 if meta.get('primary_pc_failure') else 5 if exact else 4);selected.append(-1)
        save_npz(dest,meta,trajectories=np.stack(chosen),scores=np.stack(scored),source=np.asarray(sources),candidate_id=np.asarray(ids),is_fill=np.asarray(fills),fill_parent_id=np.asarray(parents),fallback_level=np.asarray(fallback),raw_index=np.asarray(selected))
    return dict(token=token,pc_initial_count=choices['pc_mts']['final_eligible_count'],pc_primary_count=choices['pc_mts']['primary_eligible_count'])

def main(args):
    cfg=config(args.config);sc=manifest(args.manifest);out=ROOT/cfg['output_dir'];thresholds=calibrate(out,sc)
    save(out/'manifests/gt_distance_thresholds.json',dict(quantiles=[.8,.9,.95],thresholds=thresholds,source='official IL rollout-to-GT only',scene_manifest_hash=digest(sc)))
    if not args.audit_only:
        if args.defer_empty:assert args.empty_policy=='error', 'Deferral must not choose a new empty-pool policy'
        tasks=[(out,r,cfg,identity(cfg,sc),thresholds,args.empty_policy,args.methods,args.defer_empty) for r in sc['scenes']]
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(cfg['cpu_workers'],len(tasks))) as ex:rows=list(ex.map(finalize_scene,tasks,chunksize=1))
        save(out/'manifests'/('pool_construction_'+'_'.join(args.methods)+'.json'),dict(rows=rows,empty_policy=args.empty_policy,methods=args.methods,defer_empty=args.defer_empty))
        if args.defer_empty and 'pc_mts' in args.methods:
            save(out/'manifests/pc_pending_execution.json',dict(identity=identity(cfg,sc),status='partial_execution_not_full_cohort_result',reason='No accepted parent after frozen fallback levels; empty-pool rule remains undecided',pending_tokens=[r['token'] for r in rows if r['pc_initial_count']==0],eligible_tokens=[r['token'] for r in rows if r['pc_initial_count']>0],full_scene_count=len(rows)))
        if args.empty_policy=='coverage' and 'pc_mts' in args.methods:
            save(out/'manifests/coverage_completion.json',dict(identity=identity(cfg,sc),coverage_policy_hash=sha(COVERAGE_POLICY),coverage_policy=yaml.safe_load(COVERAGE_POLICY.read_text()),full_scene_count=len(rows),original_eligible_scene_count=sum(r['pc_initial_count']>0 for r in rows),coverage_scene_count=sum(r['pc_initial_count']==0 for r in rows),status='all_scene_pools_constructed'))
        return
    audits=[]
    for r in sc['scenes']:
        token=r['token'];raw=np.load(out/'raw_candidates'/f'{token}.npz');trajs=raw['trajectories'];source=raw['source'];ids=raw['candidate_id']
        scores=np.load(out/'evaluator/raw_candidates'/f'{token}.npz')['trajectories']
        loc=np.load(out/'evaluator/selection_local/common'/f'{token}.npz')['trajectories'];local=feasible(loc).mean(1)
        if args.reservoir_model_count:
            keep=np.concatenate([np.arange(cfg['raw_gt_count'])]+[cfg['raw_gt_count']+i*cfg['raw_model_count']+np.arange(args.reservoir_model_count) for i in range(4)])
            trajs,scores,local,source,ids=trajs[keep],scores[keep],local[keep],source[keep],ids[keep]
        il=np.load(out/'rollouts/official_il'/f'{token}.npz')['trajectories'];pos=policy_position(trajs,il,cfg['knn']);dg=distance(trajs,np.asarray(r['gt'])[None])[:,0]
        ref=float(np.load(out/'evaluator/rollouts/official_il'/f'{token}.npz')['reference'][0,6])
        choices,ranks=choose(scores,trajs,pos['q_policy'],local,dg,thresholds[r['command']],ref,cfg)
        audit=dict(token=token,raw_count=len(trajs),choices=choices)
        # Selection audit deliberately exposes an empty PC-MTS set instead of inventing a qualifying parent.
        audits.append(audit)
    name=f'selection_audit_raw_model_{args.reservoir_model_count or cfg["raw_model_count"]}.json'
    save(out/'manifests'/name,dict(rows=audits,pc_primary_short_ratio=float(np.mean([r['choices']['pc_mts']['primary_eligible_count']<cfg['pool_size'] for r in audits])),pc_no_parent_ratio=float(np.mean([r['choices']['pc_mts']['no_parent'] for r in audits]))))
    print(name,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--manifest');p.add_argument('--audit-only',action='store_true');p.add_argument('--reservoir-model-count',type=int);p.add_argument('--empty-policy',choices=['error','missing','emergency','coverage'],default='error');p.add_argument('--defer-empty',action='store_true');p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS);main(p.parse_args())
