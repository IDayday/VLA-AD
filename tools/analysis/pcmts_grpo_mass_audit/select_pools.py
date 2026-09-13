"""A-only selection. No B paths or B statistics are accepted by this module."""
from common_mass import *
import math, concurrent.futures, time

def matched_pairs(pool_pc,pool_other,records,scores):
    from scipy.optimize import linear_sum_assignment
    common=set(pool_pc)&set(pool_other);aa=sorted(set(pool_pc)-common);bb=sorted(set(pool_other)-common);pairs=[]
    def source(i):return records[i]['source'] if records[i]['source']!='multi_source' else 'multi_source:'+','.join(records[i]['all_source_tags'])
    keys={(source(i),float(scores[i,0]),float(scores[i,1])) for i in aa}
    for key in sorted(keys):
        a=[i for i in aa if (source(i),float(scores[i,0]),float(scores[i,1]))==key];b=[i for i in bb if (source(i),float(scores[i,0]),float(scores[i,1]))==key]
        if not b:continue
        gap=100*abs(scores[a,6,None]-scores[np.asarray(b)[None,:],6]);cost=np.full((len(a),len(b)+len(a)),1000.)
        cost[:,:len(b)]=np.where(gap<=.25+1e-12,gap,2000.)
        row,col=linear_sum_assignment(cost)
        for x,y in zip(row,col):
            if y<len(b) and cost[x,y]<1000:pairs.append(dict(pc_index=a[x],other_index=b[y],exact_source=key[0],NC=key[1],DAC=key[2],PDMS_gap_points=float(gap[x,y])))
    return dict(pairs=pairs,pc_different_count=len(aa),other_different_count=len(bb),shared_count=len(common),unmatched_pc=len(aa)-len(pairs),unmatched_other=len(bb)-len(pairs))

def rank_pareto(objectives):
    a=np.asarray(objectives);dominates=(a[:,None]>=a[None,:]-1e-12).all(2)&(a[:,None]>a[None,:]+1e-12).any(2)
    ranks=np.zeros(len(a),dtype=int);left=np.ones(len(a),bool);r=1
    while left.any():
        front=left&~dominates[left].any(0);assert front.any()
        ranks[front]=r;left[front]=False;r+=1
    return ranks

def quality_order(allowed,pdms,ids,geometry,selected=None,limit=16):
    selected=[] if selected is None else list(selected);remaining=set(int(i) for i in allowed)-set(selected)
    while remaining and len(selected)<limit:
        top=max(pdms[i] for i in remaining);ties=[i for i in remaining if top-pdms[i]<=.01/100+1e-12]
        if selected:
            separation={i:float(geometry[i,selected].min()) for i in ties};largest=max(separation.values());ties=[i for i in ties if abs(separation[i]-largest)<=1e-12]
        winner=min(ties,key=lambda i:ids[i]);selected.append(winner);remaining.remove(winner)
    return selected

def score_select(eligible,pdms,ids,geometry):return quality_order(np.flatnonzero(eligible),pdms,ids,geometry)
def pareto_select(eligible,pdms,objectives,ids,geometry):
    indices=np.flatnonzero(eligible);ranks=rank_pareto(objectives[indices]);selected=[]
    for rank in sorted(set(ranks)):
        selected=quality_order(indices[ranks==rank],pdms,ids,geometry,selected)
        if len(selected)==16:break
    return selected,dict(objective_variances=np.var(objectives[indices],0).tolist() if len(indices) else [None]*3,front_count=int(ranks.max()) if len(ranks) else 0,front_sizes={str(r):int((ranks==r).sum()) for r in sorted(set(ranks))})
def geometry_select(eligible,pdms,ids,geometry,d_gt,radius):return quality_order(np.flatnonzero(eligible&(d_gt<=radius)),pdms,ids,geometry)
def mass_select(eligible,pdms,ids,geometry,hits_A,n_A,pmin):return quality_order(np.flatnonzero(eligible&(hits_A>=math.ceil(n_A*pmin))),pdms,ids,geometry)

def select(trajectories,records,scores,reference,hits_A,n_A,d_gt,method,epsilon=.5,pmin=.03,radius=1.,include=None):
    n=len(records);valid=np.isfinite(scores).all(1)&np.isfinite(trajectories).all((1,2));valid&=np.ones(n,bool) if include is None else include
    hard=(scores[:,[0,1]]>=1-1e-8).all(1);quality=scores[:,6]>=reference-1e-12;eligible=valid&hard&quality
    ids=[r['candidate_id'] for r in records];geometry=distance(trajectories,trajectories);pdms=scores[:,6];audit={}
    if method=='score':strict=score_select(eligible,pdms,ids,geometry)
    elif method=='pareto':strict,audit=pareto_select(eligible,pdms,scores[:,[2,3,5]],ids,geometry)
    elif method=='gt_geometry':strict=geometry_select(eligible,pdms,ids,geometry,d_gt,radius)
    elif method=='grpo_mass_pc':strict=mass_select(eligible,pdms,ids,geometry,hits_A,n_A,pmin)
    else:raise ValueError(method)
    operational=list(strict);levels=[0]*len(strict)
    for level,mask in [(1,eligible),(2,valid&hard),(3,valid)]:
        before=len(operational);operational=quality_order(np.flatnonzero(mask),pdms,ids,geometry,operational);levels.extend([level]*(len(operational)-before))
    rows=[]
    for i,level in zip(operational,levels):
        qualified=(eligible[i] and (method!='gt_geometry' or d_gt[i]<=radius) and (method!='grpo_mass_pc' or hits_A[i]>=math.ceil(n_A*pmin)))
        assert qualified==(level==0)
        rows.append(dict(raw_index=i,selection_level=level,strict_qualified=bool(qualified),probability_qualified=bool(hits_A[i]>=math.ceil(n_A*pmin)),geometry_qualified=bool(d_gt[i]<=radius),hard_safe=bool(hard[i]),conservative_feasible=bool(feasible(scores[i:i+1])[0]),quality_floor_pass=bool(quality[i]),training_eligible=bool(qualified and level==0),valid_mask=True))
    return dict(strict_indices=strict,operational_indices=operational,items=rows,valid_unique_count=int(valid.sum()),strict_count=len(strict),operational_count=len(operational),status='OK' if len(operational)==16 else 'DATA_INSUFFICIENT',pareto_audit=audit)

def variants():
    rows=[dict(name='primary',epsilon=.5,pmin=.03,radius=1.,no_il=False)]
    for e in [.25,.5,1.]:
        for p in [.01,.03,.05]:
            if e==.5 and p==.03:continue
            rows.append(dict(name=f'eps{e}_p{p}',epsilon=e,pmin=p,radius=1.,no_il=False))
    for g in [.5,2.]:rows.append(dict(name=f'gt{g}',epsilon=.5,pmin=.03,radius=g,no_il=False))
    rows.append(dict(name='no_il_native',epsilon=.5,pmin=.03,radius=1.,no_il=True))
    return rows

def task(scene):
    t=scene['token'];dest=OUT/'cache/selection'/f'{t}.json'
    rawpath=OUT/'cache/raw'/f'{t}.npz';scorepath=OUT/'cache/scores/raw'/f'{t}.npz';apath=OUT/'cache/rollouts/A'/f'{t}.npz';cpath=OUT/'cache/scores/rollouts/C'/f'{t}.npz'
    hashes={str(p):sha(p) for p in [rawpath,scorepath,apath,cpath]}
    if dest.exists():
        found=read(dest);assert found['input_hashes']==hashes and found['protocol_hash']==protocol();return sha(dest)
    raw=np.load(rawpath)['trajectories'];records=read(rawpath.with_suffix('.json'))['candidates'];scores=np.load(scorepath)['scores'];a=np.load(apath)['trajectories'];c=np.load(cpath)['scores'];assert len(a)==1024 and len(c)==64
    reference=float(c[:,6].mean());d_gt=distance(raw,np.asarray(scene['gt'])[None])[:,0];d_a=distance(raw,a)
    result={};pureC=np.asarray([r['all_source_tags']==['IL_native_C'] for r in records]);hits={e:(d_a<=e).sum(1) for e in [.25,.5,1.]}
    for v in variants():
        result[v['name']]={m:select(raw,records,scores,reference,hits[v['epsilon']],len(a),d_gt,m,**{k:v[k] for k in ['epsilon','pmin','radius']},include=~pureC if v['no_il'] else None) for m in METHODS}
    matches={}
    for pool_type,key in [('strict','strict_indices'),('operational','operational_indices')]:
        matches[pool_type]={m:matched_pairs(result['primary']['grpo_mass_pc'][key],result['primary'][m][key],records,scores) for m in METHODS[:3]}
    save(dest,dict(token=t,protocol_hash=protocol(),input_hashes=hashes,raw_candidate_ids_hash=digest([r['candidate_id'] for r in records]),reference_PDMS=100*reference,A_denominator=len(a),hits_A={str(e):x.tolist() for e,x in hits.items()},d_GT=d_gt.tolist(),variants=result,matched_pairs=matches,selection_used_stream='A',frozen_at=utc()))
    return sha(dest)

def main():
    assert sha(CONFIG)==read(OUT/'manifests/protocol_frozen.json')['sha256']
    start=time.time()
    with concurrent.futures.ProcessPoolExecutor(24) as pool:hashes=list(pool.map(task,scenes()))
    seal_selection({s['token']:h for s,h in zip(scenes(),hashes)},time.time()-start)
    print('ALL PRIMARY, SENSITIVITY, NO-NATIVE SELECTIONS FROZEN',digest(hashes),flush=True)

def seal_selection(hashes,seconds):
    path=OUT/'manifests/selection_frozen.json'
    ordered=[hashes[s['token']] for s in scenes()]
    if path.exists():
        frozen=read(path)
        assert frozen['selection_hashes']==hashes and frozen['joint_hash']==digest(ordered)
        assert frozen['protocol_hash']==protocol() and frozen['variants']==variants()
        return frozen  # Preserve the original pre-B timestamp and file hash.
    assert not (OUT/'audits/B_validation.json').exists(), 'Cannot create an initial selection seal after B analysis'
    frozen=dict(protocol_hash=protocol(),scene_count=1000,variants=variants(),selection_hashes=hashes,joint_hash=digest(ordered),frozen_at=utc(),B_read_by_selector=False,seconds=seconds)
    save(path,frozen);return frozen

if __name__=='__main__':main()
