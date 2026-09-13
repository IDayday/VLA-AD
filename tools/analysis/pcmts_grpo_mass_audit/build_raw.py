"""One shared reservoir, constructed without ever reading selection/evaluation A/B."""
from common_mass import *
from perturbations import perturb

def structured(gt,token,batch=0):
    c=cfg()['perturbations'];amp=c['amplitudes_max_point_displacement_m'];out=[]
    for family,n in c['family_quotas'].items():
        for i in range(n):
            a=amp[(i//2)%len(amp)];direction=c['directions'][i%2];seed=random_seed(token,f'GT_structured_{batch}_{family}',i)
            t=perturb(gt,a,family,direction,seed,seeded_shape=True)
            out.append((t,dict(source='GT_structured',generation_seed=seed,perturbation_family=family,perturbation_parameters=dict(amplitude_max_point_m=a,direction=direction,seeded_shape=True,batch=batch),parent_candidate_id=f'{token}:GT',checkpoint_hash=None)))
    assert len(out)==128
    return out

def deduplicate(items,token):
    groups={}
    for t,r in items:
        assert np.shape(t)==(8,3) and np.isfinite(t).all()
        h=content_hash(t);groups.setdefault(h,[]).append((np.asarray(t),r))
    trajectories=[];rows=[]
    for h,parents in sorted(groups.items()):
        tags=sorted({r['source'] for _,r in parents})
        # Symmetric representative choice, independent of source insertion order.
        t=min((a for a,_ in parents),key=lambda a:a.astype('<f8').tobytes())
        source=tags[0] if len(tags)==1 else 'multi_source'
        trajectories.append(t)
        rows.append(dict(token=token,candidate_id=f'{token}:{h[:20]}',unique_parent_id=h,content_hash=h,source=source,all_source_tags=tags,all_parents=[r for _,r in sorted(parents,key=lambda x:digest(x[1]))],checkpoint_hash=sorted({r['checkpoint_hash'] for _,r in parents if r.get('checkpoint_hash')}),generation_seed=[r.get('generation_seed') for _,r in parents],perturbation_family=sorted({r['perturbation_family'] for _,r in parents if r.get('perturbation_family')}),perturbation_parameters=[r.get('perturbation_parameters') for _,r in parents],parent_candidate_id=[r.get('parent_candidate_id') for _,r in parents],preprocessing_hash=digest(cfg()['representation']),postprocessing_hash=digest(dict(dedup_decimals=6,heading_wrap=True)),score_status='PENDING',raw_index=len(rows),trajectory_path=str(OUT/'cache/raw'/f'{token}.npz')))
    return np.asarray(trajectories),rows

def build(scene):
    token=scene['token'];path=OUT/'cache/raw'/f'{token}.npz'
    if valid(path):return True
    sources=[OUT/'cache/rollouts/C'/f'{token}.npz']+[OUT/'cache/external'/s/f'{token}.npz' for s in ['ddv2','drivor']]
    if not all(p.exists() for p in sources):return False
    gt=np.asarray(scene['gt'],dtype=np.float64);items=[(gt,dict(source='GT',checkpoint_hash=None,generation_seed=None,parent_candidate_id=None))]+structured(gt,token)
    for p,source in zip(sources,['IL_native_C','ddv2','drivor']):
        z=np.load(p);meta=json.loads(str(z['metadata']));assert meta['token']==token;input_hash=sha(p)
        for i,t in enumerate(z['trajectories']):
            items.append((t,dict(source=source,parent_candidate_id=f'{source}:{token}:{i}',checkpoint_hash=meta.get('checkpoint_hash',meta.get('checkpoint_sha256')),generation_seed=meta.get('generation_seed') if source!='IL_native_C' else int(z['group_seeds'][i//8]),proposal_index=int(z['proposal_indices'][i]) if source!='IL_native_C' else i,input_cache_sha256=input_hash)))
    trajectories,records=deduplicate(items,token)
    # Trigger independent of scores, probabilities or method availability.
    extra=0
    while len(trajectories)<16 and extra<2:
        extra+=1;items+=structured(gt,token,extra);trajectories,records=deduplicate(items,token)
    meta=dict(token=token,input_hashes={str(p):sha(p) for p in sources},generation_count=len(items),unique_count=len(records),emergency_batches=extra,source_counts_pre_dedup={s:sum(r['source']==s for _,r in items) for s in ['GT','GT_structured','ddv2','drivor','IL_native_C']},A_B_used=False)
    save(path.with_suffix('.json'),dict(metadata=meta,candidates=records));npz(path,meta,trajectories=trajectories)
    return True
if __name__=='__main__':
    print('Built',sum(build(s) for s in scenes()),'of 1000')
