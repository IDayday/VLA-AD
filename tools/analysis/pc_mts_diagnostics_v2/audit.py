"""Independent arithmetic, source provenance, completeness, and V1 integrity checks."""
import argparse,concurrent.futures
from v2_common import *
from compatibility import fit,position
from select_candidate_pools import pareto_ranks

def scene_audit(scene):
    token=scene['token'];ilpath=OUT/'il_banks'/f'{token}.npz';il=np.load(ilpath);raw=np.load(OUT/'raw_candidates'/f'{token}.npz');s=np.load(OUT/'evaluator/raw_candidates'/f'{token}.npz')['trajectories'];p=np.load(OUT/'positions'/f'{token}.npz');local=np.load(OUT/'evaluator/selection_local'/f'{token}.npz')['trajectories'];rank=pareto_ranks(s);cal=fit(il['R'],il['Q']);check=position(raw['trajectories'],cal)
    assert il['R'].shape==il['Q'].shape==(128,8,3) and il['native'].shape==(32,8,3);assert raw['trajectories'].shape==(192,8,3);assert s.shape==(192,7) and local.shape==(192,4,7)
    for key in check:np.testing.assert_allclose(check[key],p[key],rtol=1e-12,atol=1e-12)
    src=raw['source'];assert (src=='GT').sum()==32 and (src=='IL-native').sum()==32 and (src=='External-MTS').sum()==32 and (src=='External-RL').sum()==32
    for alpha in [.2,.4,.6,.8]:assert (src==f'Bridge-{alpha:g}').sum()==16
    np.testing.assert_array_equal(raw['trajectories'][32:64],il['native'])
    for j in range(128,192):
        alpha=float(raw['bridge_alpha'][j]);rho=il['R'][raw['reference_index'][j]];anchor=int(np.flatnonzero(raw['candidate_id']==raw['anchor_id'][j])[0]);external=raw['trajectories'][anchor,:,:2].astype(np.float32)
        # Both original endpoints are FP32; the mixed-source reservoir stores them as FP64.
        np.testing.assert_array_equal(raw['trajectories'][j,:,:2],(1-alpha)*rho[:,:2]+alpha*external)
    ref=float(np.load(V1/'evaluator/rollouts/official_il'/f'{token}.npz')['reference'][0,6]);fills=ood=unqualified=0;maxamp=0.
    for method in METHODS:
        a=np.load(OUT/'candidate_pools'/method/f'{token}.npz');assert a['trajectories'].shape==(16,8,3);assert a['scores'].shape==(16,7)
        existing=a['raw_index']>=0;np.testing.assert_array_equal(a['trajectories'][existing],raw['trajectories'][a['raw_index'][existing]]);np.testing.assert_array_equal(a['scores'][existing],s[a['raw_index'][existing]])
        if method=='score':np.testing.assert_array_equal(a['raw_index'],sorted(range(192),key=lambda i:(-s[i,6],i))[:16])
        if method=='pc_mts':
            fills=int(a['is_fill'].sum());ood=int((a['q_holdout']>=95).sum());unqualified=int((~a['qualified']).sum())
            for j in np.flatnonzero(existing&a['qualified']):
                k=a['raw_index'][j];assert feasible(s[k:k+1])[0] and 1<=rank[k]<=2 and s[k,6]>=ref-.01-1e-10 and a['q_holdout'][j]<95
                if a['fallback_level'][j]==1:assert 50<=a['q_holdout'][j]<95 and feasible(local[k]).mean()>=.75
            for j in np.flatnonzero(a['is_fill']):
                parent=int(np.flatnonzero(a['candidate_id']==a['fill_parent_id'][j])[0]);assert np.linalg.norm(a['trajectories'][j,:,:2]-a['trajectories'][parent,:,:2],axis=-1).max()<=.01+1e-9
        h=np.load(OUT/'heldout_seeded'/method/f'{token}.npz')['trajectories'];hs=np.load(OUT/'evaluator/heldout_seeded'/method/f'{token}.npz')['trajectories'];assert h.shape==(16,24,8,3) and hs.shape==(16,24,7) and np.isfinite(hs).all()
        amplitudes=np.linalg.norm(h[...,:2]-a['trajectories'][:,None,:,:2],axis=-1).max(-1);np.testing.assert_allclose(amplitudes,np.broadcast_to(np.repeat([.05,.15,.3,.5],6),(16,24)),atol=1e-8,rtol=0);maxamp=max(maxamp,float(amplitudes.max()))
    if token in set(subset('denoising')):
        dn=np.load(OUT/'denoising'/f'{token}.npz');assert dn['candidate_errors'].shape==(4,16,6) and dn['Q_errors'].shape==(128,6);assert float(dn['epsilon'])==float(np.quantile(dn['Q_errors'],.95));np.testing.assert_array_equal(dn['return_rate'],(dn['candidate_errors']<dn['epsilon']).mean(-1))
    return dict(token=token,pc_fills=fills,pc_OOD=ood,pc_unqualified=unqualified,max_amplitude=maxamp)

def pooled_calibration():
    old_self=[];old_candidates=[];q=[];new=[];native=[];old_rates=[]
    for scene in scenes()['scenes']:
        token=scene['token'];il=np.load(V1/'rollouts/official_il'/f'{token}.npz')['trajectories'];d=distance(il,il);np.fill_diagonal(d,np.inf);ss=np.sort(d,axis=1)[:,:5].mean(1);old_self.append(ss);raw=np.load(V1/'raw_candidates'/f'{token}.npz')['trajectories'];dd=np.sort(distance(raw,il),axis=1)[:,:5].mean(1);old_candidates.append(dd);old_rates.append(float((dd>ss.max()).mean()));p=np.load(OUT/'positions'/f'{token}.npz');q.append(p['query_reference_distances']);new.append(p['policy_distance_knn']);native.append(p['policy_distance_knn'][32:64])
    vals={label:np.concatenate(arrays) for label,arrays in [('V1_IL_self',old_self),('V1_raw_candidates',old_candidates),('V2_IL_Q_to_R',q),('V2_raw_candidates',new),('V2_IL_native_to_R',native)]}
    stats={k:dict(n=len(v),mean=float(v.mean()),**{f'p{j}':float(np.percentile(v,j)) for j in [0,5,25,50,75,95,99,100]}) for k,v in vals.items()};save(OUT/'metrics/calibration_pooled_quantiles.json',dict(distributions=stats,V1_raw_fraction_above_scene_self_max=float(np.mean(old_rates)),units='ADE metres',note='Pooled across scenes; calibration_quantiles_full.csv separately reports averages of within-scene quantiles.'))

def main(a):
    if a.calibration:pooled_calibration();return
    if a.funnel:
        import pandas as pd
        with concurrent.futures.ProcessPoolExecutor(max_workers=96) as ex:d=pd.DataFrame(list(ex.map(funnel_scene,scenes()['scenes'])))
        d.to_csv(OUT/'metrics/pc_eligibility_funnel_v2.csv',index=False);pd.DataFrame([dict(gate=k,mean_candidates=d[k].mean(),zero_scenes=int((d[k]==0).sum()),at_least16_scenes=int((d[k]>=16).sum())) for k in d if k!='token']).to_csv(OUT/'metrics/pc_eligibility_funnel_v2_summary.csv',index=False);return
    with concurrent.futures.ProcessPoolExecutor(max_workers=96) as ex:rows=list(ex.map(scene_audit,scenes()['scenes'],chunksize=1))
    assert len(rows)==1000
    save(OUT/'manifests/completion_audit.json',dict(identity=ident(),rows=rows,full_scenes=len(rows),raw_count=192000,selected_count=64000,heldout_count=1536000,new_official_IL_count=288000,denoising_scene_count=len(subset('denoising')),status='All scene-level array, native score, selection, bridge, calibration, amplitude, and epsilon checks passed'))

def funnel_scene(scene):
    token=scene['token'];s=np.load(OUT/'evaluator/raw_candidates'/f'{token}.npz')['trajectories'];q=np.load(OUT/'positions'/f'{token}.npz')['q_holdout'];local=feasible(np.load(OUT/'evaluator/selection_local'/f'{token}.npz')['trajectories']).mean(1);r=pareto_ranks(s);ref=np.load(V1/'evaluator/rollouts/official_il'/f'{token}.npz')['reference'][0,6];b=(q>=50)&(q<95);safe=feasible(s);front=(r>=1)&(r<=2);quality=s[:,6]>=ref-.01-1e-10
    return dict(token=token,boundary=int(b.sum()),boundary_feasible=int((b&safe).sum()),boundary_feasible_quality=int((b&safe&quality).sum()),boundary_feasible_front=int((b&safe&front).sum()),boundary_feasible_front_quality=int((b&safe&front&quality).sum()),primary=int((b&safe&front&quality&(local>=.75)).sum()),core_all_quality=int(((q<50)&safe&front&quality&(local>=.5)).sum()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--calibration',action='store_true');p.add_argument('--funnel',action='store_true');main(p.parse_args())
