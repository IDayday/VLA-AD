"""Paired parent-cluster bootstrap with training/inference-seed structure."""
import numpy as np


def summarize_delta(old,new,high_mask=None,zero=1e-8,drop=.02):
    # old/new: training seed x scene x inference seed x metric. Old may be
    # broadcast from one fixed baseline; it is not extra training replication.
    delta=new.mean(2)-old.mean(2)
    pdms=delta[:,:,-1]
    result=dict(G=float(pdms.mean()),M_plus=float(np.maximum(pdms,0).mean()),
                M_minus=float(np.maximum(-pdms,0).mean()),material_drop_fraction=float((pdms < -drop).mean()),
                single_output_drop_probability=float(((new[:,:,:,-1]-old[:,:,:,-1]) < -drop).mean()),
                zero_output_fraction=float((new[:,:,:,-1]<=zero).mean()),
                zero_scene_mean_fraction=float((new.mean(2)[:,:,-1]<=zero).mean()),
                positive_to_zero_fraction=float(((old.mean(2)[:,:,-1]>zero)&(new.mean(2)[:,:,-1]<=zero)).mean()),
                zero_to_positive_fraction=float(((old.mean(2)[:,:,-1]<=zero)&(new.mean(2)[:,:,-1]>zero)).mean()),
                PDMS=float(new[:,:,:,-1].mean()),median_scene_PDMS=float(np.median(new.mean(2)[:,:,-1])))
    for index,name in enumerate(['NC','DAC','EP','TTC','Comfort','DDC','PDMS']):
        result['delta_'+name]=float(delta[:,:,index].mean())
    if high_mask is not None and np.any(high_mask):
        high_delta=pdms[:,high_mask]
        result['high_M_minus']=float(np.maximum(-high_delta,0).mean())
        result['high_to_zero_fraction']=float((new[:,high_mask,:,-1].mean(2)<=zero).mean())
        result['high_to_near_zero_fraction']=float((new[:,high_mask,:,-1].mean(2)<=.01).mean())
    return result


def paired_bootstrap(old,new,parents,*,high_mask=None,repetitions=10000,seed=20261020):
    old,new=np.asarray(old,dtype=np.float64),np.asarray(new,dtype=np.float64)
    if old.ndim==3:old=np.broadcast_to(old,new.shape)
    if new.ndim!=4 or old.shape!=new.shape or new.shape[-1]!=7:raise ValueError('Paired seed/scene/noise/metric axes required')
    parents=np.asarray(parents)
    if len(parents)!=new.shape[1]:raise ValueError('Scene parent metadata mismatch')
    unique=np.unique(parents);groups=[np.flatnonzero(parents==p) for p in unique]
    rng=np.random.default_rng(seed);point=summarize_delta(old,new,high_mask)
    draws={key:np.empty(repetitions) for key in point}
    for b in range(repetitions):
        train_indices=rng.integers(new.shape[0],size=new.shape[0])
        scene_indices=np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))])
        # Inference seeds remain paired across every policy in a scene. Draw
        # one shared seed set per training replicate, preserving scene blocks.
        sampled_old=[];sampled_new=[]
        for t in train_indices:
            z=rng.integers(new.shape[2],size=new.shape[2])
            sampled_old.append(old[t][scene_indices][:,z]);sampled_new.append(new[t][scene_indices][:,z])
        high=None if high_mask is None else np.asarray(high_mask)[scene_indices]
        result=summarize_delta(np.stack(sampled_old),np.stack(sampled_new),high)
        for key in point:draws[key][b]=result.get(key,np.nan)
    intervals={key:np.nanquantile(value,[.025,.975]).tolist() for key,value in draws.items()}
    centered=draws['G']-point['G']
    p_positive=float((1+np.count_nonzero(centered>=point['G']))/(repetitions+1))
    return dict(point=point,CI95=intervals,parent_count=len(unique),training_seeds=new.shape[0],
                inference_seeds=new.shape[2],bootstrap_repetitions=repetitions,
                one_sided_centered_bootstrap_p=p_positive,
                scope='Parent-block bootstrap; few parents/training seeds and rare failures may remain inconclusive')


def parent_mean_bootstrap(values,parents,repetitions=10000,seed=20261020):
    values=np.asarray(values,dtype=np.float64);parents=np.asarray(parents)
    groups=[np.flatnonzero(parents==p) for p in np.unique(parents)]
    sums=np.asarray([values[g].sum(0) for g in groups]);counts=np.asarray([len(g) for g in groups])
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(repetitions):
        indices=rng.integers(len(groups),size=len(groups))
        draws.append(sums[indices].sum(0)/counts[indices].sum())
    return dict(mean=values.mean(0).tolist(),CI95=np.quantile(draws,[.025,.975],axis=0).tolist(),parents=len(groups),scenes=len(values))


def holm_adjust(pvalues):
    order=sorted(pvalues,key=pvalues.get);n=len(order);result={};previous=0.
    for j,key in enumerate(order):
        previous=max(previous,min(1.,(n-j)*pvalues[key]));result[key]=previous
    return result


def paired_linear_bootstrap(old,new,parents,repetitions=10000,seed=20261020):
    """Exact sufficient statistics for ONE policy's linear metric differences.

    Used only for the registered harm boundary; nonlinear M+/M- still use the
    full hierarchy above. Parent counts and paired global noise IDs are sampled
    independently. No scene independence or artificial training replication.
    """
    old,new=np.asarray(old,dtype=np.float64),np.asarray(new,dtype=np.float64)
    if old.shape!=new.shape or old.ndim!=3 or old.shape[-1]!=7:raise ValueError('One paired policy x scene/noise/metric comparison required')
    parents=np.asarray(parents)
    if len(parents)!=len(old):raise ValueError('Parent metadata mismatch')
    difference=new-old;groups=[np.flatnonzero(parents==v) for v in np.unique(parents)]
    sums=np.stack([difference[g].sum(0) for g in groups]);counts=np.array([len(g) for g in groups])
    rng=np.random.default_rng(seed)
    pw=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=repetitions)
    zw=rng.multinomial(old.shape[1],np.ones(old.shape[1])/old.shape[1],size=repetitions)/old.shape[1]
    draws=np.einsum('rg,gzm,rz->rm',pw,sums,zw,optimize=True)/(pw@counts)[:,None]
    mean=difference.mean((0,1));bounds=np.quantile(draws,[.025,.975],axis=0)
    labels=['NC','DAC','EP','TTC','Comfort','DDC','PDMS']
    point={'delta_'+name:float(mean[j]) for j,name in enumerate(labels)}
    ci={'delta_'+name:bounds[:,j].tolist() for j,name in enumerate(labels)}
    point['G']=point['delta_PDMS'];ci['G']=ci['delta_PDMS']
    return dict(point=point,CI95=ci,parent_count=len(groups),training_seeds=1,inference_seeds=old.shape[1],
                bootstrap_repetitions=repetitions,scope='Linear harm-boundary metrics only; whole-parent blocks and paired inference-seed hierarchy')


def local_noise_bootstrap(noise_values,parents,repetitions=10000,seed=20261020):
    """Parent blocks + independent candidate banks + paired suffix-replicate IDs.

    noise_values is [scene, bank, test_replicate, comparison], already averaged
    over the five execution indices. Shared resampling weights keep comparisons
    paired. This is conditional on the fixed roll-in seed, which is disclosed.
    """
    value=np.asarray(noise_values,dtype=np.float64);parents=np.asarray(parents)
    if value.ndim!=4 or len(parents)!=len(value):raise ValueError('Local hierarchical axes missing')
    unique,inverse=np.unique(parents,return_inverse=True)
    rng=np.random.default_rng(seed);draws=[]
    for start in range(0,repetitions,128):
        count=min(128,repetitions-start)
        bw=rng.multinomial(value.shape[1],np.ones(value.shape[1])/value.shape[1],size=count)/value.shape[1]
        zw=rng.multinomial(value.shape[2],np.ones(value.shape[2])/value.shape[2],size=count)/value.shape[2]
        pw=rng.multinomial(len(unique),np.ones(len(unique))/len(unique),size=count)[:,inverse]
        scene=np.einsum('sblm,rb,rl->rsm',value,bw,zw,optimize=True)
        draws.append(np.einsum('rsm,rs->rm',scene,pw)/pw.sum(1)[:,None])
    return dict(mean=value.mean((0,1,2)).tolist(),CI95=np.quantile(np.concatenate(draws),[.025,.975],axis=0).tolist(),
                parents=len(unique),scenes=len(value),candidate_banks=value.shape[1],suffix_replicates=value.shape[2],
                conditional_on_fixed_rollin_seed=True)
