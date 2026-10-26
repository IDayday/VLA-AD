"""L1 fold-local geometry and signed direction clustering; arrays in/out only.

No asset reader, labels, PDMS, native model, fitting CLI or queue. A caller
must supply its own session-isolated qualified training arrays and references.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
P2=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(P2/'code'),str(P2.parent/'code')]
import hashlib,json
import numpy as np
from scipy.linalg import cholesky,solve_triangular,svd
import p2_u1_geometry_math_v1 as old_geometry
import p2_u1_direction_experts_math_v2 as old_directions
import p2_u1_paired_mapping_math_v1 as old_paired

RANKS=(32,48,64)
K_MENU=tuple(range(1,9))
SEED=2026102531
RANDOM_SEEDS=(2026102532,2026102533,2026102534)
R_RECIPES={
 ('valid_shared384',32):dict(centered=False,weighted=True,ks=2,kf=32),
 ('valid_shared384',48):dict(centered=True,weighted=True,ks=2,kf=192),
 ('valid_shared384',64):dict(centered=True,weighted=True,ks=2,kf=96),
 ('role2_768',32):dict(centered=True,weighted=False,ks=8,kf=32),
 ('role2_768',48):dict(centered=True,weighted=True,ks=4,kf=64),
 ('role2_768',64):dict(centered=True,weighted=True,ks=4,kf=384)}


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def seeded(purpose,*fields,base=SEED):
    return int(hashlib.sha256(('OL_L1|'+str(base)+'|'+purpose+'|'+json.dumps(fields,separators=(',',':'),sort_keys=True)).encode()).hexdigest()[:16],16)
def priority(tokens,seed):
    assert len(tokens)==len(set(tokens))
    return np.asarray(sorted(range(len(tokens)),key=lambda i:(hashlib.sha256((str(seed)+'|'+tokens[i]).encode()).hexdigest(),tokens[i])),np.int64)


def decomposition(x,trace):
    try:
        u,s,v=np.linalg.svd(x,full_matrices=False);trace.append(dict(driver='numpy_svd',shape=list(x.shape),fallback=False))
    except np.linalg.LinAlgError as error:
        u,s,v=svd(x,full_matrices=False,lapack_driver='gesvd');trace.append(dict(driver='scipy_gesvd',shape=list(x.shape),fallback=True,original_error=repr(error)))
    return u,s,v


def fit_space(f,s,actual_rms,route,view,rank):
    """Re-fit one fixed R/D recipe, never import an old physical basis.

    Input rows must already be this fit's Q only. Vector construction/f0 uses
    old_geometry.view_vectors with the same explicit fit rows, before here.
    """
    assert f.dtype==s.dtype==actual_rms.dtype==np.float64 and f.shape==s.shape and f.ndim==2
    assert len(f)>0 and np.isfinite(f).all() and np.isfinite(s).all() and np.isfinite(actual_rms).all() and np.all(actual_rms>=0)
    assert rank in RANKS and f.shape[1]==(384 if view=='valid_shared384' else 768)
    trace=[];info=dict(route=route,view=view,rank=rank,fit_scenes=len(f),fit_failure_SHA=digest(f),fit_success_SHA=digest(s),actual_RMS_SHA=digest(actual_rms))
    if route=='D':
        floor=max(1e-12,1e-3*float(np.median(actual_rms)));x=(s-f)/np.maximum(actual_rms,floor)[:,None]
        _,sigma,vt=decomposition(x,trace);top=float(sigma[0]) if len(sigma) else 0.
        effective=int(np.count_nonzero(sigma>top*1e-10)) if top else 0
        B=old_geometry.signs(vt[:effective].T);center=np.zeros(f.shape[1],np.float64)
        info.update(normalized=True,centered=False,normalization_floor=floor,effective_rank=effective,spectrum=sigma,relative_singular_cutoff=1e-10)
    else:
        assert route=='R';recipe=R_RECIPES[view,rank];center=(f.mean(0)+s.mean(0))/2 if recipe['centered'] else np.zeros(f.shape[1])
        classes={}
        for name,value in [('failure',f-center),('success',s-center)]:
            _,sig,vt=decomposition(value,trace);eig=sig**2/len(value);eff=int(np.count_nonzero(eig>eig[0]*1e-10)) if len(eig) and eig[0]>0 else 0
            classes[name]=dict(basis=old_geometry.signs(vt[:eff].T),eigenvalues=eig,effective_rank=eff)
        if classes['failure']['effective_rank']<recipe['kf'] or classes['success']['effective_rank']<recipe['ks']:
            return dict(status='INVALID_CLASS_EFFECTIVE_RANK',basis=np.empty((f.shape[1],0)),center=center,recipe=recipe,classes=classes,trace=trace,**info)
        us=classes['success']['basis'][:,:recipe['ks']];uf=classes['failure']['basis'][:,:recipe['kf']]
        residual=uf-us@(us.T@uf)
        if recipe['weighted']:residual*=np.sqrt(classes['failure']['eigenvalues'][:recipe['kf']]/classes['failure']['eigenvalues'][0])[None]
        u,sigma,_=decomposition(residual,trace);effective=int(np.count_nonzero(sigma>1e-6));B=old_geometry.signs(u[:,:effective])
        info.update(recipe=recipe,classes=classes,effective_rank=effective,spectrum=sigma,residual_singular_cutoff=1e-6,failure_basis_residualized_against_success=True)
    if effective<rank:return dict(status='INVALID_REQUESTED_RANK_NOT_CLAMPED',basis=B,center=center,trace=trace,**info)
    B=B[:,:rank];assert np.linalg.norm(B.T@B-np.eye(rank))<1e-8
    return dict(status='VALID_SPACE',basis=B,center=center,trace=trace,**info)


def geometry(B,view,fractions,delta,teacher_rms):
    """Compute actual-mask WLS and BOTH order representations in one metric."""
    assert B.dtype==delta.dtype==fractions.dtype==teacher_rms.dtype==np.float64
    assert fractions.shape==teacher_rms.shape==(len(delta),) and np.all((fractions>0)&(fractions<1))
    assert np.all(teacher_rms>=0) and len(delta)>0
    a,rho,rhs,error=old_directions.project(B,delta,view,fractions)
    if view['view']=='valid_shared384':
        diagonal=np.full(B.shape[0],1/384.,np.float64)
    else:
        diagonal=np.r_[np.full(384,float(np.mean(fractions/view['f0']))/384.),np.full(384,float(np.mean((1-fractions)/(1-view['f0'])))/384.)]
    H=B.T@(diagonal[:,None]*B);L=cholesky(H,lower=True)
    epsilon=np.maximum(1e-8,1e-5*teacher_rms)
    raw_P=a@L;raw_C=delta*np.sqrt(diagonal)[None]
    if view['view']=='valid_shared384':C_rms=np.sqrt(np.square(delta).mean(1))
    else:C_rms=np.sqrt(np.square(delta[:,:384]).mean(1)*fractions/view['f0']+np.square(delta[:,384:]).mean(1)*(1-fractions)/(1-view['f0']))
    outputs={}
    for order,z,physical_rho in [('P_to_C',raw_P,rho),('C_to_P',raw_C,C_rms)]:
        nodir=physical_rho<=epsilon;norm=np.linalg.norm(z,axis=1);u=np.zeros_like(z)
        assert np.all(norm[~nodir]>0);u[~nodir]=z[~nodir]/norm[~nodir,None]
        outputs[order]=dict(unit_directions=u,no_direction=nodir,physical_RMS=physical_rho,metric_norm=norm)
    return dict(basis=B.copy(),view=dict(view),fractions=fractions.copy(),delta=delta.copy(),coefficients=a,projected_RMS=rho,wls_rhs=rhs,wls_relative_error=error,
        M_diagonal=diagonal,Hbar=H,L=L,epsilon=epsilon,teacher_rms=teacher_rms.copy(),orders=outputs,
        Hbar_reconstruction_error=float(np.max(np.abs(H-L@L.T))),no_covariance_whitening=True,all_Q_including_NO_DIRECTION_in_M=True)


def _initial(u,eligible,tokens,seed,K,initial_row=None):
    """Full-token hash priority, shared seed. Never intersect P/C supports."""
    order=priority(tokens,seed);available=np.flatnonzero(eligible)
    if len(available)<K or len(np.unique(u[available],axis=0))<K:return None,dict(status='INVALID_UNIQUE_ELIGIBLE_DIRECTIONS',K=K,eligible=int(eligible.sum()))
    if initial_row is None:first=next(int(i) for i in order if eligible[i])
    else:assert eligible[initial_row];first=int(initial_row)
    chosen=[first];rng=np.random.default_rng(seed);fallback=[]
    while len(chosen)<K:
        cost=np.maximum(1-(u@u[chosen].T).max(1),0.);cost[~eligible]=0.;cost[chosen]=0.;total=float(cost.sum())
        if total>0:j=int(rng.choice(len(u),p=cost/total))
        else:j=next(int(i) for i in order if eligible[i] and i not in chosen);fallback.append(j)
        assert eligible[j] and j not in chosen;chosen.append(j)
    return np.asarray(chosen,np.int64),dict(status='VALID',seed=int(seed),full_token_priority_rows=order,rows=chosen,tokens=[tokens[i] for i in chosen],
        first_eligible_token=tokens[first],fallback_rows=fallback,initial_row_fixture_override=initial_row is not None,first_choice_rule='first own eligible in shared complete-token hash priority; never common-support intersection')


def assign_zero_rows(labels,no_direction,tokens,K):
    labels=labels.copy();counts=np.bincount(labels[~no_direction],minlength=K)
    for i in priority(tokens,seeded('NO_DIRECTION_assignment')):
        if not no_direction[i]:continue
        choices=np.flatnonzero(counts==counts.min())
        k=min(choices,key=lambda k:hashlib.sha256((tokens[i]+'|'+str(int(k))).encode()).hexdigest());labels[i]=k;counts[k]+=1
    return labels


def cluster(g,order,K,tokens,sessions,fit_identity,restarts=20,initial_row=None):
    assert order in ('P_to_C','C_to_P') and K in K_MENU and restarts==20 and len(tokens)==len(g['coefficients'])==len(sessions)
    values=g['orders'][order];u=values['unit_directions'];nodir=values['no_direction'];available=np.flatnonzero(~nodir)
    assert len(tokens)==len(set(tokens));trials=[]
    for restart in range(1 if K==1 else restarts):
        sd=seeded('CLUSTER',fit_identity,g['view']['view'],K,restart)
        if K==1 and len(available):
            mean=u[available].mean(0);length=float(np.linalg.norm(mean));center=np.zeros((1,u.shape[1])) if length<=1e-8 else mean[None]/length
            fit=dict(centers=center,labels=np.zeros(len(available),np.int64),no_op=np.array([length<=1e-8]),objective=float((1-u[available]@center[0]).sum()),iterations=0,trace=[],resets=[])
            initial=dict(seed=sd,rows=[],tokens=[],K1_all_eligible_direction_mean=True)
        else:
            chosen,initial=_initial(u,~nodir,tokens,sd,K,initial_row)
            if chosen is None:
                labels=assign_zero_rows(np.full(len(tokens),-1,np.int64),np.ones(len(tokens),bool),tokens,K)
                return dict(status='INVALID_DIRECTION_CLUSTER_NO_K_REDUCTION',K=K,labels=labels,centers=np.zeros((K,u.shape[1])),no_op=np.ones(K,bool),initialization=initial,
                    no_direction=nodir,all_scene_pairs_preserved=True,all_operator_slots_require_explicit_invalid_Base=True)
            local={int(j):i for i,j in enumerate(available)};ii=np.array([local[int(j)] for j in chosen],np.int64)
            fit=old_directions.lloyd(u[available],np.zeros(len(available)),K,ii,[tokens[i] for i in available],0.,max_iter=300)
        trials.append(dict(restart=restart,initialization=initial,**fit))
    winner=min(range(len(trials)),key=lambda i:(trials[i]['objective'],i));best=trials[winner]
    labels=np.full(len(tokens),-1,np.int64);labels[available]=best['labels'];labels=assign_zero_rows(labels,nodir,tokens,K)
    statistics=[]
    for k in range(K):
        rows=np.flatnonzero(labels==k);ss=[sessions[i] for i in rows];unique=sorted(set(ss));nz=rows[~nodir[rows]]
        statistics.append(dict(cluster=k,n=len(rows),sessions=len(unique),maximum_session_fraction=max((ss.count(s)/len(rows) for s in unique),default=0.),
            support_weak=len(rows)<20 or len(unique)<4,no_direction=int(nodir[rows].sum()),nonzero_directions=len(nz),
            angular_cost=float((1-u[nz]@best['centers'][k]).mean()) if len(nz) else None,teacher_RMS_median=float(np.median(g['teacher_rms'][rows])) if len(rows) else None,
            weak_support_not_PDMS_veto=True))
    key=dict(fit_identity=fit_identity,tokens_SHA=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),sessions=sorted(set(sessions)),view=g['view'],
        full_delta_SHA=digest(g['delta']),M_SHA=digest(g['M_diagonal']),epsilon_SHA=digest(g['epsilon']),no_direction_SHA=digest(nodir),K=K,seed=SEED,
        source_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),lloyd_source_SHA=hashlib.sha256(Path(old_directions.__file__).read_bytes()).hexdigest())
    if order=='P_to_C':key['basis_SHA']=digest(g['basis'])
    return dict(status='VALID_DIRECTION_CLUSTER',order=order,K=K,labels=labels,centers=best['centers'],no_op=best['no_op'],no_direction=nodir,
        selected_restart=winner,objective=best['objective'],trials=trials,statistics=statistics,cache_identity=key,
        cache_key=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest(),assignment_uses_training_only=True,
        zero_pair_targets_retained=True,reset_rounds_cannot_converge=True,final_noop_membership_not_reassigned=True)


def random_partition(labels,tokens,fit_identity,seed=RANDOM_SEEDS[0],K=None):
    assert seed in RANDOM_SEEDS and labels.shape==(len(tokens),) and np.all(labels>=0)
    K=int(labels.max())+1 if K is None else int(K);assert K>int(labels.max())
    sizes=np.bincount(labels,minlength=K);sd=seeded('SIZE_MATCHED_RANDOM',fit_identity,K,base=seed)
    permutation=np.random.default_rng(sd).permutation(len(labels));out=np.empty_like(labels);offset=0
    for k,n in enumerate(sizes):out[permutation[offset:offset+n]]=k;offset+=n
    assert np.array_equal(np.bincount(out,minlength=K),sizes)
    return dict(labels=out,sizes=sizes,permutation=permutation,seed=int(sd),base_seed=seed,control=True,noncompetitive=True,
        complete_pairs_including_NO_DIRECTION_permuted=True,same_13_methods_dose_and_selection_budget_required=True)


def decode_roles(q,B,view):
    v=np.asarray(q,np.float64)@B.T
    if view['view']=='valid_shared384':return np.stack([v,v],axis=-2)
    return v.reshape(*v.shape[:-1],2,384)/np.sqrt(np.array([view['f0'],1-view['f0']]))[:,None]


def request(q,B,view,D0,alpha,cap_multiple):
    assert alpha in (.25,.5,.75,1.,1.5,2.) and cap_multiple in (1.,2.) and D0>0
    assert cap_multiple==2. or alpha==.5,'Only the one .5/D0 baseline accompanies the six 2D0 main doses'
    q=np.atleast_2d(np.asarray(q,np.float64));assert np.isfinite(q).all()
    raw=decode_roles(q,B,view)*alpha
    return dict(requested_shifts_float64=raw,requested_shifts_float32=raw.astype(np.float32),cap_RMS_float64=np.full(len(q),cap_multiple*D0,np.float64),
        alpha=alpha,train_D0=D0,cap_multiple=cap_multiple,raw_prediction_not_normalized_to_D0=True,actual_FP32_cap_required=True)
