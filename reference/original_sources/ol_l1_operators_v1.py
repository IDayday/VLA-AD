"""L1 thirteen repair/centroid entries, with explicit new cluster role weights.

Array-only fit/predict building blocks, no real-data CLI or source discovery.
Negative controls are separate. Native request alpha and downward cap live in
ol_l1_geometry_v1 and the frozen native cap helper, never hidden in an MLP/OT.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
P2=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(P2/'code'),str(P2.parent/'code')]
import copy,hashlib,json
import numpy as np
from scipy.linalg import cholesky,cho_solve
import p2_u1_repair_functions_v1 as old
import p2_u1_paired_mapping_math_v1 as paired

MENU=old.REPAIRS+('paired_RRR','direction_amplitude','soft_clip','centroid_D0')
CONTROLS=('reverse','random_direction')
LAMBDA=.1
STD_FLOOR=1e-5


def support(method,nQ,nHigh,nReference):
    assert method in MENU
    reason=None
    if nQ<1:reason='NO_CLUSTER_QUALIFIED_PAIR'
    elif method in ('ot','local','soft_clip') and nQ<2:reason='UNBIASED_COVARIANCE_REQUIRES_TWO_PAIRS'
    elif method.endswith('_replay') and nHigh<1:reason='NO_FIT_HIGH_REFERENCE_REPLAY'
    elif method in ('paired_RRR','direction_amplitude') and nReference<1:reason='NO_FIT_GE70_REFERENCE'
    return dict(method=method,status='VALID' if reason is None else 'INVALID_'+reason,Base_fallback_required=reason is not None,
        cannot_borrow_other_cluster_or_held_rows=True)


def normalized_weights(parts):
    assert parts and all(n>0 for n in parts.values())
    return np.concatenate([np.full(n,1/(len(parts)*n),np.float64) for n in parts.values()])


def paired_rows(failed,successful,replay_high=None):
    """Both ordinary high replay and selected success remain separate roles."""
    data=old.paired_dataset(failed,successful,replay_high)
    roles={'paired_failure':len(failed),'selected_success':len(successful)}
    if replay_high is not None:roles['ordinary_Base_high']=len(replay_high)
    data['weights']=normalized_weights(roles);data['role_counts']=roles
    data['role_total_weights']={k:1/len(roles) for k in roles}
    return data


def standardize(x,w):
    assert x.dtype==w.dtype==np.float64 and x.ndim==2 and w.shape==(len(x),) and np.isfinite(x).all() and np.all(w>0) and abs(w.sum()-1)<1e-12
    mean=np.sum(w[:,None]*x,axis=0);scale=np.maximum(np.sqrt(np.sum(w[:,None]*np.square(x-mean),axis=0)),STD_FLOOR)
    return (x-mean)/scale,mean,scale


def ridge(x,y,w,standardize_targets):
    """Average weighted scene SUM-rank square loss + .1||W||², bias free."""
    phi,xmean,xscale=standardize(x,w)
    if standardize_targets:t,ymean,yscale=standardize(y,w)
    else:t=y;ymean=np.sum(w[:,None]*t,axis=0);yscale=np.ones(y.shape[1]);t=t-ymean
    design=np.c_[phi,np.ones(len(phi))];penalty=np.diag(np.r_[np.ones(phi.shape[1]),0.])
    C=design.T@(w[:,None]*design)+LAMBDA*penalty;rhs=design.T@(w[:,None]*t)
    W=cho_solve((cholesky(C,lower=True),True),rhs);err=float(np.linalg.norm(C@W-rhs)/max(np.linalg.norm(rhs),1e-300));assert err<1e-10
    return dict(schema='ol_l1_weighted_ridge_v1',input_mean=xmean,input_scale=xscale,target_mean=ymean,target_scale=yscale,
        weight=W[:-1],bias=W[-1],Gram=C,rhs=rhs,penalty=penalty,weights=w.copy(),lambda_average_loss=LAMBDA,bias_penalized=False,
        targets_standardized=standardize_targets,normal_equation_relative_error=err,
        objective='sum_scene(w*sum_output_squared_error)+.1*sum_weight_squared; bias unpenalized',
        training_raw_MSE=float(np.mean(np.square(predict_ridge(x,dict(input_mean=xmean,input_scale=xscale,target_mean=ymean,target_scale=yscale,weight=W[:-1],bias=W[-1]))-y))))


def predict_ridge(x,fit):return (((np.asarray(x,np.float64)-fit['input_mean'])/fit['input_scale'])@fit['weight']+fit['bias'])*fit['target_scale']+fit['target_mean']


def algebra(failed,successful,a,L,D0,projected_no_direction):
    assert failed.shape==successful.shape==a.shape and failed.ndim==2 and failed.dtype==successful.dtype==a.dtype==np.float64
    assert len(failed)>0 and D0>0 and projected_no_direction.shape==(len(failed),)
    mean=(successful-failed).mean(0);z=a@L;length=np.linalg.norm(z,axis=1);unit=np.zeros_like(z)
    active=~projected_no_direction;assert np.all(length[active]>0);unit[active]=z[active]/length[active,None]
    center=unit[active].mean(0) if active.any() else np.zeros(a.shape[1]);norm=float(np.linalg.norm(center))
    if norm>1e-8:centroid=np.linalg.solve(L.T,center/norm)*D0
    else:centroid=np.zeros(a.shape[1])
    common=dict(failure_mean=failed.mean(0),success_mean=successful.mean(0),mean_delta=mean,centroid_coefficients=centroid,
        centroid_Hbar_RMS=float(np.linalg.norm(centroid@L)),centroid_D0=D0,centroid_noop=norm<=1e-8,
        centroid_projected_pair_direction_mean_used_in_both_orders=True,fit_pairs=len(failed),covariance_valid=len(failed)>=2)
    if len(failed)>=2:common.update(old.fit_algebra(failed,successful))
    return common


def fit_original(method,failed,successful,common,replay_high=None):
    assert method in old.REPAIRS+('soft_clip','centroid_D0')
    if method in ('ot','local','soft_clip') and len(failed)<2:return dict(status='INVALID_UNBIASED_COVARIANCE_SUPPORT',method=method,Base_fallback_required=True)
    if method in ('linear_replay','mlp_replay') and (replay_high is None or not len(replay_high)):
        return dict(status='INVALID_MISSING_FIT_HIGH_REPLAY',method=method,Base_fallback_required=True)
    result=dict(status='VALID',method=method,common=common)
    if method.startswith('linear'):
        data=paired_rows(failed,successful,replay_high if method.endswith('replay') else None)
        result.update(learned=ridge(data['x'],data['y'],data['weights'],True),role_total_weights=data['role_total_weights'],row_roles=data['row_roles'])
    if method.startswith('mlp'):raise ValueError('MLP requires explicit MLPTrainingSession and approved 2000-step caller, not this analytic factory')
    return result


def predict_original(z,fit,direction=None):
    z=np.asarray(z,np.float64);one=z.ndim==1;x=z[None] if one else z
    if fit['status']!='VALID':return np.zeros_like(z)
    m=fit['method'];c=fit['common']
    if m=='centroid_D0':q=np.broadcast_to(c['centroid_coefficients'],x.shape).copy()
    elif m=='soft_clip':
        centered=x-c['success_mean'];q=np.clip(centered,-2*c['local_scale'],2*c['local_scale'])-centered
    elif m=='mean':q=np.broadcast_to(c['mean_delta'],x.shape).copy()
    elif m.startswith('linear'):q=predict_ridge(x,fit['learned'])
    else:q=old.predict_numpy(x,m,c,direction=direction)
    assert np.isfinite(q).all();return q[0] if one else q


def negative_control(z,method,common,seed=None):
    assert method in CONTROLS
    value=-common['mean_delta'] if method=='reverse' else old.random_direction(len(common['mean_delta']),seed)*np.linalg.norm(common['mean_delta'])
    return dict(coefficients=np.broadcast_to(value,np.asarray(z).shape).copy(),method=method,control=True,noncompetitive=True,
        coefficient_L2_only_not_actual_physical_dose_matched=True)


def paired_targets(B,delta,view,fractions,teacher_rms,D0):
    """Exactly the original ideal mask-WLS bounded target, not FP32 matched."""
    return paired.project_targets(B,delta,view,fractions,teacher_rms,D0)


def fit_paired(Q_v,Q_aux,reference_v,reference_aux,target,D0):
    """Full ordinary v+rawaux24; all fit >=70 references, no S->0 target."""
    assert len(Q_v)>0 and len(reference_v)>0 and Q_aux.shape==(len(Q_v),24) and reference_aux.shape==(len(reference_v),24)
    assert target['target_coefficients'].shape[0]==len(Q_v)
    x=np.r_[np.c_[Q_v,Q_aux],np.c_[reference_v,reference_aux]].astype(np.float64)
    rank=target['target_coefficients'].shape[1];T=np.zeros((len(x),2*rank+1),np.float64)
    T[:len(Q_v),:rank]=target['RRR_targets'];T[:len(Q_v),rank:2*rank]=target['DA_direction_targets'];T[:len(Q_v),-1]=target['DA_logamplitude_targets']
    w=normalized_weights(dict(Q=len(Q_v),ordinary_reference_ge70=len(reference_v)))
    return dict(status='VALID',schema='ol_l1_paired_mapping_v1',learned=ridge(x,T,w,False),rank=rank,D0=D0,
        combined_targets=T,role_total_weights=dict(Q=.5,ordinary_reference_ge70=.5),
        target_columns=['RRR[:r]','DA_direction[r:2r]','DA_logamplitude[-1]'],
        input='ordinary full-view v concatenated with raw aux24; no GT/pool/score online',no_selected_success_fixed_point_added=True)


def predict_paired(v,aux,fit,method,B,view,fractions):
    assert method in ('paired_RRR','direction_amplitude');raw=np.c_[np.atleast_2d(v),np.atleast_2d(aux)]
    predicted=predict_ridge(raw,fit['learned']);r=fit['rank']
    if method=='paired_RRR':return predicted[:,:r]*(fit['D0']*np.sqrt(384.))
    result,_=paired.da_decode(predicted[:,r:2*r],predicted[:,-1],B,view,np.asarray(fractions,np.float64),fit['D0'])
    return result


class MLPTrainingSession:
    """Explicit weighted CPU/GPU math API, no source reader or training CLI.

    Uniform replacement samples retain the old private batch generator. N*w
    makes each batch estimate the newly specified equal-role weighted loss.
    Targets use only fit D0, with no target centering/scaler.
    """
    def __init__(self,dataset,D0,initial_seed,device='cpu'):
        import torch
        assert D0>0 and 'weights' in dataset
        self.rank=dataset['x'].shape[1];self.D0=float(D0);self.initial_seed=int(initial_seed);self.device=device
        self.normalized,self.mean,self.scale=standardize(dataset['x'],dataset['weights'])
        self.X=torch.from_numpy(self.normalized.astype(np.float32)).to(device)
        self.Y=torch.from_numpy((dataset['y']/D0).astype(np.float32)).to(device)
        self.w=torch.from_numpy((len(dataset['x'])*dataset['weights']).astype(np.float32)).to(device)
        self.identity=hashlib.sha256(self.normalized.tobytes()+dataset['y'].tobytes()+dataset['weights'].tobytes()+str(D0).encode()).hexdigest()
        self.model=old.make_mlp(self.rank,initial_seed,device)
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=1e-3,weight_decay=.01)
        self.generator=torch.Generator(device='cpu').manual_seed(initial_seed+1);self.completed=0
        self.recipe=dict(hidden=max(64,min(256,2*self.rank)),activation='tanh',zero_last=True,lr=.001,AdamW_weight_decay=.01,clip1=True,
            batch128_uniform_private_generator=True,weighted_loss='mean_batch(N*w_i*mean_rank_squared_error)',target_scale='fit_D0_only',fixed_endpoint=2000)

    def step(self):
        import torch
        assert self.completed<2000
        ids=torch.randint(len(self.X),(128,),generator=self.generator,device='cpu').to(self.device)
        self.optimizer.zero_grad(set_to_none=True);loss=(torch.square(self.model(self.X[ids])-self.Y[ids]).mean(1)*self.w[ids]).mean()
        assert bool(torch.isfinite(loss));loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True)
        self.optimizer.step();self.completed+=1
        return dict(step=self.completed,loss=float(loss.detach()),gradient_norm_before_clip=float(norm),batch_indices=ids.cpu().numpy())

    def snapshot(self):
        return copy.deepcopy(dict(model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),batch_generator_state=self.generator.get_state(),
            step=self.completed,identity=self.identity,mean=self.mean,scale=self.scale,D0=self.D0,seed=self.initial_seed,recipe=self.recipe))

    def restore(self,saved):
        assert saved['identity']==self.identity and saved['D0']==self.D0 and saved['seed']==self.initial_seed and saved['recipe']==self.recipe
        assert np.array_equal(saved['mean'],self.mean) and np.array_equal(saved['scale'],self.scale)
        self.model.load_state_dict(copy.deepcopy(saved['model']));self.optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        self.generator.set_state(saved['batch_generator_state'].clone());self.completed=saved['step']

    def predict(self,z):
        """Portable ordinary single-scene FP32 neural math, FP64 decode later."""
        import torch
        x=np.atleast_2d(z).astype(np.float64);assert len(x)==1,'Online/precomputed path uses the same single-scene shape'
        with torch.no_grad():return self.model(torch.from_numpy(((x-self.mean)/self.scale).astype(np.float32)).to(self.device)).cpu().numpy().astype(np.float64)*self.D0
