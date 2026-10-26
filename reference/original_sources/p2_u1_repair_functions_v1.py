"""U1 original9 repair functions +2 controls, explicit training objectives.

Pure numpy math and lazy torch factories. There is no real-data fit CLI.
All inputs to online functions are ordinary Base coordinates z only.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import numpy as np

REPAIRS=('mean','contract','boundary','ot','local','linear','linear_replay','mlp','mlp_replay')
CONTROLS=('reverse','random_direction')
STD_FLOOR=1e-5


def covariance(x):
    assert x.ndim==2 and len(x)>1 and np.isfinite(x).all()
    z=x-x.mean(0)
    return z.T@z/(len(x)-1) # old mapping uses np.cov, ddof1


def regular(c):
    c=(c+c.T)/2;r=len(c)
    return .95*c+.05*np.eye(r)*max(float(np.trace(c)/r),1e-6)


def matrix_power(c,power):
    eigen,basis=np.linalg.eigh((c+c.T)/2)
    return (basis*np.maximum(eigen,1e-8)**power)@basis.T


def paired_dataset(failed,successful,replay_high=None):
    assert failed.dtype==successful.dtype==np.float64 and failed.shape==successful.shape and failed.ndim==2
    parts=[failed,successful];targets=[successful-failed,np.zeros_like(successful)]
    labels=['paired_failure_delta']*len(failed)+['paired_selected_success_zero']*len(successful)
    if replay_high is not None:
        assert replay_high.ndim==2 and replay_high.shape[1]==failed.shape[1] and replay_high.dtype==np.float64
        parts.append(replay_high);targets.append(np.zeros_like(replay_high));labels+=['ordinary_Base_gt90_zero']*len(replay_high)
    return dict(x=np.concatenate(parts),y=np.concatenate(targets),row_roles=labels,
        counts=dict(paired_failure=len(failed),selected_success=len(successful),ordinary_high=len(replay_high) if replay_high is not None else 0),
        replay_adds_ordinary_reference_not_replaces_selected_success=True)


def standardize(x):
    mean=x.mean(0);scale=np.maximum(x.std(0,ddof=0),STD_FLOOR)
    return (x-mean)/scale,mean,scale


def fit_algebra(failed,successful):
    assert failed.dtype==successful.dtype==np.float64 and failed.shape==successful.shape
    cf=covariance(failed);cs=covariance(successful);rf=regular(cf);rs=regular(cs)
    half=matrix_power(rf,.5);inverse=matrix_power(rf,-.5)
    transport=inverse@matrix_power(half@rs@half,.5)@inverse
    return dict(schema='u1_repair_algebra_v1',rank=failed.shape[1],failure_mean=failed.mean(0),success_mean=successful.mean(0),
        failure_covariance=cf,success_covariance=cs,failure_covariance_regularized=rf,success_covariance_regularized=rs,
        ot=transport,local_scale=np.sqrt(np.maximum(np.diag(cs),1e-6)),local_failed=failed.copy(),local_delta=successful-failed,
        covariance_ddof=1,OT_shrinkage=.05,OT_eigenvalue_floor=1e-8,local_neighbors=16,
        OT_regularized_covariance_mapping_relative_error=float(np.linalg.norm(transport@rf@transport.T-rs)/max(np.linalg.norm(rs),1e-300)))


def fit_linear(dataset,ridge_lambda=.1):
    assert ridge_lambda==.1,'Additional sensitivity fits need their own registered specification.'
    x,y=dataset['x'],dataset['y'];assert x.shape==y.shape and x.dtype==y.dtype==np.float64
    xx,xmean,xscale=standardize(x);yy,ymean,yscale=standardize(y)
    design=np.c_[xx,np.ones(len(xx))];penalty=np.diag(np.r_[np.ones(xx.shape[1]),0.])
    gram=design.T@design+ridge_lambda*len(xx)*penalty;rhs=design.T@yy
    solution=np.linalg.solve(gram,rhs)
    predicted=design@solution
    residual=float(np.linalg.norm(gram@solution-rhs)/max(np.linalg.norm(rhs),1e-300))
    assert residual<1e-10 and np.isfinite(solution).all()
    objective=float(np.square(predicted-yy).sum()/len(xx)+ridge_lambda*np.square(solution[:-1]).sum())
    return dict(schema='u1_repair_linear_v1',rank=x.shape[1],input_mean=xmean,input_scale=xscale,target_mean=ymean,target_scale=yscale,
        weight=solution[:-1],bias=solution[-1],ridge_lambda=ridge_lambda,bias_penalized=False,
        normal_equation_penalty_diagonal=np.diag(penalty),
        objective='mean_over_scenes(sum_over_rank((normalized_prediction-normalized_target)^2)) + lambda*||W||F^2',
        normal_equation_relative_residual=residual,training_objective=objective,
        training_raw_MSE=float(np.square(predicted*yscale+ymean-y).mean()),row_counts=dataset['counts'],
        normalization='Population input and target statistics from this fold training joint F/S/(ordinary high replay) only, std floor1e-5')


def linear_predict(z,fit):
    return (((z-fit['input_mean'])/fit['input_scale'])@fit['weight']+fit['bias'])*fit['target_scale']+fit['target_mean']


def random_direction(rank,seed):
    value=np.random.default_rng(seed).normal(size=rank).astype(np.float64)
    return value/np.linalg.norm(value)


def predict_numpy(z,method,common,learned=None,direction=None):
    assert method in REPAIRS+CONTROLS
    z=np.asarray(z,np.float64);one=z.ndim==1;z=z[None] if one else z
    ms,mf=common['success_mean'],common['failure_mean']
    if method in ('mean','reverse','random_direction'):
        value=ms-mf
        if method=='reverse':value=-value
        if method=='random_direction':
            assert direction is not None and direction.shape==value.shape
            value=direction/np.linalg.norm(direction)*np.linalg.norm(value)
        q=np.broadcast_to(value,z.shape).copy()
    elif method=='contract':q=ms-z
    elif method=='boundary':
        gap=mf-ms;denominator=float(gap@gap)
        q=-np.maximum((z-ms)@gap/denominator,0.)[:,None]*gap if denominator>0 else np.zeros_like(z)
    elif method=='ot':q=ms+(z-mf)@common['ot'].T-z
    elif method=='local':
        rows=[]
        for value in z:
            distance=np.square((common['local_failed']-value)/common['local_scale']).mean(1)
            indices=np.argsort(distance,kind='stable')[:min(16,len(distance))]
            scores=-distance[indices];weights=np.exp(scores-scores.max());weights/=weights.sum()
            rows.append(weights@common['local_delta'][indices])
        q=np.stack(rows)
    elif method.startswith('linear'):
        assert learned is not None;q=linear_predict(z,learned)
    else:raise ValueError('MLP inference uses the lazy torch runtime.')
    assert np.isfinite(q).all();return q[0] if one else q


def fit_risk99(ordinary_failure,ordinary_high):
    """Frozen old directional gate, never classify optimized teacher states."""
    gap=ordinary_failure.mean(0)-ordinary_high.mean(0)
    variance=float(gap@((covariance(ordinary_failure)+covariance(ordinary_high))*.5)@gap)
    risk=gap/max(np.sqrt(max(variance,0.)),1e-12);origin=ordinary_high.mean(0)
    values=(ordinary_high-origin)@risk
    return dict(risk=risk,origin=origin,threshold=float(np.quantile(values,.99,method='linear')),
        source='Corresponding training folds ordinary Base failure/high only; no teacher-state classifier',quantile=.99,comparison='strict >')


def make_mlp(rank,initial_seed,device='cpu'):
    import torch
    hidden=max(64,min(256,2*rank))
    # Only the CPU default generator is touched, and restored. Moving the
    # deterministic model afterwards never consumes a native CUDA RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(initial_seed))
        model=torch.nn.Sequential(torch.nn.Linear(rank,hidden),torch.nn.Tanh(),torch.nn.Linear(hidden,rank))
        torch.nn.init.zeros_(model[-1].weight);torch.nn.init.zeros_(model[-1].bias)
    return model.to(device)


def mlp_training_inputs(dataset,D0):
    assert D0>0 and np.isfinite(D0)
    normalized,mean,scale=standardize(dataset['x'])
    return dict(x_float32=normalized.astype(np.float32),target_float32=(dataset['y']/D0).astype(np.float32),
        input_mean=mean,input_scale=scale,D0=float(D0),target_scale='Only corresponding training fold D0; no old success p_scale or target centering')


def train_mlp(dataset,D0,initial_seed,approved_updates=2000,common_lr_multiplier=1.,device='cpu',callback=None,resume=None):
    """Caller owns actual release/source binding and budget; no fit CLI here.

    Control may pass its frozen update count. Real protocol uses2000, or an
    independently preregistered continuation. Never select a best checkpoint.
    """
    import torch
    assert isinstance(approved_updates,int) and approved_updates>0 and common_lr_multiplier in (1.,.5,.25,.125,.0625)
    prepared=mlp_training_inputs(dataset,D0);rank=dataset['x'].shape[1]
    import hashlib,json
    tensor_identity=hashlib.sha256(prepared['x_float32'].tobytes()+prepared['target_float32'].tobytes()+json.dumps(dataset['row_roles']).encode()).hexdigest()
    model=make_mlp(rank,initial_seed,device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3*common_lr_multiplier,weight_decay=.01)
    x=torch.from_numpy(prepared['x_float32']).to(device);target=torch.from_numpy(prepared['target_float32']).to(device)
    generator=torch.Generator(device='cpu').manual_seed(int(initial_seed)+1)
    def evaluate():
        with torch.no_grad():
            error=torch.square(model(x)-target)
            by_role={}
            for role in dict.fromkeys(dataset['row_roles']):
                mask=torch.tensor([v==role for v in dataset['row_roles']],device=device,dtype=torch.bool)
                by_role[role]=float(error[mask].mean())
            return dict(training_MSE=float(error.mean()),role_normalized_MSE=by_role)
    curve=[dict(step=0,**evaluate())];start=0
    if resume is not None:
        assert resume['initial_seed']==initial_seed and resume['D0']==D0
        assert resume['training_tensors_sha256']==tensor_identity
        assert np.array_equal(resume['input_mean'],prepared['input_mean']) and np.array_equal(resume['input_scale'],prepared['input_scale'])
        import copy
        model.load_state_dict(resume['model'].state_dict());optimizer.load_state_dict(copy.deepcopy(resume['optimizer'].state_dict()))
        generator.set_state(resume['batch_generator_state']);curve=list(resume['curve']);start=resume['completed_updates']
        assert start<approved_updates and optimizer.param_groups[0]['lr']==1e-3*common_lr_multiplier
    for step in range(start+1,approved_updates+1):
        indices=torch.randint(len(x),(128,),generator=generator,device='cpu').to(device)
        optimizer.zero_grad(set_to_none=True)
        loss=torch.square(model(x[indices])-target[indices]).mean()
        assert bool(torch.isfinite(loss));loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step%100==0 or step==approved_updates:curve.append(dict(step=step,**evaluate(),gradient_norm_before_clip=float(norm)))
        if callback is not None:callback(step,model,optimizer,generator,curve)
    return dict(model=model,optimizer=optimizer,batch_generator_state=generator.get_state(),input_mean=prepared['input_mean'],input_scale=prepared['input_scale'],
        D0=float(D0),initial_seed=initial_seed,approved_updates=approved_updates,curve=curve,
        completed_updates=approved_updates,
        training_tensors_sha256=tensor_identity,
        recipe=dict(hidden=max(64,min(256,2*rank)),activation='tanh',last_layer_zero=True,AdamW_lr=1e-3*common_lr_multiplier,
                    weight_decay=.01,batch=128,sampling='private CPU generator with replacement',clip_norm=1.,target_scale='training D0',endpoint='final only'))


def control_teacher(x_normalized,rank,initial_seed,teacher_seed,output_RMS):
    """Exactly realizable same-family target, including ordinary references."""
    import torch
    assert x_normalized.dtype==np.float32 and x_normalized.shape[1]==rank and output_RMS>0
    teacher=make_mlp(rank,initial_seed,'cpu');generator=torch.Generator(device='cpu').manual_seed(int(teacher_seed))
    with torch.no_grad():
        teacher[-1].weight.copy_(torch.randn(teacher[-1].weight.shape,generator=generator)/np.sqrt(teacher[-1].in_features))
        values=teacher(torch.from_numpy(x_normalized));mean=values.mean(0);rms=torch.sqrt(torch.square(values-mean).mean())
        assert float(rms)>0
        factor=float(output_RMS)/float(rms)
        teacher[-1].weight.mul_(factor);teacher[-1].bias.copy_(-mean*factor)
        target=teacher(torch.from_numpy(x_normalized)).numpy()
    return teacher,target,dict(initial_seed=initial_seed,teacher_seed=teacher_seed,output_RMS=float(np.sqrt(np.square(target.astype(np.float64)).mean())),
        all_rows_same_teacher_function=True,reference_zero_override=False)


def torch_runtime(common,method,learned=None,mlp_fit=None,direction=None,device='cpu'):
    """Returns nn.Module: z[...,r] FP32 -> requested coefficient correction.

    No cap, dose, gating, GT, aux, scores, IDs or selection is hidden inside.
    """
    import copy
    import torch
    assert method in REPAIRS+CONTROLS
    class Runtime(torch.nn.Module):
        def __init__(self):
            super().__init__();self.method=method
            for key in ('success_mean','failure_mean','ot','local_scale','local_failed','local_delta'):
                self.register_buffer(key,torch.tensor(common[key],dtype=torch.float32,device=device))
            if learned is not None:
                for key in ('input_mean','input_scale','target_mean','target_scale','weight','bias'):
                    self.register_buffer('linear_'+key,torch.tensor(learned[key],dtype=torch.float32,device=device))
            if direction is not None:self.register_buffer('direction',torch.tensor(direction,dtype=torch.float32,device=device))
            if mlp_fit is not None:
                self.network=copy.deepcopy(mlp_fit['model']).to(device).eval().requires_grad_(False)
                self.register_buffer('mlp_mean',torch.tensor(mlp_fit['input_mean'],dtype=torch.float32,device=device))
                self.register_buffer('mlp_scale',torch.tensor(mlp_fit['input_scale'],dtype=torch.float32,device=device));self.D0=mlp_fit['D0']
        def forward(self,z):
            assert z.dtype==torch.float32 and z.shape[-1]==len(self.success_mean)
            ms,mf=self.success_mean,self.failure_mean
            if self.method in ('mean','reverse','random_direction'):
                q=ms-mf
                if self.method=='reverse':q=-q
                if self.method=='random_direction':q=self.direction/self.direction.norm()*q.norm()
                return q.expand_as(z)
            if self.method=='contract':return ms-z
            if self.method=='boundary':
                gap=mf-ms;denominator=gap.square().sum()
                if float(denominator)==0:return torch.zeros_like(z)
                return -torch.clamp(((z-ms)*gap).sum(-1)/denominator,min=0.)[...,None]*gap
            if self.method=='ot':return ms+(z-mf)@self.ot.T-z
            if self.method=='local':
                shape=z.shape;outputs=[]
                for value in z.reshape(-1,shape[-1]):
                    distance=torch.square((self.local_failed-value)/self.local_scale).mean(1)
                    indices=torch.argsort(distance,stable=True)[:min(16,len(distance))]
                    outputs.append((torch.softmax(-distance[indices],0)[:,None]*self.local_delta[indices]).sum(0))
                return torch.stack(outputs).reshape(shape)
            if self.method.startswith('linear'):
                y=((z-self.linear_input_mean)/self.linear_input_scale)@self.linear_weight+self.linear_bias
                return y*self.linear_target_scale+self.linear_target_mean
            if self.method.startswith('mlp'):return self.network((z-self.mlp_mean)/self.mlp_scale)*self.D0
            raise ValueError(self.method)
    return Runtime()


def selftest(include_torch=True):
    rng=np.random.default_rng(2026102485);f=rng.normal(size=(32,4));s=f@np.diag([1.1,.9,1.2,.8])+np.array([1.,2.,3.,4.])
    replay=rng.normal(size=(7,4));base=paired_dataset(f,s);with_replay=paired_dataset(f,s,replay)
    assert len(base['x'])==64 and len(with_replay['x'])==71
    assert np.array_equal(with_replay['x'][32:64],s) and np.count_nonzero(with_replay['y'][32:])==0
    assert with_replay['row_roles'][-7:]==['ordinary_Base_gt90_zero']*7
    algebra=fit_algebra(f,s);a=algebra['ot'];cov_error=float(np.max(np.abs(a@algebra['failure_covariance_regularized']@a.T-algebra['success_covariance_regularized'])))
    assert cov_error<1e-10
    assert np.max(np.abs(algebra['success_mean']+(algebra['failure_mean']-algebra['failure_mean'])@a.T-algebra['success_mean']))==0
    linear=fit_linear(with_replay);assert not linear['bias_penalized']
    assert linear['normal_equation_penalty_diagonal'][-1]==0 and np.all(linear['normal_equation_penalty_diagonal'][:-1]==1)
    constant=dict(x=rng.normal(size=(30,4)),y=np.full((30,4),7.),counts={})
    unpenalized=fit_linear(constant);assert np.max(np.abs(linear_predict(constant['x'],unpenalized)-7.))<1e-12
    for method in ('mean','contract','boundary','ot','local','linear','linear_replay','reverse','random_direction'):
        pred=predict_numpy(f[:3],method,algebra,linear,random_direction(4,13));assert pred.shape==(3,4)
    out=dict(status='PASS_U1_REPAIR9_PLUS2_SYNTHETIC_FUNCTION_SELFTEST',real_candidate_fits=0,
        paired_success_zero_targets_preserved=True,replay_adds_ordinary_high_zero=True,bias_unpenalized_constant_exact=True,
        OT_regularized_covariance_mapping_max_abs_error=cov_error,ridge_normal_equation_relative_residual=linear['normal_equation_relative_residual'],
        loss_mean_scene_sum_rank=True,source_inputs_only_coordinates=True)
    if include_torch:
        import torch
        torch.set_num_threads(1);z=torch.from_numpy(f[:3].astype(np.float32));errors={}
        for method in ('mean','contract','boundary','ot','local','linear','linear_replay','reverse','random_direction'):
            rt=torch_runtime(algebra,method,linear,direction=random_direction(4,13))
            actual=rt(z).detach().numpy();expected=predict_numpy(f[:3],method,algebra,linear,random_direction(4,13))
            errors[method]=float(np.max(np.abs(actual-expected)));assert errors[method]<2e-5
        states=torch.random.get_rng_state().clone();counts={}
        for rank in (8,48,256):
            m=make_mlp(rank,2026102485);assert torch.equal(states,torch.random.get_rng_state())
            assert torch.count_nonzero(m(torch.zeros((4,rank))))==0
            counts[str(rank)]=sum(p.numel() for p in m.parameters())
        x=rng.normal(size=(12,8)).astype(np.float32);teacher,target,proof=control_teacher(x,8,19,23,.1)
        assert np.square(target[-4:].astype(np.float64)).mean()>1e-5
        out.update(torch_CPU_inference_max_abs_errors=errors,MLP_parameter_counts=counts,MLP_zero_init=True,
            CPU_initializer_preserves_native_RNG=True,control_reference_nonzero_same_function=True,control_teacher=proof,GPU_used=False)
    return out


if __name__=='__main__':
    import json
    print(json.dumps(selftest(),ensure_ascii=False),flush=True)
