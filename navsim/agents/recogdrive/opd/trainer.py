"""Pure distillation objective; rollout generation never enters autograd."""
import torch
from .ddim_adapter import DDIMAdapter
from .losses import transition_loss,diffusion_loss


def frozen_query(head,chain,tokens,inputs):
    if any(p.requires_grad for p in head.parameters()):raise ValueError('Teacher/anchor must be frozen')
    with torch.no_grad():return DDIMAdapter(head).evaluate_chain(chain,tokens,inputs)


def objective(student,teacher,anchor,tokens,inputs,*,experiment='E3',target=None,teacher_trajectory=None,
              teacher_chain=None,rollout_seed=0,mask=1.,protected=1.,weights=None,deterministic=False,
              loss_type='deterministic_mean_mse',kernel='native',rollout_noise=None):
    device=tokens.device;counts=dict(student_rollout=0,student_query=0,teacher_query=0,anchor_query=0,student_diffusion=0)
    mask=torch.broadcast_to(torch.as_tensor(mask,device=device,dtype=torch.float32),(len(tokens),))
    protected=torch.broadcast_to(torch.as_tensor(protected,device=device,dtype=torch.float32),(len(tokens),))
    zero=sum(p.flatten()[0]*0 for p in student.parameters() if p.requires_grad)
    opd=zero;anch=zero;il=zero;per_step=torch.zeros(1,5,device=device);chain=None;student_means=None;teacher_means=None
    if experiment=='E1':
        if teacher_trajectory is None:raise ValueError('Teacher pseudo-GT missing')
        il=diffusion_loss(student,tokens,inputs,teacher_trajectory);counts['student_diffusion']=len(tokens)
    else:
        if experiment=='E2':
            if teacher_chain is None:raise ValueError('Teacher chain missing')
            chain=teacher_chain
        else:
            with torch.random.fork_rng(devices=[device.index]):
                torch.manual_seed(rollout_seed);torch.cuda.manual_seed(rollout_seed)
                chain=DDIMAdapter(student).sample(tokens,inputs,deterministic=deterministic,kernel=kernel,
                    init=rollout_noise[0] if rollout_noise is not None else None,noises=rollout_noise[1] if rollout_noise is not None else None)
            counts['student_rollout']=5*len(tokens)
        pred=DDIMAdapter(student).evaluate_chain(chain,tokens,inputs,require_current=experiment!='E2')
        student_means=pred.detach()
        counts['student_query']=5*len(tokens)
        if mask.any():
            target_means=frozen_query(teacher,chain,tokens,inputs);counts['teacher_query']=5*len(tokens)
            teacher_means=target_means.detach()
            teacher_sigma=None
            if loss_type=='shared_variance_kl':
                c=DDIMAdapter(teacher).prepare_conditioning(tokens,inputs)
                with torch.no_grad():
                    teacher_sigma=torch.stack([(.5*DDIMAdapter(teacher).transition(chain.states[:,k],k,c,deterministic)[1]).exp().clamp_min(.0001) for k in range(5)],1)
            opd,per_step=transition_loss(pred,target_means,weights=weights,mask=mask,objective=loss_type,
                sigma=chain.sigmas,teacher_sigma=teacher_sigma,kernel=kernel)
        if protected.any() and experiment in ('E4-A','E4-C','E5','E6'):
            if anchor is None:raise ValueError('Anchor required')
            anchor_means=frozen_query(anchor,chain,tokens,inputs);anch,_=transition_loss(pred,anchor_means,weights=weights,mask=protected)
            counts['anchor_query']=5*len(tokens)
        if experiment in ('E4-B','E4-C','E5'):
            if target is None:raise ValueError('Training GT missing')
            il=diffusion_loss(student,tokens,inputs,target);counts['student_diffusion']=len(tokens)
    counts.update({k+'_batch_calls':v//len(tokens) for k,v in list(counts.items())})
    return dict(opd=opd,anchor=anch,il=il,per_step=per_step,chain=chain,forward_counts=counts,student_means=student_means,teacher_means=teacher_means)


def configure_native_grpo(head,rows,reference_path):
    """Original GRPO remains intact; fixed IL BC reference exists exactly once."""
    import copy
    from types import SimpleNamespace
    from pathlib import Path
    from navsim.agents.recogdrive.recogdrive_diffusion_planner import GRPOConfig
    from scripts.opd.scoring import Scoring
    cfg=GRPOConfig()
    for k,v in vars(cfg).items():
        if k not in ('metric_cache_path','reference_policy_checkpoint','scorer_config'):setattr(head,k,v)
    head.metric_cache_loader=SimpleNamespace(metric_cache_paths={r['token']:Path(r['metric_cache_path']) for r in rows})
    scoring=Scoring(progress_weight=cfg.scorer_config.progress_weight);head.simulator=scoring.simulator;head.train_scorer=scoring.scorer
    from .checkpoint_io import restore
    reference=copy.deepcopy(head);restore(reference,reference_path);reference.requires_grad_(False);reference.eval()
    head.old_policy=reference;head.config.grpo=True
    return head


def grpo_with_opd(student,teacher,tokens,inputs,tokens_list,*,weight=0.,anchor_weight=0.,group_size=8):
    """Capture native GRPO rollout states; never invent a policy likelihood."""
    if weight==0 and anchor_weight==0:
        return student.forward_grpo(tokens,inputs,tokens_list,sample_time=group_size),None
    if anchor_weight!=0:raise ValueError('Native GRPO already has IL BC; separate anchor requires a registered ablation')
    captured={};native=student.sample_chain
    def capture(*args,**kwargs):
        original_mean=student.p_mean_variance;observed=[]
        def observe(*values,**options):
            result=original_mean(*values,**options)
            deterministic=values[6] if len(values)>6 else options.get('deterministic',False)
            sigma=(.5*result[1]).exp().to(values[0].dtype)
            sigma=torch.zeros_like(sigma) if deterministic else sigma.clamp_min(student.min_sampling_denoising_std)
            observed.append((values[0].detach().clone(),result[0].detach().clone(),sigma.detach().clone()))
            return result
        student.p_mean_variance=observe
        try:states,trajectory=native(*args,**kwargs)
        finally:student.p_mean_variance=original_mean
        assert len(observed)==5 and torch.equal(torch.stack([v[0] for v in observed],1),states[:,:-1])
        captured.update(states=states.detach().clone(),trajectory=trajectory.detach().clone(),
            means=torch.stack([v[1] for v in observed],1),sigmas=torch.stack([v[2] for v in observed],1))
        return states,trajectory
    student.sample_chain=capture
    try:result=student.forward_grpo(tokens,inputs,tokens_list,sample_time=group_size)
    finally:student.sample_chain=native
    from .ddim_adapter import Chain
    states=captured['states'];b=len(states)
    if len(tokens)!=1:raise ValueError('E6 integration currently uses one scene per group')
    from transformers.feature_extraction_utils import BatchFeature
    expanded=BatchFeature({k:v.expand(b,*v.shape[1:]) for k,v in inputs.items()})
    context=tokens.expand(b,-1,-1)
    chain=Chain(states,captured['means'],captured['sigmas'],student.ddim_t,torch.arange(5,device=tokens.device),
        captured['trajectory'],tuple(p._version for p in student.parameters()),id(student),False,'native')
    pred=DDIMAdapter(student).evaluate_chain(chain,context,expanded,require_current=True)
    target=frozen_query(teacher,chain,context,expanded);aux,steps=transition_loss(pred,target)
    result['loss']=result.loss+weight*aux
    return result,dict(opd=aux,per_step=steps,states=states,sigmas=captured['sigmas'],timesteps=student.ddim_t.detach(),
        indices=torch.arange(5,device=tokens.device),teacher_forwards=5)
