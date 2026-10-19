import torch


def transition_loss(student,target,weights=None,mask=None,objective='deterministic_mean_mse',
                    sigma=None,teacher_sigma=None,kernel='native',sigma_floor=1e-3):
    per_step=(student.float()-target.detach().float()).square().flatten(2).mean(-1)
    if objective=='shared_variance_kl':
        if kernel!='gaussian':raise ValueError('Native clipped-normal sampler is not a Gaussian kernel; exact Gaussian KL is invalid')
        if sigma is None or teacher_sigma is None or not torch.equal(sigma,teacher_sigma):raise ValueError('Shared actual variance required')
        s=sigma.float().flatten(2)[...,0]
        valid=s>=sigma_floor
        if not valid.any():raise ValueError('No nondegenerate Gaussian steps')
        per_step=torch.where(valid,per_step*student[0,0].numel()/(2*s.square().clamp_min(sigma_floor**2)),0.)
        weights=valid.to(per_step) if weights is None else weights.to(per_step)*valid
    elif objective!='deterministic_mean_mse':raise ValueError(objective)
    w=torch.ones_like(per_step) if weights is None else torch.broadcast_to(weights.to(per_step),per_step.shape)
    if (w<0).any() or not torch.isfinite(per_step).all():raise FloatingPointError('Invalid transition loss')
    per_scene=(per_step*w).sum(-1)/w.sum(-1).clamp_min(1.)
    m=torch.ones_like(per_scene) if mask is None else mask.detach().to(per_scene)
    if not torch.isfinite(m).all() or (m<0).any():raise ValueError('Invalid gate')
    loss=(per_scene*m).sum()/m.sum().clamp_min(1.)
    return loss,per_step


def diffusion_loss(head,tokens,inputs,target):
    """Native normalized-target random-timestep noise-prediction objective."""
    from transformers.feature_extraction_utils import BatchFeature
    data=BatchFeature(dict(inputs,action=target.detach(),_allow_target_tokens_for_loss=True))
    return head(tokens,data).diffusion_loss
