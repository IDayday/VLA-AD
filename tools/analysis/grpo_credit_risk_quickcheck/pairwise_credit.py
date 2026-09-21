"""Minimal, action-dependent pairwise surrogate. No optimizer is called here.

This is not an unbiased rewrite of GRPO. Risk labels must be computed on the
current policy's original sampled actions, using independent training seeds.
Perturbed and externally constructed actions never enter the policy loss.
"""
import torch

def risk_credit(cost,nominal_safe,progress=None,tolerance=.5):
 if cost.ndim!=2 or cost.shape!=nominal_safe.shape:raise ValueError('Expected [B,G]')
 if cost.shape[1]<2:raise ValueError('G must be at least 2')
 if not torch.isfinite(cost).all() or not ((cost>=0)&(cost<=1)).all():raise ValueError('Invalid risk labels')
 mask=nominal_safe[:,:,None]&nominal_safe[:,None,:]
 if progress is not None:
  if progress.shape!=cost.shape or not torch.isfinite(progress).all():raise ValueError('Invalid simulator progress')
  mask &= abs(progress[:,:,None]-progress[:,None,:])<=tolerance
 diagonal=torch.eye(cost.shape[1],device=cost.device,dtype=torch.bool)[None];mask &= ~diagonal
 assert torch.equal(mask,mask.transpose(1,2))
 difference=cost[:,None,:]-cost[:,:,None]
 bonus=(mask.to(cost)*difference).sum(-1)/(cost.shape[1]-1)
 return bonus.detach(),mask

def combine(original,bonus,positive_eligible,clip=3.,weight=1.):
 if original.shape!=bonus.shape or original.shape!=positive_eligible.shape:raise ValueError('Mismatched credit shapes')
 pre=original.detach()+weight*bonus
 # Eligibility is applied after adding risk credit, with no later recentering.
 gated=torch.where(positive_eligible,pre,torch.minimum(pre,torch.zeros_like(pre)))
 final=gated.clamp(-clip,clip).detach()
 return final,dict(original_information_active=(original.abs()>1e-8).any(1),risk_information_active=(bonus.abs()>1e-8).any(1),bonus=bonus,pre_final_eligibility=pre,pre_final_clip=gated)

def ep_half_scalar(nc,dac,ep,ttc,comfort):
 return nc*dac*(5*ep+5*ttc+2*comfort)/17

def norm_match_delta(start,updated_a,updated_d):
 names=sorted(start)
 if set(names)!=set(updated_a) or set(names)!=set(updated_d):raise ValueError('State identity mismatch')
 # Inputs are actual trainable parameters, not gradients or advantage RMS.
 da={k:updated_a[k].double()-start[k].double() for k in names};dd={k:updated_d[k].double()-start[k].double() for k in names}
 na=torch.sqrt(sum(v.square().sum() for v in da.values()));nd=torch.sqrt(sum(v.square().sum() for v in dd.values()))
 if na==0 and nd>0:raise ValueError('Cannot rescale a zero update')
 scale=nd/na if na>0 else na
 return {k:(start[k].double()+scale*da[k]).to(start[k]) for k in names},float(na),float(nd)

def require_update_gate(identity,signal):
 if identity.get('status')!='PASS' or signal.get('status')!='WITHIN_GROUP_SIGNAL':
  raise RuntimeError('NOT_RUN: optimizer updates require verified identity/scoring AND WITHIN_GROUP_SIGNAL')


def validate_trainable_identity(model,expected_constructor_names):
 actual={n for n,v in model.named_parameters() if v.requires_grad}
 extra=actual-set(expected_constructor_names)
 if extra:raise RuntimeError('Trainable identity mismatch: '+','.join(sorted(extra)))
 for n,v in model.named_parameters():
  if v.requires_grad and not torch.isfinite(v).all():raise FloatingPointError('Nonfinite trainable parameter: '+n)
 return actual
