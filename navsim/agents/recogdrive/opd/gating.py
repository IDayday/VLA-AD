import numpy as np
import torch


def advantage_gate(il,rl,*,high=.8,z=1.,margin=.03,safety_indices=(0,1,3)):
    il=np.asarray(il,dtype=float);rl=np.asarray(rl,dtype=float)
    if il.shape!=rl.shape or il.ndim!=2 or len(il)<2 or il.shape[-1]!=7:raise ValueError('Paired multi-seed7-field scores required')
    delta=rl[:,-1]-il[:,-1];lcb=delta.mean()-z*delta.std(ddof=1)/np.sqrt(len(delta))
    ih=il[:,-1].mean()>=high;rh=rl[:,-1].mean()>=high
    category='C' if ih and rh else ('B' if ih else ('A' if rh else 'D'))
    safe=bool(np.all(rl[:,safety_indices]>=il[:,safety_indices]-1e-6))
    return dict(category=category,gate=float(lcb>margin and safe),delta_mean=float(delta.mean()),
                delta_std=float(delta.std(ddof=1)),lcb=float(lcb),safety_nondegraded=safe,samples=len(delta))


def globally_normalized_local_sum(loss,mask,total_valid,world_size=1):
    """DDP averages rank gradients; normalize by global valid scene count."""
    return world_size*(loss*mask).sum()/torch.as_tensor(total_valid,device=loss.device).clamp_min(1.)
