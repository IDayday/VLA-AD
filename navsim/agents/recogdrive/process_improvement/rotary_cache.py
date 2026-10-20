"""Scoped native RoPE constants for the audited fixed eight action positions.

Native Attention.forward always supplies arange(Nq)[None], with Nq=8. The
original rotary module ignores x values and uses x only for device/dtype. Its
unconditional CUDA max().item() is a host synchronization and blocks CUDA graph
capture. Evaluate the SAME native gather once, retain the exact constants, and
restore the original method on exit. No global baseline code is changed.
"""
from contextlib import contextmanager
import threading
import torch


@contextmanager
def fixed_rotary_cache(head):
    rope=head.model.rotary_embedder
    if getattr(rope,'_process_fixed_active',False):raise RuntimeError('Nested rotary scope forbidden')
    owner=threading.get_ident();original=rope.forward
    parameter=next(head.model.parameters());version=rope.inv_freq._version
    with torch.no_grad():
        positions=torch.arange(8,device=parameter.device).unsqueeze(0)
        cosine,sine=original(torch.empty(1,8,head.config.input_embedding_dim,device=parameter.device,dtype=parameter.dtype),positions)
    def forward(x,position_ids):
        if threading.get_ident()!=owner:raise RuntimeError('Scoped rotary wrapper cannot cross threads')
        if x.shape[1]!=8 or position_ids.shape!=(1,8) or x.dtype!=cosine.dtype or x.device!=cosine.device:
            raise ValueError('Audited native eight-position rotary contract changed')
        return cosine,sine
    rope.forward=forward;rope._process_fixed_active=True
    try:yield dict(cosine=cosine,sine=sine)
    finally:
        rope.forward=original;del rope._process_fixed_active
        if rope.inv_freq._version!=version:raise RuntimeError('Rotary frequency buffer changed inside fixed-position scope')


class _CachedNativeEta(torch.nn.Module):
    def __init__(self,original):
        super().__init__()
        # Preserve the exact original parameter object, name and state_dict.
        self.register_parameter('eta_logit',original.eta_logit)
        self.min,self.max=original.min,original.max
        with torch.no_grad():
            value=original(torch.empty(1,8,3,device=original.eta_logit.device))
        self.register_buffer('value',value,persistent=False)
    def forward(self,x):return self.value.expand(len(x),1)


@contextmanager
def fixed_eta_cache(head):
    original=head.eta
    if original.eta_logit.requires_grad:raise ValueError('Cannot cache a trainable eta')
    cached=_CachedNativeEta(original);version=original.eta_logit._version
    head.eta=cached
    try:yield cached.value
    finally:
        head.eta=original
        if original.eta_logit._version!=version:raise RuntimeError('Frozen eta changed during constant scope')
