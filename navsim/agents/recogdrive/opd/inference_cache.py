"""Opt-in, observation-local memoization of frozen native linear projections.

No sampler, precision, batching, weights or attention mathematics changes.
Never enabled during optimization. Tensor references prevent allocator reuse.
"""
from contextlib import contextmanager
import torch


@contextmanager
def native_inference_cache(head):
    if head.training or any(p.requires_grad for p in head.parameters()):
        raise ValueError('Memoization requires an eval-mode completely frozen head')
    if any(getattr(head.config,k,False) for k in ('use_expert_features','use_last_rd','use_bit_drive','use_risk_vla')):
        raise ValueError('Memoization only supports the audited baseline')
    if getattr(head,'_opd_cache_active',False):raise ValueError('Nested observation caches forbidden')
    head._opd_cache_active=True;versions=tuple(p._version for p in head.parameters());restores=[];references=[]
    stats=dict(linear_hits=0,linear_misses=0,encoder_hits=0,encoder_misses=0)
    def signature(t):return (t.data_ptr(),t._version,tuple(t.shape),tuple(t.stride()),t.dtype,t.device)
    def memoized(original,kind):
        store={}
        def forward(x,*args,**kwargs):
            if torch.is_grad_enabled():raise RuntimeError('Frozen inference memoization entered autograd')
            if kind=='encoder' and tuple(p._version for p in head.parameters())!=versions:raise RuntimeError('Head changed inside observation cache')
            key=signature(x)
            if key in store:stats[kind+'_hits']+=1;return store[key]
            result=original(x,*args,**kwargs);references.append(x);store[key]=result;stats[kind+'_misses']+=1;return result
        return forward
    original=head._encode_vlm;head._encode_vlm=memoized(original,'encoder');restores.append((head,'_encode_vlm',original))
    for i,block in enumerate(head.model.transformer_blocks):
        if head.model.interleave_attention and i%2==0:continue
        for module in (block.attn.to_k,block.attn.to_v):
            original=module.forward;module.forward=memoized(original,'linear');restores.append((module,'forward',original))
    try:yield stats
    finally:
        for obj,name,original in restores:setattr(obj,name,original)
        del head._opd_cache_active
        if tuple(p._version for p in head.parameters())!=versions:raise RuntimeError('Head changed inside observation cache')
