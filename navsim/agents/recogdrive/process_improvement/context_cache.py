"""Scoped common-subexpression reuse of native cross-attention K/V linears.

Observation conditioning is constant along a detached chain. Reusing a linear's
result keeps its autograd graph (when enabled), so contributions from all steps
still reach the shared projection and K/V parameters. No detached projection is
stored across losses or updates. This optimization requires a separate real
forward/gradient/update parity gate and is never enabled by baseline defaults.
"""
from contextlib import contextmanager
import threading
import torch


@contextmanager
def context_cache(head):
    if getattr(head, '_process_context_cache_active', False):
        raise RuntimeError('Nested/reentrant planner cache forbidden')
    if head.training:
        raise ValueError('Native planner must retain fixed eval mode (trainable parameters may still receive gradients)')
    owner = threading.get_ident()
    head._process_context_cache_active = True
    versions = tuple(p._version for p in head.parameters())
    restore = []
    references = []
    counts = dict(hits=0, misses=0)
    def wrap(original):
        store = {}
        def forward(x, *args, **kwargs):
            if threading.get_ident() != owner:
                raise RuntimeError('A planner with scoped cache cannot cross threads')
            key = (x.data_ptr(), x._version, tuple(x.shape), tuple(x.stride()), x.dtype, x.device, torch.is_grad_enabled())
            if key in store:
                counts['hits'] += 1
                return store[key]
            value = original(x, *args, **kwargs)
            references.append(x)
            store[key] = value
            counts['misses'] += 1
            return value
        return forward
    try:
        for index, block in enumerate(head.model.transformer_blocks):
            if head.model.interleave_attention and index % 2 == 0:
                continue
            for module in (block.attn.to_k, block.attn.to_v):
                original = module.forward
                module.forward = wrap(original)
                restore.append((module, original))
        yield counts
    finally:
        for module, original in restore:
            module.forward = original
        del head._process_context_cache_active
        if tuple(p._version for p in head.parameters()) != versions:
            raise RuntimeError('Parameters changed while an observation-local graph cache was active')
