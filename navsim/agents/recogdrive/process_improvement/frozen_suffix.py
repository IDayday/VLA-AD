"""Opt-in frozen suffix scheduling with exact native projection reuse.

Materialize the same expanded visual context once, rather than once per native
DiT call. This makes its unchanged K/V linears reusable across suffix steps.
The native transition/sampler body is delegated unchanged. Training heads are
forbidden; formal admission requires real batched intermediate-state parity.
"""
import torch
from navsim.agents.recogdrive.opd.inference_cache import native_inference_cache
from .suffix_rollout import NativeAdapter,expand_context


class FrozenSuffixAdapter(NativeAdapter):
    @torch.no_grad()
    def rollout_suffix(self,context,start_state,next_index,raw_noises,return_states=False):
        if self.head.training or any(p.requires_grad for p in self.head.parameters()):
            raise ValueError('Only an immutable frozen inference head can reuse suffix projections')
        context=expand_context(context,len(start_state))
        if context.get('context_mean') is None:
            raise ValueError('Native precomputed context mean is required to preserve reduction ordering')
        context=dict(context,context_embeds=context['context_embeds'].contiguous())
        with native_inference_cache(self.head) as counters:
            result=super().rollout_suffix(context,start_state,next_index,raw_noises,return_states)
        self.last_cache_counts=counters
        return result
