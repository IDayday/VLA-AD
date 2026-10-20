"""Opt-in CUDA graph scheduling of the exact frozen native B1 DDIM operators.

No precision, batch, timestep, noise, weights, or trajectory-selection changes.
Explicit noise buffers keep all randomness outside capture. Formal admission is
separate and requires actual E0 trajectories and scores, including discrete
components, to agree. This module is never used for optimization.
"""
import torch
from navsim.agents.recogdrive.opd.inference_cache import native_inference_cache
from .suffix_rollout import NativeAdapter
from .rotary_cache import fixed_rotary_cache,fixed_eta_cache


class FrozenNativeGraph:
    def __init__(self,head,tokens,inputs):
        if head.training or any(p.requires_grad for p in head.parameters()):
            raise ValueError('A native inference graph requires a completely frozen eval head')
        if tokens.shape!=(1,2800,1536) or tokens.dtype!=torch.float32:
            raise ValueError('Only qualified native single-scene FP32 conditioning is supported')
        self.head=head;self.adapter=NativeAdapter(head)
        self.parameter_versions=tuple(p._version for p in head.parameters())
        with torch.no_grad():context=self.adapter.prepare_conditioning(tokens,inputs)
        self.context={k:v.detach().clone() if v is not None else None for k,v in context.items()}
        self.initial=torch.zeros(1,8,3,device=tokens.device,dtype=tokens.dtype)
        self.noises=torch.zeros(1,5,8,3,device=tokens.device,dtype=tokens.dtype)
        self.times=[head.ddim_t[i:i+1] for i in range(5)]
        self.indices=[torch.full((1,),i,device=tokens.device,dtype=torch.long) for i in range(5)]
        with fixed_rotary_cache(head) as rotary_constants,fixed_eta_cache(head) as eta_constant:
            # External graph constants need strong references for the graph's
            # entire lifetime, even after restoring the native Python method.
            self.rotary_constants=rotary_constants
            self.eta_constant=eta_constant
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream),torch.no_grad():
                for _ in range(3):self._compute()
            torch.cuda.current_stream().wait_stream(stream)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph,stream=stream),torch.no_grad():
                self.output,self.states=self._compute()

    def _compute(self):
        x=self.initial;states=[x]
        # A fresh scope per compute is essential: K/V nodes must be captured,
        # never constants memoized during a preceding warm-up invocation.
        with native_inference_cache(self.head):
            for i in range(5):
                mean,logvar,_=self.head.p_mean_variance(x,self.times[i],self.indices[i],deterministic=False,**self.context)
                std=(.5*logvar).exp().to(torch.float32).clamp(min=getattr(self.head,'eval_min_sampling_denoising_std',.0001))
                clip=getattr(self.head,'eval_randn_clip_value',1.)
                x=mean+std*self.noises[:,i].clamp(-clip,clip)
                states.append(x)
            output=self.adapter.finalize(x)
        return output,torch.stack(states,1)

    @torch.no_grad()
    def prepare(self,tokens,inputs):
        if tuple(p._version for p in self.head.parameters())!=self.parameter_versions:
            raise ValueError('Frozen native graph weights changed')
        fresh=self.adapter.prepare_conditioning(tokens,inputs)
        for key,value in fresh.items():
            if value is not None:self.context[key].copy_(value)

    @torch.no_grad()
    def sample(self,noise,return_states=False):
        if noise.initial.shape!=(1,8,3) or noise.steps.shape!=(1,5,8,3):raise ValueError('Graph noise axes changed')
        self.initial.copy_(noise.initial);self.noises.copy_(noise.steps);self.graph.replay()
        return (self.output.clone(),self.states.clone()) if return_states else self.output.clone()
