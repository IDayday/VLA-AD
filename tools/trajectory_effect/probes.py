"""One conditional trajectory score head and explicitly scoped VLM updates."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class TrajectoryScoreHead(nn.Module):
    """Nonlinear trajectory queries attend to every native visual/language token.

    All arms use the same parameters and candidate labels. T replaces context
    tokens with zeros; its ego/history/command input is otherwise unchanged.
    No scene identity, reward, candidate source or checkpoint ID is accepted.
    """
    def __init__(self, context_dim=1536, width=192, depth=2):
        super().__init__()
        self.context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, width))
        self.query = nn.Sequential(nn.Linear(24+20, width), nn.GELU(), nn.Linear(width, width))
        self.attention = nn.ModuleList([nn.MultiheadAttention(width, 6, dropout=0, batch_first=True) for _ in range(depth)])
        self.norm = nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(width,width*4),nn.GELU(),nn.Linear(width*4,width)) for _ in range(depth)])
        self.readout = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width,1))

    def forward(self, tokens, trajectories, state, remove_visual=False):
        # One observation encoding, any number of whole-trajectory joint actions.
        if tokens.ndim != 3 or trajectories.ndim != 4:
            raise ValueError('Expected [B,T,1536] tokens and [B,C,8,3] trajectories')
        t=trajectories.detach().float().clone()
        t[...,0]=(t[...,0]+1.57)/66.74*2-1
        t[...,1]=(t[...,1]+19.68)/42*2-1
        t[...,2]=(t[...,2]+1.67)/3.53*2-1
        s=state.float().clone(); s[...,4:]/=10  # fixed units scaling, no reward normalization
        query=self.query(torch.cat([t.flatten(-2),s[:,None].expand(-1,t.shape[1],-1)],-1))
        context=self.context(torch.zeros_like(tokens) if remove_visual else tokens.float())
        for attention,norm,ffn in zip(self.attention,self.norm,self.ffn):
            attended,_=attention(query,context,context,need_weights=False)
            query=norm(query+attended)
            query=query+ffn(query)
        return self.readout(query).squeeze(-1)


def effect_loss(pred, labels, pair_indices, lambda_diff):
    labels=labels.detach().to(pred)
    absolute=(pred-labels).square().mean()
    a,b=pair_indices.unbind(-1)
    difference=((pred[...,a]-pred[...,b])-(labels[...,a]-labels[...,b])).square().mean()
    return absolute+lambda_diff*difference, absolute, difference


class LoRALinear(nn.Module):
    """Rank-8 adaptation of an existing Qwen attention projection, no dropout."""
    def __init__(self, base, rank=8):
        super().__init__(); self.base=base; self.base.requires_grad_(False)
        self.a=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.b=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a,a=math.sqrt(5)); self.scale=2.0

    def forward(self,x):
        value=self.base(x)
        return value+(F.linear(F.linear(x.float(),self.a),self.b)*self.scale).to(value.dtype)


def adapt_vlm(agent, rank=8):
    """All existing language attention Q/V projections; vision trunk stays frozen.

    This is the fixed scope of 'update VLM', not a claim about full-model tuning.
    Zero B initialization preserves the native BF16 forward bit for bit.
    """
    model=agent.backbone.model; model.requires_grad_(False)
    names=[]
    for name,module in list(model.language_model.named_modules()):
        if isinstance(module,nn.Linear) and name.rsplit('.',1)[-1] in ('q_proj','v_proj'):
            parent_name,leaf=name.rsplit('.',1)
            model.language_model.get_submodule(parent_name).__setattr__(leaf,LoRALinear(module,rank))
            names.append(name)
    if not names: raise ValueError('No native language attention projections matched')
    model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.language_model.config.use_cache=False
    return names


def joint_training_mode(agent):
    # Qwen requires training=True to enable checkpointing. Its attention dropout
    # must be exactly zero, and all ordinary dropout modules are disabled.
    language=agent.backbone.model.language_model
    if getattr(language.config,'attention_dropout',0)!=0:
        raise ValueError('Nonzero attention dropout would confound joint training')
    language.train()
    for module in language.modules():
        if isinstance(module,nn.Dropout):module.eval()
    agent.backbone.model.vision_model.eval(); agent.action_head.eval()


def vlm_parameters(agent):
    return [p for p in agent.backbone.parameters() if p.requires_grad]


def vlm_state(agent):
    return {n:p.detach().cpu().clone() for n,p in agent.backbone.named_parameters() if p.requires_grad}
