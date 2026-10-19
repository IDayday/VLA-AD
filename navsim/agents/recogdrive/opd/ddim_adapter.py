"""Native p_mean_variance is the single source of transition mathematics."""
from dataclasses import dataclass
import torch


@dataclass
class Chain:
    states: torch.Tensor
    means: torch.Tensor
    sigmas: torch.Tensor
    timesteps: torch.Tensor
    indices: torch.Tensor
    trajectory: torch.Tensor
    versions: tuple
    source: int
    deterministic: bool
    kernel: str


class DDIMAdapter:
    def __init__(self,head):
        self.head=head
        if head.config.sampling_method!='ddim':raise ValueError('DDIM required')
        for key in ('use_expert_features','use_last_rd','use_bit_drive','use_risk_vla','use_action_aware_aux'):
            if getattr(head.config,key,False):raise ValueError(f'Nonbaseline path enabled: {key}')

    def prepare_conditioning(self,tokens,inputs):
        if set(inputs)-{'state','his_traj','history_trajectory','status_feature','high_command_one_hot'}:
            raise ValueError('Labels or unaudited features in inference conditioning')
        h=self.head;c=h._prepare_dit_context(tokens,inputs,training=False)
        return dict(context_embeds=c['context_tokens'],context_mean=c['context_mean'],
            his_traj_features=h.his_traj_encoder(inputs.his_traj.unsqueeze(1)).repeat(1,h.config.action_horizon,1),
            ego_status_features=h.ego_status_encoder(inputs.status_feature),
            expert_step_condition=None,bit_action_condition=None)

    def transition(self,x,index,conditioning,deterministic=False):
        h=self.head;b=x.shape[0]
        indices=torch.full((b,),index,device=x.device,dtype=torch.long) if isinstance(index,int) else index
        if indices.shape!=(b,) or (indices<0).any() or (indices>=h.ddim_steps).any():raise ValueError('Invalid scheduler indices')
        time=h.ddim_t[indices]
        return h.p_mean_variance(x.detach(),time,indices,deterministic=deterministic,**conditioning)

    def sample(self,tokens,inputs,*,deterministic=False,init=None,kernel='native'):
        if kernel not in ('native','gaussian'):raise ValueError(kernel)
        h=self.head
        with torch.no_grad():
            c=self.prepare_conditioning(tokens,inputs)
            x=init.detach().clone() if init is not None else torch.randn((len(tokens),8,3),device=tokens.device,dtype=tokens.dtype)
            states=[x.clone()];means=[];sigmas=[]
            for k in range(h.ddim_steps):
                mean,logvar,_=self.transition(x,k,c,deterministic)
                std=(.5*logvar).exp().to(tokens.dtype)
                noise=torch.randn_like(x)
                if deterministic:std.zero_()
                else:std=std.clamp(min=getattr(h,'eval_min_sampling_denoising_std',.0001))
                if kernel=='native':noise.clamp_(-getattr(h,'eval_randn_clip_value',1.),getattr(h,'eval_randn_clip_value',1.))
                x=mean+std*noise;states.append(x.clone());means.append(mean);sigmas.append(std)
            x=x.clamp(-getattr(h,'final_action_clip_value',1.),getattr(h,'final_action_clip_value',1.))
            return Chain(torch.stack(states,1).detach().clone(),torch.stack(means,1).detach().clone(),
                torch.stack(sigmas,1).detach().clone(),h.ddim_t.detach().clone(),torch.arange(h.ddim_steps,device=tokens.device),
                h.denorm_odo(x).detach().clone(),tuple(p._version for p in h.parameters()),id(h),deterministic,kernel)

    def evaluate_chain(self,chain,tokens,inputs,*,require_current=False):
        if require_current and (chain.source!=id(self.head) or chain.versions!=tuple(p._version for p in self.head.parameters())):
            raise ValueError('Stale or nonstudent on-policy rollout')
        c=self.prepare_conditioning(tokens,inputs)
        return torch.stack([self.transition(chain.states[:,k].detach().clone(),k,c,chain.deterministic)[0]
                            for k in range(self.head.ddim_steps)],1)


def sample_student_chain(student,tokens,inputs,**kwargs):return DDIMAdapter(student).sample(tokens,inputs,**kwargs)
