"""Recompute archived LFP credit on fresh groups, never historical advantages."""
from common import *
import torch,dataclasses
install_runtime()
from navsim.agents.recogdrive.stage3_lfp_grpo import (coerce_lfp_grpo_config,compute_lfp_advantages,group_center,distributed_masked_moments)
from navsim.agents.recogdrive.stage3_metric_adapter import Stage3MetricAdapter
from navsim.agents.recogdrive.stage3_reference_cache import Stage3ReferenceCache

@lru_cache(maxsize=1)
def setup():
 m=read(OUT/'manifests/models.json');c=coerce_lfp_grpo_config(m['a5_sft']['grpo_config']['lfp_grpo_cfg']);r=Stage3ReferenceCache(c.reference_cache_path)
 assert r.metadata['stage2_checkpoint_sha256']==m['a5_sft']['sha256']
 return c,r

def scalar(nc,dac,ep,ttc,comfort,ep_factor=1.):
 # B changes the EP numerator only: retain the historical /17 scale, avoiding
 # a hidden rescaling of TTC/Comfort or the nonlinear reference-margin scale.
 return nc*dac*(10*ep_factor*ep+5*ttc+2*comfort)/17

def stages(m,r,c,frozen=None):
 native=compute_lfp_advantages(m,r,c)
 margin=c.reference_margin_weight*torch.clamp((m.scalar-r.scalar[:,None])/c.reference_margin_scale,-1,1)
 f=native.feasible if frozen is None else frozen['feasible']
 centered,base,norm,_=group_center(m.scalar+margin,f)
 moment=distributed_masked_moments(centered,norm,c.global_std_floor)
 mean=moment.mean if frozen is None else frozen['global_mean'][0,0];std=moment.std if frozen is None else frozen['global_std'][0,0]
 z=(centered-mean)/std
 positive=(native.feasible&native.progress_ok&native.ttc_ok&native.pareto_front) if frozen is None else frozen['positive_eligible']
 a=torch.where(f,torch.where(positive,z,z.clamp(max=0)),torch.where(f.any(1)[:,None],torch.full_like(z,-1),z.clamp(max=0) if c.all_infeasible_rescue else torch.zeros_like(z)))
 final=a.clamp(-c.advantage_clip,c.advantage_clip)
 if frozen is None:torch.testing.assert_close(final,native.advantages,rtol=0,atol=0)
 return dict(advantage=final,reference_margin=margin,augmented_score=m.scalar+margin,group_baseline=base[:,None].expand_as(z),centered=centered,normalized=z,pre_clip=a,
 feasible=f,progress_ok=native.progress_ok if frozen is None else frozen['progress_ok'],ttc_ok=native.ttc_ok if frozen is None else frozen['ttc_ok'],pareto=native.pareto_front if frozen is None else frozen['pareto'],positive_eligible=positive,
 normalization_mask=norm,global_mean=torch.ones_like(z)*mean,global_std=torch.ones_like(z)*std)

def ep_shadow(m,r,c,original):
 const=cfg()['shadow_ep_constant'];me=dataclasses.replace(m,ep=torch.full_like(m.ep,const));re=dataclasses.replace(r,ep=torch.full_like(r.ep,const))
 me.scalar=scalar(me.nc,me.dac,me.ep,me.ttc,me.quality);re.scalar=scalar(re.nc,re.dac,re.ep,re.ttc,re.quality)
 # Fixed eligibility and original normalisation for V1. Nonlinear margin is
 # consistently rebuilt from EP-neutral candidate/reference scalar in both.
 return stages(me,re,c,frozen=original),stages(me,re,c)

def main():
 torch.set_num_threads(1);c,refs=setup();save(OUT/'manifests/reference_metadata.json',refs.metadata)
 frame=pd.read_parquet(OUT/'trajectories.parquet');frames=[]
 adapter=Stage3MetricAdapter('navsim_v1');names=dict(NC='nc',DAC='dac',EP='ep',TTC='ttc',Comfort='comfort',DDC='ddc',training_scalar='pdms')
 tokens=sorted([r['token'] for r in scenes()],key=lambda x:digest(['native_credit_batch_v1',x]))
 for model in cfg()['models']:
  for protocol in ['native_grpo','deployment']:
   for group in range(2):
    indexed=frame[(frame.model==model)&(frame.protocol==protocol)&(frame.group==group)].set_index(['token','candidate'])
    for start in range(0,len(tokens),64):
     ts=tokens[start:start+64];b=indexed.loc[[(t,i) for t in ts for i in range(16)]].reset_index();components={v:torch.tensor(b[k].to_numpy().reshape(-1,16),dtype=torch.float32) for k,v in names.items()}
     m=adapter.canonicalize(components,batch_size=len(ts),group_size=16);r=refs.get(ts,'cpu',torch.float32)
     expected=scalar(r.nc,r.dac,r.ep,r.ttc,r.quality);assert torch.max(abs(expected-r.scalar))<1e-6
     original=stages(m,r,c);sh1,sh2=ep_shadow(m,r,c,original)
     for k,v in original.items():b[k]=v.numpy().reshape(-1)
     for v,x in [('shadow_fixed',sh1),('shadow_full',sh2)]:
      b[v+'_advantage']=x['advantage'].numpy().reshape(-1)
      b[v+'_positive_eligible']=x['positive_eligible'].numpy().reshape(-1)
      b[v+'_std']=x['global_std'].numpy().reshape(-1)
     for k in ['scalar','ep','ttc','quality','nc','dac','ddc','gt_ddc','selected_source_code','fallback_mask']:
      b['reference_'+k]=np.repeat(getattr(r,k).numpy(),16)
     b['normalization_batch']=start//64;b['original_information_active']=np.repeat((original['advantage'].abs()>1e-8).any(1).numpy(),16)
     frames.append(b)
 pd.concat(frames,ignore_index=True).to_parquet(OUT/'trajectory_credit.parquet',index=False)
 save(OUT/'audits/credit_replay.json',dict(status='PASS',native_equivalence='exact float32 tensor equality for all groups',group_size=16,batch_size=64,scope='new sample credit, not recovered historical training advantages',optimizer_steps=0,shadow_constant=cfg()['shadow_ep_constant'],shadow_fixed='replace candidate/reference EP by 1, rebuild scalar AND nonlinear margin, retain original eligibility and normalization moments',shadow_full='replace candidate/reference EP by 1; recompute full archived LFP chain'))
if __name__=='__main__':main()
