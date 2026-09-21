from common import *
import torch,copy,dataclasses
from sample import load,inputs,native
from credit import setup,scalar,stages
from pairwise_credit import require_update_gate,risk_credit,combine,validate_trainable_identity
from scoring import score,load_cache,NAMES
from perturb import deform,parameters,dynamics_ok

def initialise():
 require_update_gate(read(OUT/'audits/SMOKE_COMPLETE.json'),read(OUT/'audits/signal_gate.json'))
 p,m=load('a5_sft',readonly=False)
 assert not p.eta.eta_logit.requires_grad, 'Archived EtaFixed must remain frozen'
 assert all(torch.isfinite(v).all() for _,v in p.named_parameters() if v.requires_grad), 'Nonfinite trainable parameter'
 from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
 for n,v in p.named_parameters():
  if ReCogDriveAgent._is_last_vla_condition_parameter_key('action_head.'+n):v.requires_grad_(False)
 from navsim.agents.recogdrive.recogdrive_diffusion_planner import GRPOConfig
 g=GRPOConfig();a=m['grpo_config']
 for k in vars(g):
  if 'grpo_'+k in a:setattr(g,k,a['grpo_'+k])
  elif k in ['metric_cache_path','reference_policy_checkpoint','bc_anneal','bc_coeff_start','bc_coeff_end','bc_anneal_epochs','reference_kl_coeff','reference_kl_chunk_size'] and k in a:setattr(g,k,a[k])
 g.sample_time=16;p._init_stage3_runtime(g);p.lfp_grpo_cfg,_=setup();validate_trainable_identity(p,p._quickcheck_constructor_trainable_names);p.old_policy=copy.deepcopy(p).requires_grad_(False).eval();p.train();p.set_frozen_modules_to_eval_mode()
 return p

def state(p):return {k:v.detach().cpu().clone() for k,v in p.state_dict().items() if not k.startswith('old_policy.')}
def trainables(p):return {k:v for k,v in p.named_parameters() if v.requires_grad}
def order(seedno,step):
 epoch=(step-1)//4;offset=((step-1)%4)*64
 return sorted(scenes(split='calibration'),key=lambda r:digest(['tiny_train_order',seedno,epoch,r['token']]))[offset:offset+64]
def rollout(p,row,seedno,step):
 vl,action=inputs(row['token']);torch.manual_seed(seed('tiny_behavior',seedno,step,row['token']));d=native(p,vl,action,row['token'])
 return d['trajectory'].float().cpu().numpy(),d['chain'].float().cpu().numpy()

def scored(row,tr,needrisk):
 cache=load_cache(row);nom,diag,_=score(cache,tr);train,_,_=score(cache,tr,training=True,fast=True)
 assert np.max(abs(train[:,-1]-scalar(nom[:,0],nom[:,1],nom[:,2],nom[:,3],nom[:,4])))<=1e-8
 risk=np.zeros(len(tr));valid=np.zeros(len(tr),bool);pert=[];pertrows=[];pm=None;pdg=None
 if needrisk:
  for traj in tr:
   for lat,lon in parameters(row['token'],'training_v1'):pert.append(deform(traj,lat,lon,cache.ego_state.dynamic_car_state.speed))
  pm,pdg,_=score(copy.deepcopy(cache),np.stack(pert));base={k:np.repeat(diag[k],8) for k in ['peak_speed','peak_acceleration','peak_jerk','max_yaw_rate']};ok=dynamics_ok(pdg,base)
  assert np.isfinite(pm).all(),'Nonfinite training perturbation score'
  valid=ok.reshape(-1,8).all(1);risk=np.any(pm[:,[0,1,3]]<1,axis=1).reshape(-1,8).mean(1)
 return dict(nominal=nom,training_scalar=train[:,-1],diagnostics=diag,risk=risk,risk_valid=valid,perturbation_scores=pm,perturbation_diagnostics=pdg)

def credit_batch(scored_rows,tokens,arm):
 c,refs=setup();adapter=__import__('navsim.agents.recogdrive.stage3_metric_adapter',fromlist=['Stage3MetricAdapter']).Stage3MetricAdapter('navsim_v1');v=np.stack([x['nominal'] for x in scored_rows]);names=['nc','dac','ep','ttc','comfort','ddc','official_pdms'];comp={n:torch.tensor(v[...,i],dtype=torch.float32) for i,n in enumerate(names)};comp['pdms']=torch.tensor(np.stack([x['training_scalar'] for x in scored_rows]),dtype=torch.float32);m=adapter.canonicalize(comp,batch_size=len(tokens),group_size=16);r=refs.get(tokens,'cpu',torch.float32)
 if arm=='B':m=dataclasses.replace(m,scalar=scalar(m.nc,m.dac,m.ep,m.ttc,m.quality,.5));r=dataclasses.replace(r,scalar=scalar(r.nc,r.dac,r.ep,r.ttc,r.quality,.5))
 out=stages(m,r,c);a=out['advantage'];bonus=torch.zeros_like(a);mask=torch.zeros(*a.shape,a.shape[-1],dtype=torch.bool)
 if arm in ['C','D']:
  cost=torch.tensor(np.stack([x['risk'] for x in scored_rows]),dtype=torch.float32);valid=torch.tensor(np.stack([x['risk_valid'] for x in scored_rows]));safe=(m.nc==1)&(m.dac==1)&(m.ttc==1)&out['feasible']&valid
  progress=torch.tensor(np.stack([x['diagnostics']['raw_progress'] for x in scored_rows]),dtype=torch.float32) if arm=='D' else None
  bonus,mask=risk_credit(cost,safe,progress,.5);a,extra=combine(a,bonus,out['positive_eligible'],c.advantage_clip,1.)
 return a,out,bonus,mask

def loss_one(p,token,chain,adv):
 vl,action=inputs(token);vl=vl.expand(16,-1,-1);a=action(16);ch=torch.tensor(chain,device='cuda');p.train();p.set_frozen_modules_to_eval_mode();p.old_policy.eval()
 with torch.autocast('cuda',dtype=torch.float16):
  ctx=p._prepare_dit_context(vl,a,training=False,allow_target_tokens=False)
  lp=p.get_logprobs(vl,a.his_traj,a.status_feature,ch,action_input=a,prepared_dit_context=ctx)
  discount=p._stage3_discount(5,device=ch.device,dtype=torch.float32);reduced=p._reduce_chain_logprobs(lp,1,16,5,discount)
  with torch.no_grad():rctx=p.old_policy._prepare_dit_context(vl,a,training=False,allow_target_tokens=False)
  kl=p._chain_transition_reference_kl(vl,a.his_traj,a.status_feature,ch,a,1,16,5,discount,current_dit_context=ctx,reference_dit_context=rctx)
  from navsim.agents.recogdrive.stage3_lfp_grpo import trajectory_reinforce_loss
  policy=trajectory_reinforce_loss(adv[None].to(reduced.device),reduced);loss=policy+.005*kl
 assert torch.isfinite(lp).all() and torch.isfinite(loss),'Nonfinite log-prob/loss'
 return loss,dict(policy=float(policy.detach()),kl=float(kl.detach()),logprob_mean=float(reduced.detach().mean()),logprob_min=float(lp.detach().min()),logprob_max=float(lp.detach().max()))
