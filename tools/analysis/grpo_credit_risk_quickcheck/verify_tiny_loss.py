from common import *
from tiny_support import *
import pickle,types
from navsim.agents.recogdrive.stage3_lfp_grpo import compute_lfp_advantages as actual_adv
from navsim.agents.recogdrive.stage3_metric_adapter import Stage3MetricAdapter

def main():
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 p=initialise();before=state_hash(p);seedno=cfg()['tiny_update']['seeds'][0];rows=order(seedno,1);scores=[];chains=[]
 for r in rows:
  with (OUT/'tiny/shared'/str(seedno)/(r['token']+'.pkl')).open('rb') as f:v=pickle.load(f)
  scores.append(v['scored']);chains.append(v['chain'])
 c,refs=setup();v=np.stack([s['nominal'] for s in scores]);comp={n:torch.tensor(v[...,i],device='cuda',dtype=torch.float32) for i,n in enumerate(['nc','dac','ep','ttc','comfort','ddc','official_pdms'])};comp['pdms']=torch.tensor(np.stack([s['training_scalar'] for s in scores]),device='cuda',dtype=torch.float32);m=Stage3MetricAdapter('navsim_v1').canonicalize(comp,batch_size=64,group_size=16);ref=refs.get([r['token'] for r in rows],'cuda',torch.float32);ad=actual_adv(m,ref,c)
 # Same fixed global moment batch; select a nonzero-credit scene without looking
 # at later performance, solely to make loss-equivalence test informative.
 idx=int(ad.advantages.abs().sum(1).argmax());r=rows[idx];vl,action=inputs(r['token']);micro=dataclasses.replace(m,**{k:getattr(m,k)[idx:idx+1] for k in ['scalar','nc','dac','ep','ttc','quality','ddc_guard_value']})
 selected=dataclasses.replace(ad,**{k:getattr(ad,k)[idx:idx+1] for k in ['advantages','pareto_front','feasible','progress_ok','ttc_ok','energy']})
 module=__import__(type(p).__module__,fromlist=['compute_lfp_advantages']);old_adv=module.compute_lfp_advantages;old_sample=p.sample_chain;old_eval=p._evaluate_lfp_rollouts
 p.lfp_reference_cache=refs;p.lfp_metric_adapter=Stage3MetricAdapter('navsim_v1');p.metric_cache_loader=types.SimpleNamespace(metric_cache_paths={r['token']:Path(r['metric_cache_path'])});ch=torch.tensor(chains[idx],device='cuda')
 # Scoring is replaced by cached real rollout metrics. The returned trajectory
 # value is unused; use its actual shared-cache value, never synthetic scoring.
 with (OUT/'tiny/shared'/str(seedno)/(r['token']+'.pkl')).open('rb') as f:tr=torch.tensor(pickle.load(f)['trajectories'],device='cuda')
 module.compute_lfp_advantages=lambda *a,**k:selected;p.sample_chain=lambda *a,**k:(ch,tr);p._evaluate_lfp_rollouts=lambda *a,**k:micro
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):native_result=p.forward_grpo(vl,action(1),[r['token']],sample_time=16)
 module.compute_lfp_advantages=old_adv;p.sample_chain=old_sample;p._evaluate_lfp_rollouts=old_eval
 with torch.no_grad():direct,diag=loss_one(p,r['token'],chains[idx],ad.advantages[idx].cpu())
 error=float(abs(native_result.loss-direct));assert error<=1e-8,(error,native_result.loss,direct)
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
  vl16=vl.expand(16,-1,-1);a=action(16);ctx=p._prepare_dit_context(vl16,a,training=False,allow_target_tokens=False);dist=p._chain_transition_distribution(vl16,a.his_traj,a.status_feature,ch,action_input=a,prepared_dit_context=ctx);lp=p.get_logprobs(vl16,a.his_traj,a.status_feature,ch,action_input=a,prepared_dit_context=ctx);target=ch[:,1:].reshape(-1,8,3);manual=-(target-dist.loc).square()/(2*dist.scale.square())-torch.log(dist.scale)-.5*np.log(2*np.pi);lp_error=float((lp-manual).abs().max())
 assert lp_error<=1e-5 and torch.isfinite(lp).all(),lp_error
 assert state_hash(p)==before
 save(OUT/'audits/TINY_LOSS_COMPLETE.json',dict(status='PASS',token=r['token'],loss_equivalence_error=error,logprob_formula_error=lp_error,loss_tolerance=1e-8,logprob_float32_tolerance=1e-5,original_KL=diag['kl'],original_loss=float(direct),unchanged_state=True,optimizer_steps=0,comparison='real archived forward_grpo loss with same cached real actions and same globally computed original advantage vs extracted native loss/KL calls'))
 print('TINY LOSS PASS',error,lp_error,diag)
if __name__=='__main__':main()
