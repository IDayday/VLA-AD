from common import *
import argparse,inspect,torch,contextlib
from transformers.feature_extraction_utils import BatchFeature
install_runtime()
class Captured(Exception):pass

def load(name,readonly=True):
 models=read(OUT/'manifests/models.json');m=models[name]
 sys.path.insert(0,str(ROOT/'scripts/evaluation/distribution_audit'))
 from run import load_planner
 p=load_planner(dict(m,path=m['checkpoint_path'],config=dict(m['config'],grpo=False)))
 p._quickcheck_constructor_trainable_names=tuple(n for n,v in p.named_parameters() if v.requires_grad)
 if readonly:p.requires_grad_(False)
 module=sys.modules[type(p).__module__];g=module.GRPOConfig();a=m['grpo_config']
 for key in vars(g):
  if 'grpo_'+key in a:setattr(g,key,a['grpo_'+key])
 for key in ['denoised_clip_value','eval_randn_clip_value','randn_clip_value','final_action_clip_value','eps_clip_value','eval_min_sampling_denoising_std','min_sampling_denoising_std','min_logprob_denoising_std','gamma_denoising']:
  if hasattr(g,key):setattr(p,key,getattr(g,key))
 p.grpo_sample_time=a['grpo_sample_time'];p.stage3_algorithm=a['stage3_algorithm'];p.lfp_reference_cache=object();p.lfp_metric_adapter=object()
 assert p.stage3_algorithm=='lfp_grpo' and p.grpo_sample_time==16
 assert str(Path(inspect.getfile(type(p))).resolve())==m['runtime_planner_path']
 p.train();p.set_frozen_modules_to_eval_mode();return p,m

def inputs(token):
 f=torch.load(OUT/'cache/features'/(token+'.pt'),map_location='cpu',weights_only=False)
 assert f['_metadata']['observation_only'] and f['_metadata']['precision']=='torch.float32'
 assert f['_metadata']['vlm_state_sha256']==read(OUT/'audits/features_smoke_0.json')['state_before']
 assert f['_metadata']['feature_builder_sha256']==sha(RUNTIME/'navsim/agents/recogdrive/recogdrive_features.py')
 vl=f['last_hidden_state'][None].cuda().float();h=f['history_trajectory'].reshape(1,-1).cuda().float();s=f['status_feature'][None].cuda().float();c=f['high_command_one_hot'][None].cuda().float()
 def action(n):return BatchFeature(data={'his_traj':h.expand(n,-1),'history_trajectory':h.reshape(1,4,3).expand(n,-1,-1),'status_feature':s.expand(n,-1),'high_command_one_hot':c.expand(n,-1),'state':torch.cat([s,h],1).expand(n,-1)})
 return vl,action

def native(p,vl,action,token):
 method=p.sample_chain;found={}
 def hook(*a,**kw):
  chain,traj=method(*a,**kw);found.update(chain=chain,trajectory=traj,args=a,kwargs=kw);raise Captured()
 p.train();p.set_frozen_modules_to_eval_mode();p.sample_chain=hook
 try:
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):p.forward_grpo(vl,action(1),[token],sample_time=16)
 except Captured:pass
 finally:p.sample_chain=method
 assert found['trajectory'].shape==(16,8,3);return found

def draw(p,vl,action,t,g,protocol):
 torch.manual_seed(seed(cfg()['seed'],t,protocol,g))
 if protocol=='native_grpo':
  d=native(p,vl,action,t);return d['trajectory'].float().cpu().numpy(),d['chain'].float().cpu().numpy(),d
 p.eval()
 with torch.no_grad():traj=p.get_action(vl.expand(16,-1,-1),action(16),deterministic=False)['pred_traj']
 return traj.float().cpu().numpy(),None,None

def main(a):
 forbid_updates()
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 p,m=load(a.model);before=state_hash(p);checks=[];rows=scenes(smoke=a.smoke)[a.rank::a.world]
 if not a.smoke:assert read(OUT/'audits/SMOKE_COMPLETE.json')['status']=='PASS'
 started=time.time()
 for i,r in enumerate(rows):
  t=r['token'];dst=OUT/'cache/rollouts'/a.model/(t+'.npz')
  metadata=dict(token=t,checkpoint_sha256=m['sha256'],runtime_sha256=m['runtime_planner_sha256'],feature_sha256=sha(OUT/'cache/features'/(t+'.pt')),config_sha256=sha(OUT/'resolved_config.yaml'),group_size=16,groups=2,protocols=['native_grpo','deployment'])
  if dst.exists():
   assert json.loads(str(np.load(dst)['metadata']))==metadata;continue
  vl,action=inputs(t);arrays={}
  for protocol in ['native_grpo','deployment']:
   trajectories=[];chains=[]
   for g in range(2):
    tr,ch,detail=draw(p,vl,action,t,g,protocol);trajectories.append(tr)
    if ch is not None:chains.append(ch)
    if a.smoke and g==0:
     again,chain2,_=draw(p,vl,action,t,g,protocol);err=float(np.max(abs(tr-again)));assert err==0,(t,protocol,err)
     if protocol=='native_grpo':
      torch.manual_seed(seed(cfg()['seed'],t,protocol,g))
      with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):manual_chain,manual_traj=p.sample_chain(*detail['args'],**detail['kwargs'])
      capture_err=float(np.max(abs(tr-manual_traj.float().cpu().numpy())));assert capture_err==0 and torch.equal(detail['chain'],manual_chain)
     else:capture_err=None
     checks.append(dict(token=t,protocol=protocol,repeat_error=err,native_entry_capture_error=capture_err))
   arrays[protocol]=np.stack(trajectories)
   if chains:arrays['native_chain']=np.stack(chains)
  assert all(np.isfinite(x).all() for x in arrays.values())
  npz(dst,metadata,**arrays)
  if i%8==0:print('SAMPLE',a.model,i+1,len(rows),time.time()-started,flush=True)
 p.train();p.set_frozen_modules_to_eval_mode();after=state_hash(p);assert before==after
 save(OUT/'audits'/f"sampling_{a.model}_{'smoke' if a.smoke else 'formal'}_{a.rank}.json",dict(status='PASS',state_before=before,state_after=after,checks=checks,optimizer_steps=0,backward_calls=0,group_size=16,actual_entry='forward_grpo -> forward_lfp_grpo -> sample_chain',runtime_sha256=m['runtime_planner_sha256'],scenes=len(rows),native_precision='autocast_float16',deployment_precision='float32',seconds=time.time()-started))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1);main(p.parse_args())
