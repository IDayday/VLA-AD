from common import *
import torch,inspect
from tiny_support import initialise,trainables,state

def main():
 torch.set_num_threads(1);p=initialise();before=state_hash(p);names=sorted(trainables(p));constructor=p._quickcheck_constructor_trainable_names
 assert 'eta.eta_logit' not in names and not p.eta.eta_logit.requires_grad
 assert all(torch.isfinite(v).all() for v in trainables(p).values())
 assert set(names)<=set(constructor)
 # Read-only check against actual archived constructor, not an invented mask.
 module=__import__(type(p).__module__,fromlist=['EtaFixed']);eta_source=inspect.getsource(type(p)._init_ddim_sampler);assert 'param.requires_grad = False' in eta_source
 save(OUT/'audits/CORRECTED_TRAINABLE_IDENTITY_READONLY.json',dict(status='PASS',trainable_parameter_tensors=len(names),trainable_parameter_names=names,constructor_trainable_parameter_names=list(constructor),eta_requires_grad=p.eta.eta_logit.requires_grad,eta_value='positive_infinity (archived fixed sampler encoding)',eta_effective=float(p.eta(torch.zeros(1,8,3,device='cuda')).item()),state_before=before,state_after=state_hash(p),optimizer_steps=0,note='Implementation fix validated without rerunning any stopped arm; previous eight updates remain invalid for comparisons'))
 rows=[]
 for seedno in cfg()['tiny_update']['seeds']:
  for arm in 'ABCD':
   d=OUT/'tiny'/str(seedno)/arm;s=read(d/'status.json');old=read(d/'trainable_parameters.json');extras=sorted(set(old)-set(names));assert 'eta.eta_logit' in extras
   rows.append(dict(seed=seedno,arm=arm,status='INVALID_STOPPED',optimizer_updates=s['completed_steps'],unexpected_trainable_tensors=extras,step0_exists=(d/'step0.pt').exists(),step1_saved=(d/'step1.pt').exists(),step8='NOT_RUN',step1_evaluation='NOT_RUN_MISSING_POSTUPDATE_CHECKPOINT',norm_control='NOT_RUN',failure_file=str(d/'failure.log')))
 save(OUT/'audits/TINY_STEP1_COMPLETE.json',dict(status='FAIL',reason='Trainable identity mismatch: 105 constructor-frozen parameter tensors inadvertently unfrozen (EtaFixed plus 104 inactive planning-branch tensors); NaN diagnostic displacement from inf-inf; no evidence establishes model divergence',arms=rows,total_optimizer_updates=sum(r['optimizer_updates'] for r in rows),no_restart=True,step8='NOT_RUN',post_update_checkpoint_missing='The first implementation wrote checkpoint after JSON diagnostics; crash occurred before saving. No post-update weights recovered or fabricated.',next_only='Validate corrected trainable-parameter identity, frozen-state invariance and checkpoint-first persistence in one shared-rollout single-step comparison before considering any further updates'))
 pd.DataFrame(rows).to_csv(OUT/'tiny_update_status.csv',index=False)
 print('CORRECTED READONLY IDENTITY PASS',len(names),'STOPPED UPDATES',sum(r['optimizer_updates'] for r in rows))
if __name__=='__main__':main()
