from common import *
from tiny_support import *
import argparse,concurrent.futures,multiprocessing,pickle,traceback

def main(a):
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 assert a.seed in cfg()['tiny_update']['seeds'], 'Seed was not preregistered'
 assert not (OUT/'audits/TINY_STEP1_COMPLETE.json').exists() or read(OUT/'audits/TINY_STEP1_COMPLETE.json')['status']=='PASS', 'Campaign STOPPED; no automatic retries'
 assert read(OUT/'audits/TINY_LOSS_COMPLETE.json')['status']=='PASS'
 p=initialise();dest=OUT/'tiny'/str(a.seed)/a.arm;dest.mkdir(parents=True,exist_ok=True);assert a.resume or not (dest/'status.json').exists(),'Never rerun a stopped/completed arm or overwrite failure evidence'
 pars=trainables(p);start=state(p);ref_before=state_hash(p.old_policy);manifest=read(OUT/'tiny/shared'/str(a.seed)/'manifest.json');assert state_hash(p)==manifest['state_hash']
 if not a.resume:
  torch.save(dict(state_dict=start,seed=a.seed,arm=a.arm,step=0),dest/'step0.pt');save(dest/'trainable_parameters.json',{n:list(v.shape) for n,v in pars.items()})
 optimizer=torch.optim.AdamW(list(pars.values()),lr=1e-5,betas=(.9,.95),weight_decay=1e-4);scaler=torch.amp.GradScaler('cuda');records=[];updates=0;counts=dict(sampled_candidates=0,nominal_scored_candidates=0,perturbation_scored_candidates=0,shared_first_step_candidates=1024,optimizer_updates=0)
 if a.resume:
  assert read(OUT/'audits/TINY_STEP1_COMPLETE.json')['status']=='PASS'
  saved=torch.load(dest/'step1.pt',map_location='cpu',weights_only=False);assert saved['step']==1
  full=p.state_dict();full.update(saved['state_dict']);p.load_state_dict(full,strict=True);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler']);records=read(dest/'steps.json');counts=read(dest/'cost.json');updates=1
 try:
  with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
   for step in range(2 if a.resume else 1,a.stop_step+1):
    rows=order(a.seed,step);tokens=[r['token'] for r in rows];behavior_hash=state_hash(p);chains=[];trajectories=[];scores=[];shared_hashes=[]
    if step==1:
     for r in rows:
      path=OUT/'tiny/shared'/str(a.seed)/(r['token']+'.pkl');assert sha(path)==manifest['files'][r['token']]
      with path.open('rb') as f:v=pickle.load(f)
      assert v['behavior_state_sha256']==behavior_hash;trajectories.append(v['trajectories']);chains.append(v['chain']);scores.append(v['scored']);shared_hashes.append(sha(path))
    else:
     for r in rows:
      tr,ch=rollout(p,r,a.seed,step);trajectories.append(tr);chains.append(ch)
     counts['sampled_candidates']+=1024
     scores=list(pool.map(scored,rows,trajectories,[a.arm in ['C','D']]*64));counts['nominal_scored_candidates']+=2048;counts['perturbation_scored_candidates']+=8192 if a.arm in ['C','D'] else 0
    assert state_hash(p)==behavior_hash,'Policy changed between rollout and update'
    advantage,original,bonus,mask=credit_batch(scores,tokens,a.arm)
    npz(dest/f'step{step}_rollout_credit.npz',dict(behavior_state_sha256=behavior_hash,tokens=tokens,seed=a.seed,step=step,arm=a.arm,shared_hashes=shared_hashes,on_policy=True),trajectories=np.stack(trajectories),chains=np.stack(chains),advantages=advantage.numpy(),original_advantages=original['advantage'].numpy(),risk_bonus=bonus.numpy(),pair_mask=mask.numpy(),positive_eligible=original['positive_eligible'].numpy(),cost=np.stack([s['risk'] for s in scores]),risk_valid=np.stack([s['risk_valid'] for s in scores]),metrics=np.stack([s['nominal'] for s in scores]),raw_progress=np.stack([s['diagnostics']['raw_progress'] for s in scores]))
    with (dest/f'step{step}_scores.pkl').open('wb') as f:pickle.dump(scores,f)
    optimizer.zero_grad(set_to_none=True);losses=[]
    for i,(r,ch) in enumerate(zip(rows,chains)):
     loss,diag=loss_one(p,r['token'],ch,advantage[i]);scaler.scale(loss/64).backward();losses.append(diag)
     if i%16==0:print('BACKWARD',a.seed,a.arm,step,i+1,diag,flush=True)
    scaler.unscale_(optimizer)
    if not all(torch.isfinite(v.grad).all() for v in pars.values() if v.grad is not None):raise FloatingPointError('Nonfinite unscaled gradient; STOP before optimizer.step (no reduced-scale rerun)')
    gradnorm=torch.nn.utils.clip_grad_norm_(list(pars.values()),1.,error_if_nonfinite=True)
    scaler.step(optimizer);scaler.update();updates+=1;counts['optimizer_updates']=updates
    current=state(p)
    # Persist the actual post-update artifact before diagnostics can fail.
    if step in [1,8]:torch.save(dict(state_dict=current,optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),seed=a.seed,arm=a.arm,step=step),dest/f'step{step}.pt')
    assert all(torch.isfinite(current[n]).all() for n in pars), 'Nonfinite updated trainable parameter'
    displacement=float(torch.sqrt(sum((current[n].double()-start[n].double()).square().sum() for n in pars)))
    record=dict(step=step,optimizer_updates=updates,behavior_hash=behavior_hash,new_policy_hash=state_hash(p),loss_policy=float(np.mean([x['policy'] for x in losses])),KL=float(np.mean([x['kl'] for x in losses])),BC=0.,gradient_norm_before_clip=float(gradnorm),parameter_displacement=displacement,bonus_l1=float(bonus.abs().sum()),bonus_positive_mass=float(bonus.clamp(min=0).sum()),risk_information_active_groups=int((bonus.abs()>1e-8).any(1).sum()),original_information_active_groups=int((original['advantage'].abs()>1e-8).any(1).sum()),matching_group_fraction=float(mask.any(-1).any(-1).float().mean()),risk_invalid_candidates=int(sum((~s['risk_valid']).sum() for s in scores)) if a.arm in ['C','D'] else 0,scaler_scale=scaler.get_scale(),lr=optimizer.param_groups[0]['lr'])
    records.append(record);save(dest/'steps.json',records);save(dest/'cost.json',counts)
    assert state_hash(p.old_policy)==ref_before,'Frozen reference changed'
    print('STEP COMPLETE',a.seed,a.arm,step,record,flush=True)
  save(dest/'status.json',dict(status='PASS',completed_steps=updates,main_endpoint_reached=updates==8,cost=counts))
 except Exception:
  failure=traceback.format_exc();save(dest/'status.json',dict(status='STOPPED',completed_steps=updates,main_endpoint_reached=False,cost=counts,error=failure));(dest/'failure.log').write_text(failure);print(failure,flush=True);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--arm',choices=list('ABCD'),required=True);p.add_argument('--stop-step',type=int,default=8);p.add_argument('--resume',action='store_true');main(p.parse_args())
