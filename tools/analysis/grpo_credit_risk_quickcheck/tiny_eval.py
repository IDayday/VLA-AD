from common import *
from tiny_support import initialise,state,trainables
from sample import inputs,draw
from scoring import score,load_cache,NAMES
from perturb import deform,parameters,dynamics_ok
import argparse,concurrent.futures,multiprocessing,traceback,torch,copy

def score_evaluation(task):
 row,path,dest=task;data=np.load(path);cache=load_cache(row);nomtr=np.concatenate([data['native_grpo'].reshape(-1,8,3),data['deployment'].reshape(-1,8,3)]);before=sha(row['metric_cache_path']);nom,diag,states=score(cache,nomtr)
 altered=np.stack([deform(tr,lat,lon,cache.ego_state.dynamic_car_state.speed) for tr in nomtr for lat,lon in parameters(row['token'],'evaluation_v1')]);pert,pdiag,_=score(copy.deepcopy(cache),altered);baseline={k:np.repeat(diag[k],8) for k in ['peak_speed','peak_acceleration','peak_jerk','max_yaw_rate']};valid=dynamics_ok(pdiag,baseline).reshape(64,8);risk=np.any(pert[:,[0,1,3]]<1,axis=1).reshape(64,8);rows=[]
 from credit import setup
 c,refs=setup();ddc=refs.records[row['token']]['gt_ddc']
 for i in range(64):
  v=dict(token=row['token'],log=row['log'],protocol='native_grpo' if i<32 else 'deployment',group=(i%32)//16,candidate=i%16,**dict(zip(NAMES,nom[i])),**{k:x[i] for k,x in diag.items()},risk_valid_draws=int(valid[i].sum()),risk=float(risk[i].mean()) if valid[i].all() else np.nan,NC_fail=float((pert.reshape(64,8,7)[i,:,0]<1).mean()) if valid[i].all() else np.nan,DAC_fail=float((pert.reshape(64,8,7)[i,:,1]<1).mean()) if valid[i].all() else np.nan,TTC_fail=float((pert.reshape(64,8,7)[i,:,3]<1).mean()) if valid[i].all() else np.nan)
  v['nominal_failure']=float(np.any(nom[i,[0,1,3]]<1));v['zero_score']=float(nom[i,-1]==0);v['pdms_below50']=float(nom[i,-1]<.5);v['pdms_below95']=float(nom[i,-1]<.95);v['reference_gt_ddc']=ddc;rows.append(v)
 for first in range(0,64,16):
  q=rows[first:first+16];safe=np.array([v['NC']==1 and v['DAC']==1 and v['TTC']==1 and v['DDC']>=ddc-.01 for v in q]);d=diag['raw_progress'][first:first+16];cost=np.array([v['risk'] for v in q]);mask=safe[:,None]&safe[None,:]&(abs(d[:,None]-d[None,:])<=.5)&~np.eye(16,dtype=bool);active=mask & np.isfinite(cost[:,None]) & np.isfinite(cost[None,:]) & (abs(cost[:,None]-cost[None,:])>0)
  high=np.array([v['PDMS']>=.95 and not v['low_progress_branch'] for v in q]);matches=mask&high[:,None]&high[None,:]&(abs(nom[first:first+16,-1][:,None]-nom[first:first+16,-1][None,:])<=.01)
  for v in q:v['nominal_match_active']=bool(matches.any());v['risk_comparison_active']=bool(active.any())
 dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_parquet(dest/(row['token']+'.parquet'),index=False)
 npz(dest/(row['token']+'_perturbations.npz'),dict(token=row['token'],rollout_sha256=sha(path),metric_cache_sha256=before,stream='evaluation_v1'),metrics=pert.reshape(64,8,7),dynamics_valid=valid,raw_progress=pdiag['raw_progress'].reshape(64,8),nominal_states=states)
 assert sha(row['metric_cache_path'])==before
 return row['token']

def main(a):
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 dest=OUT/'tiny/evaluation'/a.label;dest.mkdir(parents=True,exist_ok=True);rows=scenes(split='confirmation')
 if not a.score_only:
  p=initialise();weights=torch.load(a.checkpoint,map_location='cpu',weights_only=False)['state_dict'];full=p.state_dict();full.update(weights);p.load_state_dict(full,strict=True);p.requires_grad_(False);before=state_hash(p);pred=dest/'rollouts';pred.mkdir(exist_ok=True)
  for i,r in enumerate(rows):
   vl,act=inputs(r['token']);arrays={}
   for protocol in ['native_grpo','deployment']:arrays[protocol]=np.stack([draw(p,vl,act,r['token'],g,protocol)[0] for g in range(2)])
   npz(pred/(r['token']+'.npz'),dict(checkpoint_sha256=sha(a.checkpoint),state_sha256=before,feature_sha256=sha(OUT/'cache/features'/(r['token']+'.pt')),config_sha256=sha(OUT/'resolved_config.yaml')),**arrays)
   if i%32==0:print('EXPORT',a.label,i+1,flush=True)
  assert state_hash(p)==before;save(dest/'export.json',dict(status='PASS',scenes=256,candidates=16384,checkpoint=str(a.checkpoint),checkpoint_sha256=sha(a.checkpoint),state_before=before,state_after=state_hash(p)))
  del p;torch.cuda.empty_cache()
 if a.export_only:return
 tasks=[(r,dest/'rollouts'/(r['token']+'.npz'),dest/'rows') for r in rows];done=[];errors=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
  futures=[pool.submit(score_evaluation,t) for t in tasks]
  for f in concurrent.futures.as_completed(futures):
   try:done.append(f.result())
   except Exception:errors.append(traceback.format_exc());print(errors[-1],flush=True)
   if len(done)%32==0:print('EVAL SCORE',a.label,len(done),flush=True)
 save(dest/'scoring.json',dict(status='PASS' if not errors else 'FAIL',scenes=len(done),nominal_candidates=len(done)*64,perturbation_candidates=len(done)*512,errors=errors));assert not errors
 pd.concat([pd.read_parquet(dest/'rows'/(r['token']+'.parquet')) for r in rows],ignore_index=True).to_parquet(dest/'trajectories.parquet',index=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--checkpoint');p.add_argument('--export-only',action='store_true');p.add_argument('--score-only',action='store_true');p.add_argument('--workers',type=int,default=8);main(p.parse_args())
