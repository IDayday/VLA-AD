from common import *
from scoring import *
from perturb import deform,parameters,dynamics_ok
import argparse,concurrent.futures,traceback,copy
MARGIN_FIELDS=['object_clearance_m','object_closest_time_s','object_closest_token','geometric_overlap','closest_object_ignored_initial_collision','road_margin_m','road_closest_time_s','footprint_outside_drivable']

def process(row,pairrows,source):
 t=row['token'];metric_sha=sha(row['metric_cache_path']);cache=load_cache(row)
 if source=='native_within_group':bankpath=OUT/'cache/rollouts/a5_sft'/(t+'.npz');bank=np.load(bankpath)['native_grpo'].reshape(-1,8,3)
 else:bankpath=OUT/'cache/constructed'/(t+'.npz');bank=np.load(bankpath)['trajectories']
 ids=sorted({p[k] for p in pairrows for k in ['candidate_a','candidate_b']});tr=bank[ids];nom,diag,states=score(cache,tr,margins=True)
 margins=[];results=[];error=[];initial_speed=cache.ego_state.dynamic_car_state.speed;deformed=[];labels=[]
 for index,cid in enumerate(ids):
  ident=dict(token=t,log=row['log'],split=row['split'],source=source,candidate=cid,model='a5_sft' if source=='native_within_group' else 'external',protocol='native_grpo' if source=='native_within_group' else 'constructed')
  m=dict(ident,**dict(zip(NAMES,nom[index])),**{k:v[index] for k,v in diag.items()});margins.append(m)
  for stream in ['selection_v1','evaluation_v1']:
   for k,(lat,lon) in enumerate(parameters(t,stream)):
    info=dict(ident,stream=stream,draw=k,seed=seed(cfg()['seed'],t,stream),lateral_m=lat,longitudinal_m=lon,metric_cache_sha256=metric_sha,trajectory_cache_sha256=sha(bankpath),nominal_raw_progress=diag['raw_progress'][index])
    try:
     altered=deform(tr[index],lat,lon,initial_speed);assert np.isfinite(altered).all();labels.append((info,index));deformed.append(altered)
    except Exception:results.append(dict(info,status='GENERATION_FAILED',error=traceback.format_exc()))
 if deformed:
  # All original cache/map/traffic/baseline values unchanged; score owns mutable state.
  try:
   scored,pert_diag,pert_states=score(copy.deepcopy(cache),np.stack(deformed))
   baseline={key:np.asarray([diag[key][idx] for _,idx in labels]) for key in ['peak_speed','peak_acceleration','peak_jerk','max_yaw_rate']};valid=dynamics_ok(pert_diag,baseline)
   npz(OUT/'cache/perturbed'/source/(t+'.npz'),dict(metric_cache_sha256=metric_sha,trajectory_cache_sha256=sha(bankpath)),trajectories=np.stack(deformed),simulated_states=pert_states)
   for i,(info,_) in enumerate(labels):
    finite=np.isfinite(scored[i]).all();status='OK' if finite and valid[i] else 'SCORER_NONFINITE' if not finite else 'DYNAMICS_INVALID'
    record=dict(info,status=status,**dict(zip(NAMES,scored[i])),**{k:v[i] for k,v in pert_diag.items()})
    if status=='OK':record.update(risk=float(np.any(scored[i,[0,1,3]]<1)),NC_fail=float(scored[i,0]<1),DAC_fail=float(scored[i,1]<1),TTC_fail=float(scored[i,3]<1),hard_fail=float(scored[i,-1]==0))
    results.append(record)
  except Exception:
   for info,_ in labels:results.append(dict(info,status='SCORER_EXCEPTION',error=traceback.format_exc()))
 assert sha(row['metric_cache_path'])==metric_sha
 out=OUT/'cache/risk_rows'/source;out.mkdir(parents=True,exist_ok=True);pd.DataFrame(results).to_parquet(out/(t+'.parquet'),index=False);pd.DataFrame(margins).to_parquet(out/(t+'_margins.parquet'),index=False)
 return dict(token=t,total=len(results),valid=sum(r['status']=='OK' for r in results),failures=[{k:r[k] for k in ['candidate','stream','draw','status']} for r in results if r['status']!='OK'])

def main(a):
 pairs=pd.read_parquet(OUT/('matched_pairs.parquet' if a.source=='native_within_group' else 'constructed_pairs.parquet'));subset=pairs[pairs.split==a.split]
 if a.split=='confirmation':assert read(OUT/'perturbation_manifest.json')['calibration_status']=='PASS'
 lookup={r['token']:r for r in scenes()};tasks=[(lookup[t],b.to_dict('records'),a.source) for t,b in subset.groupby('token')];status=[];failures=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
  futures=[pool.submit(process,*x) for x in tasks]
  for f in concurrent.futures.as_completed(futures):
   try:status.append(f.result())
   except Exception:failures.append(traceback.format_exc())
   if len(status)%16==0:print('RISK',a.source,a.split,len(status),len(tasks),flush=True)
 save(OUT/'audits'/f'risk_{a.source}_{a.split}.json',dict(status='PASS' if not failures else 'FAIL',scenes=status,errors=failures));assert not failures
 if a.split=='calibration' and a.source=='native_within_group':
  total=sum(x['total'] for x in status);valid=sum(x['valid'] for x in status);passed=total>0 and valid/total>=.99
  manifest=dict(calibration_status='PASS' if passed else 'BLOCKED',generation_and_dynamics_valid=valid,total=total,valid_fraction=valid/total if total else None,
   lateral_max_m=.05,longitudinal_max_m=.20,choice_uses_risk_events=False,basis=cfg()['perturbations']['basis'],config_sha256=sha(OUT/'resolved_config.yaml'),pair_manifest_sha256=sha(OUT/'matched_pairs.parquet'),
   parameters={r['token']:{stream:dict(seed=seed(cfg()['seed'],r['token'],stream),offsets=parameters(r['token'],stream).tolist()) for stream in ['selection_v1','evaluation_v1','training_v1']} for r in scenes()},
   semantics='Engineering trajectory-neighbourhood sensitivity, not measured uncertainty distribution or closed-loop accident probability',dynamics=cfg()['perturbations']['dynamics'],dyn_invalid_policy='record separately; candidate pair requires all 16 valid draws for selector comparison; no relabeling invalid as collision',initial_state='unchanged; b(0)=bprime(0)=0; simulator starts same ego state',baseline='same immutable PDM baseline and cache; deepcopy per perturbation score batch',heading='CubicHermite path tangent plus analytic derivative of Frenet displacement; endpoints displacement zero')
  save(OUT/'perturbation_manifest.json',manifest)
 # Rebuild aggregates from existing per-scene files, retaining all failure rows.
 files=list((OUT/'cache/risk_rows').glob('*/*.parquet'));risks=[p for p in files if not p.stem.endswith('_margins')];margins=[p for p in files if p.stem.endswith('_margins')]
 if risks:pd.concat([pd.read_parquet(p) for p in risks],ignore_index=True).to_parquet(OUT/'perturbation_results.parquet',index=False)
 if margins:pd.concat([pd.read_parquet(p) for p in margins],ignore_index=True).to_parquet(OUT/'nominal_margins.parquet',index=False)
 print('COMPLETE',a.split,len(status),sum(x['valid'] for x in status),sum(x['total'] for x in status))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['calibration','confirmation'],required=True);p.add_argument('--source',default='native_within_group');p.add_argument('--workers',type=int,default=16);main(p.parse_args())
