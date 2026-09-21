from common import *
from scoring import *
import argparse,concurrent.futures,traceback

def work(row,smoke):
 token=row['token'];before=sha(row['metric_cache_path']);cache=load_cache(row);rows=[];checks=[]
 for model in cfg()['models']:
  source=OUT/'cache/rollouts'/model/(token+'.npz');bank=np.load(source);nom={}
  for protocol in ['native_grpo','deployment']:
   tr=bank[protocol].reshape(-1,8,3);metrics,diag,states=score(cache,tr)
   scalar=metrics[:,0]*metrics[:,1]*(10*metrics[:,2]+5*metrics[:,3]+2*metrics[:,4])/17
   actual,_,_=score(cache,tr,training=True,fast=True)
   assert np.max(abs(actual[:,-1]-scalar))<=1e-8
   if smoke:checks.append(dict(model=model,protocol=protocol,checks=parity(row,tr[:4])))
   for i,v in enumerate(metrics):
    record=dict(token=token,log=row['log'],split=row['split'],command=row['command'],model=model,protocol=protocol,group=i//16,candidate=i%16,seed=seed(cfg()['seed'],token,protocol,i//16),training_scalar=float(scalar[i]),scene_weight=1.,group_weight=1.,trajectory_cache=str(source))
    record.update(dict(zip(NAMES,v)));record.update({k:x[i] for k,x in diag.items()});rows.append(record)
   nom[protocol+'_states']=states.reshape(2,16,41,11)
  npz(OUT/'cache/simulated'/model/(token+'.npz'),dict(trajectory_cache_sha256=sha(source),metric_cache_sha256=before),**nom)
 assert sha(row['metric_cache_path'])==before
 p=OUT/'cache/nominal_rows'/(token+'.parquet');p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_parquet(p,index=False)
 return dict(token=token,status='PASS',checks=checks,metric_cache_sha256=before)

def main(a):
 rows=scenes(smoke=a.smoke)
 if not a.smoke:assert read(OUT/'audits/SMOKE_COMPLETE.json')['status']=='PASS'
 results=[];errors=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
  jobs={pool.submit(work,r,a.smoke):r for r in rows}
  for f in concurrent.futures.as_completed(jobs):
   try:results.append(f.result())
   except Exception:errors.append(dict(token=jobs[f]['token'],traceback=traceback.format_exc()));print(errors[-1],flush=True)
   if len(results)%16==0:print('SCORED',len(results),len(rows),flush=True)
 save(OUT/'audits'/('scoring_smoke.json' if a.smoke else 'scoring_formal.json'),dict(status='FAIL' if errors else 'PASS',results=results,errors=errors))
 assert not errors,errors[:1]
 if not a.smoke:
  frame=pd.concat([pd.read_parquet(OUT/'cache/nominal_rows'/(r['token']+'.parquet')) for r in rows],ignore_index=True);frame.to_parquet(OUT/'trajectories.parquet',index=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--workers',type=int,default=16);main(p.parse_args())
