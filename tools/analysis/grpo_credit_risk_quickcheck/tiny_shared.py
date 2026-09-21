from common import *
from tiny_support import *
import argparse,concurrent.futures,multiprocessing,pickle

def main(a):
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert read(OUT/'manifests/tiny_update.json')['status']=='FROZEN_BEFORE_UPDATE'
 p=initialise();before=state_hash(p);dest=OUT/'tiny/shared'/str(a.seed);dest.mkdir(parents=True,exist_ok=True);rows=order(a.seed,1);tr=[];ch=[]
 for i,r in enumerate(rows):
  x,y=rollout(p,r,a.seed,1);tr.append(x);ch.append(y)
  if i%16==0:print('SHARED SAMPLE',a.seed,i+1,flush=True)
 with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
  scores=list(pool.map(scored,rows,tr,[True]*64))
 for r,x,y,z in zip(rows,tr,ch,scores):
  path=dest/(r['token']+'.pkl')
  with path.open('wb') as f:pickle.dump(dict(trajectories=x,chain=y,scored=z,behavior_state_sha256=before,token=r['token'],seed=a.seed,step=1),f)
 assert state_hash(p)==before
 save(dest/'manifest.json',dict(status='PASS',state_hash=before,seed=a.seed,scenes=64,files={r['token']:sha(dest/(r['token']+'.pkl')) for r in rows},sampler_calls=64,candidates=1024,nominal_score_calls=2048,perturbation_scores=8192,optimizer_steps=0))
 print('SHARED COMPLETE',a.seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);main(p.parse_args())
