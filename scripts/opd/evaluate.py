import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,concurrent.futures,multiprocessing
from scripts.opd.common import *


def score_task(task):
    row,path,dest=task
    from scripts.opd.scoring import Scoring,load_cache
    bank=np.load(path);names=json.loads(str(bank['metadata']))['models'];out={}
    scorer=Scoring();cache=load_cache(row['metric_cache_path'])
    for name in names:
        if name=='IL_screen':tr=bank[name]
        else:tr=bank[name]
        values=scorer.batch(cache,tr)
        out[name]={}
        for k,v in values.items():
            if k=='simulated_states':continue
            if k.startswith('first_'):
                out[name][k]=[float(t) if np.isfinite(t) else None for t in v]
            elif isinstance(v,np.ndarray):
                if not np.isfinite(v).all():raise FloatingPointError(f'Nonfinite scoring quantity {k}')
                out[name][k]=v.tolist()
            else:
                if not np.isfinite(v):raise FloatingPointError(f'Nonfinite scoring quantity {k}')
                out[name][k]=v
    write(dest,dict(token=row['token'],log=row['log_name'],scores=out,predictions_sha256=sha(path),model_metadata=json.loads(str(bank['metadata']))))
    return row['token']


def main():
    import torch
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head
    p=argparse.ArgumentParser();p.add_argument('--exp',required=True);p.add_argument('--name');p.add_argument('--split',default='dev')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--checkpoint-step',type=int)
    p.add_argument('--serial',action='store_true',help='Keep one native inference process; no intra-GPU shard wrapper')
    p.add_argument('--limit',type=int);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    configure();cfg=config();name=a.name or ('E0' if a.exp=='E0' else f'train_{a.exp}_{a.seed}');dest=ROOT/'evaluation'/a.split/name
    label=name+(f'_step{a.checkpoint_step}' if a.checkpoint_step is not None else '')
    dest=ROOT/'evaluation'/a.split/label
    policy=ROOT/'resources/evaluation_concurrency.json'
    if not a.serial and a.shards==1 and a.rank==0 and policy.exists():
        routing=read(policy)
        if not routing['parity_passed'] or routing['trajectory_max_error']!=0.:raise ValueError('Unqualified inference concurrency')
        count=routing['processes_per_gpu'];spec=dest/f'parallel_{os.getpid()}.json';receipt=dest/f'parallel_{os.getpid()}_receipt.json'
        commands=[['scripts/opd/evaluate.py',*sys.argv[1:],'--rank',str(k),'--shards',str(count)] for k in range(count)]
        write(spec,commands)
        subprocess.run([PYTHON,'scripts/opd/parallel.py','--spec',str(spec),'--concurrency',str(count),'--receipt',str(receipt)],cwd=SOURCE,check=True)
        return
    if a.exp=='E0':
        models={n:build_head(cfg[k],frozen=True) for n,k in [('IL','student_init'),('RL','teacher_init'),('COPY','teacher_init')]}
        checkpoint_identity={n:read(ROOT/'audit/checkpoints.json')[n]['sha256'] for n in models}
    else:
        path=ROOT/'runs'/name/(f'checkpoint_{a.checkpoint_step}.pt' if a.checkpoint_step is not None else 'latest.pt')
        models={name:build_head(path,frozen=True)}
        checkpoint_identity={name:sha(path)}
    write(dest/f'worker_source_{a.rank}.json',dict(owner=identity(),checkpoint_sha256=checkpoint_identity,config_sha256=digest(cfg),
        source_hashes={str(p.relative_to(SOURCE)):sha(p) for directory in ('navsim/agents/recogdrive/opd','scripts/opd') for p in (SOURCE/directory).glob('*.py')}))
    selected=rows(a.split)
    if a.limit:selected=selected[:a.limit]
    start=time.time();pending=[];completed=0
    with concurrent.futures.ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,row in enumerate(selected[a.rank::a.shards]):
            token=row['token'];predpath=dest/'predictions'/f'{token}.npz';scorepath=dest/'scores'/f'{token}.json'
            if scorepath.exists():
                if read(scorepath)['predictions_sha256']!=sha(predpath):raise ValueError('Conflicting scored payload')
                completed+=1;continue
            if not predpath.exists():
                tokens,inputs=conditioning(observation(row));arrays={}
                for model,h in models.items():
                    trajectories=[]
                    for z in cfg['inference_seeds']:
                        seeds(seed_for(token,z))
                        with torch.no_grad():tr=h.get_action(tokens,inputs,deterministic=False)['pred_traj'].cpu().numpy()[0]
                        trajectories.append(tr)
                    arrays[model]=np.stack(trajectories)
                if a.exp=='E0':
                    assert np.array_equal(arrays['RL'],arrays['COPY']),'E0-COPY trajectory mismatch'
                    screen=[]
                    for z in cfg['screening_seeds']:
                        seeds(seed_for(token,z))
                        with torch.no_grad():screen.append(models['IL'].get_action(tokens,inputs)['pred_traj'].cpu().numpy()[0])
                    arrays['IL_screen']=np.stack(screen)
                meta=dict(token=token,split=a.split,models=list(arrays),config_sha256=digest(cfg),seeds=cfg['inference_seeds'],owner=identity(),
                    checkpoint_sha256=checkpoint_identity)
                predpath.parent.mkdir(parents=True,exist_ok=True);temp=predpath.with_name(predpath.name+f'.{os.getpid()}.tmp.npz')
                np.savez_compressed(temp,metadata=json.dumps(meta),**arrays);temp.replace(predpath)
            pending.append(pool.submit(score_task,(row,str(predpath),str(scorepath))))
            if len(pending)>=a.workers*4:
                pending.pop(0).result();completed+=1
            if (i+1)%64==0:print('EVALUATE',name,a.split,a.rank,i+1,flush=True)
        for f in pending:f.result();completed+=1
    write(dest/f'complete_{a.rank}.json',dict(status='COMPLETE',rows=completed,shards=a.shards,rank=a.rank,seconds=time.time()-start,
        full_population=a.limit is None,population_limit=a.limit,peak_gpu_bytes=torch.cuda.max_memory_allocated(),owner=identity()))

if __name__=='__main__':main()
