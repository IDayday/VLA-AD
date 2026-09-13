"""Fixed 300 holdout scenes, 64 CRN rollouts per checkpoint; official step0 deduplicated."""
import argparse
from native import *
from train import checkpoint_path,parent_slots
from torch.utils.data import DataLoader

def specifications(stage):
    cfg=CFG['training']['micro_sft' if stage=='sft' else 'grpo'];jobs=[];aliases=[]
    if stage=='sft':jobs.append(dict(stage='sft',run='official_il',step=0,path=None))
    for method in CFG['training']['methods']:
        for sd in CFG['training']['seeds']:
            run=f'{method}_seed{sd}'
            for step in cfg['snapshots']:
                if step==0:
                    source=dict(stage='sft',run='official_il',step=0) if stage=='sft' else dict(stage='sft',run=run,step=CFG['training']['micro_sft']['steps'])
                else:
                    path=checkpoint_path(stage,method,sd,step);assert path.exists(),path;jobs.append(dict(stage=stage,run=run,step=step,path=str(path)));source=dict(stage=stage,run=run,step=step)
                aliases.append(dict(stage=stage,method=method,seed=sd,step=step,source=source))
    return jobs,aliases

def evaluate_job(p,job,device,rank):
    ckhash=v1.models()[0]['sha256'] if job['path'] is None else sha(job['path'])
    if job['path'] is not None:p.load_state_dict(torch.load(job['path'],map_location='cpu',weights_only=False)['state_dict'],strict=True)
    p.eval();p.requires_grad_(False);base=dict(identity=identity(),checkpoint_sha256=ckhash,CRN_namespace='v3_diagnostic_holdout64',rollouts=64,stage=job['stage'],run=job['run'],step=job['step']);directory=OUT/'cache/rollouts'/job['stage']/job['run']/f"step{job['step']:04d}"
    rows=[r for r in scenes() if r['token'] in set(tokens('holdout'))]
    if 'compute_shard' in job:rows=rows[job['compute_shard']::2]
    pending=[r for r in rows if not valid(directory/f"{r['token']}.npz",base)];lookup={r['token']:r for r in rows};loader=DataLoader(CachedObservations(pending,V1),batch_size=None,num_workers=4,multiprocessing_context='fork');start=time.time()
    with torch.no_grad():
        for j,(token,f) in enumerate(loader):
            vl,action=inputs(f,device);trajs=sample(p,vl,action,token,'v3_diagnostic_holdout64',64,device);embeds=encode(p,vl,action)
            native=np.load(V2/'il_banks'/f'{token}.npz')['native'][:16];raw=np.load(V2/'raw_candidates'/f'{token}.npz')['trajectories'];pools=read(OUT/'cache/pools'/f'{token}.json')['methods']
            # Official step0 is shared: cache all six target losses once. Each
            # updated model needs its own target loss and common IL retention.
            targets=[native];weights=[np.full(16,1/16)];names=['IL_retention']
            target_methods=CFG['training']['methods'] if job['run']=='official_il' else [job['run'].rsplit('_seed',1)[0]]
            for method in target_methods:
                ids=[0] if method=='gt_only' else pools[method]['indices'];slots,w,_=parent_slots(ids,token,0,0,0);targets.append(raw[slots]);weights.append(w);names.append(method)
            target=np.concatenate(targets);fit=[]
            for draw in range(4):
                losses=[]
                # Same 16 noise/timestep slots across target methods and all checkpoints.
                for b in range(0,len(target),16):
                    keys=[f'{token}:slot{k}' for k in range(16)];eps=noise_for(keys,'v3_holdout_fitting_noise',draw,device);ts=timesteps_for(keys,'v3_holdout_fitting_t',draw,device);arr=torch.as_tensor(target[b:b+16],device=device,dtype=torch.float32)
                    losses.append(diffusion(p,arr,ts,eps,embeds)[0].cpu().numpy())
                fit.append((np.stack(losses)*np.stack(weights)).sum(1))
            npz(directory/f'{token}.npz',dict(base,feature_sha256=sha(V1/'features'/f'{token}.pt')),trajectories=trajs,fitting_names=np.asarray(names),fitting_loss=np.asarray(fit).mean(0))
            if j%50==0:print(json.dumps(dict(stage=job['stage'],run=job['run'],step=job['step'],rank=rank,done=j+1,total=len(pending),seconds=time.time()-start)),flush=True)
    suffix=f"_shard{job['compute_shard']}" if 'compute_shard' in job else ''
    save(OUT/'manifests'/f"eval_{job['stage']}_{job['run']}_{job['step']}{suffix}.json",dict(**base,scenes=len(rows),seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated()))

def main(stage):
    rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1));device=setup(rank);p=model();jobs,aliases=specifications(stage)
    # 36 GRPO snapshots x 2 scene shards = 72 equal work units / 8 GPUs.
    # This only balances execution; every checkpoint still evaluates all 300
    # scenes and all 64 token-keyed CRN samples with unchanged batch shape.
    if stage=='grpo':jobs=[dict(job,compute_shard=i) for job in jobs for i in range(2)]
    if rank==0:save(OUT/'manifests'/f'evaluation_{stage}.json',dict(identity=identity(),jobs=jobs,aliases=aliases,scenes=300,rollouts=64))
    for job in jobs[rank::world]:evaluate_job(p,job,device,rank)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['sft','grpo']);main(a.parse_args().stage)
