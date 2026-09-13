"""Same V3 300-scene / 64-rollout CRN evaluation, new checkpoint/cache namespace."""
from common_r import *
import argparse, torch, math
def source_paths(method,sd,step,token):
    if step==0:
        sub=Path('sft/official_il/step0000') if method=='official_il' else Path('sft')/run_name(method,sd)/'step0200'
        return V3/'cache/rollouts'/sub/f'{token}.npz',V3/'cache/scores'/sub/f'{token}.npz'
    sub=Path(run_name(method,sd))/f'step{step:04d}'/f'{token}.npz'
    return OUT/'cache/rollouts'/sub,OUT/'cache/scores'/sub
def job(method,sd,step,shard=0,shards=1,device=None,planner=None):
    import native as n
    from train import parent_slots
    from distributed_rollout import CachedObservations
    from torch.utils.data import DataLoader
    device=device or setup(int(os.environ.get('LOCAL_RANK',0)))
    p=planner or n.model()
    expected=state_only(checkpoint(method,sd,step));incompatible=p.load_state_dict(expected,strict=False)
    assert not incompatible.unexpected_keys
    assert all(k.startswith('action_aware_aux_head.') for k in incompatible.missing_keys)
    p.eval();p.requires_grad_(False)
    base=dict(identity(),checkpoint_sha256=sha(checkpoint(method,sd,step)),CRN_namespace=CFG['evaluation']['namespace'],rollouts=64,precision='fp32')
    rows=[r for r in scenes() if r['token'] in set(tokens('holdout'))][shard::shards]
    pending=[r for r in rows if not v3.valid(source_paths(method,sd,step,r['token'])[0],base)]
    loader=DataLoader(CachedObservations(pending,V1),batch_size=None,num_workers=4,multiprocessing_context='fork');start=time.time()
    with torch.no_grad():
        for j,(token,f) in enumerate(loader):
            vl,action=n.inputs(f,device);trajs=n.sample(p,vl,action,token,CFG['evaluation']['namespace'],64,device);embeds=n.encode(p,vl,action)
            native=np.load(V2/'il_banks'/f'{token}.npz')['native'][:16];raw=np.load(V2/'raw_candidates'/f'{token}.npz')['trajectories']
            fitting_method='gt_only' if method=='official_il' else method
            pools=read(V3/'cache/pools'/f'{token}.json')['methods'];ids=[0] if fitting_method=='gt_only' else pools[fitting_method]['indices']
            slots,w,_=parent_slots(ids,token,0,0,0);targets=[native,raw[slots]];weights=[np.full(16,1/16),w];fits=[]
            for draw in range(4):
                vals=[]
                for target,weight in zip(targets,weights):
                    keys=[f'{token}:slot{k}' for k in range(16)];eps=n.noise_for(keys,'v3_holdout_fitting_noise',draw,device);ts=n.timesteps_for(keys,'v3_holdout_fitting_t',draw,device)
                    vals.append(float((n.diffusion(p,torch.as_tensor(target,device=device,dtype=torch.float32),ts,eps,embeds)[0].cpu().numpy()*weight).sum()))
                fits.append(vals)
            npz(source_paths(method,sd,step,token)[0],dict(base,feature_sha256=sha(V1/'features'/f'{token}.pt')),trajectories=trajs,
                fitting_names=np.asarray(['IL_retention',fitting_method]),fitting_loss=np.asarray(fits).mean(0))
            if j%50==0:print(json.dumps(dict(run=run_name(method,sd),step=step,shard=shard,done=j+1,total=len(pending),seconds=time.time()-start)),flush=True)
    save(OUT/'manifests'/f'eval_{run_name(method,sd)}_{step}_{shard}.json',dict(**base,scenes=len(rows),shard=shard,shards=shards,seconds=time.time()-start,host=socket.gethostname()))
def queue():
    import native as n
    rank=int(os.environ.get('LOCAL_RANK',0));device=setup(rank);p=n.model()
    claims=OUT/'queue/eval';claims.mkdir(parents=True,exist_ok=True)
    while True:
        for method in CFG['methods']:
            for sd in CFG['seeds']:
                for step in CFG['snapshots'][1:]:
                    for shard in range(4):
                        if not checkpoint(method,sd,step).exists():continue
                        key=f'{run_name(method,sd)}_{step}_{shard}';lock=claims/f'{key}.lock'
                        try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
                        except FileExistsError:continue
                        os.write(fd,json.dumps(dict(host=socket.gethostname(),rank=rank,pid=os.getpid())).encode());os.close(fd)
                        try:job(method,sd,step,shard,4,device,p)
                        except BaseException:
                            save(claims/f'{key}.failed.json',dict(host=socket.gethostname(),rank=rank));raise
        if len(list((OUT/'manifests').glob('eval_*.json')))==224:break
        time.sleep(10)
    print('Evaluation queue worker complete',socket.gethostname(),rank,flush=True)
if __name__=='__main__':queue()
