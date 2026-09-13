"""Frozen official-IL native forward-noise / DDIM reconstruction, validation only."""
import argparse,time
from v2_common import *
from gpu_banks import inputs
from distributed_rollout import CachedObservations,load_planner

def reverse(p,x,start,vl_emb,history,ego):
    import torch
    n=len(x)
    for i in range(start,p.ddim_steps):
        t=torch.full((n,),int(p.ddim_t[i]),device=x.device,dtype=torch.long);index=torch.full_like(t,i)
        x=p.p_mean_variance(x,t,index,vl_emb.expand(n,-1,-1),history.expand(n,-1,-1),ego.expand(n,*ego.shape[1:]),True)[0]
    clip=getattr(p,'final_action_clip_value',1.)
    if clip is not None:x=x.clamp(-clip,clip)
    return p.denorm_odo(x)

def reconstruct(p,trajectories,token,starts,embeds,device):
    import torch
    a=torch.as_tensor(trajectories,device=device,dtype=torch.float32);xn=p.norm_odo(a);results=[]
    # Identical XYH candidates across methods receive identical noise, reducing Monte Carlo comparison noise.
    fingerprints=[digest(t.tolist()) for t in np.asarray(trajectories)]
    for level,start in enumerate(starts):
        t=int(p.ddim_t[start]);alpha=p.ddpm_sqrt_alphas_cumprod[t];sigma=p.ddpm_sqrt_one_minus_alphas_cumprod[t]
        for repeat in range(2):
            errors=[]
            for begin in range(0,len(a),16):
                end=min(begin+16,len(a));noise=torch.stack([torch.randn((8,3),device=device,generator=torch.Generator(device=device).manual_seed(seed(token,'v2_denoise_'+fingerprints[j],level*2+repeat))) for j in range(begin,end)])
                out=reverse(p,alpha*xn[begin:end]+sigma*noise,start,*embeds)
                errors.append(torch.linalg.vector_norm(out[...,:2]-a[begin:end,...,:2],dim=-1).mean(-1).cpu().numpy())
            results.append(np.concatenate(errors))
    return np.stack(results,axis=1)

def main(a):
    import torch
    from torch.utils.data import DataLoader
    rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1));device=f'cuda:{rank}'
    torch.cuda.set_device(rank);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=load_planner(v1.models()[0]);schedule=p.ddim_t.cpu().numpy();targets=np.asarray(CFG['denoising_requested_schedule_fractions'])*(p.ddpm_num_train_timesteps-1);starts=[int(abs(schedule-v).argmin()) for v in targets];assert len(set(starts))==3
    metadata=dict(identity=ident(),schedule=schedule.tolist(),requested_targets=targets.tolist(),actual_timesteps=schedule[starts].tolist(),indices=starts,reverse='Native p_mean_variance deterministic eta=0, to t=0, same normalization and clipping',noise='Independent standard Gaussian forward noise; no reverse noise; equal candidate uses equal noise',checkpoint_hash=v1.models()[0]['sha256'])
    rows=[r for r in scenes()['scenes'] if r['token'] in set(subset('smoke' if a.smoke else 'denoising'))][rank::world]
    pending=[]
    for r in rows:
        token=r['token'];meta=dict(metadata,pool_hashes={m:sha(OUT/'candidate_pools'/m/f'{token}.npz') for m in METHODS},IL_bank_sha256=sha(OUT/'il_banks'/f'{token}.npz'))
        if not valid(OUT/'denoising'/f'{token}.npz',meta):pending.append(r)
    loader=DataLoader(CachedObservations(pending,V1),batch_size=None,num_workers=4,multiprocessing_context='fork');start_time=time.time();checks=[]
    with torch.inference_mode():
        for i,(token,f) in enumerate(loader):
            f.pop('_metadata');vl,action=inputs(f,device);inp=action(1);embeds=(p.feature_encoder(vl),p.his_traj_encoder(inp.his_traj.unsqueeze(1)).repeat(1,8,1),p.ego_status_encoder(inp.status_feature))
            if i==0:
                x=torch.randn((1,8,3),device=device,generator=torch.Generator(device=device).manual_seed(seed(token,'v2_denoise_parity')))
                manual=reverse(p,x.clone(),0,*embeds);native=p.get_action(vl,inp,init_actions=x.clone(),deterministic=True)['pred_traj'];error=float((manual-native).abs().max());assert error<1e-5;checks.append(dict(token=token,native_DDIM_max_abs=error))
            ilpath=OUT/'il_banks'/f'{token}.npz';Q=np.load(ilpath)['Q'];poolpaths={m:OUT/'candidate_pools'/m/f'{token}.npz' for m in METHODS};trajs=np.concatenate([Q]+[np.load(poolpaths[m])['trajectories'] for m in METHODS]);errors=reconstruct(p,trajs,token,starts,embeds,device);assert errors.shape==(192,6) and np.isfinite(errors).all()
            qerror=errors[:128];epsilon=float(np.quantile(qerror,.95));ce=errors[128:].reshape(4,16,6)
            meta=dict(metadata,pool_hashes={m:sha(poolpaths[m]) for m in METHODS},IL_bank_sha256=sha(ilpath))
            npz(OUT/'denoising'/f'{token}.npz',meta,Q_errors=qerror,candidate_errors=ce,E_rec=ce.mean(-1),return_rate=(ce<epsilon).mean(-1),epsilon=np.array(epsilon),actual_timesteps=schedule[starts])
            if i%10==0:print(json.dumps(dict(rank=rank,done=i+1,total=len(pending),seconds=time.time()-start_time)),flush=True)
    save(OUT/'manifests'/f'denoising_{"smoke" if a.smoke else "full"}_rank{rank}.json',dict(metadata,rank=rank,count=len(rows),new_count=len(pending),checks=checks,seconds=time.time()-start_time,peak_memory_bytes=torch.cuda.max_memory_allocated()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');main(p.parse_args())
