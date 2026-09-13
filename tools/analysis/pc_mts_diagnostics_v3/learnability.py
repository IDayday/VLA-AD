"""C GPU measurement with exactly common timestep/epsilon within every pair."""
from native import *
from torch.utils.data import DataLoader

def evaluate(p,trajs,keys,embeds,device):
    a=torch.as_tensor(trajs,device=device,dtype=torch.float32);fixed=[];uniform=[]
    for j,t in enumerate([20,50,80]):
        for repeat in range(2):
            vals=[]
            for b in range(0,len(a),16):
                noise=noise_for(keys[b:b+16],'v3_C_fixed_noise',j*2+repeat,device);ts=torch.full((len(noise),),t,device=device,dtype=torch.long)
                loss,_=diffusion(p,a[b:b+16],ts,noise,embeds);vals.append(loss.cpu().numpy())
            fixed.append(np.concatenate(vals))
    for j in range(8):
        vals=[]
        for b in range(0,len(a),16):
            kk=keys[b:b+16];noise=noise_for(kk,'v3_C_uniform_noise',j,device);ts=timesteps_for(kk,'v3_C_uniform_time',j,device)
            loss,_=diffusion(p,a[b:b+16],ts,noise,embeds);vals.append(loss.cpu().numpy())
        uniform.append(np.concatenate(vals))
    return np.stack(fixed,1),np.stack(uniform,1)

def main():
    assert (OUT/'manifests/audit_native.json').exists() and (OUT/'manifests/audit_C_matching.json').exists()
    rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1));device=setup(rank);p=model();p.requires_grad_(False);queries=pd.read_parquet(OUT/'metrics/C_query_manifest.parquet');qh=sha(OUT/'metrics/C_query_manifest.parquet')
    rows=scenes()[rank::world];base=dict(identity=identity(),query_manifest_sha256=qh)
    pending=[r for r in rows if not valid(OUT/'cache/learnability'/f"{r['token']}.npz",base)]
    loader=DataLoader(CachedObservations(pending,V1),batch_size=None,num_workers=4,multiprocessing_context='fork');start=time.time()
    with torch.no_grad():
        for j,(token,f) in enumerate(loader):
            g=queries[queries.token==token];raw=np.load(V2/'raw_candidates'/f'{token}.npz')['trajectories'];trajs=raw[g.raw_index.to_numpy()];keys=g.noise_key.tolist();vl,action=inputs(f,device);embeds=encode(p,vl,action)
            fixed,uniform=evaluate(p,trajs,keys,embeds,device);errors=recon_errors(p,trajs,keys,embeds,device)
            ep=V2/'denoising'/f'{token}.npz'
            if ep.exists():epsilon=float(np.load(ep)['epsilon']);cal='read_only_V2_Q'
            else:
                Q=np.load(V2/'il_banks'/f'{token}.npz')['Q'];qe=recon_errors(p,Q,[f'{token}:Q:{i}' for i in range(128)],embeds,device,'v3_C_Q_calibration');epsilon=float(np.quantile(qe,.95));cal='independent_V2_Q_with_V3_noise'
            npz(OUT/'cache/learnability'/f'{token}.npz',dict(base,token=token,epsilon_calibration=cal,feature_sha256=sha(V1/'features'/f'{token}.pt')),query_id=np.asarray(g.query_id,dtype=str),fixed_loss=fixed,uniform_loss=uniform,E_rec=errors.mean(1),return_rate=(errors<epsilon).mean(1),epsilon=np.array(epsilon),reconstruction_errors=errors)
            if j%10==0:print(json.dumps(dict(phase='C',rank=rank,done=j+1,total=len(pending),seconds=time.time()-start)),flush=True)
    save(OUT/'manifests'/f'C_gpu_rank{rank}.json',dict(identity=identity(),scenes=len(rows),new_scenes=len(pending),seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
if __name__=='__main__':main()
