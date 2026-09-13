"""G: independent current-policy R/Q banks on the unchanged V2 raw reservoir."""
from native import *
from train import checkpoint_path
from torch.utils.data import DataLoader
sys.path.insert(0,str(ROOT/'tools/analysis/pc_mts_diagnostics_v2'))
from compatibility import fit,position

def jobs():
    return [dict(seed=s,label=label,stage=stage,step=step,path=str(checkpoint_path(stage,'conditional_pc',s,step))) for s in CFG['progressive']['seeds'] for label,stage,step in [('sft_final','sft',200),('grpo_middle','grpo',50),('grpo_final','grpo',100)]]
def main():
    assert (OUT/'manifests/audit_F.json').exists()
    rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1));device=setup(rank);p=model();p.requires_grad_(False);rows=scenes()[rank::world];start=time.time()
    for job in jobs():
        run=f"{job['label']}_seed{job['seed']}";path=Path(job['path']);p.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state_dict'],strict=True);p.eval();meta=dict(identity=identity(),run=run,checkpoint_sha256=sha(path),reference_R=128,query_Q=128,seeds=f'v3_progressive_{run}_R/Q')
        pending=[r for r in rows if not valid(OUT/'cache/progressive'/run/f"{r['token']}.npz",meta)];loader=DataLoader(CachedObservations(pending,V1),batch_size=None,num_workers=4,multiprocessing_context='fork')
        with torch.no_grad():
            for j,(token,f) in enumerate(loader):
                vl,action=inputs(f,device);R=sample(p,vl,action,token,f'v3_progressive_{run}_R',128,device);Q=sample(p,vl,action,token,f'v3_progressive_{run}_Q',128,device)
                torch.manual_seed(0);ref=p.get_action(vl,action(1),deterministic=False)['pred_traj'].cpu().numpy();raw=np.load(V2/'raw_candidates'/f'{token}.npz')['trajectories'];cal=fit(R,Q);pos=position(raw,cal)
                npz(OUT/'cache/progressive'/run/f'{token}.npz',dict(meta,token=token,raw_sha256=sha(V2/'raw_candidates'/f'{token}.npz')),R=R,Q=Q,reference=ref,**pos)
                if j%20==0:print(json.dumps(dict(phase='G',run=run,rank=rank,done=j+1,total=len(pending),seconds=time.time()-start)),flush=True)
    save(OUT/'manifests'/f'G_gpu_rank{rank}.json',dict(identity=identity(),scenes_per_checkpoint=len(rows),checkpoints=6,seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
if __name__=='__main__':main()
