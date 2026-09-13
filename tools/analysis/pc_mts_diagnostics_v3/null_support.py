"""Supplementary unchanged-weight calibration control; primary G remains untouched."""
from native import *
from torch.utils.data import DataLoader
sys.path.insert(0,str(ROOT/'tools/analysis/pc_mts_diagnostics_v2'))
from compatibility import fit,position
def main():
    protocol=read(OUT/'manifests/calibration_control_frozen.json');path=ROOT/'configs/pc_mts_diagnostics_v3/calibration_control.yaml';assert sha(path)==protocol['sha256'];assert (OUT/'manifests/audit_G.json').exists()
    rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1));device=setup(rank);p=model();p.requires_grad_(False);base=dict(identity=identity(),control_config_sha256=sha(path),checkpoint_sha256=v1.models()[0]['sha256'],control='unchanged_official_IL_R128_Q128')
    rows=scenes()[rank::world];pending=[r for r in rows if not valid(OUT/'cache/null_support'/f"{r['token']}.npz",base)];loader=DataLoader(CachedObservations(pending,V1),batch_size=None,num_workers=4,multiprocessing_context='fork');start=time.time()
    with torch.no_grad():
        for j,(token,f) in enumerate(loader):
            vl,action=inputs(f,device);R=sample(p,vl,action,token,'v3_progressive_official_il_null_R',128,device);Q=sample(p,vl,action,token,'v3_progressive_official_il_null_Q',128,device);raw=np.load(V2/'raw_candidates'/f'{token}.npz')['trajectories'];pos=position(raw,fit(R,Q))
            npz(OUT/'cache/null_support'/f'{token}.npz',dict(base,token=token),R=R,Q=Q,**pos)
            if j%20==0:print(json.dumps(dict(phase='G_null',rank=rank,done=j+1,total=len(pending),seconds=time.time()-start)),flush=True)
    save(OUT/'manifests'/f'G_null_rank{rank}.json',dict(identity=identity(),scenes=len(rows),seconds=time.time()-start))
if __name__=='__main__':main()
