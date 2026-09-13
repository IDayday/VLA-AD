"""Reconstructed common candidates and independent smooth perturbation streams."""
import argparse, concurrent.futures
from common import *

def smooth_perturbation(traj,amplitude,seed,direction=None):
    a=np.asarray(traj,dtype=np.float64);u=np.arange(1,len(a)+1,dtype=float)/len(a)
    rng=np.random.default_rng(seed);kind=int(rng.integers(6)) if direction is None else direction%6
    angle=float(rng.uniform(-np.pi,np.pi));sign=float(rng.choice([-1,1]))
    forward=np.column_stack([np.cos(a[:,2]),np.sin(a[:,2])]);lateral=np.column_stack([-np.sin(a[:,2]),np.cos(a[:,2])])
    if kind==0:delta=sign*u[:,None]**2*lateral
    elif kind==1:delta=sign*u[:,None]**2*forward
    elif kind==2:delta=sign*u[:,None]**2*a[:,:2]
    elif kind==3:delta=u[:,None]**2*np.array([np.cos(angle),np.sin(angle)])
    elif kind==4:delta=(u**2*np.sin(np.pi*u))[:,None]*lateral*sign
    else:delta=(u**2*np.sin(np.pi*u+angle))[:,None]*lateral+.4*u[:,None]**2*forward*sign
    scale=float(np.linalg.norm(delta,axis=1).max())
    if scale<1e-12:delta=u[:,None]**2*np.array([np.cos(angle),np.sin(angle)]);scale=float(np.linalg.norm(delta,axis=1).max())
    delta*=amplitude/scale;out=a.copy();out[:,:2]+=delta
    before=np.diff(np.vstack([np.zeros((1,2)),a[:,:2]]),axis=0);after=np.diff(np.vstack([np.zeros((1,2)),out[:,:2]]),axis=0)
    change=np.arctan2(np.sin(np.arctan2(after[:,1],after[:,0])-np.arctan2(before[:,1],before[:,0])),np.cos(np.arctan2(after[:,1],after[:,0])-np.arctan2(before[:,1],before[:,0])))
    speed=np.linalg.norm(before,axis=1);change*=np.minimum(speed/.2,1.)
    out[:,2]=np.arctan2(np.sin(a[:,2]+change),np.cos(a[:,2]+change))
    assert np.linalg.norm(out[:,:2]-a[:,:2],axis=1).max()<=amplitude+1e-9
    return out

def perturb_bank(traj,token,stream,amplitudes,directions):
    fingerprint=digest(np.asarray(traj).tolist())
    return np.stack([smooth_perturbation(traj,amp,seed_for(token,stream+fingerprint,j)) for amp in amplitudes for j in range(directions)])

def build_scene(task):
    r,cfg,out,ident,selection_local=task
    token=r['token'];dest=out/'raw_candidates'/f'{token}.npz'
    if not valid_npz(dest,ident):
        gt=np.asarray(r['gt']);trajs=[gt];sources=['exact_gt'];ids=['gt_exact']
        for i in range(cfg['raw_gt_count']-1):
            amp=cfg['gt_perturbation_amplitudes'][i%5]
            trajs.append(smooth_perturbation(gt,amp,seed_for(token,'raw_gt',i),i//5));sources.append('gt_smooth');ids.append(f'gt_smooth_{i:02d}')
        parents={}
        for m in MODELS[1:]:
            path=out/'rollouts'/m/f'{token}.npz';parents[m]=sha(path)
            ext=np.load(path)['external'];assert len(ext)==cfg['raw_model_count']
            trajs.extend(ext);sources.extend([m]*len(ext));ids.extend([f'{m}_external_{i:02d}' for i in range(len(ext))])
        save_npz(dest,dict(ident,token=token,reservoir='reconstructed analysis only',parent_rollout_hashes=parents),trajectories=np.stack(trajs),source=np.asarray(sources),candidate_id=np.asarray(ids))
    if selection_local:
        data=np.load(dest);target=out/'selection_local'/'common'/f'{token}.npz';meta=dict(ident,parent_sha256=sha(dest),stream='selection_local')
        if not valid_npz(target,meta):
            a=np.stack([perturb_bank(t,token,'selection_local',cfg['selection_perturbation_amplitudes'],cfg['selection_directions_per_amplitude']) for t in data['trajectories']])
            save_npz(target,meta,trajectories=a)
    return r['token']

def main(args):
    cfg=config(args.config);sc=manifest(args.manifest);out=ROOT/cfg['output_dir'];ident=identity(cfg,sc)
    tasks=[(r,cfg,out,ident,args.selection_local) for r in sc['scenes']]
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(cfg['cpu_workers'],len(tasks))) as ex:
        for i,token in enumerate(ex.map(build_scene,tasks,chunksize=1)):
            if i%100==0:print(f'Built {i+1}/{len(tasks)} common reservoirs',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--manifest');p.add_argument('--selection-local',action='store_true');main(p.parse_args())
