"""Native pre-scorer DDV2 / DrivoR proposals, with unmodified external sources."""
from common_mass import *
import argparse, pickle, inspect, types, time, traceback, subprocess
import torch

class CapturedProposals(Exception):
    def __init__(self,tensor):self.tensor=tensor

def prepare(source,rank):
    sys.path.insert(0,str(OUT/'deps'))
    # Read-only reuse of the historical export's pure-Python dependency cache.
    sys.path.append(str(ROOT/'outputs/ddv2_internal_structured_navtrain_full103288_4shard_localg4to7_20260726/python_deps'))
    torch.cuda.set_device(rank);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    probe=torch.empty(1,device=f'cuda:{rank}');torch.cuda.synchronize();time.sleep(3)
    root=Path('/mnt/project/external/DrivoR' if source=='drivor' else '/mnt/project/external/DiffusionDriveV2-canonical-1cd12a1')
    assets=Path('/mnt/project/external/DrivoR' if source=='drivor' else '/mnt/project/external/DiffusionDriveV2')
    sys.path.insert(0,str(root));os.chdir(assets)
    if source=='ddv2':
        # Use the exact local backbone file already supported by the repository's
        # fallback; avoids a pointless remote pretrain lookup before full ckpt load.
        import timm
        original_create=timm.create_model
        def local_create(name,*args,**kwargs):
            if name=='resnet34' and kwargs.get('pretrained'):
                kwargs['pretrained_cfg_overlay']=dict(file=str(assets/'ckpts/resnet34.a1_in1k/pytorch_model.bin'))
            return original_create(name,*args,**kwargs)
        timm.create_model=local_create
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    name='drivoR' if source=='drivor' else 'diffusiondrivev2_sel_agent'
    configpath=root/'navsim/planning/script/config/common/agent'/f'{name}.yaml'
    c=OmegaConf.load(configpath);checkpoint=cfg()['raw_bank'][f'external_{source}_checkpoint'];c.checkpoint_path=checkpoint
    if source=='drivor':c.scheduler_args=None;c.batch_size=1;c.progress_bar=False
    else:c.config.bkb_path=str(assets/'ckpts/resnet34.a1_in1k/pytorch_model.bin');c.config.plan_anchor_path=str(assets/'kmeans_navsim_traj_20.npy')
    agent=instantiate(c);agent.initialize();agent.eval().float().to(f'cuda:{rank}');agent.requires_grad_(False)
    raw=torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict']
    transform=(lambda k:k.replace('agent._drivor_model','_drivor_model')) if source=='drivor' else (lambda k:k.replace('agent.',''))
    state={transform(k):v for k,v in raw.items()};missing=set(agent.state_dict())-set(state);extra=set(state)-set(agent.state_dict())
    # Released checkpoint contains a training vocabulary not referenced by the
    # public inference class. No active model tensor may be missing or unequal.
    allowed_extra={'_transfuser_model._trajectory_head.vocab'} if source=='ddv2' else set()
    assert not missing and extra<=allowed_extra,(missing,extra)
    for k,v in agent.state_dict().items():assert torch.equal(v.cpu(),state[k].cpu()),k
    if source=='ddv2':
        head=agent._transfuser_model._trajectory_head
        def capture(self,trajectories,*a,**kw):raise CapturedProposals(trajectories)
        head._get_scorer_inputs=types.MethodType(capture,head)
    else:
        # Returned proposals are untouched by scorer; scorer cannot change them.
        head=agent._drivor_model
    files={str(Path(inspect.getfile(type(m)))) for m in agent.modules() if '/mnt/project/external/' in inspect.getfile(type(m))}
    identity=dict(source=source,model_name='DrivoR NAVSIM-v1 25 epochs' if source=='drivor' else 'DiffusionDriveV2 selector',checkpoint_path=checkpoint,checkpoint_sha256=sha(checkpoint),code_root=str(root),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),runtime_files={f:sha(f) for f in sorted(files)},config_path=str(configpath),config_sha256=sha(configpath),resolved_config=OmegaConf.to_container(c,resolve=True),input_modalities=str(agent.get_sensor_config()),output_frame='ego local XY meters, heading radians',horizon_s=4,interval_s=.5,includes_t0=False,pre_scorer=True,proposal_selection='fixed protocol/source/index SHA256 ranking; never A/B or scores',original_files_modified=False)
    identity['checkpoint_unused_keys']=sorted(extra)
    save(OUT/'audits'/f'{source}_identity.json',identity)
    return agent,identity

def run(source,rank,smoke):
    agent,identity=prepare(source,rank)
    from navsim.common.dataclasses import AgentInput
    from torch.utils.data import default_collate
    rows=scenes();rows=sorted(rows,key=lambda s:digest(['external_smoke',s['token']]))[:4] if smoke else rows
    logs={};start=time.time();failures=[];records=[]
    for j,s in enumerate(rows):
        token=s['token'];path=OUT/'cache/external'/source/f'{token}.npz';meta=dict(token=token,source=source,checkpoint_sha256=identity['checkpoint_sha256'],runtime_files_hash=digest(identity['runtime_files']))
        if valid(path,meta):continue
        try:
            if s['log'] not in logs:
                if len(logs)>4:logs.clear()
                with open(Path('/mnt/navsim/trainval_navsim_logs/trainval')/(s['log']+'.pkl'),'rb') as f:logs[s['log']]=pickle.load(f)
            frames=logs[s['log']][s['frame_start']:s['frame_start']+4]
            assert frames[-1]['token']==token,(frames[-1].get('token'),token)
            observed=AgentInput.from_scene_dict_list(frames,Path('/mnt/navsim/trainval_sensor_blobs/trainval'),4,agent.get_sensor_config())
            features={}
            for builder in agent.get_feature_builders():features.update(builder.compute_features(observed))
            f=default_collate([features]);f={k:v.to(f'cuda:{rank}') if torch.is_tensor(v) else v for k,v in f.items()}
            seed=random_seed(token,'external_'+source,0);torch.manual_seed(seed)
            with torch.no_grad():
                try:
                    result=agent.forward(f)
                    assert source=='drivor';full=result['proposals']
                except CapturedProposals as capture:full=capture.tensor
            full=full.detach().float().cpu().numpy()[0];assert full.ndim==3 and full.shape[-1]==3 and full.shape[1]>=8
            n=len(full);indices=sorted(range(n),key=lambda i:digest([protocol(),source,i]))[:64]
            out=full[indices,:8].copy();out[:,:,2]=np.arctan2(np.sin(out[:,:,2]),np.cos(out[:,:,2]))
            assert np.isfinite(out).all()
            npz(path,dict(meta,generation_seed=seed,native_proposal_count=n,native_num_poses=full.shape[1],preprocessing_hash=digest(identity['resolved_config']),postprocessing='first8_poses_heading_wrap_only'),trajectories=out,proposal_indices=np.asarray(indices),native_proposals=full)
            records.append(dict(token=token,native_count=n,kept=len(out),unique=len({content_hash(t) for t in out})))
            if j%10==0:print(source,j+1,len(rows),time.time()-start,flush=True)
        except Exception as exc:
            failures.append(dict(token=token,status='ERROR',error=repr(exc),traceback=traceback.format_exc()));print(failures[-1],flush=True)
            if smoke:raise
    save(OUT/'audits'/f'{source}_{"smoke" if smoke else "generation"}.json',dict(records=records,failures=failures,seconds=time.time()-start,completed_at=utc()))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',choices=['ddv2','drivor']);p.add_argument('--rank',type=int,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.source,a.rank,a.smoke)
