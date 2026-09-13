"""Frozen 5-checkpoint, 1000-scene conditional trajectory distribution audit."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('NAVSIM_DISABLE_TQDM', '1')
os.environ.setdefault('NUPLAN_MAPS_ROOT', '/mnt/navsim/maps')
os.environ.setdefault('NUPLAN_MAP_VERSION', 'nuplan-maps-v1.0')
ROOT = Path('/mnt/project/VLA-AD')
CODE = Path('/mnt/project/VLA-AD-worktrees/a5-epdms-stage3')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''): h.update(b)
    return h.hexdigest()

def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str) + '\n'); temp.replace(path)

def features(args):
    sys.path.insert(0, str(CODE))
    import numpy as np
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SensorConfig
    from navsim.agents.recogdrive.recogdrive_features import ReCogDriveFeatureBuilder
    from transformers import AutoModel
    tokens = json.loads((args.out/'tokens.json').read_text())[args.shard::args.shards]
    if args.limit: tokens=tokens[:args.limit]
    filt=instantiate(OmegaConf.load(CODE/'navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml'))
    filt.tokens=tokens
    sensors=SensorConfig(cam_f0=True,cam_l0=False,cam_l1=False,cam_l2=False,cam_r0=False,cam_r1=False,cam_r2=False,cam_b0=False,lidar_pc=False)
    loader=SceneLoader(Path('/mnt/navsim/test_navsim_logs/test'),Path('/mnt/navsim/test_sensor_blobs/test'),filt,sensors,load_image_path=True)
    if set(loader.tokens)!=set(tokens): raise RuntimeError('Scene coverage mismatch')
    # Override the historical backbone's hardcoded BF16 loader for this FP32 audit.
    original=AutoModel.from_pretrained
    def fp32_load(*a,**kw):
        kw['torch_dtype']=torch.float32;kw['use_flash_attn']=False
        return original(*a,**kw)
    AutoModel.from_pretrained=fp32_load
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    builder=ReCogDriveFeatureBuilder(cache_hidden_state=True,cache_mode=True,model_type='internvl',checkpoint_path=str(ROOT/'checkpoints/recogdrive/ReCogDrive-VLM-2B'),device='cuda',use_expert_features=False)
    assert next(builder.backbone.model.parameters()).dtype==torch.float32
    started=time.time();records=[]
    for i,token in enumerate(tokens):
        path=args.out/'features'/f'{token}.pt';path.parent.mkdir(exist_ok=True)
        if not path.exists():
            # Inference input is constructed with history only. GT is fetched after inference.
            agent_input=loader.get_agent_input_from_token(token)
            with torch.inference_mode(): f=builder.compute_features(agent_input)
            assert all(k not in f for k in ['trajectory','teacher_score','teacher_trajectory'])
            torch.save({k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in f.items()},path.with_suffix('.tmp'))
            path.with_suffix('.tmp').replace(path)
        scene=loader.get_scene_from_token(token)
        gt=scene.get_future_trajectory(8).poses
        image_path=str(loader.get_agent_input_from_token(token).cameras[-1].cam_f0.image)
        records.append(dict(token=token,log=loader.token_to_log_file[token],gt=np.asarray(gt).tolist(),image_path=image_path,features_sha256=sha(path)))
        if (i+1)%10==0 or i==0: print(json.dumps(dict(phase='features',shard=args.shard,done=i+1,total=len(tokens),seconds=time.time()-started)),flush=True)
    save_json(args.out/f'features_shard{args.shard}.json',records)

def load_planner(model):
    import torch
    if sha(model['path']) != model['sha256']:
        raise RuntimeError('Checkpoint changed since identity audit')
    sys.path.insert(0, model['code_root'])
    from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner, ReCogDriveDiffusionPlannerConfig
    cfg=ReCogDriveDiffusionPlannerConfig(diffusion_model_cfg={'num_heads':8,'head_dim':48,'num_layers':16,'output_dim':512,'dropout':0.,'attention_bias':True,'norm_eps':1e-5,'interleave_attention':True},action_dim=3,action_horizon=8,input_embedding_dim=384,hidden_size=1024,sampling_method='ddim',num_inference_steps=5)
    for k,v in model['config'].items():
        if hasattr(cfg,k):setattr(cfg,k,v)
    cfg.vlm_size='small'
    p=ReCogDriveDiffusionPlanner(cfg).float()
    raw=torch.load(model['path'],map_location='cpu',weights_only=False)['state_dict']
    prefix='agent.action_head.'
    state={k[len(prefix):]:v for k,v in raw.items() if k.startswith(prefix) and not k.startswith(prefix+'old_policy.')}
    # Exact schema: no silent skipping of a planner parameter.
    p.load_state_dict(state,strict=True)
    p=p.cuda().eval()
    for key,val in [('denoised_clip_value',1.),('final_action_clip_value',1.),('eval_randn_clip_value',1.),('eval_min_sampling_denoising_std',0.0001)]:setattr(p,key,val)
    return p

def sample(args):
    import numpy as np
    import torch
    from transformers.feature_extraction_utils import BatchFeature
    model=next(m for m in json.loads((args.out/'models.json').read_text()) if m['id']==args.model)
    torch.set_num_threads(1);torch.manual_seed(20260912)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    p=load_planner(model)
    tokens=json.loads((args.out/'tokens.json').read_text())[args.shard::args.shards]
    if args.limit:tokens=tokens[:args.limit]
    started=time.time();dest=args.out/'predictions'/args.model;dest.mkdir(parents=True,exist_ok=True)
    replay_checked=False
    for i,token in enumerate(tokens):
        target=dest/f'{token}.npz'
        if target.exists():continue
        f=torch.load(args.out/'features'/f'{token}.pt',map_location='cpu',weights_only=False)
        assert 'trajectory' not in f
        vl=f['last_hidden_state'].unsqueeze(0).cuda().float()
        history=f['history_trajectory'].view(1,-1).cuda().float()
        status=f['status_feature'].unsqueeze(0).cuda().float()
        command=f['high_command_one_hot'].unsqueeze(0).cuda().float()
        assert command.shape==(1,3)
        def action_batch(size):
            return BatchFeature(data={'his_traj':history.expand(size,-1),'history_trajectory':history.view(1,4,3).expand(size,-1,-1),'status_feature':status.expand(size,-1),'high_command_one_hot':command.expand(size,-1),'state':torch.cat([status,history],dim=1).expand(size,-1)})
        seed=int.from_bytes(hashlib.sha256(f'20260913:{token}'.encode()).digest()[:4],'little')%(2**31-1)
        result={}
        for protocol in ['initial_noise_only','stochastic_ddim']:
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            init=torch.randn((64,8,3),device='cuda',dtype=torch.float32)
            predictions=[]
            for begin in range(0,64,args.batch):
                size=min(args.batch,64-begin)
                action=action_batch(size)
                with torch.inference_mode():
                    output=p.get_action(vl.expand(size,-1,-1),action,init_actions=init[begin:begin+size].clone(),deterministic=(protocol=='initial_noise_only'))
                predictions.append(output['pred_traj'].float().cpu().numpy())
            values=np.concatenate(predictions)
            if values.shape!=(64,8,3) or not np.isfinite(values).all():raise RuntimeError(f'Bad predictions {token}')
            result[protocol]=values
            if protocol=='initial_noise_only' and not replay_checked:
                # Same initial noise: repeat batch, then compare one member with singleton inference.
                size=min(args.batch,64)
                with torch.inference_mode():
                    replay=p.get_action(vl.expand(size,-1,-1),action_batch(size),init_actions=init[:size].clone(),deterministic=True)['pred_traj'].float().cpu().numpy()
                    single=p.get_action(vl,action_batch(1),init_actions=init[:1].clone(),deterministic=True)['pred_traj'].float().cpu().numpy()
                replay_error=float(np.max(np.abs(replay-values[:size])))
                singleton_error=float(np.max(np.abs(single-values[:1])))
                if replay_error>1e-6 or singleton_error>1e-3:raise RuntimeError(f'Inference parity failed: {replay_error}, {singleton_error}')
                save_json(dest/f'parity_shard{args.shard}.json',dict(token=token,repeated_batch_max_abs_m_or_rad=replay_error,singleton_max_abs_m_or_rad=singleton_error,all_parameter_keys_strict=True,gt_supplied_to_model=False))
                replay_checked=True
        temp=target.with_suffix('.tmp.npz');np.savez_compressed(temp,**result);temp.replace(target)
        if (i+1)%10==0 or i==0:print(json.dumps(dict(phase='sample',model=args.model,shard=args.shard,done=i+1,total=len(tokens),seconds=time.time()-started)),flush=True)
    save_json(dest/f'metadata_shard{args.shard}.json',dict(model=model,shard=args.shard,scenes=len(tokens),precision='fp32',samples=64,batch=args.batch,seconds=time.time()-started,code_sha256=sha(Path(model['code_root'])/'navsim/agents/recogdrive/recogdrive_diffusion_planner.py'),checkpoint_sha256=sha(model['path'])))

def validate(args):
    import inspect
    import torch
    from transformers.feature_extraction_utils import BatchFeature
    model=next(m for m in json.loads((args.out/'models.json').read_text()) if m['id']==args.model)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=load_planner(model)
    from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    supplied=dict(model['config'],trajectory_sampling=TrajectorySampling(time_horizon=4,interval_length=.5),checkpoint_path='',allow_random_init=True,cache_hidden_state=True,cache_mode=False,dit_type='small',vlm_type='internvl',vlm_path='',grpo=False)
    params=inspect.signature(ReCogDriveAgent.__init__).parameters
    agent=ReCogDriveAgent(**{k:v for k,v in supplied.items() if k in params})
    agent.action_head=p;agent=agent.cuda().eval()
    results=[]
    for token in json.loads((args.out/'tokens.json').read_text())[:4]:
        f=torch.load(args.out/'features'/f'{token}.pt',map_location='cpu',weights_only=False)
        features={k:v.unsqueeze(0).cuda().float() for k,v in f.items()}
        h=features['history_trajectory'].reshape(1,-1);s=features['status_feature']
        action=BatchFeature(data={'his_traj':h,'history_trajectory':features['history_trajectory'],'status_feature':s,'high_command_one_hot':features['high_command_one_hot'],'state':torch.cat([s,h],1)})
        with torch.inference_mode():
            torch.manual_seed(4321);reference=agent.forward(features)['pred_traj']
            torch.manual_seed(4321);direct=p.get_action(features['last_hidden_state'],action)['pred_traj']
        error=float((reference-direct).abs().max())
        if error>1e-6:raise RuntimeError(f'Production agent parity failed: {token}: {error}')
        results.append(dict(token=token,max_abs_error=error))
    save_json(args.out/f'production_parity_{args.model}.json',results)
    print(json.dumps(results),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['features','sample','validate']);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--model');ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--batch',type=int,default=16)
    args=ap.parse_args();globals()[args.phase](args)
