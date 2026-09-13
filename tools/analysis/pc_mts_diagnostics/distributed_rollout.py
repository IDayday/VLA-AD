"""8 GPU scene shards, one checkpoint per invocation, observation-only cache."""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('NUPLAN_MAPS_ROOT','/mnt/navsim/maps')
os.environ.setdefault('NUPLAN_MAP_VERSION','nuplan-maps-v1.0')
from common import *

class CachedObservations:
    def __init__(self,rows,root):self.rows=rows;self.root=root
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        import torch
        t=self.rows[i]['token'];f=torch.load(self.root/'features'/f'{t}.pt',map_location='cpu',weights_only=False)
        return t,f

class RawObservations:
    def __init__(self,rows):self.rows=rows;self.logs={}
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        import pickle
        from navsim.common.dataclasses import AgentInput,SensorConfig
        r=self.rows[i]
        if r['log'] not in self.logs:self.logs[r['log']]=pickle.load(open(Path('/mnt/navsim/trainval_navsim_logs/trainval')/(r['log']+'.pkl'),'rb'))
        # Deliberately pass only four observed frames, never the future slice.
        frames=self.logs[r['log']][r['frame_start']:r['frame_start']+4]
        sensors=SensorConfig(cam_f0=True,cam_l0=False,cam_l1=False,cam_l2=False,cam_r0=False,cam_r1=False,cam_r2=False,cam_b0=False,lidar_pc=False)
        return r['token'],AgentInput.from_scene_dict_list(frames,Path('/mnt/navsim/trainval_sensor_blobs/trainval'),4,sensors,load_image_path=True)

def load_planner(m):
    # This strict loader was verified against each production agent on real scenes.
    sys.path.insert(0,str(ROOT/'scripts/evaluation/distribution_audit'))
    from run import load_planner as loader
    return loader(dict(m,path=m['checkpoint_path']))

def main(args):
    import torch
    from torch.utils.data import DataLoader
    cfg=config(args.config);scene=manifest(args.manifest);rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    torch.cuda.set_device(rank);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    out=ROOT/cfg['output_dir'];out.mkdir(parents=True,exist_ok=True)
    rows=scene['scenes'][rank::world];base=identity(cfg,scene);start=time.time()
    if args.features:
        sys.path.insert(0,str(CODE))
        from navsim.agents.recogdrive.recogdrive_features import ReCogDriveFeatureBuilder
        from transformers import AutoModel
        original=AutoModel.from_pretrained
        def fp32_load(*a,**kw):
            kw['torch_dtype']=torch.float32;kw['use_flash_attn']=False
            return original(*a,**kw)
        AutoModel.from_pretrained=fp32_load
        feature_dir=out/'features';feature_dir.mkdir(exist_ok=True)
        pending=[r for r in rows if not (feature_dir/f"{r['token']}.pt").exists()]
        if pending:
            b=ReCogDriveFeatureBuilder(cache_hidden_state=True,cache_mode=True,model_type='internvl',checkpoint_path=str(ROOT/'checkpoints/recogdrive/ReCogDrive-VLM-2B'),device=f'cuda:{rank}',use_expert_features=False)
            assert next(b.backbone.model.parameters()).dtype==torch.float32
            dl=DataLoader(RawObservations(pending),batch_size=None,num_workers=cfg['loader_workers'],multiprocessing_context='fork')
            for i,(token,agent_input) in enumerate(dl):
                with torch.inference_mode():f=b.compute_features(agent_input)
                assert set(f)=={'last_hidden_state','history_trajectory','status_feature','high_command_one_hot'}
                payload={k:v.cpu() for k,v in f.items()};payload['_metadata']=dict(base,token=token,observation_only=True)
                dest=feature_dir/f'{token}.pt';tmp=dest.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(dest)
                if i%10==0:print(json.dumps(dict(phase='features',rank=rank,done=i+1,total=len(pending),seconds=time.time()-start)),flush=True)
        save(out/'manifests'/f'features_rank{rank}.json',dict(base,rank=rank,count=len(rows),seconds=time.time()-start))
        return
    m=next(m for m in models() if m['name']==args.checkpoint);p=load_planner(m)
    from transformers.feature_extraction_utils import BatchFeature
    ident=identity(cfg,scene,m)
    pending=[r for r in rows if not valid_npz(out/'rollouts'/m['name']/f"{r['token']}.npz",ident)]
    dl=DataLoader(CachedObservations(pending,out),batch_size=None,num_workers=cfg['loader_workers'],multiprocessing_context='fork')
    parity=[]
    for i,(token,f) in enumerate(dl):
        assert f.pop('_metadata')['scene_manifest_hash']==base['scene_manifest_hash']
        vl=f['last_hidden_state'].unsqueeze(0).cuda().float();h=f['history_trajectory'].view(1,-1).cuda().float();s=f['status_feature'].unsqueeze(0).cuda().float();c=f['high_command_one_hot'].unsqueeze(0).cuda().float()
        def action(n):return BatchFeature(data={'his_traj':h.expand(n,-1),'history_trajectory':h.view(1,4,3).expand(n,-1,-1),'status_feature':s.expand(n,-1),'high_command_one_hot':c.expand(n,-1),'state':torch.cat([s,h],1).expand(n,-1)})
        def sample(n,stream,default_seed=False):
            outputs=[]
            seeds=[seed_for(token,stream,j,cfg['seed']) for j in range(n)]
            for begin in range(0,n,cfg['inference_batch']):
                count=min(cfg['inference_batch'],n-begin)
                init=torch.stack([torch.randn((8,3),generator=torch.Generator(device=f'cuda:{rank}').manual_seed(seeds[j]),device=f'cuda:{rank}') for j in range(begin,begin+count)])
                torch.manual_seed(0 if default_seed else seed_for(token,stream+'_steps',begin,cfg['seed']))
                with torch.inference_mode():v=p.get_action(vl.expand(count,-1,-1),action(count),init_actions=None if default_seed else init,deterministic=False)['pred_traj']
                outputs.append(v.float().cpu().numpy())
            return np.concatenate(outputs),seeds
        trajectories,seeds=sample(cfg['num_rollouts'],'policy')
        arrays=dict(trajectories=trajectories)
        if m['name']=='official_il':arrays['reference']=sample(1,'reference',True)[0]
        else:arrays['external']=sample(cfg['raw_model_count'],'external')[0]
        assert all(np.isfinite(v).all() and v.shape[1:]==(8,3) for v in arrays.values())
        if i==0:
            replay,_=sample(cfg['num_rollouts'],'policy');err=float(np.abs(replay-trajectories).max());assert err==0
            parity.append(dict(token=token,replay_max_abs=err))
        save_npz(out/'rollouts'/m['name']/f'{token}.npz',dict(ident,token=token,initial_noise_seeds=seeds,stream_separation='policy / external / reference'),**arrays)
        if i%10==0:print(json.dumps(dict(phase='rollout',model=m['name'],rank=rank,done=i+1,total=len(pending),seconds=time.time()-start)),flush=True)
    save(out/'manifests'/f"rollout_{m['name']}_rank{rank}.json",dict(ident,rank=rank,count=len(rows),seconds=time.time()-start,parity=parity,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated()))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--config');a.add_argument('--manifest');a.add_argument('--checkpoint',choices=MODELS);a.add_argument('--features',action='store_true');a.add_argument('--num-scenes',type=int);a.add_argument('--num-rollouts',type=int);args=a.parse_args()
    if args.num_scenes and args.num_scenes!=manifest(args.manifest)['scene_count']:raise ValueError('num-scenes disagrees with frozen manifest')
    if args.num_rollouts and args.num_rollouts!=config(args.config)['num_rollouts']:raise ValueError('num-rollouts disagrees with frozen config')
    main(args)
