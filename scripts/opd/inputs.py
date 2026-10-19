"""Native observation encoding and canonical metric-cache construction.

Input preparation may inspect only the four historical frames. Metric creation
is an independent privileged label path and never enters VLM conditioning.
"""
import sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('NUPLAN_MAPS_ROOT','/mnt/navsim/maps')
import argparse,pickle,lzma
from functools import lru_cache
from scripts.opd.common import *


@lru_cache(maxsize=2)
def log(path):
    with open(path,'rb') as f:return pickle.load(f)


def agent():
    from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.agents.recogdrive.opd.checkpoint_io import restore
    c=config();a=ReCogDriveAgent(TrajectorySampling(time_horizon=4,interval_length=.5),vlm_path=c['vlm'],
        checkpoint_path=c['student_init'],cam_type='single',vlm_type='internvl',dit_type='small',vlm_size='small',
        sampling_method='ddim',cache_hidden_state=False,cache_mode=False,grpo=False,train_backbone=False,
        use_expert_features=False,use_last_rd=False,use_bit_drive=False,use_risk_vla=False,
        use_alignment_loss=False,expert_alignment_weight=0.,allow_random_init=False)
    restore(a.action_head,c['student_init']);a.eval();a.requires_grad_(False)
    return a


def features(a,row):
    from navsim.common.dataclasses import AgentInput,SensorConfig
    frames=[dict(f,lidar_path=f.get('lidar_path') or '') for f in log(row['metadata_path'])[row['start_idx']:row['start_idx']+4]]
    sensors=SensorConfig.build_no_sensors();sensors.cam_f0=[3]
    obs=AgentInput.from_scene_dict_list(frames,Path(row['sensor_root']),4,sensors,load_image_path=True)
    result={}
    for b in a.get_feature_builders():result.update(b.compute_features(obs))
    if set(result)!={'history_trajectory','status_feature','high_command_one_hot','image_path_tensor'}:raise ValueError('Unreviewed native inputs')
    return {k:v.unsqueeze(0) for k,v in result.items()}


def encode(a,f):
    import torch
    class Captured(Exception):pass
    payload={}
    def capture(module,args):payload['tokens']=args[0].detach().cpu();raise Captured()
    hook=a.action_head.feature_encoder.register_forward_pre_hook(capture)
    try:
        with torch.no_grad():a.forward({k:v.clone() for k,v in f.items()})
    except Captured:pass
    finally:hook.remove()
    if 'tokens' not in payload:raise RuntimeError('Did not reach native action conditioning')
    tokens=payload['tokens'];stored=tokens.bfloat16()
    if not torch.equal(stored.float(),tokens.float()):raise ValueError('Lossy token storage')
    return stored


def metric(row):
    from navsim.common.dataclasses import Scene,SensorConfig
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.planning.metric_caching.metric_cache import MetricCache
    path=Path(row['metric_cache_path'])
    if path.exists():return
    frames=[dict(f,lidar_path=f.get('lidar_path') or '') for f in log(row['metadata_path'])[row['start_idx']:row['start_idx']+row['frame_count']]]
    scene=Scene.from_scene_dict_list(frames,Path(row['sensor_root']),4,row['frame_count']-4,SensorConfig.build_no_sensors(),load_image_path=True)
    scenario=NavSimScenario(scene,os.environ['NUPLAN_MAPS_ROOT'],'nuplan-maps-v1.0')
    processor=MetricCacheProcessor(None,False)  # Fresh planner per independent scene.
    inputs,initialization=processor._get_planner_inputs(scenario);planner=processor._pdm_closed
    planner.initialize(initialization);reference=planner.compute_planner_trajectory(inputs)
    obs=processor._interpolate_gt_observation(scenario)
    assert len(obs._occupancy_maps)==51
    cache=MetricCache(path,reference,scenario.initial_ego_state,obs,planner._centerline,list(planner._route_lane_dict),planner._drivable_area_map)
    data=lzma.compress(pickle.dumps(cache,protocol=pickle.HIGHEST_PROTOCOL),preset=0)
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+f'.{os.getpid()}.tmp');temp.write_bytes(data);temp.replace(path)


def prepare_manifest():
    import yaml
    source=Path('/mnt/project/VLA-AD/outputs/simscale_sft_20260918/manifests/navtest.json');data=read(source)
    official=SOURCE/'navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml'
    declared=yaml.safe_load(official.read_text())
    assert len(data)==len(declared['tokens']) and {r['token'] for r in data}==set(declared['tokens'])
    assert {r['log_name'] for r in data}<=set(declared['log_names'])
    registered=sum([rows(s) for s in ('train','cal','dev')],[])
    for key in ('token','log_name'):
        assert not ({r[key] for r in data}&{r[key] for r in registered})
    parent=lambda r:r['log_name'].split('_')[0]
    assert not ({parent(r) for r in data}&{parent(r) for r in registered})
    for r in data:
        assert r['split']=='navtest' and r['frame_count']==14 and r['observation_seconds']==5.
        r['observation_path']=str(ROOT/'observations/navtest'/f"{r['token']}.pt")
        r['metric_cache_path']=str(ROOT/'metric_cache/navtest'/f"{r['token']}.pkl")
    target=ROOT/'manifests/navtest.json'
    if target.exists() and read(target)!=data:raise ValueError('Test population changed')
    write(target,data);write(ROOT/'manifests/navtest_identity.json',dict(source=str(source),source_sha256=sha(source),sha256=sha(target),
        scenes=len(data),logs=len({r['log_name'] for r in data}),purpose='Official fixed test inputs only; no gate, training or selection',
        official_filter=str(official),official_filter_sha256=sha(official),exact_official_token_membership=True,
        historical_frames=4,privileged_metric_path_separate=True))


def verify():
    import torch
    configure();a=agent();errors=[];actions=[]
    for r in rows('train')[:4]+rows('cal')[:4]:
        f=features(a,r);tokens=encode(a,f);old=observation(r)
        assert all(torch.equal(f[k].cpu(),old['features'][k].cpu()) for k in f)
        error=float((tokens.float()-old['tokens'].float()).abs().max());assert error==0.,error
        x,i=conditioning(dict(tokens=tokens,features=f));seeds(seed_for(r['token'],1101))
        with torch.no_grad():new=a.action_head.get_action(x,i)['pred_traj']
        x,i=conditioning(old);seeds(seed_for(r['token'],1101))
        with torch.no_grad():previous=a.action_head.get_action(x,i)['pred_traj']
        ae=float((new-previous).abs().max());assert ae==0.;errors.append(error);actions.append(ae)
    write(ROOT/'tests/real_online_vlm.json',dict(passed=True,scenes=len(errors),max_token_error=max(errors),max_trajectory_error=max(actions),
        VLM_sha256=read(ROOT/'audit/checkpoints.json')['VLM']['sha256'],input_contract='Native current image, navigation command and4 ego history; no answers/future',owner=identity()))


def extract(a):
    import torch
    configure();native=agent();selected=rows(a.split)[a.rank::a.shards];start=time.time();vlm=read(ROOT/'audit/checkpoints.json')['VLM']['sha256']
    for j,r in enumerate(selected):
        path=Path(r['observation_path'])
        if not path.exists():
            f=features(native,r);tokens=encode(native,f)
            save(path,dict(token=r['token'],tokens=tokens,features={k:v.cpu() for k,v in f.items()},cache_format='opd_native_vlm_v1',
                vlm_sha256=vlm,input_contract='Current front image and4 past ego states; native action-head tokens',owner=identity()))
        if a.with_metrics:metric(r)
        if (j+1)%32==0:print('INPUTS',a.rank,j+1,len(selected),time.time()-start,flush=True)
    write(ROOT/'input_jobs'/f'{a.split}_{a.rank}.json',dict(status='COMPLETE',rank=a.rank,shards=a.shards,scenes=len(selected),seconds=time.time()-start,with_metrics=a.with_metrics,owner=identity()))


def metrics(a):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    selected=rows(a.split)[a.rank::a.shards];start=time.time()
    with ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for j,_ in enumerate(pool.map(metric,selected,chunksize=1)):
            if (j+1)%128==0:print('METRICS',a.rank,j+1,len(selected),time.time()-start,flush=True)
    write(ROOT/'input_jobs'/f'metrics_{a.split}_{a.rank}.json',dict(status='COMPLETE',rank=a.rank,shards=a.shards,scenes=len(selected),seconds=time.time()-start,owner=identity()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['manifest','verify','extract','metrics']);p.add_argument('--split',default='navtest');p.add_argument('--rank',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--with-metrics',action='store_true');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.action=='manifest':prepare_manifest()
    elif a.action=='verify':verify()
    elif a.action=='metrics':metrics(a)
    else:extract(a)
