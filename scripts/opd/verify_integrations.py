"""Reproducible real-data checks formerly run as individual audit commands."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
from scripts.opd.common import *

def equivalent(a,b):
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    return a==b

def resume():
    import torch
    a,b=[torch.load(ROOT/'runs'/name/'latest.pt',map_location='cpu',weights_only=False) for name in ['resume_v2_full','resume_v2_split']]
    fields={k:equivalent(a[k],b[k]) for k in ['student','optimizer','torch_rng','cuda_rng','numpy_rng','python_rng','forward_counts']}
    assert a['step']==b['step']==4 and all(fields.values()),fields
    write(ROOT/'tests/real_resume_v2.json',dict(passed=True,fields=fields,policy_probe_recording_enabled=True,scope='Continuous4 vs2+resume2; actual corrected observations'))

def grpo_resume():
    import torch
    a,b=[torch.load(ROOT/'runs'/name/'latest.pt',map_location='cpu',weights_only=False) for name in ['E6_resume_full','E6_resume_split']]
    fields={k:equivalent(a[k],b[k]) for k in ['full_student','optimizer','torch_rng','cuda_rng','numpy_rng','python_rng','forward_counts']}
    assert a['step']==b['step']==4 and all(fields.values()),fields
    assert a['forward_counts']['teacher_examples']>0 and a['forward_counts']['reference_examples']>0
    write(ROOT/'tests/real_grpo_resume.json',dict(passed=True,fields=fields,forward_counts=a['forward_counts'],scope='Native DiffGRPO/OPD continuous4 vs2+resume2 actual optimizer updates, exact full state'))

def ddp():
    import torch
    a,b=[torch.load(ROOT/'runs'/name/'latest.pt',map_location='cpu',weights_only=False) for name in ['ddp_smoke_single','ddp_smoke_fixed']]
    error=max(float((v-b['student'][k]).abs().max()) for k,v in a['student'].items() if v.is_floating_point() and not k.startswith('eta.'))
    relative=max(abs(x['grad_norm']-y['grad_norm'])/max(abs(x['grad_norm']),1e-20) for x,y in zip(a['history'],b['history']))
    assert a['step']==b['step']==4 and error<1e-6 and relative<1e-4,(error,relative)
    write(ROOT/'tests/real_ddp.json',dict(passed=True,world=2,updates=4,max_parameter_error=error,global_gradient_norm_relative_error=relative,
        scope='Same scene schedule, noises, globalbatch8, actual corrected observations'))

def gt_parity():
    from scripts.opd.inputs import log
    from navsim.common.dataclasses import Scene,SensorConfig
    error=0.
    for row in rows('train')[:64]:
        frames=[dict(f,lidar_path=f.get('lidar_path') or '') for f in log(row['metadata_path'])[row['start_idx']:row['start_idx']+row['frame_count']]]
        scene=Scene.from_scene_dict_list(frames,Path(row['sensor_root']),4,row['frame_count']-4,SensorConfig.build_no_sensors(),load_image_path=True)
        reference=scene.get_future_trajectory(8).poses
        error=max(error,float(abs(reference-gt(row).numpy()).max()))
    assert error<=1e-5,error
    write(ROOT/'tests/real_GT_parity.json',dict(passed=True,scenes=64,max_error=error,reference='Native Scene.get_future_trajectory(8)',scope='GT supervision only, never observation'))

def auxiliary():
    import torch
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head
    from navsim.agents.recogdrive.opd.losses import diffusion_loss
    configure();s=build_head(config()['student_init']);tokens,inputs=conditioning(observation(rows('train')[0]))
    def forbidden(*args,**kwargs):raise AssertionError('Disabled auxiliary called')
    s.action_aware_aux_head.forward=forbidden
    with torch.no_grad():s.get_action(tokens,inputs)
    loss=diffusion_loss(s,tokens,inputs,gt(rows('train')[0]).unsqueeze(0).cuda());loss.backward()
    assert all(p.grad is None and not p.requires_grad for p in s.action_aware_aux_head.parameters())
    write(ROOT/'tests/real_disabled_auxiliary.json',dict(passed=True,actual_inference_and_diffusion=True,unused_auxiliary_tensors=6,gradient_none=True))

def shared_lock():
    import fcntl
    from scripts.opd.resources import remote
    path=ROOT/'resources/real_lock_test';path.parent.mkdir(exist_ok=True)
    with open(path,'a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        evidence=remote('training-rl-zt2',f"""import fcntl,json
f=open({str(path)!r},'a')
try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);blocked=False
except BlockingIOError:blocked=True
print(json.dumps(dict(blocked=blocked)))
""")
        assert evidence['blocked']
    write(ROOT/'tests/real_shared_lock.json',dict(passed=True,cross_host_lock_exclusion=True,hosts=['training-vla-zt','training-rl-zt2']))

def parallel():
    data=rows('cal')[:16];errors=[];score_errors=[]
    for row in data:
        dirs=[ROOT/'evaluation/cal'/name for name in ['E0_serial_profile','E0_parallel_profile']]
        a,b=[np.load(d/'predictions'/f"{row['token']}.npz") for d in dirs]
        for name in ['IL','RL','COPY','IL_screen']:
            errors.append(float(abs(a[name]-b[name]).max()))
            x,y=[np.asarray(read(d/'scores'/f"{row['token']}.json")['scores'][name]['scores']) for d in dirs]
            score_errors.append(float(abs(x-y).max()))
        a.close();b.close()
    assert max(errors)==max(score_errors)==0.
    serial=read(ROOT/'jobs/profile_serial/complete.json')['seconds'];concurrent=read(ROOT/'jobs/profile_parallel/complete.json')['seconds']
    write(ROOT/'tests/real_parallel_inference.json',dict(passed=True,scenes=16,model_scene_pairs=64,processes_per_gpu=4,trajectory_max_error=max(errors),canonical_score_max_error=max(score_errors),
        serial_seconds=serial,parallel_seconds=concurrent,speedup=serial/concurrent,scope='Independent native batch1 outputs, corrected official observations, no numerical changes'))
    write(ROOT/'resources/evaluation_concurrency.json',dict(parity_passed=True,processes_per_gpu=4,trajectory_max_error=0.,canonical_score_max_error=0.,evidence=str(ROOT/'tests/real_parallel_inference.json'),enabled=time.time()))

def new_host():
    errors=[];scores=[]
    for row in rows('cal')[:16]:
        old=ROOT/'evaluation/cal/E0';new=ROOT/'evaluation/cal/E0_zt5_qualification'
        with np.load(old/'predictions'/f"{row['token']}.npz") as a,np.load(new/'predictions'/f"{row['token']}.npz") as b:
            for key in ['IL','RL','COPY','IL_screen']:
                errors.append(float(abs(a[key]-b[key]).max()))
                x,y=[np.asarray(read(d/'scores'/f"{row['token']}.json")['scores'][key]['scores']) for d in [old,new]]
                scores.append(float(abs(x-y).max()))
    assert max(errors)==max(scores)==0.,(max(errors),max(scores))
    write(ROOT/'tests/real_new_host.json',dict(passed=True,host='training-vlawm-tsinghua-zt5',scenes=16,model_scene_pairs=64,
        trajectory_max_error=max(errors),canonical_score_max_error=max(scores),scope='Exact native4-process cross-host routing qualification; not a new method or additional inference draws'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['resume','grpo_resume','ddp','gt_parity','auxiliary','shared_lock','parallel','new_host']);a=p.parse_args();globals()[a.action]()
