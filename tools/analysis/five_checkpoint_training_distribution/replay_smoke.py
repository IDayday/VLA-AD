"""Read-only replay of cached evaluation draws to verify current runtime provenance."""
import argparse, inspect
from common_fd import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True,choices=MODELS);a=parser.parse_args()
    import torch
    from distributed_rollout import load_planner
    sys.path.insert(0,str(ROOT/'tools/analysis/pc_mts_diagnostics_v2'))
    from gpu_banks import inputs
    torch.cuda.set_device(0);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    # Match the original V1 module requires_grad flags as well as eval state.
    # torch.inference_mode prevents graph construction; no optimizer is created.
    p=load_planner(MODEL_INFO[a.model])
    assert not p.training and next(p.parameters()).dtype==torch.float32
    before={k:v.detach().cpu().clone() for k,v in p.state_dict().items()}
    checks=[]
    for scene in SCENES[:8]:
        token=scene['token'];f=torch.load(v1.OUT/'features'/f'{token}.pt',map_location='cpu',weights_only=False)
        meta=f.pop('_metadata');assert meta['observation_only'] and meta['token']==token
        assert set(f)=={'last_hidden_state','history_trajectory','status_feature','high_command_one_hot'}
        vl,action=inputs(f,'cuda:0')
        init=torch.stack([torch.randn((8,3),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(v1.seed_for(token,'policy',j))) for j in range(16)])
        torch.manual_seed(v1.seed_for(token,'policy_steps',0))
        # Match V1's explicit final float32 conversion, without changing sampling.
        with torch.inference_mode():
            native_out=p.get_action(vl.expand(16,-1,-1),action(16),init_actions=init,deterministic=False)['pred_traj']
            native_dtype=str(native_out.dtype)
            out=native_out.float().cpu().numpy()
        old,_,_=load_model_scene(a.model,token);error=float(np.abs(old[:16]-out).max())
        torch.manual_seed(v1.seed_for(token,'policy_steps',0))
        with torch.inference_mode():repeat=p.get_action(vl.expand(16,-1,-1),action(16),init_actions=init.clone(),deterministic=False)['pred_traj'].float().cpu().numpy()
        repeat_error=float(np.abs(out-repeat).max())
        assert repeat_error==0, 'Current runtime is not deterministically replayable'
        v1.save_npz(OUT/'cache/replay'/a.model/f'{token}.npz',dict(model=a.model,token=token,protocol_sha256=config_identity()),trajectories=out)
        checks.append(dict(token=token,max_abs_error=error,xy_max_abs_error=float(np.abs(old[:16,:,:2]-out[:,:,:2]).max()),
            heading_max_abs_error=float(np.abs(old[:16,:,2]-out[:,:,2]).max()),current_runtime_repeat_error=repeat_error,feature_hash=sha(v1.OUT/'features'/f'{token}.pt')))
    unchanged=all(torch.equal(v,p.state_dict()[k].detach().cpu()) for k,v in before.items())
    runtime=inspect.getfile(type(p))
    result=dict(model=a.model,checkpoint_sha256=sha(MODEL_INFO[a.model]['checkpoint_path']),runtime_path=runtime,runtime_sha256=sha(runtime),
        scene_count=8,rollouts_per_scene=16,max_abs_error=max(c['max_abs_error'] for c in checks),checks=checks,
        parameters_and_buffers_unchanged=unchanged,optimizer_updates=0,network_state='eval',dtype='fp32',
        native_return_dtype=native_dtype,cache_output_dtype='float32',
        protocol='Exact V1 evaluation CRN first16 replay including native V1 final .float() cast; not GRPO train sampling')
    save(OUT/'audits'/f'replay_{a.model}.json',result)
    result['bitwise_cache_replay']=result['max_abs_error']==0
    save(OUT/'audits'/f'replay_{a.model}.json',result)
    assert unchanged,result
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
