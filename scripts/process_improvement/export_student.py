"""Export a strict native head and verify real isolated online-image inference."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
from scripts.process_improvement.common import *


def main():
    import torch
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head,state_hash
    from navsim.agents.recogdrive.process_improvement.suffix_rollout import NativeAdapter
    from navsim.agents.recogdrive.process_improvement.rng_plan import RNGKey
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--name',required=True);p.add_argument('--verify',action='store_true')
    a=p.parse_args();configure();cfg=config();dest=ROOT/'exports'/a.name
    head=build_head(a.checkpoint,frozen=True);weights=dest/'student.pt'
    core={k:v.detach().cpu() for k,v in head.state_dict().items() if not k.startswith(('old_policy.','action_aware_aux_head.'))}
    if weights.exists():
        other=build_head(weights,device='cpu',frozen=True)
        assert state_hash(other)==state_hash(head),'Immutable export changed'
    else:save(weights,dict(student=core))
    write(dest/'manifest.json',dict(source_checkpoint=str(a.checkpoint),source_sha256=sha(a.checkpoint),
          exported_sha256=sha(weights),VLM_sha256=cfg['vlm_sha256'],observation_contract=CONTRACT,
          head='native ReCogDrive small DiT 384, 8x3, stochastic DDIM5, no inference search',
          excluded=['teacher','anchor','metric_cache','future_labels','process_improvement_modules']))
    if not a.verify:return
    row=membership('dev')[0];tokens,inputs=conditioning(observation(row));expected=[];bank=[]
    for seed in cfg['evaluation_seeds']:
        key=RNGKey(STUDY,seed,0,0,row['token'],0,-1);noise=key.plan('final_eval',device='cuda')
        expected.append(NativeAdapter(head).sample_rollin(tokens,inputs,noise).trajectory.cpu())
        bank.append(dict(initial=noise.initial.cpu(),steps=noise.steps.cpu()))
    from navsim.agents.recogdrive.process_improvement.cuda_inference import FrozenNativeGraph
    graph=FrozenNativeGraph(head,tokens,inputs);graph.prepare(tokens,inputs)
    accelerated=[graph.sample(RNGKey(STUDY,seed,0,0,row['token'],0,-1).plan('final_eval',device='cuda')).cpu()
                 for seed in cfg['evaluation_seeds']]
    assert torch.equal(torch.cat(accelerated),torch.cat(expected)),'Trained checkpoint CUDA graph parity failed'
    minimal={k:row[k] for k in ['token','log_name','metadata_path','sensor_root','start_idx']}
    write(dest/'online_row.json',minimal);save(dest/'online_noise.pt',bank)
    # Parent GPU allocations are released before the independent deployment process.
    del graph,head,tokens,inputs;torch.cuda.empty_cache()
    subprocess.run([PYTHON,'scripts/process_improvement/infer_export.py','--checkpoint',str(weights),
                    '--row',str(dest/'online_row.json'),'--vlm',cfg['vlm'],
                    '--noise',str(dest/'online_noise.pt'),'--output',str(dest/'online_result.pt')],cwd=SOURCE,check=True)
    result=torch.load(dest/'online_result.pt',map_location='cpu',weights_only=False)
    assert not result['improvement_imported']
    assert torch.equal(result['raw_tokens'],observation(row)['tokens'])
    error=float((result['trajectories']-torch.cat(expected)).abs().max())
    assert error==0.,('Exported online native trajectory changed',error)
    write(dest/'verification.json',dict(status='PASS',token=row['token'],seeds=cfg['evaluation_seeds'],
          trajectory_max_error=error,online_raw_token_max_error=0.,no_teacher_cache_future_or_improvement_module=True,
          trained_checkpoint_cuda_graph_exact=True,
          exported_checkpoint_sha256=sha(weights),opened_files=result['opened_files'],owner=identity()))
    print('EXPORTED_NATIVE_STUDENT_PASS',a.name,error,flush=True)


if __name__=='__main__':main()
