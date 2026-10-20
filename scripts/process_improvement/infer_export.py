"""Isolated native exported-student inference from current/past sensor inputs.

This executable deliberately cannot import the process-improvement package and
never opens a teacher, observation cache, metric cache, or supervision file.
"""
import argparse
import importlib.abc
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


class NoResearchInference(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.startswith('navsim.agents.recogdrive.process_improvement'):
            raise ImportError('Export inference may not depend on improvement modules')


def main():
    import torch
    from unittest.mock import patch
    from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
    from navsim.agents.recogdrive.opd.checkpoint_io import restore
    from navsim.agents.recogdrive.opd.conditioning import use_official_conditioning
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from scripts.opd.inputs import features,encode
    from scripts.opd.common import conditioning,configure
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--row',required=True);p.add_argument('--vlm',required=True)
    p.add_argument('--noise',required=True);p.add_argument('--output',required=True)
    p.add_argument('--benchmark-output');p.add_argument('--benchmark-repeats',type=int,default=20)
    a=p.parse_args();configure();sys.meta_path.insert(0,NoResearchInference())
    row=json.loads(Path(a.row).read_text())
    assert set(row)=={'token','log_name','metadata_path','sensor_root','start_idx'}
    opened=[]
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            name=os.fsdecode(args[0]);opened.append(name)
            if ('metric_cache' in name or '/observations/' in name or '/supervision/' in name or
                    'ReCogDrive-2B-RL' in name or 'ReCogDrive-2B-IL' in name):
                raise PermissionError('Inference tried to load a prohibited dependency: '+name)
    sys.addaudithook(audit)
    model=ReCogDriveAgent(TrajectorySampling(time_horizon=4,interval_length=.5),vlm_path=a.vlm,
        checkpoint_path=a.checkpoint,cam_type='single',vlm_type='internvl',dit_type='small',vlm_size='small',
        sampling_method='ddim',cache_hidden_state=False,cache_mode=False,grpo=False,train_backbone=False,
        use_expert_features=False,use_last_rd=False,use_bit_drive=False,use_risk_vla=False,
        use_alignment_loss=False,expert_alignment_weight=0.,allow_random_init=False)
    use_official_conditioning(model.backbone);restore(model.action_head,a.checkpoint)
    model.eval();model.requires_grad_(False)
    f=features(model,row);tokens=encode(model,f)
    raw,inputs=conditioning(dict(tokens=tokens,features=f))
    bank=torch.load(a.noise,map_location='cuda',weights_only=False);trajectories=[]
    for noise in bank:
        draws=iter(noise['steps'][:,i] for i in range(5))
        with torch.no_grad(),patch('torch.randn_like',side_effect=lambda x:next(draws).clone()):
            value=model.action_head.get_action(raw,inputs,deterministic=False,init_actions=noise['initial'].clone())['pred_traj']
        trajectories.append(value.cpu())
    torch.save(dict(trajectories=torch.cat(trajectories),raw_tokens=tokens,opened_files=opened,
                   improvement_imported=any(k.startswith('navsim.agents.recogdrive.process_improvement') for k in sys.modules)),a.output)
    if a.benchmark_output:
        import numpy as np
        target=Path(a.benchmark_output)
        if target.exists():raise FileExistsError('Completed latency measurement is immutable')
        if a.benchmark_repeats<5:raise ValueError('At least five timing repetitions required')
        def native(raw,inputs):
            with torch.no_grad():
                return model.action_head.get_action(raw,inputs,deterministic=False)['pred_traj']
        timing_seeds=[7101,7102,7103,7104,7105]
        def prepare_seed(index):
            seed=timing_seeds[index%len(timing_seeds)]
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        expected_native=[]
        for j in range(len(timing_seeds)):
            prepare_seed(j);expected_native.append(native(raw,inputs).cpu())
        measurements={}
        for full in [False,True]:
            label='online_features_VLM_and_native_head' if full else 'native_head_given_raw_VLM_tokens'
            def forward():
                if full:
                    current=features(model,row);hidden=encode(model,current)
                    current_raw,current_inputs=conditioning(dict(tokens=hidden,features=current))
                    return native(current_raw,current_inputs)
                return native(raw,inputs)
            for j in range(5):
                prepare_seed(j);result=forward()
                torch.testing.assert_close(result.cpu(),expected_native[j%len(timing_seeds)],atol=0,rtol=0)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            resident=torch.cuda.memory_allocated();seconds=[]
            for j in range(a.benchmark_repeats):
                prepare_seed(j);torch.cuda.synchronize();start=time.perf_counter()
                result=forward()
                torch.cuda.synchronize();seconds.append(time.perf_counter()-start)
                torch.testing.assert_close(result.cpu(),expected_native[j%len(timing_seeds)],atol=0,rtol=0)
            peak=torch.cuda.max_memory_allocated()
            measurements[label]=dict(warmup_calls=5,repeats=len(seconds),seconds=seconds,
                mean_seconds=float(np.mean(seconds)),median_seconds=float(np.median(seconds)),
                p95_seconds=float(np.quantile(seconds,.95)),resident_GPU_bytes=resident,
                peak_total_GPU_bytes=peak,peak_incremental_GPU_bytes=peak-resident)
        target.parent.mkdir(parents=True,exist_ok=True)
        import hashlib
        def file_hash(path):
            value=hashlib.sha256()
            with open(path,'rb') as stream:
                for block in iter(lambda:stream.read(8<<20),b''):value.update(block)
            return value.hexdigest()
        target.write_text(json.dumps(dict(status='COMPLETE',token=row['token'],
            device=torch.cuda.get_device_name(),native_K5_single_trajectory=True,
            no_teacher_scorer_future_or_improvement_dependency=True,source_checkpoint=a.checkpoint,
            checkpoint_sha256=file_hash(a.checkpoint),VLM_sha256=file_hash(Path(a.vlm)/'model.safetensors'),
            script_sha256=file_hash(__file__),
            timing_seeds=timing_seeds,randomness='Original get_action draws initial and step noise inside timed call; RNG reset and output assertions outside timing. Timing Torch RNG seeds are distinct from stateless deployment-score noise plans.',
            total_head_parameters=sum(p.numel() for p in model.action_head.parameters()),
            native_head_parameters=sum(p.numel() for n,p in model.action_head.named_parameters() if not n.startswith('action_aware_aux_head.')),
            VLM_dtype=str(next(model.backbone.parameters()).dtype),
            head_dtype=str(next(model.action_head.parameters()).dtype),measurements=measurements,
            caveat='One fixed real scene, five paired noises, repeated warm measurements; not scene-population latency CI. End-to-end includes input preparation/image IO, VLM and head; excludes model loading. Full frozen VLM remains resident for head-only measurement.'),indent=2)+'\n')
        print('REAL_NATIVE_STUDENT_LATENCY_MEASURED',a.benchmark_output,flush=True)


if __name__=='__main__':main()
