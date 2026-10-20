"""Isolated native exported-student inference from current/past sensor inputs.

This executable deliberately cannot import the process-improvement package and
never opens a teacher, observation cache, metric cache, or supervision file.
"""
import argparse
import importlib.abc
import json
import os
import sys
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


if __name__=='__main__':main()
