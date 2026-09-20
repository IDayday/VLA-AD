"""Isolated output namespace; reuse audited native runtime without changing it."""
from pathlib import Path
import sys, os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('NUPLAN_MAPS_ROOT', '/mnt/navsim/maps')
os.environ.setdefault('NUPLAN_MAP_VERSION', 'nuplan-maps-v1.0')
WORK=Path(__file__).resolve().parents[3]
MAIN=Path('/mnt/project/VLA-AD')
OLDOUT=MAIN/'outputs/historical_sft_grpo_support'
sys.path.insert(0,str(MAIN/'tools/analysis/historical_sft_grpo_support'))
import common_support as base
base.OUT=WORK/'outputs/il_rl_psi_distribution_5000'
base.CFG_PATH=WORK/'configs/il_rl_psi_distribution_5000/primary.yaml'
base.CFG=base.yaml.safe_load(base.CFG_PATH.read_text())
from common_support import *

def feature_path(token):
    old=MAIN/'outputs/pc_mts_diagnostics/features'/f'{token}.pt'
    return old if old.exists() else OUT/'cache/features'/f'{token}.pt'

def bank_path(model,token,score=False):
    suffix=Path('cache/scores/rollouts' if score else 'cache/rollouts')/model/f'{token}.npz'
    old=OLDOUT/suffix
    return old if old.exists() else OUT/suffix

def inputs(token):
    import torch
    from transformers.feature_extraction_utils import BatchFeature
    f=torch.load(feature_path(token),map_location='cpu',weights_only=False)
    assert f['_metadata']['observation_only']
    assert set(f)=={'_metadata','last_hidden_state','history_trajectory','status_feature','high_command_one_hot'}
    vl=f['last_hidden_state'].unsqueeze(0).cuda().float()
    h=f['history_trajectory'].reshape(1,-1).cuda().float()
    s=f['status_feature'].unsqueeze(0).cuda().float()
    c=f['high_command_one_hot'].unsqueeze(0).cuda().float()
    def action(n):
        return BatchFeature(data={'his_traj':h.expand(n,-1),'history_trajectory':h.reshape(1,4,3).expand(n,-1,-1),'status_feature':s.expand(n,-1),'high_command_one_hot':c.expand(n,-1),'state':torch.cat([s,h],1).expand(n,-1)})
    return vl,action
