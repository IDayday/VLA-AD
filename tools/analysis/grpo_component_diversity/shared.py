"""Independent namespace, immutable inherited cache and frozen protocol."""
import os, sys, json, hashlib
from pathlib import Path
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/grpo_component_diversity'
PREV=Path('/mnt/project/VLA-AD-worktrees/il-rl-psi-5000-20260920')
sys.path.insert(0,str(PREV/'tools/analysis/il_rl_psi_distribution_5000'))
import common_5000 as previous
import numpy as np
import pandas as pd
import yaml
sha=previous.sha;save=previous.save;npz=previous.npz;read=previous.read
CFG_PATH=ROOT/'configs/grpo_component_diversity/primary.yaml'
CFG=yaml.safe_load(CFG_PATH.read_text())
def identity():
    p=read(OUT/'manifests/protocol.json')
    assert sha(CFG_PATH)==p['config_sha256']
    assert sha(PREV/'outputs/il_rl_psi_distribution_5000/manifests/scenes.json')==p['scenes_sha256']
    return p['config_sha256']
def scenes():return previous.scenes()
def models():return previous.models()
def old_bank(model,token,score=False):return previous.bank_path(model,token,score)
def extra_bank(model,token,score=False):
    return OUT/('cache/scores/rollouts' if score else 'cache/rollouts')/model/f'{token}.npz'
def arrays(model,token,score=False):
    with np.load(old_bank(model,token,score)) as f:a=f['native_grpo'].reshape(64,7) if score else f['native_grpo'].reshape(64,8,3)
    with np.load(extra_bank(model,token,score)) as f:b=f['native_grpo'].reshape(64,7) if score else f['native_grpo'].reshape(64,8,3)
    return np.concatenate([a,b])
def csv(name,rows):
    dest=OUT/'metrics'/name;dest.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(dest,index=False)
