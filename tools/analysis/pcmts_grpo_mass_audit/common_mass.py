"""Independent mass-audit namespace; old analyses are read-only dependencies."""
from pathlib import Path
import os, sys, json, hashlib, datetime, functools
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/pcmts_grpo_mass_audit'
CONFIG=ROOT/'configs/pcmts_grpo_mass_audit/primary.yaml'
V1=ROOT/'outputs/pc_mts_diagnostics'
for name in ['pc_mts_diagnostics','pc_mts_diagnostics_v2','pc_mts_diagnostics_v3']:
    sys.path.append(str(ROOT/'tools/analysis'/name))
from common import sha, digest, read, distance, feasible, hard_failure
from common import save as _save, save_npz as _npz, valid_npz

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
@functools.lru_cache(None)
def cfg():return yaml.safe_load(CONFIG.read_text())
@functools.lru_cache(None)
def protocol():return sha(CONFIG)
def scenes():return read(OUT/'manifests/scenes_1000.json')['scenes']
def guard(path):
    p=Path(path).resolve();assert p==OUT or OUT in p.parents, p
    return p
def save(path,value):_save(guard(path),value)
def npz(path,meta,**arrays):_npz(guard(path),dict(protocol_hash=protocol(),**meta),**arrays)
def valid(path,meta=None):return valid_npz(path,dict(protocol_hash=protocol(),**(meta or {})))
def random_seed(token,stream,group):
    # One independent 64-bit counter stream per true native group. Member and
    # denoising-step identify ordered draws within that group's random stream.
    return int(digest([protocol(),cfg()['seed'],token,stream,int(group)])[:16],16) % (2**63-1)
def sample_id(token,stream,group,member,step):
    return digest([protocol(),token,stream,int(group),int(member),int(step)])
def content_hash(traj):
    # Fixed micrometre precision for exact identity, not near-mode counting.
    a=np.round(np.asarray(traj,dtype=np.float64),6);a[a==0]=0
    return hashlib.sha256(a.astype('<f8').tobytes()).hexdigest()

FIELDS=['NC','DAC','EP','TTC','Comfort','DDC','PDMS']
METHODS=['score','pareto','gt_geometry','grpo_mass_pc']
