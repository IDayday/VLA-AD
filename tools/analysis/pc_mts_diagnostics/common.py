"""Analysis-only contracts, immutable cache identities, and geometry."""
from __future__ import annotations
import hashlib, json, os, tempfile
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/pc_mts_diagnostics'
CODE = Path('/mnt/project/VLA-AD-worktrees/a5-epdms-stage3')
PY = '/root/miniconda3/envs/navsim/bin/python'
ARCHIVE = Path('/mnt/project/container_backups/container_root_backup_20260709T073417Z/workspace/recogdrive_stage3_rl')
METRIC_CACHE = ARCHIVE / 'cache/metric_cache_train_full'
OLD = ROOT / 'outputs/policy_distribution_5ckpt_1000_20260912'
FIELDS = ['no_at_fault_collisions','drivable_area_compliance','ego_progress','time_to_collision_within_bound','comfort','driving_direction_compliance','score']
MODELS = ['official_il','mts_8692','mts_8751','grpo_9041','apr_9145']
LABELS = ['Official IL','MTS-86.92','MTS-87.51','GRPO-90.41','APR-91.45']
METHODS = ['score','pareto','gt_distance','pc_mts']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    def encode(v):
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,Path):return str(v)
        raise TypeError(type(v).__name__)
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False,default=encode)+'\n');tmp.replace(path)

def save_npz(path,meta,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.stem+f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp,metadata=np.asarray(json.dumps(meta,sort_keys=True)),**arrays);tmp.replace(path)

def valid_npz(path,identity):
    if not Path(path).exists():return False
    with np.load(path,allow_pickle=False) as z:
        found=json.loads(str(z['metadata']))
        if any(found.get(k)!=v for k,v in identity.items()):
            raise RuntimeError(f'Cache identity mismatch: {path}')
    return True

def config(path=None):return yaml.safe_load(Path(path or ROOT/'configs/pc_mts_diagnostics/analysis.yaml').read_text())
def models():return yaml.safe_load((ROOT/'configs/pc_mts_diagnostics/checkpoints.yaml').read_text())
def manifest(path=None):return read(path or OUT/'manifests/scenes_1000.json')
def identity(cfg,scenes,model=None):
    d=dict(scene_manifest_hash=digest(scenes),config_hash=digest(cfg),seed=cfg['seed'])
    if model:d.update(checkpoint_hash=model['sha256'],checkpoint_path=model['checkpoint_path'])
    return d
def pending_pc_tokens(out,cfg,scenes):
    """Only explicitly deferred, audited zero-parent scenes may be skipped."""
    audit=read(Path(out)/'manifests/pc_pending_execution.json')
    assert audit['identity']==identity(cfg,scenes), 'Pending-scene provenance mismatch'
    pending=set(audit['pending_tokens']);completed=set(audit['eligible_tokens'])
    assert not pending & completed
    assert pending | completed == {r['token'] for r in scenes['scenes']}
    for token in pending:
        assert not (Path(out)/'candidate_pools/pc_mts'/f'{token}.npz').exists()
    return pending
def seed_for(token,stream,index=0,seed=20260913):return int(digest([seed,token,stream,index])[:8],16)%(2**31-1)
def distance(a,b):return np.linalg.norm(np.asarray(a)[...,:2][:,None]-np.asarray(b)[...,:2][None,:],axis=-1).mean(-1)
def pair_mean(a):
    if len(a)<2:return float('nan')
    d=distance(a,a);return float(d[np.triu_indices(len(a),1)].mean())
def feasible(scores):
    # Frozen conservative geometric safety definition; comfort reported separately.
    s=np.asarray(scores);return np.all(s[..., [0,1,3,5]]>=1-1e-8,axis=-1)
def hard_failure(scores):
    s=np.asarray(scores);return (s[...,0]<1-1e-8)|(s[...,1]<1-1e-8)
def policy_position(candidates,il,k=5):
    d=distance(il,il);np.fill_diagonal(d,np.inf)
    self_d=np.sort(d,axis=1)[:,:k].mean(1)
    cross=distance(candidates,il);pc=np.sort(cross,axis=1)[:,:k].mean(1)
    # Empirical CDF, right-continuous, explicitly handles ties.
    q=np.searchsorted(np.sort(self_d),pc,side='right')/len(il)*100
    medoid=int(np.where(np.isfinite(d),d,0).sum(1).argmin())
    return dict(d_PC=pc,q_policy=q,nearest_IL_distance=cross.min(1),distance_to_IL_medoid=cross[:,medoid],self_distances=self_d)
