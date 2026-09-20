"""Isolated matched SFT experiment; old data and code are read-only."""
import os,sys,json,hashlib,time,datetime,subprocess
from pathlib import Path
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('NUPLAN_MAPS_ROOT','/mnt/navsim/maps');os.environ.setdefault('NUPLAN_MAP_VERSION','nuplan-maps-v1.0')
import numpy as np,pandas as pd,yaml
WORK=Path(__file__).resolve().parents[3];MAIN=Path('/mnt/project/VLA-AD')
PREV=Path('/mnt/project/VLA-AD-worktrees/il-rl-psi-5000-20260920/outputs/il_rl_psi_distribution_5000')
OUT=WORK/'outputs/psi_matched_sft';CONFIG=WORK/'configs/psi_matched_sft/primary.yaml';CFG=yaml.safe_load(CONFIG.read_text())
RAW=Path('/mnt/project/VLA-AD_last_vla_dev/outputs/sg_fps_v6_full_training_20260713T1648Z/support_v6')
OLD=MAIN/'outputs/historical_sft_grpo_support'
sys.path.insert(0,str(MAIN/'tools/analysis/historical_sft_grpo_support'))
import common_support as legacy
def sha(p):return legacy.sha(p)
def digest(v):return legacy.digest(v)
def read(p):return json.loads(Path(p).read_text())
def save(p,v):legacy.save(p,v)
def npz(p,meta,**v):legacy.npz(p,meta,**v)
def identity():
    h=sha(CONFIG);assert read(OUT/'manifests/protocol.json')['config_sha256']==h;return h
def scenes(split=None):
    rows=read(OUT/'manifests/scenes.json')['scenes'];return [s for s in rows if split is None or s['split']==split]
def feature_path(t):
    paths=[MAIN/'outputs/pc_mts_diagnostics/features'/f'{t}.pt',PREV/'cache/features'/f'{t}.pt',OUT/'cache/features'/f'{t}.pt']
    return next((p for p in paths if p.exists()),paths[-1])
def baseline(t,score=False):
    s=Path('cache/scores/rollouts' if score else 'cache/rollouts')/'official_il'/f'{t}.npz'
    return OLD/s if (OLD/s).exists() else PREV/s
def seed(*parts):return int(digest(parts)[:15],16)%(2**63-1)
def archive(t):return RAW/(hashlib.sha1(t.encode()).hexdigest()+'.pkl.xz')
def table(name,rows):
    p=OUT/'metrics'/name;p.parent.mkdir(parents=True,exist_ok=True);f=pd.DataFrame(rows)
    if p.suffix=='.parquet':f.to_parquet(p,index=False)
    else:f.to_csv(p,index=False)
    return f
def safe(s):return (s[..., [0,1,3,5]]>=1-1e-8).all(-1)
FIELDS=['NC','DAC','EP','TTC','Comfort','DDC','PDMS']
def setup_legacy():
    legacy.OUT=OUT;legacy.CFG_PATH=CONFIG
    legacy.CFG=dict(CFG,protocols=CFG['evaluation']['protocols'],factorial_protocols=[],primary_models=['official_il'])
    legacy.identity=identity
def load_model():
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    sys.path.insert(0,str(MAIN/'scripts/evaluation/distribution_audit'))
    from run import load_planner
    m=read(OLD/'manifests/models.json')['official_il'];p=load_planner(dict(m,path=m['checkpoint_path']))
    return p,m
def feature(t):
    import torch
    f=torch.load(feature_path(t),map_location='cpu',weights_only=False)
    assert f['_metadata']['observation_only'];assert set(f)=={'_metadata','last_hidden_state','history_trajectory','status_feature','high_command_one_hot'}
    if 'token' in f['_metadata']:assert f['_metadata']['token']==t
    return f
def batch_input(fs,targets=None):
    import torch
    from transformers.feature_extraction_utils import BatchFeature
    vl=torch.stack([f['last_hidden_state'] for f in fs]).cuda().float()
    h=torch.stack([f['history_trajectory'].reshape(-1) for f in fs]).cuda().float()
    s=torch.stack([f['status_feature'] for f in fs]).cuda().float();c=torch.stack([f['high_command_one_hot'] for f in fs]).cuda().float()
    data=dict(his_traj=h,history_trajectory=h.reshape(-1,4,3),status_feature=s,high_command_one_hot=c,state=torch.cat([s,h],1))
    if targets is not None:data['action']=torch.as_tensor(np.asarray(targets),device='cuda',dtype=torch.float32)
    return vl,BatchFeature(data=data)
def single_input(t):
    vl,a=batch_input([feature(t)])
    from transformers.feature_extraction_utils import BatchFeature
    def action(n):return BatchFeature(data={k:v.expand(n,*v.shape[1:]) for k,v in a.items()})
    return vl,action

def execution_shard(rows,rank,world):
    """Balance measured host throughput only; union remains the frozen cohort."""
    if world==4:
        buckets=[[0,1],[2],[3,4],[5]][rank]
        return [r for i,r in enumerate(rows) if i%6 in buckets]
    return rows[rank::world]
