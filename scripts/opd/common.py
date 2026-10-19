import hashlib,json,os,pickle,random,socket,subprocess,time
from pathlib import Path
import numpy as np

SOURCE=Path(__file__).resolve().parents[2]
PROJECT=Path(os.environ.get('VLA_AD_PROJECT','/mnt/project/VLA-AD'))
ROOT=Path(os.environ.get('OPD_RUN_ROOT',str(PROJECT/'outputs/recogdrive_opd_20261020')))
REPORTS=SOURCE/'reports/recogdrive_opd_20261020'
PYTHON='/root/miniconda3/envs/navsim/bin/python'


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{os.getpid()}.tmp')
    q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n');q.replace(p)
def save(p,x):
    import torch
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{os.getpid()}.tmp');torch.save(x,q);q.replace(p)
def config(path=None):
    import yaml
    path=Path(path or SOURCE/'configs/opd/base.yaml');c=yaml.safe_load(path.read_text())
    if 'extends' in c:
        base=config(path.parent/c.pop('extends'));base.update(c);c=base
    return c
def identity():
    policy=read('/mnt/project/server_dispatch_policy.json');host=socket.gethostname().removesuffix('-worker-0')
    if host not in policy['allowed_hosts'] or host in policy.get('global_dispatch_prohibitions',{}):raise PermissionError(host)
    return dict(host=host,pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
        code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),time=time.time())
def seed_for(token,seed,stream='inference'):return int(digest([token,seed,stream])[:15],16)%(2**31-1)
def seeds(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def configure():
    import torch
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
def rows(split):return read(ROOT/'manifests'/f'{split}.json')
def observation(row):
    import torch
    cfg=config();p=Path(cfg['observation_cache'])/f"{row['token']}.pt"
    x=torch.load(p,map_location='cpu',weights_only=False)
    if x['token']!=row['token'] or x.get('protocol_sha256')!=cfg['observation_cache_protocol']:raise ValueError('Unbound observation cache')
    if set(x['features'])!={'history_trajectory','status_feature','high_command_one_hot','image_path_tensor'}:raise ValueError('Invalid cached inference inputs')
    return x
def conditioning(payload,device='cuda'):
    import torch
    from transformers.feature_extraction_utils import BatchFeature
    f=payload['features'];history=f['history_trajectory'].to(device=device,dtype=torch.float32).reshape(-1,4,3)
    status=f['status_feature'].to(device=device,dtype=torch.float32)
    return payload['tokens'].to(device=device,dtype=torch.float32),BatchFeature(dict(history_trajectory=history,his_traj=history.reshape(-1,12),
        status_feature=status,high_command_one_hot=f['high_command_one_hot'].to(device),state=torch.cat((status,history.reshape(-1,12)),1)))
def gt(row):
    from tools.training.simscale_sft.common import targets_from_frames
    with open(row['metadata_path'],'rb') as f:frames=pickle.load(f)
    return targets_from_frames(frames[row['start_idx']:row['start_idx']+row['frame_count']])
