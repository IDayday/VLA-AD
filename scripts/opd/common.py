import hashlib,json,os,pickle,random,socket,subprocess,time
from functools import lru_cache
from pathlib import Path
import numpy as np

SOURCE=Path(__file__).resolve().parents[2]
PROJECT=Path(os.environ.get('VLA_AD_PROJECT','/mnt/project/VLA-AD'))
ROOT=Path(os.environ.get('OPD_RUN_ROOT',str(PROJECT/'outputs/recogdrive_opd_20261020/official_v2')))
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
def teacher_signature(c):
    return digest({k:c[k] for k in ('teacher_init','vlm','sampling_method','num_inference_steps','rollout_deterministic','rollout_kernel','observation_cache_protocol')})
def verify_teacher_cache(meta,c):
    if meta.get('teacher_config_sha256'):
        if meta['teacher_config_sha256']!=teacher_signature(c):raise ValueError('Teacher cache sampler/model configuration mismatch')
    elif meta['config_sha256']!=digest(config()) or teacher_signature(c)!=teacher_signature(config()):
        raise ValueError('Unbound legacy teacher cache')
    if meta['teacher_sha256']!=read(ROOT/'audit/checkpoints.json')['RL']['sha256']:raise ValueError('Teacher checkpoint identity mismatch')
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
@lru_cache(maxsize=256)
def observation_by_token(token):
    import torch
    cfg=config();p=Path(cfg['observation_cache'])/f'{token}.pt'
    x=torch.load(p,map_location='cpu',weights_only=False)
    if x['token']!=token or x.get('protocol_sha256')!=cfg['observation_cache_protocol']:raise ValueError('Unbound observation cache')
    if set(x['features'])!={'history_trajectory','status_feature','high_command_one_hot','image_path_tensor'}:raise ValueError('Invalid cached inference inputs')
    return x
def observation(row):
    if 'observation_path' not in row:return observation_by_token(row['token'])
    import torch
    x=torch.load(row['observation_path'],map_location='cpu',weights_only=False)
    if x['token']!=row['token'] or x.get('cache_format')!='opd_native_vlm_v1':raise ValueError('Unbound observation cache')
    if x['vlm_sha256']!=read(ROOT/'audit/checkpoints.json')['VLM']['sha256']:raise ValueError('VLM source changed')
    if config().get('official_conditioning',False):
        if x.get('observation_contract')!='recogdrive_official_fixed2800_v2' or x.get('protocol_sha256')!=config()['observation_cache_protocol']:
            raise ValueError('Non-official conditioning cache prohibited in corrected research')
    if set(x['features'])!={'history_trajectory','status_feature','high_command_one_hot','image_path_tensor'}:raise ValueError('Invalid inference keys')
    return x
def conditioning(payload,device='cuda'):
    import torch
    from transformers.feature_extraction_utils import BatchFeature
    f=payload['features'];history=f['history_trajectory'].to(device=device,dtype=torch.float32).reshape(-1,4,3)
    status=f['status_feature'].to(device=device,dtype=torch.float32)
    return payload['tokens'].to(device=device,dtype=torch.float32),BatchFeature(dict(history_trajectory=history,his_traj=history.reshape(-1,12),
        status_feature=status,high_command_one_hot=f['high_command_one_hot'].to(device),state=torch.cat((status,history.reshape(-1,12)),1)))
@lru_cache(maxsize=1)
def gt_bank():
    import torch
    return torch.load(ROOT/'supervision/train_gt.pt',map_location='cpu',weights_only=False)
def gt(row):
    if row['split']!='navtrain' or row['token'] not in gt_bank():raise ValueError('GT requested outside registered training membership')
    return gt_bank()[row['token']]
