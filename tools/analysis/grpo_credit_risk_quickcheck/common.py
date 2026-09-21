from pathlib import Path
import os,sys,json,hashlib,time
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('NUPLAN_MAPS_ROOT','/mnt/navsim/maps')
os.environ.setdefault('NUPLAN_MAP_VERSION','nuplan-maps-v1.0')
import numpy as np
import pandas as pd
import yaml
from functools import lru_cache
ROOT=Path('/mnt/project/VLA-AD')
WORK=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/grpo_credit_risk_quickcheck'
RUNTIME=ROOT/'outputs/historical_sft_grpo_support/runtime/a5'
PYTHON='/root/miniconda3/envs/navsim/bin/python'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def seed(*args):return int(digest(args)[:15],16)%(2**63-1)
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp')
 def enc(v):
  if isinstance(v,np.generic):return v.item()
  if isinstance(v,np.ndarray):return v.tolist()
  if isinstance(v,Path):return str(v)
  raise TypeError(type(v))
 t.write_text(json.dumps(x,indent=2,allow_nan=False,default=enc)+'\n');t.replace(p)
def npz(p,metadata,**arrays):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.stem+f'.{os.getpid()}.tmp.npz');np.savez_compressed(t,metadata=np.asarray(json.dumps(metadata,sort_keys=True)),**arrays);t.replace(p)
@lru_cache(maxsize=1)
def cfg():return yaml.safe_load((OUT/'resolved_config.yaml').read_text())
def scenes(split=None,smoke=False):
 x=read(OUT/'manifests/scenes.json');rows=x['scenes']
 if split:rows=[r for r in rows if r['split']==split]
 if smoke:rows=[r for r in rows if r['token'] in x['smoke_tokens']]
 return rows
def install_runtime():
 sys.path.insert(0,str(RUNTIME))
def state_hash(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():
  h.update(k.encode());h.update(str(v.dtype).encode());h.update(v.detach().cpu().contiguous().reshape(-1).view(__import__('torch').uint8).numpy().tobytes())
 return h.hexdigest()
def forbid_updates():
 import torch
 def denied(*a,**k):raise RuntimeError('Optimizer updates forbidden in read-only experiment')
 for cls in [torch.optim.Optimizer,torch.optim.Adam,torch.optim.AdamW,torch.optim.SGD]:cls.step=denied
 torch.Tensor.backward=denied
