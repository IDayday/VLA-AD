from pathlib import Path
import os,sys,json,hashlib,time
import numpy as np
import pandas as pd
import yaml
ROOT=Path('/mnt/project/VLA-AD')
OUT=ROOT/'outputs/historical_sft_grpo_support'
CFG_PATH=ROOT/'configs/historical_sft_grpo_support/primary.yaml'
CFG=yaml.load(CFG_PATH.read_text(),Loader=yaml.CSafeLoader)
OLD=Path('/mnt/project/VLA-AD_last_vla_dev')
sys.path.insert(0,str(ROOT/'tools/analysis/pc_mts_diagnostics'))
import common as v1
sha=v1.sha;digest=v1.digest;save=v1.save;npz=v1.save_npz;distance=v1.distance;pair_mean=v1.pair_mean
def read(p):return json.loads(Path(p).read_text())
def scenes():return read(OUT/'manifests/scenes.json')['scenes']
def models():return read(OUT/'manifests/models.json')
def identity():
    frozen=read(OUT/'manifests/protocol.json');assert sha(CFG_PATH)==frozen['config_sha256']
    return frozen['config_sha256']
def seed(token,group,stream='shared_sampling'):
    return int(digest([CFG['seed'],token,stream,group])[:15],16)%(2**63-1)
def state_hash(p):
    h=hashlib.sha256()
    for k,v in p.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def csv(name,rows):
    p=OUT/'metrics'/name;p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(p,index=False)
