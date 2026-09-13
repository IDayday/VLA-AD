"""V2-only writes; V1 routines are imported only as immutable readers/pure functions."""
import sys,os,json,hashlib
from pathlib import Path
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[3]
V1=ROOT/'outputs/pc_mts_diagnostics'
OUT=ROOT/'outputs/pc_mts_diagnostics_v2'
V1TOOLS=ROOT/'tools/analysis/pc_mts_diagnostics'
sys.dont_write_bytecode=True
sys.path.insert(0,str(V1TOOLS))
sys.path.insert(0,str(OUT/'deps'))
import common as v1
from common import sha,digest,read,feasible,hard_failure,distance,pair_mean,FIELDS,MODELS,METHODS,LABELS,PY,CODE
CFG=yaml.safe_load((ROOT/'configs/pc_mts_diagnostics_v2/analysis.yaml').read_text())

def guarded(path):
    p=Path(path).absolute();assert str(p).startswith(str(OUT)+'/'),f'V2 write outside output root: {p}';return p
def save(path,value):return v1.save(guarded(path),value)
def npz(path,meta,**arrays):return v1.save_npz(guarded(path),meta,**arrays)
def valid(path,ident):return v1.valid_npz(path,ident)
def scenes():return read(V1/'manifests/scenes_1000.json')
def ident():return dict(v1_commit='2dd4b55fc658e6182aaf857bf3267b652c873cc7',scene_manifest_hash=digest(scenes()),config_hash=digest(CFG))
def seed(token,namespace,index=0):return int(digest([CFG['seed'],token,namespace,index])[:16],16)&((1<<63)-1)
def make_dirs():
    for name in ['manifests','report','logs','il_banks','raw_candidates','positions','selection_local','candidate_pools','heldout','evaluator','metrics','figures','denoising','chains']:(OUT/name).mkdir(parents=True,exist_ok=True)
def subset(name):return read(OUT/'manifests'/f'{name}_scenes.json')['tokens']
def region(q):return np.where(q<50,'core',np.where(q<95,'boundary','far'))
