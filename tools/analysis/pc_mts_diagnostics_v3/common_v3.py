"""V3 write boundary, immutable V1/V2 readers and frozen protocol identity."""
import os, sys, json, hashlib, time
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
V1 = ROOT / 'outputs/pc_mts_diagnostics'
V2 = ROOT / 'outputs/pc_mts_diagnostics_v2'
OUT = ROOT / 'outputs/pc_mts_diagnostics_v3'
sys.path.insert(0, str(ROOT / 'tools/analysis/pc_mts_diagnostics'))
sys.path.insert(0, str(V2 / 'deps'))
import numpy as np
import pandas as pd
import yaml
import common as v1
from common import sha, digest, read, distance, pair_mean, feasible, hard_failure, FIELDS, PY
CONFIG_PATH = ROOT / 'configs/pc_mts_diagnostics_v3/primary.yaml'
CFG = yaml.safe_load(CONFIG_PATH.read_text())
METHODS = ['score','pareto','gt_distance','old_pc','conditional_pc']
LABELS = {'score':'Score','pareto':'Global-Pareto','gt_distance':'GT-distance','old_pc':'Old-PC-V2','conditional_pc':'Conditional-PC','gt_only':'GT-only','quality_only':'Quality-only ablation'}
def guard(path):
    p = Path(path).resolve()
    assert p.is_relative_to(OUT.resolve()), ('V3 write outside namespace', p)
    return p
def save(path, value): return v1.save(guard(path), value)
def npz(path, meta, **arrays): return v1.save_npz(guard(path), meta, **arrays)
def valid(path, meta): return v1.valid_npz(path, meta)
def csv(df, name):
    p=guard(OUT/'metrics'/name);p.parent.mkdir(parents=True,exist_ok=True);df.to_csv(p,index=False)
def parquet(df, name): df.to_parquet(guard(OUT/'metrics'/name),index=False)
def scenes(): return read(V1/'manifests/scenes_1000.json')['scenes']
def seed(token, namespace, index=0): return int(digest([CFG['seed'],str(token),namespace,index])[:16],16)&((1<<63)-1)
def identity():
    frozen=OUT/'manifests/protocol_frozen.json'
    if frozen.exists(): assert read(frozen)['yaml_sha256']==sha(CONFIG_PATH), 'Frozen V3 configuration changed'
    return dict(base_commit=CFG['base_commit'],config_sha256=sha(CONFIG_PATH),scene_sha256=sha(V1/'manifests/scenes_1000.json'),checkpoint_sha256=v1.models()[0]['sha256'])
def tokens(name): return read(OUT/'manifests/splits.json')[name]
def region(q): return np.where(np.asarray(q)<50,'core',np.where(np.asarray(q)<95,'boundary','far'))
def stage_audit(stage, **kw):
    save(OUT/'manifests'/f'audit_{stage}.json',dict(identity=identity(),completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw))
def bootstrap(values, key='default'):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    if not len(a): return dict(n=0,mean=None,median=None,ci_low=None,ci_high=None,win_fraction=None)
    rng=np.random.default_rng(seed(key,'v3_bootstrap'));n=len(a);means=[];medians=[]
    for begin in range(0,CFG['bootstrap_replicates'],100):
        b=a[rng.integers(n,size=(min(100,CFG['bootstrap_replicates']-begin),n))];means.extend(b.mean(1));medians.extend(np.median(b,axis=1))
    return dict(n=n,mean=float(a.mean()),median=float(np.median(a)),ci_low=float(np.quantile(means,.025)),ci_high=float(np.quantile(means,.975)),median_ci_low=float(np.quantile(medians,.025)),median_ci_high=float(np.quantile(medians,.975)),win_fraction=float((a>1e-10).mean()))
def unique_indices(trajectories):
    seen=set();out=[]
    for i,t in enumerate(np.asarray(trajectories,dtype=np.float32)):
        b=t.tobytes()
        if b not in seen: seen.add(b);out.append(i)
    return np.asarray(out,dtype=int)
