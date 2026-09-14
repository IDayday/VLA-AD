"""Read-only reuse of audited V1 geometry and NAVSIM contracts."""
from __future__ import annotations
import hashlib, json, os, sys, lzma, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools/analysis/pc_mts_diagnostics'))
import common as v1
OUT = ROOT / 'outputs/five_checkpoint_training_distribution'
CONFIG = ROOT / 'configs/five_checkpoint_training_distribution/primary.yaml'
CFG = yaml.safe_load(CONFIG.read_text())
MODELS = CFG['models']
MODEL_INFO = {m['name']: m for m in v1.models()}
SCENES = v1.manifest()['scenes']
ARCHIVES = {
    'mts_8692': Path('/mnt/project/VLA-AD_last_vla_dev/outputs/sg_fps_v6_full_training_20260713T1648Z/support_v6'),
    'mts_8751': Path('/mnt/project/VLA-AD_last_vla_dev/outputs/sg_fps_support_semantic_full_20260705T032521Z/reselect_ref_relative_feasrelax_gtimprover_20260705T185120Z/support_v3'),
}
EPOCHS = {'mts_8692': 165, 'mts_8751': 155}
NATIVE = v1.CODE / 'navsim/agents/recogdrive/recogdrive_diffusion_planner.py'
sha, save, read, digest, distance, pair_mean = v1.sha, v1.save, v1.read, v1.digest, v1.distance, v1.pair_mean

def config_identity():
    frozen = read(OUT / 'manifests/protocol_frozen.json')
    assert sha(CONFIG) == frozen['config_sha256'], 'Frozen protocol changed'
    return frozen['config_sha256']

def load_model_scene(model, token):
    p = v1.OUT / 'rollouts' / model / (token + '.npz')
    q = v1.OUT / 'evaluator/rollouts' / model / (token + '.npz')
    with np.load(p) as z:
        t = z['trajectories'].astype(np.float64)
        meta = json.loads(str(z['metadata']))
    with np.load(q) as z:
        s = z['trajectories'].astype(np.float64)
        smeta = json.loads(str(z['metadata']))
    assert t.shape == (64, 8, 3) and s.shape == (64, 7)
    assert np.isfinite(t).all() and np.isfinite(s).all()
    assert meta['checkpoint_hash'] == MODEL_INFO[model]['sha256']
    assert smeta['input_sha256'] == sha(p)
    return t, s, meta

def teacher_record(model, token):
    path = ARCHIVES[model] / (hashlib.sha1(token.encode()).hexdigest() + '.pkl.xz')
    with lzma.open(path, 'rb') as f:
        rec = pickle.load(f)
    assert rec['token'] == token
    idx = np.asarray(rec['support_indices'], dtype=int)
    assert len(idx) and len(np.unique(idx)) == len(idx)
    return path, rec, idx, np.asarray(rec['candidates'], dtype=np.float64)[idx]

def exact_gt_score(scene):
    """Existing true NAVSIM-v1 score; validate trajectory and scoring identity."""
    token=scene['token'];p=v1.OUT/'raw_candidates'/f'{token}.npz'
    q=v1.OUT/'evaluator/raw_candidates'/f'{token}.npz'
    with np.load(p) as z:
        idx=np.flatnonzero(z['source']=='exact_gt');assert len(idx)==1
        np.testing.assert_allclose(z['trajectories'][idx[0]],np.asarray(scene['gt']),atol=2e-6,rtol=0)
    with np.load(q) as z:
        meta=json.loads(str(z['metadata']));assert meta['input_sha256']==sha(p)
        return z['trajectories'][idx[0]],dict(trajectory_path=str(p),score_path=str(q),trajectory_file_sha256=sha(p),score_file_sha256=sha(q))

def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))

def center(t):
    c = np.asarray(t).mean(0)
    c[:, 2] = np.arctan2(np.sin(t[..., 2]).mean(0), np.cos(t[..., 2]).mean(0))
    return c

def residual(t):
    r = np.asarray(t) - center(t)
    r[..., 2] = wrap(r[..., 2])
    return r

def combine(c, r):
    z = c[None] + r
    z[..., 2] = wrap(z[..., 2])
    return z

def spread(t):
    return np.sqrt(np.var(t[..., :2], axis=0, ddof=1).sum(-1)).mean()

def make_counterfactuals(bank):
    result = {}
    for a, b in CFG['counterfactual']['contrasts']:
        name = a + '__' + b
        result[name + '__new_center_old_residual'] = combine(center(bank[b]), residual(bank[a]))
        result[name + '__old_center_new_residual'] = combine(center(bank[a]), residual(bank[b]))
    a, b = bank['official_il'], bank['grpo_9041']
    ratio = spread(b) / max(spread(a), 1e-12)
    t = a.copy()
    t[..., :2] = center(a)[None, :, :2] + ratio * residual(a)[..., :2]
    result['official_il__grpo_width_only'] = t
    return result

def shapley(q00, q10, q01, q11):
    return .5 * ((q10-q00)+(q11-q01)), .5 * ((q01-q00)+(q11-q10))

def score_metrics(s, threshold=None):
    p = s[:, 6] * 100
    good = v1.feasible(s)
    result = dict(PDMS=p.mean(), median_PDMS=np.median(p), P10_PDMS=np.quantile(p,.1),
                  feasible=good.mean(), hard_failure=v1.hard_failure(s).mean(),
                  zero_score=np.mean(p == 0), oracle64=p.max(), oracle_gap=p.max()-p.mean(),
                  CVaR20=np.sort(p)[:max(1,int(np.ceil(len(p)*.2)))].mean())
    result.update({k:s[:,i].mean() for i,k in enumerate(['NC','DAC','EP','TTC','Comfort','DDC'])})
    if threshold is not None:
        hq = good & (p >= threshold)
        result.update(HQ_mass=hq.mean(), Hit8=hq.reshape(-1,8).any(1).mean(),
                      best8=p.reshape(-1,8).max(1).mean())
    return result

def csv(name, rows):
    p = OUT / 'metrics' / name
    p.parent.mkdir(parents=True,exist_ok=True)
    frame = rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(p,index=False)
    return frame
