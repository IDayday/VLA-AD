"""Independent rerun namespace; earlier studies are immutable inputs."""
import os, sys, json, time, hashlib, socket
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools/analysis/pc_mts_diagnostics_v3'))
import common_v3 as v3
import numpy as np
import yaml
V1, V2, V3 = v3.V1, v3.V2, v3.OUT
OUT = ROOT / 'outputs/pc_mts_v3_stage3_rerun'
CONFIG = ROOT / 'configs/pc_mts_v3_stage3_rerun/primary.yaml'
CFG = yaml.safe_load(CONFIG.read_text())
sha, read, digest = v3.sha, v3.read, v3.digest
PY = v3.PY
def guard(p):
    p = Path(p).resolve()
    assert p.is_relative_to(OUT.resolve()), p
    return p
def save(p, value): return v3.v1.save(guard(p), value)
def npz(p, meta, **arrays): return v3.v1.save_npz(guard(p), meta, **arrays)
def identity():
    f = read(OUT/'manifests/frozen.json')
    assert f['config_sha256'] == sha(CONFIG)
    assert f['split_sha256'] == sha(V3/'manifests/splits.json')
    return {k:f[k] for k in ['config_sha256','split_sha256','base_commit']}
def tokens(part): return v3.tokens(part)
def scenes(): return v3.scenes()
def run_name(method, sd): return f'{method}_seed{sd}'
def initial_path(method, sd):
    return OUT/'initializations'/f'{run_name(method,sd)}.ckpt'
def checkpoint(method, sd, step):
    return OUT/'checkpoints'/run_name(method,sd)/f'step{step:04d}.pt'
def state_only(path):
    import torch
    sd = torch.load(path,map_location='cpu',weights_only=False)['state_dict']
    if any(k.startswith('agent.action_head.') for k in sd):
        sd={k[len('agent.action_head.'):]:v for k,v in sd.items() if k.startswith('agent.action_head.')}
    return {k:v for k,v in sd.items() if not k.startswith('old_policy.')}
def setup(rank=0):
    import torch
    torch.cuda.set_device(rank);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    probe=torch.empty(1,device=f'cuda:{rank}');torch.cuda.synchronize();time.sleep(3)
    return f'cuda:{rank}'
def build_agent(method, sd, grpo=True):
    # Resolve the formal checkout, never the archived V3 planner class.
    sys.path.insert(0,str(ROOT))
    from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    kwargs=dict(CFG['agent']);kwargs['grpo']=grpo
    path=initial_path(method,sd)
    agent=ReCogDriveAgent(trajectory_sampling=TrajectorySampling(time_horizon=4,interval_length=.5),
        checkpoint_path=str(path),reference_policy_checkpoint=str(path),
        metric_cache_path=str(v3.v1.METRIC_CACHE),**kwargs)
    agent.initialize()
    expected=state_only(path);student=agent.action_head.state_dict()
    extra={k for k in student if not k.startswith('old_policy.')} - set(expected)
    assert all(k.startswith('action_aware_aux_head.') for k in extra), extra
    assert set(expected)<=set(student)
    import torch
    assert all(torch.equal(v.cpu(),student[k].cpu()) for k,v in expected.items())
    if grpo:
        assert all(torch.equal(v.cpu(),agent.action_head.old_policy.state_dict()[k].cpu()) for k,v in expected.items())
        assert not any(p.requires_grad for p in agent.action_head.old_policy.parameters())
    # The current checkout already freezes this disabled later auxiliary head.
    if extra:
        assert not agent.action_head.config.use_action_aware_aux
        assert not any(p.requires_grad for p in agent.action_head.action_aware_aux_head.parameters())
    if grpo:
        # Relocate historical mount prefixes to the audited per-token caches.
        agent.action_head.metric_cache_loader.metric_cache_paths={r['token']:r['metric_cache_path'] for r in scenes()}
    return agent
