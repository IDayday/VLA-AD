"""Experiment identity, atomic artifacts, and deployment admission."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

SOURCE = Path(__file__).resolve().parents[2]
PROJECT = Path('/mnt/project/VLA-AD')
ROOT = PROJECT / 'outputs/trajectory_effect_validation_20261019'
PYTHON = '/root/miniconda3/envs/navsim/bin/python'
RL = PROJECT / 'checkpoints/recogdrive/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt'
IL = PROJECT / 'checkpoints/recogdrive/ReCogDrive-2B-IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt'
VLM = PROJECT / 'checkpoints/recogdrive/ReCogDrive-VLM-2B'
TRAIN_MANIFEST = PROJECT / 'outputs/simscale_sft_20260918/manifests/navtrain.json'
AUDIT_MANIFEST = Path('/mnt/project/ddp-foresight-artifacts/20260928/student_dev_v1/index.json')
METRIC_ROOT = Path('/mnt/project/container_backups/container_root_backup_20260709T073417Z/workspace/recogdrive_stage3_rl/cache/metric_cache_train_full')
FIELDS = ['no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress',
          'time_to_collision_within_bound', 'comfort', 'driving_direction_compliance', 'score']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    import numpy as np
    def default(x):
        if isinstance(x, Path): return str(x)
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=default) + '\n')
    tmp.replace(path)


def seed_for(token, seed, stream='inference'):
    return int(digest([token, seed, stream])[:15], 16) % (2 ** 31 - 1)


def require_host():
    policy = read('/mnt/project/server_dispatch_policy.json')
    host = socket.gethostname().removesuffix('-worker-0')
    if host not in policy['allowed_hosts'] or host in policy.get('global_dispatch_prohibitions', {}):
        raise PermissionError(f'Host is not authorized: {host}')
    return host


def runtime_identity():
    return dict(host=require_host(), pid=os.getpid(),
                start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
                started_unix=time.time(), source=str(SOURCE),
                commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip())


def scene_rows(split):
    return read(ROOT / 'manifests' / (split + '.json'))


def driver_record(log):
    """nuPlan log segments share date/time/vehicle before the final time bounds."""
    return log.rsplit('_', 2)[0]


def atomic_npz(path, metadata, **arrays):
    import numpy as np
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp, metadata=json.dumps(metadata, sort_keys=True), **arrays)
    tmp.replace(path)


def protocol():
    import yaml
    lock=read(ROOT/'protocol.lock.json')
    if sha(lock['path'])!=lock['sha256']:
        raise ValueError('Locked protocol changed')
    return yaml.safe_load(Path(lock['path']).read_text()),lock['sha256']


def atomic_torch(path,value):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    torch.save(value,tmp);tmp.replace(path)
