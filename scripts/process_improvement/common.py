"""Study identity, atomic records, and immutable observation inputs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

SOURCE = Path(__file__).resolve().parents[2]
PROJECT = Path('/mnt/project/VLA-AD')
ROOT = Path(os.environ.get('PROCESS_RUN_ROOT', str(PROJECT / 'outputs/recogdrive_process_improvement_20261020')))
HISTORY = PROJECT / 'outputs/recogdrive_opd_20261020/official_v2'
REPORTS = SOURCE / 'reports/recogdrive_process_improvement_20261020'
PYTHON = '/root/miniconda3/envs/navsim/bin/python'
STUDY = 'recogdrive_process_improvement_20261020'
BASE = '560bb257cd493613130f4b3643a0a2e4a88a7115'
CONTRACT = '98ce682857a3c0b5cc7ca521499db4d8a175444729b472465a15d6adb43f9093'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')
    temp.replace(path)


def save(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(value, temp)
    temp.replace(path)


def configure():
    import torch
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def config(path=None):
    import yaml
    path = Path(path or SOURCE / 'configs/process_improvement/base.yaml')
    result = yaml.safe_load(path.read_text())
    if 'extends' in result:
        parent = config(path.parent / result.pop('extends'))
        parent.update(result)
        result = parent
    return result


def membership(name):
    return read(ROOT / 'manifests' / f'{name}.json')


def observation(row):
    import torch
    path = Path(row['observation_path'])
    value = torch.load(path, map_location='cpu', weights_only=False)
    if (value.get('token') != row['token'] or value.get('protocol_sha256') != CONTRACT or
            value.get('observation_contract') != 'recogdrive_official_fixed2800_v2'):
        raise ValueError(f'Observation identity/contract mismatch: {path}')
    if value['vlm_sha256'] != config()['vlm_sha256']:
        raise ValueError('Observation VLM differs from frozen shared VLM')
    if value['tokens'].shape != (1, 2800, 1536):
        raise ValueError('Raw padded VLM hidden-state contract changed')
    return value


def conditioning(payload, device='cuda'):
    # This wrapper only moves *raw* frozen VLM outputs. Each planner projects them itself.
    from scripts.opd.common import conditioning as move_raw
    return move_raw(payload, device)


def identity():
    import socket
    host = socket.gethostname().removesuffix('-worker-0')
    policy = read(PROJECT.parent / 'server_dispatch_policy.json')
    if host not in policy['allowed_hosts'] or host in policy.get('global_dispatch_prohibitions', {}):
        raise PermissionError(f'Host withdrawn or not authorized: {host}')
    return dict(host=host, pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().split()[21],
                code_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip())
