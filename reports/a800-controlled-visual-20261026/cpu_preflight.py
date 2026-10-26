"""Check actual independent interpreter and portable settings before GPU work."""
from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import runpy
import sys

ROOT = Path(__file__).resolve().parent
FOLLOWUP = ROOT / 'audit_repo/reports/ppu-a800-alignment-followup-20261027'
reference = json.loads((FOLLOWUP / 'PPU_CONTROLLED_CONFIGURATION.json').read_text())
expected = reference['portable_conditions']
import torch
import torchvision

actual = {
    'python_version': platform.python_version(),
    'torch_release': str(torch.__version__).split('+')[0],
    'torch_build': str(torch.__version__),
    'torch_module': torch.__file__,
    'torchvision_module': torchvision.__file__,
    'common_package_versions': {key: metadata.version(key) for key in expected['common_package_versions']},
    'sys_path': sys.path,
    'read_only_dependency_fallback': '/root/miniconda3/envs/ddp/lib/python3.10/site-packages',
    'shared_environment_modified': False,
}
checks = {key: actual[key] == expected[key]
          for key in ['python_version', 'torch_release', 'common_package_versions']}
namespace = runpy.run_path(str(FOLLOWUP / 'controlled_conditions.py'))
namespace['configure_fresh_receiver'](FOLLOWUP / 'PPU_CONTROLLED_CONFIGURATION.json')
actual['numeric_policy_before_CUDA'] = namespace['numeric_policy']()
actual['relevant_environment'] = {key: os.environ.get(key) for key in namespace['ENV_KEYS']}
checks['numeric_policy'] = actual['numeric_policy_before_CUDA'] == expected['numeric_policy']
checks['relevant_environment'] = actual['relevant_environment'] == expected['relevant_environment']
checks['CUDA_not_initialized'] = not torch.cuda.is_initialized()
actual['checks'] = checks
actual['status'] = 'PASS_BEFORE_GPU' if all(checks.values()) else 'STOP_BEFORE_GPU'
(ROOT / 'CPU_ENVIRONMENT_PREFLIGHT.json').write_text(json.dumps(actual, indent=2) + '\n')
print(json.dumps({key: value for key, value in actual.items() if key != 'sys_path'}, indent=2))
assert all(checks.values()), checks
