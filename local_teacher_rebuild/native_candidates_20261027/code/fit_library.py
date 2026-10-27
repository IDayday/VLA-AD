#!/usr/bin/env python3
"""New-root entry/import shim for the source-bound local candidate metadata adapter."""
from pathlib import Path
import importlib.util
import sys
sys.dont_write_bytecode = True
ADAPTER_PATH = Path('/mnt/project/VLA-AD/experiments/recogdrive_dc8_refit_20261026/engineering/local_candidate_recheck_20261027/fit_library_local.py')
_SPEC = importlib.util.spec_from_file_location('nrkd_local_candidate_adapter', ADAPTER_PATH)
_ADAPTER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _ADAPTER
_SPEC.loader.exec_module(_ADAPTER)
_FITTER = _ADAPTER.cloned_fitter(Path(__file__).resolve().parents[1])
# Imported numerical functions keep the original code objects and use private
# globals rooted here. No historical fitter module/global is mutated.
for _name, _value in vars(_FITTER).items():
    if not _name.startswith('__'):
        globals()[_name] = _value

if __name__ == '__main__':
    _ADAPTER.main()
