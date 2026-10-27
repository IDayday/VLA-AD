#!/usr/bin/env python3
"""New-root entry/import shim for the source-bound local candidate metadata adapter."""
from pathlib import Path
import hashlib
import importlib.util
import sys
sys.dont_write_bytecode = True
ADAPTER_PATH = Path('/mnt/project/VLA-AD/experiments/recogdrive_dc8_refit_20261026/engineering/local_candidate_recheck_20261027/fit_library_local.py')
_APPROVED_SOURCES={
    ADAPTER_PATH:'a1683540c91a9c42045249086d7e0bb08396e68f28db78306441e6115117a55f',
    Path('/mnt/project/VLA-AD/experiments/recogdrive_dc8_refit_20261026/code/fit_library.py'):
        'a86c7fee440909b769d2e17067f1a48aab6094d2871c9d76a124c475ac7c811a',
}
for _path,_expected in _APPROVED_SOURCES.items():
    if hashlib.sha256(_path.read_bytes()).hexdigest()!=_expected:
        raise ValueError('Approved local fitter source changed before import: '+str(_path))
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
