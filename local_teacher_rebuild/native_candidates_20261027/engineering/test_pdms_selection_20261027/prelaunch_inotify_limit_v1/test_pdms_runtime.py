"""Explicit user-authorized Test selection without changing frozen training."""
from __future__ import annotations

import ast
import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CODE = Path('/mnt/project/recogdrive-no-routing-round1-20261026')
RESULTS = Path('/mnt/project/recogdrive-no-routing-round1-results-20261026')
OLD_SCOPE = ROOT / 'engineering/single_seed_resource_saving_20261027'
sys.path[:0] = [str(HERE), str(OLD_SCOPE), str(CODE)]
from student_distillation.checkpointing import atomic_json, digest, sha256


def verify_scope():
    """Verify immutable administrative code separately from training identity."""
    manifest = json.loads((HERE / 'SCOPE_FREEZE.json').read_text())
    for name, checksum in manifest['files'].items():
        if sha256(HERE / name) != checksum:
            raise ValueError('Test selection overlay changed: ' + name)
    request = json.loads((HERE / 'REQUEST.json').read_text())
    expected = {'selection_population': 'test', 'test_scenes': 12146,
                'noise_indices': [0, 1, 2, 3, 4], 'selection_epochs': [8, 16, 24, 32, 40],
                'training_seeds': [20261027], 'test_used_for_model_selection': True,
                'independent_held_out_test': False, 'validation_used_for_model_selection': False}
    if any(request.get(k) != v for k, v in expected.items()):
        raise ValueError('Missing latest user Test-PDMS instruction')
    frozen = json.loads((RESULTS / 'protocol/CODE_FREEZE.json').read_text())
    for name, checksum in frozen['code_files'].items():
        if sha256(CODE / name) != checksum:
            raise ValueError('Frozen numerical training source changed: ' + name)
    return request


def test_protocol(cfg, checkpoint, out, protocol):
    """Admit an actual epoch endpoint; never fabricate a preexisting Test lock."""
    verify_scope()
    import test_pdms_selector as selection
    run = RESULTS / f"{cfg['experiment']}_seed{cfg['training_seed']}"
    checkpoint = Path(checkpoint).resolve()
    if Path(cfg['paths']['results_root']).resolve() != RESULTS:
        raise ValueError('Test selection belongs to this registered suite only')
    if checkpoint.parent != (run / 'checkpoints').resolve():
        raise ValueError('Test checkpoint is outside the registered run')
    try:
        epoch = int(checkpoint.stem.removeprefix('epoch_'))
    except ValueError as exc:
        raise ValueError('Test selection requires a sealed epoch checkpoint') from exc
    endpoint = selection.endpoint_checkpoint(run, epoch)
    if checkpoint != Path(endpoint['checkpoint']):
        raise ValueError('Different epoch checkpoint path')
    expected = {'population': 'test', 'scene_count': 12146, 'noise_indices': [0, 1, 2, 3, 4],
                'full_population': True, 'mode': 'student', 'artifact_kind': 'behavior',
                'target_version': None, 'checkpoint_sha256': endpoint['sha256']}
    if any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError('Test selection requires the complete paired formal evaluation')
    if digest(cfg) != endpoint['protocol_sha256']:
        raise ValueError('Test configuration differs from the trained checkpoint')
    destination = Path(out).resolve()
    endpoint_destination = (run / 'test' / f'epoch_{epoch:03d}').resolve()
    if destination != endpoint_destination:
        if destination not in ((run / 'test/best').resolve(), (run / 'test/last').resolve()):
            raise ValueError('Unregistered Test selection destination')
        locked = selection.verify_selection_lock(RESULTS)
        label = 'best' if Path(out).name == 'best' else 'last'
        if locked['runs'][run.name][label]['sha256'] != endpoint['sha256']:
            raise ValueError('Final Test label is not the selected checkpoint')
    extra = {'selection_population': 'test', 'test_used_for_model_selection': True,
             'independent_held_out_test': False, 'validation_used_for_model_selection': False,
             'selection_epochs': list(selection.ENDPOINTS),
             'user_instruction_sha256': sha256(HERE / 'REQUEST.json')}
    protocol.update(extra)
    atomic_json(Path(out) / f'TEST_SELECTION_ADMISSION_rank{os.environ.get("RANK", "0")}.json',
                {'status': 'USER_AUTHORIZED_TEST_SELECTION_ENDPOINT', 'endpoint': endpoint,
                 'user_instruction': selection.request_identity(), **extra})
    return protocol


def evaluator_source(source):
    """Change only Test admission and its protocol annotation, preserving math."""
    begin = source.index('    protocol_sha = digest(protocol)\n    if population == "test":\n')
    end = source.index('    marker = out / "COMPLETE.json"\n', begin)
    replacement = ('    if population == "test":\n'
                   '        from test_pdms_runtime import test_protocol\n'
                   '        protocol = test_protocol(cfg, checkpoint, out, protocol)\n'
                   '    protocol_sha = digest(protocol)\n')
    changed = source[:begin] + replacement + source[end:]
    # Assert every other original function is exactly the same AST.
    before = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    after = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(changed).body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if {k:v for k,v in before.items() if k != 'evaluate_model'} != {
            k:v for k,v in after.items() if k != 'evaluate_model'}:
        raise ValueError('Unexpected numerical evaluator function modification')
    return changed


def wait_for_artifact(path, failed=None):
    """Linux file events, not repeated training-progress/log polling."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'inotify_init1')
    try:
        # Atomic writes arrive as MOVED_TO. CREATE/CLOSE_WRITE cover ordinary writes.
        if libc.inotify_add_watch(fd, os.fsencode(path.parent), 0x80 | 0x100 | 0x8) < 0:
            raise OSError(ctypes.get_errno(), 'inotify_add_watch')
        while True:
            if failed and Path(failed).is_file():
                raise RuntimeError('Test endpoint owner failed: ' + Path(failed).read_text())
            if path.is_file():
                return
            select.select([fd], [], [])
            try:
                os.read(fd, 65536)
            except BlockingIOError:
                pass
    finally:
        os.close(fd)


def install():
    """Adopt the same live trainers; replace only selection/evaluation routing."""
    verify_scope()
    from single_seed_scope import install as single_seed_install
    suite = single_seed_install()
    if getattr(suite, '_test_pdms_scope_installed', False):
        return suite
    import student_distillation.select_checkpoint as original_selection
    import test_pdms_selector as selection
    original_selection.select = selection.select
    original_selection.verify_selection_lock = selection.verify_selection_lock
    active_registry = []
    original_registry = suite.Registry

    class TestSelectionRegistry(original_registry):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            active_registry.append(self)

    suite.Registry = TestSelectionRegistry

    def lock_suite(out):
        wait_for_artifact(HERE / 'TEST_ENDPOINTS_COMPLETE.json', HERE / 'TEST_ENDPOINTS_FAILED.json')
        # Merge measured Test costs once, after the independent owner is done;
        # do not let two live controllers overwrite the same training registry.
        executed = json.loads((HERE / 'TEST_EXECUTION_REGISTRY.json').read_text())
        if len(active_registry) != 1:
            raise ValueError('Expected exactly one authoritative suite registry')
        registry = active_registry[0]
        for key, task in executed['tasks'].items():
            if key in registry.value['tasks'] and registry.value['tasks'][key] != task:
                raise ValueError('Test task conflicts with another measured job')
            registry.value['tasks'][key] = task
        registry.value['test_endpoint_execution'] = {
            'path': str(HERE / 'TEST_EXECUTION_REGISTRY.json'),
            'sha256': sha256(HERE / 'TEST_EXECUTION_REGISTRY.json')}
        registry.save()
        return selection.lock_suite(out)

    original_selection.lock_suite = lock_suite
    original_command = suite.child_command

    def child_command(module, arguments, world_size):
        if module == 'student_distillation.train':
            return original_command(module, arguments, world_size)
        if module != 'student_distillation.evaluate':
            raise ValueError('Unexpected Test-selection child module')
        prefix = [sys.executable, '-B']
        if world_size > 1:
            prefix += ['-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={world_size}']
        return prefix + [str(HERE / 'test_pdms_worker.py'), '--', *map(str, arguments)]

    suite.child_command = child_command
    original_evaluation = suite.evaluation_task

    def evaluation_task(controller, cfg_path, checkpoint, population, destination, **kwargs):
        if population == 'test' and Path(destination).name in ('best', 'last') and not kwargs:
            cfg = suite.load_config(cfg_path)
            run = RESULTS / f"{cfg['experiment']}_seed{cfg['training_seed']}"
            entry = selection.verify_selection_lock(RESULTS)['runs'][run.name][Path(destination).name]
            source = run / 'test' / f"epoch_{entry['epoch']:03d}"
            if sha256(checkpoint) != entry['sha256'] or not suite._same_checkpoint_result(
                    source, destination, entry['sha256']):
                raise ValueError('Cannot reuse the verified actual Test endpoint')
            return
        return original_evaluation(controller, cfg_path, checkpoint, population, destination, **kwargs)

    suite.evaluation_task = evaluation_task
    import report_scope
    report_scope.install()
    suite._test_pdms_scope_installed = True
    return suite
