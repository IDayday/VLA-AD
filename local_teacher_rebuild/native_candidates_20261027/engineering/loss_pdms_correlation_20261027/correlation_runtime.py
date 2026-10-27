"""Actual A-checkpoint diagnostics; frozen training/primary selection unchanged."""
from __future__ import annotations
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEST_SCOPE = ROOT / 'engineering/test_pdms_selection_20261027'
CODE = Path('/mnt/project/recogdrive-no-routing-round1-20261026')
RESULTS = Path('/mnt/project/recogdrive-no-routing-round1-results-20261026')
RUN = RESULTS / 'H_full_seed20261027'
EPOCHS = (1, 4, 8, 12, 16)
EXTRA_TEST_EPOCHS = (1, 4, 12)
sys.path[:0] = [str(HERE), str(TEST_SCOPE), str(CODE)]
from student_distillation.checkpointing import atomic_json, digest, read_checkpoint, sha256


def verify_scope():
    import test_pdms_runtime
    test_pdms_runtime.verify_scope()
    freeze = json.loads((HERE / 'SCOPE_FREEZE.json').read_text())
    for name, checksum in freeze['files'].items():
        if sha256(HERE / name) != checksum:
            raise ValueError('Correlation diagnostic source/authority changed: ' + name)
    request = json.loads((HERE / 'REQUEST.json').read_text())
    if request['epochs'] != list(EPOCHS) or request['run_id'] != RUN.name:
        raise ValueError('Diagnostic checkpoint plan changed')
    return request


def checkpoint_identity(epoch):
    if type(epoch) is not int or epoch not in EPOCHS:
        raise ValueError('Unregistered loss-correlation checkpoint')
    identities = json.loads((RUN / 'ASSET_IDENTITIES.json').read_text())
    for item in identities['files'].values():
        if sha256(item['path']) != item['sha256']:
            raise ValueError('An original training asset changed')
    checkpoint = RUN / 'checkpoints' / f'epoch_{epoch:03d}.pt'
    state = read_checkpoint(checkpoint, identities)
    expected = {'epoch_index': epoch, 'global_batch_cursor': 0,
                'global_step': epoch * 1341, 'real_scenes_seen': epoch * 85813}
    if any(state['progress'].get(k) != v for k, v in expected.items()) or state['world_size'] != 4:
        raise ValueError('Incomplete real checkpoint exposure or different world size')
    return {'epoch': epoch, 'checkpoint': str(checkpoint.resolve()), 'sha256': sha256(checkpoint),
            'checkpoint_complete_sha256': sha256(str(checkpoint) + '.COMPLETE.json'),
            'protocol_sha256': identities['protocol_sha256'], 'progress': state['progress']}


def admit_test_checkpoint(cfg, checkpoint, out, protocol):
    verify_scope()
    if cfg['experiment'] != 'H_full' or cfg['training_seed'] != 20261027:
        raise ValueError('Only A seed20261027 is authorized for this diagnostic')
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.parent != (RUN / 'checkpoints').resolve():
        raise ValueError('Wrong diagnostic checkpoint directory')
    epoch = int(checkpoint.stem.removeprefix('epoch_'))
    if epoch not in EXTRA_TEST_EPOCHS:
        raise ValueError('Epoch8/16 must reuse formal Test, not duplicate it')
    identity = checkpoint_identity(epoch)
    destination = RUN / 'diagnostics/loss_pdms_correlation' / f'epoch_{epoch:03d}' / 'test'
    expected = {'population': 'test', 'scene_count': 12146, 'noise_indices': [0, 1, 2, 3, 4],
                'full_population': True, 'mode': 'student', 'artifact_kind': 'behavior',
                'target_version': None, 'checkpoint_sha256': identity['sha256']}
    if (Path(out).resolve() != destination.resolve() or
            any(protocol.get(k) != v for k, v in expected.items()) or
            digest(cfg) != identity['protocol_sha256']):
        raise ValueError('Wrong diagnostic population/configuration/checkpoint')
    protocol.update(test_used_for_model_selection=True, independent_held_out_test=False,
                    analysis='A_checkpoint_loss_PDMS_correlation',
                    analysis_epochs=list(EPOCHS), primary_formal_selection_candidate=False,
                    user_instruction_sha256=sha256(HERE / 'REQUEST.json'))
    return protocol


def evaluator_source(source):
    from test_pdms_runtime import evaluator_source as registered_evaluator_source
    value = registered_evaluator_source(source)
    old = 'from test_pdms_runtime import test_protocol'
    if value.count(old) != 1:
        raise ValueError('Frozen evaluator admission location changed')
    return value.replace(old, 'from correlation_runtime import admit_test_checkpoint as test_protocol', 1)


def install():
    verify_scope()
    import student_distillation.run_suite as suite
    def child_command(module, arguments, world_size):
        if module != 'student_distillation.evaluate':
            raise ValueError('Correlation owner is evaluation-only')
        prefix = [sys.executable, '-B']
        if world_size > 1:
            prefix += ['-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={world_size}']
        return prefix + [str(HERE / 'correlation_worker.py'), '--', *map(str, arguments)]
    suite.child_command = child_command
    return suite
