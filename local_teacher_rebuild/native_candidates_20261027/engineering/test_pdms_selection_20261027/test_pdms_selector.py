"""User-authorized full-Test PDMS selection, separate from frozen training code.

Test is explicitly a model-selection population in this protocol. Its best
checkpoint score is not an independently held-out generalization estimate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

CODE = Path('/mnt/project/recogdrive-no-routing-round1-20261026')
sys.path.insert(0, str(CODE))
from student_distillation.checkpointing import atomic_json, digest, read_checkpoint, sha256
from student_distillation.evaluate import verify_evaluation

ROOT = Path(__file__).resolve().parent
ENDPOINTS = (8, 16, 24, 32, 40)
NOISE_INDICES = (0, 1, 2, 3, 4)
TEST_SCENES = 12146
RUNS = tuple(f'{arm}_seed20261027' for arm in
             ('H_full', 'H_gain1', 'H_mu_teacher', 'H_mu_student'))
LOCK_NAME = 'TEST_PDMS_SELECTION_LOCK.json'
BEST_NAME = 'test_pdms_best.json'


def request_identity():
    path = ROOT / 'REQUEST.json'
    value = json.loads(path.read_text())
    return {'path': str(path), 'sha256': sha256(path)}


def _selection_protocol_sha256():
    return digest({'endpoints': ENDPOINTS, 'noise_indices': NOISE_INDICES,
                   'population': 'test', 'scene_count': TEST_SCENES,
                   'tie_break': 'earlier_epoch', 'Test_is_used_for_selection': True})


def _identities(run_dir):
    value = json.loads((run_dir / 'ASSET_IDENTITIES.json').read_text())
    for name in ('feature_manifest', 'teacher_bundle', 'targets_manifest', 'calibration'):
        item = value['files'][name]
        if sha256(item['path']) != item['sha256']:
            raise ValueError(f'Frozen training asset changed: {name}')
    return value


def endpoint_checkpoint(run_dir, epoch):
    """Admit only a complete registered epoch checkpoint; no future-step gate."""
    run_dir = Path(run_dir).resolve()
    if run_dir.name not in RUNS or type(epoch) is not int or epoch not in ENDPOINTS:
        raise ValueError('Test selection accepts only the four retained runs and five endpoints')
    identities = _identities(run_dir)
    checkpoint = run_dir / 'checkpoints' / f'epoch_{epoch:03d}.pt'
    state = read_checkpoint(checkpoint, identities)
    expected = {'global_step': epoch * 1341, 'real_scenes_seen': epoch * 85813,
                'epoch_index': epoch, 'global_batch_cursor': 0}
    if any(state['progress'].get(k) != v for k, v in expected.items()):
        raise ValueError('Checkpoint endpoint has an incomplete actual training exposure')
    if state.get('world_size') != identities['qualification']['execution']['world_size']:
        raise ValueError('Checkpoint world size differs from registered execution identity')
    result = {'run': run_dir.name, 'epoch': epoch, 'checkpoint': str(checkpoint),
              'sha256': sha256(checkpoint),
              'checkpoint_complete_sha256': sha256(str(checkpoint) + '.COMPLETE.json'),
              'asset_identities_sha256': sha256(run_dir / 'ASSET_IDENTITIES.json'),
              'progress': dict(state['progress']),
              'qualification': identities['qualification'],
              'protocol_sha256': identities['protocol_sha256'],
              'teacher_bundle_sha256': identities['files']['teacher_bundle']['sha256'],
              'feature_manifest_sha256': identities['files']['feature_manifest']['sha256'],
              'calibration_sha256': identities['files']['calibration']['sha256']}
    noise_manifest = Path(state['config']['paths']['evaluation_noise_manifest']).resolve()
    result['noise_manifest'] = str(noise_manifest)
    result['noise_manifest_sha256'] = sha256(noise_manifest)
    return result


def _noise_identity(endpoint):
    manifest = Path(endpoint['noise_manifest'])
    value = json.loads(manifest.read_text())
    if value.get('schema') != 'native_explicit_evaluation_noise_v1':
        raise ValueError('Wrong explicit evaluation noise manifest')
    entry = value['populations']['test']
    resolve = lambda text: str((Path(text) if Path(text).is_absolute() else manifest.parent / text).resolve())
    return {'manifest': str(manifest), 'manifest_sha256': endpoint['noise_manifest_sha256'],
            'population': 'test', 'path': resolve(entry['path']), 'sha256': entry['sha256'],
            'tokens_path': resolve(entry['tokens_path']), 'tokens_sha256': entry['tokens_sha256'],
            'noise_indices': list(entry['noise_indices']), 'source': entry.get('source'),
            'seed_fallback': False}


def endpoint_candidate(run_dir, epoch):
    """Verify paired full-Test scoring for one endpoint, including every row."""
    endpoint = endpoint_checkpoint(run_dir, epoch)
    folder = Path(run_dir) / 'test' / f'epoch_{epoch:03d}'
    expected = {'population': 'test', 'scene_count': TEST_SCENES,
                'noise_indices': list(NOISE_INDICES), 'full_population': True,
                'checkpoint_sha256': endpoint['sha256'], 'artifact_kind': 'behavior',
                'mode': 'student', 'target_version': None, 'representation_only': False,
                'paired_scene_noise_base': True, 'targets_manifest_sha256': None,
                'teacher_bundle_sha256': endpoint['teacher_bundle_sha256'],
                'feature_manifest_sha256': endpoint['feature_manifest_sha256'],
                'calibration_sha256': endpoint['calibration_sha256'],
                'noise_identity': _noise_identity(endpoint)}
    summary = verify_evaluation(folder, expected)
    score = summary['behavior']['pdms_points']
    if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError('Full-Test selection requires finite PDMS in points [0,100]')
    return {**endpoint, 'pdms_points': score,
            'test_summary': str((folder / 'SUMMARY.json').resolve()),
            'test_summary_sha256': sha256(folder / 'SUMMARY.json'),
            'test_complete_sha256': sha256(folder / 'COMPLETE.json'),
            'test_protocol_sha256': sha256(folder / 'PROTOCOL.json')}


def _training_complete(run_dir, identities):
    path = run_dir / 'TRAIN_COMPLETE.json'
    value = json.loads(path.read_text())
    expected = {'status': 'TRAIN_COMPLETE', 'global_step': 53640,
                'real_scenes_seen': 3432520, 'epoch_index': 40}
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('Final Test selection requires the complete 40-epoch training budget')
    if value.get('qualification') != identities.get('qualification'):
        raise ValueError('Training completion has a different registered asset/execution identity')
    return sha256(path)


def select(run_dir):
    """Finalize best/last only after all training and all five full-Test runs."""
    run_dir = Path(run_dir).resolve()
    identities = _identities(run_dir)
    complete_sha = _training_complete(run_dir, identities)
    candidates = [endpoint_candidate(run_dir, epoch) for epoch in ENDPOINTS]
    best = min(candidates, key=lambda x: (-x['pdms_points'], x['epoch']))
    result = {'schema': 'nrkd_test_pdms_selection_v1',
              'rule': 'full_Test12146_scene_mean_S0_S4_PDMS_max_exact_ties_earlier_epoch',
              'selection_population': 'test', 'test_used_for_model_selection': True,
              'independent_held_out_test': False,
              'validation_used_for_model_selection': False,
              'user_instruction': request_identity(), 'candidates': candidates,
              'best': best, 'last': candidates[-1], 'base_is_not_a_candidate': True,
              'protocol_sha256': identities['protocol_sha256'],
              'teacher_bundle_sha256': identities['files']['teacher_bundle']['sha256'],
              'qualification': identities['qualification'],
              'train_complete_sha256': complete_sha}
    path = run_dir / 'checkpoints' / BEST_NAME
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError('Cannot replace an already finalized Test-PDMS selection')
    atomic_json(path, result)
    return result


def lock_suite(out):
    out = Path(out).resolve()
    registry = json.loads((out / 'EXPERIMENT_REGISTRY.json').read_text())
    if set(registry['runs']) != set(RUNS):
        raise ValueError('Test-PDMS final lock requires exactly four retained runs')
    selections = {name: select(out / name) for name in sorted(RUNS)}
    value = {'schema': 'nrkd_test_pdms_selection_lock_v1', 'test_may_start': True,
             'all_registered_train_and_full_test_complete': True,
             'registered_training_seeds': [20261027], 'registered_run_count': 4,
             'selection_population': 'test', 'test_used_for_model_selection': True,
             'independent_held_out_test': False, 'validation_used_for_model_selection': False,
             'user_instruction': request_identity(), 'runs': selections,
             'selection_protocol_sha256': _selection_protocol_sha256()}
    value['lock_sha256'] = digest(value)
    path = out / LOCK_NAME
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Cannot replace an existing final Test-PDMS selection lock')
    atomic_json(path, value)
    return value


def verify_selection_lock(out, *, full=False):
    out = Path(out).resolve()
    value = json.loads((out / LOCK_NAME).read_text())
    if digest({k: v for k, v in value.items() if k != 'lock_sha256'}) != value.get('lock_sha256'):
        raise ValueError('Final Test-PDMS selection lock digest changed')
    if (value.get('schema') != 'nrkd_test_pdms_selection_lock_v1' or
        set(value.get('runs', {})) != set(RUNS) or
        value.get('test_may_start') is not True or
        value.get('all_registered_train_and_full_test_complete') is not True or
        value.get('registered_training_seeds') != [20261027] or
        value.get('registered_run_count') != 4 or
        value.get('selection_population') != 'test' or
        value.get('test_used_for_model_selection') is not True or
        value.get('independent_held_out_test') is not False or
        value.get('validation_used_for_model_selection') is not False or
        value.get('selection_protocol_sha256') != _selection_protocol_sha256() or
        value.get('user_instruction') != request_identity()):
        raise ValueError('Incomplete or differently authorized final Test-PDMS selection lock')
    for run, selected in value['runs'].items():
        folder = out / run
        if sha256(folder / 'TRAIN_COMPLETE.json') != selected['train_complete_sha256']:
            raise ValueError('Training completion changed after Test-PDMS selection')
        if full:
            if select(folder) != selected:
                raise ValueError('Full-Test PDMS selection evidence changed')
            continue
        if selected.get('user_instruction') != request_identity():
            raise ValueError('Selected checkpoint belongs to a different user instruction')
        if [x['epoch'] for x in selected['candidates']] != list(ENDPOINTS):
            raise ValueError('Final selection does not contain all five Test endpoints')
        expected_best = min(selected['candidates'], key=lambda x: (-x['pdms_points'], x['epoch']))
        if selected['best'] != expected_best or selected['last'] != selected['candidates'][-1]:
            raise ValueError('Best/last does not match the frozen full-Test selection rule')
        for candidate in selected['candidates']:
            epoch = candidate['epoch']
            endpoint = endpoint_checkpoint(folder, epoch)
            if any(candidate[k] != v for k, v in endpoint.items()):
                raise ValueError('Checkpoint identity changed after selection')
            ev = folder / 'test' / f'epoch_{epoch:03d}'
            for artifact, field in [('SUMMARY.json', 'test_summary_sha256'),
                                    ('COMPLETE.json', 'test_complete_sha256'),
                                    ('PROTOCOL.json', 'test_protocol_sha256')]:
                if sha256(ev / artifact) != candidate[field]:
                    raise ValueError('Full-Test selection evidence changed after final lock')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run', type=Path)
    group.add_argument('--suite', type=Path)
    group.add_argument('--verify', type=Path)
    parser.add_argument('--epoch', type=int, choices=ENDPOINTS)
    args = parser.parse_args()
    if args.epoch is not None and args.run is None:
        parser.error('--epoch requires --run')
    value = endpoint_candidate(args.run, args.epoch) if args.run and args.epoch else (
        select(args.run) if args.run else lock_suite(args.suite) if args.suite else
        verify_selection_lock(args.verify, full=True))
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
