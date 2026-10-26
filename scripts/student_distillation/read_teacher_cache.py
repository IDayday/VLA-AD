#!/usr/bin/env python3
"""Lossless CPU reader/indexer for the newly refit local teacher cache.

No model, action head, GPU or historical expert is loaded. The stored native
FP32 Base mmap and the scored actual shift reconstruct the complete interface;
paired trajectories and seven PDM fields come from that same saved execution.
"""
from __future__ import annotations
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import uuid
import numpy as np

SCHEMA = 'nrkd_repaired_cache_index_v1'
COUNTS = {'train': 85813, 'validation': 17475, 'test': 12146}
LAYOUT = ('image_mask', 'text_mask', 'token_ids', 'attention_mask', 'position_ids')
SCORE_FIELDS = ['score', 'no_at_fault_collisions', 'drivable_area_compliance',
                'time_to_collision_within_bound', 'ego_progress', 'comfort', 'driving_direction_compliance']


def sha_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def sha_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def reference(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha_file(path)}


def resolve(reference_value, root, *, verify=True):
    path = Path(reference_value['path'] if isinstance(reference_value, dict) else reference_value)
    path = path if path.is_absolute() else Path(root) / path
    if verify and isinstance(reference_value, dict):
        if sha_file(path) != reference_value.get('sha256'):
            raise ValueError(f'Artifact SHA mismatch: {path}')
    return path.resolve()


def context(paths, population, *, teacher_root=None, allow_synthetic=False, verify_selection=True):
    values = dict(paths) if isinstance(paths, dict) else read_json(paths)
    values = values.get('paths', values)
    root = Path(teacher_root).resolve() if teacher_root else None
    bundle_path = Path(values.get('teacher_bundle') or (root / 'frozen_teacher/BUNDLE.json' if root else ''))
    if not bundle_path.is_file():
        raise FileNotFoundError('The NEW frozen local teacher bundle is not ready')
    bundle_bytes = bundle_path.read_bytes()
    bundle = json.loads(bundle_bytes)
    if bundle.get('status') != 'FROZEN_LOCAL_TEACHER' or bundle.get('teacher_origin') != 'local_refit' or bundle.get('fit_population') != 'train':
        raise ValueError('This reader accepts only the newly refit, frozen local Train teacher')
    root = root or bundle_path.resolve().parent.parent
    features_path = Path(values['feature_manifest']).resolve()
    features_bytes = features_path.read_bytes()
    features = json.loads(features_bytes)
    if features.get('schema') != 'nrkd_native_post_mmap_v1' or not features.get('preserve_native_layout'):
        raise ValueError('A lossless native FP32 mmap feature manifest is required')
    spec = features['populations'][population]
    if int(spec['count']) != COUNTS[population] and not (allow_synthetic and features.get('synthetic_only')):
        raise ValueError('The complete original population is required for this cache index')
    registry_path = resolve(bundle['selection_registry'], bundle_path.parent, verify=False)
    registry_bytes = registry_path.read_bytes()
    registry = json.loads(registry_bytes)
    registry_body = dict(registry); registered_digest = registry_body.pop('identity_sha256', None)
    if registry.get('schema') != 'nrkd_local_selection_registry_v1' or digest(registry_body) != registered_digest:
        raise ValueError('Local selection registry self-digest mismatch')
    if registry.get('teacher_identity') != bundle['teacher_identity'] or registry.get('source_checkpoint_sha256') != bundle['source_checkpoint_sha256']:
        raise ValueError('Selection evidence belongs to another local teacher')
    selection_ref = registry['populations'][population]
    if selection_ref.get('units') != 'points' or (population != 'train' and not selection_ref.get('diagnostic_only')):
        raise ValueError('Selection units/diagnostic flags are not explicit')
    selection_path = resolve(selection_ref, registry_path.parent, verify=verify_selection)
    return {'paths': values, 'root': root, 'population': population, 'bundle': bundle,
            'bundle_path': bundle_path.resolve(), 'bundle_sha': hashlib.sha256(bundle_bytes).hexdigest(),
            'features': features, 'features_path': features_path, 'features_sha': hashlib.sha256(features_bytes).hexdigest(),
            'feature_spec': spec, 'selection_ref': dict(selection_ref, path=str(selection_path)),
            'registry_snapshot_b64': base64.b64encode(registry_bytes).decode('ascii'),
            'synthetic_only': bool(features.get('synthetic_only'))}


def index_files(ctx):
    folder = ctx['root'] / 'requests' / ctx['population']
    return folder, folder / 'REPAIRED_CACHE_INDEX.jsonl', folder / 'REPAIRED_CACHE_INDEX.sqlite3', folder / 'REPAIRED_CACHE_INDEX.COMPLETE.json'


def encode_shift(shift):
    value = np.ascontiguousarray(shift, dtype='<f4')
    if value.shape != (384,) or not np.isfinite(value).all():
        raise ValueError('Actual compact shared shift must be finite FP32[384]')
    return {'dtype': '<f4', 'shape': [384], 'base64': base64.b64encode(value.tobytes()).decode('ascii'),
            'bytes_sha256': sha_array(value)}


def decode_shift(value):
    raw = base64.b64decode(value['base64'], validate=True)
    if value.get('dtype') != '<f4' or value.get('shape') != [384] or hashlib.sha256(raw).hexdigest() != value['bytes_sha256']:
        raise ValueError('Compact actual-shift bytes are corrupt')
    return np.frombuffer(raw, dtype='<f4').copy()


def build_index(paths, population, *, workers=4, teacher_root=None, allow_synthetic=False):
    """Append new index artifacts only; no existing score/selection is rewritten."""
    ctx = context(paths, population, teacher_root=teacher_root, allow_synthetic=allow_synthetic)
    folder, index_path, lookup_path, complete_path = index_files(ctx)
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'REPAIRED_CACHE_INDEX.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX)
    evaluation_path = ctx['root'] / 'frozen_teacher' / (population.upper() + '_EVALUATION.json')
    evaluation = read_json(evaluation_path)
    native_complete = read_json(folder / 'COMPLETE.json')
    if evaluation.get('status') != 'COMPLETE' or native_complete.get('status') != 'COMPLETE':
        raise ValueError('Full actual native candidate scoring/selection is not complete')
    if evaluation['scenes'] != ctx['feature_spec']['count'] or native_complete['scenes'] != evaluation['scenes'] or native_complete['actions'] != 9:
        raise ValueError('Teacher scoring coverage differs from the complete feature population')
    if evaluation['noise_indices'] != [0, 1, 2, 3, 4] or native_complete['noise_indices'] != [0, 1, 2, 3, 4]:
        raise ValueError('The actual paired S0-S4 population is incomplete')
    inputs = {'teacher_bundle': {'path': str(ctx['bundle_path']), 'sha256': ctx['bundle_sha']},
              'feature_manifest': {'path': str(ctx['features_path']), 'sha256': ctx['features_sha']},
              'feature_population_sha256': digest(ctx['feature_spec']), 'selection': ctx['selection_ref'],
              'population_evaluation': reference(evaluation_path), 'native_complete': reference(folder / 'COMPLETE.json')}
    if complete_path.exists():
        previous = read_json(complete_path)
        if previous['inputs'] != inputs:
            raise ValueError('Refusing to replace a completed repaired-cache index of another identity')
        for artifact in previous['artifacts'].values():
            resolve(artifact, folder)
        return previous
    index_ref = ctx['feature_spec']['index']
    feature_rows = [json.loads(line) for line in resolve(index_ref, ctx['features_path'].parent).read_text().splitlines() if line]
    by_token = {row['token']: row for row in feature_rows}
    if len(by_token) != len(feature_rows):
        raise ValueError('Duplicate feature token')
    with np.load(ctx['selection_ref']['path'], allow_pickle=False) as selection:
        tokens = selection['tokens'].astype(str).tolist()
        q_points = np.array(selection['q_points'])
        choices = np.array(selection['choice'])
        indices = np.array(selection['indices'])
        scored_shifts = np.array(selection['selected_actual_role_shifts'])
    if (q_points.dtype != np.float64 or q_points.shape != (len(tokens), 9) or
            not np.isfinite(q_points).all() or not np.array_equal(q_points.argmax(1), choices) or
            set(tokens) != set(by_token) or len(tokens) != len(by_token)):
        raise ValueError('Frozen point choices/feature token coverage are inconsistent')
    if scored_shifts.dtype != np.float32 or scored_shifts.shape != (len(tokens), 2, 384):
        raise ValueError('Frozen selected actual shifts are missing')
    selected_position = {token: position for position, token in enumerate(tokens)}
    queue = read_json(folder / 'QUEUE.json')
    request_path = resolve(queue['request'], folder)
    request = read_json(request_path)
    if request.get('scope') != 'full_population' or request['population'] != population or len(queue['tasks']) != len(tokens):
        raise ValueError('One full nine-branch execution per population scene is required')
    if request['feature_manifest']['sha256'] != ctx['features_sha'] or request['bundle']['sha256'] != ctx['bundle_sha']:
        raise ValueError('Scored teacher/Base identity differs from the supplied paths')
    if sha_file(native_complete['matrix']['path']) != native_complete['matrix']['sha256']:
        raise ValueError('Actual native score matrix is corrupt')

    def make_row(task_id):
        receipt_path = folder / 'receipts' / (task_id + '.json')
        receipt = read_json(receipt_path)
        if receipt.get('status') != 'COMPLETE':
            raise ValueError('Actual scene scoring receipt is incomplete')
        result_path = resolve(receipt['result'], folder)
        result = read_json(result_path)
        token = result['token']; feature = by_token[token]; pos = selected_position[token]
        task = result['task']; row_index = int(task['row_index'])
        if result['population'] != population or request['tokens'][row_index] != token or int(task['index']) != int(feature['global_index']) or int(indices[pos]) != int(feature['global_index']):
            raise ValueError('Scored token/global index/feature join mismatch')
        positions = task['positions']
        branch_choices = [int(request['actions'][position]['choice']) for position in positions]
        if branch_choices != list(range(9)):
            raise ValueError('Actual native score payload does not contain Base-first choices0..8')
        arrays_path = resolve(result['arrays'], folder)
        with np.load(arrays_path, allow_pickle=False) as arrays:
            scores = np.array(arrays['scores']); base_scores = np.array(arrays['base_scores'])
            roles = np.array(arrays['actual_role_shifts']); rms = np.array(arrays['actual_RMS']); caps = np.array(arrays['cap_RMS'])
            noise_indices = np.array(arrays['noise_indices'])
            if scores.dtype != np.float64 or scores.shape != (9, 5, 7) or base_scores.shape != (5, 7) or not np.isfinite(scores).all():
                raise ValueError('Seven actual native score fields are incomplete')
            if not np.array_equal(scores[0], base_scores) or noise_indices.tolist() != [0, 1, 2, 3, 4]:
                raise ValueError('Actual paired Base/five-noise scores disagree')
            actual_points = scores[:, :, 0].mean(1, dtype=np.float64) * np.float64(100.)
            if not np.array_equal(actual_points, q_points[pos]) or not np.array_equal(roles[int(choices[pos])], scored_shifts[pos]):
                raise ValueError('The selected label differs from its actual scored execution')
            full = int(choices[pos]); gain_points = float(actual_points[full] - actual_points[0]); keep = gain_points > 1.
            if roles.dtype != np.float32 or roles.shape != (9, 2, 384) or not np.array_equal(roles[:, 0], roles[:, 1]) or np.any(roles[0]):
                raise ValueError('Native actual shifts are not shared valid384 with literal Base first')
            compact = encode_shift(roles[full, 0])
        required_audit = ('selected_teacher_interface_sha256', 'selected_branch_point_argmax', 'Base_padding_bytes_sha256',
                          'selected_teacher_padding_bytes_sha256', 'Base_interface_energy_all', 'Base_interface_energy_valid',
                          'selected_teacher_residual_energy_valid', 'selected_teacher_residual_energy_all')
        if any(key not in result for key in required_audit):
            raise ValueError('The scored cache lacks the required same-execution FP32 hash/energy audit; do not rerun a head silently')
        if result['selected_branch_point_argmax'] != full or result['Base_interface_sha256'] != feature['actual_Base_interface_sha256']:
            raise ValueError('Same-execution branch/Base byte identity mismatch')
        if result['Base_padding_bytes_sha256'] != result['selected_teacher_padding_bytes_sha256']:
            raise ValueError('Native teacher changed padding bytes')
        location = result['immutable_full_padded_H0']
        if any(int(location[key]) != int(feature[key]) for key in ('shard', 'offset', 'length', 'state_row')):
            raise ValueError('Native mmap row reference changed')
        shard = ctx['feature_spec']['shards'][int(feature['shard'])]
        return {'schema': SCHEMA, 'token': token, 'population': population, 'global_index': int(feature['global_index']),
                'log_id': str(feature['log_id']), 'native_sequence_length': int(feature['length']),
                'Base_immutable_mmap': {'manifest': result['feature_manifest'], 'shard': int(feature['shard']),
                    'offset': int(feature['offset']), 'length': int(feature['length']), 'state_row': int(feature['state_row']),
                    'arrays': shard['arrays'], 'Base_interface_sha256': result['Base_interface_sha256'],
                    'token_layout_sha256': feature['token_layout_sha256']},
                'source_scores': dict(result['arrays'], path=str(arrays_path)), 'source_result': reference(result_path),
                'source_receipt': reference(receipt_path), 'source_row_index': row_index, 'branch_choices': branch_choices,
                'full_choice': full, 'full_branch_position': branch_choices.index(full),
                'gain_points_fp64': gain_points, 'gain1_keep': bool(keep), 'gain1_choice': full if keep else 0,
                'gain1_branch_position': branch_choices.index(full if keep else 0),
                'full_actual_shift_fp32': compact, 'full_actual_RMS': float(rms[full]), 'full_cap_RMS': float(caps[full]),
                'Base_actual_RMS': float(rms[0]), 'Base_cap_RMS': float(caps[0]),
                'selected_teacher_interface_sha256': result['selected_teacher_interface_sha256'],
                'Base_padding_bytes_sha256': result['Base_padding_bytes_sha256'],
                'selected_teacher_padding_bytes_sha256': result['selected_teacher_padding_bytes_sha256'],
                'valid_token_count': int(result['valid_token_count']), 'noise_indices': [0, 1, 2, 3, 4],
                'noise_identity': result['noise_identity'], 'native_identity_sha256': result['native_identity_sha256'],
                'metric_cache': result['metric_cache'], 'local_metric_sha256': result['local_metric_sha256'],
                'reference_metric_sha256': result.get('reference_metric_sha256'),
                'local_metric_registry_sha256': result.get('local_metric_registry_sha256'),
                'energy_statistics_audit_only': {key: result[key] for key in required_audit if 'energy' in key},
                'residual_energy_all_formula': result['residual_energy_all_formula']}

    temporary_index = index_path.with_name('.' + index_path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary_lookup = lookup_path.with_name('.' + lookup_path.name + '.' + uuid.uuid4().hex + '.tmp')
    database = sqlite3.connect(temporary_lookup)
    database.execute('PRAGMA journal_mode=OFF')
    database.execute('CREATE TABLE scenes(token TEXT PRIMARY KEY, global_index INTEGER UNIQUE NOT NULL, byte_offset INTEGER NOT NULL, byte_length INTEGER NOT NULL, row_sha256 TEXT NOT NULL)')
    seen = set()
    try:
        with temporary_index.open('wb') as output, ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            for count, row in enumerate(pool.map(make_row, queue['tasks']), 1):
                token = row['token']
                if token in seen:
                    raise ValueError('Duplicate actual scene result')
                seen.add(token)
                raw = (json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
                offset = output.tell(); output.write(raw)
                database.execute('INSERT INTO scenes VALUES (?,?,?,?,?)',
                                 (token, row['global_index'], offset, len(raw), hashlib.sha256(raw).hexdigest()))
                if count % 512 == 0:
                    print(json.dumps({'population': population, 'index_scenes': count, 'total': len(tokens)}), flush=True)
            output.flush(); os.fsync(output.fileno())
        if seen != set(tokens):
            raise ValueError('The repaired-cache index did not cover the full population')
        database.commit(); database.close()
        temporary_index.replace(index_path); temporary_lookup.replace(lookup_path)
        marker = {'schema': SCHEMA, 'status': 'COMPLETE', 'population': population, 'scenes': len(seen),
                  'inputs': inputs, 'synthetic_only': ctx['synthetic_only'],
                  'teacher_identity': ctx['bundle']['teacher_identity'], 'source_checkpoint_sha256': ctx['bundle']['source_checkpoint_sha256'],
                  'selection_registry_snapshot_base64': ctx['registry_snapshot_b64'],
                  'noise_indices': [0, 1, 2, 3, 4], 'score_fields': SCORE_FIELDS,
                  'artifacts': {'jsonl': reference(index_path), 'sqlite_lookup': reference(lookup_path)},
                  'original_scores_selections_teacher_parameters_or_Base_mmap_modified': False,
                  'reader_source': reference(__file__), 'no_model_or_head_rerun': True}
        atomic_json(complete_path, marker)
        return marker
    finally:
        database.close()


def _mmap_scene(row):
    spec = row['Base_immutable_mmap']; arrays = {}
    for name in ('base_post', *LAYOUT, 'state24'):
        artifact = spec['arrays'][name]
        mapped = np.memmap(artifact['path'], mode='r', dtype=np.dtype(artifact['dtype']), shape=tuple(artifact['shape']))
        view = mapped[int(spec['state_row'])] if name == 'state24' else mapped[int(spec['offset']):int(spec['offset']) + int(spec['length'])]
        arrays[name] = np.array(view, copy=True)
    return arrays


class TeacherCacheReader:
    """Reuse verified lookup metadata across many direct CPU scene reads."""
    def __init__(self, paths, population, *, teacher_root=None, verify_index=False, allow_synthetic=False):
        self.paths, self.population = paths, population
        self.teacher_root, self.allow_synthetic = teacher_root, allow_synthetic
        self.ctx = context(paths, population, teacher_root=teacher_root, allow_synthetic=allow_synthetic,
                           verify_selection=verify_index)
        folder, index, lookup, complete = index_files(self.ctx)
        self.marker = read_json(complete)
        if self.marker.get('status') != 'COMPLETE' or self.marker['inputs']['teacher_bundle']['sha256'] != self.ctx['bundle_sha'] or self.marker['inputs']['selection']['sha256'] != self.ctx['selection_ref']['sha256']:
            raise ValueError('Cache reader identity differs from the completed local teacher index')
        if sha_file(lookup) != self.marker['artifacts']['sqlite_lookup']['sha256']:
            raise ValueError('Direct token lookup is corrupt')
        if verify_index and sha_file(index) != self.marker['artifacts']['jsonl']['sha256']:
            raise ValueError('Complete repaired-cache index bytes are corrupt')
        self.database = sqlite3.connect('file:' + str(lookup.resolve()) + '?mode=ro', uri=True)

    def read(self, token, target_version='full'):
        return read_teacher_cache(self.paths, self.population, token, target_version,
                                  teacher_root=self.teacher_root, allow_synthetic=self.allow_synthetic, _state=self)

    def __getitem__(self, token):
        return self.read(token)

    def close(self):
        self.database.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def read_teacher_cache(paths, population, token, target_version='full', *, teacher_root=None,
                       verify_index=False, allow_synthetic=False, _state=None):
    """Return numpy arrays and metadata for exactly one scene; no torch import."""
    if target_version not in ('full', 'gain1'):
        raise ValueError('Target version must be full or gain1')
    ctx = (_state.ctx if _state else context(paths, population, teacher_root=teacher_root,
                   allow_synthetic=allow_synthetic, verify_selection=verify_index))
    folder, index_path, lookup_path, complete_path = index_files(ctx)
    marker = _state.marker if _state else read_json(complete_path)
    if marker.get('status') != 'COMPLETE' or marker['inputs']['teacher_bundle']['sha256'] != ctx['bundle_sha'] or marker['inputs']['selection']['sha256'] != ctx['selection_ref']['sha256']:
        raise ValueError('Completed repaired-cache index belongs to another local teacher/selection')
    if marker['inputs']['feature_manifest']['sha256'] != ctx['features_sha']:
        raise ValueError('Completed repaired-cache index belongs to another native Base mmap')
    if _state is None and sha_file(lookup_path) != marker['artifacts']['sqlite_lookup']['sha256']:
        raise ValueError('Direct token lookup is corrupt')
    if verify_index and sha_file(index_path) != marker['artifacts']['jsonl']['sha256']:
        raise ValueError('Complete repaired-cache JSONL is corrupt')
    database = _state.database if _state else sqlite3.connect('file:' + str(lookup_path.resolve()) + '?mode=ro', uri=True)
    record = database.execute('SELECT byte_offset,byte_length,row_sha256 FROM scenes WHERE token=?', (token,)).fetchone()
    if _state is None:
        database.close()
    if record is None:
        raise KeyError(f'{token} is absent from the complete {population} cache')
    with index_path.open('rb') as stream:
        stream.seek(record[0]); raw = stream.read(record[1])
    if hashlib.sha256(raw).hexdigest() != record[2]:
        raise ValueError('Indexed scene record bytes are corrupt')
    row = json.loads(raw)
    if row['token'] != token or row['population'] != population:
        raise ValueError('Direct lookup token/population mismatch')
    arrays = _mmap_scene(row)
    base = arrays['base_post']
    if base.dtype != np.float32 or base.shape != (2800, 384) or sha_array(base) != row['Base_immutable_mmap']['Base_interface_sha256']:
        raise ValueError('Native full padded FP32 Base bytes changed')
    image, text = arrays['image_mask'], arrays['text_mask']
    valid = image | text
    if image.dtype != np.bool_ or text.dtype != np.bool_ or np.any(image & text) or not image.any() or not text.any() or not np.array_equal(valid, arrays['attention_mask'].astype(bool)):
        raise ValueError('Native role/layout masks are invalid')
    if digest({name: sha_array(arrays[name]) for name in LAYOUT}) != row['Base_immutable_mmap']['token_layout_sha256']:
        raise ValueError('Native token IDs/masks/positions changed')
    source = resolve(row['source_scores'], folder)
    full_shift = decode_shift(row['full_actual_shift_fp32'])
    choice = row['full_choice'] if target_version == 'full' else row['gain1_choice']
    position = row['full_branch_position'] if target_version == 'full' else row['gain1_branch_position']
    literal_base = target_version == 'gain1' and not row['gain1_keep']
    with np.load(source, allow_pickle=False) as scored:
        role_shift = np.array(scored['actual_role_shifts'][position])
        if not np.array_equal(role_shift[0], role_shift[1]) or (choice == row['full_choice'] and not np.array_equal(role_shift[0], full_shift)):
            raise ValueError('Actual scored shift differs from indexed compact shift')
        shift = role_shift[0].copy()
        rms, cap = float(scored['actual_RMS'][position]), float(scored['cap_RMS'][position])
        if literal_base:
            if choice != 0 or np.any(shift):
                raise ValueError('gain1 keep targets must use actual literal Base branch0')
            trajectory, quality = np.array(scored['base_trajectories']), np.array(scored['base_scores'])
        else:
            trajectory, quality = np.array(scored['trajectories'][position]), np.array(scored['scores'][position])
        base_trajectory, base_quality = np.array(scored['base_trajectories']), np.array(scored['base_scores'])
        if not np.array_equal(scored['image_mask'], image) or not np.array_equal(scored['text_mask'], text) or not np.array_equal(scored['state24'], arrays['state24']):
            raise ValueError('Saved execution masks/state differ from the immutable native inputs')
    teacher = base.copy()
    if not literal_base:
        teacher[image] = np.add(base[image], role_shift[0][None], dtype=np.float32)
        teacher[text] = np.add(base[text], role_shift[1][None], dtype=np.float32)
    padding_sha = sha_array(teacher[~valid])
    if padding_sha != sha_array(base[~valid]) or padding_sha != row['Base_padding_bytes_sha256']:
        raise ValueError('Reconstruction changed original FP32 padding bytes')
    expected_teacher_sha = row['Base_immutable_mmap']['Base_interface_sha256'] if literal_base else row['selected_teacher_interface_sha256']
    teacher_sha = sha_array(teacher)
    if teacher_sha != expected_teacher_sha:
        raise ValueError('Reconstructed complete FP32 teacher differs from the same scored execution SHA')
    residual_valid = float(np.square(teacher[valid].astype(np.float64) - base[valid].astype(np.float64)).mean())
    actual_rms = float(np.sqrt(residual_valid))
    if not np.isclose(actual_rms, rms, rtol=1e-10, atol=1e-12) or actual_rms > cap + max(1e-7, cap * 1e-4):
        raise ValueError('Reconstructed native RMS/cap differs from the actual scored teacher')
    if trajectory.dtype != np.float32 or trajectory.shape != (5, 8, 3) or quality.dtype != np.float64 or quality.shape != (5, 7) or not np.isfinite(trajectory).all() or not np.isfinite(quality).all():
        raise ValueError('Actual paired five trajectories/seven score fields are invalid')
    state = arrays['state24']
    if state.dtype != np.float32 or state.shape != (24,) or not np.isfinite(state).all():
        raise ValueError('Ordinary state must remain the original finite FP32[24]')
    energy = dict(row['energy_statistics_audit_only'])
    energy['returned_teacher_residual_energy_valid'] = residual_valid
    energy['returned_teacher_residual_energy_all'] = residual_valid * (int(valid.sum()) / len(valid))
    metadata = {'schema': 'nrkd_lossless_full_teacher_scene_v1', 'token': token, 'population': population,
                'global_index': row['global_index'], 'log_id': row['log_id'], 'target_version': target_version,
                'original_full_choice': row['full_choice'], 'returned_choice': choice, 'gain1_keep': row['gain1_keep'],
                'gain1_repair_retained': row['gain1_keep'], 'gain1_uses_Base': not row['gain1_keep'],
                'gain_points_fp64': row['gain_points_fp64'], 'diagnostic_only': population != 'train',
                'teacher_identity': marker['teacher_identity'], 'teacher_bundle': marker['inputs']['teacher_bundle'],
                'source_checkpoint_sha256': marker['source_checkpoint_sha256'], 'noise_indices': [0, 1, 2, 3, 4],
                'noise_identity': row['noise_identity'], 'native_identity_sha256': row['native_identity_sha256'],
                'score_fields': SCORE_FIELDS, 'score_units': 'fraction_0_to_1', 'PDMS_points': float(quality[:, 0].mean() * 100.),
                'trajectory_units': 'ego_frame_xy_metres_heading_radians_8poses_dt0.5seconds',
                'teacher_interface_sha256': teacher_sha, 'same_execution_full_FP32_SHA_verified': True,
                'padding_bytes_sha256': padding_sha, 'padding_preserved_bytewise': True,
                'actual_RMS': actual_rms, 'scored_actual_RMS': rms, 'cap_RMS': cap, 'energy_statistics_audit_only': energy,
                'local_metric_sha256': row['local_metric_sha256'], 'reference_metric_sha256': row.get('reference_metric_sha256'),
                'local_metric_registry_sha256': row.get('local_metric_registry_sha256'),
                'Base_immutable_mmap': row['Base_immutable_mmap'], 'source_scores': row['source_scores'],
                'source_result': row['source_result'], 'source_receipt': row['source_receipt'],
                'repaired_cache_index': marker['artifacts']['jsonl'], 'no_model_or_head_rerun': True,
                'synthetic_only': marker.get('synthetic_only', False)}
    return {'teacher_interface': teacher, 'Base_interface': base, 'actual_shift': shift,
            'actual_role_shifts': role_shift, 'image_mask': image, 'text_mask': text, 'valid_mask': valid,
            'padding_mask': ~valid, 'token_ids': arrays['token_ids'], 'attention_mask': arrays['attention_mask'],
            'position_ids': arrays['position_ids'], 'state24': state, 'history12': state[:12].copy(),
            'history_trajectory': state[:12].reshape(4, 3).copy(), 'status8': state[12:20].copy(),
            'status_feature': state[12:20].copy(), 'high_command_one_hot': state[20:24].copy(),
            'teacher_trajectories': trajectory, 'teacher_quality': quality, 'Base_trajectories': base_trajectory,
            'Base_quality': base_quality, 'noise_indices': np.arange(5, dtype=np.int64),
            'integer_seeds': np.asarray([int(hashlib.sha256(f'{token}|20261008|{noise}'.encode()).hexdigest()[:8], 16) for noise in range(5)], np.int64),
            'metadata': metadata}


def save_scene(path, scene):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: value for name, value in scene.items() if name != 'metadata'}
    arrays['metadata_json'] = np.asarray(json.dumps(scene['metadata'], sort_keys=True, allow_nan=False))
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    with np.load(temporary, allow_pickle=False) as saved:
        for name, value in arrays.items():
            if saved[name].tobytes() != np.asarray(value).tobytes():
                raise ValueError('Exported cache scene failed bytewise readback: ' + name)
    temporary.replace(path)
    return reference(path)


def watch_build(paths, populations, *, teacher_root=None, workers=4, poll_seconds=30.):
    """CPU-only watcher; each population is indexed once after real collection."""
    pending = set(populations)
    status_root = Path(teacher_root).resolve() if teacher_root else None
    while pending:
        if not Path(paths).is_file():
            print(json.dumps({'status': 'WAITING_PATHS', 'paths': str(paths)}), flush=True)
            time.sleep(poll_seconds); continue
        values = read_json(paths); values = values.get('paths', values)
        bundle_path = Path(values.get('teacher_bundle') or (status_root / 'frozen_teacher/BUNDLE.json' if status_root else ''))
        if not bundle_path.is_file():
            print(json.dumps({'status': 'WAITING_NEW_LOCAL_TEACHER', 'populations': sorted(pending)}), flush=True)
            time.sleep(poll_seconds); continue
        actual_root = status_root or bundle_path.resolve().parent.parent
        for population in sorted(pending):
            if not (actual_root / 'requests' / population / 'COMPLETE.json').is_file() or not (actual_root / 'frozen_teacher' / (population.upper() + '_EVALUATION.json')).is_file():
                continue
            build_index(paths, population, workers=workers, teacher_root=actual_root)
            pending.remove(population)
            atomic_json(actual_root / 'repaired_cache_index_watcher/STATE.json', {'status': 'INDEXING' if pending else 'COMPLETE',
                        'pending_populations': sorted(pending), 'completed_populations': sorted(set(populations) - pending)})
        if pending:
            print(json.dumps({'status': 'WAITING_COMPLETE_POPULATION_SCORING', 'pending': sorted(pending)}), flush=True)
            time.sleep(poll_seconds)
    atomic_json(actual_root / 'repaired_cache_index_watcher/COMPLETE.json', {'status': 'COMPLETE', 'populations': sorted(populations),
                'artifacts': {population: reference(actual_root / 'requests' / population / 'REPAIRED_CACHE_INDEX.COMPLETE.json')
                              for population in sorted(populations)},
                'reader_source': reference(__file__), 'no_model_GPU_or_scoring_calls': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--population', choices=tuple(COUNTS))
    parser.add_argument('--token')
    parser.add_argument('--target-version', choices=('full', 'gain1'), default='full')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--build-index', action='store_true')
    parser.add_argument('--watch-build-index', action='store_true')
    parser.add_argument('--populations', nargs='+', choices=tuple(COUNTS), default=list(COUNTS))
    parser.add_argument('--teacher-root', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--poll-seconds', type=float, default=30.)
    parser.add_argument('--verify-index', action='store_true')
    args = parser.parse_args()
    if not 0 < args.poll_seconds <= 60 or args.workers < 1:
        parser.error('poll-seconds must be in (0,60] and workers must be positive')
    if args.watch_build_index:
        watch_build(args.paths, args.populations, teacher_root=args.teacher_root, workers=args.workers, poll_seconds=args.poll_seconds)
        return
    if not args.population:
        parser.error('--population is required for indexing or reading one population')
    if args.build_index:
        value = build_index(args.paths, args.population, workers=args.workers, teacher_root=args.teacher_root)
        print(json.dumps({'status': value['status'], 'population': args.population, 'scenes': value['scenes']}), flush=True)
        return
    if not args.token:
        parser.error('--token is required to read one complete teacher scene')
    scene = read_teacher_cache(args.paths, args.population, args.token, args.target_version,
                               teacher_root=args.teacher_root, verify_index=args.verify_index)
    result = dict(scene['metadata'], teacher_interface_shape=list(scene['teacher_interface'].shape),
                  teacher_interface_dtype=str(scene['teacher_interface'].dtype))
    if args.out:
        result['export'] = save_scene(args.out, scene)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
