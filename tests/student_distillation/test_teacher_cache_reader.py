"""Private CPU cache fixtures; no native model, GPU or real score is fabricated."""
import builtins
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/student_distillation/read_teacher_cache.py'
SPEC = importlib.util.spec_from_file_location('nrkd_standalone_teacher_reader', SCRIPT)
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        np.savez(stream, **arrays)
    return reader.reference(path)


@pytest.fixture
def cache(tmp_path):
    root = tmp_path / 'private_synthetic_teacher'
    frozen = root / 'frozen_teacher'; frozen.mkdir(parents=True)
    features = root / 'features'; features.mkdir()
    random = np.random.default_rng(2026102711)
    posts = random.normal(size=(2, 2800, 384)).astype(np.float32)
    # Negative zero at a valid token checks FP32 addition bits; nonzero padding
    # must survive reconstruction, never be zero-filled or dropped.
    posts[:, 200, 0] = np.float32(-0.)
    image = np.zeros((2, 2800), bool); image[:, 128:1128] = True
    text = np.zeros((2, 2800), bool); text[:, 1128:2328] = True
    states = random.normal(size=(2, 24)).astype(np.float32)
    states[:, 12:16] = states[:, 20:24]
    inputs = {'base_post': posts.reshape(-1, 384), 'image_mask': image.reshape(-1), 'text_mask': text.reshape(-1),
              'token_ids': np.tile(np.arange(2800, dtype=np.int64), 2),
              'attention_mask': (image | text).reshape(-1).astype(np.int64),
              'position_ids': np.tile(np.arange(2800, dtype=np.int64), 2), 'state24': states}
    artifacts = {}
    for name, value in inputs.items():
        path = features / (name + '.bin'); value.tofile(path)
        artifacts[name] = dict(reader.reference(path), dtype=value.dtype.str, shape=list(value.shape))
    feature_rows = []
    for row, token in enumerate(('private_fixture_repair', 'private_fixture_gain1_keep')):
        layout = {name: inputs[name][row * 2800:(row + 1) * 2800] for name in reader.LAYOUT}
        feature_rows.append({'token': token, 'population': 'train', 'global_index': 100 + row, 'log_id': 'private_log',
                             'shard': 0, 'offset': row * 2800, 'length': 2800, 'state_row': row,
                             'actual_Base_interface_sha256': reader.sha_array(posts[row]),
                             'token_layout_sha256': reader.digest({name: reader.sha_array(value) for name, value in layout.items()})})
    index = features / 'INDEX.jsonl'
    index.write_text(''.join(json.dumps(row) + '\n' for row in feature_rows))
    feature_manifest = features / 'MANIFEST.json'
    reader.atomic_json(feature_manifest, {'schema': 'nrkd_native_post_mmap_v1', 'preserve_native_layout': True,
                      'synthetic_only': True, 'populations': {'train': {'count': 2, 'index': reader.reference(index),
                                              'shards': [{'arrays': artifacts}]}}})
    bundle_path = frozen / 'BUNDLE.json'
    reader.atomic_json(bundle_path, {'status': 'FROZEN_LOCAL_TEACHER', 'teacher_origin': 'local_refit', 'fit_population': 'train',
                      'teacher_identity': 'PRIVATE_SYNTHETIC_ONLY_NOT_A_REAL_TEACHER', 'source_checkpoint_sha256': 'a' * 64,
                      'selection_registry': {'path': str(frozen / 'SELECTIONS.json')}})
    folder = root / 'requests/train'; folder.mkdir(parents=True)
    tokens = [row['token'] for row in feature_rows]
    request_path = folder / 'REQUEST.json'
    reader.atomic_json(request_path, {'scope': 'full_population', 'population': 'train', 'tokens': tokens,
                      'indices': [100, 101], 'actions': [{'choice': choice} for choice in range(9)],
                      'feature_manifest': reader.reference(feature_manifest), 'bundle': reader.reference(bundle_path)})
    all_points, selected_shifts, expected = [], [], {}
    for row, token in enumerate(tokens):
        task_id = f'private_task{row}'
        valid = image[row] | text[row]
        roles = np.zeros((9, 2, 384), np.float32)
        for branch in range(1, 9):
            roles[branch, 0] = random.normal(scale=.005 * branch, size=384).astype(np.float32)
            roles[branch, 1] = roles[branch, 0]
        interfaces = np.repeat(posts[row][None], 9, axis=0)
        rms = np.zeros(9, np.float64)
        for branch in range(9):
            interfaces[branch, image[row]] = np.add(posts[row, image[row]], roles[branch, 0][None], dtype=np.float32)
            interfaces[branch, text[row]] = np.add(posts[row, text[row]], roles[branch, 1][None], dtype=np.float32)
            rms[branch] = np.sqrt(np.square(interfaces[branch, valid].astype(np.float64) - posts[row, valid].astype(np.float64)).mean())
        caps = np.array([0., *([.3] * 8)], np.float64)
        trajectories = np.empty((9, 5, 8, 3), np.float32)
        scores = np.ones((9, 5, 7), np.float64)
        for branch in range(9):
            for noise in range(5):
                trajectories[branch, noise] = branch + noise * .01 + np.arange(24, dtype=np.float32).reshape(8, 3) * .001
        scores[:, :, 0] = .6
        scores[0, :, 0] = .7 if row == 0 else .9
        scores[1 if row == 0 else 2, :, 0] = .74 if row == 0 else .905
        points = scores[:, :, 0].mean(1, dtype=np.float64) * 100.
        choice = int(points.argmax())
        all_points.append(points); selected_shifts.append(roles[choice])
        arrays = save_npz(folder / 'results' / task_id / 'SCORES.npz', positions=np.arange(9, dtype=np.int64),
                          noise_indices=np.arange(5, dtype=np.int64), trajectories=trajectories,
                          base_trajectories=trajectories[0], scores=scores, base_scores=scores[0],
                          actual_role_shifts=roles, actual_RMS=rms, cap_RMS=caps,
                          image_mask=image[row], text_mask=text[row], state24=states[row])
        result_path = folder / 'results' / task_id / 'COMPLETE.json'
        energy = float(np.square(posts[row].astype(np.float64)).mean())
        valid_energy = float(np.square(posts[row, valid].astype(np.float64)).mean())
        reader.atomic_json(result_path, {'status': 'COMPLETE', 'synthetic_only': True, 'token': token, 'population': 'train',
                          'task': {'id': task_id, 'row_index': row, 'index': 100 + row, 'positions': list(range(9))},
                          'arrays': arrays, 'feature_manifest': reader.reference(feature_manifest),
                          'Base_interface_sha256': reader.sha_array(posts[row]),
                          'immutable_full_padded_H0': {key: feature_rows[row][key] for key in ('shard', 'offset', 'length', 'state_row')},
                          'selected_branch_point_argmax': choice, 'selected_teacher_interface_sha256': reader.sha_array(interfaces[choice]),
                          'Base_padding_bytes_sha256': reader.sha_array(posts[row, ~valid]),
                          'selected_teacher_padding_bytes_sha256': reader.sha_array(interfaces[choice, ~valid]),
                          'valid_token_count': int(valid.sum()), 'Base_interface_energy_all': energy,
                          'Base_interface_energy_valid': valid_energy,
                          'selected_teacher_residual_energy_valid': float(rms[choice] ** 2),
                          'selected_teacher_residual_energy_all': float(rms[choice] ** 2) * float(valid.sum() / len(valid)),
                          'residual_energy_all_formula': 'valid/L', 'noise_identity': {'synthetic_only': True, 'not_real_noise': True},
                          'native_identity_sha256': 'b' * 64, 'metric_cache': 'PRIVATE_NOT_REAL_SCORE', 'local_metric_sha256': 'c' * 64})
        reader.atomic_json(folder / 'receipts' / (task_id + '.json'), {'status': 'COMPLETE', 'result': reader.reference(result_path)})
        expected[token] = {'full': interfaces[choice], 'base': posts[row], 'full_trajectories': trajectories[choice],
                           'base_trajectories': trajectories[0], 'full_quality': scores[choice], 'base_quality': scores[0]}
    points = np.asarray(all_points, np.float64)
    selection = save_npz(frozen / 'train.npz', tokens=np.asarray(tokens), indices=np.array([100, 101], np.int64),
                          choice=points.argmax(1).astype(np.int64), q_points=points,
                          selected_actual_role_shifts=np.asarray(selected_shifts, np.float32))
    registry = {'schema': 'nrkd_local_selection_registry_v1', 'teacher_identity': 'PRIVATE_SYNTHETIC_ONLY_NOT_A_REAL_TEACHER',
                'source_checkpoint_sha256': 'a' * 64, 'populations': {'train': dict(selection, units='points', diagnostic_only=False)}}
    registry['identity_sha256'] = reader.digest(registry)
    reader.atomic_json(frozen / 'SELECTIONS.json', registry)
    matrix = save_npz(folder / 'MATRIX.npz', synthetic_matrix=np.zeros((2, 9, 5, 7), np.float64))
    reader.atomic_json(folder / 'COMPLETE.json', {'status': 'COMPLETE', 'scenes': 2, 'actions': 9,
                                                'noise_indices': [0, 1, 2, 3, 4], 'matrix': matrix})
    reader.atomic_json(folder / 'QUEUE.json', {'request': reader.reference(request_path), 'tasks': ['private_task0', 'private_task1']})
    reader.atomic_json(frozen / 'TRAIN_EVALUATION.json', {'status': 'COMPLETE', 'population': 'train', 'scenes': 2,
                                                       'noise_indices': [0, 1, 2, 3, 4], 'selection': selection})
    paths = root / 'PATHS.json'
    reader.atomic_json(paths, {'paths': {'teacher_bundle': str(bundle_path), 'feature_manifest': str(feature_manifest)}})
    return root, paths, tokens, expected, artifacts


def test_full_reader_is_byte_exact_and_uses_saved_execution(cache, monkeypatch):
    root, paths, tokens, expected, _ = cache
    marker = reader.build_index(paths, 'train', workers=2, allow_synthetic=True)
    assert marker['scenes'] == 2
    original_import = builtins.__import__
    def reject_model_import(name, *args, **kwargs):
        if name == 'torch' or name == 'model' or name.startswith(('model.', 'interface_teacher')):
            raise AssertionError('Lossless reader attempted to load a model/head')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', reject_model_import)
    scene = reader.read_teacher_cache(paths, 'train', tokens[0], 'full', allow_synthetic=True)
    assert scene['teacher_interface'].shape == (2800, 384) and scene['teacher_interface'].dtype == np.float32
    assert scene['teacher_interface'].tobytes() == expected[tokens[0]]['full'].tobytes()
    assert scene['teacher_trajectories'].tobytes() == expected[tokens[0]]['full_trajectories'].tobytes()
    assert scene['teacher_quality'].tobytes() == expected[tokens[0]]['full_quality'].tobytes()
    assert scene['teacher_interface'][scene['padding_mask']].tobytes() == scene['Base_interface'][scene['padding_mask']].tobytes()
    assert scene['metadata']['global_index'] == 100
    assert scene['metadata']['same_execution_full_FP32_SHA_verified']
    artifact = reader.save_scene(root / 'readback.npz', scene)
    with np.load(artifact['path'], allow_pickle=False) as exported:
        assert exported['teacher_interface'].tobytes() == scene['teacher_interface'].tobytes()
        assert json.loads(str(exported['metadata_json']))['no_model_or_head_rerun']


def test_gain1_keep_uses_literal_Base_branch_even_with_nonBase_full_choice(cache):
    _, paths, tokens, expected, _ = cache
    reader.build_index(paths, 'train', allow_synthetic=True)
    scene = reader.read_teacher_cache(paths, 'train', tokens[1], 'gain1', allow_synthetic=True)
    assert scene['metadata']['original_full_choice'] == 2
    assert scene['metadata']['returned_choice'] == 0 and not scene['metadata']['gain1_keep']
    assert scene['teacher_interface'].tobytes() == expected[tokens[1]]['base'].tobytes()
    assert scene['teacher_trajectories'].tobytes() == expected[tokens[1]]['base_trajectories'].tobytes()
    assert scene['teacher_quality'].tobytes() == expected[tokens[1]]['base_quality'].tobytes()
    assert not scene['actual_shift'].any()


def test_reader_rejects_changed_Base_mmap_bytes(cache):
    _, paths, tokens, _, artifacts = cache
    reader.build_index(paths, 'train', allow_synthetic=True)
    array = np.memmap(artifacts['base_post']['path'], mode='r+', dtype=np.float32, shape=tuple(artifacts['base_post']['shape']))
    array[10, 10] += np.float32(1.)
    array.flush()
    with pytest.raises(ValueError, match='Base bytes changed'):
        reader.read_teacher_cache(paths, 'train', tokens[0], allow_synthetic=True)


def test_population_registry_append_does_not_mutate_existing_index(cache):
    root, paths, tokens, _, _ = cache
    first = reader.build_index(paths, 'train', allow_synthetic=True)
    registry_path = root / 'frozen_teacher/SELECTIONS.json'
    registry = reader.read_json(registry_path)
    registry['populations']['validation'] = {'path': 'NOT_READ_BY_TRAIN', 'sha256': 'd' * 64, 'units': 'points', 'diagnostic_only': True}
    registry.pop('identity_sha256'); registry['identity_sha256'] = reader.digest(registry)
    reader.atomic_json(registry_path, registry)
    second = reader.build_index(paths, 'train', allow_synthetic=True)
    assert first == second
    assert reader.read_teacher_cache(paths, 'train', tokens[0], allow_synthetic=True)['metadata']['target_version'] == 'full'
    with reader.TeacherCacheReader(paths, 'train', allow_synthetic=True) as direct:
        assert direct[tokens[0]]['teacher_interface'].shape == (2800, 384)
        assert direct.read(tokens[1], 'gain1')['metadata']['returned_choice'] == 0


def test_cpu_watcher_builds_population_immediately_and_binds_artifacts(cache, monkeypatch):
    root, paths, tokens, _, _ = cache
    monkeypatch.setitem(reader.COUNTS, 'train', 2)
    reader.watch_build(paths, ['train'], workers=2, poll_seconds=.01)
    marker = reader.read_json(root / 'repaired_cache_index_watcher/COMPLETE.json')
    assert marker['status'] == 'COMPLETE' and marker['populations'] == ['train']
    assert marker['no_model_GPU_or_scoring_calls']
    assert reader.resolve(marker['artifacts']['train'], root).is_file()
    assert reader.read_teacher_cache(paths, 'train', tokens[0])['metadata']['same_execution_full_FP32_SHA_verified']
