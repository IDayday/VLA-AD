"""Local-only D/C_to_P/K8 double refit, with immutable original math kernels.

Stages: prepare -> fit-inner -> pack-pilot -> collect-pilot -> pack-cal ->
collect-cal -> refit -> freeze -> queue-population -> collect-population.
Only the two native score stages and full-population evaluation need GPUs.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
import uuid
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
from gt_numeric import atomic, bind_function, digest, read, sha, stamp
from teacher_library import coefficient, original_modules

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / 'source_snapshot/ReCogDrive'
FEATURES = Path('/mnt/project/recogdrive-no-routing-round1-results-20261026/features/MANIFEST.ALL_POPULATIONS.json')
BASE_POOL = Path('/mnt/project/recogdrive-dc8-local-refit-20261026/inputs/LOCAL_TRAIN_BASE_POOL.npz')
ROUND1 = Path('/mnt/project/recogdrive-no-routing-round1-20261026')
CHECKPOINT_SHA = '23fb949b10387efeff87e093703d5faf41b61b9e5ead16d4076661e3c7c08a20'
_FIT_CONTEXT = None


def ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def save_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    with np.load(temporary, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays) or any(not np.array_equal(saved[key], np.asarray(value)) for key, value in arrays.items()):
            raise ValueError('Array persistence readback failed')
    temporary.replace(path)
    return ref(path)


def save_state(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    return ref(path)


def load_state(reference):
    import torch
    path = Path(reference['path'] if isinstance(reference, dict) else reference)
    if isinstance(reference, dict) and sha(path) != reference['sha256']:
        raise ValueError('State SHA mismatch')
    return torch.load(path, map_location='cpu', weights_only=False)


def source_paths(source):
    code = Path(source) / '1024-representation_distillation_diagnosis_batch5/stage2/offline_library/code'
    return code, code.parent


def source_bindings(source):
    code, _ = source_paths(source)
    names = ('ol_stage_a_final_factory_v1', 'ol_l1_pilot_fitter_v1', 'ol_l1_geometry_v1',
             'ol_l1_operators_v1', 'ol_l1_pilot_selection_v1', 'ol_l2_formal_selection_v2',
             'ol_l3_outer_refit_v1', 'ol_cv_shared48_cpu_v1')
    paths = [code / (name + '.py') for name in names]
    ops, geo = original_modules(code)
    paths += [Path(module.__file__) for module in (ops.old, ops.paired, geo.old_geometry, geo.old_directions, geo.old_paired)]
    return [ref(path) for path in sorted(set(paths))]


def bind_math(source):
    """Bind original functions without their historical persistence/release globals."""
    code, _ = source_paths(source)
    ops, geo = original_modules(code)
    namespace = {'np': np, 'json': json, 'hashlib': hashlib, 'itertools': itertools,
                 'ops': ops, 'geo': geo, 'native': SimpleNamespace(fitter=SimpleNamespace(ops=ops))}
    functions, identities = {}, []
    for filename, names in (
        ('ol_l1_pilot_fitter_v1.py', ('ordinary_view', 'coordinates', 'initialization')),
        ('ol_l1_pilot_selection_v1.py', ('canonical', 'objective', 'optimize', 'marginal_methods')),
        ('ol_l2_formal_selection_v2.py', ('optimize',)),
        ('ol_l3_outer_refit_v1.py', ('max_overlap', 'exact_matching', 'adjusted_rand', 'physical_hash', 'fit_selected_paired')),
        ('ol_cv_shared48_cpu_v1.py', ('hash_panel', 'pilot_panel'))):
        for name in names:
            from scipy.optimize import linear_sum_assignment
            namespace['linear_sum_assignment'] = linear_sum_assignment
            function, identity = bind_function(code / filename, name, namespace)
            # bind_function gives each function private globals; link original
            # dependencies as bindings accumulate, never rewrite its __code__.
            key = ('formal_optimize' if filename == 'ol_l2_formal_selection_v2.py' else name)
            functions[key] = function
            namespace[name] = function
            identities.append(identity)
    for key, function in functions.items():
        own_name = function.__name__
        function.__globals__.update(namespace)
        function.__globals__[own_name] = function
    return ops, geo, functions, identities


def prepare(args):
    code, old = source_paths(args.source_root)
    split_path = old / 'preparation/session_cv_metadata_v1/SPLITS.json'
    split = read(split_path)
    inputs_path = old / 'preparation/l1_data_pack_v1/INPUTS10042.json'
    rows = read(inputs_path)['cases']
    rows = [{**row, 'category': row['audit_category']} for row in rows]
    by_index = {int(row['index']): row for row in rows}
    fit, cal = split['final_locked_family_inner_fit'], split['final_locked_family_inner_calibration']
    if len(rows) != 10042 or len(fit['selected_indices']) != 8021 or len(cal['selected_indices']) != 2021 or len(fit['sessions']) != 104 or len(cal['sessions']) != 26:
        raise ValueError('The original final 10042/8021/2021 session split changed')
    if set(fit['selected_indices']) & set(cal['selected_indices']) or set(fit['sessions']) & set(cal['sessions']):
        raise ValueError('Original final split is not session isolated')
    if set(by_index) != set(fit['selected_indices']) | set(cal['selected_indices']):
        raise ValueError('Original final split index coverage mismatch')
    if args.features:
        sys.path.insert(0, str(ROUND1))
        from student_distillation.data import FeatureStore
        store = FeatureStore(args.features, 'train')
    else:
        raise ValueError('Prepare needs the verified local ordinary feature mmap to recover state24')
    with np.load(args.base_pool, allow_pickle=False) as pool:
        position = {int(value): index for index, value in enumerate(pool['indices_int64'])}
        take = [position[int(row['index'])] for row in rows]
        if pool['tokens'][take].tolist() != [row['token'] for row in rows] or pool['sessions'][take].tolist() != [row['session'] for row in rows]:
            raise ValueError('Local Base pool cannot be joined by original global index/token/session')
        arrays = {'indices_int64': np.asarray([row['index'] for row in rows], np.int64),
                  'tokens': np.asarray([row['token'] for row in rows]),
                  'Base_pooled_float64': np.array(pool['Base_pooled_float64'][take]),
                  'token_counts_int64': np.array(pool['token_counts_int64'][take]),
                  'Base_quality_float64': np.array(pool['Base_quality_float64'][take]),
                  'raw_aux24_float32': np.stack([np.array(store.read(store.by_token[row['token']])['state24']) for row in rows])}
    ops, geo, math, binding = bind_math(args.source_root)
    doses = read(old / 'preparation/l1_pilot_metadata_v1/MANIFEST.json')['doses']
    panel = math['pilot_panel'](0, cal, by_index)
    folder = args.root / 'fitting'
    feature_snapshot = folder / ('FEATURES_' + sha(args.features) + '.json')
    feature_snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not feature_snapshot.exists():
        shutil.copyfile(args.features, feature_snapshot)
    if sha(feature_snapshot) != sha(args.features):
        raise ValueError('Frozen local feature-manifest snapshot SHA mismatch')
    ordinary = save_npz(folder / 'ORDINARY10042.npz', arrays)
    plan = {'schema': 'local_dc8_refit_plan_v1', 'status': 'PREPARED_LOCAL_ORDINARY_METADATA',
            'root': str(args.root), 'source_root': str(args.source_root), 'source_code_root': str(code),
            'original_math_sources': source_bindings(args.source_root), 'math_bindings': binding,
            'original_split': ref(split_path), 'original_inputs': ref(inputs_path), 'local_Base_pool': ref(args.base_pool),
            'feature_manifest': ref(feature_snapshot), 'ordinary': ordinary, 'rows': rows, 'fit': fit, 'cal': cal,
            'panel': panel, 'doses': doses, 'menu': list(ops.MENU), 'family': {'route': 'D', 'order': 'C_to_P', 'K': 8, 'rank': 48, 'view': 'valid_shared384'},
            'whole_variant': {'raw_shift_factor': 2., 'cap_factor': 2.}, 'gt_root': str(args.gt_root),
            'new_qualified_pair_count': None, 'historical_teacher_parameters_choices_or_subspace_used': False}
    atomic(folder / 'PLAN.json', plan)
    return {'status': plan['status'], 'plan': str(folder / 'PLAN.json'), 'pilot32': [row['index'] for row in panel],
            'next': 'fit-inner after all1850 newly selected GT-post exports exist'}


def load_pairs(plan):
    rows = [row for row in plan['rows'] if row['audit_category'] == 'failure']
    if len(rows) != 1850:
        raise ValueError('Local pair export population must retain original1850 failure IDs')
    records, arrays, missing = [], [], []
    for row in rows:
        folder = Path(plan['gt_root']) / 'exports/post' / row['token']
        marker = folder / 'COMPLETE.json'
        if not marker.exists():
            missing.append(row['token']); continue
        meta = read(marker)
        if meta['status'] != 'PASS_LOCAL_FROZEN_BASE_GT_POST_PAIR' or meta['token'] != row['token'] or int(meta['global_index']) != int(row['index']) or meta['population'] != 'train':
            raise ValueError('GT export metadata mismatch')
        file = Path(plan['gt_root']) / meta['arrays']['path']
        if sha(file) != meta['arrays']['sha256']:
            raise ValueError('GT export SHA mismatch')
        with np.load(file, allow_pickle=False) as saved:
            base, teacher = np.asarray(saved['Base_interface']), np.asarray(saved['teacher_interface'])
            image, text = np.asarray(saved['image_mask']), np.asarray(saved['text_mask'])
            valid = image | text
            if base.dtype != np.float32 or teacher.dtype != np.float32 or base.shape != teacher.shape or base.ndim != 2 or base.shape[1] != 384 or not np.isfinite(teacher).all():
                raise ValueError('GT post must preserve finite native FP32 full layout')
            if not np.array_equal(base[~valid], teacher[~valid]):
                raise ValueError('GT post changed native padding')
            base_quality = np.asarray(saved['Base_quality']) if 'Base_quality' in saved else None
            if base_quality is None:
                input_file = ROOT / 'native_base/train/cases' / row['token'] / 'BASE_INPUTS.npz'
                with np.load(input_file, allow_pickle=False) as native:
                    if not np.array_equal(base, native['Base_interface']):
                        raise ValueError('GT/Base local input mismatch')
                    base_quality = np.array(native['Base_quality'])
            b0, t0 = float(base_quality[0, 0]), float(saved['quality'][0, 0])
            qualified = b0 < .7 and t0 > .9
            counts = np.asarray([image.sum(), text.sum()], np.int64)
            pooled_base = np.stack([base[mask].astype(np.float64).mean(0) for mask in (image, text)])
            pooled_teacher = np.stack([teacher[mask].astype(np.float64).mean(0) for mask in (image, text)])
            rms = float(np.sqrt(np.square(teacher[valid].astype(np.float64) - base[valid].astype(np.float64)).mean()))
        records.append({'index': row['index'], 'token': row['token'], 'session': row['session'], 'qualified': qualified,
                        'Base_S0': b0, 'teacher_S0': t0, 'actual_fulltoken_RMS': rms, 'source': ref(marker)})
        arrays.append((pooled_base, pooled_teacher, counts, rms))
    if missing:
        atomic(Path(plan['root']) / 'fitting/PAIR_EXPORTS_PENDING.json', {'status': 'WAITING_NEW_GT_POST_EXPORTS', 'missing_count': len(missing), 'tokens': missing})
        raise RuntimeError(f'New local GT-post exports not ready: {len(missing)}/1850 missing')
    qualified = [record for record in records if record['qualified']]
    atomic(Path(plan['root']) / 'fitting/LOCAL_PAIR_QUALIFICATION.json',
           {'status': 'COMPLETE', 'source': 'actual local S0', 'total': 1850, 'qualified': len(qualified),
            'rule': 'Base_S0<0.7 and selected_GT_post_S0>0.9', 'rows': records, 'no_historical_count_assumption': True})
    return records, {record['index']: arrays[index] for index, record in enumerate(records)}


def build_space(plan, scope, records, pairs):
    ops, geo, math, binding = bind_math(plan['source_root'])
    ordinary = dict(np.load(plan['ordinary']['path'], allow_pickle=False))
    lookup = {int(value): index for index, value in enumerate(ordinary['indices_int64'])}
    rows = {row['index']: row for row in plan['rows']}
    indices = plan['fit']['selected_indices'] if scope == 'INNER' else [row['index'] for row in plan['rows']]
    qualified = {record['index'] for record in records if record['qualified']}
    q = [index for index in indices if index in qualified]
    normal = [index for index in indices if rows[index]['audit_category'] in ('mid', 'high')]
    high = [index for index in normal if rows[index]['audit_category'] == 'high']
    if not q:
        raise ValueError('No newly qualified local GT-post pairs')
    base = np.stack([pairs[index][0] for index in q]); teacher = np.stack([pairs[index][1] for index in q])
    counts = np.stack([pairs[index][2] for index in q]); rms = np.asarray([pairs[index][3] for index in q], np.float64)
    expected = ordinary['Base_pooled_float64'][[lookup[index] for index in q]]
    if not np.array_equal(base, expected) or not np.array_equal(counts, ordinary['token_counts_int64'][[lookup[index] for index in q]]):
        raise ValueError('New local GT pair Base differs from independent frozen Base pool')
    failed, success, view = geo.old_geometry.view_vectors(base, teacher, counts, 'valid_shared384', np.arange(len(q)))
    space = geo.fit_space(failed, success, rms, 'D', 'valid_shared384', 48)
    if space['status'] != 'VALID_SPACE':
        raise ValueError(f'Fresh D48 invalid: {space["status"]}; no old-basis fallback')
    identity = digest({'scope': scope, 'plan': digest(plan), 'Q': q, 'normal': normal, 'high': high,
                       'sessions': sorted({rows[index]['session'] for index in indices}), 'local_pair_sources': [record['source'] for record in records if record['index'] in q]})
    geometry = geo.geometry(space['basis'], view, counts[:, 0] / counts.sum(1), success - failed, rms)
    state = {'space': space, 'view': view, 'D0': float(np.median(rms)), 'Q_indices': np.asarray(q, np.int64),
             'normal_indices': normal, 'high_indices': high, 'fail_view': failed, 'success_view': success,
             'actual_fulltoken_RMS': rms, 'raw_aux': ordinary['raw_aux24_float32'][[lookup[index] for index in q]],
             'fit_identity': identity, 'recipe': {'route': 'D', 'view': 'valid_shared384', 'rank': 48},
             'tokens': [rows[index]['token'] for index in q], 'sessions': [rows[index]['session'] for index in q], 'geometry': geometry}
    cluster = geo.cluster(geometry, 'C_to_P', 8, state['tokens'], state['sessions'], identity)
    if cluster['status'] != 'VALID_DIRECTION_CLUSTER':
        raise ValueError('Fresh full384 C_to_P K8 cluster invalid; no old cluster fallback')
    state['cluster'] = cluster
    ordinary_view = math['ordinary_view'](ordinary['Base_pooled_float64'], ordinary['token_counts_int64'], view)
    z = math['coordinates'](ordinary_view, space['center'], space['basis'])
    common = []
    for slot in range(8):
        take = np.flatnonzero(cluster['labels'] == slot)
        common.append(ops.algebra(math['coordinates'](failed[take], space['center'], space['basis']),
                                 math['coordinates'](success[take], space['center'], space['basis']),
                                 geometry['coefficients'][take], geometry['L'], state['D0'],
                                 geometry['orders']['P_to_C']['no_direction'][take]) if len(take) else None)
    return state, ordinary, ordinary_view, z, lookup, common, (ops, geo, math, binding)


def mlp_fit(dataset, d0, seed, destination):
    """Use the original exact2000 step kernel and independently replay1999->2000."""
    import torch
    ops = mlp_fit.ops
    marker = Path(destination) / 'COMPLETE.json'
    identity = digest({'x_sha': hashlib.sha256(dataset['x'].tobytes()).hexdigest(),
                       'y_sha': hashlib.sha256(dataset['y'].tobytes()).hexdigest(),
                       'weights_sha': hashlib.sha256(dataset['weights'].tobytes()).hexdigest(),
                       'd0': d0, 'seed': int(seed)})
    if marker.exists():
        meta = read(marker)
        saved = load_state(meta['endpoint'])
        if meta.get('dataset_identity') is None:
            expected = ops.MLPTrainingSession(dataset, d0, seed, 'cpu')
            if saved['identity'] != expected.identity or saved['seed'] != seed or saved['D0'] != d0:
                raise ValueError('Saved local MLP endpoint belongs to another dataset/seed')
        elif meta['dataset_identity'] != identity:
            raise ValueError('Saved local MLP endpoint belongs to another dataset/seed')
        return saved
    session = ops.MLPTrainingSession(dataset, d0, seed, 'cpu')
    curves = []
    for update in range(2000):
        before = session.snapshot() if update == 1999 else None
        event = session.step()
        if update in (0, 99, 399, 999, 1999):
            curves.append({key: value for key, value in event.items() if key != 'batch_indices'})
        if update == 1999:
            after = session.snapshot()
            second = ops.MLPTrainingSession(dataset, d0, seed, 'cpu'); second.restore(before)
            again = second.step()
            from student_distillation.train import _assert_same_nested
            _assert_same_nested(event, again)
            _assert_same_nested(after, second.snapshot())
    snapshot = session.snapshot()
    endpoint = save_state(Path(destination) / 'ENDPOINT2000.pt', snapshot)
    replay = save_state(Path(destination) / 'REPLAY1999_TO2000.pt', {'before': before, 'after': after, 'second_after': second.snapshot()})
    atomic(marker, {'status': 'COMPLETE', 'updates': 2000, 'endpoint': endpoint, 'replay': replay, 'dataset_identity': identity,
                    'curves': curves, 'initial_seed': int(seed), 'weak_fit_is_not_PDMS_veto': True})
    return snapshot


def fit_payload(method, slot, state, ordinary, ordinary_view, z, lookup, common, math_bundle, destination, selected_only=False):
    ops, geo, math, _ = math_bundle
    take = np.flatnonzero(state['cluster']['labels'] == slot)
    support = ops.support(method, len(take), len(state['high_indices']), len(state['normal_indices'])) if method != 'Base' else {'Base_fallback_required': True, 'status': 'FROZEN_BASE_SLOT'}
    if support['Base_fallback_required']:
        return {'kind': 'base', 'method': method, 'Base_fallback_required': True, 'support': support}
    basis, center = state['space']['basis'], state['space']['center']
    failed = math['coordinates'](state['fail_view'][take], center, basis)
    success = math['coordinates'](state['success_view'][take], center, basis)
    high = z[[lookup[index] for index in state['high_indices']]]
    if method.startswith('mlp'):
        dataset = ops.paired_rows(failed, success, high if method == 'mlp_replay' else None)
        seed = math['initialization'](state, 8, method)
        mlp_fit.ops = ops
        return {'kind': 'mlp', 'method': method, 'snapshot': mlp_fit(dataset, state['D0'], seed, destination), 'Base_fallback_required': False}
    if method in ('paired_RRR', 'direction_amplitude'):
        target, proof = ops.paired_targets(basis, state['success_view'][take] - state['fail_view'][take], state['view'],
                                          state['geometry']['fractions'][take], state['actual_fulltoken_RMS'][take], state['D0'])
        normal = [lookup[index] for index in state['normal_indices']]
        arguments = (state['fail_view'][take], state['raw_aux'][take].astype(np.float64), ordinary_view[normal],
                     ordinary['raw_aux24_float32'][normal].astype(np.float64), target, state['D0'])
        model = math['fit_selected_paired'](*arguments, method) if selected_only else ops.fit_paired(*arguments)
        return {'kind': 'paired_single' if selected_only else 'paired', 'method': method, 'model': model,
                'target': target, 'proof': proof, 'Base_fallback_required': False}
    return {'kind': 'analytic', 'method': method, 'model': ops.fit_original(method, failed, success, common[slot], high), 'Base_fallback_required': False}


def _fit_inner_cluster(slot):
    state, ordinary, ordinary_view, z, lookup, common, math_bundle, folder = _FIT_CONTEXT
    methods = {}
    for method in math_bundle[0].MENU:
        destination = folder / f'cluster{slot:02d}' / method
        marker = destination / 'MODEL.json'
        if marker.exists():
            previous = read(marker)
            if previous['fit_identity'] != state['fit_identity']:
                raise ValueError('Existing inner operator belongs to a different local fit identity')
            model_ref = previous['payload']
            load_state(model_ref)
        else:
            payload = fit_payload(method, slot, state, ordinary, ordinary_view, z, lookup, common, math_bundle, destination / 'MLP2000')
            model_ref = save_state(destination / 'PAYLOAD.pt', payload)
            atomic(marker, {'status': 'COMPLETE', 'payload': model_ref, 'fit_identity': state['fit_identity']})
        methods[method] = model_ref
        print(json.dumps({'stage': 'fit-inner', 'cluster': slot, 'method': method, 'status': 'COMPLETE'}), flush=True)
    return {'cluster': slot, 'methods': methods}


def cpu_cluster_map(function, workers):
    if workers == 1:
        return [function(slot) for slot in range(8)]
    # Only CPU arrays/kernels have initialized here. Fork preserves shared
    # read-only arrays and exact per-method private generators; one cluster
    # writes one folder. No GPU context is inherited or used for fitting.
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU cluster workers must start before CUDA initialization')
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=min(8, workers), mp_context=multiprocessing.get_context('fork')) as executor:
        return list(executor.map(function, range(8)))


def fit_inner(args):
    global _FIT_CONTEXT
    plan = read(args.root / 'fitting/PLAN.json')
    known_path = args.root / 'selftests/all13_cpu/COMPLETE.json'
    known = read(known_path) if known_path.exists() else None
    if not known or known.get('source_bindings') != plan['original_math_sources'] or not known.get('known_realizable_MLP_role_absorptions'):
        selftest(args)
    records, pairs = load_pairs(plan)
    state, ordinary, ordinary_view, z, lookup, common, math_bundle = build_space(plan, 'INNER', records, pairs)
    folder = args.root / 'fitting/INNER'
    state_ref = save_state(folder / 'SPACE_AND_CLUSTER.pt', state)
    _FIT_CONTEXT = (state, ordinary, ordinary_view, z, lookup, common, math_bundle, folder)
    entries = cpu_cluster_map(_fit_inner_cluster, args.fit_workers)
    result = {'status': 'COMPLETE_LOCAL_INNER_ALL13', 'state': state_ref, 'clusters': entries,
              'new_local_qualified_total': sum(record['qualified'] for record in records),
              'qualified_inner': len(state['Q_indices']), 'scope': 'original8021/104sessions',
              'source_binding': plan['math_bindings'], 'no_calibration_teacher_arrays_fitted': True}
    atomic(folder / 'BANK.json', result)
    return result


def queue_request(root, request, action_chunk=8):
    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    request_ref = ref(request)
    value = read(request)
    tasks = []
    for row_index, index in enumerate(value['indices']):
        for first in range(0, len(value['actions']), action_chunk):
            positions = list(range(first, min(first + action_chunk, len(value['actions']))))
            identity = {'request': request_ref, 'row_index': row_index, 'index': index, 'positions': positions}
            task = dict(id=digest(identity), **identity)
            atomic(destination / 'tasks' / (task['id'] + '.json'), task)
            tasks.append(task['id'])
    atomic(destination / 'QUEUE.json', {'status': 'READY_NATIVE_SCORING', 'request': request_ref,
           'tasks': tasks, 'total_tasks': len(tasks), 'scene_count': len(value['indices']), 'noise_indices': value['noise_indices'],
           'logical_expert_trajectories': len(value['indices']) * len(value['actions']) * len(value['noise_indices']), 'fitting_allowed': False})
    return {'queue': str(destination / 'QUEUE.json'), 'tasks': len(tasks)}


def validate_task(task, identity, request_ref, request):
    fields = {key: task[key] for key in ('request', 'row_index', 'index', 'positions')}
    if task.get('id') != identity or digest(fields) != identity or task['request'] != request_ref:
        raise ValueError('Native task content/request SHA identity mismatch')
    row, positions = task['row_index'], task['positions']
    if not isinstance(row, int) or not 0 <= row < len(request['indices']) or request['indices'][row] != task['index']:
        raise ValueError('Native task row/global index mismatch')
    if not positions or len(set(positions)) != len(positions) or min(positions) < 0 or max(positions) >= len(request['actions']):
        raise ValueError('Native task action positions are invalid')


def pack(args, scope):
    plan = read(args.root / 'fitting/PLAN.json')
    bank = read(args.root / 'fitting/INNER/BANK.json')
    state = load_state(bank['state'])
    ops, geo, math, _ = bind_math(plan['source_root'])
    ordinary = dict(np.load(plan['ordinary']['path'], allow_pickle=False))
    lookup = {int(value): index for index, value in enumerate(ordinary['indices_int64'])}
    rows = {row['index']: row for row in plan['rows']}
    if scope == 'pilot':
        indices = [row['index'] for row in plan['panel']]
        weights = [row['natural_mean_weight'] for row in plan['panel']]
        menus = [list(ops.MENU) for _ in range(8)]
        noise = [0, 1]
    else:
        coarse = read(args.root / 'fitting/COARSE_SELECTION.json')
        menus = [row['retained_methods'] for row in coarse['retained_per_slot']]
        indices = plan['cal']['selected_indices']
        weights = [plan['cal']['conditional_natural_mean_weight_per_sample'][rows[index]['audit_category']] for index in indices]
        noise = [0, 1, 2]
    take = [lookup[index] for index in indices]
    v = math['ordinary_view'](ordinary['Base_pooled_float64'][take], ordinary['token_counts_int64'][take], state['view'])
    raw = np.zeros((len(indices), 8, len(menus[0]), 48), np.float64)
    actions, networks = [], {}
    for slot in range(8):
        for position, method in enumerate(menus[slot]):
            payload = load_state(bank['clusters'][slot]['methods'][method]); payload['_ops'] = ops
            for row_index, original in enumerate(take):
                raw[row_index, slot, position] = coefficient(payload, v[row_index], ordinary['raw_aux24_float32'][original].astype(np.float64),
                       float(ordinary['token_counts_int64'][original, 0] / ordinary['token_counts_int64'][original].sum()), state['space']['basis'], state['space']['center'], state['view'], networks, (slot, method))
            for dose in plan['doses']:
                action = {'cluster': slot, 'method': method, 'method_position': position, **dose,
                          'valid': not payload['Base_fallback_required']}
                actions.append(dict(action_id=digest(action), **action))
    folder = args.root / 'requests' / scope
    compact = save_npz(folder / 'COMPACT.npz', {'indices': np.asarray(indices, np.int64), 'raw_coefficients': raw,
                         'natural_weights': np.asarray(weights, np.float64), 'basis': state['space']['basis'], 'D0': np.asarray(state['D0'])})
    request = {'schema': 'local_dc8_native_request_v1', 'scope': scope, 'population': 'train', 'indices': indices,
               'tokens': [rows[index]['token'] for index in indices], 'actions': actions, 'noise_indices': noise,
               'compact': compact, 'feature_manifest': plan['feature_manifest'], 'raw_shift_factor': 1., 'cap_factor': 1.,
               'score_units': 'fraction', 'all_experts_cross_applied_without_routing': True, 'plan': ref(args.root / 'fitting/PLAN.json')}
    atomic(folder / 'REQUEST.json', request)
    return queue_request(folder, folder / 'REQUEST.json', args.action_chunk)


def collect_request(folder):
    folder = Path(folder)
    queue = read(folder / 'QUEUE.json'); request = read(queue['request']['path'])
    n, a, s = len(request['indices']), len(request['actions']), len(request['noise_indices'])
    scores = np.full((n, a, s, 7), np.nan, np.float64); base = np.full((n, s, 7), np.nan, np.float64)
    seen = np.zeros((n, a), bool); missing = []
    for ident in queue['tasks']:
        marker = folder / 'receipts' / (ident + '.json')
        if not marker.exists():
            missing.append(ident); continue
        meta = read(marker)
        result = read(meta['result']['path'])
        if sha(meta['result']['path']) != meta['result']['sha256'] or sha(result['arrays']['path']) != result['arrays']['sha256']:
            raise ValueError('Native scoring receipt SHA mismatch')
        task = result['task']; row, positions = task['row_index'], task['positions']
        validate_task(task, ident, queue['request'], request)
        if task != read(folder / 'tasks' / (ident + '.json')) or seen[row, positions].any():
            raise ValueError('Native task result differs from registered task or duplicates action coverage')
        with np.load(result['arrays']['path'], allow_pickle=False) as values:
            scores[row, positions] = values['scores']
            if np.isfinite(base[row]).all() and not np.array_equal(base[row], values['base_scores']):
                raise ValueError('Base scores differ between action chunks')
            base[row] = values['base_scores']; seen[row, positions] = True
    if missing:
        atomic(folder / 'PENDING.json', {'status': 'WAITING_NATIVE_SCORE_TASKS', 'missing': len(missing), 'total': len(queue['tasks']), 'task_ids': missing})
        raise RuntimeError(f'Native score queue incomplete: {len(missing)}/{len(queue["tasks"])} tasks pending')
    if not seen.all() or not np.isfinite(scores).all() or not np.isfinite(base).all():
        raise ValueError('Incomplete/nonfinite native score matrix')
    matrix = save_npz(folder / 'MATRIX.npz', {'scores': scores, 'base_scores': base, 'indices': np.asarray(request['indices'], np.int64)})
    atomic(folder / 'COMPLETE.json', {'status': 'COMPLETE', 'matrix': matrix, 'scenes': n, 'actions': a, 'noise_indices': request['noise_indices']})
    return request, scores, base


def collect(args, scope):
    request, scores, base = collect_request(args.root / 'requests' / scope)
    plan = read(args.root / 'fitting/PLAN.json'); _, _, math, _ = bind_math(plan['source_root'])
    with np.load(request['compact']['path'], allow_pickle=False) as compact:
        weights = np.array(compact['natural_weights'])
    bmean, table = base[:, :, 0].mean(1), scores[:, :, :, 0].mean(2)
    if scope == 'pilot':
        centroid = next(index for index, action in enumerate(request['actions']) if action['cluster'] == 0 and action['method'] == 'centroid_D0' and action['alpha'] == 1 and action['cap_D0'] == 2)
        selection = math['optimize'](bmean, table, weights, request['actions'], 8, centroid)
        retained = math['marginal_methods'](bmean, table, weights, request['actions'], 8, centroid)
        result = {'status': 'COMPLETE_LOCAL_COARSE_SELECTION', 'selection': selection, 'retained_per_slot': retained, 'request': ref(args.root / 'requests/pilot/REQUEST.json')}
        atomic(args.root / 'fitting/COARSE_SELECTION.json', result)
    else:
        selection = math['formal_optimize'](bmean, table, weights, request['actions'], 8)
        slots = []
        for cluster, position in enumerate(selection['positions']):
            slots.append({'cluster': cluster, 'Base': True, 'method': 'Base', 'alpha': 0., 'cap_D0': 0.} if position < 0 else
                         dict(Base=False, **{key: request['actions'][position][key] for key in ('cluster', 'method', 'alpha', 'cap_D0')}))
        result = {'status': 'FROZEN_LOCAL_INNER_SELECTION', 'selection': selection, 'slots': slots,
                  'raw_shift_factor': 2., 'cap_factor': 2., 'whole_variant_preselected_by_user': True,
                  'original2021_calibration_indices': True, 'natural_conditional_weighted_PDMS': selection['J'] * 100,
                  'request': ref(args.root / 'requests/cal/REQUEST.json'), 'no_test_or_validation_selection': True}
        atomic(args.root / 'fitting/FROZEN_INNER_SELECTION.json', result)
    return result


def _refit_slot(old_slot):
    state, ordinary, ordinary_view, z, lookup, common, math_bundle, folder, selection, mapping = _FIT_CONTEXT
    new_cluster = mapping['old_to_new'][old_slot]
    specification = selection['slots'][old_slot]
    destination = folder / f'slot{old_slot:02d}'
    payload = fit_payload(specification['method'], new_cluster, state, ordinary, ordinary_view, z, lookup, common,
                          math_bundle, destination / 'MLP2000', selected_only=True)
    payload_ref = save_state(destination / 'PAYLOAD.pt', payload)
    print(json.dumps({'stage': 'refit-selected', 'old_slot': old_slot, 'new_cluster': new_cluster, 'method': specification['method']}), flush=True)
    return {'old_slot': old_slot, 'new_cluster': new_cluster, 'selection': specification, 'payload': payload_ref}


def refit(args):
    global _FIT_CONTEXT
    plan = read(args.root / 'fitting/PLAN.json'); selection = read(args.root / 'fitting/FROZEN_INNER_SELECTION.json')
    records, pairs = load_pairs(plan)
    state, ordinary, ordinary_view, z, lookup, common, math_bundle = build_space(plan, 'ALLTRAIN', records, pairs)
    old = load_state(read(args.root / 'fitting/INNER/BANK.json')['state'])
    hashes = [math_bundle[2]['physical_hash'](value, state)[0] for value in common]
    where = {int(index): row for row, index in enumerate(state['Q_indices'])}
    new_labels = state['cluster']['labels'][[where[int(index)] for index in old['Q_indices']]]
    contingency = np.zeros((8, 8), np.int64)
    np.add.at(contingency, (old['cluster']['labels'], new_labels), 1)
    mapping = math_bundle[2]['exact_matching'](contingency, hashes)
    mapping.update(contingency=contingency, ARI=math_bundle[2]['adjusted_rand'](contingency), new_physical_center_hashes=hashes)
    folder = args.root / 'fitting/ALLTRAIN'
    state_ref = save_state(folder / 'SPACE_AND_CLUSTER.pt', state)
    mapping_ref = save_state(folder / 'CLUSTER_TRANSFER.pt', mapping)
    _FIT_CONTEXT = (state, ordinary, ordinary_view, z, lookup, common, math_bundle, folder, selection, mapping)
    slots = cpu_cluster_map(_refit_slot, args.fit_workers)
    result = {'status': 'COMPLETE_LOCAL_ALLTRAIN_SELECTED_REFIT', 'state': state_ref, 'mapping': mapping_ref,
              'slots': slots, 'only_selected_methods_refitted': True, 'new_local_qualified': len(state['Q_indices']),
              'original10042_selectedTrain_all130sessions': True, 'test_validation_GT_never_fitted': True}
    atomic(folder / 'BANK.json', result)
    return result


def flatten_arrays(value, prefix='', output=None):
    import torch
    output = {} if output is None else output
    if isinstance(value, dict):
        for key, item in value.items():
            flatten_arrays(item, f'{prefix}.{key}'.strip('.'), output)
    elif torch.is_tensor(value):
        output[prefix] = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        output[prefix] = value
    elif isinstance(value, (int, float, bool, np.number)):
        output[prefix] = np.asarray(value)
    return output


def freeze(args):
    plan = read(args.root / 'fitting/PLAN.json'); final = read(args.root / 'fitting/ALLTRAIN/BANK.json')
    state = load_state(final['state']); folder = args.root / 'frozen_teacher'
    if (folder / 'BUNDLE.json').exists():
        bundle = read(folder / 'BUNDLE.json')
        if bundle.get('alltrain_refit') != ref(args.root / 'fitting/ALLTRAIN/BANK.json'):
            raise ValueError('Frozen teacher already exists with a different final fit; never overwrite parameters')
        sys.path.insert(0, str(ROUND1))
        from student_distillation.asset_identity import load_library_bundle
        load_library_bundle(folder / 'BUNDLE.json')
        return {'status': 'FROZEN_LOCAL_TEACHER', 'bundle': str(folder / 'BUNDLE.json'), 'identity': bundle['teacher_identity'], 'reused_existing_frozen_parameters': True}
    folder.mkdir(parents=True, exist_ok=True)
    space = save_npz(folder / 'space.npz', {'basis': state['space']['basis'], 'center': state['space']['center'], 'D0': np.asarray(state['D0'])})
    experts = []
    for slot in final['slots']:
        selected = slot['selection']; payload = load_state(slot['payload'])
        copied = folder / f"slot{slot['old_slot']:02d}.pt"
        shutil.copyfile(slot['payload']['path'], copied)
        arrays = flatten_arrays(payload)
        parameters = save_npz(folder / f"slot{slot['old_slot']:02d}.npz", arrays)
        experts.append({'slot': slot['old_slot'], 'method': selected['method'],
                        'alpha': selected['alpha'], 'cap_D0': selected['cap_D0'],
                        'Base': selected['Base'], 'new_cluster': slot['new_cluster'],
                        'parameters': parameters, 'runtime_state': ref(copied)})
    config = {'schema': 1, 'id': 'local_refit_D_C8_double_20261026', 'K': 8,
              'route': 'D', 'order': 'C_to_P', 'rank': 48, 'view': 'valid_shared384',
              'space': space, 'experts': experts, 'raw_shift_factor': 2., 'cap_factor': 2.,
              'local_refit': True, 'original_13family_menu': plan['menu']}
    atomic(folder / 'library.json', config)
    runtime = folder / 'teacher_library.py'
    shutil.copyfile(Path(__file__).with_name('teacher_library.py'), runtime)
    bundle = {'schema': 'local_refit_teacher_bundle_v1', 'status': 'FROZEN_LOCAL_TEACHER',
              'teacher_origin': 'local_refit', 'fit_population': 'train', 'source_checkpoint_sha256': CHECKPOINT_SHA,
              'teacher_identity': digest({'library': ref(folder / 'library.json'), 'final_refit': ref(args.root / 'fitting/ALLTRAIN/BANK.json')}),
              'library': ref(folder / 'library.json'), 'artifact_root': str(folder),
              'runtime_predictor': dict(ref(runtime), **{'class': 'LocalLibrary'}),
              'source_code_root': plan['source_code_root'], 'original_math_sources': plan['original_math_sources'],
              'math_bindings': plan['math_bindings'], 'new_local_qualified_pairs': final['new_local_qualified'],
              'alltrain_refit': ref(args.root / 'fitting/ALLTRAIN/BANK.json'),
              'inner_selection': ref(args.root / 'fitting/FROZEN_INNER_SELECTION.json'),
              'selections': {}, 'selection_registry': {'path': str(folder / 'SELECTIONS.json')},
              'parameters_frozen_before_full_population_train_test_validation_scoring': True,
              'per_slot_cap_in_d0': [expert['cap_D0'] * 2 for expert in experts],
              'student_cap_in_d0': max(expert['cap_D0'] * 2 for expert in experts)}
    atomic(folder / 'BUNDLE.json', bundle)
    registry = {'schema': 'nrkd_local_selection_registry_v1', 'teacher_identity': bundle['teacher_identity'],
                'source_checkpoint_sha256': CHECKPOINT_SHA, 'populations': {}}
    registry['identity_sha256'] = digest(registry)
    atomic(folder / 'SELECTIONS.json', registry)
    # Load through the actual consumer contract before announcing freeze.
    sys.path.insert(0, str(ROUND1))
    from student_distillation.asset_identity import load_library_bundle
    load_library_bundle(folder / 'BUNDLE.json')
    return {'status': 'FROZEN_LOCAL_TEACHER', 'bundle': str(folder / 'BUNDLE.json'), 'identity': bundle['teacher_identity']}


def queue_population(args):
    sys.path.insert(0, str(ROUND1))
    from student_distillation.data import FeatureStore
    bundle_path = args.root / 'frozen_teacher/BUNDLE.json'; bundle = read(bundle_path)
    store = FeatureStore(args.features, args.population)
    expected = {'train': 85813, 'validation': 17475, 'test': 12146}[args.population]
    if len(store) != expected:
        raise ValueError('Full teacher evaluation requires the entire original population')
    actions = [{'choice': choice, 'action_id': digest({'bundle': bundle['teacher_identity'], 'choice': choice})} for choice in range(9)]
    request = {'schema': 'local_dc8_native_request_v1', 'scope': 'full_population', 'population': args.population,
               'indices': [int(row['global_index']) for row in store.rows], 'tokens': [row['token'] for row in store.rows],
               'actions': actions, 'noise_indices': [0, 1, 2, 3, 4], 'bundle': ref(bundle_path),
               'feature_manifest': ref(args.features), 'score_units': 'fraction', 'fitting_allowed': False,
               'diagnostic_only': args.population != 'train', 'all_experts_cross_applied_without_routing': True}
    folder = args.root / 'requests' / args.population
    atomic(folder / 'REQUEST.json', request)
    # One scene with all nine branches avoids repeating Base/noise/cap work.
    return queue_request(folder, folder / 'REQUEST.json', 9)


def queue_base_qualification(args):
    """Literal Base-only path check, available while GT optimization is running."""
    plan = read(args.root / 'fitting/PLAN.json')
    selected = [next(row for row in plan['panel'] if row['category'] == category) for category in ('failure', 'high')]
    folder = args.root / 'requests/base_qualification'
    compact = save_npz(folder / 'COMPACT.npz', {'indices': np.asarray([row['index'] for row in selected], np.int64),
               'raw_coefficients': np.zeros((2, 1, 1, 48), np.float64), 'natural_weights': np.full(2, .5),
               'basis': np.zeros((384, 48), np.float64), 'D0': np.array(1., np.float64)})
    action = {'cluster': 0, 'method': 'literal_Base_engineering_check', 'method_position': 0,
              'alpha': 0., 'cap_D0': 0., 'valid': False}
    request = {'schema': 'local_dc8_native_request_v1', 'scope': 'base_qualification', 'population': 'train',
               'indices': [row['index'] for row in selected], 'tokens': [row['token'] for row in selected],
               'actions': [dict(action_id=digest(action), **action)], 'noise_indices': [0, 1, 2, 3, 4],
               'compact': compact, 'feature_manifest': plan['feature_manifest'], 'raw_shift_factor': 1., 'cap_factor': 1.,
               'no_operator_or_subspace_fitted': True, 'score_units': 'fraction'}
    atomic(folder / 'REQUEST.json', request)
    return queue_request(folder, folder / 'REQUEST.json', 1)


def collect_population(args):
    folder = args.root / 'requests' / args.population
    request, scores, base = collect_request(folder)
    means = scores[:, :, :, 0].mean(2, dtype=np.float64)
    if means.shape[1] != 9 or not np.array_equal(scores[:, 0], base):
        raise ValueError('Full teacher matrix must put literal Base first')
    q_points = means * np.float64(100.)
    choice = q_points.argmax(1).astype(np.int64)
    actual_shifts = np.empty((len(choice), 2, 384), np.float32)
    actual_rms = np.empty(len(choice), np.float64)
    cap_rms = np.empty(len(choice), np.float64)
    queue = read(folder / 'QUEUE.json')
    seen = np.zeros(len(choice), bool)
    for identity in queue['tasks']:
        receipt = read(folder / 'receipts' / (identity + '.json'))
        result = read(receipt['result']['path'])
        task = result['task']; row = task['row_index']
        position = task['positions'].index(int(choice[row]))
        with np.load(result['arrays']['path'], allow_pickle=False) as arrays:
            actual_shifts[row] = arrays['actual_role_shifts'][position]
            actual_rms[row] = arrays['actual_RMS'][position]
            cap_rms[row] = arrays['cap_RMS'][position]
        seen[row] = True
    if not seen.all() or not np.array_equal(actual_shifts[:, 0], actual_shifts[:, 1]):
        raise ValueError('Selected native actual shared-role shifts are incomplete or unequal')
    output = args.root / 'frozen_teacher/selections' / (args.population + '.npz')
    selection = save_npz(output, {'tokens': np.asarray(request['tokens']), 'indices': np.asarray(request['indices'], np.int64),
                                 'choice': choice, 'q_points': q_points,
                                 'selected_actual_role_shifts': actual_shifts, 'selected_actual_RMS': actual_rms,
                                 'selected_cap_RMS': cap_rms})
    bundle_path = args.root / 'frozen_teacher/BUNDLE.json'; bundle = read(bundle_path)
    registry_path = args.root / 'frozen_teacher/SELECTIONS.json'; registry = read(registry_path)
    entry = dict(selection, units='points', diagnostic_only=args.population != 'train')
    if args.population in registry['populations'] and registry['populations'][args.population] != entry:
        raise ValueError('Existing measured population selections cannot change')
    registry['populations'][args.population] = entry
    registry.pop('identity_sha256', None)
    registry['identity_sha256'] = digest(registry)
    atomic(registry_path, registry)
    selected = q_points[np.arange(len(choice)), choice]
    stats = {'status': 'COMPLETE', 'population': args.population, 'scenes': len(choice), 'noise_indices': [0, 1, 2, 3, 4],
             'selection': selection, 'Base_PDMS_points': float(q_points[:, 0].mean()),
             'full_teacher_PDMS_points': float(selected.mean()), 'choice_counts': np.bincount(choice, minlength=9).tolist(),
             'five_noise_scene_fixed_choice': True, 'Base_first_exact_ties': True,
             'argmax_units': 'FP64_points_after_five_noise_mean_then_times100_Base_first_exact_POINT_ties',
             'parameters_refitted_on_test_or_validation': False, 'diagnostic_only': args.population != 'train'}
    atomic(args.root / 'frozen_teacher' / f'{args.population.upper()}_EVALUATION.json', stats)
    return stats


def selftest(args):
    """Exercise every original method and matching/selection on private arrays."""
    ops, geo, math, binding = bind_math(args.source_root)
    rng = np.random.default_rng(2026102717)
    failed = rng.normal(size=(96, 384)).astype(np.float64)
    success = failed + rng.normal(size=failed.shape) * .03
    rms = np.sqrt(np.square(success - failed).mean(1))
    space = geo.fit_space(failed, success, rms, 'D', 'valid_shared384', 48)
    view = {'view': 'valid_shared384', 'f0': .6}
    geometry = geo.geometry(space['basis'], view, np.full(96, .6), success - failed, rms)
    state = {'space': space, 'view': view, 'D0': float(np.median(rms)), 'Q_indices': np.arange(96),
             'normal_indices': list(range(96, 160)), 'high_indices': list(range(96, 160)),
             'fail_view': failed, 'success_view': success, 'actual_fulltoken_RMS': rms,
             'raw_aux': rng.normal(size=(96, 24)).astype(np.float32), 'fit_identity': 'LOCAL_SYNTHETIC_ALL13',
             'recipe': {'route': 'D', 'view': 'valid_shared384', 'rank': 48}, 'geometry': geometry,
             'cluster': {'labels': np.zeros(96, np.int64)}}
    ordinary_view = np.r_[failed, rng.normal(size=(64, 384))]
    ordinary = {'raw_aux24_float32': rng.normal(size=(160, 24)).astype(np.float32)}
    z = math['coordinates'](ordinary_view, space['center'], space['basis'])
    common = [ops.algebra(z[:96], math['coordinates'](success, space['center'], space['basis']), geometry['coefficients'],
                          geometry['L'], state['D0'], geometry['orders']['P_to_C']['no_direction'])]
    folder = args.root / 'selftests/all13_cpu'
    checked, payloads = [], {}
    for method in ops.MENU:
        payload = fit_payload(method, 0, state, ordinary, ordinary_view, z, {i: i for i in range(160)},
                              common, (ops, geo, math, binding), folder / method / 'MLP2000')
        payload['_ops'] = ops
        payloads[method] = payload
        value = coefficient(payload, ordinary_view[0], ordinary['raw_aux24_float32'][0].astype(np.float64), .6,
                            space['basis'], space['center'], view)
        if value.dtype != np.float64 or value.shape != (48,) or not np.isfinite(value).all():
            raise AssertionError(f'All13 runtime coefficient failed for {method}')
        if method in ('paired_RRR', 'direction_amplitude'):
            try:
                coefficient(payload, ordinary_view[0], None, .6, space['basis'], space['center'], view)
            except ValueError:
                pass
            else:
                raise AssertionError('Paired operator silently accepted missing ordinary aux')
        checked.append(method)
    from teacher_library import LocalLibrary
    import torch
    for first in (0, 8):
        names = [ops.MENU[(first + offset) % len(ops.MENU)] for offset in range(8)]
        library = LocalLibrary()
        library.basis, library.center, library.D0 = space['basis'], space['center'], state['D0']
        library.config = {'K': 8, 'raw_shift_factor': 2., 'cap_factor': 2.,
                          'experts': [{'method': name, 'alpha': 2., 'cap_D0': 1. if slot == 0 else 2.}
                                      for slot, name in enumerate(names)]}
        library.payloads, library.networks = [payloads[name] for name in names], {}
        base_post = torch.from_numpy(rng.normal(size=(7, 384)).astype(np.float32))
        image = torch.tensor([False, True, True, True, False, False, False])
        text = torch.tensor([False, False, False, False, True, True, True])
        result = library.targets(base_post, image, text, list(range(9)), raw_aux24=ordinary['raw_aux24_float32'][0])
        if not torch.equal(result['interfaces'][0], base_post) or not torch.equal(result['interfaces'][:, 0], base_post[0].expand(9, -1)):
            raise AssertionError('General local runtime changed Base or native padding')
        if not torch.equal(result['actual_role_shifts'][:, 0], result['actual_role_shifts'][:, 1]):
            raise AssertionError('General local runtime did not preserve shared valid384 repair')
        if not bool((result['actual_RMS'] <= result['cap_RMS'] + result['tolerance']).all()):
            raise AssertionError('General local runtime violated post-rounding cap')
        if result['cap_RMS'][1].item() != library.D0 * 2 or result['cap_RMS'][2].item() != library.D0 * 4:
            raise AssertionError('General local runtime altered per-expert mixed caps')
    contingency = np.array([[100, 99], [99, 0]], np.int64)
    matching = math['exact_matching'](contingency, ['a', 'z'])
    if matching['old_to_new'] != [1, 0] or matching['maximum_integer_overlap'] != 198:
        raise AssertionError('Original exact integer matching binding failed')
    base = np.array([.5, .5]); weights = np.array([.5, .5])
    actions = [{'cluster': 0, 'method': 'centroid_D0' if i == 0 else str(i), 'alpha': 1, 'cap_D0': 2} for i in range(91)]
    table = np.zeros((2, 91)); table[:, 5] = .8
    pilot = math['optimize'](base, table, weights, actions, 1, 0)
    if pilot['positions'] != [5] or pilot['J'] != .8:
        raise AssertionError('Original coarse optimizer private binding failed')
    formal = math['formal_optimize'](base, table[:, :21], weights, actions[:21], 1)
    if formal['positions'] != [5] or formal['J'] != .8:
        raise AssertionError('Original formal optimizer private binding failed')
    known = {}
    for method in ('mlp', 'mlp_replay'):
        dataset = ops.paired_rows(z[:96], math['coordinates'](success, space['center'], space['basis']),
                                  z[96:] if method == 'mlp_replay' else None)
        seed = math['initialization'](state, 8, method)
        normalized, _, _ = ops.standardize(dataset['x'], dataset['weights'])
        _, target, _ = ops.old.control_teacher(normalized.astype(np.float32), 48, seed, seed + 17, .25 * state['D0'])
        dataset['y'] = target.astype(np.float64)
        mlp_fit.ops = ops
        saved = mlp_fit(dataset, state['D0'], seed, folder / 'known_positive_controls' / method)
        session = ops.MLPTrainingSession(dataset, state['D0'], seed, 'cpu'); session.restore(saved)
        with torch.no_grad():
            predicted = session.model(session.X).cpu().numpy().astype(np.float64) * session.D0
        errors = np.square(predicted - dataset['y']).mean(1)
        baseline = np.square(dataset['y']).mean(1)
        roles = np.asarray(dataset['row_roles'])
        absorptions = {role: 1 - float(dataset['weights'][roles == role] @ errors[roles == role]) / float(dataset['weights'][roles == role] @ baseline[roles == role])
                       for role in dict.fromkeys(dataset['row_roles'])}
        if not all(value >= .9 for value in absorptions.values()):
            raise AssertionError(f'Original MLP known positive control failed: {method} {absorptions}')
        known[method] = absorptions
    result = {'status': 'PASS_PRIVATE_SYNTHETIC_ALL13_ORIGINAL_MATH', 'methods': checked,
              'MLP_fixed_updates_each': 2000, 'MLP_endpoint_replay': True, 'integer_matching': True,
              'original_two_start_two_pass_pilot_and_formal_selection': True, 'real_GT_or_old_teacher_parameters_used': False}
    result['generalized_runtime_all13_native_padding_Base_shared_shift_mixed_caps'] = True
    result['known_realizable_MLP_role_absorptions'] = known
    result['source_bindings'] = source_bindings(args.source_root)
    atomic(folder / 'COMPLETE.json', result)
    return result


def status(args):
    states = []
    for path in sorted((args.root / 'requests').glob('*/QUEUE.json')):
        queue = read(path)
        folder = path.parent
        complete = sum((folder / 'receipts' / (value + '.json')).exists() for value in queue['tasks'])
        failures = sum(1 for _ in (folder / 'failures').glob('*/*.json'))
        states.append({'queue': str(path), 'scenes': queue['scene_count'], 'total_tasks': queue['total_tasks'],
                       'completed_tasks': complete, 'pending_tasks': queue['total_tasks'] - complete,
                       'recorded_failed_attempts': failures, 'complete_matrix': (folder / 'COMPLETE.json').exists()})
    return {'time': stamp(), 'GT_exports_completed': len(list((args.root / 'native_gt/exports/post').glob('*/COMPLETE.json'))),
            'GT_exports_required': 1850, 'queues': states,
            'teacher_parameters_frozen': (args.root / 'frozen_teacher/BUNDLE.json').exists()}


def emit_validation_diagnostics(args):
    if args.targets_manifest is None:
        raise ValueError('Validation diagnostics require the completed new validation targets manifest')
    bundle = args.root / 'frozen_teacher/BUNDLE.json'
    summary_path = args.root / 'frozen_teacher/VALIDATION_EVALUATION.json'
    summary = read(summary_path)
    if summary.get('status') != 'COMPLETE' or summary.get('population') != 'validation' or summary.get('scenes') != 17475 or summary.get('noise_indices') != [0, 1, 2, 3, 4]:
        raise ValueError('Full validation teacher scoring is incomplete')
    targets = read(args.targets_manifest)
    spec = targets['populations']['validation']
    if targets.get('schema') != 'nrkd_compact_targets_v1' or targets.get('teacher_origin') != 'local_refit' or spec.get('count') != 17475 or not spec.get('diagnostic_only') or set(spec['versions']) != {'full', 'gain1'}:
        raise ValueError('Validation diagnostic full/gain1 targets are incomplete')
    if spec['identity']['teacher']['bundle_sha256'] != sha(bundle):
        raise ValueError('Validation diagnostic targets belong to another frozen teacher')
    complete_path = args.root / 'requests/validation/COMPLETE.json'
    complete = read(complete_path)
    if complete['scenes'] != 17475 or complete['actions'] != 9 or complete['noise_indices'] != [0, 1, 2, 3, 4]:
        raise ValueError('Full validation nine-branch matrix coverage is incomplete')
    artifacts = {'teacher_bundle': ref(bundle), 'full_validation_teacher_summary': ref(summary_path),
                 'full_validation_candidate_matrix': complete['matrix'], 'validation_selection': summary['selection'],
                 'target_manifest': ref(args.targets_manifest), 'target_index': spec['index'],
                 'target_full_shifts': spec['versions']['full']['shifts'],
                 'target_gain1_shifts': spec['versions']['gain1']['shifts']}
    for name, artifact in artifacts.items():
        if sha(artifact['path']) != artifact['sha256']:
            raise ValueError(f'Validation diagnostic artifact SHA mismatch: {name}')
    value = {'schema': 'nrkd_validation_teacher_diagnostics_v1', 'status': 'COMPLETE',
             'population': 'validation', 'scenes': 17475, 'noise_indices': [0, 1, 2, 3, 4],
             'teacher_bundle_sha256': sha(bundle), 'versions': ['full', 'gain1'], 'artifacts': artifacts,
             'posthoc_diagnostic_teacher': True, 'training_or_calibration_feedback': False,
             'original_threshold_gain_points_strict_gt': 1.0}
    destination = args.diagnostics_out or Path('/mnt/project/recogdrive-no-routing-round1-results-20261026/VAL_TEACHER_DIAGNOSTICS.json')
    atomic(destination, value)
    return value


def emit_teacher_report(args):
    folder = args.root / 'frozen_teacher'
    bundle = read(folder / 'BUNDLE.json'); library = read(bundle['library']['path'])
    lines = ['# New local frozen D_C8 teacher', '',
             'The subspace, eight expert slots, calibration choices and full-population scene choices were rebuilt from local native inputs. No historical expert weights, D48, or scene choices were transplanted.', '',
             f"New locally qualified GT-post pairs: {bundle['new_local_qualified_pairs']}. The criterion is local Base S0 <70 and selected GT-post S0 >90; 1,807 is not imposed as a constraint.", '',
             '| Population | Scenes | Base PDMS | Teacher PDMS | Gain |', '|---|---:|---:|---:|---:|']
    summaries = {}
    for population in ('train', 'test', 'validation'):
        path = folder / f'{population.upper()}_EVALUATION.json'
        if not path.exists():
            continue
        summary = read(path); summaries[population] = summary
        lines.append(f"| {population} | {summary['scenes']} | {summary['Base_PDMS_points']:.7f} | {summary['full_teacher_PDMS_points']:.7f} | {summary['full_teacher_PDMS_points'] - summary['Base_PDMS_points']:+.7f} |")
    if 'train' not in summaries or 'test' not in summaries:
        raise ValueError('The local teacher report requires complete Train and Test measurements')
    lines += ['', 'Every scene has one fixed Base/expert choice across S0–S4, selected after averaging those five scores. These are post-hoc oracle teacher measurements, not a deployable router or student performance. Test and validation never fit the subspace, experts or dose choices.', '',
              '| Slot | Newly selected method | Alpha | Cap/D0 before double |', '|---:|---|---:|---:|']
    for expert in library['experts']:
        lines.append(f"| {expert['slot']} | {expert['method']} | {expert['alpha']} | {expert['cap_D0']} |")
    lines += ['', 'The whole-library raw-shift and cap factors are both 2. The continuous student bound is the maximum frozen per-slot cap, with no input-dependent cap selection.', '',
              'All candidate trajectories, seven native PDM components and actual FP32 shifts are retained under requests/<population>/results/. Exact full padded Base interfaces remain in the immutable feature mmap; each result pins its row/shard/offset/length and verifies its H0 bytes SHA. Selection matrices and target manifests are checksum identified. GPU actor wall time, failures and retries are recorded by the durable stage owner; no unmeasured cost is presented as a measured value.', '']
    (folder / 'LOCAL_TEACHER_REPORT.md').write_text('\n'.join(lines))
    atomic(folder / 'LOCAL_TEACHER_RESULTS.json', {'status': 'COMPLETE_TRAIN_TEST_TEACHER_MEASUREMENT',
            'teacher_bundle_sha256': sha(folder / 'BUNDLE.json'), 'populations': summaries,
            'report': ref(folder / 'LOCAL_TEACHER_REPORT.md')})
    return {'report': str(folder / 'LOCAL_TEACHER_REPORT.md'), 'populations': list(summaries)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'fit-inner', 'pack-pilot', 'collect-pilot', 'pack-cal', 'collect-cal',
                                         'refit', 'freeze', 'queue-population', 'collect-population', 'queue-base-qualification',
                                         'emit-validation-diagnostics', 'emit-teacher-report', 'selftest', 'status'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--gt-root', type=Path)
    parser.add_argument('--features', type=Path, default=FEATURES)
    parser.add_argument('--base-pool', type=Path, default=BASE_POOL)
    parser.add_argument('--population', choices=('train', 'validation', 'test'), default='train')
    parser.add_argument('--action-chunk', type=int, default=8)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--fit-workers', type=int, choices=range(1, 9), default=1)
    parser.add_argument('--targets-manifest', type=Path)
    parser.add_argument('--diagnostics-out', type=Path)
    args = parser.parse_args()
    args.root, args.source_root = args.root.resolve(), args.source_root.resolve()
    args.gt_root = (args.gt_root or args.root / 'native_gt').resolve()
    sys.path.insert(0, str(ROUND1))
    import torch
    torch.set_num_threads(args.threads)
    actions = {'prepare': prepare, 'fit-inner': fit_inner, 'pack-pilot': lambda value: pack(value, 'pilot'),
               'collect-pilot': lambda value: collect(value, 'pilot'), 'pack-cal': lambda value: pack(value, 'cal'),
               'collect-cal': lambda value: collect(value, 'cal'), 'refit': refit, 'freeze': freeze,
               'queue-population': queue_population, 'collect-population': collect_population,
               'queue-base-qualification': queue_base_qualification, 'emit-validation-diagnostics': emit_validation_diagnostics,
               'emit-teacher-report': emit_teacher_report, 'selftest': selftest, 'status': status}
    print(json.dumps(actions[args.action](args), allow_nan=False, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value)), flush=True)


if __name__ == '__main__':
    main()
