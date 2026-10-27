"""Local transport for unchanged archived GT optimization function code.

No archived output, release, teacher tensor, or score table is used as a result.
The only changed pieces are local input loading, explicit-noise transport,
atomic persistence, and the new cumulative summary/controller.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import datetime as dt
import hashlib
import io
import inspect
import json
import marshal
import os
from pathlib import Path
import sys
import time
from types import FunctionType, ModuleType
import uuid
import zipfile

import numpy as np

FIELDS = ['score', 'no_at_fault_collisions', 'drivable_area_compliance',
          'time_to_collision_within_bound', 'ego_progress', 'comfort',
          'driving_direction_compliance']
STAGES = ['baseline', 'strength_A', 'strength_B', 'strength_C', 'pre_lift',
          'pre_A', 'pre_B', 'pre_C', 'pre_progress', 'pre_hard_gt',
          'post_A', 'post_B', 'post_C', 'post_from_pre', 'post_progress']
G5 = '1022-gt_assisted_representation_optimization/code'
R2 = '1018-representation_repair_batch2/code'
R3 = '1021-representation_repair_batch3/code'


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def seed(token, j=0):
    return int(hashlib.sha256(f'{token}|20261008|{j}'.encode()).hexdigest()[:8], 16)


def arrays_at(root, path, values):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('GT numeric outputs must remain below the new root')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(values) or any(not np.array_equal(saved[k], np.asarray(v)) for k, v in values.items()):
                raise ValueError(f'Existing numeric output differs: {path}')
    else:
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for key, value in values.items():
                stream = io.BytesIO()
                np.lib.format.write_array(stream, np.asarray(value), allow_pickle=False)
                archive.writestr(key + '.npy', stream.getvalue())
        temporary.replace(path)
    return {'path': str(path.relative_to(root)), 'sha256': sha(path)}


def bind_function(path, name, namespace):
    """Compile only the original function, then rebind its unchanged code."""
    path = Path(path)
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    scope = dict(namespace, __file__=str(path), __name__='local_gt_bound_original')
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)
    original = scope[name]
    bound = FunctionType(original.__code__, scope, original.__name__, original.__defaults__, original.__closure__)
    scope[name] = bound
    identity = {'file': str(path), 'file_sha256': sha(path), 'function': name,
                'function_code_sha256': hashlib.sha256(marshal.dumps(bound.__code__)).hexdigest(),
                'binding': 'unchanged __code__ / private FunctionType globals'}
    return bound, identity


@contextmanager
def import_shims(values):
    previous = {name: sys.modules.get(name) for name in values}
    try:
        for name, members in values.items():
            module = ModuleType(name)
            module.__dict__.update(members)
            sys.modules[name] = module
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class ReferenceNoise:
    def __init__(self, protocol, token_index, bank=None):
        protocol = Path(protocol).resolve()
        metadata = read(protocol)
        entry = next(item for item in metadata['files'] if item['cohort'] == 'navtrain')
        bank = Path(bank).resolve() if bank else protocol.parent.parent / entry['file']
        if sha(bank) != entry['sha256']:
            raise ValueError('Explicit reference noise SHA mismatch')
        self.bank = np.load(bank, mmap_mode='r', allow_pickle=False)
        if list(self.bank.shape) != entry['shape'] or self.bank.dtype != np.float32 or self.bank.shape[1:] != (5, 6, 1, 8, 3):
            raise ValueError('Reference bank must preserve all five native six-draw tapes')
        rows = read(token_index)
        if not isinstance(rows, list) or len(rows) != len(self.bank):
            raise ValueError('Reference noise token index must be the verified ordered population')
        self.positions = {row['token']: position for position, row in enumerate(rows)}
        if len(self.positions) != len(rows):
            raise ValueError('Duplicate reference noise token')
        self.identity = {'protocol': str(protocol), 'protocol_sha256': sha(protocol),
                         'bank': str(bank), 'bank_sha256': entry['sha256'],
                         'token_index': str(Path(token_index).resolve()), 'token_index_sha256': sha(token_index),
                         'explicit_initial_then_five_step_noise': True, 'seed_only_sampling_forbidden': True}

    def tape(self, token, j):
        if type(j) is not int or not 0 <= j < 5:
            raise ValueError('Reference noise index must be int 0..4')
        tape = np.asarray(self.bank[self.positions[token], j])
        if not np.isfinite(tape).all():
            raise ValueError('Nonfinite reference noise')
        return tape


class LocalGTNumeric:
    """One process owns one frozen head, one original optimizer, and one tape store."""
    def __init__(self, root, source_root, frozen_repo, paths, base_root, noise, *, agent=None, config=None, runtime=None):
        import torch
        self.root, self.sources = Path(root).resolve(), Path(source_root).resolve()
        self.base_root, self.noise = Path(base_root).resolve(), noise
        self.frozen_repo = Path(frozen_repo).resolve()
        if str(self.frozen_repo) not in sys.path:
            sys.path.insert(0, str(self.frozen_repo))
        if agent is None:
            from interface_teacher.runtime import bootstrap
            bootstrap(paths, self.root / 'runtime')
            from worker import load_model
            from portable_common import read as read_config
            config = read_config(self.frozen_repo / 'configs/repair_final_baseline.json')
            agent, runtime = load_model(config)
        self.agent = agent.eval().requires_grad_(False)
        self.cfg, self.planner_runtime = config, dict(runtime or {})
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(1)
        self.planner_runtime['tf32'] = {'matmul': False, 'cudnn': False}
        if torch.__version__.split('+')[0] != '2.5.1' or torch.backends.cudnn.version() != 90100:
            raise ValueError('Local GT qualification requires the declared Torch2.5.1/cuDNN90100 environment')
        if any(parameter.dtype != torch.float32 for parameter in self.agent.action_head.parameters()):
            raise ValueError('The native frozen action head must be FP32')
        head_source = Path(inspect.getfile(type(self.agent.action_head))).resolve()
        self.planner_runtime['local_native_numerics'] = {
            'torch': torch.__version__, 'cuda_build': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(), 'threads': torch.get_num_threads(),
            'action_head_dtype': 'float32', 'autocast': False,
            'head_source': str(head_source), 'head_source_sha256': sha(head_source),
            'load_model_source_sha256': sha(self.frozen_repo / 'vendor/worker.py'),
            'local_numeric_transport_sha256': sha(__file__),
            'tf32': {'matmul': False, 'cudnn': False},
            'timesteps': self.agent.action_head.ddim_t.detach().cpu().tolist(),
            'std_floor': float(getattr(self.agent.action_head, 'eval_min_sampling_denoising_std', 1e-4)),
            'noise_clip': float(getattr(self.agent.action_head, 'eval_randn_clip_value', 1.0))}
        self.parameter_versions = {name: value._version for name, value in self.agent.named_parameters()}
        common = {'ROOT': self.root, 'B2': self.root, 'Path': Path, 'read': read,
                  'sha': sha, 'atomic': self._atomic, 'arrays': self._arrays,
                  'stamp': stamp, 'seed': seed, 'time': time, 'FIELDS': FIELDS}
        self.bindings = []
        def bind(relative, name, extra=None):
            function, identity = bind_function(self.sources / relative, name, dict(common, **(extra or {})))
            self.bindings.append(identity)
            return function
        self.original_edited_value = bind(R2 + '/r2_counterfactual.py', 'edited_value')
        self.original_predict_gradient = bind(R2 + '/r2_counterfactual.py', 'predict_gradient')
        self.original_prediction = bind(R3 + '/r3_runtime.py', 'prediction')
        self.augment = bind(G5 + '/g5_targets.py', 'augment')
        self.changed = bind(G5 + '/g5_runtime.py', 'changed')
        self.scoring = bind(G5 + '/g5_runtime.py', 'scoring')
        self.replay = bind(G5 + '/g5_runtime.py', 'replay', {'changed': self.changed, 'scoring': self.scoring})
        self.optimize = bind(G5 + '/g5_runtime.py', 'optimize',
                             {'changed': self.changed, 'scoring': self.scoring, 'replay': self.replay})
        self.configurations = bind(G5 + '/g5_runtime.py', 'configurations')
        self.current_data = None
        self.identity = {'status': 'BOUND_LOCAL_ORIGINAL_GT_NUMERICS', 'bindings': self.bindings,
                         'reference_noise': noise.identity, 'checkpoint': self.planner_runtime,
                         'source_outputs_readonly': True, 'teacher_targets_imported': False,
                         'scope': 'locally screened Train failure IDs only; no Validation/Test optimization',
                         'candidate_count': len(read(self.root / 'manifests/SCENES.json')),
                         'local_candidates_sha256': read(self.root / 'PROTOCOL.json')['local_candidates']['sha256']}

    def _atomic(self, path, value):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Original code attempted a write outside the new GT root')
        atomic(path, value)

    def _arrays(self, path, values):
        return arrays_at(self.root, path, values)

    @contextmanager
    def shims(self):
        with import_shims({'r2_counterfactual': {'edited_value': self.original_edited_value,
                                                 'predict_gradient': self.predict_gradient},
                           'r3_runtime': {'prediction': self.prediction},
                           'g5_targets': {'augment': self.augment}}):
            yield

    @contextmanager
    def replay_noise(self, token, j):
        from noise_replay import replay_bank
        with replay_bank(self.noise.tape(token, j)):
            yield

    def predict_gradient(self, engine, features, hidden, interface, value):
        token = self.current_data['row']['token']
        if value != seed(token):
            raise ValueError('GT gradients may only use S0')
        with self.replay_noise(token, 0):
            return self.original_predict_gradient(engine, features, hidden, interface, value)

    def prediction(self, engine, features, hidden, j, token, interface=None, gradient=False, capture=True):
        with self.replay_noise(token, j):
            return self.original_prediction(engine, features, hidden, j, token, interface,
                                             gradient=gradient, capture=capture)

    def trajectory(self, features, hidden, value, interface=None):
        token = self.current_data['row']['token']
        choices = [j for j in range(5) if seed(token, j) == value]
        if len(choices) != 1:
            raise ValueError('Trajectory request does not match one of the five reference noises')
        return self.prediction(self, features, hidden, choices[0], token, interface, capture=False)[0]

    def mask(self, block, group):
        masks = {'valid_mean': self.valid_mask, 'image_mean': self.image_mask, 'text_mean': self.text_mask}
        value = masks[group].reshape(-1)
        return value, bool(value.any())

    def suffix(self, block, value):
        if block == 'interface_pre':
            return value, None
        if block == 'interface_post':
            return self.prefix['interface_pre'], value
        raise ValueError(block)

    def verify_engine(self):
        if self.agent.training or any(value.requires_grad or value.grad is not None or value._version != self.parameter_versions[name]
                                       for name, value in self.agent.named_parameters()):
            raise ValueError('Frozen native Base was modified')

    def prepare(self, row):
        import torch
        began = time.monotonic()
        if row.get('population', 'train') != 'train' or row['subset'] != 'exploration':
            raise ValueError('GT optimizer must never read Validation/Test')
        folder = self.base_root / 'train/cases' / row['token']
        complete = read(folder / 'COMPLETE.json')
        path = folder / 'BASE_INPUTS.npz'
        if complete['token'] != row['token'] or int(complete['global_index']) != int(row['global_index']) or complete['population'] != 'train':
            raise ValueError('Fresh local Base identity mismatch')
        if sha(path) != complete['arrays_sha256']:
            raise ValueError('Fresh local Base arrays SHA mismatch')
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {name: np.array(loaded[name]) for name in loaded.files}
        required = ['hidden_FP32', 'Base_interface', 'image_mask', 'text_mask', 'attention_mask',
                    'token_ids', 'position_ids', 'history_trajectory', 'status_feature',
                    'high_command_one_hot', 'GT', 'Base_trajectories', 'Base_quality']
        if any(name not in arrays for name in required):
            raise ValueError('Incomplete fresh Base numeric inputs')
        hidden, post = arrays['hidden_FP32'], arrays['Base_interface']
        if hidden.dtype != np.float32 or post.dtype != np.float32 or hidden.ndim != 2 or hidden.shape[1] != 1536 or post.shape != (len(hidden), 384):
            raise ValueError('Native hidden/post shape or dtype changed')
        image, text = arrays['image_mask'].astype(bool), arrays['text_mask'].astype(bool)
        if image.shape != (len(hidden),) or text.shape != image.shape or (image & text).any() or not np.array_equal(image | text, arrays['attention_mask'].astype(bool)):
            raise ValueError('Native token mask mismatch')
        if not image.any() or not text.any() or not all(np.isfinite(arrays[k]).all() for k in required):
            raise ValueError('Invalid finite/native input groups')
        tensor = lambda value: torch.as_tensor(value, device='cuda')
        self.prefix = {'interface_pre': tensor(hidden).unsqueeze(0),
                       'interface_post': tensor(post).unsqueeze(0)}
        self.image_mask, self.text_mask = tensor(image), tensor(text)
        self.valid_mask = self.image_mask | self.text_mask
        self.inputs = {name: tensor(arrays[key]).unsqueeze(0) for name, key in
                       [('input_ids', 'token_ids'), ('attention_mask', 'attention_mask'), ('position_ids', 'position_ids')]}
        features = {name: tensor(arrays[name]).float() for name in
                    ['history_trajectory', 'status_feature', 'high_command_one_hot']}
        features['last_hidden_state'] = self.prefix['interface_pre'][0]
        quality = arrays['Base_quality']
        if quality.shape != (5, 7) or arrays['Base_trajectories'].shape != (5, 8, 3):
            raise ValueError('Fresh Base five-noise records malformed')
        expected_metric_sha = complete.get('metric_cache_sha256', row.get('metric_sha256'))
        if not expected_metric_sha or sha(row['metric_cache']) != expected_metric_sha:
            raise ValueError('Actual local metric file does not match its verified registry/Base identity')
        if row.get('metric_sha256') and row['metric_sha256'] != expected_metric_sha:
            raise ValueError('Base and controller local metric identities disagree')
        reference = {'trajectory': arrays['Base_trajectories'][0],
                     'score': dict(zip(FIELDS, quality[0].tolist())),
                     'metric_sha256': expected_metric_sha}
        data = {'row': row, 'features': features, 'target': tensor(arrays['GT']).float(),
                'identity': complete, 'hidden': self.prefix['interface_pre'],
                'reference': reference, 'read_seconds': time.monotonic() - began,
                'base_arrays': arrays, 'base_arrays_sha256': complete['arrays_sha256']}
        self.current_data = data
        with torch.no_grad():
            actual_post = self.agent.action_head.feature_encoder(data['hidden'])
            if not torch.equal(actual_post, self.prefix['interface_post']):
                raise ValueError('Fresh Base post differs bytewise from the native local projection')
            data['native'] = self.trajectory(features, data['hidden'], seed(row['token']))
        if not np.array_equal(data['native'].cpu().numpy(), reference['trajectory']):
            raise ValueError('Reference-noise Base replay differs from newly verified local input')
        if tuple(data['target'].shape) != (8, 3):
            raise ValueError('GT must preserve the original [8,3] trajectory')
        return data

    def run_config(self, row, config, pool, destination):
        data = self.prepare(row)
        with self.shims():
            result = self.optimize(self, data, pool, config, Path(destination), expected=None)
        result['local_input_sha256'] = data['base_arrays_sha256']
        result['numerics_binding_sha256'] = digest(self.identity)
        result['reference_noise'] = self.noise.identity
        self._atomic(Path(destination) / 'RESULT.json', result)
        return result

    def export_selected(self, row, selection, pool, destination):
        """Capture complete H0/selected post384 from the same executed rollout."""
        data = self.prepare(row)
        if sha(selection['arrays']) != selection['arrays_sha256']:
            raise ValueError('Selected perturbation SHA mismatch')
        with np.load(selection['arrays'], allow_pickle=False) as saved:
            delta = np.array(saved['delta'])
        with self.shims():
            result = self.replay(self, data, pool, delta, selection['config'], expected=None, capture=True)
        if result['selected'][0]['score']['score'] != selection['S0_score']:
            raise ValueError('Selected branch export score differs from its actual S0 selection')
        source = read(selection['source'])
        if sha(selection['source']) != selection['source_sha256']:
            raise ValueError('Selected branch source SHA mismatch')
        expected = source['records'][source['selected_index']]
        if result['selected'][0]['trajectory'] != expected['trajectory'] or result['selected'][0]['score'] != expected['score']:
            raise ValueError('Selected branch export differs from its original native S0 record')
        base_trajectories = np.asarray([r['trajectory'] for r in result['baselines']], dtype=np.float32)
        base_quality = np.asarray([[r['score'][f] for f in FIELDS] for r in result['baselines']], dtype=np.float64)
        if not np.array_equal(base_trajectories, data['base_arrays']['Base_trajectories'][:3]):
            raise ValueError('Same-execution Base S0-S2 trajectories differ from the verified local Base cache')
        if not np.array_equal(base_quality, data['base_arrays']['Base_quality'][:3]):
            raise ValueError('Same-execution Base S0-S2 quality differs from the verified local Base cache')
        state24 = np.concatenate([data['base_arrays']['history_trajectory'].reshape(12),
                                  data['base_arrays']['status_feature'],
                                  data['base_arrays']['high_command_one_hot']]).astype(np.float32)
        payload = {'Base_interface': data['base_arrays']['Base_interface'], 'teacher_interface': result['interface'],
                   'image_mask': data['base_arrays']['image_mask'], 'text_mask': data['base_arrays']['text_mask'],
                   'selected_delta': delta, 'trajectories': np.asarray([r['trajectory'] for r in result['selected']], dtype=np.float32),
                   'quality': np.asarray([[r['score'][f] for f in FIELDS] for r in result['selected']], dtype=np.float64),
                   'GT': data['base_arrays']['GT'], 'Base_trajectories': base_trajectories,
                   'Base_quality': base_quality, 'state24': state24}
        for key in ['token_ids', 'attention_mask', 'position_ids', 'history_trajectory',
                    'status_feature', 'high_command_one_hot']:
            payload[key] = data['base_arrays'][key]
        numeric = self._arrays(Path(destination) / 'TEACHER.npz', payload)
        value = {'status': 'PASS_LOCAL_FROZEN_BASE_GT_POST_PAIR', 'token': row['token'],
                 'global_index': row['global_index'], 'population': 'train', 'subset': 'exploration',
                 'selection': selection, 'arrays': numeric, 'selected': result['selected'],
                 'baselines': result['baselines'], 'base_input_sha256': data['base_arrays_sha256'],
                 'same_execution_interface_and_trajectory': True, 'reference_noise': self.noise.identity,
                 'Base_pairing': 'Base S0/S1/S2 trajectories and quality rerun in this same export; exact against current local cache',
                 'noise_indices': [0, 1, 2], 'Base_noise_indices': [0, 1, 2],
                 'state24_order': ['history_trajectory.reshape(12)', 'status_feature[8]', 'high_command_one_hot[4]'],
                 'numerics_binding_sha256': digest(self.identity), 'test_used': False}
        self._atomic(Path(destination) / 'COMPLETE.json', value)
        return value


def descriptor(root, path):
    root, path = Path(root), Path(path)
    value = read(path)
    numeric = value['arrays']
    array_path = root / numeric['path']
    if sha(array_path) != numeric['sha256']:
        raise ValueError('GT descriptor array SHA mismatch')
    return {'source': str(path.resolve()), 'source_sha256': sha(path),
            'arrays': str(array_path.resolve()), 'arrays_sha256': numeric['sha256'],
            'config': value['config'], 'S0_score': value['records'][value['selected_index']]['score']['score'],
            'selected_index': value['selected_index'], 'transfer': value['transfer']}
