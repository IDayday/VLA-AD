"""Durable native per-scene candidate scoring; main agent owns all GPU dispatch."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
import numpy as np
from gt_numeric import atomic, digest, read, sha, stamp, FIELDS
from fit_library import save_npz, ref, validate_task
from teacher_library import LocalLibrary

ROOT = Path(__file__).resolve().parents[1]
ROUND1 = Path('/mnt/project/recogdrive-no-routing-round1-20261026')


class ExplicitNoise:
    """Validated reference tape bank, with no generation or seed-only fallback."""
    def __init__(self, protocol, index, bank, population):
        protocol = Path(protocol).resolve()
        metadata = read(protocol)
        cohort = 'navtest' if population == 'test' else 'navtrain'
        candidates = [entry for entry in metadata.get('files', []) if entry.get('cohort') == cohort]
        if not candidates:
            raise ValueError(f'No explicit reference noise protocol for {cohort}')
        if bank is None:
            bank = protocol.parent.parent / candidates[0]['file']
        bank = Path(bank).resolve()
        checksum = sha(bank)
        entry = next((entry for entry in candidates if entry.get('sha256') == checksum), None)
        if entry is None:
            raise ValueError('Explicit reference noise bank SHA does not match declared protocol')
        self.array = np.load(bank, mmap_mode='r', allow_pickle=False)
        if self.array.dtype != np.float32 or self.array.ndim != 6 or self.array.shape[2:] != (6, 1, 8, 3) or self.array.shape[1] < 5:
            raise ValueError('Explicit native noise must have shape [N,>=5,6,1,8,3] FP32')
        if entry.get('shape') is not None and list(self.array.shape) != entry['shape']:
            raise ValueError('Explicit reference bank shape differs from its protocol')
        rows = read(index)
        if isinstance(rows, dict):
            rows = rows.get('rows', rows.get('scenes'))
        if len(rows) != len(self.array):
            raise ValueError('Explicit noise token index length mismatch')
        self.position = {row['token']: index for index, row in enumerate(rows)}
        if len(self.position) != len(rows):
            raise ValueError('Duplicate token in explicit noise index')
        self.identity = {'protocol': ref(protocol), 'bank': ref(bank), 'index': ref(index),
                         'shape': list(self.array.shape), 'seed_only_sampling_forbidden': True}

    def tape(self, token, noise, device):
        import torch
        value = np.array(self.array[self.position[token], noise], copy=True)
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite explicit native noise tape')
        return torch.from_numpy(value).to(device)


def claim(folder, worker, max_attempts):
    queue = read(folder / 'QUEUE.json')
    ids = sorted(queue['tasks'], key=lambda value: digest([worker, value]))
    for identity in ids:
        marker = folder / 'receipts' / (identity + '.json')
        if marker.exists() or len(list((folder / 'failures' / identity).glob('*.json'))) >= max_attempts:
            continue
        path = folder / 'claims' / (identity + '.lock')
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open('a+')
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close(); continue
        if marker.exists():
            stream.close(); continue
        task = read(folder / 'tasks' / (identity + '.json'))
        stream.seek(0); stream.truncate()
        json.dump({'worker': worker, 'pid': os.getpid(), 'host': socket.gethostname(), 'time': stamp()}, stream)
        stream.flush(); os.fsync(stream.fileno())
        return task, stream
    return None, None


def run(args):
    import torch
    sys.path.insert(0, str(args.round1_repo))
    from student_distillation.data import FeatureStore
    from student_distillation.train import bridge_factory
    from interface_teacher.bank import pool_base
    from interface_teacher.native_cap import apply
    folder = args.queue.parent if args.queue.name == 'QUEUE.json' else args.queue
    queue = read(folder / 'QUEUE.json')
    if sha(queue['request']['path']) != queue['request']['sha256']:
        raise ValueError('Queue request identity changed')
    request = read(queue['request']['path'])
    store = FeatureStore(request['feature_manifest']['path'], request['population'])
    noise = ExplicitNoise(args.noise_protocol, args.noise_index, args.noise_bank, request['population'])
    if torch.__version__.split('+')[0] != '2.5.1' or torch.backends.cudnn.version() != 90100:
        raise ValueError('Native teacher scoring requires original local Torch2.5.1/cuDNN90100')
    torch.set_num_threads(1)
    device = f'cuda:{int(os.environ.get("LOCAL_RANK", "0"))}'
    worker = args.worker_id or f'{socket.gethostname()}_gpu{os.environ.get("CUDA_VISIBLE_DEVICES", "0")}_pid{os.getpid()}'
    worker_folder = folder / 'workers' / worker
    worker_folder.mkdir(parents=True, exist_ok=True)
    paths = read(args.paths)
    paths.update(paths_json=str(args.paths.resolve()), results_root=str(worker_folder / 'runtime'))
    bridge = bridge_factory(paths, device)
    if args.inference_graph_receipt:
        from student_distillation.inference_graph import qualified_inference_bridge
        bridge = qualified_inference_bridge(bridge, args.inference_graph_receipt,
                                            feature_manifest=request['feature_manifest']['path'])
    from repair_score import score_group
    scoring = json.loads((args.round1_repo / 'configs/repair_final_baseline.json').read_text())['scoring']
    identity = {'request': queue['request'], 'noise': noise.identity, 'worker_source': ref(__file__),
                'bridge_source': ref(args.round1_repo / 'student_distillation/native_bridge.py'),
                'bridge_factory_source': ref(args.round1_repo / 'student_distillation/train.py'),
                'native_cap_source': ref(args.round1_repo / 'interface_teacher/native_cap.py'),
                'paths_identity': ref(args.paths),
                'checkpoint_sha256': '23fb949b10387efeff87e093703d5faf41b61b9e5ead16d4076661e3c7c08a20',
                'head_dtype': 'float32', 'torch': torch.__version__, 'cudnn': torch.backends.cudnn.version(),
                'tf32': {'matmul': False, 'cudnn': False}, 'scoring': scoring}
    identity['inference_graph_qualification'] = ref(args.inference_graph_receipt) if args.inference_graph_receipt else None
    atomic(worker_folder / 'IDENTITY.json', identity)
    compact = None; library = None
    if request['scope'] == 'full_population':
        if sha(request['bundle']['path']) != request['bundle']['sha256']:
            raise ValueError('Frozen teacher bundle identity changed')
        library = LocalLibrary.from_bundle(request['bundle']['path'])
    else:
        if sha(request['compact']['path']) != request['compact']['sha256']:
            raise ValueError('Candidate coefficient matrix SHA changed')
        compact = dict(np.load(request['compact']['path'], allow_pickle=False))
    from student_distillation.checkpointing import rng_state, restore_rng
    completed = failed = 0
    start = time.monotonic()
    while True:
        if args.deadline_unix and time.time() >= args.deadline_unix:
            break
        if (args.root / 'STOP_NEW_CLAIMS.json').exists() or (folder / 'STOP_NEW_CLAIMS.json').exists():
            break
        task, lock = claim(folder, worker, args.max_attempts)
        if task is None:
            break
        began = time.monotonic()
        try:
            validate_task(task, task['id'], queue['request'], request)
            token = request['tokens'][task['row_index']]
            feature_index = store.by_token[token]
            meta = store.rows[feature_index]
            if int(meta['global_index']) != int(task['index']):
                raise ValueError('Native feature/global index mismatch')
            arrays = store.read(feature_index)
            base = torch.from_numpy(np.array(arrays['base_post'], copy=True)).unsqueeze(0).to(device)
            image = torch.from_numpy(np.array(arrays['image_mask'], copy=True)).unsqueeze(0).to(device)
            text = torch.from_numpy(np.array(arrays['text_mask'], copy=True)).unsqueeze(0).to(device)
            state = np.array(arrays['state24'], copy=True)
            context = bridge.encode_context(torch.from_numpy(state[:12]).unsqueeze(0).to(device),
                                            torch.from_numpy(state[12:20]).unsqueeze(0).to(device))
            actions = [request['actions'][position] for position in task['positions']]
            if library is not None:
                target = library.targets(base, image, text, [action['choice'] for action in actions], raw_aux24=state)
            else:
                raw, caps = [], []
                for action in actions:
                    if not action['valid']:
                        physical = np.zeros((2, 384), np.float64)
                    else:
                        q = compact['raw_coefficients'][task['row_index'], action['cluster'], action['method_position']]
                        decoded = q[None] @ compact['basis'].T
                        physical = (np.stack([decoded, decoded], axis=-2) * np.float64(action['alpha']))[0]
                        physical = physical * np.float64(request['raw_shift_factor'])
                    raw.append(physical)
                    caps.append(float(action['cap_D0']) * float(compact['D0']) * float(request['cap_factor']))
                raw = np.asarray(raw, np.float64)
                target = apply(base, image, text, torch.from_numpy(raw.astype(np.float32)).to(device),
                               torch.from_numpy(np.asarray(caps, np.float64)).to(device))
                target['requested_shifts_float64'] = raw
            tapes = [noise.tape(token, index, device) for index in request['noise_indices']]
            trajectories = np.empty((len(actions), len(tapes), 8, 3), np.float32)
            base_trajectories = np.empty((len(tapes), 8, 3), np.float32)
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                for j, tape in enumerate(tapes):
                    base_trajectories[j] = bridge.states_from_post(base, context, tape)['trajectory'][0].cpu().numpy()
                for position in range(len(actions)):
                    post = target['interfaces'][position:position + 1]
                    for j, tape in enumerate(tapes):
                        trajectories[position, j] = bridge.states_from_post(post, context, tape)['trajectory'][0].cpu().numpy()
            records = [{'arm': 'base', 'seed_index': noise_index, 'trajectory': trajectory.tolist()}
                       for noise_index, trajectory in zip(request['noise_indices'], base_trajectories)]
            for position, action in enumerate(actions):
                records.extend({'arm': str(task['positions'][position]), 'seed_index': noise_index, 'trajectory': trajectory.tolist()}
                               for noise_index, trajectory in zip(request['noise_indices'], trajectories[position]))
            metric_path = Path(meta['metric_cache'])
            # Pin the converted local bytes to the independently verified
            # registry. Hashing the current file alone would bless corruption.
            metric_sha = meta.get('local_metric_sha256') or meta.get('metric_cache_sha256')
            if not isinstance(metric_sha, str) or len(metric_sha) != 64:
                raise ValueError('A verified local metric-cache SHA is required')
            if sha(metric_path) != metric_sha:
                raise ValueError(f'Local metric cache differs from its pinned registry: {token}')
            scored = score_group({'metric_cache': str(metric_path), 'metric_sha256': metric_sha,
                                  'scoring': scoring, 'records': records})
            values = np.asarray([[record['score'][key] for key in FIELDS] for record in scored], np.float64)
            count_noise = len(tapes)
            base_scores = values[:count_noise]
            scores = values[count_noise:].reshape(len(actions), count_noise, 7)
            baseline_validation = 'new_validation_Base_measurement'
            if request['population'] in ('train', 'test'):
                native_cache = Path('/mnt/project/recogdrive-teacher2-a800-20261026/cache') / request['population'] / 'cases' / token[:2] / token / 'TARGETS.npz'
                if not native_cache.exists():
                    raise FileNotFoundError(f'Independent local Base cache missing: {native_cache}')
                with np.load(native_cache, allow_pickle=False) as frozen_base:
                    expected_trajectories = np.array(frozen_base['Base_trajectories'])[request['noise_indices']]
                    expected_scores = np.array(frozen_base['Base_quality'])[request['noise_indices']]
                if not np.array_equal(expected_trajectories, base_trajectories):
                    error = float(np.abs(expected_trajectories.astype(np.float64) - base_trajectories.astype(np.float64)).max())
                    raise ValueError(f'Native bridge/reference noise Base trajectory mismatch {token}: max_abs={error}')
                if not np.array_equal(expected_scores, base_scores):
                    error = float(np.abs(expected_scores - base_scores).max())
                    raise ValueError(f'Local converted-cache Base PDM mismatch {token}: max_abs={error}')
                baseline_validation = 'all_Base_trajectory_and_seven_metric_values_exact_to_independent_local_cache'
            output = folder / 'results' / task['id']
            result_arrays = {'positions': np.asarray(task['positions'], np.int64), 'noise_indices': np.asarray(request['noise_indices'], np.int64),
                       'trajectories': trajectories, 'base_trajectories': base_trajectories,
                       'scores': scores, 'base_scores': base_scores,
                       'actual_role_shifts': target['actual_role_shifts'].cpu().numpy(),
                       'actual_RMS': target['actual_RMS'].cpu().numpy(), 'cap_RMS': target['cap_RMS'].cpu().numpy(),
                       'requested_shifts_float64': target['requested_shifts_float64'],
                       'image_mask': np.array(arrays['image_mask']), 'text_mask': np.array(arrays['text_mask']), 'state24': state}
            numeric = save_npz(output / 'SCORES.npz', result_arrays)
            base_sha = hashlib_sha(arrays['base_post'])
            if base_sha != meta['actual_Base_interface_sha256']:
                raise ValueError('Immutable mmap H0 bytes changed during native scoring')
            audit = {}
            if request['scope'] == 'full_population':
                # Audit only: use the same FP64 point argmax as the independent
                # population collector, without changing any branch execution.
                point_scores = scores[:, :, 0].mean(axis=1, dtype=np.float64) * np.float64(100.)
                selected_position = int(point_scores.argmax())
                selected_choice = int(actions[selected_position]['choice'])
                selected_post = target['interfaces'][selected_position].detach().cpu().numpy()
                valid_cpu = np.asarray(arrays['image_mask']) | np.asarray(arrays['text_mask'])
                base_padding_sha = hashlib_sha(np.asarray(arrays['base_post'])[~valid_cpu])
                teacher_padding_sha = hashlib_sha(selected_post[~valid_cpu])
                if base_padding_sha != teacher_padding_sha:
                    raise ValueError('Same-execution selected teacher changed original FP32 padding bytes')
                valid_count = int(valid_cpu.sum())
                length = len(valid_cpu)
                base_energy_all = float(base.double().square().mean())
                base_energy_valid = float(base[:, torch.from_numpy(valid_cpu).to(device)].double().square().mean())
                residual_valid = float(target['actual_RMS'][selected_position].square())
                audit = {'selected_branch_point_argmax': selected_choice,
                         'selected_branch_position': task['positions'][selected_position],
                         'candidate_q_points_fp64': point_scores.tolist(),
                         'selected_teacher_interface_sha256': hashlib_sha(selected_post),
                         'Base_padding_bytes_sha256': base_padding_sha,
                         'selected_teacher_padding_bytes_sha256': teacher_padding_sha,
                         'valid_token_count': valid_count, 'native_sequence_length': length,
                         'Base_interface_energy_all': base_energy_all,
                         'Base_interface_energy_valid': base_energy_valid,
                         'selected_teacher_residual_energy_valid': residual_valid,
                         'selected_teacher_residual_energy_all': residual_valid * (valid_count / length),
                         'residual_energy_all_formula': 'residual_energy_valid * valid_token_count / native_sequence_length',
                         'energy_and_hash_fields_are_audit_only': True}
            result = {'status': 'COMPLETE', 'task': task, 'token': token, 'population': request['population'],
                      'arrays': numeric, 'metric_cache': str(metric_path), 'local_metric_sha256': metric_sha,
                      'reference_metric_sha256': meta.get('reference_metric_sha256'),
                      'local_metric_registry_sha256': meta.get('local_metric_registry_sha256'),
                      'feature_manifest': request['feature_manifest'], 'Base_interface_sha256': base_sha,
                      'immutable_full_padded_H0': {'manifest': request['feature_manifest'], 'population': request['population'],
                         'token': token, 'global_index': int(meta['global_index']), 'shard': int(meta['shard']),
                         'offset': int(meta['offset']), 'length': int(meta['length']), 'state_row': int(meta['state_row']),
                         'bytes_sha256': base_sha, 'duplicate_array_saved': False},
                      'native_identity_sha256': digest(identity), 'noise_identity': noise.identity,
                      **audit,
                      'baseline_validation': baseline_validation,
                      'elapsed_seconds': time.monotonic() - began, 'worker': worker}
            atomic(output / 'COMPLETE.json', result)
            atomic(folder / 'receipts' / (task['id'] + '.json'), {'status': 'COMPLETE', 'result': ref(output / 'COMPLETE.json'), 'worker': worker})
            completed += 1
            atomic(worker_folder / 'LIVE.json', {'status': 'SCORING', 'completed': completed, 'failed': failed,
                   'elapsed_seconds': time.monotonic() - start, 'last_token': token, 'time': stamp()})
            if completed % 10 == 0:
                print(json.dumps({'worker': worker, 'completed': completed, 'failed': failed, 'elapsed_seconds': time.monotonic() - start}), flush=True)
        except Exception:
            failed += 1
            destination = folder / 'failures' / task['id'] / (str(time.time_ns()) + '.json')
            atomic(destination, {'status': 'FAILED', 'worker': worker, 'task': task, 'traceback': traceback.format_exc()})
            if args.fail_fast:
                raise
        finally:
            lock.close()
        if args.limit and completed >= args.limit:
            break
    atomic(worker_folder / 'COMPLETE.json', {'status': 'WORKER_EXITED', 'completed': completed, 'failed': failed,
                                          'elapsed_seconds': time.monotonic() - start, 'time': stamp(),
                                          'inference_graph_statistics': getattr(bridge, 'graph_statistics', None)})


def hashlib_sha(value):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--round1-repo', type=Path, default=ROUND1)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--noise-protocol', type=Path, required=True)
    parser.add_argument('--noise-index', type=Path, required=True)
    parser.add_argument('--noise-bank', type=Path)
    parser.add_argument('--worker-id')
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--fail-fast', action='store_true')
    parser.add_argument('--inference-graph-receipt', type=Path)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
