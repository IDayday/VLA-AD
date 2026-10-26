"""Convert audited native post NPZ cases once into indexed contiguous mmap shards."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
from .asset_identity import atomic_json, canonical_sha, population_rows, read_json, sha256_file

FIELDS = {'base_post': ('Base_interface', np.float32), 'image_mask': ('image_mask', np.bool_),
          'text_mask': ('text_mask', np.bool_), 'token_ids': ('token_ids', np.int64),
          'attention_mask': ('attention_mask', np.int64), 'position_ids': ('position_ids', np.int64)}


def validate_arrays(arrays):
    post = arrays['base_post']
    if post.dtype != np.float32 or post.ndim != 2 or post.shape[1] != 384 or not np.isfinite(post).all():
        raise ValueError('Base_interface must be finite native FP32[L,384]')
    length = len(post)
    for name in FIELDS:
        a = arrays[name]
        if name != 'base_post' and a.shape != (length,):
            raise ValueError(f'Native layout mismatch: {name}')
    im, tx = arrays['image_mask'], arrays['text_mask']
    if im.dtype != np.bool_ or tx.dtype != np.bool_ or np.any(im & tx) or not im.any() or not tx.any():
        raise ValueError('Both nonoverlapping image/text roles required')
    if not np.array_equal(im | tx, arrays['attention_mask'].astype(bool)):
        raise ValueError('Valid masks and attention layout disagree')
    state = arrays['state24']
    if state.shape != (24,) or state.dtype != np.float32 or not np.isfinite(state).all():
        raise ValueError('state24 must be finite FP32[history12,status8,command4]')
    if not np.array_equal(state[12:16], state[20:24]):
        raise ValueError('Command does not match ordinary builder status ordering')
    return length


class OrdinaryStateReader:
    """Use actual NAVSIM coordinate conversion, excluding cameras and all future GT."""
    def __init__(self, paths, out):
        from interface_teacher.runtime import bootstrap
        bootstrap(paths, out)
        self.paths = read_json(paths)
        self._logs = {}

    def __call__(self, row, population):
        import pickle
        import torch
        from navsim.common.dataclasses import AgentInput, SensorConfig
        cohort = 'navtest' if population == 'test' else 'navtrain'
        path = str(Path(self.paths[cohort + '_logs']) / row['raw_rel'])
        if path not in self._logs:
            if len(self._logs) >= 2:
                self._logs.pop(next(iter(self._logs)))
            with open(path, 'rb') as f:
                self._logs[path] = pickle.load(f)
        data = self._logs[path]
        i = int(row['frame_index'])
        if data[i]['token'] != row['token']:
            raise ValueError('Raw frame/token mismatch')
        sensors = SensorConfig(cam_f0=[], cam_l0=[], cam_l1=[], cam_l2=[], cam_r0=[], cam_r1=[], cam_r2=[], cam_b0=[], lidar_pc=[])
        frames = data[i-3:i+1]
        if len(frames) != 4:
            raise ValueError('Native history requires four frames')
        inp = AgentInput.from_scene_dict_list(frames, Path(self.paths[cohort + '_sensor']) / row['sensor_rel'], 4, sensors)
        ego = inp.ego_statuses
        # Verbatim ordinary feature-builder ordering and FP32 conversion.
        history = torch.tensor([[float(e.ego_pose[0]), float(e.ego_pose[1]), float(e.ego_pose[2])] for e in ego[:4]], dtype=torch.float32)
        command = torch.tensor(ego[-1].driving_command, dtype=torch.float32)
        status = torch.cat([command.clone(), torch.tensor(ego[-1].ego_velocity, dtype=torch.float32), torch.tensor(ego[-1].ego_acceleration, dtype=torch.float32)], dim=-1)
        return torch.cat([history.flatten(), status, command]).numpy()


def state_npz_reader(path):
    with np.load(path, allow_pickle=False) as z:
        tokens = z['tokens'].astype(str).tolist()
        if 'state24' in z:
            states = z['state24'].copy()
        else:
            states = np.concatenate([z['history12'].reshape(-1, 12), z['status8'], z['command4']], 1)
    if len(set(tokens)) != len(tokens):
        raise ValueError('Duplicate state tokens')
    mapping = dict(zip(tokens, states))
    return lambda row, population: np.asarray(mapping[row['token']], dtype=np.float32)


def join_original_global_indices(rows, manifest):
    """Join only identity columns from the original 103288-scene source list."""
    original = read_json(manifest)
    lookup = {r['token']: r for r in original}
    if len(lookup) != len(original):
        raise ValueError('Duplicate original scene tokens')
    result = []
    for row in rows:
        source = lookup[row['token']]
        for field in ('session', 'log_name', 'frame_index'):
            if str(row[field]) != str(source[field]):
                raise ValueError(f'Original global-index identity mismatch: {row["token"]}/{field}')
        index = int(source['global_index'])
        if 'index' in row and int(row['index']) != index:
            raise ValueError('Original global index disagrees with frozen population')
        result.append(dict(row, index=str(index)))
    return result


def local_metric_entries(paths, population, rows):
    registry_path = paths.get('local_metric_registries', {}).get(population)
    if registry_path is None:
        return {}, None
    registry = read_json(registry_path)
    entries = registry['entries']
    selected = {}
    for row in rows:
        entry = entries[row['token']]
        if entry['relative'] != row['metric_rel'] or entry['reference_sha256'] != row['metric_sha256']:
            raise ValueError(f'Local metric registry source identity mismatch: {row["token"]}')
        if len(entry['local_sha256']) != 64:
            raise ValueError('Invalid local metric checksum')
        selected[row['token']] = entry
    return selected, sha256_file(registry_path)


def convert_population(source_index, population, out, *, frozen_repo=None, paths=None,
                       state_reader=None, shard_scenes=64, verify_source=True, require_complete=True, rank=0, world_size=1):
    out = Path(out).resolve()
    source_index = Path(source_index).resolve()
    if not 0 <= rank < world_size:
        raise ValueError('Invalid rank/world size')
    pop_dir = out / population if world_size == 1 else out / '.parts' / population / f'rank_{rank:03d}_of_{world_size:03d}'
    pop_dir.mkdir(parents=True, exist_ok=True)
    complete_path = pop_dir / 'COMPLETE.json'
    if complete_path.exists():
        previous = read_json(complete_path)
        if previous['source_index_sha256'] != sha256_file(source_index) or previous.get('world_size', 1) != world_size or previous.get('rank', 0) != rank:
            raise ValueError('Completed feature population has different source identity')
        return previous['population_spec']
    source_rows = [json.loads(s) for s in source_index.read_text().splitlines() if s.strip()]
    paths_dict = read_json(paths) if paths else {}
    ordinary = population_rows(population, frozen_repo)
    if population == 'validation' and any('index' not in r for r in ordinary):
        source_manifest = paths_dict.get('global_index_manifest')
        if not source_manifest:
            raise ValueError('Validation requires the original token-joined global_index_manifest')
        ordinary = join_original_global_indices(ordinary, source_manifest)
    metrics, metric_registry_sha = local_metric_entries(paths_dict, population, ordinary)
    by_token = {r['token']: r for r in ordinary}
    if len({r['token'] for r in source_rows}) != len(source_rows):
        raise ValueError('Duplicate source tokens')
    if require_complete and set(by_token) != {r['token'] for r in source_rows}:
        raise ValueError(f'Incomplete {population} Base cache')
    if not {r['token'] for r in source_rows} <= set(by_token):
        raise ValueError('Foreign-population source token')
    source_rows.sort(key=lambda r: (by_token[r['token']]['raw_rel'], int(by_token[r['token']]['frame_index']), r['token']))
    source_rows = source_rows[len(source_rows) * rank // world_size:len(source_rows) * (rank + 1) // world_size]
    # Actual source lengths can exceed 2800. Estimate is a lower bound for admission,
    # followed by per-shard checks before all writes; never truncate a native layout.
    estimate = len(source_rows) * 2800 * (384 * 4 + 2 + 3 * 8)
    if shutil.disk_usage(out).free < estimate * 1.02:
        raise OSError(f'Insufficient disk for native mmap cache: estimated {estimate} bytes')
    shards, records = [], []
    for begin in range(0, len(source_rows), shard_scenes):
        chunk = source_rows[begin:begin + shard_scenes]
        shard_id = len(shards)
        destination = pop_dir / f'shard_{shard_id:05d}'
        temp = destination.with_name(destination.name + '.partial')
        if temp.exists():
            shutil.rmtree(temp)
        temp.mkdir()
        handles = {k: (temp / (k + '.bin')).open('wb') for k in (*FIELDS, 'state24')}
        total = 0
        hashes = {k: hashlib.sha256() for k in handles}
        try:
            for state_row, src in enumerate(chunk):
                row = by_token[src['token']]
                source = source_index.parent / src['npz']
                if verify_source and sha256_file(source) != src['sha256']:
                    raise ValueError(f'Native source SHA mismatch: {source}')
                with np.load(source, allow_pickle=False) as z:
                    arrays = {name: np.array(z[field], copy=True) for name, (field, dtype) in FIELDS.items()}
                    if any(arrays[k].dtype != np.dtype(dtype) for k, (_, dtype) in FIELDS.items()):
                        raise ValueError('Source cache changed native dtype')
                    if 'state24' in z:
                        arrays['state24'] = np.array(z['state24'], copy=True)
                        ordinary_names = ('history_trajectory', 'status_feature', 'high_command_one_hot')
                        if all(k in z for k in ordinary_names):
                            builder_state = np.concatenate([np.asarray(z[k]).reshape(-1) for k in ordinary_names])
                            if builder_state.dtype != np.float32 or not np.array_equal(builder_state, arrays['state24']):
                                raise ValueError('Native state24 does not match ordinary builder fields')
                    else:
                        if state_reader is None:
                            if not paths:
                                raise ValueError('Supply ordinary state24 reader or verified raw paths')
                            state_reader = OrdinaryStateReader(paths, out)
                        arrays['state24'] = np.asarray(state_reader(row, population))
                    length = validate_arrays(arrays)
                    rec = {'token': row['token'], 'population': population, 'log_id': row['session'],
                           'log_name': row['log_name'], 'global_index': int(row['index']) if 'index' in row else None,
                           'shard': shard_id, 'offset': total, 'length': length, 'state_row': state_row,
                           'source_npz_sha256': src['sha256'],
                           'actual_Base_interface_sha256': hashlib.sha256(arrays['base_post'].tobytes()).hexdigest(),
                           'token_layout_sha256': canonical_sha({k: hashlib.sha256(arrays[k].tobytes()).hexdigest() for k in FIELDS if k != 'base_post'})}
                    cohort = 'navtest' if population == 'test' else 'navtrain'
                    if paths_dict.get(cohort + '_metric'):
                        rec['metric_cache'] = str(Path(paths_dict[cohort + '_metric']) / row['metric_rel'])
                    rec['metric_rel'] = row['metric_rel']
                    rec['reference_metric_sha256'] = row.get('metric_sha256')
                    if row['token'] in metrics:
                        rec['local_metric_sha256'] = metrics[row['token']]['local_sha256']
                        rec['metric_cache_sha256'] = metrics[row['token']]['local_sha256']
                        rec['local_metric_registry_sha256'] = metric_registry_sha
                    if 'Base_quality' in z:
                        rec['base_scores'] = np.asarray(z['Base_quality'][:, 0], np.float64).tolist()
                        rec['base_mean_points'] = float(np.mean(z['Base_quality'][:, 0], dtype=np.float64) * 100.0)
                for name, a in arrays.items():
                    payload = np.ascontiguousarray(a).tobytes()
                    handles[name].write(payload)
                    hashes[name].update(payload)
                records.append(rec)
                total += length
        finally:
            for h in handles.values():
                h.flush()
                os.fsync(h.fileno())
                h.close()
        refs = {}
        for name in handles:
            dtype = np.float32 if name == 'state24' else FIELDS[name][1]
            shape = [len(chunk), 24] if name == 'state24' else ([total, 384] if name == 'base_post' else [total])
            file = temp / (name + '.bin')
            if file.stat().st_size != int(np.prod(shape)) * np.dtype(dtype).itemsize:
                raise ValueError('Incomplete shard write')
            if sha256_file(file) != hashes[name].hexdigest():
                raise ValueError('Shard readback SHA mismatch')
            refs[name] = {'path': str(destination / file.name), 'sha256': hashes[name].hexdigest(), 'dtype': np.dtype(dtype).str, 'shape': shape}
        atomic_json(temp / 'COMPLETE.json', {'arrays': refs, 'scenes': len(chunk), 'tokens': total})
        if destination.exists():
            # Only an unregistered interrupted output is replaced; completed population
            # returned above before this point.
            shutil.rmtree(destination)
        temp.rename(destination)
        shards.append({'arrays': refs, 'scenes': len(chunk), 'tokens': total})
        progress = {'population': population, 'scenes_cached': len(records), 'total': len(source_rows), 'rank': rank, 'world_size': world_size, 'status': 'CONVERTING_BASE_ONLY'}
        atomic_json(pop_dir / 'LIVE.json', progress)
        print(json.dumps(progress), flush=True)
    index_path = pop_dir / 'INDEX.jsonl'
    temp = index_path.with_suffix('.tmp')
    with temp.open('w') as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    temp.replace(index_path)
    spec = {'count': len(records), 'index': {'path': str(index_path), 'sha256': sha256_file(index_path)}, 'shards': shards,
            'source_index_sha256': sha256_file(source_index), 'native_layout': 'full_unmodified_including_padding'}
    atomic_json(complete_path, {'status': 'COMPLETE', 'population_spec': spec, 'source_index_sha256': sha256_file(source_index), 'rank': rank, 'world_size': world_size})
    return spec


def index_native_population(native_root, population, *, frozen_repo=None, global_index_manifest=None, verify_arrays=True):
    """Index a newly generated full native Base population only after every receipt."""
    native_root = Path(native_root).resolve()
    rows = population_rows(population, frozen_repo)
    if population == 'validation' and any('index' not in r for r in rows):
        if not global_index_manifest:
            raise ValueError('Validation indexing requires original global_index_manifest')
        rows = join_original_global_indices(rows, global_index_manifest)
    records = []
    for row in rows:
        token = row['token']
        folder = native_root / 'cases' / token
        if not folder.exists():
            folder = native_root / 'cases' / token[:2] / token
        receipt_path = folder / 'COMPLETE.json'
        if not receipt_path.is_file():
            raise FileNotFoundError(f'Native Base population not ready: {receipt_path}')
        receipt = read_json(receipt_path)
        arrays = folder / 'BASE_INPUTS.npz'
        expected_sha = receipt.get('arrays_sha256', receipt.get('sha256'))
        if expected_sha is None or not arrays.is_file() or (verify_arrays and sha256_file(arrays) != expected_sha):
            raise ValueError(f'New native Base arrays SHA mismatch: {token}')
        if 'index' in row and receipt.get('global_index') != int(row['index']):
            raise ValueError('Native receipt global index differs from original token join')
        record = {'token': token, 'global_index': int(row['index']) if 'index' in row else None,
                  'session': row['session'], 'npz': str(arrays.relative_to(native_root)),
                  'sha256': expected_sha, 'bytes': arrays.stat().st_size}
        records.append(record)
    path = native_root / 'INDEX.jsonl'
    temp = path.with_suffix('.tmp')
    with temp.open('w') as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + '\n')
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)
    atomic_json(native_root / 'INDEX_COMPLETE.json', {'population': population, 'scenes': len(records),
                'index_sha256': sha256_file(path), 'teacher_labels_used': False,
                'array_bytes_verified_during_index': verify_arrays,
                'conversion_must_verify_each_source_sha': True,
                'global_index_manifest_sha256': sha256_file(global_index_manifest) if global_index_manifest else None})
    return path


def merge_population(out, population, world_size, source_index, *, frozen_repo=None):
    """Publish only after disjoint rank artifacts cover the complete true population."""
    out = Path(out).resolve()
    source_sha = sha256_file(source_index)
    shards, rows = [], []
    for rank in range(world_size):
        folder = out / '.parts' / population / f'rank_{rank:03d}_of_{world_size:03d}'
        done = read_json(folder / 'COMPLETE.json')
        if done['status'] != 'COMPLETE' or done['source_index_sha256'] != source_sha or done['rank'] != rank or done['world_size'] != world_size:
            raise ValueError('Rank source/identity mismatch')
        spec = done['population_spec']
        index = Path(spec['index']['path'])
        if sha256_file(index) != spec['index']['sha256']:
            raise ValueError('Rank index changed')
        for line in index.read_text().splitlines():
            row = json.loads(line)
            row['shard'] += len(shards)
            rows.append(row)
        shards.extend(spec['shards'])
    expected = population_rows(population, frozen_repo)
    if len(rows) != len(expected) or len({r['token'] for r in rows}) != len(rows) or {r['token'] for r in rows} != {r['token'] for r in expected}:
        raise ValueError('Disjoint rank coverage is incomplete or duplicated')
    pop_dir = out / population
    pop_dir.mkdir(parents=True, exist_ok=True)
    index = pop_dir / 'INDEX.jsonl'
    temp = index.with_suffix('.tmp')
    with temp.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    temp.replace(index)
    spec = {'count': len(rows), 'index': {'path': str(index), 'sha256': sha256_file(index)}, 'shards': shards,
            'source_index_sha256': source_sha, 'world_size': world_size, 'native_layout': 'full_unmodified_including_padding'}
    atomic_json(pop_dir / 'COMPLETE.json', {'status': 'COMPLETE', 'population_spec': spec, 'source_index_sha256': source_sha,
                                          'world_size': world_size, 'all_rank_exit_and_artifact_checks_required': True})
    return spec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths', required=True)
    p.add_argument('--population', nargs='+', default=['train', 'validation', 'test'], choices=['train', 'validation', 'test'])
    p.add_argument('--source-index', help='Explicit source INDEX for a single population')
    p.add_argument('--index-native-root', help='Index complete BASE_INPUTS.npz native exports before conversion; one population')
    p.add_argument('--state-manifest', help='NPZ tokens/state24 with actual ordinary builder outputs')
    p.add_argument('--preserve-native-layout', action='store_true', required=True)
    p.add_argument('--shard-scenes', type=int, default=64)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world-size', type=int, default=1)
    p.add_argument('--merge-ranks', action='store_true')
    p.add_argument('--out', required=True)
    args = p.parse_args(argv)
    paths = read_json(args.paths)
    out = Path(args.out).resolve()
    import fcntl
    out.mkdir(parents=True, exist_ok=True)
    lock_path = out / ('.merge.lock' if args.merge_ranks else f'.rank_{args.rank:03d}_of_{args.world_size:03d}.lock')
    rank_lock = lock_path.open('a')
    fcntl.flock(rank_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = out / 'MANIFEST.json'
    manifest = read_json(manifest_path) if manifest_path.exists() else {'schema': 'nrkd_native_post_mmap_v1', 'preserve_native_layout': True, 'populations': {}, 'source_checkpoint_sha256': paths.get('source_checkpoint_sha256', '23fb949b10387efeff87e093703d5faf41b61b9e5ead16d4076661e3c7c08a20')}
    if (args.source_index or args.index_native_root) and len(args.population) != 1:
        p.error('--source-index requires one population')
    for population in args.population:
        source = args.source_index or paths.get('post_cache_indices', {}).get(population)
        if args.index_native_root:
            source = index_native_population(args.index_native_root, population, frozen_repo=paths.get('frozen_repo'), global_index_manifest=paths.get('global_index_manifest'))
        if not source:
            raise FileNotFoundError(f'Missing audited native post INDEX for {population}; generate it before conversion')
        if args.merge_ranks:
            spec = merge_population(out, population, args.world_size, source, frozen_repo=paths.get('frozen_repo'))
        else:
            reader = state_npz_reader(args.state_manifest) if args.state_manifest else None
            spec = convert_population(source, population, out, frozen_repo=paths.get('frozen_repo'), paths=args.paths, state_reader=reader, shard_scenes=args.shard_scenes, rank=args.rank, world_size=args.world_size)
        if args.world_size == 1 or args.merge_ranks:
            manifest['populations'][population] = spec
            atomic_json(manifest_path, manifest)
    print(manifest_path)


if __name__ == '__main__':
    main()
