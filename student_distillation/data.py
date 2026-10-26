"""Lossless mmap scene inputs, joined to offline labels by token only."""
from __future__ import annotations
import json
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from .asset_identity import read_json, resolve_ref, sha256_file

INPUT_KEYS = ('base_post', 'image_mask', 'text_mask', 'state24',
              'teacher_actual_shift', 'history12', 'status8')


def _load_manifest(value):
    if isinstance(value, dict):
        return value, Path(value.get('_root', '.')).resolve()
    p = Path(value).resolve()
    return read_json(p), p.parent


class FeatureStore:
    """Native full-token FP32 arrays; source NPZ is never opened during training."""
    def __init__(self, manifest, population, *, verify=False, mmap_cache_shards=8):
        self.manifest, self.root = _load_manifest(manifest)
        if self.manifest.get('schema') != 'nrkd_native_post_mmap_v1':
            raise ValueError('Expected lossless nrkd_native_post_mmap_v1 feature manifest')
        if not self.manifest.get('preserve_native_layout'):
            raise ValueError('Native padding/layout must be preserved')
        self.population = population
        self.spec = self.manifest['populations'][population]
        index_path = resolve_ref(self.spec['index'], self.root, verify=True)
        self.rows = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
        if len(self.rows) != self.spec['count'] or len({r['token'] for r in self.rows}) != len(self.rows):
            raise ValueError('Feature coverage/duplicate mismatch')
        self.by_token = {r['token']: i for i, r in enumerate(self.rows)}
        registry_ref = self.manifest.get('local_metric_registries', {}).get(population)
        if registry_ref:
            from .asset_identity import population_rows
            registry_path = resolve_ref(registry_ref, self.root, verify=True)
            entries = read_json(registry_path)['entries']
            registry_sha = sha256_file(registry_path)
            ordinary = {r['token']: r for r in population_rows(population, self.manifest.get('frozen_repo'))}
            for row in self.rows:
                entry, source = entries[row['token']], ordinary[row['token']]
                if entry['relative'] != source['metric_rel'] or entry['reference_sha256'] != source['metric_sha256'] or row['metric_rel'] != entry['relative']:
                    raise ValueError('Local metric registry differs from frozen scene identity')
                row['local_metric_sha256'] = entry['local_sha256']
                row['metric_cache_sha256'] = entry['local_sha256']
                row['reference_metric_sha256'] = entry['reference_sha256']
                row['local_metric_registry_sha256'] = registry_sha
        self._arrays = {}
        self._shard_order = OrderedDict()
        self._mmap_cache_shards = int(mmap_cache_shards)
        if self._mmap_cache_shards <= 0:
            raise ValueError('mmap_cache_shards must be positive')
        if verify:
            for shard in self.spec['shards']:
                for ref in shard['arrays'].values():
                    resolve_ref(ref, self.root, verify=True)

    def _array(self, shard, name):
        key = (shard, name)
        if shard in self._shard_order:
            self._shard_order.move_to_end(shard)
        else:
            while len(self._shard_order) >= self._mmap_cache_shards:
                evicted, _ = self._shard_order.popitem(last=False)
                # Drop ownership only: any returned scene view keeps its own
                # numpy base alive. Explicit mmap.close would invalidate it.
                for old_key in [k for k in self._arrays if k[0] == evicted]:
                    del self._arrays[old_key]
            self._shard_order[shard] = None
        if key not in self._arrays:
            ref = self.spec['shards'][shard]['arrays'][name]
            p = resolve_ref(ref, self.root, verify=False)
            self._arrays[key] = np.memmap(p, mode='r', dtype=np.dtype(ref['dtype']), shape=tuple(ref['shape']))
        return self._arrays[key]

    def read(self, index):
        r = self.rows[index]
        s, o, n = int(r['shard']), int(r['offset']), int(r['length'])
        state = np.asarray(self._array(s, 'state24')[int(r['state_row'])])
        result = {name: np.asarray(self._array(s, name)[o:o+n]) for name in
                  ('base_post', 'image_mask', 'text_mask', 'token_ids', 'attention_mask', 'position_ids')}
        result['state24'] = state
        result['history12'], result['status8'] = state[:12], state[12:20]
        return result

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_arrays'] = {}
        state['_shard_order'] = OrderedDict()
        return state


class SceneDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, population='train', target_version='full'):
        top, root = _load_manifest(manifest)
        if 'feature_manifest' in top:
            features = resolve_ref(top['feature_manifest'], root, verify=False)
        else:
            features = manifest
        self.features = FeatureStore(features, population)
        self.population, self.target_version = population, target_version
        self.metadata = [dict(r) for r in self.features.rows]
        self._target_shifts = None
        self._target_lookup = None
        targets = top.get('targets_manifest') or self.features.manifest.get('targets_manifest')
        if target_version not in (None, 'none'):
            if not targets:
                raise ValueError(f'No frozen {target_version} targets for {population}')
            tp = resolve_ref(targets, root, verify=False)
            tm = read_json(tp)
            if tm.get('schema') != 'nrkd_compact_targets_v1' or tm.get('teacher_origin') != 'local_refit':
                raise ValueError('Only new local-refit compact targets are accepted')
            spec = tm['populations'][population]
            if population != 'train' and not spec.get('diagnostic_only'):
                raise ValueError('Validation/Test targets must be explicitly diagnostic only')
            target_rows_path = resolve_ref(spec['index'], tp.parent, verify=True)
            target_rows = [json.loads(l) for l in target_rows_path.read_text().splitlines() if l.strip()]
            target_map = {r['token']: i for i, r in enumerate(target_rows)}
            if len(target_map) != len(target_rows) or set(target_map) != set(self.features.by_token):
                raise ValueError('Feature/target token coverage mismatch')
            ref = spec['versions'][target_version]['shifts']
            self._target_shifts = np.load(resolve_ref(ref, tp.parent, verify=True), mmap_mode='r', allow_pickle=False)
            if self._target_shifts.dtype != np.float32 or self._target_shifts.shape != (len(target_rows), 384):
                raise ValueError('Invalid compact shifts')
            self._target_lookup = [target_map[r['token']] for r in self.metadata]
            for r in self.metadata:
                tr = target_rows[target_map[r['token']]]
                for key in ('log_id', 'global_index'):
                    if key in tr and key in r and str(tr[key]) != str(r[key]):
                        raise ValueError(f'Target metadata mismatch: {r["token"]}/{key}')
                r.update(tr)
                r.update(target_is_zero=bool(tr[f'{target_version}_is_zero']), target_available=True,
                         teacher_version=target_version, teacher_identity=tm['teacher_identity'])
        else:
            for r in self.metadata:
                r.update(target_available=False, target_is_zero=None)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        arrays = self.features.read(index)
        shift = (np.zeros(384, np.float32) if self._target_shifts is None else
                 np.asarray(self._target_shifts[self._target_lookup[index]]))
        # Copy each scene out of read-only mmap so tensors cannot mutate the cache.
        result = {k: torch.from_numpy(np.array(arrays[k], copy=True)) for k in INPUT_KEYS if k != 'teacher_actual_shift'}
        result['teacher_actual_shift'] = torch.from_numpy(np.array(shift, copy=True)).detach()
        result['metadata'] = dict(self.metadata[index])
        return result


def batch_scene(item, device):
    """Only ordinary tensors enter the model; all privileged fields stay in metadata."""
    result = {k: item[k].unsqueeze(0).to(device, non_blocking=True) for k in INPUT_KEYS}
    result['metadata'] = item.get('metadata', {})
    # Token is needed only for stateless sampling, never passed to adapter.forward.
    if 'token' in result['metadata']:
        result['token'] = result['metadata']['token']
    for key in ('time_index', 'noise_bank', 'noise_tape'):
        if key in item:
            result[key] = item[key]
    return result
