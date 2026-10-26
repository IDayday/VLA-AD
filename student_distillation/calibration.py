"""Freeze Train-only normalization, loss scales and population-identical panels."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import numpy as np
from .asset_identity import atomic_json, canonical_sha, read_json, sha256_file
from .data import SceneDataset, batch_scene


def _rank(token, namespace):
    return hashlib.sha256((namespace + '|' + token).encode()).hexdigest()


def deterministic_train_stats_indices(rows, count=4096):
    if len(rows) < count:
        raise ValueError(f'Train statistics require {count} real scenes')
    return sorted(range(len(rows)), key=lambda i: (_rank(rows[i]['token'], 'NRKD_train_stats_v1'), rows[i]['token']))[:count]


def deterministic_log_coverage_indices(rows, count, *, repaired_only=False, namespace='NRKD_train_mu_calibration_v1'):
    groups = {}
    for i, r in enumerate(rows):
        if repaired_only and r['target_is_zero']:
            continue
        groups.setdefault(r['log_id'], []).append(i)
    if sum(map(len, groups.values())) < count:
        raise ValueError(f'Not enough repaired Train scenes for fixed {count} calibration')
    for log, indices in groups.items():
        indices.sort(key=lambda i: (_rank(rows[i]['token'], namespace), rows[i]['token']))
    # Round-robin deterministically covers every eligible Train session before
    # assigning second slots; no label or score ordering inside a log.
    selected = []
    depth = 0
    while len(selected) < count:
        for log in sorted(groups):
            if depth < len(groups[log]):
                selected.append(groups[log][depth])
                if len(selected) == count:
                    break
        depth += 1
    return selected


def build_train_behavior_panel(rows, count=8192):
    if len(rows) < count:
        raise ValueError('Cannot shorten frozen Train behavior panel')
    groups = {}
    for i, row in enumerate(rows):
        score = float(row['base_mean_points'])
        bucket = 0 if score < 70.0 else (1 if score <= 90.0 else 2)
        groups.setdefault((row['log_id'], bucket), []).append(i)
    # Integer numerators avoid floating-point tie ambiguity in largest remainder.
    total = len(rows)
    quotas = {key: len(ix) * count // total for key, ix in groups.items()}
    remainders = {key: len(ix) * count % total for key, ix in groups.items()}
    remaining = count - sum(quotas.values())
    for key in sorted(groups, key=lambda k: (-remainders[k], k[0], k[1]))[:remaining]:
        quotas[key] += 1
    selected = []
    strata = []
    for key in sorted(groups):
        ix = sorted(groups[key], key=lambda i: (_rank(rows[i]['token'], 'NRKD_train_panel_v1'), rows[i]['token']))
        selected.extend(ix[:quotas[key]])
        strata.append({'log_id': key[0], 'base_bucket': key[1], 'population': len(ix), 'quota': quotas[key], 'remainder_numerator': remainders[key]})
    if len(selected) != count or len(set(selected)) != count:
        raise AssertionError('Panel allocation failed')
    tokens = [rows[i]['token'] for i in selected]
    return {'population': 'train', 'population_size': len(rows), 'panel_size': count,
            'description': 'Fixed Train behavior panel, not full-Train PDMS',
            'bucket_definition': ['<70', '[70,90]', '>90'], 'strata': strata,
            'tokens': tokens, 'tokens_sha256': canonical_sha(tokens), 'indices': selected,
            'noise_indices': [0, 1, 2, 3, 4], 'teacher_denominator': 'same_panel_same_target_version_same_noise'}


def _freeze_list(rows, indices, purpose):
    tokens = [rows[i]['token'] for i in indices]
    return {'population': 'train', 'purpose': purpose, 'count': len(tokens), 'tokens': tokens,
            'tokens_sha256': canonical_sha(tokens), 'log_ids': sorted({rows[i]['log_id'] for i in indices})}


def input_statistics(dataset, indices, std_floor=1e-3):
    token_sum, token_sq = np.zeros(384, np.float64), np.zeros(384, np.float64)
    state_sum, state_sq = np.zeros(24, np.float64), np.zeros(24, np.float64)
    token_count = 0
    for i in indices:
        arrays = dataset.features.read(i)
        valid = arrays['image_mask'] | arrays['text_mask']
        tokens = arrays['base_post'][valid].astype(np.float64)
        state = arrays['state24'].astype(np.float64)
        token_sum += tokens.sum(0)
        token_sq += np.square(tokens).sum(0)
        token_count += len(tokens)
        state_sum += state
        state_sq += np.square(state)
    token_mean = token_sum / token_count
    state_mean = state_sum / len(indices)
    token_std_raw = np.sqrt(np.maximum(token_sq / token_count - token_mean ** 2, 0))
    state_std_raw = np.sqrt(np.maximum(state_sq / len(indices) - state_mean ** 2, 0))
    return {'token_mean': token_mean.tolist(), 'token_std': np.maximum(token_std_raw, std_floor).tolist(),
            'state_mean': state_mean.tolist(), 'state_std': np.maximum(state_std_raw, std_floor).tolist(),
            'input_std_floor': std_floor, 'token_std_floor_channels': int((token_std_raw < std_floor).sum()),
            'state_std_floor_channels': int((state_std_raw < std_floor).sum()), 'valid_tokens_in_statistics': token_count}


def h_mse_scale(full_dataset):
    if full_dataset.population != 'train' or full_dataset.target_version != 'full':
        raise ValueError('H scale must use full Train teacher shifts')
    total, count = 0.0, 0
    for i, j in enumerate(full_dataset._target_lookup):
        shift = np.asarray(full_dataset._target_shifts[j], np.float64)
        if np.any(shift):
            total += float(np.square(shift).mean())
            count += 1
    value = total / count if count else float('nan')
    if not np.isfinite(value) or value <= 0:
        raise ValueError('Full local Train teacher has invalid/zero shift scale')
    return {'h_mse_scale': value, 'h_scale_nonzero_scene_count': count,
            'h_scale_epsilon_guard_active': value < 1e-12,
            'h_scale_definition': 'mean_scene_channel_squared_full_train_nonzero_actual_shift'}


def transition_statistics(dataset, indices, bridge, *, device='cuda', calibration_seed=20261027):
    import torch
    from .noise import make_noise_tape, train_noise_seed
    if dataset.population != 'train' or dataset.target_version != 'gain1':
        raise ValueError('Transition calibration must use gain1 Train targets')
    sums = np.zeros(5, np.float64)
    observations = 0
    for n, index in enumerate(indices):
        item = dataset[index]
        if item['metadata']['target_is_zero']:
            raise ValueError('Zero repair in repaired calibration set')
        batch = batch_scene(item, device)
        base = batch['base_post']
        valid = batch['image_mask'] | batch['text_mask']
        teacher = (base + valid.unsqueeze(-1).to(base.dtype) * batch['teacher_actual_shift'].unsqueeze(1)).detach()
        with torch.no_grad():
            context = bridge.encode_context(batch['history12'], batch['status8'])
            for bank in (0, 1):
                tape = make_noise_tape(train_noise_seed(item['metadata']['token'], calibration_seed, bank), device=device)
                chain = bridge.states_from_post(base, context, tape)
                for j in range(5):
                    state = chain['states'][j].detach()
                    base_mean = bridge.mean_from_post(state, j, base, context)[0]
                    teacher_mean = bridge.mean_from_post(state, j, teacher, context)[0]
                    mse = (base_mean.double() - teacher_mean.double()).square().mean()
                    sums[j] += float(mse)
                observations += 1
        if (n + 1) % 32 == 0:
            print(json.dumps({'mu_calibration_scenes': n + 1, 'total': len(indices)}), flush=True)
    raw = sums / observations
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite normalized-action mean calibration')
    floor = max(0.1 * float(raw.mean()), 1e-6)
    final = np.maximum(raw, floor)
    return {'mu_mse_raw': raw.tolist(), 'mu_mse_scale': final.tolist(), 'mu_floor': floor,
            'mu_floor_active': (raw < floor).tolist(), 'mu_all_steps_floor_dominated': bool((raw < floor).all()),
            'mu_units': 'normalized_action_squared', 'mu_calibration_state_source': 'frozen_Base',
            'mu_calibration_seed': calibration_seed, 'mu_noise_banks': [0, 1], 'mu_time_indices': list(range(5)),
            'mu_noise_namespace': 'NRKD_train_v1', 'mu_same_state_query': True, 'mu_observations_per_step': observations}


def prepare_input_statistics(feature_manifest, out, *, stats_count=4096, panel_count=8192):
    """Base-only CPU work can finish while the independent local teacher is fitting."""
    dataset = SceneDataset(feature_manifest, 'train', None)
    if len(dataset) != 85813:
        raise ValueError('Input normalization requires complete Train85813')
    indices = deterministic_train_stats_indices(dataset.metadata, stats_count)
    result = {'schema': 'nrkd_base_input_statistics_v1', 'status': 'BASE_INPUT_STATISTICS_ONLY',
              'population': 'train', 'feature_manifest_sha256': sha256_file(feature_manifest),
              'feature_population_identity_sha256': canonical_sha(dataset.features.spec),
              'stats_scenes': _freeze_list(dataset.metadata, indices, 'ordinary_input_normalization'),
              'teacher_labels_used': False}
    result.update(input_statistics(dataset, indices))
    result['identity_sha256'] = canonical_sha(result)
    out = Path(out).resolve()
    atomic_json(out / 'BASE_INPUT_STATISTICS.json', result)
    atomic_json(out / 'TRAIN_BEHAVIOR_PANEL.json', build_train_behavior_panel(dataset.metadata, panel_count))
    return result


def calibrate(feature_manifest, targets_manifest, out, *, bridge=None, device='cuda', stats_count=4096,
              calibration_count=2048, panel_count=8192, calibration_seed=20261027, statistics_only=False):
    if not statistics_only and bridge is None:
        raise ValueError('A real frozen native bridge is required for complete calibration')
    combined = {'feature_manifest': str(Path(feature_manifest).resolve()), 'targets_manifest': str(Path(targets_manifest).resolve())}
    full = SceneDataset(combined, 'train', 'full')
    gain1 = SceneDataset(combined, 'train', 'gain1')
    if len(full) != 85813:
        raise ValueError('Formal calibration requires complete Train85813')
    stats_ix = deterministic_train_stats_indices(full.metadata, stats_count)
    mu_ix = deterministic_log_coverage_indices(gain1.metadata, calibration_count, repaired_only=True)
    panel = build_train_behavior_panel(full.metadata, panel_count)
    result = {'schema': 'nrkd_train_calibration_v1', 'population': 'train', 'teacher_origin': 'local_refit',
              'status': 'INPUT_STATISTICS_ONLY' if statistics_only else 'COMPLETE',
              'feature_manifest_sha256': sha256_file(feature_manifest), 'targets_manifest_sha256': sha256_file(targets_manifest),
              'stats_scenes': _freeze_list(full.metadata, stats_ix, 'ordinary_input_normalization'),
              'mu_scenes': _freeze_list(gain1.metadata, mu_ix, 'gain1_nonzero_transition_calibration'),
              'teacher_identity': full.metadata[0]['teacher_identity']}
    precomputed_path = Path(out).resolve() / 'BASE_INPUT_STATISTICS.json'
    if precomputed_path.is_file():
        precomputed = read_json(precomputed_path)
        if precomputed.get('feature_population_identity_sha256') != canonical_sha(full.features.spec) or precomputed.get('stats_scenes') != result['stats_scenes']:
            raise ValueError('Precomputed Train input statistics have a different identity')
        for key in ('token_mean', 'token_std', 'state_mean', 'state_std', 'input_std_floor', 'token_std_floor_channels', 'state_std_floor_channels', 'valid_tokens_in_statistics'):
            result[key] = precomputed[key]
        result['base_input_statistics_sha256'] = sha256_file(precomputed_path)
    else:
        result.update(input_statistics(full, stats_ix))
    result.update(h_mse_scale(full))
    if not statistics_only:
        result.update(transition_statistics(gain1, mu_ix, bridge, device=device, calibration_seed=calibration_seed))
    result['identity_sha256'] = canonical_sha(result)
    out = Path(out).resolve()
    path = out / ('INPUT_STATISTICS.json' if statistics_only else 'CALIBRATION.json')
    atomic_json(out / 'TRAIN_BEHAVIOR_PANEL.json', panel)
    atomic_json(path, result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--train-only', action='store_true', required=True)
    p.add_argument('--bridge-factory', help='module:function(paths=...,device=...) -> frozen bridge')
    p.add_argument('--device', default='cuda')
    p.add_argument('--base-input-only', action='store_true', help='Prepare ordinary Train statistics/panel without teacher labels')
    p.add_argument('--statistics-only', action='store_true', help='Write partial input statistics only; never a complete calibration')
    p.add_argument('--out', required=True)
    args = p.parse_args(argv)
    import yaml
    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text())
    if 'extends' in cfg:
        raise ValueError('Use fully expanded effective config for calibration')
    paths = cfg['paths']
    if args.base_input_only:
        result = prepare_input_statistics(paths['feature_manifest'], args.out)
        print(json.dumps({'status': result['status'], 'identity_sha256': result['identity_sha256']}))
        return
    bridge = None
    factory = args.bridge_factory or paths.get('bridge_factory')
    if not args.statistics_only:
        if not factory:
            raise ValueError('Complete calibration needs configured real bridge_factory')
        module, name = factory.split(':', 1)
        bridge = getattr(importlib.import_module(module), name)(paths=paths, device=args.device)
    result = calibrate(paths['feature_manifest'], paths['targets_manifest'], args.out,
                       bridge=bridge, device=args.device, statistics_only=args.statistics_only)
    print(json.dumps({'status': result['status'], 'identity_sha256': result['identity_sha256']}))


if __name__ == '__main__':
    main()
