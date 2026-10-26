"""Build compact full/gain1 labels from the NEW frozen local K8 teacher only."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from .asset_identity import (atomic_json, canonical_sha, load_library_bundle, population_rows,
                             read_json, resolve_ref, sha256_file)
from .data import FeatureStore


def score_selection(q_points, saved_choice=None):
    """Float64 POINT units; first-column argmax implements exact Base-first ties."""
    q = np.asarray(q_points)
    if q.dtype != np.float64 or q.ndim != 2 or q.shape[1] != 9 or not np.isfinite(q).all():
        raise ValueError('Expected finite FP64 point scores [N,9]')
    if np.any(q < -1e-8) or np.any(q > 100 + 1e-8):
        raise ValueError('PDMS points outside [0,100]')
    choice = q.argmax(1)
    if saved_choice is not None and not np.array_equal(choice, np.asarray(saved_choice)):
        raise ValueError('Saved choice does not match fixed-scene S0-S4 argmax')
    gain = q[np.arange(len(q)), choice] - q[:, 0]
    return choice.astype(np.int64), gain, gain > 1.0


def read_local_selection(bundle, population):
    b = bundle['bundle']
    ref = b.get('selections', {}).get(population)
    ref_root = Path(bundle['bundle_path']).parent
    registry_sha = None
    registry_value = None
    if 'selection_registry' in b:
        registry_path = resolve_ref(b['selection_registry'], ref_root, verify=False)
        registry = read_json(registry_path)
        registry_value = dict(registry)
        recorded_digest = registry_value.pop('identity_sha256', None)
        if registry.get('schema') != 'nrkd_local_selection_registry_v1' or recorded_digest != canonical_sha(registry_value):
            raise ValueError('Mutable selection evidence registry self-digest mismatch')
        if not b.get('teacher_identity') or registry.get('teacher_identity') != b['teacher_identity']:
            raise ValueError('Selection registry belongs to a different local teacher')
        if registry.get('source_checkpoint_sha256') != b['source_checkpoint_sha256']:
            raise ValueError('Selection registry checkpoint mismatch')
        ref = registry.get('populations', {}).get(population)
        ref_root = registry_path.parent
        registry_sha = sha256_file(registry_path)
    if ref is None:
        raise FileNotFoundError(f'New local {population} candidate scores/choices are not frozen yet')
    if not isinstance(ref, dict) or ref.get('units') != 'points':
        raise ValueError('Local selection must explicitly declare units=points')
    if population != 'train' and not ref.get('diagnostic_only', False):
        raise ValueError('Validation/Test oracle selections must be diagnostic only')
    path = resolve_ref(ref, ref_root)
    with np.load(path, allow_pickle=False) as z:
        tokens = z['tokens'].astype(str).tolist()
        q = np.array(z['q_points'])
        saved = np.array(z['choice'])
        indices = np.array(z['indices']) if 'indices' in z else None
        scored = {k: np.array(z[k]) for k in ('selected_actual_role_shifts', 'selected_actual_RMS', 'selected_cap_RMS') if k in z}
    if scored:
        if set(scored) != {'selected_actual_role_shifts', 'selected_actual_RMS', 'selected_cap_RMS'}:
            raise ValueError('Scored actual shifts/RMS/caps must be exported together')
        a = scored['selected_actual_role_shifts']
        if a.dtype != np.float32 or a.shape != (len(tokens), 2, 384) or not np.isfinite(a).all() or not np.array_equal(a[:, 0], a[:, 1]):
            raise ValueError('Scored actual role shifts must be identical-role finite FP32[N,2,384]')
        for key in ('selected_actual_RMS', 'selected_cap_RMS'):
            if scored[key].shape != (len(tokens),) or not np.isfinite(scored[key]).all() or (scored[key] < 0).any():
                raise ValueError('Invalid scored actual RMS/cap')
    if len(set(tokens)) != len(tokens) or len(tokens) != len(q):
        raise ValueError('Invalid local selection token coverage')
    choice, gain, keep = score_selection(q, saved)
    return {'tokens': tokens, 'q_points': q, 'choice': choice, 'gain_points': gain,
            'gain1_keep': keep, 'indices': indices, 'source_sha256': sha256_file(path),
            'selection_registry_sha256': registry_sha, 'selection_registry_snapshot': registry_value, **scored}


def verify_historical_matrices(frozen_repo=None):
    """Audit the 11 original matrices as HISTORICAL evidence; never training labels."""
    import csv
    import gzip
    root = Path(frozen_repo or Path(__file__).resolve().parents[1])
    folder = root / 'decisions/train/D_C8_double'
    historical_manifest = read_json(folder / 'MANIFEST.json')
    expected_hashes = {Path(r['path']).name: r['sha256'] for r in historical_manifest['chunks']}
    if sha256_file(folder / 'choices.csv.gz') != historical_manifest['choice_table']['sha256']:
        raise ValueError('Historical CSV SHA mismatch')
    matrices = sorted(folder.glob('pdms_*.npz'))
    if len(matrices) != 11:
        raise ValueError('Historical audit requires all 11 original matrices')
    all_indices, all_q, saved_choices = [], [], []
    hashes = []
    for path in matrices:
        if sha256_file(path) != expected_hashes.get(path.name):
            raise ValueError('Historical matrix SHA mismatch')
        with np.load(path, allow_pickle=False) as z:
            all_indices.append(np.array(z['indices']))
            # The historical matrices are fractions. The ONE conversion is FP64.
            all_q.append(100.0 * np.asarray(z['mean_PDMS_score'], dtype=np.float64))
            saved_choices.append(np.array(z['choice']))
        hashes.append({'name': path.name, 'sha256': sha256_file(path)})
    indices, q, saved = np.concatenate(all_indices), np.concatenate(all_q), np.concatenate(saved_choices)
    choice, gain, repair = score_selection(q, saved)
    if len(indices) != 85813 or len(set(indices.tolist())) != 85813:
        raise ValueError('Historical index coverage mismatch')
    with gzip.open(folder / 'choices.csv.gz', 'rt') as f:
        rows = list(csv.DictReader(f))
    by_index = {int(r['index']): r for r in rows}
    if set(by_index) != set(indices.tolist()):
        raise ValueError('Historical matrix/CSV index mismatch')
    for i, c in zip(indices, choice):
        if int(by_index[int(i)]['choice']) != int(c):
            raise ValueError('Historical choice mismatch')
    counts = np.bincount(choice, minlength=9).tolist()
    expected = [40154, 1059, 1174, 2000, 3215, 2667, 17265, 14976, 3303]
    if counts != expected or int(repair.sum()) != 31511:
        raise ValueError('Historical counts no longer match frozen evidence')
    full = q[np.arange(len(q)), choice]
    gain1 = np.where(repair, full, q[:, 0])
    stats = {'identity': 'HISTORICAL_PPU_ONLY_NOT_LOCAL_TRAINING_TARGETS', 'scenes': len(q),
             'choice_counts': counts, 'gain1_repair_count': int(repair.sum()),
             'base_points': float(q[:, 0].mean()), 'full_points': float(full.mean()),
             'gain1_points': float(gain1.mean()), 'matrix_hashes': hashes}
    for field, expected_value in [('base_points', 94.9163155), ('full_points', 97.3053145), ('gain1_points', 97.2233742)]:
        if not np.isclose(stats[field], expected_value, atol=5e-7, rtol=0):
            raise ValueError(f'Historical {field} mismatch')
    return stats


def compact_actual_shift(base, image, text, result):
    """No role averaging: actual image/text shifts must be exactly identical."""
    import torch
    shifts = result['actual_role_shifts']
    if shifts.shape != (1, 2, 384) or shifts.dtype != torch.float32:
        raise ValueError('Expected actual_role_shifts FP32[1,2,384]')
    if not torch.equal(shifts[:, 0], shifts[:, 1]):
        raise ValueError('Current protocol requires identical image/text actual shifts')
    delta = shifts[0, 0].detach()
    reconstructed = base + (image | text).unsqueeze(-1).to(base.dtype) * delta
    if not torch.equal(reconstructed, result['interfaces']):
        raise ValueError('Compact actual shift does not exactly reconstruct native teacher')
    valid = (image | text)[0]
    if bool((~valid).any()) and not torch.equal(reconstructed[:, ~valid], base[:, ~valid]):
        raise ValueError('Teacher modified native padding')
    rms = (reconstructed[:, valid].double() - base[:, valid].double()).square().mean().sqrt()
    cap = result['cap_RMS'][0]
    tolerance = torch.maximum(cap * 1e-4, cap.new_tensor(1e-7))
    if not bool(torch.isfinite(rms)) or bool(rms > cap + tolerance):
        raise ValueError('Native post-rounding RMS cap violated')
    return delta.cpu().numpy().copy(), float(rms), float(cap)


def build_population(feature_manifest, bundle_path, population, out, *, target_versions=('full', 'gain1'), diagnostic_only=False, device='cpu'):
    import torch
    if population != 'train' and not diagnostic_only:
        raise ValueError('Validation/Test targets are diagnostic only; never training labels')
    if set(target_versions) - {'full', 'gain1'}:
        raise ValueError('Unknown target version')
    bundle = load_library_bundle(bundle_path)
    selection = read_local_selection(bundle, population)
    features = FeatureStore(feature_manifest, population)
    by_token = {t: i for i, t in enumerate(selection['tokens'])}
    if set(by_token) != set(features.by_token):
        raise ValueError('Local selection/native feature token coverage mismatch')
    feature_checkpoint = features.manifest.get('source_checkpoint_sha256')
    if feature_checkpoint != bundle['identity']['source_checkpoint_sha256']:
        raise ValueError('Teacher/Base feature checkpoint mismatch')
    folder = Path(out).resolve() / population
    folder.mkdir(parents=True, exist_ok=True)
    identity = {'teacher': bundle['identity'], 'selection_sha256': selection['source_sha256'],
                'feature_manifest_sha256': sha256_file(feature_manifest), 'feature_population_identity_sha256': canonical_sha(features.spec), 'target_versions': list(target_versions),
                'population': population, 'diagnostic_only': diagnostic_only, 'execution_device_type': str(device).split(':')[0]}
    identity_sha = canonical_sha(identity)
    if selection['selection_registry_snapshot'] is not None:
        # Snapshot append-only evidence separately: adding another population to
        # the registry cannot change this population's already frozen targets.
        snapshot_path = folder / 'SELECTION_REGISTRY_USED.json'
        if not snapshot_path.exists():
            atomic_json(snapshot_path, {'registry': selection['selection_registry_snapshot'],
                        'source_registry_sha256': selection['selection_registry_sha256']})
    complete = folder / 'COMPLETE.json'
    if complete.exists():
        done = read_json(complete)
        if done['identity_sha256'] != identity_sha:
            raise ValueError('Completed target artifact has different identity')
        return done['population_spec'], bundle['identity']
    temporary = {v: folder / (v + '_shifts.partial.npy') for v in target_versions}
    shifts = {v: np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(features), 384)) for v, path in temporary.items()}
    records = []
    try:
        for n, row in enumerate(features.rows):
            i = by_token[row['token']]
            arrays = features.read(n)
            if selection['indices'] is not None and row.get('global_index') is not None and int(selection['indices'][i]) != int(row['global_index']):
                raise ValueError('Local selection global index/token mismatch')
            base = torch.from_numpy(np.array(arrays['base_post'], copy=True)).unsqueeze(0).to(device)
            image = torch.from_numpy(np.array(arrays['image_mask'], copy=True)).unsqueeze(0).to(device)
            text = torch.from_numpy(np.array(arrays['text_mask'], copy=True)).unsqueeze(0).to(device)
            choice = int(selection['choice'][i])
            target_kwargs = ({'raw_aux24': np.asarray(arrays['state24'], np.float32)} if getattr(bundle['library'], 'requires_aux24', False) else {})
            result = bundle['library'].targets(base, image, text, [choice], **target_kwargs)
            shift, rms, cap = compact_actual_shift(base, image, text, result)
            if 'selected_actual_role_shifts' in selection:
                scored_shift = selection['selected_actual_role_shifts'][i, 0]
                if not np.array_equal(shift, scored_shift):
                    raise ValueError(f'Native recomputation differs from the actually scored local shift: {row["token"]}; use its exact native execution device/protocol')
                if not np.isclose(rms, selection['selected_actual_RMS'][i], rtol=1e-10, atol=1e-12) or cap != float(selection['selected_cap_RMS'][i]):
                    raise ValueError('Recomputed actual target RMS/cap differs from scored teacher')
                shift = scored_shift.copy()
            if 'full' in shifts:
                shifts['full'][n] = shift
            keep = bool(selection['gain1_keep'][i])
            if 'gain1' in shifts:
                shifts['gain1'][n] = shift if keep else 0.0
            rec = {k: row[k] for k in ('token', 'log_id', 'global_index', 'population', 'actual_Base_interface_sha256', 'token_layout_sha256')}
            rec.update(original_choice=choice, gain_points=float(selection['gain_points'][i]),
                       full_is_zero=bool(not np.any(shift)), gain1_is_zero=bool(not keep or not np.any(shift)),
                       gain1_keep=keep, actual_target_RMS=rms, cap_RMS=cap,
                       base_mean_points=float(selection['q_points'][i, 0]),
                       full_teacher_mean_points=float(selection['q_points'][i, choice]),
                       gain1_teacher_mean_points=float(selection['q_points'][i, choice if keep else 0]),
                       source_checkpoint_sha256=bundle['identity']['source_checkpoint_sha256'],
                       library_sha256=bundle['identity']['library_sha256'], basis_sha256=bundle['identity']['basis_sha256'])
            records.append(rec)
            if (n + 1) % 256 == 0:
                print(json.dumps({'population': population, 'targets_completed': n + 1, 'total': len(features)}), flush=True)
    finally:
        for a in shifts.values():
            a.flush()
    version_specs = {}
    for v, a in shifts.items():
        path = folder / (v + '_shifts.npy')
        temporary[v].replace(path)
        version_specs[v] = {'shifts': {'path': str(path), 'sha256': sha256_file(path)}, 'zero_targets': sum(r[f'{v}_is_zero'] for r in records)}
    index = folder / 'INDEX.jsonl'
    temp = index.with_suffix('.tmp')
    with temp.open('w') as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    temp.replace(index)
    selected_q = selection['q_points'][np.arange(len(by_token)), selection['choice']]
    gain1_q = np.where(selection['gain1_keep'], selected_q, selection['q_points'][:, 0])
    stats = {'count': len(records), 'original_choice_counts': np.bincount(selection['choice'], minlength=9).tolist(),
             'gain1_repair_count': int(selection['gain1_keep'].sum()), 'base_points': float(selection['q_points'][:, 0].mean()),
             'full_points': float(selected_q.mean()), 'gain1_points': float(gain1_q.mean()),
             'identity': 'NEW_LOCAL_FROZEN_TEACHER', 'historical_counts_are_not_constraints': True}
    spec = {'count': len(records), 'diagnostic_only': diagnostic_only, 'identity': identity, 'identity_sha256': identity_sha,
            'index': {'path': str(index), 'sha256': sha256_file(index)}, 'versions': version_specs, 'statistics': stats}
    atomic_json(complete, {'status': 'COMPLETE', 'identity_sha256': identity_sha, 'population_spec': spec})
    return spec, bundle['identity']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths', required=True)
    p.add_argument('--library', default='D_C8_double')
    p.add_argument('--population', choices=['train', 'validation', 'test'], default='train')
    p.add_argument('--target-versions', nargs='+', choices=['full', 'gain1'], default=['full', 'gain1'])
    p.add_argument('--diagnostic-only', action='store_true')
    p.add_argument('--score-frozen-candidates', action='store_true')
    p.add_argument('--noise-indices', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--device', default='cpu')
    p.add_argument('--historical-audit-only', action='store_true')
    p.add_argument('--out', required=True)
    args = p.parse_args(argv)
    paths = read_json(args.paths)
    if args.library != 'D_C8_double':
        raise ValueError('Round1 authorizes only the new local D_C8_double library')
    out = Path(args.out).resolve()
    population_out = out if out.name == args.population else out / args.population
    audit = verify_historical_matrices(paths.get('frozen_repo'))
    # Train/Test can build concurrently. Their audits are independent artifacts,
    # avoiding two writers racing on a shared temporary filename.
    atomic_json(population_out / 'HISTORICAL_PPU_AUDIT.json', audit)
    if args.historical_audit_only:
        print(json.dumps(audit, sort_keys=True))
        return
    if args.score_frozen_candidates:
        # Real scoring is delegated to the registered evaluator, never fabricated
        # from old decisions or from saved labels of another teacher.
        scorer = paths.get('frozen_candidate_scorer')
        if not scorer:
            raise ValueError('Configure frozen_candidate_scorer=module:function to score the new local validation candidates')
        import importlib
        module, function = scorer.split(':', 1)
        getattr(importlib.import_module(module), function)(paths=paths, population=args.population,
            noise_indices=args.noise_indices, output_dir=str(out), diagnostic_only=args.diagnostic_only)
    feature_manifest = paths['feature_manifest']
    # One immutable manifest per population. Independent Validation/Test oracle
    # work must never mutate the Train manifest bound into Adam checkpoints.
    spec, identity = build_population(feature_manifest, paths['teacher_bundle'], args.population, population_out.parent,
        target_versions=args.target_versions, diagnostic_only=args.diagnostic_only, device=args.device)
    path = population_out / 'MANIFEST.json'
    manifest = {'schema': 'nrkd_compact_targets_v1', 'teacher_origin': 'local_refit',
                'teacher_identity': identity, 'immutable_population': args.population,
                'populations': {args.population: spec}}
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError('Refusing to mutate an already frozen population target manifest')
    else:
        atomic_json(path, manifest)
    print(path)


if __name__ == '__main__':
    main()
