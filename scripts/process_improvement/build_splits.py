"""Reconstruct official navtrain, independent of pre-existing cache membership.

Scene legality follows SceneFilter(history=4,future=10,interval=1,route required).
Only observation files actually consumed by the native front-camera model are
required. Missing metric/hidden caches are repair jobs, never exclusions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import pickle
import yaml
from scripts.process_improvement.common import *


def legacy(name):
    from scripts.opd.splits import population
    rows = population(name)
    for row in rows:
        row['observation_path'] = str(HISTORY / 'observations' / name / f"{row['token']}.pt")
    return rows


def scan(job):
    path, tokens, sensor, metric_root, output = job
    path = Path(path)
    if not path.is_file():
        return [], [], dict(missing_log=1), dict(path=str(path), missing=True)
    raw = path.read_bytes()
    frames = pickle.loads(raw)
    counts = Counter()
    rows, excluded = [], []
    for i, frame in enumerate(frames):
        token = frame['token']
        if token not in tokens:
            continue
        counts['declared_token_occurrences'] += 1
        reason = None
        start = i - 3
        if start < 0:
            reason = 'insufficient_history'
        elif i + 10 >= len(frames):
            reason = 'insufficient_required_future'
        elif not frame.get('roadblock_ids'):
            reason = 'no_route'
        elif not Path('/mnt/navsim/maps', frame['map_location']).is_dir():
            reason = 'missing_map'
        elif not frame.get('cams', {}).get('CAM_F0', {}).get('data_path'):
            reason = 'missing_observation_metadata'
        elif not Path(sensor, frame['cams']['CAM_F0']['data_path']).is_file():
            reason = 'missing_current_front_image'
        elif any(any(k not in f for k in ('ego2global_translation', 'ego2global_rotation', 'ego_dynamic_state', 'driving_command'))
                 for f in frames[start:i+1]):
            reason = 'missing_historical_ego_inputs'
        if reason:
            excluded.append(dict(token=token, log_name=path.stem, reason=reason))
            counts[reason] += 1
            continue
        metric = Path(metric_root, path.stem, 'unknown', token, 'metric_cache.pkl')
        counts['metric_cache_present' if metric.is_file() else 'metric_cache_rebuild_required'] += 1
        rows.append(dict(token=token, log_name=path.stem, driver_record=path.stem.rsplit('_', 2)[0],
                         metadata_path=str(path), sensor_root=sensor, start_idx=start, frame_count=14,
                         split='navtrain', observation_seconds=5.0,
                         metric_cache_path=str(metric) if metric.is_file() else str(Path(output, 'metric_cache', token[:2], token+'.pkl')),
                         observation_path=str(Path(output, 'observations', CONTRACT[:12], token[:2], token+'.pt'))))
    return rows, excluded, dict(counts), dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), frames=len(frames))


def whole_prefix(groups, order, target):
    selected = []
    for parent in order:
        if len(selected) >= target:
            break
        selected.extend(groups[parent])
    return selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if (ROOT/'manifests/identity.json').exists():
        if not args.resume:
            raise FileExistsError('Immutable population already exists; use --resume to verify it')
        record = read(ROOT/'manifests/identity.json')
        for name, entry in record['populations'].items():
            assert sha(ROOT/'manifests'/f'{name}.json') == entry['sha256']
        print('IMMUTABLE_POPULATIONS_VERIFIED', record['populations'])
        return
    cfg = config()
    source = SOURCE/cfg['official_filter']
    declared = yaml.safe_load(source.read_text())
    assert (declared['num_history_frames'], declared['num_future_frames'], declared['frame_interval'], declared['has_route']) == (4, 10, 1, True)
    official = set(declared['tokens'])
    old = {name: legacy(name) for name in ['train', 'cal', 'dev', 'navtest']}
    metric_root = Path(old['train'][0]['metric_cache_path']).parents[3]
    jobs = [(str(Path(cfg['raw_logs'], name+'.pkl')), official, cfg['sensor_root'], str(metric_root), str(ROOT))
            for name in sorted(set(declared['log_names']))]
    all_rows, excluded, versions = [], [], []
    stats = Counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for index, (rows, exclusions, counts, version) in enumerate(pool.map(scan, jobs, chunksize=1)):
            all_rows.extend(rows)
            excluded.extend(exclusions)
            stats.update(counts)
            versions.append(version)
            if (index+1) % 50 == 0:
                print('POPULATION', index+1, len(jobs), len(all_rows), dict(stats), flush=True)
    occurrences = Counter(r['token'] for r in all_rows)
    # Do not silently choose a duplicate occurrence (the native dict loader would overwrite).
    duplicates = {token for token, count in occurrences.items() if count > 1}
    if duplicates:
        write(ROOT/'audit/duplicate_tokens.json', [r for r in all_rows if r['token'] in duplicates])
        raise ValueError('Ambiguous official tokens; inspect duplicate provenance before proceeding')
    observed = {r['token'] for r in all_rows} | {r['token'] for r in excluded}
    excluded.extend(dict(token=token, reason='official_token_not_in_available_declared_logs') for token in sorted(official-observed))
    cache_index = {r['token']: r['observation_path'] for rows in old.values() for r in rows}
    metric_index = {r['token']: r['metric_cache_path'] for rows in old.values() for r in rows}
    for row in all_rows:
        if row['token'] in cache_index and Path(cache_index[row['token']]).is_file():
            row['observation_path'] = cache_index[row['token']]
        if row['token'] in metric_index and Path(metric_index[row['token']]).is_file():
            row['metric_cache_path'] = metric_index[row['token']]
    groups = defaultdict(list)
    for row in sorted(all_rows, key=lambda r: (r['log_name'], r['start_idx'])):
        groups[row['driver_record']].append(row)
    old_parents = {name: {r['driver_record'] for r in rows} for name, rows in old.items()}
    dev_parents, cal_parents = old_parents['dev'], set(old_parents['cal'])
    available_cal = sorted(set(groups)-dev_parents-cal_parents-old_parents['train'], key=lambda s: digest([STUDY, 'cal-parent', s]))
    for parent in available_cal:
        if len(cal_parents) >= 20 and sum(len(groups[p]) for p in cal_parents) >= 3000:
            break
        cal_parents.add(parent)
    train_parents = set(groups)-dev_parents-cal_parents
    assert not train_parents & (cal_parents | dev_parents)
    assert not cal_parents & dev_parents
    order = sorted(train_parents, key=lambda s: digest([STUDY, 'train-parent-order', s]))
    train_full = sum([groups[p] for p in order], [])
    # All scenes of reserved calibration parents are withheld, including adjacent chunks.
    cal_order = sorted(cal_parents, key=lambda s: digest([STUDY, 'cal-purpose', s]))
    cal_parts = {purpose: sum([groups[p] for p in cal_order[i::3]], [])
                 for i, purpose in enumerate(['cal_tune', 'cal_confirm', 'cal_promotion'])}
    by_token = {r['token']: r for r in all_rows}
    legacy_retained = [by_token[r['token']] for r in old['train'] if r['token'] in by_token and r['driver_record'] in train_parents]
    removed_legacy = [dict(r, reason='parent_reserved' if r['driver_record'] not in train_parents else 'unavailable_native_scene')
                      for r in old['train'] if r['token'] not in {v['token'] for v in legacy_retained}]
    # Dev is frozen exactly; reserve all its parents without expanding the audit sample.
    populations = dict(train_full=train_full, train_25k=whole_prefix(groups, order, 25000),
                       train_50k=whole_prefix(groups, order, 50000), legacy_4k=legacy_retained,
                       dev=old['dev'], navtest=old['navtest'], cal=sum(cal_parts.values(), []), **cal_parts)
    populations['diagnostic_train'] = sorted(train_full, key=lambda r: digest([STUDY, 'diagnostic', r['token']]))[:512]
    populations['diagnostic_cal'] = sorted(cal_parts['cal_tune'], key=lambda r: digest([STUDY, 'diagnostic-cal', r['token']]))[:512]
    details = {}
    for name, rows in populations.items():
        write(ROOT/'manifests'/f'{name}.json', rows)
        details[name] = dict(scenes=len(rows), logs=len({r['log_name'] for r in rows}),
                             parents=len({r['driver_record'] for r in rows}), sha256=sha(ROOT/'manifests'/f'{name}.json'))
    write(ROOT/'manifests/source_versions.json', versions)
    write(ROOT/'manifests/exclusions.json', excluded)
    write(ROOT/'manifests/legacy_removed.json', removed_legacy)
    write(ROOT/'manifests/parent_splits.json', dict(train=order, dev=sorted(dev_parents),
          **{purpose: cal_order[i::3] for i,purpose in enumerate(cal_parts)}))
    write(ROOT/'manifests/identity.json', dict(official_filter_sha256=sha(source), official_token_entries=len(declared['tokens']),
          official_unique_tokens=len(official), official_log_entries=len(declared['log_names']), stats=dict(stats),
          legal_scenes=len(all_rows), excluded_scenes=len(excluded), duplicate_valid_tokens=len(duplicates), populations=details,
          cache_absence_is_not_exclusion=True, base_training_exposure_unknown=True, navtest_previously_viewed=True,
          source_versions_sha256=sha(ROOT/'manifests/source_versions.json'), exclusions_sha256=sha(ROOT/'manifests/exclusions.json')))
    print('POPULATIONS_COMPLETE', details, flush=True)


if __name__ == '__main__':
    main()
