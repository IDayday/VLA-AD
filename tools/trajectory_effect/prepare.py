"""Freeze complete-log membership before looking at any new policy outcomes."""
from collections import defaultdict
from pathlib import Path
import argparse

from .common import *


def main():
    require_host()
    all_rows = read(TRAIN_MANIFEST)
    dev = read(AUDIT_MANIFEST)
    dev_tokens = {r['token'] for r in dev}
    audit = [r for r in all_rows if r['token'] in dev_tokens]
    assert len(audit) == len(dev) == 1696
    assert len({r['log_name'] for r in audit}) == 16
    audit_parents = {driver_record(r['log_name']) for r in audit}
    logs = defaultdict(list)
    for row in all_rows:
        if driver_record(row['log_name']) not in audit_parents:
            logs[row['log_name']].append(row)
    ordered = sorted(logs, key=lambda log: digest(['split-v1', 20261019, log]))
    cal, fit, cal_parents = [], [], set()
    for log in ordered:
        if len(cal) < 512:
            cal.extend(logs[log]); cal_parents.add(driver_record(log))
    for log in ordered:
        if driver_record(log) not in cal_parents and len(fit) < 4096:
            fit.extend(logs[log])
    groups = dict(D_fit=fit, D_cal=cal, D_audit=audit)
    for a, aa in groups.items():
        for b, bb in groups.items():
            if a != b:
                assert not ({driver_record(r['log_name']) for r in aa} &
                            {driver_record(r['log_name']) for r in bb})
    counts = {}
    for split, rows in groups.items():
        for r in rows:
            paths = list((METRIC_ROOT / r['log_name']).glob(f"*/{r['token']}/metric_cache.pkl"))
            if len(paths) != 1:
                raise FileNotFoundError((r['token'], paths))
            r['metric_cache_path'] = str(paths[0])
            r['driver_record'] = driver_record(r['log_name'])
        rows.sort(key=lambda r: (r['log_name'], r['start_idx'], r['token']))
        dest = ROOT / 'manifests' / (split + '.json')
        if dest.exists() and read(dest) != rows:
            raise ValueError(f'Existing split differs: {split}')
        atomic_json(dest, rows)
        counts[split] = dict(scenes=len(rows), logs=len({r['log_name'] for r in rows}),
                             driver_records=len({r['driver_record'] for r in rows}), sha256=sha(dest))
    # Pilot scenes come only from D_fit. Stratification follows its existing scores
    # if available; this initial deterministic list is not an outcome-selected cohort.
    pilot = sorted(fit, key=lambda r: digest(['pilot-v1', r['token']]))[:64]
    atomic_json(ROOT / 'manifests/pilot.json', pilot)
    families = [dict(id=f'update_{k:02d}', split=('fit' if k < 6 else 'cal' if k < 9 else 'audit'),
                     source_kind=('natural' if k % 2 == 0 else 'low_score_reward_diverse'),
                     train_seed=7300+k, updates=[1, 4]) for k in range(12)]
    atomic_json(ROOT / 'manifests/update_families.json', families)
    atomic_json(ROOT / 'manifests/splits_identity.json', dict(
        counts=counts, all_official_navtrain_scenes=len(all_rows),
        sources={str(p): sha(p) for p in (TRAIN_MANIFEST, AUDIT_MANIFEST)},
        no_overlap_units=['token', 'complete_log', 'parent_date_time_vehicle_driving_record'],
        audit_excluded_parent_records=sorted(audit_parents),
        previously_seen_by_base='All splits belong to official navtrain; no claim of base-model-unseen data.',
        navtest_opened=False, purpose='pilot and membership preparation; formal protocol awaits real cost measurements'))
    print(counts, flush=True)


if __name__ == '__main__': main()
