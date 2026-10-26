"""Finite original GT stage generator plus new-root configuration task queue."""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import gzip
import json
import time
from pathlib import Path

from gt_numeric import (G5, STAGES, atomic, arrays_at, bind_function, descriptor,
                        digest, import_shims, read, seed, sha, stamp)


def lock_root(root):
    (root / 'runs').mkdir(parents=True, exist_ok=True)
    stream = (root / 'runs/CONTROLLER.lock').open('a+')
    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return stream


def configuration_tasks(root, stage):
    tasks = read(root / 'runs' / stage / 'TASKS.json')
    for scene in tasks:
        for position, config in enumerate(scene['configs']):
            task_id = f"{scene['id']}__c{position:04d}"
            task = {'id': task_id, 'scene_id': scene['id'], 'kind': stage,
                    'index': scene['index'], 'config': config,
                    'position': position, 'scene_task_sha256': digest(scene)}
            path = root / 'runs' / stage / 'configuration_tasks' / (task_id + '.json')
            if path.exists():
                if read(path) != task:
                    raise ValueError('Persisted task differs from original stage generator')
            else:
                atomic(path, task)
    atomic(root / 'runs' / stage / 'CONFIGURATION_QUEUE.json',
           {'stage': stage, 'scene_tasks': len(tasks), 'configuration_tasks': sum(len(t['configs']) for t in tasks),
            'claim_granularity': 'one original configuration', 'configuration_boundary_resume': True,
            'generated_at': stamp()})


def initialize(args, root):
    if (root / 'runs/QUEUE.json').exists():
        raise ValueError('GT root already initialized; use advance or status')
    source = Path(args.source_root).resolve()
    archive_path = source / '1022-gt_assisted_representation_optimization/manifests/SCENES.json'
    archived = [r for r in read(archive_path) if r['subset'] == 'exploration']
    if len(archived) != 1850:
        raise ValueError('Original Train failure population must contain exactly 1850 IDs')
    manifest = Path(args.train_manifest).resolve()
    with gzip.open(manifest, 'rt') if manifest.suffix == '.gz' else manifest.open() as stream:
        train = {r['token']: r for r in csv.DictReader(stream)}
    if len(train) != 85813:
        raise ValueError('Full frozen Train population must contain 85813 scenes')
    paths = read(args.paths)
    metric_registry = read(args.metric_registry)
    metric_registry_sha256 = sha(args.metric_registry)
    rows = []
    for old in archived:
        current = train[old['token']]
        if int(current['index']) != int(old['global_index']) or current['session'] != old['session']:
            raise ValueError('Original failure token/index/session mapping changed')
        metric = metric_registry['entries'][old['token']]
        if metric['reference_sha256'] != current['metric_sha256'] or metric['relative'] != current['metric_rel']:
            raise ValueError('Local metric registry does not match the frozen scene/metric identity')
        row = dict(old, population='train', legacy_best=None, legacy_configurations={},
                   metric_cache=str(Path(paths['navtrain_metric']) / current['metric_rel']),
                   metric_sha256=metric['local_sha256'], metric_reference_sha256=metric['reference_sha256'],
                   local_metric_registry_sha256=metric_registry_sha256, log_name=current['log_name'])
        # Retain the archived identity as evidence, while all local numerical
        # inputs and Base scores will be read from NEW/native_base.
        for key in ['raw_path', 'sensor_root', 'feature_dir', 'feature_sha256']:
            row.pop(key, None)
        rows.append(row)
    atomic(root / 'manifests/SCENES.json', rows)
    common = {'Path': Path}
    configurations, identity = bind_function(source / G5 / 'g5_runtime.py', 'configurations', common)
    tasks = [{'id': f"scene_{r['global_index']:06d}", 'kind': 'baseline',
              'index': r['global_index'], 'configs': configurations()} for r in rows]
    atomic(root / 'runs/baseline/TASKS.json', tasks)
    atomic(root / 'runs/baseline/SPEC.json', {'kind': 'baseline', 'case_count': len(tasks), 'enabled': True})
    atomic(root / 'runs/QUEUE.json', {'stages': ['baseline'], 'active': True})
    atomic(root / 'PROTOCOL.json', {'status': 'LOCAL_NEW_GT_REFIT_REGISTERED', 'created': stamp(),
           'source_root': str(source), 'source_population': str(archive_path), 'source_population_sha256': sha(archive_path),
           'train_manifest': str(manifest), 'train_manifest_sha256': sha(manifest), 'paths_sha256': sha(args.paths),
           'local_metric_registry': str(Path(args.metric_registry).resolve()), 'local_metric_registry_sha256': metric_registry_sha256,
           'original_configurations_binding': identity, 'stages': STAGES, 'train_failures': 1850,
           'selection_noise': 'actual S0 PDMS', 'transfer_noise_indices': [1, 2],
           'transfer_not_used_for_selection': True, 'no_validation_or_test_optimization': True,
           'old_teachers_or_scores_as_results': False, 'library_to_fit': 'D_C8_double only'})
    configuration_tasks(root, 'baseline')


def aggregate_receipts(root, stage):
    """Produce each original scene receipt only once every config is complete."""
    tasks = read(root / 'runs' / stage / 'TASKS.json')
    done, configurations_done, failed_terminal = 0, 0, 0
    for scene in tasks:
        results = []
        for position, config in enumerate(scene['configs']):
            receipt = root / 'runs' / stage / 'configuration_receipts' / f"{scene['id']}__c{position:04d}.json"
            if not receipt.exists():
                failure_dir = root / 'runs' / stage / 'failures' / f"{scene['id']}__c{position:04d}"
                failed_terminal += len(list(failure_dir.glob('*.json'))) >= 3
                continue
            value = read(receipt)
            if value['status'] != 'COMPLETE' or value['config_sha256'] != digest(config):
                raise ValueError('Configuration receipt identity mismatch')
            path = root / value['result_path']
            if sha(path) != value['result_sha256']:
                raise ValueError('Configuration RESULT SHA mismatch')
            results.append({'path': str(path.relative_to(root)), 'sha256': value['result_sha256']})
        configurations_done += len(results)
        if len(results) == len(scene['configs']):
            done += 1
            receipt_path = root / 'runs' / stage / 'receipts' / (scene['id'] + '.json')
            if not receipt_path.exists():
                atomic(receipt_path, {'status': 'COMPLETE', 'time': stamp(),
                                     'task_sha256': digest(scene), 'result': {'status': 'PASS', 'results': results}})
    return {'stage': stage, 'scenes_complete': done, 'scenes_total': len(tasks),
            'configurations_complete': configurations_done, 'configurations_total': sum(len(t['configs']) for t in tasks),
            'terminal_failed_configurations': failed_terminal, 'complete': done == len(tasks)}


def summarize(root, stage):
    """Retain previous choices on ties; no historical score array is read."""
    summary_path = root / 'analysis' / ('summary_' + stage) / 'SUMMARY.json'
    if summary_path.exists():
        return read(summary_path)
    progress = aggregate_receipts(root, stage)
    if not progress['complete']:
        raise ValueError('Cannot summarize an incomplete stage')
    tasks = read(root / 'runs' / stage / 'TASKS.json')
    rows = read(root / 'manifests/SCENES.json')
    if stage == 'baseline':
        by_index = {}
        for row in rows:
            by_index[row['global_index']] = {'index': row['global_index'], 'token': row['token'],
                                             'subset': 'exploration', 'perblock': {}, 'reused': False}
    else:
        previous = read(root / 'runs' / stage / 'PREVIOUS.json')
        previous_path = root / previous['summary']
        if sha(previous_path) != previous['sha256']:
            raise ValueError('Previous cumulative summary identity changed')
        by_index = {r['index']: copy.deepcopy(r) for r in read(previous_path)['cases']}
    gt_quality = []
    for scene in tasks:
        record = by_index[scene['index']]
        receipt = read(root / 'runs' / stage / 'receipts' / (scene['id'] + '.json'))
        candidates = sorted((root / value['path'] for value in receipt['result']['results']),
                            key=lambda path: (path.parent.name, str(path)))
        for path in candidates:
            current = descriptor(root, path)
            value = read(path)
            block = current['config']['block']
            if stage == 'baseline' and 'base' not in record:
                base_record = next(r for r in value['records'] if r.get('iteration') == 0)
                record['base'] = base_record['score']['score']
                gt = next(r for r in value['records'] if r['arm'] == 'GT_reference')
                gt_quality.append({'index': record['index'], 'GT_PDMS': gt['score']['score'] * 100})
            if 'selection' not in record or current['S0_score'] > record['selection']['S0_score']:
                record['selection'] = current
            if block not in record['perblock'] or current['S0_score'] > record['perblock'][block]['S0_score']:
                record['perblock'][block] = current
        if stage == 'baseline' and set(record['perblock']) != {'interface_pre', 'interface_post'}:
            raise ValueError('Baseline did not retain both original optimization locations')
    records = [by_index[r['global_index']] for r in rows]
    statistics = {'cases': len(records), 'S0_mean': sum(r['selection']['S0_score'] for r in records) / len(records),
                  'S0_gt90': sum(r['selection']['S0_score'] > .9 for r in records),
                  'pre_S0_gt90': sum(r['perblock']['interface_pre']['S0_score'] > .9 for r in records),
                  'post_S0_gt90': sum(r['perblock']['interface_post']['S0_score'] > .9 for r in records)}
    value = {'status': 'PASS_COMPLETE_CUMULATIVE_LOCAL_S0_SELECTION', 'time': stamp(), 'stage': stage,
             'cases': records, 'statistics': statistics, 'scope': '1850 original Train failure IDs',
             'selection_noise': 'actual S0 PDMS only', 'transfer_noise_indices': [1, 2],
             'transfer_not_used_for_selection': True, 'old_results_not_overwritten': True,
             'tie_policy': 'Retain previous selections on ties; lexicographic new configuration order',
             'test_used': False, 'aggregator_sha256': sha(__file__)}
    atomic(summary_path, value)
    if stage == 'baseline':
        atomic(root / 'analysis/GT_exception_interim.json', {'cases': gt_quality})
    atomic(root / 'runs' / stage / 'ACCEPTED.json', {'status': 'PASS_COMPLETE', 'summary': str(summary_path.relative_to(root)),
                                                 'sha256': sha(summary_path), 'scenes': len(records)})
    return value


def generate_next(root, source, name, previous):
    common = {'ROOT': root, 'Path': Path, 'read': read, 'sha': sha, 'atomic': atomic,
              'json': json, 'B2': root}
    minimum, minimum_identity = bind_function(source / G5 / 'g5_next_stage.py', 'minimum_original_GT_loss', common)
    main, main_identity = bind_function(source / G5 / 'g5_next_stage.py', 'main',
                                        dict(common, minimum_original_GT_loss=minimum))
    with import_shims({'g5_summarize': {'descriptor': lambda path: descriptor(root, path)},
                       'g5_delegate': {'prepare': lambda *args: None}}):
        main(name, previous)
    atomic(root / 'runs' / name / 'ORIGINAL_GENERATOR_BINDINGS.json',
           {'main': main_identity, 'minimum_original_GT_loss': minimum_identity,
            'transport_change': 'one configuration per dynamic claim; original scene receipt after all configs'})
    configuration_tasks(root, name)


def advance(root, source):
    queue = read(root / 'runs/QUEUE.json')
    stage = queue['stages'][-1]
    progress = aggregate_receipts(root, stage)
    if not progress['complete']:
        return progress
    if stage == 'export_post':
        queue['active'] = False
        atomic(root / 'runs/QUEUE.json', queue)
        atomic(root / 'GT_POST_COMPLETE.json', {'status': 'ALL1850_LOCAL_GT_POST_PAIRS_EXPORTED',
               'time': stamp(), 'train_failures': 1850, 'actual_exports': progress['configurations_complete'],
               'search_summary': 'analysis/summary_post_progress/SUMMARY.json',
               'search_summary_sha256': sha(root / 'analysis/summary_post_progress/SUMMARY.json'),
               'complete_does_not_imply_fitted_library': True})
        return {'status': 'GT_POST_COMPLETE', **progress}
    summarize(root, stage)
    position = STAGES.index(stage)
    if position + 1 == len(STAGES):
        atomic(root / 'SEARCH_COMPLETE.json', {'status': 'FULL_ORIGINAL_FINITE_GT_SEARCH_COMPLETE',
                                               'time': stamp(), 'stages': STAGES, 'train_failures': 1850,
                                               'exports_pending': True})
        summary = read(root / 'analysis/summary_post_progress/SUMMARY.json')
        tasks = [{'id': f"scene_{r['index']:06d}", 'kind': 'export_post', 'index': r['index'],
                  'configs': [r['perblock']['interface_post']]} for r in summary['cases']]
        atomic(root / 'runs/export_post/TASKS.json', tasks)
        configuration_tasks(root, 'export_post')
        queue['stages'].append('export_post')
        atomic(root / 'runs/QUEUE.json', queue)
        return {'status': 'SEARCH_COMPLETE_EXPORT_READY', **aggregate_receipts(root, 'export_post')}
    name = STAGES[position + 1]
    if not (root / 'runs' / name / 'TASKS.json').exists():
        generate_next(root, source, name, stage)
    return {'status': 'STAGE_READY', **aggregate_receipts(root, name)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['init', 'advance', 'status', 'watch'])
    parser.add_argument('--root', required=True)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--train-manifest')
    parser.add_argument('--paths')
    parser.add_argument('--metric-registry', default='/mnt/project/recogdrive-first42-a800-20261022/LOCAL_METRIC_REGISTRY.json')
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--deadline-unix', type=float)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    owner = lock_root(root)
    if args.command == 'init':
        if not args.train_manifest or not args.paths:
            parser.error('init requires --train-manifest and --paths')
        initialize(args, root)
        result = aggregate_receipts(root, 'baseline')
    elif args.command == 'advance':
        result = advance(root, Path(args.source_root).resolve())
    elif args.command == 'watch':
        if not 0 < args.poll_seconds <= 60:
            parser.error('poll-seconds must be in (0,60]')
        while read(root / 'runs/QUEUE.json')['active']:
            if (root / 'STOP_NEW_CLAIMS.json').exists() or (args.deadline_unix and time.time() >= args.deadline_unix):
                break
            result = advance(root, Path(args.source_root).resolve())
            atomic(root / 'CONTROLLER_STATUS.json', {'time': stamp(), **result})
            print(json.dumps(result, sort_keys=True), flush=True)
            if result.get('status') == 'GT_POST_COMPLETE':
                break
            time.sleep(args.poll_seconds)
        result = read(root / 'CONTROLLER_STATUS.json') if (root / 'CONTROLLER_STATUS.json').exists() else {'status': 'WAITING'}
    else:
        queue = read(root / 'runs/QUEUE.json')
        result = {'active': queue['active'], 'stages': [aggregate_receipts(root, stage) for stage in queue['stages']]}
    print(json.dumps(result, sort_keys=True), flush=True)
    owner.close()


if __name__ == '__main__':
    main()
