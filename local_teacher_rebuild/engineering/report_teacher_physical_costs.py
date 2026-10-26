"""Measured actor wall time and physical GPU interval unions, without hot edits."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from gt_numeric import atomic, read, sha, stamp


def reference(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def union_seconds(intervals):
    total, start, end = 0., None, None
    for left, right in sorted(intervals):
        if right < left:
            raise ValueError('Actor end precedes its start')
        if start is None:
            start, end = left, right
        elif left <= end:
            end = max(end, right)
        else:
            total += end - start
            start, end = left, right
    return total + (0. if start is None else end - start)


def actor_times(status_path, value):
    """Terminal launcher records drop started_at; use the actual PID log line."""
    status_path = Path(status_path)
    start, source = value.get('started_at'), 'terminal_status.started_at'
    evidence = reference(status_path)
    if not start:
        command_log = status_path.parent.parent / 'commands.log'
        pattern = re.compile(r'^\[([^]]+)\] name=' + re.escape(value['name']) + r' pid=' + str(value['pid']) + r'\s*$')
        matches = [match.group(1) for line in command_log.read_text().splitlines() if (match := pattern.match(line))]
        if len(matches) != 1:
            raise ValueError(f'No unique actual process-start PID timestamp: {status_path}')
        start, source, evidence = matches[0], 'commands.log_actual_PID_registration_timestamp', reference(command_log)
    end = value.get('ended_at')
    if not end:
        raise ValueError(f'Actual terminal actor end timestamp is missing: {status_path}')
    left = datetime.fromisoformat(start.replace('Z', '+00:00')).timestamp()
    right = datetime.fromisoformat(end.replace('Z', '+00:00')).timestamp()
    if right < left:
        raise ValueError('Actual actor timestamps run backwards')
    return left, right, {'start': start, 'end': end, 'start_source': source, 'start_evidence': evidence}


def host_for_status(status_path, value):
    if value.get('host') or value.get('hostname'):
        return value.get('host', value.get('hostname'))
    directory = Path(status_path).parent.parent
    for ancestor in [directory, *directory.parents[:4]]:
        launch = ancestor / 'LAUNCH.json'
        if launch.exists() and read(launch).get('host'):
            return read(launch)['host']
        command = ancestor / 'COMMAND.json'
        if command.exists():
            data = read(command)
            if isinstance(data, dict) and data.get('host'):
                return data['host']
    if directory.name.startswith('training-'):
        return directory.name
    raise ValueError(f'Actual host identity missing for GPU actor: {status_path}')


def compute(manifests):
    by_gpu = defaultdict(list)
    actors = []
    seen = set()
    for manifest in manifests:
        for job in read(manifest):
            status_path = Path(job['status'])
            if str(status_path.resolve()) in seen:
                continue
            seen.add(str(status_path.resolve()))
            value = read(status_path)
            if value.get('state') not in ('done', 'failed') or value.get('returncode') is None:
                raise ValueError(f'Actor has not actually exited: {status_path}')
            start, end, timing = actor_times(status_path, value)
            host, uuid = job['host'], job['uuid']
            by_gpu[(host, uuid)].append((start, end))
            actors.append({'name': job['name'], 'host': host, 'uuid': uuid, 'returncode': value['returncode'],
                           'started_at': timing['start'], 'ended_at': timing['end'], 'timing_evidence': timing,
                           'actor_wall_seconds': end - start, 'actual_exit_receipt': reference(status_path)})
    groups = []
    for (host, uuid), intervals in sorted(by_gpu.items()):
        actor = sum(end - start for start, end in intervals)
        physical = union_seconds(intervals)
        groups.append({'host': host, 'uuid': uuid, 'actor_count': len(intervals),
                       'actor_GPU_wall_hours': actor / 3600., 'physical_GPU_union_hours': physical / 3600.,
                       'overlap_factor_actor_to_physical': actor / physical if physical else None})
    return {'actor_GPU_wall_hours': sum(row['actor_wall_seconds'] for row in actors) / 3600.,
            'physical_GPU_union_hours': sum(row['physical_GPU_union_hours'] for row in groups),
            'GPU_groups': groups, 'actors': actors, 'failed_actor_attempts': sum(row['returncode'] != 0 for row in actors),
            'manifest_references': [reference(path) for path in manifests]}


def total_preparation_cost(root, results):
    """Enumerate only the explicitly authorized new-task dispatch namespaces."""
    root, results = Path(root), Path(results)
    paths = set((root / 'dispatch').glob('**/status/*.json'))
    paths.update((results / 'qualification').glob('dispatch*/**/status/*.json'))
    paths.update((results / 'reference_noise/dispatch').glob('**/status/*.json'))
    paths.update((results / 'preparation_owner/dispatch').glob('**/status/*.json'))
    groups = defaultdict(list)
    actors, unknown, cpu = [], [], []
    for path in sorted(paths):
        try:
            value = read(path)
        except (ValueError, OSError) as error:
            unknown.append({'status_path': str(path), 'reason': repr(error)})
            continue
        gpu = str(value.get('gpu', '')).strip()
        if gpu in ('-', '', 'cpu', 'CPU', 'none', 'None'):
            cpu.append({'status_path': str(path), 'state': value.get('state'), 'GPU_cost': 'excluded_CPU_actor'})
            continue
        uuids = [part.strip() for part in gpu.split(',') if part.strip()]
        if not all(uuid.startswith('GPU-') for uuid in uuids):
            unknown.append({'status_path': str(path), 'reason': 'GPU UUID identity missing; numeric index is not a physical identity', 'raw_gpu': gpu})
            continue
        if value.get('state') not in ('done', 'failed') or value.get('returncode') is None:
            unknown.append({'status_path': str(path), 'reason': 'Actual terminal exit is unknown; no end time assumed',
                            'state': value.get('state'), 'pid': value.get('pid'), 'gpu_uuids': uuids})
            continue
        try:
            host = host_for_status(path, value)
            start, end, timing = actor_times(path, value)
        except (ValueError, KeyError, OSError) as error:
            unknown.append({'status_path': str(path), 'reason': repr(error), 'state': value.get('state'), 'gpu_uuids': uuids})
            continue
        for uuid in uuids:
            groups[(host, uuid)].append((start, end))
        actors.append({'name': value['name'], 'host': host, 'gpu_uuids': uuids, 'returncode': value['returncode'],
                       'wall_seconds': end - start, 'GPU_actor_seconds': (end - start) * len(uuids),
                       'timing_evidence': timing, 'actual_exit_receipt': reference(path)})
    physical = [{'host': host, 'uuid': uuid, 'physical_GPU_union_hours': union_seconds(intervals) / 3600.,
                 'actor_GPU_wall_hours': sum(end - start for start, end in intervals) / 3600.,
                 'measured_actor_intervals': len(intervals)} for (host, uuid), intervals in sorted(groups.items())]
    return {'schema': 'nrkd_total_preparation_GPU_cost_v1',
            'status': 'COMPLETE_MEASURED_COST' if not unknown else 'CONFIRMED_INTERVALS_WITH_UNKNOWN_RECORDS', 'time': stamp(),
            'actor_GPU_wall_hours': sum(row['GPU_actor_seconds'] for row in actors) / 3600.,
            'physical_GPU_union_hours': sum(row['physical_GPU_union_hours'] for row in physical),
            'is_lower_bound_due_to_unknown_records': bool(unknown), 'physical_GPU_groups': physical, 'actors': actors,
            'failed_actor_attempts_included': sum(row['returncode'] != 0 for row in actors),
            'unknown_exit_or_timing_records': unknown, 'CPU_actors_excluded_from_GPU_union': cpu,
            'enumeration_namespaces': [str(root / 'dispatch/**/status/*.json'), str(results / 'qualification/dispatch*/**/status/*.json'),
                 str(results / 'reference_noise/dispatch/**/status/*.json'), str(results / 'preparation_owner/dispatch/**/status/*.json')],
            'historical_frozen_legacy_cache_generation_included': False,
            'all_physical_preparation_intervals_unioned_together_not_component_sum': True}


def report(root=ROOT, *, include_gt=False, round1_report=None, results=None):
    root = Path(root)
    complete = root / 'teacher_suite/COMPLETE.json'
    if not complete.exists() or read(complete).get('status') != 'LOCAL_TEACHER_COMPLETE':
        raise ValueError('Physical cost finalization waits for the teacher owner to complete all actual scoring/export actors')
    manifests = sorted((root / 'dispatch/local_teacher').glob('*/attempt_*/JOBS.json'))
    if not manifests:
        raise ValueError('No registered teacher GPU actor manifests')
    value = {'schema': 'nrkd_teacher_physical_GPU_cost_v1', 'status': 'COMPLETE', 'time': stamp(),
             'teacher_scoring_and_validation_target_export': compute(manifests),
             'interpretation': 'actor_GPU_wall_hours sums actor wall times; physical_GPU_union_hours merges overlapping intervals per (host,UUID). Both include failed attempts, scoring and IO wait; neither is pure GPU kernel time.',
             'CPU_phase_wall_seconds_are_separate_not_GPU_cost': (read(root / 'teacher_suite/COSTS.json').get('CPU_phases', {}) if (root / 'teacher_suite/COSTS.json').exists() else {}),
             'original_actor_accounting_preserved': reference(root / 'teacher_suite/COSTS.json')}
    if include_gt:
        requirements = read(root / 'dispatch/GT_EXIT_REQUIREMENTS.json')
        records = requirements if isinstance(requirements, list) else requirements['job_manifests']
        gt = []
        for record in records:
            path = Path(record['path'] if isinstance(record, dict) else record)
            if isinstance(record, dict) and sha(path) != record['sha256']:
                raise ValueError('GT manifest changed after registration')
            gt.append(path)
        value['GT_search_separate'] = compute(gt)
    destination = root / 'teacher_suite/PHYSICAL_GPU_COSTS.json'
    atomic(destination, value)
    results = Path(results or '/mnt/project/recogdrive-no-routing-round1-results-20261026')
    preparation = total_preparation_cost(root, results)
    atomic(root / 'teacher_suite/TOTAL_PREPARATION_GPU_COSTS.json', preparation)
    measured = value['teacher_scoring_and_validation_target_export']
    text = ['# Measured local teacher cost', '',
            f"Actor GPU wall hours: {measured['actor_GPU_wall_hours']:.6f}.",
            f"Physical GPU interval-union hours: {measured['physical_GPU_union_hours']:.6f}.", '',
            'Overlapping actors on the same physical GPU are counted once in physical GPU hours. Failed attempts remain included. These measurements include CPU scoring and IO wait within a GPU actor; they are not kernel FLOP measurements. CPU fitting phase wall times remain separate.', '']
    (root / 'teacher_suite/PHYSICAL_GPU_COSTS.md').write_text('\n'.join(text))
    if round1_report:
        path = Path(round1_report)
        body = path.read_text()
        marker = '<!-- NRKD_PHYSICAL_TEACHER_GPU_COST -->'
        if marker in body:
            body = body.split(marker, 1)[0].rstrip() + '\n'
        section = ['\n' + marker, '', '## Measured teacher GPU occupancy', '',
                   f"Teacher scoring/export actor wall time totals {measured['actor_GPU_wall_hours']:.6f} GPU actor hours. Physical GPU occupancy after merging intervals for each (host, UUID) is {measured['physical_GPU_union_hours']:.6f} GPU hours. Failed attempts are included. CPU fitting wall time is separate; overlapping actors are not counted twice as physical GPU time.", '',
                   f"Evidence: `{destination}` (SHA256 `{sha(destination)}`).", '']
        section += [f"Across all enumerated new-task preparation, generation, GT search, graph checks, engineering and formal qualification, confirmed actor wall time totals {preparation['actor_GPU_wall_hours']:.6f} GPU actor hours and confirmed physical GPU interval union is {preparation['physical_GPU_union_hours']:.6f} GPU hours. Unknown exit/timing records: {len(preparation['unknown_exit_or_timing_records'])}; these totals are {'lower bounds' if preparation['is_lower_bound_due_to_unknown_records'] else 'complete for the enumerated dispatch namespaces'}. Historical frozen cache generation is excluded.", '',
                    f"Preparation evidence: `{root / 'teacher_suite/TOTAL_PREPARATION_GPU_COSTS.json'}`.", '']
        path.write_text(body + '\n'.join(section))
        paired = root / 'engineering/report_round1_paired_comparisons.py'
        if not paired.is_file():
            raise FileNotFoundError('Final report requires the complete paired cross-arm comparison postprocessor')
        subprocess.run([sys.executable, '-B', str(paired), '--out', str(results), '--round1-report', str(path)], check=True)
    return value


def selftest(root=ROOT):
    root = Path(root)
    folder = root / 'selftests/physical_gpu_cost'
    folder.mkdir(parents=True, exist_ok=True)
    jobs = []
    # Two overlapping actors on GPU_A, one independent actor on GPU_B.
    for name, uuid, start, end, code in [('a', 'GPU_A', 0, 10, 0), ('b', 'GPU_A', 5, 15, 1), ('c', 'GPU_B', 0, 4, 0)]:
        path = folder / (name + '.json')
        atomic(path, {'state': 'done' if code == 0 else 'failed', 'returncode': code,
                     'started_at': f'2026-10-26T00:00:{start:02d}+00:00', 'ended_at': f'2026-10-26T00:00:{end:02d}+00:00'})
        jobs.append({'name': name, 'host': 'private_fixture', 'uuid': uuid, 'status': str(path)})
    manifest = folder / 'JOBS.json'; atomic(manifest, jobs)
    value = compute([manifest])
    assert value['actor_GPU_wall_hours'] == 24 / 3600
    assert value['physical_GPU_union_hours'] == 19 / 3600
    assert value['failed_actor_attempts'] == 1
    assert union_seconds([(0, 2), (2, 5), (10, 12), (11, 15)]) == 10
    terminal = folder / 'status/terminal_without_started.json'
    atomic(terminal, {'name': 'terminal_without_started', 'pid': 1234, 'state': 'done', 'returncode': 0,
                      'ended_at': '2026-10-26T00:00:10+00:00'})
    (folder / 'commands.log').write_text('[2026-10-26T00:00:02+00:00] name=terminal_without_started gpu=GPU_A\n'
                                       '[2026-10-26T00:00:03+00:00] name=terminal_without_started pid=1234\n')
    left, right, timing = actor_times(terminal, read(terminal))
    assert right - left == 7 and timing['start_source'] == 'commands.log_actual_PID_registration_timestamp'
    result = {'status': 'PASS_PHYSICAL_GPU_INTERVAL_UNION', 'synthetic_only': True,
              'actor_seconds': 24, 'physical_GPU_union_seconds': 19, 'failed_attempt_included': True,
              'actual_terminal_status_without_started_at_supported': True}
    atomic(folder / 'COMPLETE.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--include-gt', action='store_true')
    parser.add_argument('--round1-report', type=Path)
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--results', type=Path)
    args = parser.parse_args()
    value = selftest(args.root) if args.selftest else report(args.root, include_gt=args.include_gt, round1_report=args.round1_report, results=args.results)
    print(json.dumps({'status': value['status'], 'evidence_root': str(args.root / 'teacher_suite')}), flush=True)


if __name__ == '__main__':
    main()
