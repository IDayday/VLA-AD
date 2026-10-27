"""External final report hook: retained preparation attempts across both roots.

No experiment source, score, exit or runtime status is changed by this reader.
Only intervals with an actual terminal actor receipt enter measured GPU totals.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
FAST = ROOT.parent / 'recogdrive_dc8_direct48_20261027'
RESULTS = Path('/mnt/project/recogdrive-no-routing-round1-results-20261026')
POLICY = Path('/mnt/project/server_dispatch_policy.json')


def load_legacy(root=ROOT):
    path = Path(root) / 'engineering/report_teacher_physical_costs.py'
    spec = importlib.util.spec_from_file_location('nrkd_original_physical_cost_reporter', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEGACY = load_legacy()
read, atomic, sha = LEGACY.read, LEGACY.atomic, LEGACY.sha


def ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path) if path.is_file() else None}


def host_aliases(policy):
    """Retain historical aliases too: revocation does not erase measured cost."""
    aliases = {}
    def visit(value, key=None):
        if isinstance(value, dict):
            names = value.get('aliases')
            if isinstance(names, list):
                canonical = value.get('host') or (key if str(key).startswith('training-') else None)
                if canonical:
                    canonical = str(canonical).removesuffix('-worker-0')
                    for name in [canonical, *names]:
                        aliases.setdefault(str(name), canonical)
            for child_key, child in value.items():
                visit(child, child_key)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    # Current canonical mappings take precedence over older snapshots embedded
    # in the policy. Withdrawn host identities are still recognized below.
    visit(policy.get('allowed_hosts', {}))
    visit(policy)
    return aliases


def canonical_host(name, aliases):
    if not isinstance(name, str) or not name.strip():
        raise ValueError('Actual host identity is missing')
    name = name.strip()
    return aliases.get(name, aliases.get(name.removesuffix('-worker-0'), name.removesuffix('-worker-0')))


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError('Actual timestamp is missing')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Timestamp has no timezone; UTC is not assumed')
    return parsed.timestamp()


def discover(roots, results):
    """Read only registered task namespaces, not student runs or source fixtures."""
    paths, registrations, scopes = set(), defaultdict(list), []
    namespaces = [(Path(root) / 'dispatch', str(Path(root).resolve())) for root in roots]
    namespaces += [(Path(results) / suffix, 'results/' + suffix)
                   for suffix in ('qualification', 'reference_noise', 'preparation_owner')]
    def add(path, scope, registration=None):
        path = Path(path).resolve()
        paths.add(path)
        item = {'scope': scope}
        if registration is not None:
            item.update(registration)
        if item not in registrations[path]:
            registrations[path].append(item)
    for namespace, scope in namespaces:
        scopes.append(str(namespace.resolve()))
        if not namespace.is_dir():
            continue
        for path in namespace.glob('**/status/*.json'):
            add(path, scope)
        for path in namespace.glob('**/STATUS.json'):
            add(path, scope)
        # A registered actor whose status never appeared must remain unknown.
        # Multiple manifests referencing the same status do not duplicate cost.
        for pattern in ('**/JOBS.json', '**/manifest.json'):
            for manifest in namespace.glob(pattern):
                try:
                    data = read(manifest)
                except (OSError, ValueError):
                    continue  # The manifest itself is not an actor exit record.
                jobs = data if isinstance(data, list) else data.get('jobs', []) if isinstance(data, dict) else []
                if isinstance(jobs, dict):
                    jobs = list(jobs.values())
                for job in jobs:
                    if not isinstance(job, dict):
                        continue
                    status = job.get('status', job.get('status_path'))
                    if not isinstance(status, str) or not status.endswith('.json'):
                        continue
                    path = Path(status)
                    path = path if path.is_absolute() else manifest.parent / path
                    add(path, scope, {'manifest': ref(manifest), 'job': job})
    return sorted(paths), registrations, scopes


def parse_actor(path, value, aliases, registration):
    """Separate actual terminal children, prelaunch exclusions and unknown exits."""
    is_core = path.name == 'STATUS.json'
    spec = {}
    spec_reference = None
    if is_core:
        spec_path = path.parent / 'SPEC.json'
        if not spec_path.is_file():
            raise ValueError('Controller status has no immutable SPEC.json')
        spec = read(spec_path)
        spec_reference = ref(spec_path)
        if value.get('spec_sha256') != spec_reference['sha256']:
            raise ValueError('Controller status/SPEC SHA mismatch')
        uuids = value.get('gpu_uuids', spec.get('gpu_uuids', []))
        if not isinstance(uuids, list) or uuids != spec.get('gpu_uuids'):
            raise ValueError('Controller status/SPEC physical GPU identities differ')
        host = canonical_host(spec.get('host'), aliases)
        if value.get('host') and canonical_host(value['host'], aliases) != host:
            raise ValueError('Controller status/SPEC canonical hosts differ')
        state, code = value.get('status'), value.get('returncode')
        child = value.get('child_started')
        if child is False and state in ('STARTUP_FAILED', 'TRANSPORT_FAILED', 'LAUNCH_UNCONFIRMED',
                                        'SUPERSEDED_BEFORE_START'):
            return 'excluded', {'reason': 'No scientific child admitted; GPU UUID visibility alone is not GPU execution',
                'status': state, 'child_started': False, 'returncode': code,
                'actual_process_exit': value.get('actual_process_exit'), 'gpu_uuids': uuids,
                'spec_reference': spec_reference}
        if child is False and state == 'EXITED':
            raise ValueError('Controller terminal-child receipt contradicts child_started=False')
        actual_terminal = state == 'EXITED' and type(code) is int
        # Older wrapper revisions omit child_started and child_pid. Their EXITED
        # status is written only after waiting for the real child process.
        child_evidence = ('explicit_child_started_true' if child is True else
                          'legacy_EXITED_child_wait_receipt' if actual_terminal else
                          'explicit_child_not_yet_started' if child is False else 'unknown')
        start, end = value.get('start'), value.get('end') if actual_terminal else None
        timing = {'start': start, 'end': end, 'start_source': 'Controller.STATUS.start',
                  'end_source': 'Controller.STATUS.EXITED.end' if actual_terminal else None,
                  'wrapper_observation_end': value.get('end') if not actual_terminal else None,
                  'child_evidence': child_evidence, 'spec_reference': spec_reference}
    else:
        gpu = value.get('gpu', '')
        if str(gpu).strip() in ('-', '', 'cpu', 'CPU', 'none', 'None'):
            # A manifest may identify a missing GPU field on an older status.
            jobs = [row.get('job', {}) for row in registration]
            manifest_uuids = [job.get('uuid') for job in jobs if job.get('uuid')]
            if not manifest_uuids:
                if 'gpu' not in value or gpu == '':
                    raise ValueError('Actor has no physical GPU or explicit CPU identity')
                return 'excluded', {'reason': 'Explicit CPU/no-GPU actor', 'state': value.get('state')}
            uuids = sorted(set(manifest_uuids))
        else:
            uuids = [part.strip() for part in str(gpu).split(',') if part.strip()]
        host = canonical_host(LEGACY.host_for_status(path, value), aliases)
        state, code = value.get('state'), value.get('returncode')
        actual_terminal = state in ('done', 'failed') and type(code) is int
        start, end = value.get('started_at'), value.get('ended_at') if actual_terminal else None
        timing = {'start': start, 'end': end, 'start_source': 'legacy.status.started_at',
                  'end_source': 'legacy.terminal.ended_at' if actual_terminal else None}
        if actual_terminal:
            _, _, timing = LEGACY.actor_times(path, value)
            start, end = timing['start'], timing['end']
    if not uuids or len(set(uuids)) != len(uuids) or any(not str(gpu).startswith('GPU-') for gpu in uuids):
        raise ValueError('Unique physical GPU UUIDs are missing; numeric indices are not substituted')
    common = {'name': value.get('name', path.parent.name), 'host': host,
              'reported_host': value.get('host', value.get('hostname')), 'gpu_uuids': uuids,
              'returncode': code if actual_terminal else None, 'reported_status': state,
              'pid': value.get('child_pid', value.get('pid')),
              'process_start_ticks': value.get('child_start_ticks', value.get('process_start_ticks')),
              'timing_evidence': timing, 'actual_start': start, 'actual_end': end,
              'status_reference': ref(path), 'spec_reference': spec_reference,
              'registrations': registration, 'format': 'Controller_UPPERCASE' if is_core else 'stable_launcher'}
    if not actual_terminal:
        return 'unknown', {**common, 'reason': 'Actual terminal child exit is unknown; no end or interval is assumed'}
    left, right = timestamp(start), timestamp(end)
    if right < left:
        raise ValueError('Actual terminal timestamps run backwards')
    return 'actor', {**common, 'start_epoch_seconds': left, 'end_epoch_seconds': right,
                     'wall_seconds': right-left, 'GPU_actor_seconds': (right-left)*len(uuids)}


def combined_preparation_cost(root=ROOT, fast_root=FAST, results=RESULTS, *, policy_path=POLICY):
    roots = list(dict.fromkeys(str(Path(path).resolve()) for path in (root, fast_root) if path is not None))
    policy = read(policy_path) if policy_path is not None else {}
    aliases = host_aliases(policy)
    paths, registrations, scopes = discover(roots, results)
    actors, unknown, excluded = [], [], []
    groups = defaultdict(list)
    for path in paths:
        source = ref(path)
        try:
            kind, record = parse_actor(path, read(path), aliases, registrations[path])
        except (OSError, ValueError, KeyError, TypeError) as error:
            unknown.append({'status_reference': source, 'reason': repr(error), 'actual_end': None,
                            'registrations': registrations[path]})
            continue
        if kind == 'excluded':
            excluded.append({'status_reference': source, **record})
        elif kind == 'unknown':
            unknown.append(record)
        else:
            actors.append(record)
            for gpu in record['gpu_uuids']:
                groups[(record['host'], gpu)].append((record['start_epoch_seconds'], record['end_epoch_seconds']))
    physical = [{'host': host, 'uuid': gpu, 'physical_GPU_union_seconds': LEGACY.union_seconds(intervals),
                 'physical_GPU_union_hours': LEGACY.union_seconds(intervals)/3600.,
                 'GPU_actor_seconds': sum(right-left for left,right in intervals),
                 'measured_actor_intervals': len(intervals)} for (host,gpu), intervals in sorted(groups.items())]
    return {'schema': 'nrkd_combined_preparation_GPU_cost_v2',
            'status': 'CONFIRMED_INTERVALS_WITH_UNKNOWN_RECORDS' if unknown else 'COMPLETE_MEASURED_COST',
            'time': datetime.now(timezone.utc).isoformat(), 'roots': roots, 'results': str(Path(results).resolve()),
            'actor_GPU_wall_seconds': sum(row['GPU_actor_seconds'] for row in actors),
            'actor_GPU_wall_hours': sum(row['GPU_actor_seconds'] for row in actors)/3600.,
            'physical_GPU_union_seconds': sum(row['physical_GPU_union_seconds'] for row in physical),
            'physical_GPU_union_hours': sum(row['physical_GPU_union_seconds'] for row in physical)/3600.,
            'physical_GPU_groups': physical, 'actors': actors,
            'failed_GPU_actor_attempts_included': sum(row['returncode'] != 0 for row in actors),
            'unknown_exit_or_timing_records': unknown, 'is_lower_bound_due_to_unknown_records': bool(unknown),
            'CPU_and_no_child_startups_excluded_from_GPU_cost': excluded,
            'status_reference_count_after_realpath_deduplication': len(paths),
            'host_aliases_used': aliases, 'host_alias_policy': ref(policy_path) if policy_path is not None else None,
            'enumeration_namespaces': scopes, 'student_training_namespace_included': False,
            'historical_frozen_legacy_cache_generation_included': False,
            'abandoned_FAST_intervals_are_cost_only_never_teacher_data': True,
            'all_physical_preparation_intervals_unioned_together_not_component_sum': True,
            'interpretation': 'Confirmed actor wall time includes failed attempts and IO/scoring waits. Physical occupancy merges all observed intervals once per canonical host and physical UUID across roots. Unknown exits/timing contribute no invented end time; totals are lower bounds when they exist. Controller intervals use recorded admission-to-child-wait completion and include admission overhead. These are allocation/actor intervals, not measured CUDA-active time.'}


def replace_preparation_section(body, section):
    lines = body.splitlines()
    left = next((i for i,line in enumerate(lines) if line.startswith('Across all enumerated new-task preparation,')), None)
    if left is None:
        raise ValueError('Original teacher reporter did not emit its expected preparation paragraph')
    right = next((i for i in range(left+1,len(lines)) if lines[i].startswith('Preparation evidence:')), None)
    if right is None:
        raise ValueError('Original preparation evidence boundary is missing')
    return '\n'.join([*lines[:left], *section, *lines[right+1:]]) + '\n'


def report(root=ROOT, fast_root=FAST, results=RESULTS, round1_report=None, *, policy_path=POLICY):
    root, results = Path(root), Path(results)
    round1_report = Path(round1_report) if round1_report else results/'ROUND1_REPORT.md'
    teacher_reporter = LEGACY if root.resolve() == ROOT.resolve() else load_legacy(root)
    teacher = teacher_reporter.report(root, include_gt=True, round1_report=round1_report, results=results)
    combined = combined_preparation_cost(root, fast_root, results, policy_path=policy_path)
    destination = root/'teacher_suite/COMBINED_PREPARATION_GPU_COSTS.json'
    atomic(destination, combined)
    unknown = len(combined['unknown_exit_or_timing_records'])
    section = ['<!-- NRKD_COMBINED_PREPARATION_GPU_COST -->', '',
        'Across the original full-flow root, abandoned FAST dispatch root, and registered results qualification/noise/preparation scopes, '
        f"confirmed actor wall time totals {combined['actor_GPU_wall_hours']:.6f} GPU actor hours. "
        f"The combined interval union is {combined['physical_GPU_union_hours']:.6f} physical GPU hours after canonicalizing hosts and merging all overlaps per UUID. "
        f"Failed GPU actor attempts included: {combined['failed_GPU_actor_attempts_included']}. Unknown exit/timing records: {unknown}; "
        + ('the totals are lower bounds.' if unknown else 'the totals cover all enumerated actor status references.'), '',
        'This combined total already includes the teacher subset above. FAST intervals contribute cost only; FAST data does not enter the restored original teacher. '
        'Controller startup failures with no admitted child contribute no GPU actor interval. Student training cost remains recorded separately. '
        'Historical frozen cache generation is outside these namespaces. Actor intervals include scoring/IO waits and do not measure CUDA kernel activity.', '',
        f"Preparation evidence: `{destination}` (SHA256 `{sha(destination)}`).", '']
    round1_report.write_text(replace_preparation_section(round1_report.read_text(), section))
    receipt = {'schema':'nrkd_round1_physical_cost_finalizer_v1','status':'COMPLETE_MEASURED_COMBINED_COST_POSTPROCESSING',
               'teacher_subset':ref(root/'teacher_suite/PHYSICAL_GPU_COSTS.json'), 'combined_preparation':ref(destination),
               'round1_report':ref(round1_report),'source':ref(__file__),
               'is_lower_bound_due_to_unknown_records':combined['is_lower_bound_due_to_unknown_records'],
               'formal_training_or_scoring_executed':False}
    atomic(root/'teacher_suite/ROUND1_PHYSICAL_COST_FINALIZATION.json',receipt)
    return receipt


def selftest(out):
    folder = Path(out)/('fixture_'+uuid.uuid4().hex)
    folder.mkdir(parents=True)
    old, fast, results = folder/'old', folder/'fast', folder/'results'
    policy = folder/'policy.json'
    atomic(policy,{'allowed_hosts':{'training-host':{'aliases':['training-host','training-host-worker-0','host-alias']}}})
    def at(second):return f'2026-10-27T00:00:{second:02d}+00:00'
    def stable(root,name,left,right,code=0,host='training-host'):
        path=root/'dispatch/batch/status'/f'{name}.json'
        atomic(path,{'name':name,'host':host,'gpu':'GPU-A','state':'done' if code==0 else 'failed',
                     'pid':100+left,'started_at':at(left),'ended_at':at(right),'returncode':code})
        return path
    first=stable(old,'old',0,10)
    stable(fast,'fast_failed',5,15,3,'training-host-worker-0')
    qualifier=results/'qualification'
    stable(qualifier,'engineering',8,12,host='host-alias')
    duplicate=fast/'dispatch/duplicate/status/old_alias.json'
    duplicate.parent.mkdir(parents=True);duplicate.symlink_to(first)
    def core(name,start,end,state,code,child,uuids=('GPU-A',)):
        path=results/'preparation_owner/remote_execution/jobs'/name/'attempt_00000/STATUS.json'
        spec=path.parent/'SPEC.json'
        atomic(spec,{'host':'training-host','gpu_uuids':list(uuids),'command':['private_CPU_fixture']})
        value={'spec_sha256':sha(spec),'host':'training-host-worker-0','gpu_uuids':list(uuids),
               'start':at(start),'status':state,'returncode':code,'child_started':child,'pid':200+start}
        if end is not None:value['end']=at(end)
        atomic(path,value)
        return path
    core('terminal_failed',12,20,'EXITED',1,True,('GPU-A','GPU-B'))
    core('unknown_running',20,None,'RUNNING',None,True)
    core('unknown_wrapper_end',20,50,'EXIT_UNKNOWN',None,True)
    core('no_child_startup',20,30,'STARTUP_FAILED',1,False)
    core('still_admitting',20,None,'ADMITTING',None,False)
    atomic(fast/'dispatch/registered/JOBS.json', [
        {'name':'same_registered_old_actor','status':str(first),'host':'training-host','uuid':'GPU-A'},
        {'name':'missing_status','status':str(fast/'dispatch/missing/status/never_written.json'),
         'host':'training-host','uuid':'GPU-A'}])
    value=combined_preparation_cost(old,fast,results,policy_path=policy)
    assert len(value['actors'])==4
    assert value['actor_GPU_wall_seconds']==40
    assert value['physical_GPU_union_seconds']==28
    assert len(value['physical_GPU_groups'])==2
    assert value['failed_GPU_actor_attempts_included']==2
    assert len(value['unknown_exit_or_timing_records'])==4 and value['is_lower_bound_due_to_unknown_records']
    assert all(row['actual_end'] is None for row in value['unknown_exit_or_timing_records'])
    assert len(value['CPU_and_no_child_startups_excluded_from_GPU_cost'])==1
    assert value['physical_GPU_groups'][0]['host']=='training-host'
    evidence=folder/'COMBINED.json';atomic(evidence,value)
    assert replace_preparation_section('before\nAcross all enumerated new-task preparation, old\n\nPreparation evidence: old\n\npaired section\n',['replacement']).endswith('paired section\n')
    # Exercise the composition hook using a private synthetic reporter. The
    # actual report path still always calls the unchanged original reporter.
    calls=[]
    class SyntheticReporter:
        def report(self, root, *, include_gt, round1_report, results):
            calls.append({'include_gt':include_gt,'round1_report':str(round1_report)})
            atomic(Path(root)/'teacher_suite/PHYSICAL_GPU_COSTS.json',{'synthetic':True})
            Path(round1_report).parent.mkdir(parents=True,exist_ok=True)
            Path(round1_report).write_text('teacher section\nAcross all enumerated new-task preparation, old\n\nPreparation evidence: old\n\n<!-- PAIRED_COMPARISONS -->\npaired section\n')
            return {'synthetic':True}
    loader=globals()['load_legacy']
    globals()['load_legacy']=lambda root:SyntheticReporter()
    try:
        final=report(old,fast,results,results/'ROUND1_REPORT.md',policy_path=policy)
    finally:
        globals()['load_legacy']=loader
    body=(results/'ROUND1_REPORT.md').read_text()
    assert len(calls)==1 and calls[0]['include_gt'] is True
    assert 'NRKD_COMBINED_PREPARATION_GPU_COST' in body and 'PAIRED_COMPARISONS' in body
    assert 'Across all enumerated new-task preparation, old' not in body
    assert final['is_lower_bound_due_to_unknown_records'] is True
    receipt={'status':'PASS_CPU_COMBINED_PREPARATION_COST','synthetic_only':True,
             'actor_GPU_wall_seconds':40,'physical_GPU_union_seconds':28,
             'OLD_FAST_and_Controller_intervals_unioned':True,'canonical_aliases_merged':True,
             'duplicate_realpath_reference_counted_once':True,'failed_attempts_included':2,
             'unknown_exits_have_no_invented_end':True,'no_child_startup_has_no_GPU_interval':True,
             'paired_report_section_preserved':True,'original_report_called_with_include_GT':True,
             'missing_registered_status_stays_unknown':True,'pending_prelaunch_remains_unknown':True,
             'fixture':ref(evidence),
             'composition_receipt':ref(old/'teacher_suite/ROUND1_PHYSICAL_COST_FINALIZATION.json'),
             'source':ref(__file__)}
    destination=Path(out)/'SELFTEST.json';atomic(destination,receipt)
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--fast-root',type=Path,default=FAST)
    parser.add_argument('--results','--out',dest='results',type=Path,default=RESULTS)
    parser.add_argument('--round1-report',type=Path)
    parser.add_argument('--host-policy',type=Path,default=POLICY)
    parser.add_argument('--selftest',action='store_true')
    parser.add_argument('--selftest-out',type=Path,default=ROOT/'engineering/permanent_fullflow/cost_selftest')
    args=parser.parse_args(argv)
    value=(selftest(args.selftest_out) if args.selftest else
           report(args.root,args.fast_root,args.results,args.round1_report,policy_path=args.host_policy))
    print(json.dumps(value,indent=2),flush=True)


if __name__=='__main__':main()
