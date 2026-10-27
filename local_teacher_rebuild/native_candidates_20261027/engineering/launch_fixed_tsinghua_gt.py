"""Add measured three-per-GPU GT actors on the newly free fixed host.

Resource orchestration only: the active worker, search, noise and budgets are
unchanged. Register all actual exits before dispatch so teacher fitting cannot
overtake a newly registered actor. Existing workers and allocations are untouched.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import copy
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'engineering/permanent_fullflow'), str(ROOT / 'code')]
from gt_numeric import atomic, read, sha, stamp
from permanent_dispatch import grant, install

HOST = 'training-vlawm-tsinghua-zt'


def main():
    inventory = ROOT / 'execution_watch/PERMANENT_TSINGHUA_INVENTORY_20261027T021348Z.json'
    evidence = read(inventory)
    current = grant()
    if (HOST not in current['temporary_hosts'] or evidence['host'] != HOST or
            evidence['inventory']['compute_processes']):
        raise ValueError('The fixed host must be granted and actually free at admission')
    observed = datetime.fromisoformat(evidence['observed_at'])
    if (datetime.now(timezone.utc) - observed).total_seconds() > 600:
        raise ValueError('Resource inventory is too old for this one-time dispatch')
    folder = ROOT / 'dispatch/gt_fixed_tsinghua24'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'DISPATCH.lock').open('a+') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder / HOST / 'launcher.pid').exists():
            raise RuntimeError('Existing launcher requires exact invocation recovery, not redispatch')
        template = read(ROOT / 'dispatch/gt_local64/JOBS.json')[0]
        jobs, allocation = [], []
        for line in evidence['inventory']['GPUs']:
            index, uuid, used, free, utilization = [part.strip() for part in line.split(',')]
            index, used, free, utilization = map(int, (index, used, free, utilization))
            if (index not in current['authorized_gpu_indices'][HOST] or
                    index in current.get('excluded_gpu_indices', {}).get(HOST, []) or
                    free < 60000 or used != 0 or utilization != 0):
                raise ValueError('Unexpected GPU identity, grant or free capacity')
            allocation.append({'host': HOST, 'gpu': index, 'uuid': uuid, 'actors_per_gpu': 3,
                               'minimum_free_mb_at_admission': 60000})
            for actor in range(3):
                rank = index * 3 + actor
                name = f'local_gt_fixed_tsinghua_{rank:03d}'
                job = copy.deepcopy(template)
                args = job['arguments']
                args[args.index('--worker-id') + 1] = name
                job.update(name=name, worker_id=name, host=HOST, gpu=index, uuid=uuid,
                           rank=64 + rank, arguments=args,
                           status=str(folder / HOST / 'status' / (name + '.json')),
                           heartbeat=str(ROOT / 'native_gt/workers' / name / 'STATUS.json'))
                jobs.append(job)
        if len(jobs) != 24 or len({x['uuid'] for x in allocation}) != 8:
            raise ValueError('Exactly eight distinct physical GPUs / 24 actors required')
        atomic(folder / 'JOBS.json', jobs)
        atomic(folder / 'ALLOCATION.json', allocation)
        requirements = ROOT / 'dispatch/GT_EXIT_REQUIREMENTS.json'
        with (requirements.parent / 'GT_EXIT_REQUIREMENTS.edit.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            value = read(requirements)
            record = {'path': str(folder / 'JOBS.json'), 'sha256': sha(folder / 'JOBS.json')}
            if not any(x['path'] == record['path'] for x in value['job_manifests']):
                value['job_manifests'].append(record)
                value.update(persistent_workers=88, physical_GPUs=40,
                             fixed_tsinghua_expansion_inventory={'path': str(inventory), 'sha256': sha(inventory)})
                atomic(requirements, value)
            elif record not in value['job_manifests']:
                raise ValueError('Registered job manifest identity changed')
        install()
        import dispatch
        launched = dispatch.launch(HOST, folder / HOST,
                                   [(j['name'], j['uuid'], j['arguments']) for j in jobs])
        now = datetime.now(timezone.utc)
        # 24 sequential eight-second starts, action-only model loading and buffer.
        atomic(folder / 'STARTUP_READ_PLAN.json', {
            'created_at': stamp(), 'read_not_before': (now + timedelta(minutes=5)).isoformat(),
            'estimate_seconds': 24 * 8 + 60, 'buffer_seconds': 48,
            'reason': 'one startup verification after all staggered actors should have loaded',
            'no_intermediate_result_polling': True})
        atomic(folder / 'DISPATCH_COMPLETE.json', {
            'status': 'REGISTERED_AND_LAUNCHED_NOT_SCIENTIFIC_COMPLETION', 'at': stamp(),
            'launcher': launched, 'actor_count': 24, 'physical_GPU_count': 8,
            'job_manifest': {'path': str(folder / 'JOBS.json'), 'sha256': sha(folder / 'JOBS.json')},
            'worker_source': {'path': args[2], 'sha256': sha(args[2])},
            'exit_requirements': {'path': str(requirements), 'sha256': sha(requirements)},
            'existing_64_actors_and_teacher_student_resource_pins_unchanged': True})
        print(json.dumps({'actors': 24, 'physical_GPUs': 8, 'launch': launched,
                          'startup_read_plan': str(folder / 'STARTUP_READ_PLAN.json')}, indent=2))


if __name__ == '__main__':
    main()
