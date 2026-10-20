"""Account executed research costs without substituting expected measurements.

Run ledgers and job leases are two views of the same work: never add their GPU
seconds. Incomplete ledgers are snapshots, not final costs. Missing telemetry
stays null and every failed/interrupted job remains in the lease inventory.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import csv
from collections import defaultdict

from scripts.process_improvement.common import *


def sum_ledgers(paths):
    payloads = {path:path.read_bytes() for path in paths}
    values = [json.loads(payload) for payload in payloads.values()]
    if not values:
        return None
    # During live checkpointing, rank snapshots may momentarily differ. Do not
    # combine those into a fictitious synchronized training endpoint.
    if len({v['step'] for v in values}) != 1:
        return dict(status='RANK_SNAPSHOTS_NOT_SYNCHRONIZED', source_files=[str(p) for p in paths])
    compute = defaultdict(lambda: defaultdict(float))
    budget = defaultdict(float)
    for value in values:
        for role, record in value['compute'].items():
            for key, count in record.items():
                compute[role][key] += count
        for key, count in value['budget'].items():
            budget[key] += count
    return dict(status='MEASURED', checkpoint_step=values[0]['step'],
                GPU_seconds=sum(v['allocated_GPU_seconds'] for v in values),
                compute={k:dict(v) for k,v in compute.items()}, budget=dict(budget),
                peak_gpu_bytes_per_rank=[v.get('peak_gpu_bytes') for v in values],
                peak_history_complete=all(v.get('peak_history_complete',False) for v in values),
                source_files={str(p):hashlib.sha256(payload).hexdigest() for p,payload in payloads.items()})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--snapshot-name',default='actual_costs')
    a=p.parse_args()
    jobs=[]
    for path in sorted((ROOT/'jobs').glob('*/spec.json')):
        spec=read(path);done=path.parent/'complete.json'
        value=read(done) if done.exists() else None
        admission=read(path.parent/'admission_failure.json') if (path.parent/'admission_failure.json').exists() else None
        rejected=bool(admission and 'RuntimeError: GPU occupied:' in admission['error'] and not (path.parent/'launch.json').exists())
        jobs.append(dict(name=path.parent.name,host=spec['host'],gpus=spec['gpus'],
            status=value['status'] if value else 'PRELAUNCH_GPU_REJECTED' if rejected else 'RUNNING_OR_UNFINISHED',
            allocated_GPU_seconds=(value['seconds']*len(spec['gpus']) if value and 'seconds' in value else 0. if rejected else None),
            cost_complete=bool((value and 'seconds' in value) or rejected),
            admission_failure=admission,no_worker_launched=rejected,
            commands=spec['commands'],source_spec_sha256=sha(path)))
    runs=[]
    for path in sorted((ROOT/'runs').glob('*/config.json')):
        cfg=read(path);directory=path.parent
        done=read(directory/'complete.json') if (directory/'complete.json').exists() else None
        paused=read(directory/'paused.json') if (directory/'paused.json').exists() else None
        paths=sorted(directory.glob('cost_rank_*.json'))
        ledger=sum_ledgers(paths) if len(paths)==cfg.get('world',1) else None
        if done and ledger and ledger['status']=='MEASURED':
            if ledger['checkpoint_step']!=done['steps'] or ledger['budget']['physical_scores']!=done['physical_scores']:
                raise ValueError('Final cost ledger does not match completed policy: '+directory.name)
        progress=read(directory/'progress.json') if (directory/'progress.json').exists() else []
        runs.append(dict(name=directory.name,method=cfg['method'],split=cfg['split'],
            seed=cfg['seed'],world=cfg.get('world',1),parameter_count=cfg.get('parameter_count'),
            status=done['status'] if done else 'PAUSED' if paused else 'RUNNING_OR_UNFINISHED',
            actual_steps=done['steps'] if done else paused['step'] if paused else progress[-1]['step'] if progress else 0,
            completion=done,ledger=ledger,
            telemetry_missing=ledger is None,
            source_config_sha256=sha(path)))
    diagnostics=[]
    for path in sorted((ROOT/'diagnostics').glob('*/complete_*.json')):
        value=read(path)
        diagnostics.append(dict(name=path.parent.name,shard=path.stem,status=value['status'],
            budget=value.get('budget'),seconds=value.get('seconds'),source_sha256=sha(path),
            sampling_role=read(path.parent/'sampling_role.json') if (path.parent/'sampling_role.json').exists() else None))
    evaluations=[]
    for path in sorted((ROOT/'evaluation').glob('*/*/complete_*.json')):
        value=read(path)
        evaluations.append(dict(split=path.parents[1].name,name=path.parent.name,shard=path.stem,
            scenes=value['scenes'],seconds=value.get('seconds'),budget=value.get('budget'),source_sha256=sha(path)))
    result=dict(study=STUDY,protocol_sha256=sha(ROOT/'protocol.yaml'),jobs=jobs,runs=runs,
        diagnostics=diagnostics,evaluations=evaluations,
        completed_job_lease_GPU_hours=sum(v['allocated_GPU_seconds'] for v in jobs if v['allocated_GPU_seconds'] is not None)/3600,
        unfinished_jobs=sum(v['status']=='RUNNING_OR_UNFINISHED' for v in jobs),
        unknown_finished_job_lease_costs=[v['name'] for v in jobs if not v['cost_complete'] and v['status']!='RUNNING_OR_UNFINISHED'],
        interpretation=[
            'Run ledgers and exclusive job leases overlap; do not sum these GPU cost views.',
            'Finished lease costs include setup, inference, CPU scoring waits and failed qualifications.',
            'Qualifying/scientific diagnostic scores and tuning/evaluation costs are separate from each policy training budget.',
            'A warm-start method inherits its selected training-seed-specific warm cost; this is not zero-cost initialization.',
            'Frozen VLM extraction is a shared study asset; raw producer/lease records retain its actual cost.',
            'Original IL/RL/VLM pretraining costs are unknown shared prerequisites; COPY has zero new planner optimization.',
            'Missing telemetry is null, never inferred from a different run; unfinished snapshots are lower-bound observations only.',
        ])
    write(ROOT/'analysis'/f'{a.snapshot_name}.json',result)
    write(REPORTS/'tables'/f'{a.snapshot_name}.json',result)
    table=REPORTS/'tables'/f'{a.snapshot_name}_runs.csv'
    with table.open('w',newline='') as stream:
        keys=['name','method','split','seed','world','status','actual_steps','GPU_seconds','physical_scores','logical_scores','teacher_DiT_calls','student_DiT_calls','telemetry_missing']
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        for run in runs:
            row={k:run[k] for k in keys if k in run}
            measured=run['ledger'] if run['ledger'] and run['ledger']['status']=='MEASURED' else None
            if measured:
                row.update(GPU_seconds=measured['GPU_seconds'],physical_scores=measured['budget'].get('physical_scores'),
                    logical_scores=measured['budget'].get('logical_scores'),
                    teacher_DiT_calls=measured['compute'].get('teacher',{}).get('forward_calls'),
                    student_DiT_calls=measured['compute'].get('student',{}).get('forward_calls'))
            writer.writerow(row)
    print('ACTUAL_COST_SNAPSHOT',len(jobs),len(runs),len(diagnostics),len(evaluations),result['completed_job_lease_GPU_hours'],flush=True)


if __name__=='__main__':main()
