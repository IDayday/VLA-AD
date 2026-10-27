"""Launch the finite downstream owners for the verified local candidate scope.

These CPU owners wait for actual complete artifacts. They never substitute a
submission receipt for teacher, qualification, training, evaluation or report.
"""
from __future__ import annotations
import fcntl
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'engineering/permanent_fullflow'
sys.path[:0]=[str(HERE),str(ROOT/'code')]
from gt_numeric import atomic, sha, stamp
from permanent_dispatch import PYTHON, grant, install

RESULTS=Path('/mnt/project/recogdrive-no-routing-round1-results-20261026')


def verify_unheld(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)


def main():
    grant()
    requirements=ROOT/'dispatch/GT_EXIT_REQUIREMENTS.json'
    value=json.loads(requirements.read_text())
    for record in value['job_manifests']:
        if sha(record['path'])!=record['sha256']:
            raise ValueError('Registered GT actor requirements changed')
    specifications=[
        ('full_teacher_native_candidates_owner',HERE/'fullflow_teacher_owner.py',ROOT/'teacher_suite/OWNER.lock'),
        ('round1_preparation_native_candidates_owner',ROOT/'code/round1_prepare_and_run.py',RESULTS/'preparation_owner/OWNER.lock'),
    ]
    install();import dispatch
    launches=[]
    for name,source,lock in specifications:
        directory=ROOT/'dispatch'/(name+'_20261027')
        if (directory/'launcher.pid').exists():
            raise RuntimeError('A registered owner launcher exists; reattach its exact invocation before any recovery')
        verify_unheld(lock)
        command=[PYTHON,'-B',str(source),'--root',str(ROOT),'--poll-seconds','30']
        result=dispatch.launch('training-vla-zt',directory,[(name,'-',command)])
        launches.append({'name':name,'source':{'path':str(source),'sha256':sha(source)},
                         'command':command,'launch':result,
                         'actual_exit_status':str(directory/'status'/(name+'.json'))})
    atomic(ROOT/'dispatch/LOCAL_CANDIDATE_DOWNSTREAM_OWNERS.json',{
        'status':'CPU_WAITING_OWNERS_LAUNCHED_NOT_SCIENTIFIC_COMPLETION','time':stamp(),
        'teacher_root':str(ROOT),'GT_requirements':{'path':str(requirements),'sha256':sha(requirements)},
        'owners':launches,'Val_and_preparation_may_share_authorized_GPUs':True,
        'student_ready_requires_actual_local_teacher_targets_calibration_formal_qualification':True})
    print(json.dumps(launches,indent=2),flush=True)


if __name__=='__main__':main()
