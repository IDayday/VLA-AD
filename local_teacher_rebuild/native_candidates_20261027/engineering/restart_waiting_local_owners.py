"""Replace two exact CPU WAIT owners after reviewed recovery-gate changes.

No GT/GPU worker or unknown process is signalled. Original resource files and
all old invocation receipts remain intact.
"""
import ctypes
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
RESULTS=Path('/mnt/project/recogdrive-no-routing-round1-results-20261026')
CODE=Path('/mnt/project/recogdrive-no-routing-round1-20261026')
HERE=ROOT/'engineering/permanent_fullflow'
sys.path[:0]=[str(HERE),str(ROOT/'code')]
from gt_numeric import atomic,read,sha,stamp
from permanent_dispatch import PYTHON,grant,install


def stop_exact_wait_owner(kind,pid,registry_path,source,lock,waiting_artifact):
    registry=read(registry_path)
    recorded=registry.get('owner_pid',registry.get('pid'))
    expected_ticks=str(registry.get('owner_start_ticks',registry.get('process_start_ticks')))
    if recorded!=pid or 'WAITING' not in registry['status'] or waiting_artifact.exists():
        raise RuntimeError('Owner is no longer the exact safe WAIT invocation')
    proc=Path(f'/proc/{pid}')
    command=[x.decode() for x in (proc/'cmdline').read_bytes().split(bytes([0])) if x]
    if str(source) not in command or '--root' not in command or command[command.index('--root')+1]!=str(ROOT):
        raise RuntimeError('CPU owner command identity differs; preserve it')
    ticks=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
    if ticks!=expected_ticks or (proc/'task'/str(pid)/'children').read_text().strip():
        raise RuntimeError('CPU owner start identity changed or has a child; preserve it')
    libc=ctypes.CDLL(None,use_errno=True);fd=libc.syscall(434,pid,0)
    if fd<0:raise OSError(ctypes.get_errno(),'pidfd_open')
    try:
        if (proc/'stat').read_text().rsplit(')',1)[1].split()[19]!=ticks:
            raise RuntimeError('CPU owner changed while opening pidfd')
        if libc.syscall(424,fd,signal.SIGTERM,0,0)<0:raise OSError(ctypes.get_errno(),'pidfd_send_signal')
        if not select.select([fd],[],[],20.)[0]:raise RuntimeError('CPU owner exit not observed')
    finally:os.close(fd)
    with lock.open('a+') as stream:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return {'kind':kind,'pid':pid,'process_start_ticks':ticks,'verified_command':command,
            'status_at_stop':registry['status'],'kernel_pidfd_exit_event':True,'OWNER_lock_released':True,
            'no_children_at_stop':True,'no_GT_or_GPU_signal':True,'ended_at':stamp()}


def main():
    grant()
    freeze=read(RESULTS/'protocol/CODE_FREEZE.json')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()
    if freeze['git_commit']!=commit:raise ValueError('Revised source commit has not been frozen')
    snapshot=freeze['production_source_snapshot']
    if sha(snapshot['path'])!=snapshot['sha256']:raise ValueError('Frozen production source manifest changed')
    for row in read(snapshot['path'])['files']:
        if sha(row['active_source'])!=row['sha256']:raise ValueError('Reviewed production source changed')
    requests=[
        ('full_teacher_native_candidates_owner',362980,ROOT/'teacher_suite/REGISTRY.json',HERE/'fullflow_teacher_owner.py',
         ROOT/'teacher_suite/OWNER.lock',ROOT/'native_gt/GT_POST_COMPLETE.json',
         ['--allocation',str(HERE/'PERMANENT_TEACHER_40GPU_ALLOCATION.json')]),
        ('round1_preparation_native_candidates_owner',362988,RESULTS/'preparation_owner/REGISTRY.json',ROOT/'code/round1_prepare_and_run.py',
         RESULTS/'preparation_owner/OWNER.lock',ROOT/'teacher_suite/FULL_TRAIN_TEST_TEACHER_READY.json',
         ['--resources',str(RESULTS/'resources/ROUND1_FIXED_32GPU_WITH_TSINGHUA_ALLOCATION.json')]),
    ]
    receipt=ROOT/'dispatch/WAIT_OWNER_IDENTITY_RECOVERY.json'
    if receipt.exists():raise RuntimeError('Recovery already recorded; inspect its exact invocation before any restart')
    stops=[];launches=[];install();import dispatch
    for name,pid,registry,source,lock,artifact,extra in requests:
        directory=ROOT/'dispatch'/(name+'_identity_recovery_20261027')
        if (directory/'launcher.pid').exists():raise RuntimeError('Recovery owner already registered')
        stops.append(stop_exact_wait_owner(name,pid,registry,source,lock,artifact))
        atomic(receipt,{'status':'CONTROLLED_WAIT_OWNER_REPLACEMENT_IN_PROGRESS','stops':stops,'launches':launches})
        command=[PYTHON,'-B',str(source),'--root',str(ROOT),'--poll-seconds','30',*extra]
        launch=dispatch.launch('training-vla-zt',directory,[(name,'-',command)])
        launches.append({'name':name,'command':command,'source_sha256':sha(source),'launch':launch,
                         'actual_status':str(directory/'status'/(name+'.json'))})
        atomic(receipt,{'status':'REVIEWED_WAIT_OWNERS_REPLACED_NOT_SCIENTIFIC_COMPLETION','time':stamp(),
                       'git_commit':commit,'code_freeze_sha256':sha(RESULTS/'protocol/CODE_FREEZE.json'),
                       'stops':stops,'launches':launches,'GT_GPU_workers_preserved':True})
    print(json.dumps(read(receipt),indent=2))


if __name__=='__main__':main()
