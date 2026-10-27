"""Permanent-only dispatch transport; no scientific data/model transformations."""
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from gt_numeric import atomic, read

TASK='recogdrive_no_routing_round1_20261026'
PERMANENT={'training-vla-zt','training-vlawm-tsinghua-zt','training-vlawm-tsinghua-zt3',
           'training-vlawm-tsinghua-zt4','training-vlawm-tsinghua-zt5'}
PYTHON='/mnt/project/recogdrive-distillation-first42-a800-20261022/.venv/bin/python'
LEGACY=Path('/mnt/project/recogdrive-teacher2-a800-20261026')

def grant():
    value=read('/mnt/project/server_dispatch_policy.json')['task_authorizations'][TASK]
    if value.get('status')!='ACTIVE_TASK_BOUNDED_GRANT' or value.get('host_scope')!='EXISTING_PERMANENT_ALLOWLIST_ONLY':
        raise RuntimeError('Current permanent-only user task grant is required')
    if set(value.get('authorized_hosts',value['temporary_hosts']))-PERMANENT:
        raise ValueError('The task grant contains a withdrawn host')
    return value

def admit(allocation):
    current=grant()
    if not allocation or len({a['uuid'] for a in allocation})!=len(allocation):raise ValueError('Unique physical UUID allocation required')
    for item in allocation:
        host=item['host'];gpu=int(item['gpu'])
        if host not in PERMANENT or host not in current['temporary_hosts'] or gpu not in current['authorized_gpu_indices'][host] or gpu in current.get('excluded_gpu_indices',{}).get(host,[]):
            raise ValueError('GPU is outside the current permanent-only grant')
        if int(item.get('actors_per_gpu',2)) not in (1,2):raise ValueError('Only one/two registered actors per GPU are supported')
    return current

def install():
    """Private module adapter; preserve immutable legacy launcher source."""
    sys.path.insert(0,str(LEGACY/'local'));import dispatch
    if getattr(dispatch,'_permanent_fullflow_adapter',False):return
    original=dispatch.launch
    def launch(host,directory,jobs):
        current=grant()
        if host not in PERMANENT or host not in current['temporary_hosts']:raise RuntimeError('Host permission withdrawn')
        if socket.gethostname() not in (host,host+'-worker-0'):return original(host,directory,jobs)
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        if (directory/'launcher.pid').exists():raise ValueError('Existing stable launcher must be resumed')
        lines=[]
        for name,gpu,arguments in jobs:
            command=shlex.join(['env','ZT_AUTO_PROXY=0','OPENBLAS_CORETYPE=Sandybridge','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',*arguments])
            lines.append(name+'\t'+str(gpu)+'\t'+command+'\n')
        (directory/'jobs.tsv').write_text(''.join(lines))
        env=os.environ.copy();env.update(ZT_AUTO_PROXY='0',OPENBLAS_CORETYPE='Sandybridge',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        with (directory/'launcher.log').open('ab',buffering=0) as log:
            child=subprocess.Popen([PYTHON,'/mnt/project/skill/stable-gpu-job-launch/scripts/stable_gpu_launcher.py',
                    '--jobs-tsv',str(directory/'jobs.tsv'),'--out-root',str(directory),
                    '--stagger-seconds','8','--poll-seconds','5'],env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (directory/'launcher.pid').write_text(str(child.pid)+'\n')
        value={'host':host,'pid':child.pid,'transport':'canonical_local_same_stable_launcher'}
        atomic(directory/'LAUNCH.json',value);return value
    dispatch.TASK=TASK;dispatch.launch=launch;dispatch._permanent_fullflow_adapter=True
