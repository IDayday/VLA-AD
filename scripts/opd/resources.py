"""Owned detached workers; policy and actual device capacity checked at admission."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,shlex
from scripts.opd.common import *


def remote(host,code):
    local=socket.gethostname().removesuffix('-worker-0')
    cmd=[PYTHON,'-'] if host==local else ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,'python -']
    r=subprocess.run(cmd,input=code,text=True,capture_output=True,timeout=40)
    if r.returncode:raise RuntimeError(f'{host}: {r.stderr[-2000:]}')
    return json.loads(r.stdout)


def launch(name,host,gpu,commands):
    import fcntl
    lock=ROOT/'resources/dispatch.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with open(lock,'a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        return _launch(name,host,gpu,commands)


def _launch(name,host,gpu,commands):
    policy=read('/mnt/project/server_dispatch_policy.json')
    if host not in policy['allowed_hosts'] or host in policy.get('global_dispatch_prohibitions',{}):raise PermissionError(host)
    dest=ROOT/'jobs'/name
    if (dest/'spec.json').exists():raise ValueError('Job name already used')
    for p in (ROOT/'jobs').glob('*/spec.json'):
        if (p.parent/'complete.json').exists():continue
        if not (p.parent/'launch.json').exists():continue
        s=read(p)
        old=set(s['gpu'] if isinstance(s['gpu'],list) else ([] if s['gpu'] is None else [s['gpu']]))
        requested=set(gpu if isinstance(gpu,list) else ([] if gpu is None else [gpu]))
        if s['host']==host and old&requested:raise RuntimeError(f'GPU reserved by {p.parent.name}')
    write(dest/'spec.json',dict(name=name,host=host,gpu=gpu,commands=commands))
    code=f"""import os,json,subprocess
from pathlib import Path
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
gpus={gpu if isinstance(gpu,list) else ([] if gpu is None else [gpu])!r}
for gpu in gpus:
 row=next(x for x in rows if int(x.split(',')[0])==gpu)
 if int(row.split(',')[1])>100 or int(row.split(',')[2])>5:raise RuntimeError('GPU occupied '+row)
env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
env['OPD_RUN_ROOT']={str(ROOT)!r}
f=open({str(dest/'worker.log')!r},'a')
p=subprocess.Popen([{PYTHON!r},'scripts/opd/resources.py','worker','--spec',{str(dest/'spec.json')!r}],cwd={str(SOURCE)!r},env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,start_ticks=int(Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21]),host={host!r})))
"""
    try:receipt=remote(host,code)
    except Exception as e:write(dest/'admission_failure.json',dict(error=str(e)));raise
    write(dest/'launch.json',receipt);return receipt


def worker(path):
    s=read(path);dest=Path(path).parent;owner=identity();start=time.time();results=[]
    owner['source_hashes']={str(p.relative_to(SOURCE)):sha(p) for d in ('scripts/opd','navsim/agents/recogdrive/opd') for p in (SOURCE/d).glob('*.py')}
    write(dest/'owner.json',owner)
    for i,command in enumerate(s['commands']):
        dispatched=identity();source_hashes={str(p.relative_to(SOURCE)):sha(p) for d in ('scripts/opd','navsim/agents/recogdrive/opd') for p in (SOURCE/d).glob('*.py')}
        child=subprocess.Popen([PYTHON,*command],cwd=SOURCE)
        write(dest/f'child_{i}.json',dict(pid=child.pid,start_ticks=int(Path(f'/proc/{child.pid}/stat').read_text().split()[21]),command=command,source_hashes=source_hashes,dispatch=dispatched))
        rc=child.wait();results.append(dict(command=command,returncode=rc))
        if rc:break
    write(dest/'complete.json',dict(status='COMPLETE' if all(x['returncode']==0 for x in results) else 'FAILED',seconds=time.time()-start,results=results,owner=owner))
    if rc:raise SystemExit(rc)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['worker']);p.add_argument('--spec');a=p.parse_args();worker(a.spec)
