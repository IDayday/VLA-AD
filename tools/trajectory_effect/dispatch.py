"""Explicit owned jobs on currently authorized, idle devices; no auto retries."""
from __future__ import annotations
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import fcntl
import shlex
from .common import *


def run_remote(host, code):
    local = socket.gethostname().removesuffix('-worker-0')
    command = [PYTHON, '-'] if host == local else ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, 'python -']
    r = subprocess.run(command, input=code, text=True, capture_output=True, timeout=40)
    if r.returncode: raise RuntimeError(f'{host}: {r.stderr[-3000:]} {r.stdout[-1000:]}')
    return json.loads(r.stdout)


def inventory():
    policy = read('/mnt/project/server_dispatch_policy.json')
    code = """import subprocess,json,socket
print(json.dumps(dict(host=socket.gethostname(),gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True),processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True))))
"""
    def one(h):
        try: return h, run_remote(h, code)
        except Exception as e: return h, dict(error=str(e))
    with ThreadPoolExecutor(8) as pool: value = dict(pool.map(one, policy['allowed_hosts']))
    atomic_json(ROOT/'resources'/f'inventory_{int(time.time())}.json', value)
    return value


def launch(spec):
    policy = read('/mnt/project/server_dispatch_policy.json')
    host = spec['host']
    if host not in policy['allowed_hosts'] or host in policy.get('global_dispatch_prohibitions', {}):
        raise PermissionError(host)
    path = ROOT/'jobs'/spec['name']/'spec.json'
    if path.exists(): raise ValueError(f'Job identity already exists: {path}')
    atomic_json(path, spec)
    code = f"""import subprocess,os,json
from pathlib import Path
source={str(SOURCE)!r}; job={str(path.parent)!r}; gpu={spec.get('gpu')!r}
if gpu is not None:
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
 row=next(x for x in rows if int(x.split(',')[0])==gpu)
 if int(row.split(',')[1])>100 or int(row.split(',')[2])>5: raise RuntimeError('Device became occupied: '+row)
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
env['CUDA_VISIBLE_DEVICES']=str(gpu) if gpu is not None else ''
f=open(job+'/worker.log','a')
p=subprocess.Popen([{PYTHON!r},'-m','tools.trajectory_effect.dispatch','worker','--spec',job+'/spec.json'],cwd=source,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,start_ticks=int(Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21]),host={host!r},job=job)))
"""
    result = run_remote(host, code)
    atomic_json(path.parent/'launch.json', result)
    return result


def worker(spec_path):
    require_host(); spec=read(spec_path); dest=Path(spec_path).parent
    lock=open(dest/'worker.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    owner=runtime_identity();owner['spec']=spec
    owner['source_hashes']={str(p.relative_to(SOURCE)):sha(p) for p in (SOURCE/'tools/trajectory_effect').glob('*.py')}
    atomic_json(dest/'owner.json',owner);start=time.time();results=[]
    for command in spec['commands']:
        require_host()
        t=time.time();p=subprocess.Popen([PYTHON,'-m',*command],cwd=SOURCE)
        atomic_json(dest/'child.json',dict(pid=p.pid,start_ticks=int(Path(f'/proc/{p.pid}/stat').read_text().split()[21]),command=command))
        rc=p.wait();results.append(dict(command=command,returncode=rc,seconds=time.time()-t))
        if rc:break
    atomic_json(dest/'complete.json',dict(status='COMPLETE' if all(r['returncode']==0 for r in results) else 'FAILED',
        seconds=time.time()-start,results=results,owner=owner))
    if results[-1]['returncode']:raise SystemExit(results[-1]['returncode'])


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['inventory','launch','worker']);p.add_argument('--spec')
    args=p.parse_args()
    if args.action=='inventory':print(json.dumps(inventory(),indent=2))
    elif args.action=='launch':print(json.dumps(launch(read(args.spec)),indent=2))
    else:worker(args.spec)


if __name__=='__main__':main()
