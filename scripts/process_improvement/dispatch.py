"""Owned detached jobs, checked against the live host allowlist and GPU UUIDs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import fcntl
import subprocess
import socket
import time
from scripts.process_improvement.common import *


def remote(host, code):
    policy = read(PROJECT.parent/'server_dispatch_policy.json')
    if host not in policy['allowed_hosts'] or host in policy.get('global_dispatch_prohibitions', {}):
        raise PermissionError(host)
    cmd = [PYTHON, '-'] if host == socket.gethostname().removesuffix('-worker-0') else ['ssh', '-oBatchMode=yes', '-oConnectTimeout=10', host, 'python -']
    out = subprocess.run(cmd, input=code, text=True, capture_output=True, timeout=60)
    if out.returncode:
        raise RuntimeError(f'{host}: {out.stderr[-2000:]}')
    return json.loads(out.stdout)


def launch(name, host, gpus, commands):
    dest = ROOT/'jobs'/name
    lock = ROOT/'jobs/dispatch.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, 'a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if dest.exists():
            raise FileExistsError(f'Preserve existing job: {name}')
        for path in (ROOT/'jobs').glob('*/spec.json'):
            old = read(path)
            if (old['host'] == host and set(old['gpus']) & set(gpus) and
                    not (path.parent/'complete.json').exists() and (path.parent/'launch.json').exists()):
                raise RuntimeError(f'GPU owned by running job {path.parent.name}')
        spec = dict(name=name, host=host, gpus=gpus, commands=commands, source=str(SOURCE))
        write(dest/'spec.json', spec)
        code = f"""import json,os,subprocess
from pathlib import Path
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
gpus={gpus!r}; uuids=[]
for gpu in gpus:
 r=next(row.split(',') for row in rows if int(row.split(',')[0])==gpu)
 if int(r[2])>100 or int(r[3])>5:raise RuntimeError('GPU occupied: '+str(r))
 uuids.append(r[1].strip())
env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',PROCESS_RUN_ROOT={str(ROOT)!r})
out=open({str(dest/'worker.log')!r},'a')
p=subprocess.Popen([{PYTHON!r},'scripts/process_improvement/dispatch.py','worker','--spec',{str(dest/'spec.json')!r}],cwd={str(SOURCE)!r},env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,start_ticks=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],physical_gpu_uuids=uuids)))
"""
        try:
            receipt = remote(host, code)
        except Exception as exc:
            write(dest/'admission_failure.json', dict(error=repr(exc)))
            raise
        write(dest/'launch.json', receipt)
        return receipt


def worker(path):
    spec = read(path)
    dest = Path(path).parent
    owner = identity()
    started = time.time()
    write(dest/'owner.json', owner)
    results = []
    for index, command in enumerate(spec['commands']):
        identity()  # Recheck live host permission before each independently queued job.
        hashes = {str(p.relative_to(SOURCE)): sha(p) for folder in ('scripts/process_improvement', 'navsim/agents/recogdrive/process_improvement')
                  for p in (SOURCE/folder).glob('*.py')}
        proc = subprocess.Popen([PYTHON, *command], cwd=SOURCE)
        write(dest/f'child_{index}.json', dict(pid=proc.pid, command=command, source_hashes=hashes))
        rc = proc.wait()
        results.append(dict(command=command, returncode=rc))
        if rc:
            break
    passed = len(results) == len(spec['commands']) and all(r['returncode'] == 0 for r in results)
    write(dest/'complete.json', dict(status='COMPLETE' if passed else 'FAILED', results=results, owner=owner, seconds=time.time()-started))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['worker'])
    parser.add_argument('--spec', required=True)
    worker(parser.parse_args().spec)
