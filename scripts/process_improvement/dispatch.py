"""Owned detached jobs, checked against the live host allowlist and GPU UUIDs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import fcntl
import subprocess
import socket
import time
import tarfile
import io
import hashlib
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


def archive_launch_sources(dest,index):
    """Retain the actual small source bytes, including uncommitted development.

    This is a launch snapshot, not a claim of execution in an isolated checkout.
    It complements the HEAD ID and hashes of older jobs, without inventing
    archives for those already launched. Models, caches and secrets are absent.
    """
    folders=['scripts/process_improvement','scripts/opd','navsim/agents/recogdrive/process_improvement',
             'navsim/agents/recogdrive/opd']
    paths={p for folder in folders for p in (SOURCE/folder).glob('*.py')}
    for name in ['navsim/agents/recogdrive/recogdrive_agent.py',
                 'navsim/agents/recogdrive/recogdrive_diffusion_planner.py',
                 'navsim/agents/recogdrive/recogdrive_dit.py','navsim/evaluate/pdm_score.py',
                 'navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py']:
        paths.add(SOURCE/name)
    paths.update((SOURCE/'configs/process_improvement').glob('*.yaml'))
    payloads={str(p.relative_to(SOURCE)):p.read_bytes() for p in sorted(paths) if p.is_file()}
    hashes={name:hashlib.sha256(value).hexdigest() for name,value in payloads.items()}
    target=Path(dest)/f'source_snapshot_{index}.tar.gz'
    if target.exists():raise FileExistsError('Launch source snapshot is immutable')
    with tarfile.open(target,'w:gz') as archive:
        for name,value in payloads.items():
            info=tarfile.TarInfo(name);info.size=len(value);info.mode=0o644
            archive.addfile(info,io.BytesIO(value))
    return hashes,dict(path=str(target),sha256=sha(target),files=len(payloads),
                       scope='Source bytes at launch; shared-checkout execution; historical jobs retain original hashes only')


def worker(path):
    spec = read(path)
    dest = Path(path).parent
    owner = identity()
    started = time.time()
    write(dest/'owner.json', owner)
    results = []
    for index, command in enumerate(spec['commands']):
        identity()  # Recheck live host permission before each independently queued job.
        try:
            hashes,archive=archive_launch_sources(dest,index)
            commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
        except Exception as exc:
            results.append(dict(command=command,returncode=None,prelaunch_error=repr(exc)))
            break # Provenance failure must not start an unrecorded child.
        proc = subprocess.Popen([PYTHON, *command], cwd=SOURCE)
        write(dest/f'child_{index}.json', dict(pid=proc.pid, command=command, source_hashes=hashes,
                                             source_archive=archive,source_HEAD_at_launch=commit))
        rc = proc.wait()
        results.append(dict(command=command, returncode=rc))
        if rc:
            break
    passed = len(results) == len(spec['commands']) and all(r['returncode'] == 0 for r in results)
    write(dest/'complete.json', dict(status='COMPLETE' if passed else 'FAILED', results=results, owner=owner, seconds=time.time()-started))
    if not passed:
        raise SystemExit(1)


def stop_owned(name,reason):
    """Stop only this registry's exact process group; retain an explicit receipt."""
    dest=ROOT/'jobs'/name;spec=read(dest/'spec.json');receipt=read(dest/'launch.json')
    if (dest/'complete.json').exists():raise ValueError('Job is already finished')
    code=f"""import os,json,signal,time
from pathlib import Path
pid={receipt['pid']!r}; path=Path('/proc')/str(pid)
args=(path/'cmdline').read_bytes().split(b'\\0')
assert int((path/'stat').read_text().split()[21])==int({receipt['start_ticks']!r})
assert path.stat().st_uid==os.getuid()
assert b'scripts/process_improvement/dispatch.py' in args and {str(dest/'spec.json').encode()!r} in args
assert os.getpgid(pid)==pid
elapsed=time.monotonic()-int((path/'stat').read_text().split()[21])/os.sysconf('SC_CLK_TCK')
os.killpg(pid,signal.SIGTERM)
print(json.dumps(dict(parent_pid=pid,start_ticks={receipt['start_ticks']!r},reason={reason!r},allocated_seconds=elapsed,signal='SIGTERM',exact_owned_process_group=True)))
"""
    result=remote(spec['host'],code)
    write(dest/'termination.json',result)
    write(dest/'complete.json',dict(status='INTERRUPTED',reason=reason,termination=result,results=[]))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['worker'])
    parser.add_argument('--spec', required=True)
    worker(parser.parse_args().spec)
