"""Detached CPU coordinator lifecycle; never signals its scientific GPU jobs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import ctypes
import signal
import time
from scripts.process_improvement.common import *


def require_restart_authorization(action):
    if action in ['start','restart'] and (ROOT/'controller/user_stop.json').exists():
        raise RuntimeError('User stopped this study; do not restart without new explicit authorization')


def owned():
    path=ROOT/'controller/owner.json'
    if not path.exists():return None
    value=read(path);proc=Path('/proc')/str(value['pid'])
    try:
        status=(proc/'stat').read_text().split()
        if int(status[21])!=int(value['start_ticks']) or status[2] in ['Z','X']:return None
        if proc.stat().st_uid!=os.getuid():raise PermissionError('Controller UID changed')
        if b'scripts/process_improvement/coordinator.py' not in (proc/'cmdline').read_bytes().split(b'\0'):
            raise PermissionError('Controller command identity changed')
    except FileNotFoundError:return None
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','stop','status','restart']);a=p.parse_args();identity()
    require_restart_authorization(a.action)
    prior=owned()
    if a.action=='status':print(prior or {'status':'NOT_RUNNING'});return
    if a.action in ['stop','restart'] and prior:
        deadline=time.monotonic()+40
        while time.monotonic()<deadline:
            age=time.time()-(ROOT/'controller/state.json').stat().st_mtime
            if read(ROOT/'controller/phase.json')['phase']=='idle' and 1<age<24:break
            time.sleep(.25)
        else:raise TimeoutError('Coordinator is making an atomic selection; do not interrupt it')
        # Revalidate after waiting. The process may have naturally exited.
        current=owned();assert current==prior
        # The pinned navsim Python lacks os.pidfd_open; use the same audited
        # Linux pidfd syscalls as the existing resource controller.
        libc=ctypes.CDLL(None,use_errno=True);fd=int(libc.syscall(434,prior['pid'],0))
        if fd<0:raise OSError(ctypes.get_errno(),'pidfd_open failed; no signal sent')
        try:
            assert owned()==prior
            if libc.syscall(424,fd,int(signal.SIGTERM),0,0)<0:
                raise OSError(ctypes.get_errno(),'pidfd_send_signal failed')
        finally:os.close(fd)
        for _ in range(100):
            # During exit, cmdline can disappear before the state becomes Z.
            # Ownership was checked immediately before SIGTERM; now observe
            # only liveness and never send another signal to this PID.
            try:status=(Path('/proc')/str(prior['pid'])/'stat').read_text().split()
            except FileNotFoundError:break
            if int(status[21])!=int(prior['start_ticks']) or status[2] in ['Z','X']:break
            time.sleep(.05)
        else:raise TimeoutError('Coordinator has not exited; refuse a duplicate start')
        write(ROOT/'controller'/f"stop_{prior['pid']}.json",dict(prior,signal='SIGTERM',scientific_children_untouched=True))
    elif a.action=='start' and prior:raise RuntimeError('An owned coordinator is already running')
    if a.action=='stop':return
    folder=ROOT/'controller';folder.mkdir(parents=True,exist_ok=True)
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    with (folder/'controller.log').open('a') as log:
        proc=subprocess.Popen([PYTHON,'scripts/process_improvement/coordinator.py','--formal-controls'],cwd=SOURCE,
                env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(pid=proc.pid,start_ticks=Path('/proc')/str(proc.pid)/'stat')
    value['start_ticks']=Path(value['start_ticks']).read_text().split()[21]
    value.update(detached=True,source_sha256=sha(SOURCE/'scripts/process_improvement/coordinator.py'),parent_action=a.action)
    write(folder/'detached_launch.json',value)
    for _ in range(100):
        if proc.poll() is not None:raise RuntimeError('Detached coordinator exited before acknowledgement')
        if (folder/'owner.json').exists() and read(folder/'owner.json')['pid']==proc.pid:break
        time.sleep(.05)
    else:raise TimeoutError('Coordinator acknowledgement not received')
    print(value,flush=True)


if __name__=='__main__':main()
