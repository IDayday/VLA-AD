"""Preempt only the user-authorized, container-local known idle-pressure workers."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
from concurrent.futures import ThreadPoolExecutor
from scripts.opd.common import *
from scripts.opd.resources import remote

TAG='recogdrive-opd-20261020-official-v2'
HOSTS=['training-vla-zt','training-rl-zt2','training-vlawm-tsinghua-zt2','training-vlawm-tsinghua-zt3','training-vlawm-tsinghua-zt5']

def operate(host,action):
    policy=read('/mnt/project/server_dispatch_policy.json')
    if host not in policy['allowed_hosts'] or host in policy.get('global_dispatch_prohibitions',{}):raise PermissionError(host)
    result=remote(host,"""import os,json,signal,ctypes,hashlib,time
from pathlib import Path
p=Path('/tmp/zt-idle-pressure-v1');pause=p/'PAUSED';tag=TAG
prior=pause.read_text() if pause.exists() else None
released=[]
if ACTION=='pause':
 if prior not in (None,tag):raise RuntimeError('Another operator owns pause marker')
 pause.write_text(tag)
 for q in Path('/proc').iterdir():
  if not q.name.isdigit():continue
  try:
   args=(q/'cmdline').read_bytes().split(b'\\0')
   if len(args)!=5 or args[1]!=str(p/'agent.py').encode() or args[2]!=b'--worker' or not args[3].startswith(b'GPU-'):continue
   if q.stat().st_uid!=os.getuid():continue
   pid=int(q.name);start=int((q/'stat').read_text().split()[21]);libc=ctypes.CDLL(None,use_errno=True);fd=libc.syscall(434,pid,0)
   if fd<0:continue
   try:
    if (q/'cmdline').read_bytes().split(b'\\0')!=args or int((q/'stat').read_text().split()[21])!=start:continue
    libc.syscall(424,fd,int(signal.SIGTERM),0,0)
    released.append(dict(pid=pid,start_ticks=start,argv=[a.decode() for a in args if a],uid=q.stat().st_uid))
   finally:os.close(fd)
  except FileNotFoundError:pass
elif ACTION=='resume':
 if prior!=tag:raise RuntimeError('Own pause marker missing or changed; refuse removal')
 pause.unlink()
else:raise ValueError(ACTION)
print(json.dumps(dict(host=os.uname().nodename,prior_pause=prior,tag=tag,action=ACTION,released=released,
 agent_sha256=hashlib.sha256((p/'agent.py').read_bytes()).hexdigest(),time=time.time(),
 resume_policy='Existing original automation reobserves fresh30min mean<15%; no manual filler launch')))
""".replace('TAG',repr(TAG)).replace('ACTION',repr(action)))
    dest=ROOT/'resources'/f'{host}_pressure_{action}_v2.json';write(dest,result);dest.chmod(0o600)
    return dict(host=host,action=action,released=len(result['released']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['pause','resume']);a=p.parse_args()
    with ThreadPoolExecutor(4) as pool:
        for value in pool.map(lambda host:operate(host,a.action),HOSTS):print(value,flush=True)
