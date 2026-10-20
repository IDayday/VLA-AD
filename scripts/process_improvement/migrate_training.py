"""Pause only an exact owned training descendant; resume its immutable lineage."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import time
from scripts.process_improvement.common import *
from scripts.process_improvement.dispatch import remote


def main():
    p=argparse.ArgumentParser();p.add_argument('--old-job',required=True);p.add_argument('--run',required=True);a=p.parse_args()
    old=ROOT/'jobs'/a.old_job;spec=read(old/'spec.json');receipt=read(old/'launch.json')
    dest=ROOT/'runs'/a.run
    assert not (dest/'paused.json').exists() and not (dest/'complete.json').exists()
    code=f"""import os,json,signal
from pathlib import Path
root={receipt['pid']!r};proc=Path('/proc')/str(root)
assert int((proc/'stat').read_text().split()[21])==int({receipt['start_ticks']!r})
assert {str(old/'spec.json').encode()!r} in (proc/'cmdline').read_bytes().split(b'\\0')
matches=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().split(b'\\0')
  if b'scripts/process_improvement/train.py' not in args or {a.run.encode()!r} not in args:continue
  assert p.stat().st_uid==os.getuid()
  parent=int((p/'stat').read_text().split()[3]);ancestry=[]
  while parent not in (0,1,root):
   ancestry.append(parent);parent=int((Path('/proc')/str(parent)/'stat').read_text().split()[3])
  if parent==root:matches.append((int(p.name),int((p/'stat').read_text().split()[21])))
 except FileNotFoundError:continue
assert len(matches)==1,matches
pid,start=matches[0];fd=os.pidfd_open(pid)
try:
 assert int((Path('/proc')/str(pid)/'stat').read_text().split()[21])==start
 signal.pidfd_send_signal(fd,signal.SIGUSR1,None,0)
finally:os.close(fd)
print(json.dumps(dict(training_pid=pid,start_ticks=start,signal='SIGUSR1',owned_dispatch_ancestor=root,unknown_load_untouched=True)))
"""
    request=remote(spec['host'],code)
    write(dest/'migration_request.json',dict(request,reason='Unidentified extra GPU load; move to verified idle authorized device',old_host=spec['host'],new_owner=identity()))
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        if (dest/'paused.json').exists() and (old/'complete.json').exists():break
        time.sleep(1)
    else:raise TimeoutError('Owned training did not acknowledge its native checkpoint boundary')
    assert read(old/'complete.json')['status']=='COMPLETE'
    cfg=read(dest/'config.json');assert cfg['world']==1 and cfg['method']=='OPD' and cfg['steps']==9361
    write(dest/'migration_checkpoint.json',dict(paused=read(dest/'paused.json'),checkpoint_sha256=sha(dest/'latest.pt'),
          original_configuration=cfg,restoration='Same world size, global/micro batch, native precision, optimizer, all RNG and sampler cursor'))
    subprocess.run([PYTHON,'scripts/process_improvement/train.py','--method','OPD','--name',a.run,
                    '--split',cfg['split'],'--seed',str(cfg['seed']),'--lr',str(cfg['lr']),
                    '--epochs','4','--stop-after','4681','--resume'],cwd=SOURCE,check=True)


if __name__=='__main__':main()
