"""Bounded independent native inference processes on one admitted GPU.

No batching, precision, sampler or RNG changes. Combined CPU scorer capacity
and actual GPU memory must be measured before choosing concurrency.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
from scripts.opd.common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--concurrency',type=int,default=4);p.add_argument('--receipt',required=True);a=p.parse_args()
    tasks=read(a.spec);remaining=list(enumerate(tasks));active={};completed=[];start=time.time()
    if not 1<=a.concurrency<=4:raise ValueError('Concurrency must be qualified and bounded')
    while remaining or active:
        while remaining and len(active)<a.concurrency:
            i,command=remaining.pop(0);identity();proc=subprocess.Popen([PYTHON,*command],cwd=SOURCE)
            active[i]=(proc,command,int(Path(f'/proc/{proc.pid}/stat').read_text().split()[21]))
        for i,(proc,command,ticks) in list(active.items()):
            rc=proc.poll()
            if rc is not None:
                completed.append(dict(index=i,command=command,pid=proc.pid,start_ticks=ticks,returncode=rc));del active[i]
                if rc:
                    # Siblings own disjoint shards; allow already-running work to
                    # finish and retain its valid files, but admit no new shard.
                    remaining=[]
        if active:time.sleep(.25)
    passed=len(completed)==len(tasks) and all(v['returncode']==0 for v in completed)
    write(a.receipt,dict(status='COMPLETE' if passed else 'FAILED',concurrency=a.concurrency,physical_GPUs=1,model_processes=len(tasks),seconds=time.time()-start,children=completed,owner=identity()))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
