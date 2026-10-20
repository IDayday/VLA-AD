"""Bounded independent model processes within one admitted physical GPU lease."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import subprocess
import time
from scripts.process_improvement.common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--receipt',required=True)
    p.add_argument('--concurrency',type=int,default=3);a=p.parse_args()
    if not 1<=a.concurrency<=3:raise ValueError('Unqualified inference process concurrency')
    tasks=list(enumerate(read(a.spec)));active={};results=[];started=time.time()
    while tasks or active:
        while tasks and len(active)<a.concurrency:
            index,command=tasks.pop(0);identity();proc=subprocess.Popen([PYTHON,*command],cwd=SOURCE)
            active[index]=(proc,command)
        for index,(proc,command) in list(active.items()):
            rc=proc.poll()
            if rc is not None:
                results.append(dict(index=index,command=command,pid=proc.pid,returncode=rc));del active[index]
                if rc:tasks=[]
        if active:time.sleep(.2)
    passed=len(results)==len(read(a.spec)) and all(r['returncode']==0 for r in results)
    write(a.receipt,dict(status='COMPLETE' if passed else 'FAILED',results=results,
          physical_gpu_count=1,model_processes=a.concurrency,seconds=time.time()-started,owner=identity()))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
