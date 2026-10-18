"""Finished/failed jobs, actual configurations and conservative GPU-time accounting."""
from __future__ import annotations
import argparse
import shlex
from .common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');args=p.parse_args()
    jobs=[];gpu_hours=0.;running=[];now=time.time();commands=[]
    for specfile in sorted((ROOT/'jobs').glob('*/spec.json')):
        s=read(specfile);dest=specfile.parent
        if not (dest/'launch.json').exists():
            jobs.append(dict(name=s['name'],status='NOT_LAUNCHED',spec=s));continue
        finish=read(dest/'complete.json') if (dest/'complete.json').exists() else None
        owner=read(dest/'owner.json') if (dest/'owner.json').exists() else {}
        elapsed=finish['seconds'] if finish else now-owner.get('started_unix',now)
        devices=s.get('gpus',[s['gpu']] if s.get('gpu') is not None else [])
        cost=len(devices)*elapsed/3600;gpu_hours+=cost
        status=finish['status'] if finish else 'RUNNING'
        if not finish:running.append(s['name'])
        jobs.append(dict(name=s['name'],status=status,host=s['host'],devices=devices,seconds=elapsed,
            reserved_gpu_hours=cost,spec=s,launch=read(dest/'launch.json'),owner=owner,
            results=finish['results'] if finish else None))
        for command in s['commands']:
            commands.append('# '+s['name']+' on '+s['host']+'\n'+shlex.join(['env','CUDA_VISIBLE_DEVICES='+','.join(map(str,devices)),PYTHON,'-m',*command]))
    if args.final and running:raise RuntimeError('Owned workers still running: '+','.join(running))
    sizes={};total=0
    for root,dirs,files in os.walk(ROOT):
        for name in files:
            path=Path(root)/name
            if path.is_symlink():continue
            size=path.stat().st_size;total+=size;key=path.relative_to(ROOT).parts[0];sizes[key]=sizes.get(key,0)+size
    audit=read(ROOT/'pilot/inference_complete.json');score=read(ROOT/'pilot/score_complete.json')
    atomic_json(ROOT/'analysis/accounting.json',dict(status='FINAL' if args.final else 'LIVE',reserved_gpu_hours=gpu_hours,
        definition='GPU count times owned worker wall time, including startup and dependency waits; upper bound on active GPU compute time.',
        pilot_gpu_seconds=sum(read(ROOT/'pilot'/f'{k}_complete.json').get('seconds',0) for k in ['inference','train','joint']),
        storage_bytes=total,storage_by_directory=sizes,running=running,jobs=jobs,
        protocol_sha256=protocol()[1],commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()))
    (ROOT/'analysis/executed_commands.sh').write_text('# Actual executed module commands; GPU assignments are historical.\n# Workdir: '+str(SOURCE)+'\n\n'+'\n\n'.join(commands)+'\n')
    print(json.dumps(dict(reserved_gpu_hours=gpu_hours,storage_GiB=total/1024**3,running=len(running),jobs=len(jobs))))


if __name__=='__main__':main()
