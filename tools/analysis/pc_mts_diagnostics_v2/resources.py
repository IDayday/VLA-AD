"""Stop/restart only disposable gpu_stress jobs; never touch real experiments."""
import argparse,subprocess,psutil,time
from v2_common import *

def main(a):
    path=OUT/'manifests/pressure_snapshot.json'
    if a.action=='stop':
        jobs=[];procs=[]
        for p in psutil.process_iter(['pid','cmdline']):
            cmd=p.info['cmdline'] or []
            if '/mnt/project/gpu_stress.py' not in cmd:continue
            jobs.append(dict(pid=p.pid,command=cmd,cuda_visible_devices=p.environ().get('CUDA_VISIBLE_DEVICES'),cwd=p.cwd()))
            procs.extend(p.children(recursive=True));procs.append(p)
        if jobs:save(path,dict(jobs=jobs,stopped_at=time.time()))
        for p in procs:
            try:p.terminate()
            except psutil.NoSuchProcess:pass
        _,alive=psutil.wait_procs(procs,timeout=3)
        for p in alive:p.kill()
        print('Stopped only gpu_stress:',len(jobs),flush=True)
    else:
        if not path.exists():return
        running=[p for p in psutil.process_iter(['cmdline']) if '/mnt/project/gpu_stress.py' in (p.info['cmdline'] or [])]
        if running:print('Pressure already running; no duplicate restart');return
        jobs=[]
        for i,row in enumerate(read(path)['jobs']):
            env=dict(os.environ)
            if row['cuda_visible_devices'] is not None:env['CUDA_VISIBLE_DEVICES']=row['cuda_visible_devices']
            with (OUT/'logs'/f'pressure_restored_{i}.log').open('a') as f:p=subprocess.Popen(row['command'],cwd=row['cwd'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            jobs.append(dict(pid=p.pid,cuda_visible_devices=row['cuda_visible_devices']))
        save(OUT/'manifests/pressure_restored.json',dict(jobs=jobs));print('Restored pressure jobs',len(jobs))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stop','restore']);main(p.parse_args())
