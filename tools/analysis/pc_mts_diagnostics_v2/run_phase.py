"""Record actual commands, resource samples, return codes and wall-clock time."""
import argparse,subprocess,time,threading
from v2_common import *

def main(a):
    make_dirs();cmd=a.command
    if cmd and cmd[0]=='--':cmd=cmd[1:]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1')
    start=time.time();stop=threading.Event()
    with (OUT/'logs'/f'{a.name}.log').open('a') as log:
        p=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        save(OUT/'manifests/active_process.json',dict(pid=p.pid,command=cmd,stage=a.name,started=start))
        def monitor():
            import psutil
            with (OUT/'logs'/f'{a.name}_resources.jsonl').open('a') as f:
                while not stop.is_set():
                    try:g=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=4)
                    except Exception:g='unavailable'
                    f.write(json.dumps(dict(time=time.time(),cpu_percent=psutil.cpu_percent(),ram_used=psutil.virtual_memory().used,gpus=g))+'\n');f.flush();stop.wait(3)
        t=threading.Thread(target=monitor,daemon=True);t.start();rc=p.wait();stop.set();t.join(timeout=5)
    save(OUT/'manifests'/f'{a.name}_execution.json',dict(command=cmd,returncode=rc,wall_seconds=time.time()-start));assert rc==0,f'{a.name} failed; see its log'
    print(a.name,'completed',time.time()-start,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('command',nargs=argparse.REMAINDER);main(p.parse_args())
