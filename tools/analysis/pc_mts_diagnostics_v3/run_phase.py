"""Record commands, exit status and passive resource telemetry; no process eviction."""
import argparse,subprocess,threading
from common_v3 import *
def main(a):
    cmd=a.command
    if cmd and cmd[0]=='--':cmd=cmd[1:]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1')
    start=time.time();stop=threading.Event()
    code_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'tools/analysis/pc_mts_diagnostics_v3').glob('*.py')}
    with (OUT/'logs'/f'{a.name}.log').open('a') as log:
        p=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        save(OUT/'manifests/active_process.json',dict(pid=p.pid,command=cmd,stage=a.name,started=start))
        def monitor():
            import psutil
            with (OUT/'logs'/f'{a.name}_resources.jsonl').open('a') as f:
                while not stop.is_set():
                    try:g=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=4)
                    except Exception:g='unavailable'
                    f.write(json.dumps(dict(time=time.time(),cpu=psutil.cpu_percent(),ram=psutil.virtual_memory().used,gpus=g))+'\n');f.flush();stop.wait(10)
        t=threading.Thread(target=monitor,daemon=True);t.start();rc=p.wait();stop.set();t.join(timeout=5)
    save(OUT/'manifests'/f'{a.name}_execution.json',dict(command=cmd,returncode=rc,seconds=time.time()-start,code_sha256_at_start=code_hashes));assert rc==0,f'{a.name} failed; inspect preserved log'
    print(a.name,'completed',time.time()-start,flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('name');a.add_argument('command',nargs=argparse.REMAINDER);main(a.parse_args())
