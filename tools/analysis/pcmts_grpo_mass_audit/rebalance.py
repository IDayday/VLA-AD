"""Reassign remaining fixed tokens after early GPU shards finish; no new samples."""
from common_mass import *
import time,signal,subprocess
def main():
    while not all((OUT/'audits'/f'rollouts_rank{r}.json').exists() for r in range(6)):time.sleep(10)
    interrupted=[]
    for rank in [6,7]:
        entry=read(OUT/'manifests'/f'launch_native_native_{rank}.json')[0];pid=entry['pid'];proc=Path(f'/proc/{pid}/cmdline')
        if proc.exists():
            command=proc.read_bytes().replace(b'\0',b' ').decode();assert str(Path(__file__).parent/'sample_native.py') in command and f'--rank {rank}' in command
            os.kill(pid,signal.SIGTERM);interrupted.append(dict(pid=pid,rank=rank,command=command))
    # All interrupted outputs use atomic writes. A partially drawn file is not
    # a completed probability bank and is never counted as additional samples.
    time.sleep(5)
    missing=[s['token'] for s in scenes() if not all((OUT/'cache/rollouts'/stream/f'{s["token"]}.npz').exists() for stream in ['A','B','C'])]
    path=OUT/'manifests/rebalanced_remaining_tokens.json';save(path,dict(tokens=missing,reason='resource-only scheduling, no score or sampling-mass access',interrupted_own_workers=interrupted,created_at=utc(),protocol_hash=protocol()))
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1');runs=[]
    for rank in range(8):
        command=[sys.executable,'-u',str(Path(__file__).parent/'sample_native.py'),'--rank',str(rank),'--world','8','--token-file',str(path)]
        with open(OUT/'logs'/f'rebalanced_rank{rank}.log','a') as f:p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        runs.append(dict(pid=p.pid,rank=rank,command=command))
    save(OUT/'manifests/rebalanced_launch.json',dict(runs=runs,remaining_scene_count=len(missing),started_at=utc()))
    print('Rebalanced fixed remaining tokens',len(missing),flush=True)
if __name__=='__main__':main()
