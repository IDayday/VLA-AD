"""One full eight-GPU formal run per host; atomic shared work queue."""
from common_r import *
import subprocess
def main():
    identity();assert (OUT/'manifests'/f'host_parity_{socket.gethostname()}.json').exists()
    jobs=[(m,sd) for m in CFG['methods'] for sd in CFG['seeds']]
    directory=OUT/'queue/train';directory.mkdir(parents=True,exist_ok=True)
    for method,sd in jobs:
        key=run_name(method,sd);lock=directory/f'{key}.lock'
        if (OUT/'manifests'/f'train_{key}.json').exists():continue
        try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:continue
        os.write(fd,json.dumps(dict(host=socket.gethostname(),pid=os.getpid(),python=sys.executable)).encode());os.close(fd)
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(ROOT/'tools/analysis/pc_mts_v3_stage3_rerun/train_r.py'),method,str(sd)]
        print('START',key,socket.gethostname(),flush=True)
        with open(OUT/'logs'/f'train_{key}.log','w') as f:result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            save(directory/f'{key}.failed.json',dict(host=socket.gethostname(),exit_code=result.returncode));raise RuntimeError(f'Training failed: {key}')
        print('DONE',key,flush=True)
    print('TRAIN QUEUE DONE',socket.gethostname(),flush=True)
if __name__=='__main__':main()
