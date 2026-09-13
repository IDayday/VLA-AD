"""Independent runs scheduled using free GPUs; no other process is touched."""
import argparse,subprocess
from common_v3 import *
def main(stage):
    status=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True);gpu=[]
    for line in status.splitlines():
        i,free,util=map(int,map(str.strip,line.split(',')))
        if free>=16000:gpu.append(i)
    assert gpu,'No GPU has the required free memory; no unrelated jobs will be stopped'
    # Two small action-head runs fit comfortably on an idle 80GB device.
    # At most 12 experiments, allocated round-robin over all available GPUs.
    jobs=[(m,s) for s in CFG['training']['seeds'] for m in CFG['training']['methods']]
    maxjobs=min(len(jobs),2*len(gpu));active=[];results=[];start=time.time();env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1')
    save(OUT/'manifests'/f'{stage}_resource_plan.json',dict(identity=identity(),status=status,available_gpus=gpu,max_concurrent_jobs=maxjobs,max_per_gpu=2,reason='34.3M action-head parameters; maximum two runs/device, VLM uses cached observations',stop_other_processes=False))
    assignment=0
    while jobs or active:
        while jobs and len(active)<maxjobs:
            method,sd=jobs.pop(0);device=gpu[assignment%len(gpu)];assignment+=1
            used=sum(r['device']==device for r in active)
            if used>=2:
                available=[i for i in gpu if sum(r['device']==i for r in active)<2];device=available[0]
            log=(OUT/'logs'/f'train_{stage}_{method}_{sd}.log').open('a');cmd=[PY,str(ROOT/'tools/analysis/pc_mts_diagnostics_v3/train.py'),stage,method,str(sd)];p=subprocess.Popen(cmd,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(device)),stdout=log,stderr=subprocess.STDOUT)
            active.append(dict(p=p,log=log,method=method,seed=sd,device=device,command=cmd));print('Launched',stage,method,sd,'GPU',device,flush=True)
        for r in active[:]:
            code=r['p'].poll()
            if code is not None:
                r['log'].close();results.append({k:v for k,v in r.items() if k not in ['p','log']}|{'returncode':code});active.remove(r);print('Finished',r['method'],r['seed'],code,flush=True)
        if active:time.sleep(5)
    save(OUT/'manifests'/f'{stage}_training_execution.json',dict(identity=identity(),jobs=results,seconds=time.time()-start));assert all(r['returncode']==0 for r in results),results
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['sft','grpo']);main(a.parse_args().stage)
