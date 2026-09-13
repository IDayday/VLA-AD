"""Eight GPU workers; retain completed scenes and fail on any subprocess error."""
import concurrent.futures,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/project/VLA-AD');OUT=ROOT/'outputs/policy_distribution_5ckpt_1000_20260912'
PY='/root/miniconda3/envs/navsim/bin/python';RUN=ROOT/'scripts/evaluation/distribution_audit/run.py'
MODELS=[x['id'] for x in json.load(open(OUT/'models.json'))]

def worker(gpu):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    for phase,model in [('features',None)]+[('sample',m) for m in MODELS]:
        tokens=json.load(open(OUT/'tokens.json'))[gpu::8]
        if phase=='features' and (OUT/f'features_shard{gpu}.json').is_file() and all((OUT/'features'/f'{t}.pt').is_file() for t in tokens):continue
        if phase=='sample' and all((OUT/'predictions'/model/f'{t}.npz').is_file() for t in tokens):continue
        label=f'{phase}_{model or "shared"}_shard{gpu}'
        cmd=[PY,str(RUN),phase,'--out',str(OUT),'--shard',str(gpu),'--shards','8']
        if model:cmd+=['--model',model]
        with open(OUT/(label+'.log'),'a') as log:
            p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            print(json.dumps(dict(event='start',gpu=gpu,label=label,pid=p.pid,time=time.time())),flush=True)
            rc=p.wait()
        if rc:raise RuntimeError(f'{label} exited {rc}')
        print(json.dumps(dict(event='complete',gpu=gpu,label=label,time=time.time())),flush=True)

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    for f in concurrent.futures.as_completed([pool.submit(worker,g) for g in range(8)]):f.result()
(OUT/'sampling_complete.json').write_text(json.dumps({'complete':True,'time':time.time()}))
