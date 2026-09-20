"""Resumable one-process-per-GPU orchestration; no unrelated processes touched."""
from common_matched import *
import subprocess,shlex
PY='/root/miniconda3/envs/navsim/bin/python';CODE=WORK/'tools/analysis/psi_matched_sft'
HOSTS=[None,'training-rl-zt4','training-vla-zt2','training-vla-zt3']
def launch(cmd,label,gpu=None):
    env=dict(os.environ)
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    log=OUT/'logs'/f'{label}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as f:p=subprocess.Popen(cmd,cwd=WORK,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    save(OUT/'manifests/jobs'/f'{label}.json',dict(pid=p.pid,cmd=list(map(str,cmd)),gpu=gpu,started=time.time()))
    return p
def launch_evaluation(name,step,gpu,rank,host):
    cmd=[PY,str(CODE/'evaluate.py'),name,'--step',str(step),'--rank',str(rank),'--world',str(len(HOSTS))]
    if host:
        remote='cd '+shlex.quote(str(WORK))+' && CUDA_VISIBLE_DEVICES='+str(gpu)+' '+shlex.join(cmd)
        cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,remote]
    return launch(cmd,f'eval_{name}_step{step}_rank{rank}',gpu if not host else None)
def main():
    while not (OUT/'manifests/selection_frozen.json').exists() or not all(feature_path(s['token']).exists() for s in scenes('train')):time.sleep(5)
    subprocess.run([PY,str(CODE/'test_matched.py')],cwd=WORK,check=True)
    jobs={};attached={}
    for gpu,(method,runseed) in enumerate((m,s) for m in CFG['methods'] for s in CFG['train_seeds']):
        name=f'{method}_seed{runseed}'
        if (OUT/'audits'/f'train_{name}.json').exists():continue
        previous=OUT/'manifests/jobs'/f'train_{name}.json'
        if previous.exists():
            pid=read(previous)['pid'];path=Path(f'/proc/{pid}/cmdline')
            if path.exists() and str(CODE/'train.py') in path.read_text():attached[name]=pid;continue
            raise RuntimeError(f'{name} has a previous incomplete run; inspect instead of silently restarting')
        jobs[gpu]=(name,launch([PY,str(CODE/'train.py'),method,str(runseed)],f'train_{name}',gpu))
    for gpu,(name,p) in jobs.items():
        code=p.wait()
        if code:raise RuntimeError(f'{name} training failed: {code}')
    for name,pid in attached.items():
        while not (OUT/'audits'/f'train_{name}.json').exists():
            path=Path(f'/proc/{pid}/cmdline')
            if not path.exists() or str(CODE/'train.py') not in path.read_text():raise RuntimeError(f'Attached run {name} stopped without completion audit')
            time.sleep(5)
    save(OUT/'audits/training_complete.json',dict(status='PASS',protocol_hash=identity(),runs=8,optimizer_updates_per_run=512))
    scorer=launch([PY,str(CODE/'score.py')],'scoring')
    # Training and all initialization choices finish before evaluation summaries.
    jobs=[]
    for gpu,(method,runseed) in enumerate((m,s) for m in CFG['methods'] for s in CFG['train_seeds']):
        name=f'{method}_seed{runseed}'
        for rank,host in enumerate(HOSTS):jobs.append(launch_evaluation(name,512,gpu,rank,host))
    for p in jobs:
        if p.wait():raise RuntimeError('Final evaluation failed')
    jobs=[]
    for gpu,(method,runseed) in enumerate((m,s) for m in CFG['methods'] for s in CFG['train_seeds']):
        name=f'{method}_seed{runseed}'
        for rank,host in enumerate(HOSTS):jobs.append(launch_evaluation(name,128,gpu,rank,host))
    for p in jobs:
        if p.wait():raise RuntimeError('Early evaluation failed')
    jobs=[launch([PY,str(CODE/'evaluate.py'),'official_il','--rank',str(g),'--world','8'],f'baseline_{g}',g) for g in range(8)]
    for p in jobs:
        if p.wait():raise RuntimeError('Baseline probe failed')
    save(OUT/'audits/gpu_sampling_complete.json',dict(status='PASS',protocol_hash=identity()))
    if scorer.wait():raise RuntimeError('Scoring failed')
    save(OUT/'audits/data_ready.json',dict(status='PASS',protocol_hash=identity()))
if __name__=='__main__':main()
