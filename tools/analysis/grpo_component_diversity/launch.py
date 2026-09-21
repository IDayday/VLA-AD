"""Bounded eight-GPU read-only sampling, with resumable CPU scoring."""
from shared import *
import subprocess,time
PY='/root/miniconda3/envs/navsim/bin/python'
def start(script,args,log,gpu=None):
    env=dict(os.environ)
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    f=open(OUT/'logs'/log,'a')
    return subprocess.Popen([PY,str(ROOT/'tools/analysis/grpo_component_diversity'/script)]+args,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
def main():
    identity();jobs=[]
    for i,m in enumerate(CFG['models']):
        assert read(OUT/'audits'/f'sampler_parity_{m}.json')['status']=='PASS'
        for rank in range(4):
            p=start('sample_extra.py',['--model',m,'--rank',str(rank),'--world','4'],f'sample_{m}_{rank}.log',i*4+rank);jobs.append(p)
    score=start('score_extra.py',['--workers','80','--watch'],'score.log')
    save(OUT/'manifests/launch.json',dict(gpu_pids=[p.pid for p in jobs],scoring_pid=score.pid,gpus=8,scoring_workers=80,optimizer_updates=0))
    for p in jobs:assert p.wait()==0,('sampling failed',p.pid)
    save(OUT/'audits/gpu_sampling_complete.json',dict(status='PASS'))
    assert score.wait()==0,'scoring failed'
    for m in CFG['models']:
        assert len(list((OUT/'cache/scores/rollouts'/m).glob('*.npz')))==5000
    save(OUT/'audits/all_sampling_scoring_complete.json',dict(status='PASS',new_draws=640000,total_draws=1280000))
    p=start('native_audit.py',[],'native_audit.log',0);assert p.wait()==0,'native audit failed'
    p=start('analyze_components.py',['--workers','32'],'analyze.log');assert p.wait()==0,'analysis failed'
if __name__=='__main__':main()
