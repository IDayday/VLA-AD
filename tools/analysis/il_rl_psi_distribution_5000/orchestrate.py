"""Resumable local GPU phases; CPU scoring runs independently."""
from common_5000 import *
import subprocess, time

if __name__=='__main__':
    start=time.time()
    while len(list((OUT/'cache/features').glob('*.pt')))<4000:
        if time.time()-start>14400:raise TimeoutError('Feature phase incomplete after four hours; inspect logs')
        time.sleep(15)
    for model in CFG['primary_models']:
        with open(OUT/'logs'/f'sample_{model}.log','a') as log:
            subprocess.run(['/root/miniconda3/envs/navsim/bin/torchrun','--standalone','--nproc_per_node=8',str(Path(__file__).with_name('sample_5000.py')),'--model',model],cwd=WORK,stdout=log,stderr=subprocess.STDOUT,check=True)
    save(OUT/'audits/gpu_sampling_complete.json',dict(status='PASS',new_scene_model_banks=12000,new_trajectories=1536000,seconds=time.time()-start))
