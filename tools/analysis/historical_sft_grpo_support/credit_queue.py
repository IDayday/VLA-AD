"""CPU credit diagnostics start once each complete frozen rollout bank is scored."""
from common_support import *
import concurrent.futures,subprocess

def work(model):
    expected={s['token'] for s in scenes()};folder=OUT/'cache/scores/rollouts'/model
    while {p.stem for p in folder.glob('*.npz') if p.stem in expected}!=expected:time.sleep(10)
    with (OUT/'logs'/f'credit_{model}.log').open('w') as f:
        r=subprocess.run(['/root/miniconda3/envs/navsim/bin/python',str(Path(__file__).with_name('advantage.py')),'--model',model],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    return dict(model=model,returncode=r.returncode)

if __name__=='__main__':
    names=['a5_sft','v6_sft','a5_grpo_300','a5_grpo_4842','v6_grpo_300','v6_grpo_3300']
    with concurrent.futures.ThreadPoolExecutor(3) as ex:
        results=list(ex.map(work,names))
    save(OUT/'audits/credit_queue.json',results)
