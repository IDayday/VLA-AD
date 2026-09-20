"""Wait for all inputs, then compute fixed analyses, tests and complete report."""
from common_5000 import *
import subprocess,time,datetime

if __name__=='__main__':
    start=time.time()
    while not (OUT/'audits/gpu_sampling_complete.json').exists() or len(list((OUT/'cache/scores/rollouts').glob('*/*.npz')))<12000:
        if time.time()-start>21600:raise TimeoutError('Sampling/scoring incomplete; no partial result is called complete')
        time.sleep(20)
    scripts=Path(__file__).parent
    save(OUT/'manifests/analysis_code.json',dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_hash=identity(),files={str(p):sha(p) for p in scripts.glob('*.py')}))
    for script,extra in [('analyze_5000.py',[]),('figures_5000.py',[]),('test_5000.py',[]),('audit_5000.py',['--final']),('report_5000.py',[])]:
        with open(OUT/'logs'/f'{Path(script).stem}.log','a') as log:
            subprocess.run(['/root/miniconda3/envs/navsim/bin/python',str(scripts/script),*extra],cwd=WORK,stdout=log,stderr=subprocess.STDOUT,check=True)
    save(OUT/'audits/finalize_complete.json',dict(status='PASS',seconds=time.time()-start))
