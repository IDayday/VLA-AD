"""Queue read-only follow-up diagnostics after each of our primary GPU jobs."""
from common_support import *
import subprocess,threading,queue

def main():
    q=queue.Queue();jobs=[]
    for name in CFG['historical_followups']:
        for rank in range(2):jobs.append(dict(script='batched.py',model=name,rank=rank,world=2))
    for name in CFG['primary_models']:jobs.append(dict(script='fitting.py',model=name,rank=0,world=1))
    for j in jobs:q.put(j)
    status=[];lock=threading.Lock()
    def worker(gpu,primary,rank):
        marker=OUT/'audits'/f'run_{primary}_{rank}.json'
        while not marker.exists():time.sleep(10)
        while True:
            try:j=q.get_nowait()
            except queue.Empty:return
            label=f"{j['script'][:-3]}_{j['model']}_{j['rank']}";log=OUT/'logs'/f'{label}.log'
            cmd=['/root/miniconda3/envs/navsim/bin/python',str(Path(__file__).with_name(j['script'])),'--model',j['model'],'--rank',str(j['rank']),'--world',str(j['world'])]
            with log.open('w') as f:
                p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),start_new_session=True)
                print('START',gpu,label,p.pid,flush=True);rc=p.wait()
            with lock:
                status.append(dict(**j,gpu=gpu,returncode=rc,log=str(log)));save(OUT/'audits/queue_progress.json',status)
            print('DONE',gpu,label,rc,flush=True);q.task_done()
    threads=[]
    for gpu in range(8):
        primary=['official_il','psi_sft','a5_sft','v6_sft'][gpu//2]
        t=threading.Thread(target=worker,args=(gpu,primary,gpu%2));t.start();threads.append(t)
    for t in threads:t.join()
    if all(r['returncode']==0 for r in status) and len(status)==len(jobs):
        save(OUT/'audits/gpu_sampling_complete.json',dict(status='PASS',jobs=status,optimizer_updates=0))
    else:raise RuntimeError('Some queued read-only jobs failed; inspect queue_progress.json')

if __name__=='__main__':main()
