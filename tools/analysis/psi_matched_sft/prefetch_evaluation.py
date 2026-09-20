"""Use remote inference slots while local, unchanged SFT jobs continue.

Only predeclared step128 and initial-IL probes are evaluated. No results feed
back into training or checkpoint selection. The main driver safely resumes the
same content-addressed caches and independently repeats parity checks.
"""
from common_matched import *
import shlex
def main():
    py='/root/miniconda3/envs/navsim/bin/python';code=WORK/'tools/analysis/psi_matched_sft/evaluate.py'
    jobs=[dict(name='official_il',step=512,rank=r,world=8) for r in range(8)]
    jobs += [dict(name=f'{m}_seed{s}',step=128,rank=r,world=3) for m in CFG['methods'] for s in CFG['train_seeds'] for r in range(3)]
    slots=[(host,g) for host in ['training-rl-zt4','training-vla-zt2'] for g in range(8)];running={};finished=[]
    while jobs or running:
        for slot,(proc,job) in list(running.items()):
            status=proc.poll()
            if status is not None:
                finished.append(dict(job,returncode=status));del running[slot]
        for slot in slots:
            if slot in running:continue
            ready=next((j for j in jobs if j['name']=='official_il' or (OUT/'checkpoints'/j['name']/f"step{j['step']:04d}.json").exists()),None)
            if ready is None:continue
            jobs.remove(ready);host,gpu=slot;label=f"prefetch_{ready['name']}_step{ready['step']}_rank{ready['rank']}"
            cmd=[py,str(code),ready['name'],'--step',str(ready['step']),'--rank',str(ready['rank']),'--world',str(ready['world'])]
            remote='cd '+shlex.quote(str(WORK))+' && CUDA_VISIBLE_DEVICES='+str(gpu)+' '+shlex.join(cmd)
            full=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,remote]
            with (OUT/'logs'/f'{label}.log').open('a') as f:p=subprocess.Popen(full,cwd=WORK,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            running[slot]=(p,ready);save(OUT/'manifests/jobs'/f'{label}.json',dict(pid=p.pid,cmd=full,host=host,gpu=gpu,started=time.time()))
        save(OUT/'audits/prefetch_progress.json',dict(pending=len(jobs),running=len(running),finished=finished))
        time.sleep(5)
    save(OUT/'audits/prefetch_complete.json',dict(status='PASS' if all(j['returncode']==0 for j in finished) else 'RETRY_IN_MAIN_DRIVER',jobs=finished))
if __name__=='__main__':main()
