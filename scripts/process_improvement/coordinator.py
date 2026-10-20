"""Resumable first-stage real experiment orchestration over owned GPU leases.

Only predefined protocols/selection rules are used. Empty devices are checked
again atomically by dispatch.launch. Completed and failed artifacts are never
overwritten; failures require explicit review, never automatic success/retry.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import fcntl
import time
from scripts.process_improvement.common import *
from scripts.process_improvement.dispatch import remote,launch

SINGLE_HOSTS=['training-vla-zt','training-vlawm-tsinghua-zt2']
RL_HOSTS=['training-rl-zt2','training-vlawm-tsinghua-zt5','training-vlawm-tsinghua-zt6']


def evaluation_task(name,checkpoint=None,baseline=None,split='cal_tune',priority=0):
    dest=ROOT/'evaluation'/split/name
    if all((dest/f'complete_{r}.json').exists() for r in range(3)):return None
    commands=[]
    for rank in range(3):
        cmd=['scripts/process_improvement/evaluate.py','--name',name,'--split',split,
             '--cuda-graph','--rank',str(rank),'--shards','3']
        cmd+=['--checkpoint',str(checkpoint)] if checkpoint else ['--baseline',baseline]
        commands.append(cmd)
    path=ROOT/'suite'/f'auto_eval_{split}_{name}.json'
    if path.exists():assert read(path)==commands
    else:write(path,commands)
    return dict(name=f'auto_eval_{split}_{name}',priority=priority,gpus=1,hosts=SINGLE_HOSTS,
                commands=[['scripts/process_improvement/parallel.py','--spec',str(path),
                           '--receipt',str(path.with_suffix('.receipt.json'))]])


def eval_done(name,split='cal_tune'):
    dest=ROOT/'evaluation'/split/name
    return all((dest/f'complete_{rank}.json').exists() for rank in range(3))


def ready_tasks(include_formal=False):
    from scripts.process_improvement import calibrate
    extra=ROOT/'suite/extra_queue.json'
    tasks=read(extra) if extra.exists() else []
    # Fixed-hyperparameter safety monitoring baselines, no test data access.
    for base in ['IL','RL']:
        task=evaluation_task('B0_'+base,baseline=base,split='cal_confirm',priority=25)
        if task:tasks.append(task)
    names=['tune_'+m+'_'+lr for m in ['OPD','ANCHOR'] for lr in ['1e-5','3e-5']]
    rates=ROOT/'selection/learning_rates.json'
    if not rates.exists() and all(eval_done(name) for name in names):calibrate.rates()
    if rates.exists():
        selected=read(rates)['selected']
        for method in ['OPD','ANCHOR']:
            for seed in [42,43,44]:
                name=f'full_{method}_{seed}';run=ROOT/'runs'/name
                if not (run/'paused.json').exists() and not (run/'complete.json').exists():
                    tasks.append(dict(name='auto_'+name,priority=10,gpus=1,hosts=SINGLE_HOSTS,
                        commands=[['scripts/process_improvement/run_suite.py','--phase','scaling','--method',method,'--seed',str(seed)]]))
                for epoch,step in [(1,2341),(2,4681)]:
                    checkpoint=run/f'head_{step}.pt'
                    if checkpoint.exists():
                        task=evaluation_task(f'{name}_epoch{epoch}',checkpoint,priority=1)
                        if task:tasks.append(task)
        for population in ['legacy_4k','train_25k']:
            name=f'scaling_{population}_OPD_42';run=ROOT/'runs'/name
            if not (run/'complete.json').exists():
                tasks.append(dict(name='auto_'+name,priority=11,gpus=1,hosts=SINGLE_HOSTS,
                    commands=[['scripts/process_improvement/run_suite.py','--phase','scaling','--population',population]]))
        warm=ROOT/'selection/warm_start.json'
        if not warm.exists() and all(eval_done(f'full_{m}_{s}_epoch{e}') for m in ['OPD','ANCHOR'] for s in [42,43,44] for e in [1,2]):
            calibrate.warm()
    # The original native algorithm is independent of every process gate.
    for initial in ['IL','RL']:
        for lr in ['1e-5','3e-5']:
            name=f'tune_GRPO_{initial}_{lr}';run=ROOT/'runs'/name
            if (run/'complete.json').exists():
                task=evaluation_task(name,run/'head_812.pt')
                if task:tasks.append(task)
            elif name!='tune_GRPO_IL_1e-5':
                tasks.append(dict(name='auto_'+name,priority=5,gpus=8,hosts=RL_HOSTS,
                    commands=[['scripts/process_improvement/verify_inputs.py'],
                              ['-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                               'scripts/process_improvement/train_grpo.py','--name',name,'--init',initial,
                               '--lr',lr,'--split','train_25k','--steps','812','--score-budget','207872']]))
    grpo_names=[f'tune_GRPO_{initial}_{lr}' for initial in ['IL','RL'] for lr in ['1e-5','3e-5']]
    if not (ROOT/'selection/grpo_rates.json').exists() and all(eval_done(n) for n in grpo_names):calibrate.grpo_rates()
    # Calibration selection and confirmation occupy disjoint physical records.
    local=ROOT/'diagnostics/local_cal_grid'
    choice=ROOT/'selection/process_hyperparameters.json'
    if not choice.exists() and all((local/f'complete_{r}.json').exists() for r in range(24)):calibrate.mechanism()
    if choice.exists():
        phase='local_confirm';expansion=False
        if all((ROOT/'diagnostics'/phase/f'complete_{r}.json').exists() for r in range(8)):
            if not (ROOT/'selection/process_gate_local_confirm.json').exists():calibrate.process_gate()
            decision=read(ROOT/'selection/process_gate_local_confirm.json')
            if decision['expansion_allowed']:
                phase='local_confirm_expand32';expansion=True
            else:phase=None
        if phase:
            for rank in range(8):
                if (ROOT/'diagnostics'/phase/f'complete_{rank}.json').exists():continue
                command=['scripts/process_improvement/diagnose.py','--name',phase,'--split','diagnostic_confirm',
                         '--config','configs/process_improvement/selected_process.yaml',
                         '--rank',str(rank),'--shards','8','--scoring-workers','4']
                if expansion:command+=['--test','32']
                tasks.append(dict(name=f'auto_{phase}_{rank}',priority=20,gpus=1,hosts=SINGLE_HOSTS,commands=[command]))
            if expansion and all((ROOT/'diagnostics'/phase/f'complete_{r}.json').exists() for r in range(8)):
                if not (ROOT/'selection/process_gate_local_confirm_expand32.json').exists():calibrate.process_gate()
    if include_formal:
        from scripts.process_improvement.formal_controls import tasks as formal_tasks
        tasks.extend(formal_tasks())
    return sorted([t for t in tasks if t],key=lambda t:(t['priority'],t['name']))


def available(host):
    occupied=set()
    for spec in (ROOT/'jobs').glob('*/spec.json'):
        if (spec.parent/'launch.json').exists() and not (spec.parent/'complete.json').exists():
            value=read(spec)
            if value['host']==host:occupied.update(value['gpus'])
    code="""import json,subprocess
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
print(json.dumps([int(r.split(',')[0]) for r in rows if int(r.split(',')[1])<=100 and int(r.split(',')[2])<=5]))
"""
    return [gpu for gpu in remote(host,code) if gpu not in occupied]


def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');p.add_argument('--formal-controls',action='store_true');a=p.parse_args()
    folder=ROOT/'controller';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'lease.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(folder/'owner.json',identity())
        while True:
            identity();assert sha(ROOT/'protocol.yaml')==read(ROOT/'protocol.lock.json')['sha256']
            write(folder/'phase.json',dict(phase='scheduling',owner=identity()))
            tasks=ready_tasks(a.formal_controls);free={};started=[];errors=[]
            for task in tasks:
                job=ROOT/'jobs'/task['name']
                if job.exists():continue # No implicit retry/overwrite of failed jobs.
                for host in task['hosts']:
                    if host not in free:
                        try:free[host]=available(host)
                        except Exception as exc:errors.append(dict(host=host,error=repr(exc)));free[host]=[]
                    if len(free[host])<task['gpus']:continue
                    gpus=free[host][:task['gpus']]
                    try:
                        launch(task['name'],host,gpus,task['commands']);free[host]=free[host][task['gpus']:]
                        started.append(dict(name=task['name'],host=host,gpus=gpus))
                    except Exception as exc:errors.append(dict(name=task['name'],error=repr(exc)))
                    break
            state=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),ready_tasks=[t['name'] for t in tasks],
                       launched=started,errors=errors,owner=identity())
            write(folder/'state.json',state)
            write(folder/'phase.json',dict(phase='idle',owner=identity()))
            print('COORDINATOR',state,flush=True)
            if a.once:return
            time.sleep(30)


if __name__=='__main__':main()
