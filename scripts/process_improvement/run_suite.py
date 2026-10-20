"""Unified executable suite; scientific gates are local to each lineage.

Without --host commands run on the already leased GPU(s). With --host they are
dispatched through the live allowlist and exact GPU ownership registry. A dry
run is explicitly recorded as planning and never as experimental completion.
"""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',required=True,choices=['parity','scaling','local_diagnostics','calibration','main','rl_controls','final_eval'])
    p.add_argument('--dry-run',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--run-root');p.add_argument('--resource-policy',default='/mnt/project/server_dispatch_policy.json')
    p.add_argument('--host');p.add_argument('--gpus',nargs='+',type=int,default=[0]);p.add_argument('--job-name')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--rank',type=int,default=0)
    p.add_argument('--shards',type=int,default=1);p.add_argument('--method',default='OPD')
    p.add_argument('--population',default='train_full');p.add_argument('--stage',default='rates')
    p.add_argument('--name');p.add_argument('--checkpoint');p.add_argument('--init',default='IL')
    p.add_argument('--score-budget',type=int,default=4000000)
    p.add_argument('--steps',type=int)
    p.add_argument('--cold',action='store_true');p.add_argument('--extend-four-epochs',action='store_true')
    a=p.parse_args()
    if a.run_root:os.environ['PROCESS_RUN_ROOT']=a.run_root
    from scripts.process_improvement.common import ROOT,SOURCE,PYTHON,read,write,identity
    if Path(a.resource_policy).resolve()!=Path('/mnt/project/server_dispatch_policy.json'):
        raise ValueError('The live host policy cannot be replaced by a different file')
    commands=[];gate=None
    if a.phase=='parity':
        for receipt,script,extra in [('parity.json','audit.py',['--online']),
                                     ('disabled_method.json','qualify_disabled.py',[]),
                                     ('distributed_resume.json','qualify_distributed.py',[])]:
            prior=ROOT/'tests'/receipt
            if a.resume and prior.exists() and read(prior)['status']=='PASS':continue
            commands.append(['scripts/process_improvement/'+script,*extra])
    elif a.phase=='scaling':
        rates=read(ROOT/'selection/learning_rates.json')['selected'];method=a.method
        if method not in rates:raise ValueError('Uncalibrated method')
        name=a.name or (f'full_{method}_{a.seed}' if a.population=='train_full' else f'scaling_{a.population}_{method}_{a.seed}')
        command=['scripts/process_improvement/train.py','--method',method,'--name',name,'--split',a.population,
                 '--seed',str(a.seed),'--lr',str(rates[method]['lr'])]
        if a.population=='legacy_4k':command+=['--exposures','64000']
        elif a.population=='train_25k':command+=['--epochs','4']
        elif a.population=='train_full':
            command+=['--epochs','4']
            if a.extend_four_epochs:
                choice=read(ROOT/'selection/four_epoch_extension.json')
                assert choice['decisions'][method]['extend_to_four_epochs']
                if not a.resume:raise ValueError('Four epochs must extend the existing lineage with --resume')
            else:command+=['--stop-after','4681']
        else:raise ValueError('Unregistered scaling population')
        commands=[command]
    elif a.phase=='local_diagnostics':
        commands=[['scripts/process_improvement/diagnose.py','--name','local_'+part+'_grid','--split',split,
                   '--grid','--rank',str(a.rank),'--shards',str(a.shards),'--scoring-workers','4']
                  for part,split in [('train','diagnostic_train'),('cal','diagnostic_cal')]]
    elif a.phase=='calibration':
        commands=[['scripts/process_improvement/calibrate.py','--stage',a.stage]]
    elif a.phase=='main':
        gate=read(ROOT/'selection/process_gate.json')
        if gate['status']!='SUPPORTED':
            write(ROOT/'suite'/f'process_not_run_seed_{a.seed}.json',dict(status='NOT_RUN',reason=gate,independent_controls_continue=True))
            print('Process gate did not support expansion; OPD and GRPO remain independent',flush=True)
            return
        rates=read(ROOT/'selection/learning_rates.json')['selected'];method=a.method
        if a.cold:
            from scripts.process_improvement.common import config
            init=config()['student_init'];rate=rates['OPD']['lr']
        else:
            warm=read(ROOT/'selection/warm_start.json')['chosen']
            init=warm['checkpoints'][str(a.seed)]['path'];rate=rates[warm['method']]['lr']
        if method not in ['PROCESS','UNIFORM_TARGET','SHUFFLED_VALUE','NO_GUARD','OPD']:raise ValueError(method)
        if method=='OPD' and a.steps is None:raise ValueError('C1 requires fixed exposure/compute-matched --steps; OPD consumes no scoring budget')
        command=['scripts/process_improvement/train.py','--method',method,'--name',a.name or f'warm_{method}_{a.seed}',
                 '--split',a.population,'--steps',str(a.steps or 100000),'--init',init,'--seed',str(a.seed),
                 '--lr',str(rate),'--config','configs/process_improvement/selected_process.yaml']
        if method!='OPD':command+=['--score-budget',str(a.score_budget)]
        commands=[command]
    elif a.phase=='rl_controls':
        if a.init=='WARM':
            selected=read(ROOT/'selection/grpo_warm_rate.json')['selected']
            initial=read(ROOT/'selection/warm_start.json')['chosen']['checkpoints'][str(a.seed)]['path']
        else:
            selected=read(ROOT/'selection/grpo_rates.json')['selected'][a.init];initial=a.init
        command=['scripts/process_improvement/train_grpo.py','--name',a.name or f'GRPO_{a.init}_{a.seed}',
                 '--init',initial,'--seed',str(a.seed),'--lr',str(selected['lr']),'--score-budget',str(a.score_budget)]
        if len(a.gpus)>1:command=['-m','torch.distributed.run','--standalone',f'--nproc_per_node={len(a.gpus)}',*command]
        commands=[command]
    elif a.phase=='final_eval':
        lock=read(ROOT/'selection/final.lock.json')
        commands=[['scripts/process_improvement/evaluate.py','--name',item['name'],'--checkpoint',item['checkpoint'],
                   '--split',split,'--cuda-graph','--rank',str(a.rank),'--shards',str(a.shards)]
                  for split in ['dev','navtest'] for item in lock['candidates']]
    if a.resume:
        for command in commands:
            if 'scripts/process_improvement/train.py' in command or 'scripts/process_improvement/train_grpo.py' in command:
                command.append('--resume')
    record=dict(phase=a.phase,status='DRY_RUN' if a.dry_run else 'EXECUTING',commands=commands,gate=gate)
    print(record,flush=True)
    if a.dry_run:return
    if a.host:
        from scripts.process_improvement.dispatch import launch
        if not a.job_name:raise ValueError('--host requires a unique --job-name')
        print(launch(a.job_name,a.host,a.gpus,commands),flush=True)
    else:
        import subprocess
        identity()
        for command in commands:subprocess.run([PYTHON,*command],cwd=SOURCE,check=True)


if __name__=='__main__':main()
