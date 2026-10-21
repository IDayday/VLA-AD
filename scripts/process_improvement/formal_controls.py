"""Independent native GRPO controls with fixed scoring endpoints and harm checks.

This scheduler never reads dev/navtest to pick a model or stop a lineage.
Training resumes on its original host, with its original world size and full
native optimizer/RNG state. No old E5 or process-projection gate is consulted.
"""
from scripts.process_improvement.common import *


def freeze_endpoint_cost(run, step):
    """Archive the actual stable rank ledgers before a resume overwrites them."""
    from scripts.process_improvement.calibrate import immutable
    from scripts.process_improvement.summarize_costs import sum_ledgers
    dest=run/'endpoint_costs'/str(step);receipt=dest/'receipt.json'
    checkpoint=run/f'head_{step}.pt'
    if receipt.exists():
        value=read(receipt)
        if value['checkpoint_sha256']!=sha(checkpoint):raise ValueError('Endpoint checkpoint changed')
        for path,expected in value['ledger']['source_files'].items():
            if sha(path)!=expected:raise ValueError('Archived endpoint cost ledger changed')
        return value
    cfg=read(run/'config.json')
    sources=[run/f'cost_rank_{rank}.json' for rank in range(cfg['world'])]
    if not all(path.exists() for path in sources):raise ValueError('Missing actual rank cost ledger')
    if any(read(path)['step']!=step for path in sources):
        raise ValueError('Never infer an endpoint cost from another training step')
    archived=[]
    for rank,path in enumerate(sources):
        target=dest/f'rank_{rank}.json';immutable(target,read(path));archived.append(target)
    ledger=sum_ledgers(archived)
    if ledger['budget']['physical_scores']!=step*cfg['global_batch']*cfg['group_size']:
        raise ValueError('Actual endpoint scoring count differs from native group accounting')
    value=dict(status='MEASURED_IMMUTABLE_ENDPOINT',step=step,
               checkpoint_sha256=sha(checkpoint),run_config_sha256=sha(run/'config.json'),
               actual_source_files={str(path):sha(path) for path in sources},ledger=ledger,
               generator_sha256=sha(__file__),aggregator_sha256=sha(SOURCE/'scripts/process_improvement/summarize_costs.py'),
               interpretation='Actual paused/completed rank ledgers; cumulative training cost, distinct from whole-job leases; no historical imputation')
    immutable(receipt,value)
    return value


def tasks():
    from scripts.process_improvement.coordinator import evaluation_task,SINGLE_HOSTS,RL_HOSTS,eval_done
    from scripts.process_improvement.calibrate import immutable
    ready=[];rates=ROOT/'selection/grpo_rates.json';warm=ROOT/'selection/warm_start.json'
    if not rates.exists():return ready
    gate=ROOT/'tests/native_GRPO_distributed_resume.json'
    if not gate.exists() or read(gate)['status']!='PASS':return ready
    recipes={initial:dict(init=initial,lr=read(rates)['selected'][initial]['lr']) for initial in ['IL','RL']}
    if warm.exists():
        chosen=read(warm)['chosen'];warm_init=chosen['checkpoints']['42']['path']
        for lr in ['1e-5','3e-5']:
            name='tune_GRPO_WARM_'+lr;run=ROOT/'runs'/name
            if (run/'complete.json').exists():
                task=evaluation_task(name,run/'head_812.pt')
                if task:ready.append(task)
            else:
                ready.append(dict(name='auto_'+name,priority=4,gpus=8,hosts=RL_HOSTS,
                    commands=[['-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                               'scripts/process_improvement/train_grpo.py','--name',name,'--init',warm_init,
                               '--lr',lr,'--split','train_25k','--steps','812','--score-budget','207872']]))
        rate=ROOT/'selection/grpo_warm_rate.json'
        if not rate.exists() and all(eval_done('tune_GRPO_WARM_'+lr) for lr in ['1e-5','3e-5']):
            from scripts.process_improvement.analyze import predictions
            trials=[]
            for lr in ['1e-5','3e-5']:
                values,_=predictions('tune_GRPO_WARM_'+lr,'cal_tune')
                run=ROOT/'runs'/('tune_GRPO_WARM_'+lr);done=read(run/'complete.json')
                assert done['steps']==812 and done['physical_scores']==207872
                trials.append(dict(lr=float(lr),PDMS=float(values[:,:,-1].mean()),checkpoint_sha256=sha(run/'head_812.pt')))
            immutable(rate,dict(status='LOCKED',selected=sorted(trials,key=lambda x:(-x['PDMS'],x['lr']))[0],trials=trials,
                                source='Same selected warm recipe, training seed42, complete train25k epoch per LR, cal-tune only'))
        if rate.exists():recipes['WARM']=dict(init_by_seed=chosen['checkpoints'],lr=read(rate)['selected']['lr'])
    for family,recipe in recipes.items():
        for seed in [42,43,44]:
            name=f'GRPO_{family}_{seed}';run=ROOT/'runs'/name
            for marker,field in [('paused.json','step'),('complete.json','steps')]:
                if (run/marker).exists():
                    step=read(run/marker)[field]
                    if step in [977,3907,15625]:
                        freeze_endpoint_cost(run,step)
                        if not (run/'endpoint_states'/str(step)/'receipt.json').exists():
                            from scripts.process_improvement.endpoint_state import freeze_endpoint_state
                            freeze_endpoint_state(run,step)
            for step in [977,3907,15625]:
                checkpoint=run/f'head_{step}.pt'
                if checkpoint.exists():
                    # Schedule the fixed endpoint report before reserving all
                    # remaining cards for another gang. These results never
                    # select a checkpoint or stop training (cal-confirm does).
                    task=evaluation_task(f'{name}_step{step}',checkpoint,split='dev',priority=1)
                    if task:ready.append(task)
            if (run/'complete.json').exists():continue
            init=recipe['init_by_seed'][str(seed)]['path'] if family=='WARM' else recipe['init']
            baseline='WARM_'+str(seed) if family=='WARM' else 'B0_'+family
            if family=='WARM':
                task=evaluation_task(baseline,init,split='cal_confirm',priority=0)
                if task:ready.append(task)
            target=977;resume=False;hosts=RL_HOSTS
            world=read(run/'config.json')['world'] if (run/'config.json').exists() else 8
            if world not in [4,8]:raise ValueError('Unqualified formal GRPO world size')
            if (run/'paused.json').exists():
                paused=read(run/'paused.json');step=paused['step'];label=f'{name}_step{step}'
                checkpoint=run/f'head_{step}.pt'
                task=evaluation_task(label,checkpoint,split='cal_confirm')
                if task:ready.append(task)
                audit=run/f'harm_audit_{step}.json'
                if not audit.exists() and eval_done(label,'cal_confirm') and eval_done(baseline,'cal_confirm'):
                    log=run/f'harm_audit_{step}.log'
                    with log.open('w') as stream:
                        subprocess.run([PYTHON,'scripts/process_improvement/harm_audit.py','--run',name,
                                       '--evaluation',label,'--baseline',baseline],cwd=SOURCE,stdout=stream,stderr=subprocess.STDOUT,check=True)
                if not audit.exists():continue
                decision=read(audit)
                if decision['status']=='HARM_BOUNDARY_REACHED':
                    with (run/f'harm_finalize_{step}.log').open('w') as stream:
                        subprocess.run([PYTHON,'scripts/process_improvement/harm_audit.py','--run',name,
                                       '--evaluation',label,'--baseline',baseline,'--finalize-stop'],cwd=SOURCE,stdout=stream,stderr=subprocess.STDOUT,check=True)
                    continue
                target=3907 if step==977 else 15625
                assert step in [977,3907],'Unexpected paused formal GRPO endpoint'
                resume=True
                # Actual running owner is authoritative if a pre-launch
                # admission was explicitly retried under a new job identity.
                hosts=[read(run/'config.json')['owner']['host']]
            command=['-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',
                     'scripts/process_improvement/train_grpo.py','--name',name,'--init',init,
                     '--seed',str(seed),'--lr',str(recipe['lr']),'--score-budget','4000000']
            if target<15625:command+=['--stop-after',str(target)]
            if resume:command+=['--resume']
            # Complete the same low-cost endpoint across independent seeds
            # before occupying an entire host with a larger-budget extension.
            priority=7 if not resume else 8 if target==3907 else 9
            ready.append(dict(name=f'formal_{name}_to{target}',priority=priority,gpus=world,hosts=hosts,commands=[command]))
    return ready
