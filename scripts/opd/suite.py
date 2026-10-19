"""Bounded dispatcher for the registered matrix; no outcome-directed retries."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
from scripts.opd.common import *
from scripts.opd.resources import launch,remote

HOSTS=['training-rl-zt2','training-vla-zt','training-vlawm-tsinghua-zt2','training-vlawm-tsinghua-zt3']
EXPS=['E1','E2','E3','E4-A','E4-B','E4-C','E5']


def slots():
    reserved=set()
    for p in (ROOT/'jobs').glob('*/spec.json'):
        if (p.parent/'complete.json').exists() or not (p.parent/'launch.json').exists():continue
        spec=read(p);g=spec['gpu'];g=g if isinstance(g,list) else ([] if g is None else [g])
        reserved.update((spec['host'],i) for i in g)
    from concurrent.futures import ThreadPoolExecutor
    policy=read('/mnt/project/server_dispatch_policy.json')
    hosts=[h for h in HOSTS if h in policy['allowed_hosts'] and h not in policy.get('global_dispatch_prohibitions',{})]
    def inspect(host):
        try:
            data=remote(host,"""import subprocess,json
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
print(json.dumps([list(map(int,row.split(','))) for row in rows]))
""")
            return host,data,None
        except Exception as error:return host,[],str(error)
    with ThreadPoolExecutor(len(hosts) or 1) as pool:inventory=list(pool.map(inspect,hosts))
    write(ROOT/'resources/live_slots.json',dict(time=time.time(),inventory={h:dict(gpus=data,error=error) for h,data,error in inventory},preserve_unrelated_workloads=True))
    return [(h,g) for h,data,error in inventory for g,memory,util in data if memory<=100 and util<=5 and (h,g) not in reserved]


def run_queue(tasks,reserve_for_e6=False):
    remaining=list(tasks);running=[];admissions={}
    while remaining or running:
        for name in list(running):
            p=ROOT/'jobs'/name/'complete.json'
            if p.exists():
                if read(p)['status']!='COMPLETE':raise RuntimeError(f'Failed experiment {name}; fix and explicitly rerun affected run')
                running.remove(name)
        # Existing tasks can be resumed/observed without a free GPU. In
        # particular, completed tasks must not wait behind unrelated GPU jobs.
        for task in list(remaining):
            name,_=task;path=ROOT/'jobs'/name
            if (path/'launch.json').exists():
                remaining.remove(task)
                if (path/'complete.json').exists():
                    if read(path/'complete.json')['status']!='COMPLETE':raise RuntimeError('Failed experiment '+name)
                else:running.append(name)
        free=slots() if remaining else []
        if reserve_for_e6:
            decision=ROOT/'selection/E6_admission.json'
            e6_dispatched=len(list((ROOT/'jobs').glob('formal_E6_*/launch.json')))
            hold=not decision.exists() or (read(decision)['admitted'] and e6_dispatched<9)
            if hold:free=free[:max(0,len(HOSTS)*8-9-len(running))]
        while remaining and free:
            name,commands=remaining[0]
            host,gpu=free.pop(0)
            print('DISPATCH',name,host,gpu,flush=True)
            try:launch(name,host,gpu,commands)
            except RuntimeError as error:
                if 'GPU reserved by' in str(error) or 'GPU occupied' in str(error):continue
                raise
            remaining.pop(0);running.append(name)
        write(ROOT/'suite/progress.json',dict(remaining=[r[0] for r in remaining],running=running,time=time.time()))
        if remaining or running:time.sleep(15)


def formal():
    expected=[f'tune_{e}_{j}' for e in EXPS for j in range(2)]
    while not all((ROOT/'jobs'/n/'complete.json').exists() for n in expected):
        for n in expected:
            p=ROOT/'jobs'/n/'complete.json'
            if p.exists() and read(p)['status']!='COMPLETE':raise RuntimeError(f'Tuning failed: {n}')
        time.sleep(15)
    if not (ROOT/'selection/tuning.json').exists():
        from scripts.opd.analyze import select_tuning
        select_tuning()
    chosen=read(ROOT/'selection/tuning.json');w=chosen['gradient_weights'];tasks=[]
    for e in EXPS:
        for seed in config()['train_seeds']:
            name=f'train_{e}_{seed}'
            train=['scripts/opd/train.py','--exp',e,'--stage','train','--seed',str(seed),'--lr',str(chosen['selected'][e]['lr']),
                '--anchor-weight',str(w['anchor']),'--il-weight',str(w['il'])]
            evaluate=[['scripts/opd/evaluate.py','--exp',e,'--seed',str(seed),'--split',split] for split in ('cal','dev')]
            tasks.append((f'formal_{e}_{seed}',[train,*evaluate]))
    run_queue(tasks)
    write(ROOT/'suite/formal_complete.json',dict(status='COMPLETE',runs=len(tasks),time=time.time()))


def test_selection():
    import numpy as np
    from scripts.opd.analyze import bank
    choices={e:float(np.mean([bank('cal',f'train_{e}_{s}')[...,-1].mean() for s in config()['train_seeds']])) for e in ['E4-A','E4-B','E4-C']}
    best=max(choices,key=lambda e:(choices[e],-['E4-A','E4-B','E4-C'].index(e)))
    value=dict(status='LOCKED',E4_variant=best,calibration_PDMS=choices,methods=['E2','E3','E5',best],training_seeds=config()['train_seeds'],
        checkpoint='fixed1000; no dev or navtest selection',population_sha256=sha(ROOT/'manifests/navtest.json'))
    p=ROOT/'selection/navtest.json'
    if p.exists():
        if read(p)!=value:raise ValueError('Final test selection changed')
    else:write(p,value)
    return value


def final_test():
    choice=test_selection()
    # Require every official test cache before dispatch, never mark partial evaluation complete.
    for r in rows('navtest'):
        if not Path(r['observation_path']).exists() or not Path(r['metric_cache_path']).exists():raise FileNotFoundError('Test input preparation incomplete')
    routing=read(ROOT/'resources/evaluation_concurrency.json')
    if not routing['parity_passed'] or routing['processes_per_gpu']!=4:raise ValueError('Four native process routing not qualified')
    tasks=[]
    def bundle(name,commands):
        path=ROOT/'resources/navtest_bundles'/f'{name}.json';write(path,commands)
        tasks.append((name,[['scripts/opd/parallel.py','--spec',str(path),'--concurrency','4','--receipt',str(path.with_suffix('.receipt.json'))]]))
    for group in range(8):
        bundle(f'navtest_E0_{group}',[['scripts/opd/evaluate.py','--exp','E0','--split','navtest','--rank',str(group*4+k),'--shards','32','--workers','1'] for k in range(4)])
    for e in choice['methods']:
        for s in choice['training_seeds']:
            for group in range(2):
                bundle(f'navtest_{e}_{s}_{group}',[['scripts/opd/evaluate.py','--exp',e,'--seed',str(s),'--split','navtest','--rank',str(group*4+k),'--shards','8','--workers','1'] for k in range(4)])
    run_queue(tasks,reserve_for_e6=True);write(ROOT/'suite/navtest_complete.json',dict(status='COMPLETE',jobs=len(tasks),time=time.time()))


def confirmation():
    def ready(e,s):
        markers=[read(p) for p in (ROOT/'evaluation/cal'/f'train_{e}_{s}').glob('complete_*.json')]
        return bool(markers) and len(markers)==markers[0]['shards'] and sum(v['rows'] for v in markers)==len(rows('cal'))
    while not all(ready(e,s) for e in EXPS for s in config()['train_seeds']):time.sleep(15)
    final_test()


def e6():
    while not (ROOT/'suite/formal_complete.json').exists():time.sleep(15)
    from scripts.opd.analyze import formal as analyze
    if not (ROOT/'analysis/formal_dev.json').exists():analyze('dev')
    from scripts.opd.report import e6_admission
    admission=e6_admission()
    if not admission['admitted']:
        write(ROOT/'suite/E6_NOT_RUN.json',dict(status='NOT_RUN',reason=admission['reason']));return
    tasks=[]
    for variant in ['IL_GRPO','E5_GRPO','IL_JOINT']:
        for seed in config()['train_seeds']:
            name=f'E6_{variant}_{seed}_train'
            commands=[['scripts/opd/train.py','--exp','E6','--stage','train','--variant',variant,'--seed',str(seed)]]
            commands.extend([['scripts/opd/evaluate.py','--exp','E6','--name',name,'--split',s] for s in ['cal','dev']])
            tasks.append((f'formal_{name}',commands))
    run_queue(tasks);write(ROOT/'suite/E6_training_complete.json',dict(status='COMPLETE',jobs=len(tasks),time=time.time()))
    from scripts.opd.analyze_grpo import analyze as analyze_e6
    analyze_e6('dev');analyze_e6('cal')
    variants=['IL_GRPO','E5_GRPO','IL_JOINT'];cal=read(ROOT/'analysis/E6_cal.json')['methods']
    best=max(variants,key=lambda v:(cal[v]['metrics']['PDMS']['estimate'],-variants.index(v)))
    selection=dict(status='LOCKED',variant=best,source='Full D_cal mean PDMS across3 seeds, fixed registered endpoint; first listed variant on ties',
        calibration_PDMS={v:cal[v]['metrics']['PDMS']['estimate'] for v in variants})
    target=ROOT/'selection/E6_navtest.json'
    if target.exists():
        if read(target)!=selection:raise ValueError('E6 final test selection changed')
    else:write(target,selection)
    tasks=[]
    for seed in config()['train_seeds']:
        for group in range(2):
            name=f'navtest_E6_{best}_{seed}_{group}';path=ROOT/'resources/navtest_bundles'/f'{name}.json'
            write(path,[['scripts/opd/evaluate.py','--exp','E6','--name',f'E6_{best}_{seed}_train','--split','navtest',
                '--rank',str(group*4+k),'--shards','8','--workers','1'] for k in range(4)])
            tasks.append((name,[['scripts/opd/parallel.py','--spec',str(path),'--concurrency','4','--receipt',str(path.with_suffix('.receipt.json'))]]))
    run_queue(tasks)
    while not (ROOT/'suite/navtest_complete.json').exists():time.sleep(15)
    analyze_e6('navtest');write(ROOT/'suite/E6_complete.json',dict(status='COMPLETE',training_runs=9,test_policies=3,time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['formal','test','e6','confirm']);a=p.parse_args()
    if a.action=='formal':formal()
    elif a.action=='test':final_test()
    elif a.action=='confirm':confirmation()
    else:e6()
