"""Bounded dispatcher for the registered matrix; no outcome-directed retries."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
from scripts.opd.common import *
from scripts.opd.resources import launch

HOSTS=['training-rl-zt2','training-vla-zt','training-vlawm-tsinghua-zt2','training-vlawm-tsinghua-zt3']
EXPS=['E1','E2','E3','E4-A','E4-B','E4-C','E5']


def slots():
    reserved=set()
    for p in (ROOT/'jobs').glob('*/spec.json'):
        if (p.parent/'complete.json').exists() or not (p.parent/'launch.json').exists():continue
        spec=read(p);g=spec['gpu'];g=g if isinstance(g,list) else ([] if g is None else [g])
        reserved.update((spec['host'],i) for i in g)
    return [(h,g) for h in HOSTS for g in range(8) if (h,g) not in reserved]


def run_queue(tasks):
    remaining=list(tasks);running=[];admissions={}
    while remaining or running:
        for name in list(running):
            p=ROOT/'jobs'/name/'complete.json'
            if p.exists():
                if read(p)['status']!='COMPLETE':raise RuntimeError(f'Failed experiment {name}; fix and explicitly rerun affected run')
                running.remove(name)
        free=slots()
        while remaining and free:
            name,commands=remaining[0]
            if (ROOT/'jobs'/name/'spec.json').exists():
                if not (ROOT/'jobs'/name/'launch.json').exists():raise RuntimeError(f'Previous admission failed for {name}')
                remaining.pop(0);running.append(name);continue
            host,gpu=free.pop(0)
            print('DISPATCH',name,host,gpu,flush=True);launch(name,host,gpu,commands);remaining.pop(0);running.append(name)
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
    tasks=[]
    for rank in range(8):tasks.append((f'navtest_E0_{rank}',[['scripts/opd/evaluate.py','--exp','E0','--split','navtest','--rank',str(rank),'--shards','8']]))
    for e in choice['methods']:
        for s in choice['training_seeds']:
            for rank in range(2):tasks.append((f'navtest_{e}_{s}_{rank}',[['scripts/opd/evaluate.py','--exp',e,'--seed',str(s),'--split','navtest','--rank',str(rank),'--shards','2']]))
    run_queue(tasks);write(ROOT/'suite/navtest_complete.json',dict(status='COMPLETE',jobs=len(tasks),time=time.time()))


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
    run_queue(tasks);write(ROOT/'suite/E6_complete.json',dict(status='COMPLETE',jobs=len(tasks),time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['formal','test','e6']);a=p.parse_args()
    if a.action=='formal':formal()
    elif a.action=='test':final_test()
    else:e6()
