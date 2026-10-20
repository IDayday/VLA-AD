"""Generate auditable results from complete real experiment artifacts only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
from collections import defaultdict
import numpy as np
from scripts.process_improvement.common import *
from navsim.agents.recogdrive.process_improvement.statistics import parent_mean_bootstrap,paired_bootstrap,local_noise_bootstrap


def local(name,split,repetitions=10000,require_complete=True):
    directory=ROOT/'diagnostics'/name;rows=membership(split)
    paths=[directory/'scenes'/f"{r['token']}.json" for r in rows]
    missing=[str(p) for p in paths if not p.exists()]
    if missing and require_complete:raise ValueError(f'Incomplete diagnostics: {len(missing)}/{len(rows)} scenes missing')
    groups=defaultdict(list)
    for row,path in zip(rows,paths):
        if not path.exists():continue
        by_recipe=defaultdict(list)
        for record in read(path)['records']:
            by_recipe[(record['source'],record['temperature_base'],record['rho'])].append(record)
        for recipe,records in by_recipe.items():
            values=[np.mean([r[key] for r in records]) for key in ['selection_gain','projection_gain','guarded_gain','accepted','ess','radius','fit_value_range']]
            records=sorted(records,key=lambda r:(r['index'],r['bank']))
            noise=np.stack([np.stack([np.asarray(r[field])[:,-1]-np.asarray(r['P0'])[:,-1]
                                      for field in ['P1','proposed','P2']],axis=-1) for r in records])
            banks=len({r['bank'] for r in records})
            noise=noise.reshape(5,banks,noise.shape[1],3).mean(0)
            groups[recipe].append((row['token'],row['driver_record'],values,noise))
    result=[]
    for (source,temperature,rho),data in sorted(groups.items()):
        estimate=parent_mean_bootstrap([r[2] for r in data],[r[1] for r in data],repetitions)
        hierarchy=local_noise_bootstrap([r[3] for r in data],[r[1] for r in data],repetitions)
        estimate['CI95'][0][:3]=hierarchy['CI95'][0]
        estimate['CI95'][1][:3]=hierarchy['CI95'][1]
        result.append(dict(source=source,temperature=temperature,rho=rho,
                           fields=['selection_gain','projection_gain','guarded_gain','accepted_fraction','ess','radius','fit_value_range'],
                           noise_hierarchy=hierarchy,**estimate))
    record=dict(name=name,split=split,complete=not missing,missing_scenes=len(missing),results=result,
                caveat='Banks/steps averaged within each scene; parent record is the independent bootstrap cluster; no fit-winner gain substituted for held-out gain')
    write(ROOT/'analysis'/f'{name}.json',record)
    return record


def predictions(name,split,limit=None):
    rows=membership(split)
    if limit:rows=rows[:limit]
    root=ROOT/'evaluation'/split/name
    arrays=[]
    identities=set()
    for row in rows:
        path=root/'scenes'/f"{row['token']}.npz";meta=path.with_suffix('.json')
        if not path.exists() or not meta.exists():raise ValueError(f'Incomplete evaluation {name}: {row["token"]}')
        metadata=read(meta)
        if metadata['prediction_sha256']!=sha(path):raise ValueError(f'Prediction payload hash changed: {path}')
        identities.add(metadata['evaluation_identity'])
        with np.load(path) as data:arrays.append(data['scores'])
    if len(identities)!=1:raise ValueError('Mixed evaluation identities')
    return np.stack(arrays),rows


def compare(names,baseline,split,repetitions=10000,limit=None,screening=None):
    old,rows=predictions(baseline,split,limit);new=[]
    for name in names:
        value,other=predictions(name,split,limit)
        assert [r['token'] for r in rows]==[r['token'] for r in other]
        new.append(value)
    high=None
    if screening:
        screen,_=predictions(screening,split,limit)
        high=(screen[:,:,-1]>=.90).all(1)
    result=paired_bootstrap(old,np.stack(new),[r['driver_record'] for r in rows],high_mask=high,repetitions=repetitions)
    result.update(methods=names,baseline=baseline,split=split,scenes=len(rows),high_screening=screening)
    write(ROOT/'analysis'/('compare_'+split+'_'+names[0]+'.json'),result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--local');p.add_argument('--split',default='dev')
    p.add_argument('--methods',nargs='+');p.add_argument('--baseline',default='B0_IL')
    p.add_argument('--screening');p.add_argument('--limit',type=int);p.add_argument('--repetitions',type=int,default=10000)
    p.add_argument('--suite');p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    if a.local:
        result=local(a.local,a.split,a.repetitions,not a.allow_partial)
    elif a.methods:
        result=compare(a.methods,a.baseline,a.split,a.repetitions,a.limit,a.screening)
    else:
        runs={p.parent.name:read(p) for p in (ROOT/'runs').glob('*/complete.json')}
        result=dict(study=STUDY,actual_completed_runs=runs,protocol_lock=read(ROOT/'protocol.lock.json'))
        write(ROOT/'analysis/suite_inventory.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
