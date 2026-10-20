"""Generate auditable results from complete real experiment artifacts only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import re
from collections import defaultdict
import numpy as np
from scripts.process_improvement.common import *
from navsim.agents.recogdrive.process_improvement.statistics import parent_mean_bootstrap,paired_bootstrap,local_noise_bootstrap


def analysis_source(split):
    return dict(script_sha256=sha(__file__),
        statistics_sha256=sha(SOURCE/'navsim/agents/recogdrive/process_improvement/statistics.py'),
        protocol_sha256=sha(ROOT/'protocol.yaml'),population_sha256=sha(ROOT/'manifests'/f'{split}.json'),
        bootstrap_seed=20261020)


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


def compare(names,baseline,split,repetitions=10000,limit=None,screening=None,cluster='parent',result_name=None):
    provenance=analysis_source(split)
    old,rows=predictions(baseline,split,limit);new=[]
    for name in names:
        value,other=predictions(name,split,limit)
        assert [r['token'] for r in rows]==[r['token'] for r in other]
        new.append(value)
    high=None
    if screening:
        screen,_=predictions(screening,split,limit)
        high=(screen[:,:,-1]>=.90).all(1)
    field={'parent':'driver_record','log':'log_name'}[cluster]
    result=paired_bootstrap(old,np.stack(new),[r[field] for r in rows],high_mask=high,repetitions=repetitions)
    result.update(methods=names,baseline=baseline,split=split,scenes=len(rows),high_screening=screening,cluster_unit=cluster,analysis_source=provenance)
    suffix='' if cluster=='parent' else '_cluster_log'
    suffix+=('' if baseline=='B0_IL' else '_vs_'+baseline)
    suffix+=('' if limit is None else '_limit'+str(limit))
    suffix+=('' if repetitions==10000 else '_boot'+str(repetitions))
    if cluster=='log':
        result['log_count']=result.pop('parent_count')
        result['scope']='Complete-log sensitivity analysis; globally paired inference IDs and training-policy draws retained; parent-record analysis is primary'
    if result_name and not re.fullmatch('[A-Za-z0-9_.-]+',result_name):raise ValueError('Invalid result identity')
    write(ROOT/'analysis'/((result_name or 'compare_'+split+'_'+names[0]+suffix)+'.json'),result)
    return result


def contrast(names,controls,baseline,split,repetitions=10000,screening=None,cluster='parent',result_name=None):
    provenance=analysis_source(split)
    from navsim.agents.recogdrive.process_improvement.statistics import risk_difference_bootstrap
    base,rows=predictions(baseline,split)
    new=np.stack([predictions(name,split)[0] for name in names])
    control=np.stack([predictions(name,split)[0] for name in controls])
    if len(control)==1:control=np.broadcast_to(control,new.shape)
    if control.shape!=new.shape:raise ValueError('Controls must be fixed or paired one-to-one with training policies')
    high=None
    if screening:
        values,_=predictions(screening,split);high=(values[:,:,-1]>=.9).all(1)
    field={'parent':'driver_record','log':'log_name'}[cluster]
    result=risk_difference_bootstrap(base,control,new,[r[field] for r in rows],high_mask=high,repetitions=repetitions)
    result.update(new_methods=names,control_methods=controls,fixed_baseline=baseline,split=split,
                  control_training_replicates=len(controls),new_training_replicates=len(names),cluster_unit=cluster,analysis_source=provenance)
    suffix='' if cluster=='parent' else '_cluster_log'
    suffix+=('' if baseline=='B0_IL' else '_baseline_'+baseline)
    suffix+=('' if repetitions==10000 else '_boot'+str(repetitions))
    if cluster=='log':result['log_count']=result.pop('parent_count')
    if result_name and not re.fullmatch('[A-Za-z0-9_.-]+',result_name):raise ValueError('Invalid result identity')
    write(ROOT/'analysis'/((result_name or f'contrast_{split}_{names[0]}_vs_{controls[0]}{suffix}')+'.json'),result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--local');p.add_argument('--split',default='dev')
    p.add_argument('--methods',nargs='+');p.add_argument('--baseline',default='B0_IL')
    p.add_argument('--screening');p.add_argument('--limit',type=int);p.add_argument('--repetitions',type=int,default=10000)
    p.add_argument('--control-methods',nargs='+')
    p.add_argument('--cluster',choices=['parent','log'],default='parent')
    p.add_argument('--result-name',help='Explicit unique identity for grouped training-seed results')
    p.add_argument('--suite');p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    if a.local:
        result=local(a.local,a.split,a.repetitions,not a.allow_partial)
    elif a.methods:
        if a.control_methods:result=contrast(a.methods,a.control_methods,a.baseline,a.split,a.repetitions,a.screening,a.cluster,a.result_name)
        else:result=compare(a.methods,a.baseline,a.split,a.repetitions,a.limit,a.screening,a.cluster,a.result_name)
    else:
        runs={p.parent.name:read(p) for p in (ROOT/'runs').glob('*/complete.json')}
        result=dict(study=STUDY,actual_completed_runs=runs,protocol_lock=read(ROOT/'protocol.lock.json'))
        write(ROOT/'analysis/suite_inventory.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
