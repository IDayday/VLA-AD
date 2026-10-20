"""One-time selection using calibration-purpose data only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import numpy as np
from scripts.process_improvement.common import *
from scripts.process_improvement.analyze import predictions,local


def immutable(path,value):
    if Path(path).exists():
        if read(path)!=value:raise ValueError('A calibration lock cannot be overwritten')
    else:write(path,value)


def rates():
    selected={};trials=[]
    for method in ['OPD','ANCHOR']:
        own=[]
        for lr in ['1e-5','3e-5']:
            name='tune_'+method+'_'+lr
            values,rows=predictions(name,'cal_tune')
            complete=read(ROOT/'runs'/name/'complete.json')
            assert complete['steps']==812 and complete['scene_exposures']==25984
            own.append(dict(name=name,lr=float(lr),PDMS=float(values[:,:,-1].mean()),
                            scene_exposures=complete['scene_exposures'],data_split='cal_tune',
                            checkpoint_sha256=sha(ROOT/'runs'/name/'head_812.pt')))
        selected[method]=sorted(own,key=lambda x:(-x['PDMS'],x['lr']))[0]
        trials.extend(own)
    value=dict(status='LOCKED',selection='Complete train_25k epoch; highest cal_tune mean PDMS; ties smaller LR',
               selected=selected,trials=trials,calibration_manifest_sha256=sha(ROOT/'manifests/cal_tune.json'))
    immutable(ROOT/'selection/learning_rates.json',value)
    return value


def mechanism():
    # This procedure cannot look at dev/navtest or cal_confirm when choosing.
    plan=diagnostic_plan();result=local(plan['cal_name'],plan['cal_split'])
    source=[r for r in result['results'] if r['source']=='OLD_OPD']
    chosen=sorted(source,key=lambda r:(-r['mean'][2],r['rho'],-r['temperature']))[0]
    value=dict(status='LOCKED_FOR_INDEPENDENT_CONFIRMATION',primary_source='OLD_OPD',chosen=chosen,
               rule='Highest cal-tune guarded gain; exact ties prefer smaller rho then larger temperature',
               scope='Choice only, not a confirmation claim; cal-confirm remains unused',
               source_manifest_sha256=sha(ROOT/'manifests'/f"{plan['cal_split']}.json"),diagnostic_plan=plan)
    immutable(ROOT/'selection'/plan['choice_file'],value)
    import yaml
    config_path=SOURCE/plan['config']
    payload=dict(extends='base.yaml',temperature=chosen['temperature'],rho=chosen['rho'])
    if config_path.exists():assert yaml.safe_load(config_path.read_text())==payload
    else:config_path.write_text(yaml.safe_dump(payload,sort_keys=False))
    return value


def grpo_rates():
    selected={};trials=[]
    for initial in ['IL','RL']:
        own=[]
        for lr in ['1e-5','3e-5']:
            name=f'tune_GRPO_{initial}_{lr}'
            values,_=predictions(name,'cal_tune');complete=read(ROOT/'runs'/name/'complete.json')
            assert complete['steps']==812 and complete['scene_exposures']==25984 and complete['physical_scores']==207872
            own.append(dict(name=name,lr=float(lr),PDMS=float(values[:,:,-1].mean()),
                            data_split='cal_tune',physical_scores=complete['physical_scores'],
                            checkpoint_sha256=sha(ROOT/'runs'/name/'head_812.pt')))
        selected[initial]=sorted(own,key=lambda x:(-x['PDMS'],x['lr']))[0];trials.extend(own)
    value=dict(status='LOCKED',selected=selected,trials=trials,
               rule='One complete train_25k epoch each, same canonical scoring budget, best cal-tune PDMS, ties lower LR')
    immutable(ROOT/'selection/grpo_rates.json',value)
    return value


def warm():
    cfg=config();trials=[]
    extension_choice=read(ROOT/'selection/four_epoch_extension.json')
    for method in ['OPD','ANCHOR']:
        endpoints=[(1,2341),(2,4681)]
        if extension_choice['decisions'][method]['extend_to_four_epochs']:endpoints.append((4,9361))
        for epoch,step in endpoints:
            means=[];checkpoints={}
            for seed in cfg['train_seeds']:
                name=f'full_{method}_{seed}_epoch{epoch}'
                scores,_=predictions(name,'cal_tune');means.append(float(scores[:,:,-1].mean()))
                path=ROOT/'runs'/f'full_{method}_{seed}'/f'head_{step}.pt'
                checkpoints[str(seed)]=dict(path=str(path),sha256=sha(path))
            trials.append(dict(method=method,epoch=epoch,step=step,mean_PDMS=float(np.mean(means)),per_seed=means,checkpoints=checkpoints))
    chosen=sorted(trials,key=lambda r:(-r['mean_PDMS'],r['epoch'],r['method']!='OPD'))[0]
    value=dict(status='LOCKED',chosen=chosen,trials=trials,
               rule='Mean over all three training seeds on cal-tune; all registered eligible endpoints (1/2 plus conditional4); one recipe/epoch applied to each seeds own checkpoint')
    immutable(ROOT/'selection/warm_start.json',value)
    return value


def confirmation_membership():
    if (ROOT/'selection/diagnostic_protocol_correction.json').exists():
        value=read(ROOT/'selection/diagnostic_protocol_correction.json');name=value['active_plan']['confirm_split']
        return dict(status='LOCKED_PARENT_HASH_CORRECTION',**value['manifests'][name])
    original=membership('cal_confirm')
    selected=sorted(original,key=lambda r:digest([r['token'],'local-confirm-v1']))[:512]
    assert len(selected)==512
    value=ROOT/'manifests/diagnostic_confirm.json'
    immutable(value,selected)
    immutable(ROOT/'selection/confirmation_population.json',dict(status='LOCKED',rule='Fixed token hash local-confirm-v1; no score filtering',
              scenes=512,parents=len({r['driver_record'] for r in selected}),sha256=sha(value)))
    return read(ROOT/'selection/confirmation_population.json')


def extension():
    if (ROOT/'selection/four_epoch_extension.json').exists():return read(ROOT/'selection/four_epoch_extension.json')
    from navsim.agents.recogdrive.process_improvement.statistics import paired_bootstrap
    import shutil
    decisions={}
    for method in ['OPD','ANCHOR']:
        before=[];after=[]
        for seed in config()['train_seeds']:
            values,rows=predictions(f'full_{method}_{seed}_epoch1','cal_tune');before.append(values)
            values,_=predictions(f'full_{method}_{seed}_epoch2','cal_tune');after.append(values)
        result=paired_bootstrap(np.stack(before),np.stack(after),[r['driver_record'] for r in rows])
        # This exact registered rule is independent of dev/test results and
        # never changes a lineage's already planned optimizer/sampler schedule.
        free=shutil.disk_usage(ROOT).free
        storage=free>=config()['shared_free_floor_bytes']+(10<<30)
        extend=result['point']['G']>0 and result['CI95']['G'][1]>=.002 and storage
        decisions[method]=dict(extend_to_four_epochs=bool(extend),paired_cal_tune=result,
                               storage_available_bytes=free,storage_allows=storage)
    value=dict(status='LOCKED',decisions=decisions,rule='Positive epoch2-minus-epoch1 cal gain, CI upper >=0.002, and storage budget permits')
    immutable(ROOT/'selection/four_epoch_extension.json',value)
    return value


def process_gate():
    plan=diagnostic_plan();chosen=read(ROOT/'selection'/plan['choice_file'])['chosen']
    expanded=ROOT/'diagnostics'/plan['expansion_name']
    expansion_complete=expanded.exists() and len(list((expanded/'scenes').glob('*.json')))==512
    name=plan['expansion_name'] if expansion_complete else plan['confirm_name']
    result=local(name,plan['confirm_split'])
    item=next(r for r in result['results'] if r['source']=='OLD_OPD' and r['temperature']==chosen['temperature'] and r['rho']==chosen['rho'])
    mean=item['mean'][2];lower=item['CI95'][0][2];upper=item['CI95'][1][2]
    if mean>=.002 and lower>0:status='SUPPORTED'
    elif upper<.002:status='NOT_SUPPORTED'
    else:status='INCONCLUSIVE'
    value=dict(status=status,primary='OLD_OPD independent guarded native mean kernel gain',mean=mean,CI95=[lower,upper],
               material_gain=.002,source=result['name'],diagnostic=result,
               expansion_used=expansion_complete,expansion_allowed=status=='INCONCLUSIVE' and not expansion_complete,
               decision='Expand P2 only with supported held-out guarded kernel gain; independent OPD/GRPO always continue')
    # A registered expansion may supersede an initial inconclusive decision;
    # preserve both immutable decisions and their chronology.
    immutable(ROOT/'selection'/f'process_gate_{name}.json',value)
    write(ROOT/'selection/process_gate.json',value)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['rates','grpo_rates','mechanism','warm','extension','confirmation_membership','process_gate'],required=True)
    args=p.parse_args();print(json.dumps(globals()[args.stage](),ensure_ascii=False,indent=2),flush=True)
