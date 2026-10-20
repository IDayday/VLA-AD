"""Correct specification section16 before confirmatory diagnostic observations.

Preserve token-hash pilot artifacts. Parent hashes determine a fixed round-robin
order, with deterministic within-parent ordering. This spreads 512 fixed scenes
over all available independent driving records without score-based filtering.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import shutil
from collections import defaultdict
from scripts.process_improvement.common import *


def parent_sample(rows,count,purpose):
    groups=defaultdict(list)
    for row in rows:groups[row['driver_record']].append(row)
    parents=sorted(groups,key=lambda p:digest([STUDY,'diagnostic-parent-v2',purpose,p]))
    for parent in parents:groups[parent].sort(key=lambda r:digest([STUDY,'within-diagnostic-parent-v2',purpose,r['token']]))
    selected=[];depth=0
    while len(selected)<count:
        size=len(selected)
        for parent in parents:
            if depth<len(groups[parent]):selected.append(groups[parent][depth])
            if len(selected)==count:break
        if len(selected)==size:raise ValueError('Insufficient real scenes for parent-hash sample')
        depth+=1
    if len({r['token'] for r in selected})!=count:raise ValueError('Duplicate token in diagnostic population')
    return selected


def main():
    correction=ROOT/'selection/diagnostic_protocol_correction.json'
    if correction.exists():
        value=read(correction)
        for name,record in value['manifests'].items():assert sha(ROOT/'manifests'/f'{name}.json')==record['sha256']
        print('PARENT_HASH_CORRECTION_ALREADY_LOCKED',flush=True);return
    assert not list((ROOT/'diagnostics/local_confirm/scenes').glob('*.json')),'Confirmatory labels already exist; stop for explicit audit rather than silently changing samples'
    populations={};identities={}
    for purpose,source in [('train','train_full'),('cal','cal_tune'),('confirm','cal_confirm')]:
        name='diagnostic_parent_'+purpose;rows=parent_sample(membership(source),512,purpose)
        path=ROOT/'manifests'/f'{name}.json'
        if path.exists():assert read(path)==rows
        else:write(path,rows)
        populations[purpose]=rows
        identities[name]=dict(sha256=sha(path),source_split=source,scenes=512,
                             parents=len({r['driver_record'] for r in rows}),rule='Parent-hash round robin; fixed within-parent token order; no score filter')
    active=dict(train_split='diagnostic_parent_train',cal_split='diagnostic_parent_cal',confirm_split='diagnostic_parent_confirm',
                train_name='local_parent_train_grid',cal_name='local_parent_cal_grid',confirm_name='local_parent_confirm',
                expansion_name='local_parent_confirm_expand32',choice_file='process_hyperparameters_parenthash.json',
                config='configs/process_improvement/selected_process_parenthash.yaml')
    reused=[]
    for purpose,old_name,new_name in [('train','local_train_grid',active['train_name']),('cal','local_cal_grid',active['cal_name'])]:
        old=ROOT/'diagnostics'/old_name
        for row in populations[purpose]:
            path=old/'scenes'/f"{row['token']}.json"
            if not path.exists():continue
            value=read(path);assert value['token']==row['token'] and value['driver_record']==row['driver_record']
            assert len(value['records'])==180
            assert {(r['temperature_base'],r['rho']) for r in value['records']}=={(t,r) for t in [.02,.05,.1] for r in [.25,.5]}
            target=ROOT/'diagnostics'/new_name/'scenes'/path.name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert sha(target)==sha(path)
            else:shutil.copyfile(path,target)
            reused.append(dict(token=row['token'],source=str(path),destination=str(target),sha256=sha(path),
                               scope='Same scene/state source/checkpoints/MLK/CRN and score; valid label reuse; no new physical score charged'))
        if (old/'scenes').exists():
            write(old/'sampling_role.json',dict(status='EXPLORATORY_NONCONFORMING_SAMPLER',
                  reason='Token-hash subset instead of requested parent-hash subset; labels and parent-purpose isolation remain valid',
                  replacement=new_name,old_results_preserved=True))
    value=dict(status='LOCKED_BEFORE_CONFIRMATION',reason='Correct specification section16 diagnostic sampling unit',
               specification_quote='先在固定512个train场景上诊断，按父记录哈希取样而不是只取已知可修复场景。',
               original_protocol_sha256=sha(ROOT/'protocol.yaml'),active_plan=active,manifests=identities,
               earlier_tokenhash_choice_preserved_sha256=sha(ROOT/'selection/process_hyperparameters.json') if (ROOT/'selection/process_hyperparameters.json').exists() else None,
               reused_valid_labels=reused,confirmatory_labels_observed_before_correction=0,
               unchanged=['score protocol','six hyperparameter pairs','M16/Lfit8/Lguard8/Ltest16','three rollin sources',
                          'full training populations','OPD/Anchor LR selection','RL controls','thresholds','one registered confirmation expansion'],
               extra_cost='Engineering retest cost is recorded separately; not a new scientific hyperparameter expansion')
    write(correction,value);write(SOURCE/'configs/process_improvement/diagnostic_sampling_correction.json',value)
    print('PARENT_HASH_DIAGNOSTICS_LOCKED',identities,'reused',len(reused),flush=True)


if __name__=='__main__':main()
