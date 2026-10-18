"""Replay the completed scientific matrix in a NEW artifact root on four idle GPUs.

Reuses locked membership/protocol, reconstructs predictions/labels/training, and
never overwrites the original study. Slower than the original multi-host dispatch.
"""
from __future__ import annotations
import argparse
import shutil
from .common import *
from .dispatch import launch


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--gpus',default='0,1,2,3');a=p.parse_args()
    original=PROJECT/'outputs/trajectory_effect_validation_20261019';gpus=[int(x) for x in a.gpus.split(',')]
    if len(gpus)!=4 or len(set(gpus))!=4:raise ValueError('Specify four distinct idle GPUs')
    if ROOT==original or ROOT.exists():raise ValueError('Set TRAJECTORY_EFFECT_RUN_ROOT to a new, nonexistent directory')
    host=require_host();ROOT.mkdir(parents=True)
    shutil.copy2(original/'protocol.lock.json',ROOT/'protocol.lock.json');(ROOT/'manifests').mkdir()
    for name in ['D_fit','D_cal','D_audit','pilot','update_families','source_weights','splits_identity']:
        shutil.copy2(original/'manifests'/f'{name}.json',ROOT/'manifests'/f'{name}.json')
    atomic_json(ROOT/'replication.json',dict(original=str(original),protocol_sha256=protocol()[1],host=host,gpus=gpus,
        mode='Independent fresh execution with the original frozen membership, seeds and protocol'))
    counter=0
    def start(command,devices):
        nonlocal counter
        counter+=1;name=f'replay_{counter:04}'
        launch(dict(name=name,host=host,gpus=devices,commands=[command]));return ROOT/'jobs'/name
    def join(paths):
        for path in paths:
            while not (path/'complete.json').exists():time.sleep(2)
            if read(path/'complete.json')['status']!='COMPLETE':raise RuntimeError(f'Replay job failed: {path}/worker.log')
    def run(command,devices=[]):join([start(command,devices)])
    def parallel(module,arguments,shards=4):
        join([start([module,*arguments,'--rank',str(r),'--shards',str(shards)],[gpus[r]]) for r in range(shards)])
    def distributed(command,world=4):
        run(['torch.distributed.run','--standalone','--nproc_per_node',str(world),'--module',*command],gpus[:world])
    def update_range(first,last):
        for i in range(first,last):
            family=f'update_{i:02}'
            run(['tools.trajectory_effect.updates','train','--family',family],[gpus[0]])
            parallel('tools.trajectory_effect.updates',['evaluate','--family',family])
            run(['tools.trajectory_effect.updates','score','--family',family,'--workers','8'])
    for phase in ['inference','score','train','joint']:
        run(['tools.trajectory_effect.audit','--phase',phase,'--limit','1' if phase=='train' else '64'],[gpus[0]] if phase!='score' else [])
    for split in ['D_fit','D_cal','D_audit']:
        parallel('tools.trajectory_effect.bank',['extract','--split',split])
        run(['tools.trajectory_effect.bank','score','--split',split,'--workers','8'])
    parallel('tools.trajectory_effect.screen',['extract']);run(['tools.trajectory_effect.screen','score','--workers','8'])
    update_range(0,12)
    run(['tools.trajectory_effect.candidates','scales']);parallel('tools.trajectory_effect.candidates',['extract'])
    run(['tools.trajectory_effect.candidates','score','--workers','8'])
    for split in ['D_fit','D_cal','D_audit']:run(['tools.trajectory_effect.candidates','assemble','--split',split])
    run(['tools.trajectory_effect.statistics','h1'])
    for arm in ['T','FQ','FE','JQ','JE']:
        for slot in range(3):
            name=f'tuning_{arm}_{slot}';lr=[1e-4,3e-4,1e-3][slot] if arm in ['T','FQ','JQ'] else 3e-4
            lam=[.25,1.,4.][slot] if arm in ['FE','JE'] else (1. if arm=='T' else 0.)
            train=['tools.trajectory_effect.probe_runner','train','--name',name,'--arm',arm,'--seed','41','--steps','300','--phase','tuning','--head-lr',str(lr),'--lambda-diff',str(lam)]
            evaluate=['tools.trajectory_effect.probe_runner','evaluate','--name',name,'--split','D_cal']
            if arm in ['JQ','JE']:distributed(train);distributed(evaluate)
            else:run(train,[gpus[0]]);run(evaluate,[gpus[0]])
    run(['tools.trajectory_effect.probe_runner','select'])
    if read(ROOT/'analysis/h1.json')['gate_B']!='INSUFFICIENT_EVENTS':raise RuntimeError('Initial gate differs from original study; review before any extension')
    (ROOT/'confirmation').mkdir()
    activation=read(original/'confirmation/activation.json');activation['replication_of']=str(original)
    activation['trigger_record_sha256']=sha(ROOT/'analysis/h1.json');atomic_json(ROOT/'confirmation/activation.json',activation)
    shutil.copy2(original/'manifests/confirmation_families.json',ROOT/'manifests/confirmation_families.json')
    for name in ['h1','h1_scene_changes']:shutil.copy2(ROOT/'analysis'/f'{name}.json',ROOT/'analysis'/f'{name}_initial.json')
    update_range(12,24)
    for split in ['D_fit','D_cal','D_audit']:run(['tools.trajectory_effect.candidates','assemble','--split',split,'--package','confirmation'])
    for i in [9,10,11,21,22,23]:parallel('tools.trajectory_effect.severe',['evaluate','--family',f'update_{i:02}'])
    run(['tools.trajectory_effect.severe','analyze']);run(['tools.trajectory_effect.statistics','h1'])
    run(['tools.trajectory_effect.controls','shuffle','--split','D_audit'])
    selection=read(ROOT/'probes/selection.json')['selected']
    for arm in ['T','FQ','FE','JQ','JE']:
        for seed in [42,43,44]:
            name=f'formal_{arm}_{seed}';c=selection[arm];world=4 if arm in ['JQ','JE'] or (arm=='T' and seed==42) else 2
            distributed(['tools.trajectory_effect.probe_runner','train','--name',name,'--arm',arm,'--seed',str(seed),'--steps','2000','--phase','formal',
                         '--head-lr',str(c['head_lr']),'--lambda-diff',str(c['lambda_diff'])],world)
            for split in ['D_fit','D_cal','D_audit']:
                distributed(['tools.trajectory_effect.probe_runner','evaluate','--name',name,'--split',split],world)
            if arm=='JE':distributed(['tools.trajectory_effect.probe_runner','evaluate','--name',name,'--split','D_audit','--shuffle'],world)
    for module,arguments in [
        ('controls',['ridge']),('probe_metrics',['--prefix','formal','--split','D_audit']),('probe_diagnostics',[]),('secondary_intervals',[]),
        ('diagnostics',['high']),('diagnostics',['coverage']),('diagnostics',['tracking']),('input_identity',[]),('conclude',[])]:
        run(['tools.trajectory_effect.'+module,*arguments])
    if read(ROOT/'analysis/stage_admission.json')['stage_E']=='PASS':raise RuntimeError('Replication admits a new stage; review the changed evidence before entering RL')
    # Invoke final accounting outside an owned job, so it can verify all jobs exited.
    subprocess.run([PYTHON,'-m','tools.trajectory_effect.accounting','--final'],cwd=SOURCE,check=True)
    subprocess.run([PYTHON,'-m','tools.trajectory_effect','report'],cwd=SOURCE,check=True)


if __name__=='__main__':main()
