"""Orchestration-only scope adapter; frozen training/config/source stay intact."""
import json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
CODE=Path('/mnt/project/recogdrive-no-routing-round1-20261026')
INSTRUCTION=ROOT/'engineering/single_seed_20261027/USER_INSTRUCTION.json'
SEEDS=(20261027,)

def install():
    sys.path.insert(0,str(CODE))
    import student_distillation.experiment_registry as reg
    import student_distillation.run_suite as suite
    import student_distillation.select_checkpoint as selection
    import student_distillation.aggregate as aggregate
    from student_distillation.checkpointing import atomic_json,digest,sha256
    if getattr(suite,'_single_seed_scope_installed',False):return suite
    expected={f'{arm}_seed20261027' for arm in reg.EXPERIMENTS}
    original_registry=reg.Registry
    class SingleSeedRegistry(original_registry):
        def __init__(self,out,experiments=reg.EXPERIMENTS,seeds=SEEDS):
            if tuple(seeds)!=SEEDS:raise ValueError('Only the user-retained seed may be scheduled')
            super().__init__(out,experiments=experiments,seeds=SEEDS)
            if set(self.value['runs'])!=expected:raise ValueError('Cancelled runs remain in the active registry')
    reg.SEEDS=SEEDS;reg.Registry=SingleSeedRegistry;suite.SEEDS=SEEDS;suite.Registry=SingleSeedRegistry
    instruction={'path':str(INSTRUCTION),'sha256':sha256(INSTRUCTION)}

    def lock_suite(out):
        out=Path(out);registry=json.loads((out/'EXPERIMENT_REGISTRY.json').read_text())
        if set(registry['runs'])!=expected:raise ValueError('Test requires all four retained runs')
        selections={run:selection.select(out/run) for run in sorted(expected)}
        value={'schema':'nrkd_single_seed_selection_lock_v1','test_may_start':True,
          'all_registered_train_and_full_validation_complete':True,'registered_training_seeds':[20261027],
          'registered_run_count':4,'user_instruction':instruction,'runs':selections,
          'selection_protocol_sha256':digest({'endpoints':selection.ENDPOINTS,'noise_indices':[0,1,2,3,4],
                                             'population':17475,'tie_break':'earlier_epoch'})}
        value['lock_sha256']=digest(value);path=out/'CHECKPOINT_SELECTION_LOCK.json'
        if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Cannot replace a frozen selection lock')
        atomic_json(path,value);return value

    def verify_selection_lock(out,*,full=False):
        out=Path(out);value=json.loads((out/'CHECKPOINT_SELECTION_LOCK.json').read_text())
        if digest({k:v for k,v in value.items() if k!='lock_sha256'})!=value.get('lock_sha256'):
            raise ValueError('Single-seed Test selection lock digest changed')
        if (value.get('schema')!='nrkd_single_seed_selection_lock_v1' or set(value.get('runs',{}))!=expected or
            value.get('test_may_start') is not True or value.get('all_registered_train_and_full_validation_complete') is not True or
            value.get('registered_training_seeds')!=[20261027] or value.get('registered_run_count')!=4 or
            value.get('user_instruction')!=instruction or sha256(INSTRUCTION)!=instruction['sha256']):
            raise ValueError('Test requires the complete user-retained four-run selection lock')
        for run_id,entry in value['runs'].items():
            folder=out/run_id
            if sha256(folder/'TRAIN_COMPLETE.json')!=entry.get('train_complete_sha256'):
                raise ValueError('Training completion changed after selection')
            if full:
                if selection.select(folder)!=entry:raise ValueError('Full Validation selection changed')
            else:
                for candidate in entry['candidates']:
                    marker=folder/'validation'/f"epoch_{candidate['epoch']:03d}"/'COMPLETE.json'
                    if sha256(marker)!=candidate['validation_complete_sha256']:raise ValueError('Full Validation evidence changed')
                for label in ('last','best'):
                    candidate=entry[label]
                    if (sha256(candidate['checkpoint'])!=candidate['sha256'] or
                        sha256(candidate['checkpoint']+'.COMPLETE.json')!=candidate['checkpoint_complete_sha256']):
                        raise ValueError('Locked checkpoint bytes changed')
        return value
    selection.lock_suite=lock_suite;selection.verify_selection_lock=verify_selection_lock

    original_command=suite.child_command
    worker=Path(__file__).with_name('single_seed_worker.py')
    def child_command(module,arguments,world_size):
        # Existing training specs must remain exactly identical for live adoption.
        if module=='student_distillation.train':return original_command(module,arguments,world_size)
        if module!='student_distillation.evaluate':raise ValueError('Unexpected scoped child module')
        prefix=[sys.executable,'-B']
        if world_size>1:prefix+=['-m','torch.distributed.run','--standalone',f'--nproc_per_node={world_size}']
        return prefix+[str(worker),'--module',module,'--',*map(str,arguments)]
    suite.child_command=child_command
    original_report=aggregate.report
    def report(out,*args,**kwargs):
        value=original_report(out,*args,**kwargs)
        path=Path(out)/'ROUND1_REPORT.md';text=path.read_text()
        text=text.replace('20261027 and 20261028 are the two independent training seeds',
                          '20261027 is the sole retained training seed; 20261028 was cancelled by the user')
        text=text.replace('Two training seeds provide limited information about training randomness.',
                          'One retained training seed provides no estimate of variability across training seeds.')
        text+='\nUser scope: four A/B/C/D runs with seed 20261027; each retains 40 epochs and 53,640 updates. Cancelled seed20261028 partial runs are preserved and excluded from comparisons.\n'
        path.write_text(text);return value
    aggregate.report=report
    suite._single_seed_scope_installed=True
    return suite
