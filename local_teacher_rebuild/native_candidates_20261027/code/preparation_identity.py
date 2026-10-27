"""CPU-only completion identities for the current local teacher preparation.

These checks authorize reuse of saved results. They never load a GPU, rerun a
teacher, or replace the independent formal qualification checks.
"""
from pathlib import Path
import json
import numpy as np

COUNTS={'train':85813,'test':12146}


def require(condition,message):
    if not condition:raise ValueError(message)


def qualification_identities(paths,results):
    from student_distillation.asset_identity import sha256_file
    files={name:paths[name] for name in ('teacher_bundle','feature_manifest','targets_manifest','calibration','checkpoint')}
    files.update(input_statistics=Path(results)/'calibration/BASE_INPUT_STATISTICS.json',
                 reference_noise_manifest=paths['evaluation_noise_manifest'])
    return {name+'_sha256':sha256_file(path) for name,path in files.items()}


def validate_current_output(name,path,*,paths,results,code,world_size):
    from student_distillation.asset_identity import canonical_sha,load_library_bundle,read_json,resolve_ref,sha256_file
    value=read_json(path)
    bundle=load_library_bundle(paths['teacher_bundle']);teacher=bundle['identity']
    feature_path=Path(paths['feature_manifest']);features=read_json(feature_path)
    if name.startswith('targets_'):
        from student_distillation.build_targets import read_local_selection
        population=name.removeprefix('targets_');spec=value['populations'][population]
        require(value.get('schema')=='nrkd_compact_targets_v1' and value.get('teacher_origin')=='local_refit' and
                value.get('immutable_population')==population and set(value['populations'])=={population},'Target schema/population differs')
        require(value.get('teacher_identity')==teacher,'Targets belong to another frozen teacher')
        identity=spec['identity'];selection=read_local_selection(bundle,population)
        expected={'teacher':teacher,'selection_sha256':selection['source_sha256'],
                  'feature_manifest_sha256':sha256_file(feature_path),
                  'feature_population_identity_sha256':canonical_sha(features['populations'][population]),
                  'target_versions':['full','gain1'],'population':population,
                  'diagnostic_only':population!='train','execution_device_type':'cuda'}
        require(identity==expected and spec.get('identity_sha256')==canonical_sha(identity),'Target source/selection identity differs')
        require(spec.get('count')==COUNTS[population] and spec.get('diagnostic_only')==(population!='train') and
                set(spec.get('versions',{}))=={'full','gain1'},'Full target coverage/versions differ')
        resolve_ref(spec['index'],Path(path).parent)
        for version in spec['versions'].values():
            array=np.load(resolve_ref(version['shifts'],Path(path).parent),mmap_mode='r',allow_pickle=False)
            require(array.dtype==np.float32 and array.shape==(COUNTS[population],384),'Target array layout differs')
        done=read_json(Path(path).parent/'COMPLETE.json')
        require(done.get('status')=='COMPLETE' and done.get('identity_sha256')==spec['identity_sha256'] and
                done.get('population_spec')==spec,'Target completion receipt differs from its manifest')
        return True
    if name=='train_calibration':
        body=dict(value);recorded=body.pop('identity_sha256',None)
        require(value.get('schema')=='nrkd_train_calibration_v1' and value.get('status')=='COMPLETE' and
                value.get('population')=='train' and value.get('teacher_origin')=='local_refit','Calibration schema is incomplete')
        require(value.get('teacher_identity')==teacher and value.get('feature_manifest_sha256')==sha256_file(feature_path) and
                value.get('targets_manifest_sha256')==sha256_file(paths['targets_manifest']) and
                value.get('base_input_statistics_sha256')==sha256_file(Path(results)/'calibration/BASE_INPUT_STATISTICS.json') and
                recorded==canonical_sha(body),'Calibration current teacher/assets/self identity differs')
        return True
    expected=qualification_identities(paths,results)
    require(all(value.get('identities',{}).get(k)==v for k,v in expected.items()),'Qualification belongs to different current assets')
    current_code={str(p.relative_to(code)):sha256_file(p) for p in sorted((Path(code)/'student_distillation').glob('*.py'))}
    require(value.get('code_sha256')==current_code,'Qualification belongs to different current code')
    if name=='formal_ddp':
        from student_distillation.qualification import validate_ddp_execution
        execution=validate_ddp_execution(value,world_size)
        require(value.get('execution')==execution,'DDP qualification execution identity differs')
        return True
    if name=='formal_single':
        from student_distillation.experiment_registry import load_config
        from student_distillation.train import verify_qualification
        require(value.get('formal_asset_qualification') is True and not value.get('pending_checks'),'Formal qualification is pending')
        for experiment in ('H_full','H_mu_teacher'):
            cfg=load_config(Path(code)/'configs/student_distillation'/f'{experiment}.yaml',Path(code)/'configs/student_distillation/common.yaml')
            cfg['paths']=dict(paths,qualification_receipt=str(path));cfg['runtime']={'world_size':world_size}
            cfg['training']['gradient_accumulation_steps']=64//world_size
            verify_qualification(cfg)
        return True
    raise ValueError('Unknown production preparation output: '+name)
