"""Private CPU fixtures for stale-scope rejection and safe output recovery."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import runpy
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CODE=Path('/mnt/project/recogdrive-no-routing-round1-20261026')
sys.path[:0]=[str(ROOT/'code'),str(CODE)]
from gt_numeric import atomic,sha,stamp
from preparation_identity import validate_current_output,qualification_identities
from round1_prepare_and_run import PreparationOwner
from student_distillation.asset_identity import canonical_sha
from student_distillation.train import QUALIFICATION_CHECKS


def main():
    with tempfile.TemporaryDirectory(prefix='native_prep_identity_') as temporary:
        folder=Path(temporary);results=folder/'results';results.mkdir()
        def put(path,value):atomic(path,value);return path
        def ref(path):return {'path':str(path),'sha256':sha(path)}
        bundle=put(folder/'BUNDLE.json',{'teacher':'NEW_SYNTHETIC_ONLY'})
        feature=put(folder/'FEATURES.json',{'populations':{'train':{'count':3},'test':{'count':2}}})
        checkpoint=put(folder/'CHECKPOINT.json',{'synthetic':True})
        noise=put(folder/'NOISE.json',{'synthetic':True})
        stats=put(results/'calibration/BASE_INPUT_STATISTICS.json',{'synthetic':True})
        selected=put(folder/'SELECTION.json',{'synthetic_current_selection':True})
        teacher={'bundle_path':str(bundle),'bundle_sha256':sha(bundle),'library_sha256':'a'*64,
                 'basis_sha256':'b'*64,'source_checkpoint_sha256':sha(checkpoint),
                 'runtime_predictor_sha256':None,'teacher_identity':'NEW_ONLY','teacher_origin':'local_refit'}
        paths={'teacher_bundle':str(bundle),'feature_manifest':str(feature),'checkpoint':str(checkpoint),
               'evaluation_noise_manifest':str(noise),'targets_manifest':str(results/'targets/train/MANIFEST.json'),
               'calibration':str(results/'calibration/CALIBRATION.json'),'results_root':str(results)}
        def target():
            f=results/'targets/train';f.mkdir(parents=True,exist_ok=True)
            index=f/'INDEX.jsonl';index.write_text('synthetic fixture index\n')
            versions={}
            for version in ('full','gain1'):
                p=f/(version+'_shifts.npy');np.save(p,np.zeros((3,384),np.float32));versions[version]={'shifts':ref(p)}
            identity={'teacher':teacher,'selection_sha256':sha(selected),'feature_manifest_sha256':sha(feature),
                      'feature_population_identity_sha256':canonical_sha({'count':3}),'target_versions':['full','gain1'],
                      'population':'train','diagnostic_only':False,'execution_device_type':'cuda'}
            spec={'count':3,'diagnostic_only':False,'identity':identity,'identity_sha256':canonical_sha(identity),
                  'index':ref(index),'versions':versions}
            put(f/'COMPLETE.json',{'status':'COMPLETE','identity_sha256':spec['identity_sha256'],'population_spec':spec})
            return put(f/'MANIFEST.json',{'schema':'nrkd_compact_targets_v1','teacher_origin':'local_refit',
                'teacher_identity':teacher,'immutable_population':'train','populations':{'train':spec}})
        def validate(name,path):
            return validate_current_output(name,path,paths=paths,results=results,code=CODE,world_size=4)
        def reject(name,path):
            try:validate(name,path)
            except (ValueError,KeyError,TypeError,OSError):return
            raise AssertionError('Incompatible output was accepted: '+name)
        with patch('student_distillation.asset_identity.load_library_bundle',return_value={'identity':teacher}), \
             patch('student_distillation.build_targets.read_local_selection',return_value={'source_sha256':sha(selected)}), \
             patch('preparation_identity.COUNTS',{'train':3,'test':2}), \
             patch('student_distillation.losses.teacher_query_cache_settings',return_value=None):
            target_path=target();assert validate('targets_train',target_path)
            good=json.loads(target_path.read_text());stale=copy.deepcopy(good);stale['teacher_identity']['bundle_sha256']='0'*64
            put(target_path,stale);reject('targets_train',target_path);put(target_path,good)
            stale=copy.deepcopy(good);stale['populations']['train']['identity']['selection_sha256']='0'*64
            put(target_path,stale);reject('targets_train',target_path);put(target_path,good)
            calibration={'schema':'nrkd_train_calibration_v1','status':'COMPLETE','population':'train','teacher_origin':'local_refit',
                'teacher_identity':teacher,'feature_manifest_sha256':sha(feature),'targets_manifest_sha256':sha(target_path),
                'base_input_statistics_sha256':sha(stats),'synthetic_fixture_only':True}
            calibration['identity_sha256']=canonical_sha(calibration);cal=put(Path(paths['calibration']),calibration)
            assert validate('train_calibration',cal)
            bad=copy.deepcopy(calibration);bad['teacher_identity']['bundle_sha256']='0'*64;body=dict(bad);body.pop('identity_sha256');bad['identity_sha256']=canonical_sha(body)
            put(cal,bad);reject('train_calibration',cal);put(cal,calibration)
            ids=qualification_identities(paths,results);code={str(p.relative_to(CODE)):sha(p) for p in sorted((CODE/'student_distillation').glob('*.py'))}
            execution={'world_size':4,'global_batch_size':64,'microbatch_per_device':1,'gradient_accumulation_steps':16,'DDP_adapter_only':True}
            ddp={'schema':'nrkd_ddp_tail_check_v1','status':'PASS','world_size':4,'global_slots':64,'real_scene_count':53,
                 'zero_weight_padded_slots':11,'native_microbatch':1,'gradient_accumulation_steps':16,'DDP_adapter_only':True,
                 'execution':execution,'identities':ids,'code_sha256':code}
            ddp_path=put(results/'qualification/formal_ddp/DDP_TAIL_53.json',ddp);assert validate('formal_ddp',ddp_path)
            bad=copy.deepcopy(ddp);bad['identities']['teacher_bundle_sha256']='0'*64;put(ddp_path,bad);reject('formal_ddp',ddp_path);put(ddp_path,ddp)
            checks={}
            for name in QUALIFICATION_CHECKS:
                body={'synthetic_fixture_only':True}
                if name=='next_update_recovery':body['active_transition_recovery']={s:{'status':'PASS','trained_Adam_state':True,'transition_loss_active':True,'independent_loads':2} for s in ('teacher','student')}
                if name=='dit_input_gradient':body['student_forward_native_parity']={'status':'PASS','scenes':64,'time_indices':[0,1,2,3,4],'bitwise':True}
                if name=='ddp_tail_objective':body=ddp
                artifact=put(results/'qualification/formal_single'/(name+'.json'),body);checks[name]={'status':'PASS','artifact':ref(artifact)}
            single={'schema':'nrkd_code_checks_v1','status':'CODE_CHECKS_PASS','formal_asset_qualification':True,
                    'pending_checks':[],'execution':execution,'identities':ids,'code_sha256':code,'checks':checks,'synthetic_fixture_only':True}
            single_path=put(results/'qualification/formal_single/QUALIFICATION.json',single);assert validate('formal_single',single_path)
            bad=copy.deepcopy(single);bad['identities']['teacher_bundle_sha256']='0'*64;put(single_path,bad);reject('formal_single',single_path)
            bad=copy.deepcopy(single);bad['checks']['native_bridge_replay']['artifact']['sha256']='0'*64;put(single_path,bad);reject('formal_single',single_path);put(single_path,single)
            # Current identity is reusable; stale outputs are preserved only
            # after a terminal child proof. Unknown/live ownership is retained.
            owner=PreparationOwner.__new__(PreparationOwner);owner.folder=results/'preparation_owner';owner.folder.mkdir()
            owner.args=SimpleNamespace(results=results,code=CODE);owner.current_paths=paths;owner.training_world_size=4
            class Unknown(RuntimeError):pass
            assert owner.gpu_output_complete('targets_train',target_path)
            owner.retire_incompatible_output('targets_train',target_path,Unknown);assert target_path.exists()
            put(target_path,stale);family=owner.folder/'remote_execution/jobs/targets_train';attempt=family/'attempt_00000';attempt.mkdir(parents=True)
            spec=put(attempt/'SPEC.json',{});put(family/'ACTIVE_ATTEMPT.json',{'spec':str(spec)})
            put(attempt/'STATUS.json',{'status':'RUNNING','child_started':True})
            try:owner.retire_incompatible_output('targets_train',target_path,Unknown)
            except Unknown:pass
            else:raise AssertionError('UNKNOWN/live child output was moved')
            assert target_path.exists()
            put(attempt/'STATUS.json',{'status':'EXITED','returncode':0})
            owner.retire_incompatible_output('targets_train',target_path,Unknown);assert not target_path.exists()
            assert list((owner.folder/'incompatible_outputs/targets_train').glob('*/MANIFEST.json'))
            target_path=target();assert owner.gpu_output_complete('targets_train',target_path)
            from student_distillation.run_suite import Controller
            class Registry:
                value={'tasks':{}}
                def task_update(self,*args,**kwargs):raise AssertionError('Valid current output triggered a new attempt')
            allocation={'_resource_status':'READY_FOR_FORMAL_DISPATCH','gpu_uuids':['PRIVATE_CPU_ONLY'],
                'python':sys.executable,'workdir':str(CODE),'host':'NO_REAL_HOST',
                '_live_policy_path':'NO_REAL_POLICY','_task':'PRIVATE_CPU_ONLY'}
            put(attempt/'STATUS.json',{'status':'EXITED','returncode':0,'wall_seconds':123.456})
            controller=Controller(owner.folder/'remote_execution',Registry(),{'targets_train':allocation})
            for _ in range(2):
                assert controller.execute('targets_train',[sys.executable,'-c','raise RuntimeError("must never launch")'],
                    allocation_run='targets_train',success_check=lambda:owner.gpu_output_complete('targets_train',target_path))==123.456
            assert len(list(family.glob('attempt_*')))==1
        original_read=Path.read_bytes
        for target_source in (
            Path('/mnt/project/VLA-AD/experiments/recogdrive_dc8_refit_20261026/engineering/local_candidate_recheck_20261027/fit_library_local.py'),
            Path('/mnt/project/VLA-AD/experiments/recogdrive_dc8_refit_20261026/code/fit_library.py')):
            def changed_source(path):
                return b'PRIVATE_CORRUPTED_SOURCE' if path==target_source else original_read(path)
            with patch.object(Path,'read_bytes',changed_source):
                try:runpy.run_path(str(ROOT/'code/fit_library.py'),run_name='private_fitter_import_guard_check')
                except ValueError as error:assert 'before import' in str(error)
                else:raise AssertionError('Changed fitter dependency passed before-import SHA guard')
    receipt={'status':'PASS_PRIVATE_CPU_PREPARATION_IDENTITY_AND_RECOVERY','time':stamp(),'source':str(ROOT/'code/preparation_identity.py'),
        'source_sha256':sha(ROOT/'code/preparation_identity.py'),'old_teacher_targets_calibration_DDP_single_all_rejected':True,
        'current_identity_all_four_outputs_accepted':True,'selection_identity_and_nine_check_artifact_corruption_rejected':True,
        'live_unknown_output_preserved':True,'terminal_incompatible_output_preserved_then_current_identity_rebuilt':True,
        'both_external_fitter_source_changes_rejected_before_import':True,
        'real_Controller_current_identity_completed_attempt_reused_twice_without_launch':True,
        'GPU_model_training_or_real_asset_mutations':0,'synthetic_fixture_only':True,'formal_qualification':False}
    atomic(ROOT/'engineering/PREPARATION_IDENTITY_CPU_CHECK.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
