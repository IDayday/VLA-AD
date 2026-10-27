"""CPU-only admission tests using the completed candidate plus private mutations."""
import ast,copy,json,os,subprocess,sys,tempfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
CODE=Path('/mnt/project/recogdrive-no-routing-round1-20261026')
RESULTS=Path('/mnt/project/recogdrive-no-routing-round1-results-20261026')
sys.path[:0]=[str(ROOT/'code'),str(CODE)]
from gt_numeric import atomic,read,sha,stamp
from cached_target_publication import validate_publication,reference
from preparation_identity import validate_current_output,validate_cached_target_construction

def main():
    manifest=RESULTS/'targets_verified_native_cache_stage/train/MANIFEST.json'
    if not manifest.exists():manifest=RESULTS/'targets/train/MANIFEST.json'
    paths=read(RESULTS/'protocol/PATHS.json');spec=read(manifest)['populations']['train']
    from student_distillation.asset_identity import load_library_bundle,read_json,resolve_ref
    from student_distillation.build_targets import read_local_selection
    features=read(paths['feature_manifest']);bundle=load_library_bundle(paths['teacher_bundle']);selection=read_local_selection(bundle,'train')
    assert validate_current_output('targets_train',manifest,paths=paths,results=RESULTS,code=CODE,world_size=4)
    construction=resolve_ref(spec['construction_receipt'],manifest.parent);good=read(construction)
    mutations=[('private_synthetic_fixture',True),('scenes_verified',85812),('ordered_tokens_sha256','0'*64),
               ('per_row_check_counts',{}),('source_refs',{}),('builder_source',{'path':'none','sha256':'0'*64})]
    for key,value in mutations:
        bad=copy.deepcopy(good);bad[key]=value
        def modified(path):return bad if Path(path)==construction else read_json(path)
        with patch('student_distillation.asset_identity.read_json',side_effect=modified):
            try:validate_cached_target_construction(spec,manifest,paths=paths,features=features,teacher=bundle['identity'],selection=selection,code=CODE)
            except (ValueError,KeyError,OSError):pass
            else:raise AssertionError('Bad cached target admission accepted: '+key)
    with tempfile.TemporaryDirectory(prefix='cached_target_publication_') as d:
        f=Path(d);target=f/'train';target.mkdir();m=target/'MANIFEST.json';atomic(m,{})
        def put(name,v):p=f/name;atomic(p,v);return reference(p)
        preserved=f/'preserved';preserved.mkdir()
        v={'schema':'nrkd_cached_actual_target_publication_v1','status':'COMPLETE','published_manifest':reference(m),
          'all_previous_writers_exited':True,'family_and_attempt_leases_free':True,'old_partial_preserved':True,
          'producer_exit':put('producer.json',{'actual_returncode':0}),
          'superseded_status':put('status.json',{'status':'EXITED','returncode':-15,'child_pid':1,'child_start_ticks':'2'}),
          'supersession_proof':put('proof.json',{'exact_pid_start_command_verified':True,'pidfd_exit_observed':True,
            'wrapper_exit_observed':True,'family_and_attempt_leases_free':True,'child_pid':1,'child_start_ticks':'2'}),
          'preserved_partial_directory':str(preserved),'candidate_acceptance':put('acceptance.json',{}),'superseded_spec':put('spec.json',{})}
        atomic(target/'ADOPTION.json',v);assert validate_publication(m,RESULTS)
        for key in ('all_previous_writers_exited','family_and_attempt_leases_free','old_partial_preserved'):
            bad=dict(v);bad[key]=False;atomic(target/'ADOPTION.json',bad)
            try:validate_publication(m,RESULTS)
            except ValueError:pass
            else:raise AssertionError('Live/unknown transition accepted')
        atomic(target/'ADOPTION.json',v);atomic(f/'status.json',{'status':'RUNNING','returncode':None})
        try:validate_publication(m,RESULTS)
        except ValueError:pass
        else:raise AssertionError('Changed/surviving CUDA writer accepted')
    # Parent-owned preparation uses an explicit four-device allocation while the
    # training resource map and all mathematical core source bytes stay intact.
    plan=read(RESULTS/'resources/ROUND1_PREPARATION_FIXED_ROOT_4GPU_20261027.json')
    assert plan['allocation']['host']=='training-vla-zt' and len(set(plan['allocation']['gpu_uuids']))==4
    source=(ROOT/'code/local_teacher_suite.py').read_text();tree=ast.parse(source)
    target=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='targets')
    assert "'PYTHONPATH='" in ast.unparse(target)
    # A fresh CPU process launched from an unrelated cwd must resolve core
    # before a GPU job has ever inserted its path (the actual recovery case).
    with tempfile.TemporaryDirectory(prefix='prep_import_recovery_') as d:
        program='''import importlib.util,sys\nfrom pathlib import Path\nfrom types import SimpleNamespace\nsys.path.insert(0,sys.argv[1])\nfrom round1_prepare_and_run import PreparationOwner\na=SimpleNamespace(results=Path(sys.argv[2]),root=Path(sys.argv[3]),code=Path(sys.argv[4]))\no=PreparationOwner(a)\nassert importlib.util.find_spec("student_distillation.asset_identity") is not None\n'''
        environment=dict(os.environ);environment.pop('PYTHONPATH',None)
        subprocess.run([sys.executable,'-B','-c',program,str(ROOT/'code'),d,str(ROOT),str(CODE)],
                       cwd=d,env=environment,check=True,capture_output=True,text=True)
    receipt={'status':'PASS','time':stamp(),'actual_all85813_candidate_admitted':True,
      'construction_negative_mutations':len(mutations),'publication_live_unknown_or_tamper_rejections':4,
      'preparation_world4_student_pins_preserved':True,'Val_target_explicit_module_import_environment':True,
      'fresh_unrelated_cwd_CPU_admission_import':True,
      'no_GPU_or_model_execution':True,'source_sha256':{p.name:sha(p) for p in (ROOT/'code/preparation_identity.py',ROOT/'code/cached_target_publication.py',ROOT/'code/round1_prepare_and_run.py')}}
    path=ROOT/'engineering/cached_actual_target_adoption_20261027'/('CPU_ADMISSION_CHECK.'+sha(ROOT/'code/round1_prepare_and_run.py')+'.json');atomic(path,receipt)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
