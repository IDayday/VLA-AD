"""Restart two known exited CPU owners under the revised frozen transport."""
import fcntl,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RESULTS=Path('/mnt/project/recogdrive-no-routing-round1-results-20261026')
HERE=ROOT/'engineering/permanent_fullflow'
sys.path[:0]=[str(HERE),str(ROOT/'code')]
from gt_numeric import atomic,read,sha,stamp
from permanent_dispatch import PYTHON,grant,install
from round1_prepare_and_run import PreparationOwner
from cached_target_publication import validate_publication

def main():
    grant();validate_publication(RESULTS/'targets/train/MANIFEST.json',RESULTS)
    for pid in (643796,582673):
        if Path(f'/proc/{pid}').exists():raise RuntimeError('Previous known CPU owner is not fully gone')
    main_exit=ROOT/'dispatch/full_teacher_native_candidates_owner_parallel_test_20261027/status/full_teacher_native_candidates_owner.json'
    prep_exit=ROOT/'dispatch/round1_preparation_native_candidates_owner_edquot_recovery_20261027/status/round1_preparation_native_candidates_owner.json'
    if read(main_exit).get('returncode')!=1 or read(prep_exit).get('returncode')!=-15:
        raise RuntimeError('Previous CPU owner actual exits differ')
    resources=RESULTS/'resources/ROUND1_FIXED_32GPU_WITH_TSINGHUA_ALLOCATION.json'
    # A minimal owner object runs the exact production source/commit gate.
    from types import SimpleNamespace
    owner=PreparationOwner.__new__(PreparationOwner)
    owner.args=SimpleNamespace(code=Path('/mnt/project/recogdrive-no-routing-round1-20261026'),
        code_freeze=RESULTS/'protocol/CODE_FREEZE.json')
    owner.code_gate()
    requests=[('round1_preparation_native_candidates_owner',ROOT/'code/round1_prepare_and_run.py',
      RESULTS/'preparation_owner/OWNER.lock',['--resources',str(resources),'--preparation-resources',
        str(RESULTS/'resources/ROUND1_PREPARATION_FIXED_ROOT_4GPU_20261027.json')]),
      ('full_teacher_native_candidates_owner',HERE/'fullflow_teacher_owner.py',ROOT/'teacher_suite/OWNER.lock',
       ['--allocation',str(HERE/'PERMANENT_TEACHER_40GPU_ALLOCATION.json'),'--test-allocation',
        str(ROOT/'engineering/parallel_test_20261027/TEST_ALLOCATION.json'),'--engineering-retries','3'])]
    receipt=ROOT/'dispatch/CACHED_TARGET_ADOPTION_OWNERS_RESTARTED.json'
    if receipt.exists():raise RuntimeError('Restart was already registered')
    install();import dispatch
    rows=[]
    for name,source,lock,extra in requests:
        with lock.open('a+') as stream:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        directory=ROOT/'dispatch'/(name+'_cached_actual_targets_20261027')
        if (directory/'launcher.pid').exists():raise RuntimeError('Replacement CPU owner was already launched')
        command=[PYTHON,'-B',str(source),'--root',str(ROOT),'--poll-seconds','30',*extra]
        launch=dispatch.launch('training-vla-zt',directory,[(name,'-',command)])
        rows.append({'name':name,'command':command,'launch':launch,'source_sha256':sha(source),
          'actual_status':str(directory/'status'/(name+'.json'))})
        atomic(receipt,{'status':'DURABLE_OWNERS_RESTARTED_NOT_FORMAL_TRAINING_COMPLETION','time':stamp(),
          'owners':rows,'freeze_sha256':sha(RESULTS/'protocol/CODE_FREEZE.json'),
          'all_old_Target_and_CPU_exit_receipts_preserved':True})
    print(json.dumps(read(receipt),indent=2))

def preparation_import_recovery():
    grant()
    old=ROOT/'dispatch/round1_preparation_native_candidates_owner_cached_actual_targets_20261027/status/round1_preparation_native_candidates_owner.json'
    if read(old).get('pid')!=697344 or read(old).get('returncode')!=1 or Path('/proc/697344').exists():
        raise RuntimeError('Previous prep invocation has not actually exited as recorded')
    lock=RESULTS/'preparation_owner/OWNER.lock'
    with lock.open('a+') as stream:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from types import SimpleNamespace
    owner=PreparationOwner.__new__(PreparationOwner);owner.args=SimpleNamespace(
      code=Path('/mnt/project/recogdrive-no-routing-round1-20261026'),code_freeze=RESULTS/'protocol/CODE_FREEZE.json')
    owner.code_gate();validate_publication(RESULTS/'targets/train/MANIFEST.json',RESULTS)
    name='round1_preparation_native_candidates_owner'
    directory=ROOT/'dispatch'/(name+'_import_recovery_20261027')
    if (directory/'launcher.pid').exists():raise RuntimeError('Replacement already registered')
    source=ROOT/'code/round1_prepare_and_run.py'
    command=[PYTHON,'-B',str(source),'--root',str(ROOT),'--poll-seconds','30','--resources',
      str(RESULTS/'resources/ROUND1_FIXED_32GPU_C_ROOT_ALLOCATION_20261027.json'),'--preparation-resources',
      str(RESULTS/'resources/ROUND1_PREPARATION_FIXED_ROOT_4GPU_20261027.json')]
    install();import dispatch
    launch=dispatch.launch('training-vla-zt',directory,[(name,'-',command)])
    value={'status':'CPU_IMPORT_RECOVERY_LAUNCHED_NOT_FORMAL_TRAINING_COMPLETION','time':stamp(),
      'old_actual_exit':{'path':str(old),'sha256':sha(old)},'command':command,'launch':launch,
      'actual_status':str(directory/'status'/(name+'.json')),'source_sha256':sha(source),
      'freeze_sha256':sha(RESULTS/'protocol/CODE_FREEZE.json'),'healthy_main_and_all_GPU_jobs_preserved':True}
    atomic(ROOT/'dispatch/CACHED_TARGET_PREP_IMPORT_RECOVERY.json',value);print(json.dumps(value,indent=2))

if __name__=='__main__':
    preparation_import_recovery() if '--preparation-import-recovery' in sys.argv else main()
