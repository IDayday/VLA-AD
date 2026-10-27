"""Actual full-budget suite activation under the user's latest cancellation."""
import fcntl,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
RESULTS=Path('/mnt/project/recogdrive-no-routing-round1-results-20261026')
CODE=Path('/mnt/project/recogdrive-no-routing-round1-20261026')
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'engineering/permanent_fullflow'),str(CODE)]
from gt_numeric import read,atomic,sha,stamp
from permanent_dispatch import install,grant,PYTHON
from round1_prepare_and_run import PreparationOwner

def main():
    grant()
    if Path('/proc/698578').exists():raise RuntimeError('Replaced CPU owner remains alive')
    cal=RESULTS/'preparation_owner/remote_execution/jobs/train_calibration/attempt_00000/STATUS.json'
    if read(cal).get('returncode')!=0:raise RuntimeError('Immutable Train calibration has not actually completed')
    with (RESULTS/'preparation_owner/OWNER.lock').open('a+') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    owner=PreparationOwner.__new__(PreparationOwner);owner.args=SimpleNamespace(code=CODE,code_freeze=RESULTS/'protocol/CODE_FREEZE.json');owner.code_gate()
    name='round1_preparation_native_candidates_owner';directory=ROOT/'dispatch'/(name+'_basic_normal_start_20261027')
    if (directory/'launcher.pid').exists():raise RuntimeError('Basic activation already registered')
    command=[PYTHON,'-B',str(ROOT/'code/round1_prepare_and_run.py'),'--root',str(ROOT),'--poll-seconds','30',
      '--resources',str(RESULTS/'resources/ROUND1_FIXED_32GPU_C_ROOT_ALLOCATION_20261027.json'),
      '--preparation-resources',str(RESULTS/'resources/ROUND1_PREPARATION_FIXED_ROOT_4GPU_20261027.json'),
      '--user-authorized-basic-normal-start']
    install();import dispatch
    launch=dispatch.launch('training-vla-zt',directory,[(name,'-',command)])
    value={'status':'USER_AUTHORIZED_BASIC_NORMAL_START_ACTIVATED_NOT_YET_TRAINING_COMPLETION','time':stamp(),
      'command':command,'launch':launch,'actual_status':str(directory/'status'/(name+'.json')),
      'freeze_sha256':sha(RESULTS/'protocol/CODE_FREEZE.json'),'five_extra_qualification_families':'CANCELLED_BY_USER',
      'initial_periodic_recovery_probes':'CANCELLED_BY_USER','intrinsic_actual_forward_backward_finite_clip_checkpoints_retained':True}
    atomic(ROOT/'engineering/basic_normal_start_20261027/LAUNCH.json',value);print(value)

if __name__=='__main__':main()
