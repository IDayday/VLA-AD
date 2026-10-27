"""Small CPU-only regression for the explicit user startup override."""
import copy,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
CODE=Path('/mnt/project/recogdrive-no-routing-round1-20261026')
RESULTS=Path('/mnt/project/recogdrive-no-routing-round1-results-20261026')
sys.path[:0]=[str(ROOT/'code'),str(CODE)]
from types import SimpleNamespace
from gt_numeric import atomic,read,sha,stamp
from round1_prepare_and_run import PreparationOwner
from student_distillation.train import verify_qualification,verify_formal_config
from student_distillation.experiment_registry import load_config

def main():
    with tempfile.TemporaryDirectory(prefix='basic_start_cpu_') as d:
        values=read(RESULTS/'protocol/PATHS.json');values['qualification_receipt']=str(Path(d)/'ADMISSION.json')
        owner=PreparationOwner.__new__(PreparationOwner);owner.args=SimpleNamespace(root=ROOT,results=RESULTS,code=CODE)
        owner.training_world_size=4;owner.update=lambda *a,**k:None
        owner.basic_normal_start(values)
        for experiment in ('H_full','H_gain1','H_mu_teacher','H_mu_student'):
            cfg=load_config(CODE/'configs/student_distillation'/f'{experiment}.yaml',CODE/'configs/student_distillation/common.yaml')
            cfg['paths']=values;cfg['runtime']={'world_size':4};cfg['training']['gradient_accumulation_steps']=16
            cfg['startup_admission_mode']='USER_AUTHORIZED_BASIC_NORMAL_START'
            verify_formal_config(cfg);v=verify_qualification(cfg)
            assert v['formal_asset_qualification'] is False and v['runtime_extra_recovery_probes']=='CANCELLED_BY_USER'
            wrong=copy.deepcopy(cfg);wrong.pop('startup_admission_mode')
            try:verify_qualification(wrong)
            except ValueError:pass
            else:raise AssertionError('Implicit cancellation accepted')
        receipt=read(values['qualification_receipt']);receipt['identities']['calibration_sha256']='0'*64
        atomic(values['qualification_receipt'],receipt)
        try:verify_qualification(cfg)
        except ValueError:pass
        else:raise AssertionError('Stale assets accepted')
    proof={'status':'PASS_BASIC_START_CPU','time':stamp(),'all_four_configs_explicit_user_mode_accepted':True,
      'implicit_mode_and_stale_assets_rejected':True,'no_CODE_CHECKS_PASS_claim':True,'GPU_work':0,
      'extra_five_qualification_families':'CANCELLED_BY_USER','formal_budget_unchanged':True}
    atomic(ROOT/'engineering/basic_normal_start_20261027/CPU_CHECK.json',proof);print(proof)

if __name__=='__main__':main()
