"""Adopt four live seed27 runs and continue their full Val/Test/report chain."""
import fcntl,json,os,subprocess,sys
from pathlib import Path
from single_seed_scope import install,ROOT,CODE
RESULTS=Path('/mnt/project/recogdrive-no-routing-round1-results-20261026')
sys.path.insert(0,str(ROOT/'code'))
from gt_numeric import atomic,read,sha,stamp

def main():
    lock=(RESULTS/'preparation_owner/OWNER.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    spec=read(ROOT/'engineering/single_seed_resource_saving_20261027/OWNER_SPEC.json')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()!='0a58bd8e6271e26103512add2c5cb7816d7eb8f1':
        raise ValueError('Retained training HEAD changed')
    source_map=read(RESULTS/'protocol/CODE_FREEZE.json')['code_files']
    if source_map!={str(p.relative_to(CODE)):sha(p) for p in sorted((CODE/'student_distillation').glob('*.py'))}:
        raise ValueError('Frozen training code changed')
    registry=read(RESULTS/'preparation_owner/REGISTRY.json');registry.update(status='EXECUTING_SINGLE_SEED_FOUR_FULL_RUNS',
      pid=os.getpid(),process_start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19],
      updated_at=stamp(),single_seed_scope=spec['user_instruction'],extra_qualification='CANCELLED_BY_USER')
    atomic(RESULTS/'preparation_owner/REGISTRY.json',registry)
    suite=install();sys.argv=[str(Path(__file__).resolve()),*spec['arguments']];suite.main()
    actual=read(RESULTS/'EXPERIMENT_REGISTRY.json')
    if actual.get('status')!='REPORT_COMPLETE':raise RuntimeError('Retained full suite returned without final report')
    subprocess.run([sys.executable,'-B',str(ROOT/'engineering/permanent_fullflow/finalize_round1_physical_costs.py'),
      '--root',str(ROOT),'--prior-root','/mnt/project/VLA-AD/experiments/recogdrive_dc8_refit_20261026',
      '--results',str(RESULTS),'--round1-report',str(RESULTS/'ROUND1_REPORT.md')],cwd=CODE,check=True)
    registry.update(status='REPORT_COMPLETE',updated_at=stamp(),final_report={'path':str(RESULTS/'ROUND1_REPORT.md'),'sha256':sha(RESULTS/'ROUND1_REPORT.md')})
    atomic(RESULTS/'preparation_owner/REGISTRY.json',registry);atomic(RESULTS/'preparation_owner/COMPLETE.json',registry)

if __name__=='__main__':main()
