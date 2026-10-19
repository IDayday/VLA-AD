"""Complete real diagnostics, independent certificates and reports; never publish."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import shutil
from scripts.opd.common import *
from scripts.opd.suite import run_queue,EXPS

def wait_until(predicate):
    while not predicate():time.sleep(15)

def call(*args):subprocess.run([PYTHON,*args],cwd=SOURCE,check=True)

def main():
    wait_until(lambda:all((ROOT/'runs'/f'train_{e}_{s}/complete.json').exists() for e in EXPS for s in config()['train_seeds']))
    tasks=[('diagnostic_gradients',[['scripts/opd/diagnostics.py','gradients']])]
    for exp in ['E3','E5']:
        for seed in config()['train_seeds']:
            tasks.append((f'gate_diagnostic_{exp}_{seed}',[['scripts/opd/evaluate.py','--exp',exp,'--seed',str(seed),'--split','train_gate_diagnostic']]))
    run_queue(tasks)
    call('scripts/opd/diagnostics.py','chains');call('scripts/opd/gate_diagnostics.py','analyze')
    chosen=rows('dev')[:8]
    wait_until(lambda:all((ROOT/'evaluation/dev'/f'train_{e}_42/scores'/f"{r['token']}.json").exists() for e in ['E3','E5'] for r in chosen))
    run_queue([('qualified_student_deployment',[['scripts/opd/inputs.py','deployment'],['scripts/opd/inputs.py','benchmark']])])
    call('scripts/opd/certify.py','models')
    wait_until(lambda:(ROOT/'suite/formal_complete.json').exists())
    if not (ROOT/'analysis/baseline_dev.json').exists():call('scripts/opd/analyze.py','--phase','baseline','--split','dev')
    if not (ROOT/'analysis/formal_dev.json').exists():call('scripts/opd/analyze.py','--split','dev')
    call('scripts/opd/additional_statistics.py','--split','dev')
    marker=ROOT/'analysis/review_directories_reset.json'
    if not marker.exists():
        archive=ROOT.parent/'legacy_review_files_before_v2';archive.mkdir(exist_ok=True)
        for name in ['figures','tables']:
            src=REPORTS/name;dest=archive/name
            if src.exists():
                if dest.exists():raise FileExistsError('Legacy review archive already exists')
                shutil.move(str(src),str(dest))
            src.mkdir(exist_ok=True)
        write(marker,dict(preserved_previous_files=str(archive),time=time.time(),purpose='Prevent stale INVALID v1 tables/figures from appearing as corrected outputs'))
    call('scripts/opd/export.py');call('scripts/opd/figures.py','--split','dev','--cases')
    call('scripts/opd/certify.py','evaluation','--split','dev')
    wait_until(lambda:(ROOT/'suite/navtest_complete.json').exists())
    call('scripts/opd/analyze.py','--phase','baseline','--split','navtest')
    call('scripts/opd/analyze.py','--split','navtest');call('scripts/opd/additional_statistics.py','--split','navtest')
    call('scripts/opd/certify.py','evaluation','--split','navtest');call('scripts/opd/figures.py','--split','navtest')
    wait_until(lambda:(ROOT/'selection/E6_admission.json').exists())
    if read(ROOT/'selection/E6_admission.json')['admitted']:
        wait_until(lambda:(ROOT/'suite/E6_complete.json').exists())
    call('scripts/opd/final_audit.py','cache_versions')
    with open(ROOT/'tests/final_cpu.stdout','w') as output:
        subprocess.run([PYTHON,'-m','pytest','tests/opd','-q','--junitxml='+str(ROOT/'tests/cpu.xml')],cwd=SOURCE,stdout=output,stderr=subprocess.STDOUT,check=True)
    call('scripts/opd/final_audit.py','tests');call('scripts/opd/final_audit.py','accounting')
    call('scripts/opd/export.py');call('scripts/opd/report.py');call('scripts/opd/export.py')
    write(ROOT/'suite/reports_complete.json',dict(status='COMPLETE',time=time.time(),owner=identity(),
        scope='Real complete results and certificates generated; final human-readable review, git diff and authorized push remain separate'))

if __name__=='__main__':main()
