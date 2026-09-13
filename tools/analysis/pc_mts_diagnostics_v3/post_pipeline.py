"""Finish the predeclared additive control and all audits; leave publication for final review."""
from common_v3 import *
from pipeline import python,gpu
def main():
    while not (OUT/'manifests/pipeline_complete.json').exists():time.sleep(5)
    if not (OUT/'manifests/audit_G_null.json').exists():
        gpu('G_null','null_support.py');python('G_null_statistics','analyze_null.py')
    python('additional_final','additional_audits.py')
    python('initialization_tensors','initialization_audit.py')
    python('supplementary','supplementary.py')
    python('figures_final','figures.py')
    python('tests_final','tests.py')
    python('immutability_final','prepare.py','--verify')
    python('full_final_audit','final_audit.py')
    python('report_final','report.py')
    save(OUT/'manifests/READY_FOR_REVIEW.json',dict(identity=identity(),all_experiments_and_audits_complete=True,publication_pending=True))
if __name__=='__main__':main()
