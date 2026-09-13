"""Final tests, artifact checks and report; publication remains ordinary git."""
from common_mass import *
import subprocess

def main():
    assert sha(CONFIG)==read(OUT/'manifests/protocol_frozen.json')['sha256']
    assert read(OUT/'audits/completion.json')['status']=='PASS'
    assert read(OUT/'audits/independent_real_data_review.json')['status']=='PASS'
    command=[sys.executable,'-m','pytest','-q',str(Path(__file__).parent/'tests')]
    result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
    log=OUT/'audits/final_tests.txt';log.write_text(result.stdout+result.stderr)
    save(OUT/'audits/final_tests.json',dict(command=command,returncode=result.returncode,completed_at=utc(),output=result.stdout+result.stderr))
    print(result.stdout,flush=True);assert result.returncode==0
    required=['raw_source_summary.csv','candidate_metrics.parquet','selected_pools_4x1000x16.parquet','pool_scene_metrics.csv','method_summary.csv','paired_comparisons.csv','source_quality_matched.csv','no_il_native_control.csv','sensitivity.csv','global_policy_sampling_baseline.csv','missing_and_errors.csv']
    for name in required:assert (OUT/'metrics'/name).is_file(),name
    figures={}
    for i in range(1,7):
        png=list((OUT/'figures').glob(f'Fig{i}_*.png'));assert len(png)==1
        for ext in ['png','svg','pdf']:
            p=png[0].with_suffix('.'+ext);assert p.is_file();figures[str(p.relative_to(ROOT))]=sha(p)
        assert png[0].with_name(png[0].stem+'_data.csv').is_file()
    workers=[]
    for p in sorted((OUT/'audits').glob('rollouts*json')):
        a=read(p);assert a['state_before']==a['state_after']
        workers.append(dict(path=str(p.relative_to(ROOT)),scenes=a['scenes'],state_unchanged=True))
    codehash={str(p.relative_to(ROOT)):sha(p) for p in sorted(Path(__file__).parent.rglob('*.py'))}
    save(OUT/'audits/final_publication_review.json',dict(status='PASS',completed_at=utc(),tests_passed=True,scientific_config_unchanged=True,code_hashes=codehash,figure_hashes=figures,worker_state_checks=workers,preempted_worker_limit='Original rank6/rank7 were stopped only for resource redistribution and do not have final process-level hash audits. Their completed atomic banks were retained. Native frozen parameters, no optimizer/active BN/dropout, eight-scene parity and all 14 completed worker state checks support unchanged policy.',new_weight_updates=0))
    run=read(OUT/'manifests/run_summary.json');run.update(status='COMPLETE',review_completed_at=utc(),test_audit_hash=sha(OUT/'audits/final_tests.json'),publication_review_hash=sha(OUT/'audits/final_publication_review.json'));save(OUT/'manifests/run_summary.json',run)
    subprocess.run([sys.executable,str(Path(__file__).parent/'write_report.py')],cwd=ROOT,check=True)
    save(OUT/'manifests/pipeline_finished.json',dict(status='COMPLETE_PENDING_GIT_PUBLICATION',completed_at=utc(),report_sha256=sha(ROOT/'reports/PC_MTS_GRPO_MASS_AUDIT_1000_20260913.md')))

if __name__=='__main__':main()
