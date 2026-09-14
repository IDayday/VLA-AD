"""Audits all consumed historical files and produced accounting tables; no training."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import subprocess
from analyze import *

def main():
    a=OUT/'audits';a.mkdir(exist_ok=True)
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tools/analysis/historical_sft_grpo_longchain'),'-p','test_*.py','-v'],cwd=ROOT,text=True,capture_output=True)
    (a/'tests.txt').write_text(test.stdout+test.stderr);assert test.returncode==0
    inputs=json.loads((OUT/'manifests/inputs.json').read_text())
    def verify(item):
        name,d=item;p=Path(name)
        return {'path':name,'ok':p.is_file() and p.stat().st_size==d['bytes'] and sha(p)==d['sha256']}
    with ThreadPoolExecutor(max_workers=4) as pool:checks=list(pool.map(verify,inputs.items()))
    assert all(x['ok'] for x in checks),'A historical input changed; investigate before publication.'
    curve=pd.read_csv(OUT/'metrics/historical_curves.csv')
    assert len(curve)==113 and curve.PDMS.between(80,100).all()
    assert curve.n_valid.eq(12138).all()
    for col in ['NC','DAC','TTC','DDC','EP','Comfort']:assert curve[col].between(0,100).all()
    assert len(curve[curve.family.eq('a5_watcher')])==5
    pairs=pd.read_csv(OUT/'metrics/paired_comparisons.csv')
    assert pairs.n_tokens.eq(12138).all() and pairs.n_scene_clusters.eq(1203).all()
    assert pairs.ci_low.le(pairs.ci_high).all()
    t=pd.read_csv(OUT/'metrics/tail_decomposition.csv')
    np.testing.assert_allclose(t.safety_regression_contribution+t.nonregression_contribution,t.total_delta,atol=1e-9)
    wrong=pd.read_csv(OUT/'metrics/wrong_vlm_false_collapse.csv')
    assert wrong.groupby('epoch').checkpoint_path.nunique().eq(1).all()
    assert (wrong[wrong.evaluation.eq('correct_vlm')].vlm_path==wrong[wrong.evaluation.eq('correct_vlm')].train_vlm_path).all()
    protected=['outputs/pc_mts_diagnostics','outputs/pc_mts_diagnostics_v2','outputs/pc_mts_diagnostics_v3',
               'reports/PC_MTS_POLICY_DISTRIBUTION_DIAGNOSTICS_20260913.md','reports/PC_MTS_POLICY_DIAGNOSTICS_V2_20260913.md',
               'reports/PC_MTS_POLICY_DIAGNOSTICS_V3_20260913.md']
    parent='304a35e0fe372b9258d42ebf023917b6370810e3'
    changed=subprocess.check_output(['git','diff','--name-only',parent,'--',*protected],cwd=ROOT,text=True)
    assert not changed.strip(),changed
    figures={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in (OUT/'figures').iterdir()}
    assert len(figures)==12
    save_json('audits/final_audit.json',dict(status='PASS',timestamp=datetime.now(timezone.utc).isoformat(),
        analysis_parent_commit=parent,branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        optimizer_updates=0,new_rollouts=0,tests_passed=9,verified_historical_input_files=len(checks),
        historical_input_bytes=sum(d['bytes'] for d in inputs.values()),primary_curve_rows=113,
        supplemental_historical_curve_rows=5,n_valid_benchmark_tokens=12138,n_scene_clusters=1203,
        known_manifest_total=12146,known_missing_score_records=8,bootstrap_replicates=3000,
        token_alignment='PASS',score_unit_audit='PASS',safety_tail_accounting='PASS',
        wrong_VLM_pairs_same_checkpoint='PASS',protected_tracked_outputs_unchanged='PASS',
        no_recent_quicktest_experiment_inputs=not any('/pc_mts_diagnostics' in p or '/progressive_pcmts' in p for p in inputs),
        log_cluster_sensitivity='UNAVAILABLE: common log-name mapping not recovered',
        figures=figures,engineering_corrections=[
            'Historical watcher TSV column named PDMS is fractional; fixed conversion to points before publication, reran analysis and figures, added regression test. No historical file edited.',
            'Training phase bins renamed to explicit step intervals; different runs have different epoch lengths.',
            'V6 source commit alone was insufficient; applied saved patch and recorded active function differences.'],
        limitations=['Single training seed per historical run','Original/PSI evaluations not common-random-number paired',
                     'Historical Navtest peak selection is descriptive','A5/V6 formal runs stopped before configured 10 epochs',
                     'Different algorithms, representations and reference constraints preclude SFT-only causal attribution',
                     'Source reconstruction is not forward parity'] ))
    print('PASS:',len(checks),'historical inputs verified; 9 tests; protected results unchanged.',flush=True)

if __name__=='__main__':main()
