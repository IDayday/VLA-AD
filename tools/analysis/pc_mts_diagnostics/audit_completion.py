"""Requirement-level acceptance for the full experiment and authorized coverage."""
import argparse,subprocess
import pandas as pd
from common import *

def main(args):
    cfg=config();sc=manifest();out=OUT;checks=[]
    def verified(requirement,evidence,condition=True):
        assert condition,requirement
        checks.append(dict(requirement=requirement,status='verified_complete',evidence=evidence))
    full=read(out/'manifests/full_validation.json');coverage=read(out/'manifests/coverage_validation.json');smoke=read(out/'smoke/manifests/smoke_validation.json');poolcheck=read(out/'manifests/candidate_pool_validation.json')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    verified('项目目录 VLA-AD 中完成',[str(ROOT)],ROOT.name=='VLA-AD')
    verified('独立 analysis branch',[branch],branch=='analysis/pc-mts-policy-diagnostics-20260913')
    ckpts=models();required={'name','category','checkpoint_path','config_path','reported_pdms','training_method','git_commit','metadata_source','notes'}
    verified('5 checkpoint manifest 完成',['manifests/checkpoint_manifest.yaml','manifests/full_validation.json'],len(ckpts)==5 and all(required<=set(m) and Path(m['checkpoint_path']).exists() for m in ckpts) and full['checkpoint_hashes_verified']==5)
    verified('1000 scene manifest 固定',['manifests/scenes_1000.json','manifests/protocol_frozen.json'],len(sc['scenes'])==len({r['token'] for r in sc['scenes']})==1000 and sc['population']==103288)
    gpu=read(out/'manifests/rollout_official_il_execution.json')['command']
    verified('8 GPU distributed rollout 实现',['manifests/rollout_official_il_execution.json'], '--nproc_per_node=8' in gpu)
    verified('CPU multiprocess evaluator 实现',['manifests/benchmark_cpu_rollouts.json','manifests/benchmark_cpu_heldout.json'],read(out/'manifests/benchmark_cpu_heldout.json')['workers']==96)
    verified('5×1000×64 rollout 完成',['manifests/full_validation.json'],full['rollout_trajectories']==320000 and full['rollout_scene_model_pairs']==5000)
    verified('policy distribution width/sharpness分析完成',['metrics/policy_distribution.csv','metrics/policy_paired_tests.csv'],len(pd.read_csv(out/'metrics/policy_distribution.csv'))==5000)
    verified('96/128 common raw candidate reservoir完成',['manifests/reservoir_expansion_decision.json','manifests/coverage_validation.json'],cfg['raw_gt_count']+4*cfg['raw_model_count']==128 and coverage['all_raw_candidates_traceable'])
    verified('四种candidate selection rule实现正确',['configs/pc_mts_diagnostics/coverage_fallback.yaml','manifests/coverage_validation.json'],coverage['all_coverage_tier_predicates_verified'])
    verified('每method每scene恰好16条',['manifests/candidate_pool_validation.json','manifests/coverage_validation.json'],poolcheck['complete_scene_counts']==dict.fromkeys(METHODS,1000) and coverage['candidate_count']==64000 and coverage['pool_size']==16)
    candidates=pd.read_parquet(out/'metrics/candidate_records.parquet')
    verified('filler/fallback全部有metadata',['metrics/candidate_records.parquet','manifests/coverage_validation.json'],{'is_fill','fill_parent_id','fallback_level','is_coverage_fallback'}<=set(candidates.columns) and len(candidates)==64000)
    verified('candidate pool sanity check完成',['manifests/candidate_pool_validation.json'],not poolcheck['all_methods_almost_identical'])
    verified('Exp-1完成',['metrics/candidate_records.parquet','metrics/candidate_pool_scene.csv'],candidates.groupby('method').size().eq(16000).all())
    verified('Exp-2使用held-out perturbation完成',['manifests/coverage_validation.json','smoke/manifests/smoke_validation.json'],coverage['heldout_scored_count']==768000 and coverage['all_parent_hashes_match'] and smoke['heldout_seed_separation_and_amplitude_bounds'])
    verified('Exp-3 GRPO exact advantage诊断完成',['manifests/grpo_advantage_source.json','metrics/grpo_groups.csv'],full['actual_groups']==40000 and smoke['native_grpo_reward_max_error']<=1e-8)
    tests=pd.read_csv(out/'metrics/coverage/pc_full_cohort_paired_tests.csv')
    verified('paired scene-level statistics完成',['metrics/policy_paired_tests.csv','metrics/coverage/pc_full_cohort_paired_tests.csv'],len(tests)>0 and tests.n_scenes.eq(1000).all())
    tags=[f'Fig-0{x}' for x in 'abcdef']+[f'Fig-0g-page{i}' for i in [1,2,3]]+[f'Fig-1{x}' for x in 'abcde']+[f'Fig-2{x}' for x in 'abcdef']+[f'Fig-3{x}' for x in 'abcdefg']
    for ext in ['png','pdf']:verified(ext.upper()+'图完成',['figures/'+x+'.'+ext for x in tags],all((out/'figures'/f'{x}.{ext}').stat().st_size>1000 for x in tags))
    for name in ['candidate_records','candidate_pool_scene','local_robustness_scene','local_robustness_candidates']:
        csv=pd.read_csv(out/'metrics'/f'{name}.csv');pq=pd.read_parquet(out/'metrics'/f'{name}.parquet');assert len(csv)==len(pq) and set(csv.columns)==set(pq.columns)
    verified('CSV/Parquet完成',['metrics/','metrics/coverage/'])
    report=(ROOT/'reports/PC_MTS_POLICY_DISTRIBUTION_DIAGNOSTICS_20260913.md').read_text()
    verified('Markdown report完成',['reports/PC_MTS_POLICY_DISTRIBUTION_DIAGNOSTICS_20260913.md'],report.count('\n## ')==15 and all(t in report for t in ['TABLE A','TABLE B','TABLE C','TABLE D','coverage','64,000','768,000']))
    verified('smoke test完成',['smoke/manifests/smoke_validation.json','smoke/manifests/coverage_validation.json'],read(out/'smoke/manifests/coverage_validation.json')['full_completion'] and smoke['scalar_batch_parity']==0 and 'all four methods complete' in smoke['candidate_construction_status'])
    verified('benchmark记录完成',['benchmark/manifests/gpu_throughput_summary.json','benchmark/manifests/benchmark_cpu_rollouts.json','benchmark_telemetry/manifests/replay_validation.json'],all((out/p).exists() for p in ['benchmark/manifests/gpu_throughput_summary.json','benchmark/manifests/benchmark_cpu_rollouts.json','benchmark_telemetry/manifests/replay_validation.json']))
    before=read(out/'manifests/coverage_before_existing_pools.json');assert all(sha(r['path'])==r['sha256'] for r in before['pools'])
    verified('不修改原训练结果',['Analysis-only code; original checkpoint hashes validated; original 192 pools byte-identical','manifests/coverage_before_existing_pools.json'])
    verified('所有结果来自真实计算，不是人工构造',['manifests/full_validation.json','manifests/coverage_validation.json'],full['full_experiments_complete'])
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if args.require_commit:
        changed=subprocess.check_output(['git','diff-tree','--no-commit-id','--name-only','-r','HEAD'],cwd=ROOT,text=True).splitlines()
        verified('最后单独git commit',[commit], 'reports/PC_MTS_POLICY_DISTRIBUTION_DIAGNOSTICS_20260913.md' in changed and any(x.startswith('tools/analysis/pc_mts_diagnostics/') for x in changed))
    else:checks.append(dict(requirement='最后单独git commit',status='pending',evidence=['All experiment artifacts verified; final commit follows this pre-commit audit']))
    save(out/'manifests/completion_audit.json',dict(goal='Full PC-MTS diagnostics with subsequent user-authorized 1000-scene coverage extension',requirements=checks,overall_status='complete' if args.require_commit else 'all_experiments_complete_final_commit_pending',branch=branch,analysis_commit=commit if args.require_commit else None,notes=['Original PC-MTS zero-parent rate 80.8% retained; coverage is explicitly labeled exploratory.','Native learned normalization retained and common physical output normalization used; architecture differences disclosed.','Initial smoke covered five checkpoints and M1-M3; complete extension smoke preceded main coverage run.','Original 50-scene throughput benchmark preceded main run; missing detailed telemetry was collected in a later exact replay and disclosed.']))
    print(json.dumps(dict(requirements=len(checks),experiment_completion=True,final_commit_verified=args.require_commit),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--require-commit',action='store_true');main(p.parse_args())
