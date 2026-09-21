from common import *
import subprocess,collections
p=OUT/'native_vs_constructed_coverage.csv';f=pd.read_csv(p);f['status']='MEASURED'
for split in ['calibration','confirmation']:
 f=pd.concat([f,pd.DataFrame([dict(split=split,model='external',protocol='constructed',scope='external_constructed',tolerance_m=.5,scenes=64,status='NOT_RUN_TRIGGER_FALSE',generated_candidates=0)])],ignore_index=True)
f.to_csv(p,index=False)
p=OUT/'signal_coverage_plot_data.csv';f=pd.read_csv(p);f['status']='MEASURED';f=pd.concat([f,pd.DataFrame([dict(split='confirmation',model='external',protocol='constructed',scope='external_constructed',tolerance_m=.5,scenes=64,status='NOT_RUN_TRIGGER_FALSE',generated_candidates=0)])],ignore_index=True);f.to_csv(p,index=False)
# Association is descriptive and retains the counterexamples.
f=pd.read_csv(OUT/'ep_credit_degradation_association.csv');f['ep_component_positive_credit']=f.ep_only_components_mass>1e-8
from clustered_stats import summarise
summarise(f,['failure','PDMS','EP','raw_progress'],['split','protocol','ep_component_positive_credit'],'association_only').to_csv(OUT/'ep_credit_association_summary.csv',index=False)
# Exact statuses and costs; no invented step8 data or plots.
status=read(OUT/'audits/TINY_STEP1_COMPLETE.json');cost=[dict(stage='stage1_primary',scope='two checkpoints,512 scenes,two native groups16 and two deployment pools16',sampled_candidates=65536,nominal_candidate_scores=131072,perturbation_candidate_scores=0,optimizer_updates=0),dict(stage='stage1_smoke_extra',scope='repeat/capture draws plus repeated nominal and scoring-equivalence calls; PDM baseline excluded from candidate counts',sampled_candidates=1536,nominal_candidate_scores=6784,perturbation_candidate_scores=0,optimizer_updates=0),dict(stage='stage2_matched_pairs',scope='1192 distinct selected candidates; 8 selection +8 evaluation',sampled_candidates=0,nominal_candidate_scores=1192,perturbation_candidate_scores=19072,optimizer_updates=0),dict(stage='stage3_shared_step1',scope='two seeds shared by all four arms; costs are amortized, not charged four times',sampled_candidates=2048,nominal_candidate_scores=4096,perturbation_candidate_scores=16384,optimizer_updates=0)]
for r in status['arms']:cost.append(dict(stage='stage3_update_attempt',scope=f"seed={r['seed']},arm={r['arm']}; shared step1 cache reused; INVALID_STOPPED",sampled_candidates=0,nominal_candidate_scores=0,perturbation_candidate_scores=0,optimizer_updates=r['optimizer_updates']))
pd.DataFrame(cost).to_csv(OUT/'compute_cost.csv',index=False)
# Small read-only ad-hoc checks are separately declared rather than conflated
# with primary candidate counts.
save(OUT/'manifests/cost_notes.json',dict(primary_counts='candidate evaluations excluding PDM baseline; smoke counts include repeated standard/fast/single/batch tests',ad_hoc_debug_scores=6,ad_hoc_debug_scope='2 nominal margin debug candidates + 2 original and 2 zero-deformation candidates',unit_test_model_data='synthetic tensors only for unit assertions, never experimental evidence',updates_total=8,updates_valid_for_arm_comparison=0,step8='NOT_RUN',updated_policy_evaluations='NOT_RUN',norm_control='NOT_RUN'))
checks=[]
for item in read(OUT/'manifests/input_files.json'):
 checks.append(dict(path=item['path'],expected_sha256=item['sha256'],actual_sha256=sha(item['path'])));assert checks[-1]['expected_sha256']==checks[-1]['actual_sha256']
save(OUT/'audits/frozen_input_files_unchanged.json',dict(status='PASS',files=checks))
# Source and cache indices support later replay without resampling.
save(OUT/'manifests/analysis_sources.json',[dict(path=str(p),sha256=sha(p)) for p in sorted((WORK/'tools/analysis/grpo_credit_risk_quickcheck').glob('*.py'))])
items=[]
for model in cfg()['models']:
 for row in scenes():
  path=OUT/'cache/rollouts'/model/(row['token']+'.npz');items.append(dict(model=model,token=row['token'],path=str(path),sha256=sha(path)))
pd.DataFrame(items).to_parquet(OUT/'manifests/trajectory_cache_index.parquet',index=False)
save(OUT/'manifests/overall_status.json',dict(stage1='PASS_READONLY',stage2='WITHIN_GROUP_SIGNAL',stage3='INVALID_STOPPED_AFTER_ONE_UPDATE_PER_ARM_SEED',stage3_optimizer_updates=8,stage3_valid_comparisons=0,step8='NOT_RUN',scenes_changed=False,amplitudes_changed=False,automatic_retries=False,canonical_checkpoints_modified=False,report=str(WORK/'reports/GRPO_CREDIT_RISK_QUICKCHECK.md')))
print('EVIDENCE FINALIZED')
