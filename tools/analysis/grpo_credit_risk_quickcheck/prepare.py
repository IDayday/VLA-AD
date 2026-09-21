from common import *
import subprocess,collections,ast
assert not (OUT/'resolved_config.yaml').exists(),'Frozen configuration already exists'
models0=read(ROOT/'outputs/historical_sft_grpo_support/manifests/models.json')
models={n:models0[n] for n in ['a5_sft','a5_grpo_300']}
for n,m in models.items():
 assert Path(m['checkpoint_path']).is_file(),m['checkpoint_path']
 assert sha(m['checkpoint_path'])==m['sha256']
 assert sha(m['runtime_planner_path'])==m['runtime_planner_sha256']
run=yaml.safe_load(Path(models['a5_sft']['grpo_config_path']).read_text());agent=run['agent']
assert agent['checkpoint_path']==models['a5_sft']['checkpoint_path']==agent['reference_policy_checkpoint']
assert agent['stage3_algorithm']=='lfp_grpo' and agent['grpo_sample_time']==16
assert run['trainer']['params']['precision']=='16-mixed'
source=read(ROOT/'outputs/historical_sft_grpo_support/manifests/scenes.json')
rows=source['scenes'];lognames=sorted({r['log'] for r in rows},key=lambda x:digest(['quickcheck_log_split_v1',x]))
logsplit={x:'calibration' if i%2==0 else 'confirmation' for i,x in enumerate(lognames)}
quotas={'straight':163,'left':64,'right':29};selected=[]
for split in ['calibration','confirmation']:
 for command,n in quotas.items():
  candidates=sorted([r for r in rows if logsplit[r['log']]==split and r['command']==command],key=lambda x:digest(['quickcheck_scene_v1',x['token']]))
  assert len(candidates)>=n,(split,command,len(candidates),n)
  selected += [dict(r,split=split) for r in candidates[:n]]
selected.sort(key=lambda x:(x['split'],digest(['ordering',x['token']])))
smoke=[]
for command,n in [('straight',8),('left',4),('right',4)]:smoke += [r['token'] for r in selected if r['split']=='calibration' and r['command']==command][:n]
external={s:[r['token'] for r in sorted(selected,key=lambda x:digest(['external_fixed64',x['token']])) if r['split']==s][:64] for s in ['calibration','confirmation']}
manifest=dict(sampling_frame='Previously hash-random, command-stratified Navtrain 1000; outcome-unconditioned subset of 103288',
 source_path=str(ROOT/'outputs/historical_sft_grpo_support/manifests/scenes.json'),source_sha256=sha(ROOT/'outputs/historical_sft_grpo_support/manifests/scenes.json'),
 source_scene_count=1000,scene_count=512,selection='hash log split then command-stratified fixed token hash',quotas_per_split=quotas,
 smoke_tokens=smoke,external_tokens=external,scenes=selected,
 confirmation_semantics='This probe does not update/tune on confirmation; historical Navtrain can have participated in original training. Not an untouched generalization test.',
 logs_by_split={s:len({r['log'] for r in selected if r['split']==s}) for s in ['calibration','confirmation']})
assert not {r['log'] for r in selected if r['split']=='calibration'}&{r['log'] for r in selected if r['split']=='confirmation'}
config=dict(version=1,models=list(models),main_chain='A5 epoch155 -> formal LFP r4 step300',early_checkpoint_selection='fixed earliest formal step300 before new results',
 checkpoint_selection_uses_current_results=False,training_group_size=16,groups_per_scene=2,deployment_groups_per_scene=2,deployment_draws_per_group=16,
 native_precision='autocast_float16_with_fp32_parameters (archived 16-mixed)',deployment_precision='float32 (repository evaluation standard)',
 vlm_precision='bfloat16 using archived feature builder',historical_hidden_cache_missing=True,
 observation_policy='Rebuild from 4 observed frames with archived VLM builder; not byte-identical recovery of deleted historical hidden cache.',
 old_fp32_trajectory_cache_reuse=False,read_only=True,optimizer_updates=0,seed=2026092107,bootstrap_replicates=3000,bootstrap_unit='log_cluster_paired_scene_equal',
 smoke_scenes=16,formal_scenes=512,positive_credit_epsilon=1e-8,component_equality_atol=1e-6,
 tolerances=dict(scalar_batch=1e-8,batch_companions=1e-8,native_capture=0.,parameter_buffer_hash='exact',repeated_sampling=0.),
 train_scalar_weights=dict(EP=10.,TTC=5.,Comfort=2.,DDC=0.),official_scalar_weights=dict(EP=5.,TTC=5.,Comfort=2.,DDC=0.),
 shadow_ep_constant=1.0,advantage_batch_size=64,advantage_batch_semantics='Fixed hash batches of 64 newly sampled scenes; equivalent DDP global reduction, NOT historical minibatches or historical advantages.',
 matching=dict(primary_model='a5_sft',protocol='native_grpo',require_NC_DAC_TTC_1=True,DDC='actual LFP GT-relative threshold',
 min_pdms=.95,max_pdms_difference=.01,progress_tolerance_m=.5,progress_sensitivities_m=[.25,1.],max_pairs_per_scene=2,
 pair_selection='fixed hash, nonoverlapping candidate IDs; within original G16 groups primary; merged G32 coverage descriptive',exclude_low_progress_branch_from_primary=True),
 margins=dict(object_clearance='time-aligned official footprints, exclude red-light pseudo-objects, cap unavailable at null',road_margin='signed footprint clearance to union of official drivable semantic layers',selector='min(object_clearance capped at 10m, signed road clearance capped at 10m); null when both unavailable'),
 perturbations=dict(selection_n=8,evaluation_n=8,streams=['selection_v1','evaluation_v1','training_v1'],
 candidate_amplitudes=[dict(lateral_m=.05,longitudinal_m=.20),dict(lateral_m=.10,longitudinal_m=.50)],
 primary_choice_rule='Use smaller amplitude if calibration numerical/dynamics pass >=99%; otherwise BLOCKED; no risk-based amplitude selection and no confirmation-based increase',
 basis='b(u)=16*u^2*(1-u)^2, b(0)=b(1)=b\'(0)=b\'(1)=0; local path-tangent/Frenet frame; analytic derivative-consistent heading',
 dynamics=dict(max_speed_mps=40.,max_acceleration_mps2=12.,max_jerk_mps3=40.,max_yaw_rate_rps=2.,allow_baseline_exceedance_if_not_worse=True),
 primary_risk='mean(NC<1 or DAC<1 or TTC<1)',external_probe_trigger='confirmation within-group eligible scenes<50 OR high-score-group coverage<0.10',
 external_max_scenes_per_split=64,external_max_candidates=6),
 update_gate=dict(min_confirmation_scenes=50,min_high_score_group_coverage=.10,risk_ci_upper_lt=0.,nominal_pdms_ci_lower_ge=-.005,
 require_no_NC_DAC_reverse_harm=True,require_identity_and_smoke=True),
 tiny_update=dict(start='a5_sft',arms=['A','B','C','D'],seeds=[2026092111,2026092112],steps=[0,1,8],lambda_risk=1.,
 ep_weight_arm_B=.5,lr=agent['lr'],global_batch=64,per_gpu_batch=8,world_size=8,optimizer='actual archived AdamW recipe',
 only_if='WITHIN_GROUP_SIGNAL and identity/scoring gate PASS',no_automatic_additional_experiments=True))
assert agent['bc_coeff_start']==0 and agent['lfp_grpo_cfg']['reference_kl_coeff']==.005
(OUT/'resolved_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
(WORK/'configs/grpo_credit_risk_quickcheck/primary.yaml').write_text((OUT/'resolved_config.yaml').read_text())
save(OUT/'manifests/scenes.json',manifest);save(OUT/'manifests/models.json',models);save(OUT/'manifests/archived_training_config.json',run)
files=[Path(m['checkpoint_path']) for m in models.values()]+[Path(models['a5_sft']['config']['fs_norm_stats_path']),Path(agent['lfp_grpo_cfg']['reference_cache_path']),Path(models['a5_sft']['grpo_config_path'])]
files+=list((OUT/'source_snapshots').iterdir())
files+=list((RUNTIME/'navsim/agents/recogdrive').glob('*.py'))
files+=[RUNTIME/p for p in ['navsim/evaluate/pdm_score.py','navsim/evaluate/pdm_score_batch.py','navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py','navsim/planning/simulation/planner/pdm_planner/scoring/fast_pdm_scorer.py','navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py']]
vlm=Path(agent['vlm_path']);files += [p for p in vlm.iterdir() if p.is_file()]
save(OUT/'manifests/input_files.json',[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in files])
save(OUT/'manifests/frozen.json',dict(unix=time.time(),branch=subprocess.check_output(['git','branch','--show-current'],cwd=WORK,text=True).strip(),
 base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip(),config_sha256=sha(OUT/'resolved_config.yaml'),scene_manifest_sha256=sha(OUT/'manifests/scenes.json'),
 runtime_commit='f350e56408584c0d375b1ff10744da2750442580',historical_dirty_diff_missing=True,
 runtime_limit='Archived git source matches recovered sampler/advantage. Original A5 uncommitted edits and original deleted hidden-cache bytes cannot be proven.',
 current_run_selection='Recent actual runs found were SimScale SFT/LoRA and original-GRPO diagnostic probes, not a more specifically identified failing GRPO training chain; fallback A5 r4 selected as instructed.',
 agents_note='Root AGENTS.md absent; AGENT.md read; user read-only and isolated-worktree instructions override unrelated training/pressure defaults.'.replace('ag ents','agents')))
print('FROZEN',manifest['logs_by_split'],len(selected))
