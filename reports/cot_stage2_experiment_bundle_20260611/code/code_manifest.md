# Code Manifest

Generated: 2026-06-11 UTC

This manifest lists the source/config/script files that define the CoT stage2 experiment family. Snapshots are copied under this bundle; canonical files remain at repo root.

## Config snapshots

- `configs/last_vla_v2/decoupled_highcap_no_risk/base.yaml`
- `configs/last_vla_v2/decoupled_highcap_no_risk/cot_alignment.yaml`
- `configs/last_vla_v2/decoupled_highcap_no_risk/progressive_sft.yaml`
- `configs/last_vla_v2/decoupled_highcap_no_risk/progressive_sft_eval.yaml`
- `configs/last_vla_v2/decoupled_highcap_no_risk/progressive_sft_eval_flat.yaml`
- `configs/last_vla_v2/decoupled_highcap_no_risk/original_recogdrive_residual_anchor_eval_flat.yaml`
- `configs/last_vla_v2/decoupled_highcap_no_risk/progressive_sft_vlm_text_residual_eval_flat.yaml`
- `configs/last_vla_v2/decoupled_highcap_no_risk/vlm_lora_cot_alignment.yaml`
- `navsim/planning/script/config/common/agent/recogdrive_agent.yaml`
- `navsim/planning/script/config/experiment/last_vla_decoupled_cot_alignment_highcap_no_risk.yaml`
- `navsim/planning/script/config/experiment/last_vla_decoupled_progressive_highcap_no_risk.yaml`
- `navsim/planning/script/config/experiment/last_vla_decoupled_progressive_highcap_no_risk_vlm_text_residual.yaml`
- `navsim/planning/script/config/experiment/last_vla_decoupled_vlm_lora_cot_alignment_highcap_no_risk.yaml`

## Launch/eval script snapshots

- `scripts/last_vla_v2/decoupled_highcap_no_risk/eval_checkpoint_sweep_decoupled.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/launch_no_residual_stage2_ab_top5val.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/launch_vlm_text_residual_anchor_stage2_ab.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/resume_vlm_text_residual_anchor_stage2_ab.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/launch_residual_anchor_fixed_step_eval.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/launch_residual_anchor_live_eval_watcher.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/launch_residual_anchor_eval_after_navtest_anchor.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/run_b_lora_coarse_traj_parallel_eval.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/run_b_lora_pred_traj_parallel_eval.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/run_b_progressive_ckpt_navtest_parallel_eval.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/run_b_progressive_ckpt_navtest_direct_online_watcher.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/run_current_ab_ckpt_navtest_parallel_eval.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/watch_ab_top5_eval_after_training.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/watch_top5_ckpts_live.py`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/watch_residual_anchor_live_eval.py`
- `scripts/build_vlm_text_traj_anchor_cache.py`
- `scripts/merge_vlm_text_traj_anchor_cache_into_chunks.py`
- `scripts/run_recogdrive_stage2_residual_anchor_8gpu.sh`
- `scripts/eval_recogdrive_expert_pdm.py`
- `scripts/eval_vlm_direct_text_pdm.py`
- `scripts/aggregate_vlm_direct_text_eval.py`
- `scripts/monitor_last_vla_stage2_ab_progress.py`
- `scripts/monitor_direct_text_eval_progress.py`

## Source/test snapshots

- `navsim/agents/recogdrive/recogdrive_diffusion_planner.py`
- `navsim/agents/recogdrive/recogdrive_agent.py`
- `navsim/agents/recogdrive/recogdrive_features.py`
- `navsim/agents/recogdrive/expert_backends.py`
- `navsim/agents/recogdrive/expert_cache.py`
- `navsim/planning/script/run_training_recogdrive.py`
- `navsim/planning/script/run_training_recogdrive_rl.py`
- `navsim/planning/training/agent_lightning_module.py`
- `tests/test_last_vla_diffusion_target_no_residual.py`
- `tests/test_last_vla_sample_chain_cot_condition.py`
- `tests/test_last_vla_vlm_text_residual_anchor.py`
- `tests/test_recogdrive_original_stage2_residual_anchor.py`

## Report/doc snapshots

- `reports/a0_stage2_repro_current_results.md`
- `reports/last_vla_v2_decoupled_highcap_no_risk_pdms_summary_20260608.md`
- `reports/last_vla_v2_highcap_no_risk_serverA_navtest_results_20260606.md`
- `reports/last_vla_v2_highcap_no_risk_serverA_pdms_diagnosis_20260606.md`
- `reports/last_vla_v2_no_residual_diffusion_review_fix_report.md`
- `docs/ReCogDrive_Experiment_Index_zh.md`
- `docs/Last_VLA_v2_Decoupled_HighCap_Runbook.md`

## Current Worktree Diff

`current_worktree_diff.patch` captures tracked-file modifications at bundle generation time. Untracked files present at generation time are listed in `untracked_files_at_generation.txt`; stage2-relevant snapshots are copied above.
