# ReCogDrive / Last-VLA Experiment Index

Updated: 2026-06-10

This document is the lookup index for our ReCogDrive reproduction and
Last-VLA improvement attempts. It records which code, scripts, cache roots,
outputs, and reports belong to each stage. It is not a cleanup manifest: do not
delete reports, evaluation summaries, manifests, or useful launch/eval scripts
based only on this document.

## Quick Conclusion

| Stage | Purpose | Current conclusion | Main artifacts |
| --- | --- | --- | --- |
| A0 official alignment | Reproduce original ReCogDrive stage2 IL with official-style training | Reproduced successfully. Best full NAVTEST PDMS: `0.864891` at `step_00100000`. This is the baseline for later stage2 comparisons. | `reports/a0_stage2_repro_current_results.md` |
| A4-V2 | Add expert feature path on top of the successful A0 official-aligned setup | Confirmed useful improvement in the A0-based comparison. Paired A4-V2 deltas are mostly positive in later checkpoints; e.g. `step_00160000` improves by `+0.003789` PDMS in the packaged paired run. | `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501_summary_bundle_20260530_040157` |
| Last-VLA latent CoT / decoupled high-cap | Add latent CoT condition from JEPA/VGGT teachers without replacing original VLM context | No confirmed improvement over A0. Earlier hard-bottleneck / summary variants were poor and are deprecated. | `docs/LaST_VLA_ReCogDrive_v2_Design.md`, `docs/Last_VLA_v2_Decoupled_HighCap_Runbook.md` |
| B LoRA variants | LoRA-tune VLM for CoT / self-driving context and then train/evaluate stage2 | No confirmed improvement. Direct-text NAVTEST: base `0.845804`, LoRA epoch002 `0.822256`, epoch003 `0.811971`, epoch005 `0.812922`. | `docs/Last_VLA_v2_VLM_LoRA_Design.md` |
| No-residual stage2 AB | Disable residual target and restore original ReCogDrive diffusion target semantics | Code semantics fixed; experiments did not establish improvement over A0. | `reports/last_vla_v2_no_residual_diffusion_review_fix_report.md` |
| VLM text residual-anchor | Use fixed 2B-base direct text trajectory as the residual anchor for diffusion | Active / unproven. Correct goal: diffusion predicts residual from fixed base-text anchor, not from live coarse trajectory. | `scripts/build_vlm_text_traj_anchor_cache.py`, `scripts/run_recogdrive_stage2_residual_anchor_8gpu.sh` |
| Stage3 RL reproduction | Reproduce stage3 RL / GRPO-style phase | Reproduction infrastructure exists. No final confirmed improvement should be claimed until same-protocol full summary is promoted. | `scripts/training/launch_recogdrive_stage3_rl_2b_local_stable.sh` |
| Stage3 safe DiffGRPO attempt | Improved safer stage3 launch/eval flow | Active / needs final verification. Stream eval produced raw full NAVTEST score means around `0.8785` and `0.8899` for early ckpts, but these are not yet treated as the official stage3 conclusion. | `/mnt/project/VLA-AD/outputs/stage3_safe_diffgrpo_ckpt_stream_eval_live_20260610T030355Z` |

Working rule: the only stage2 improvement that is currently confirmed and
should be treated as a useful milestone is A4-V2. Later Last-VLA/LoRA/residual
stage2 attempts are useful engineering records but not better models unless a
new same-protocol full NAVTEST report proves otherwise.

## Environment And Data Roots

| Item | Path / value | Notes |
| --- | --- | --- |
| Code workspace | `/mnt/project/VLA-AD_last_vla_dev` | Current branch: `feature/recogdrive-last-vla-v2`. |
| Main experiment/data mount | `/mnt/project/VLA-AD` | Outputs, caches, checkpoints, experiment bundles. |
| Training data mount | `/mnt/navsim` | Same backing mount as `/mnt/project` in this container. |
| Conda env | `/root/miniconda3/envs/navsim` | Use this Python for tests/eval scripts. |
| Stable launch skill | `/mnt/project/skill/multi-container-training-dispatch/SKILL.md` | Uses detached `setsid` launchers and status files. |

## A0 Official Alignment

Purpose: reproduce the original ReCogDrive stage2 IL baseline with the
official-aligned Lightning-style training path.

Result:
- Best full NAVTEST PDMS: `0.864891` at `step_00100000`.
- Final PDMS: `0.860038`.
- Evaluation: full navtest, `12146` samples, `12138` valid PDM rows, fp32 eval,
  expert target tokens disabled.

Important paths:
- Report: `reports/a0_stage2_repro_current_results.md`
- Guardrails: `docs/OfficialAlignedBaselineGuardrails.md`
- Output root:
  `/mnt/project/VLA-AD/outputs/a0_stage2_repro_20260531_003029/a0_official_aligned_a0complete`
- Launch script: `scripts/run_a0_official_aligned_8gpu.sh`
- Eval script: `scripts/eval_a0_stage2_checkpoints.sh`
- Summary script: `scripts/summarize_a0_stage2_repro.py`
- Loader smoke: `scripts/smoke_check_official_aligned_local_loader.py`

Cache notes:
- Original A0 complete cache root was
  `/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1_a0_complete`.
  That directory is no longer present after cleanup.
- Current preserved base hidden/expert cache root is
  `/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1`.
- The 252 missing A0-complete records were documented, not preserved as a full
  training cache. See `reports/navtrain_missing_252_manifest.jsonl` and
  `reports/navtrain_missing_252_backfill_report.md`.

## A4-V2

Purpose: improve the successfully reproduced A0 official-aligned stage2 by
adding the A4-V2 expert feature path. This is the stage2 attempt we currently
treat as the useful improved milestone.

Result summary from the packaged paired run:
- Experiment bundle:
  `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501_summary_bundle_20260530_040157`
- Experiment root:
  `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501`
- Best available packaged A4-V2 PDMS: `0.858101` at `step_00160000`.
- Same packaged paired A0 at `step_00160000`: `0.854313`.
- Delta at `step_00160000`: `+0.003789` PDMS.
- Paired deltas are positive at `50k`, `60k`, `100k`, `120k`, `140k`, and
  `160k`; `80k` is the main negative outlier.

Use this evidence with the correct comparison scope: A4-V2 was developed after
A0 official alignment succeeded, and its useful conclusion is the same-line
A0-based improvement. Do not compare a partial/older bundle row against a later
different A0 eval protocol and then conclude A4-V2 failed.

Important files in the bundle:
- `README.md`
- `evaluation/evaluation_pairwise_delta.csv`
- `evaluation/evaluation_summary_long.csv`
- `training/training_summary.csv`
- `training/training_metrics_at_key_steps.csv`
- `configs/recogdrive2b_A0_base_no_expert.yaml`
- `configs/recogdrive2b_A4_v2.yaml`
- `runs/a0_no_expert/final_report.md`
- `runs/a4_v2/final_report.md`
- `checkpoints/checkpoint_inventory.csv`

Code/config/scripts:
- Agent config:
  `navsim/planning/script/config/common/agent/recogdrive_agent_a4_v2.yaml`
- Alignment configs:
  `navsim/planning/script/config/common/agent/recogdrive_agent_a4_v2_align_stage1.yaml`,
  `navsim/planning/script/config/common/agent/recogdrive_agent_a4_v2_stage2_noalign.yaml`
- Main launch: `scripts/run_a4_v2_official_aligned_8gpu.sh`
- Align-first launch: `scripts/run_a4_v2_official_align_first_8gpu.sh`
- Eval: `scripts/eval_a4_v2_official_aligned_checkpoints.sh`
- Matrix launch/summarize:
  `scripts/run_a4_8gpu_continuation_matrix.sh`,
  `scripts/summarize_a4_8gpu_continuation.py`
- Core code:
  `navsim/agents/recogdrive/expert_backends.py`,
  `navsim/agents/recogdrive/expert_cache.py`,
  `navsim/agents/recogdrive/recogdrive_features.py`,
  `navsim/agents/recogdrive/recogdrive_diffusion_planner.py`,
  `navsim/agents/recogdrive/recogdrive_agent.py`,
  `navsim/planning/script/run_training_recogdrive.py`

Cache:
- Main A4 cache root:
  `/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1`
- Approximate size observed: `197G`.
- It contains the train hidden cache plus JEPA/VGGT expert fields used by A4.
- Do not point A4-V2 to the removed `full_v1_a0_complete` overlay. The 252
  hidden-only A0 backfill records do not provide the same full expert fields.

## Last-VLA Latent CoT / Decoupled High-Cap

Purpose: add latent CoT as an extra diffusion condition while keeping the raw
ReCogDrive VLM hidden context. The intended current design is the decoupled
high-cap no-risk line, not the old summary replacement line.

Design:
- Use full raw ReCogDrive VLM hidden tokens as the base DiT context.
- Add latent CoT through a separate residual condition branch.
- JEPA future/dynamic tokens and VGGT geometry tokens supervise/condition CoT.
- Avoid hard summary bottlenecks and old `12`-token fallback settings.

Important docs:
- `docs/LaST_VLA_ReCogDrive_v2_Design.md`
- `docs/Last_VLA_v2_Decoupled_CoT_Design.md`
- `docs/Last_VLA_v2_Decoupled_HighCap_Runbook.md`
- `reports/last_vla_v2_decoupled_highcap_no_risk_pdms_summary_20260608.md`
- `reports/last_vla_v2_highcap_no_risk_serverA_pdms_diagnosis_20260606.md`

Current useful cache entrypoint:
- Env file:
  `/mnt/project/VLA-AD/cache/last_vla_v2/experiments/decoupled_highcap_no_risk/cache.env`
- Train high-cap chunks:
  `/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/train_full_highcap_chunks`
- NAVTEST high-cap chunks:
  `/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/navtest_full_highcap_chunks`
- NAVTEST metric cache:
  `/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1`

Cache sizes observed:
- `train_full_highcap_chunks`: about `1.8T`.
- `navtest_full_highcap_chunks`: about `216G`.

Current high-cap schema:
- JEPA: `128 x 1024`.
- VGGT geometry: `192 x 512`.
- CoT target length: high-cap design uses `192` latent tokens.
- Train records: `103036`.
- NAVTEST records: `12146`.

Key scripts:
- `scripts/last_vla_v2/decoupled_highcap_no_risk/prepare_decoupled_highcap_no_risk_data.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/run_strict_preflight_decoupled.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/serverA_frozen_vlm_decoupled_highcap_no_risk.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/serverB_vlm_lora_decoupled_highcap_no_risk.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/run_lora_cache_and_progressive_decoupled.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/build_navtest_highcap_cache_decoupled.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/eval_checkpoint_sweep_decoupled.sh`
- `scripts/monitor_last_vla_stage2_ab_progress.py`

Representative outputs:
- `/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_ab_cacheenv_setsid_20260607T022706Z`
- `/mnt/project/VLA-AD/outputs/last_vla_v2/no_residual_stage2_ab_top5val_20260609T021059Z`
- `/mnt/project/VLA-AD/outputs/last_vla_v2/no_residual_stage2_ab_ckpt_parallel_eval_20260609T124343Z`
- `/mnt/project/VLA-AD/outputs/last_vla_v2/no_residual_stage2_ab_fixed_step_eval_20260609T164130Z`
- `/mnt/project/VLA-AD/outputs/last_vla_v2/no_residual_stage2_ab_A_remaining_eval_20260610T034415Z`

Known results:
- Base 2B IL full NAVTEST `pred_traj`: `0.858575`.
- B LoRA direct-online epoch003 `pred_coarse_traj`: `0.766702`.
- A CoT-only `pred_coarse_traj`: `0.561656`.
- A progressive `step_00080000` `pred_traj`: `0.374729`.
- No confirmed improvement over A0.

Deprecated / do-not-use notes:
- Do not use old hard-bottleneck summary caches.
- Do not use old `12`-token fallback settings.
- Do not use VLM summary-compressed variants as the current design reference.

## B LoRA And Direct VLM Text Evaluation

Purpose: test whether LoRA tuning the VLM improves CoT/scene understanding and
whether the VLM itself can still directly emit trajectories.

Design notes:
- B group has a LoRA stage before progressive stage2.
- Saved LoRA checkpoints/adapters of interest: epochs `2`, `3`, and `5`.
- Recommended LoRA design is documented in `docs/Last_VLA_v2_VLM_LoRA_Design.md`.

Important scripts:
- `scripts/last_vla_v2/decoupled_highcap_no_risk/serverB_vlm_lora_decoupled_highcap_no_risk.sh`
- `scripts/build_recogdrive_hidden_cache_with_lora.py`
- `scripts/last_vla_v2/extract_vlm_lora_and_cot_adapters.py`
- `scripts/last_vla_v2/lora/run_lora_cot_alignment_sweep.sh`
- `scripts/eval_vlm_direct_text_pdm.py`
- `scripts/eval_vlm_vqa_smoke.py`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/run_vlm_direct_text_base_b_lora_eval_8shard_parallel_all.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/run_b_lora_pred_traj_parallel_eval.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/run_b_lora_coarse_traj_parallel_eval.sh`

Direct-text full NAVTEST result file:
- `/mnt/project/VLA-AD/outputs/last_vla_v2/direct_text_navtest_eval_8shard_parallel_20260608T135058Z/summary.csv`

Direct-text result summary:
- Base ReCogDrive VLM 2B: `0.845804`.
- B LoRA epoch002: `0.822256`.
- B LoRA epoch003: `0.811971`.
- B LoRA epoch005: `0.812922`.

Interpretation:
- These LoRA checkpoints did not improve direct text trajectory output.
- Earlier confusion between `coarse_traj`, `pred_traj`, and stage2 IL output
  should not be repeated. For direct VLM text tests, use the explicit direct
  text eval script and record the trajectory output key.

VQA smoke:
- Output root:
  `/mnt/project/VLA-AD/outputs/last_vla_v2/vqa_scene_understanding_20260608T142120Z`
- Summary file: `summary.csv`.
- This is diagnostic only; do not use it as a PDMS claim.

## Residual Diffusion Attempts

There are two different residual concepts. Keep them separate.

### Old coarse residual design

Old idea:
- Target: `selected_target_norm - alpha * coarse_traj_norm`.
- Final action could be interpreted as coarse plus residual.

Current status:
- This was intentionally disabled in the no-residual stage2 fix.
- Current no-residual semantics:
  diffusion target is `selected_target_norm`, final `pred_traj` is
  `denorm(current_actions)`, `pred_coarse_traj` is diagnostic only, and
  `pred_residual_norm` is not emitted.

Report:
- `reports/last_vla_v2_no_residual_diffusion_review_fix_report.md`

Tests:
- `tests/test_last_vla_diffusion_target_no_residual.py`
- `tests/test_last_vla_sample_chain_cot_condition.py`

### VLM text residual-anchor design

New idea:
- Decode fixed 2B-base direct text trajectories first.
- Store them as an anchor cache.
- In diffusion training, predict residual from the fixed anchor:
  `selected_target_norm - anchor_norm`.
- At inference, reconstruct final trajectory from fixed anchor plus predicted
  residual.
- This anchor is not the live CoT coarse trajectory and should not update with
  the diffusion model.

Anchor cache:
- `/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/vlm_text_anchor_2bbase_train_20260609T052233Z`
- Contains `index.jsonl`, `metadata.json`, and per-sample `.pt` files.
- Parse failures were repaired; keep `repair_backup_20260610_parse_fail_2`
  unless a later cleanup explicitly archives it.

Scripts:
- `scripts/build_vlm_text_traj_anchor_cache.py`
- `scripts/merge_vlm_text_traj_anchor_cache_into_chunks.py`
- `scripts/run_recogdrive_stage2_residual_anchor_8gpu.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/launch_vlm_text_residual_anchor_stage2_ab.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/resume_vlm_text_residual_anchor_stage2_ab.sh`

Configs/tests:
- `configs/last_vla_v2/decoupled_highcap_no_risk/progressive_sft_vlm_text_residual_eval_flat.yaml`
- `navsim/planning/script/config/experiment/last_vla_decoupled_progressive_highcap_no_risk_vlm_text_residual.yaml`
- `tests/test_last_vla_vlm_text_residual_anchor.py`
- `tests/test_recogdrive_original_stage2_residual_anchor.py`

Active / recent output roots:
- Pure ReCogDrive residual-anchor stage2, no Last-VLA/CoT/external features:
  `/mnt/project/VLA-AD/outputs/recogdrive_stage2_residual_anchor_base2b_20260610T034055Z_setsid_freezecot`
- Last-VLA CoT residual-anchor A remote:
  `/mnt/project/VLA-AD/outputs/last_vla_v2/vlm_text_residual_anchor_stage2_A_remote_20260610T030302Z/A_frozen_vlm_stage2_progressive`

Important comparison caveat:
- The pure ReCogDrive residual-anchor run and the Last-VLA CoT residual-anchor
  run are not a perfectly isolated "only CoT differs" ablation. The Last-VLA
  side also includes CoT/expert condition machinery.
- No confirmed residual-anchor improvement should be claimed yet.

## Stage3 RL / Safe DiffGRPO

Purpose: reproduce and improve the stage3 RL phase after stage2.

Important scripts:
- Metric cache:
  `scripts/cache_dataset/launch_metric_caching_train_local_full_stable.sh`,
  `scripts/cache_dataset/run_metric_caching_train_local_full.sh`
- Stage3 training:
  `scripts/training/launch_recogdrive_stage3_rl_2b_local_stable.sh`,
  `scripts/training/run_recogdrive_stage3_rl_2b_local.sh`,
  `scripts/training/watch_metric_cache_then_launch_stage3_rl_2b_local.sh`
- Stage3 eval:
  `scripts/evaluation/run_recogdrive_stage3_safe_diffgrpo_eval_8gpu.sh`,
  `scripts/evaluation/watch_stage3_training_then_eval_8gpu.sh`,
  `scripts/evaluation/watch_stage3_checkpoints_eval_8gpu.sh`

Output roots:
- Metric cache:
  `/mnt/project/VLA-AD/outputs/metric_cache_navtrain_full_safe_diffgrpo_20260609T175818Z`
- First stage3 launch:
  `/mnt/project/VLA-AD/outputs/stage3_rl_2b_safe_diffgrpo_online_20260609T175818Z`
- Relaunch:
  `/mnt/project/VLA-AD/outputs/stage3_rl_2b_safe_diffgrpo_online_relaunch_20260609T1944Z`
- Stream checkpoint eval:
  `/mnt/project/VLA-AD/outputs/stage3_safe_diffgrpo_ckpt_stream_eval_live_20260610T030355Z`

Known status:
- First launch summary reports `failed`.
- Metric cache job reports `done`.
- Relaunch produced early checkpoints, and stream eval processed
  `epoch_0-step_1330` and `epoch_1-step_2660`.
- Raw full NAVTEST score means observed from the stream CSVs:
  `0.878473` and `0.889881`.

Interpretation:
- Treat these as stage3 diagnostic / in-progress results until a formal
  same-protocol summary is written and compared against the correct baseline.
- Do not overwrite the stage2 conclusion that A4-V2 is the only confirmed
  stage2 improvement.

## Cache / Data Map

| Path | Approx size / records | Used by | Notes |
| --- | ---: | --- | --- |
| `/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1` | about `197G` | A0-style hidden cache, A4-V2 expert cache | Current preserved base expert chunk cache. |
| `/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1_a0_complete` | missing | Old A0 complete run | Removed; use reports/manifests only. |
| `/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/train_full_highcap_chunks` | about `1.8T`, `103036` train records | Last-VLA high-cap stage2 | Uses JEPA `128 x 1024`, VGGT geometry `192 x 512`. |
| `/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/navtest_full_highcap_chunks` | about `216G`, `12146` navtest records | Last-VLA NAVTEST eval | Generated to avoid repeating base high-cap features. |
| `/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/vlm_text_anchor_2bbase_train_20260609T052233Z` | many per-sample `.pt` anchors | VLM text residual-anchor | Fixed 2B-base direct text trajectory anchor. |
| `/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1` | not fully remeasured in this pass | NAVTEST PDMS scoring | Keep; reused by full eval. |
| `/mnt/project/VLA-AD/cache/metric_cache_navtrain_full_v1` | not fully remeasured in this pass | Stage3 / train metric cache | Keep if stage3 work continues. |
| `/mnt/project/VLA-AD/cache/last_vla_v2/experiments/decoupled_highcap_no_risk/cache.env` | small | Last-VLA scripts | Preferred env entrypoint for cache roots. |

## System Disk Root-Level Artifacts

The user remembered cache generation outside the mounted project disk. The
system disk was checked with `/mnt` excluded.

Findings:
- `/` filesystem: `3.0T` total, `913G` used, `2.1T` available.
- Mounted project disk `/mnt/project` and `/mnt/navsim`: full / near full.
- Root filesystem actual directory usage with mount exclusion is about `38G`.
- `/tmp` total is about `1.5G`.
- `/root/.cache` is small, about `760K`.
- No large `.pt`, `.ckpt`, `.pth`, or `.safetensors` files above `1G` were
  found under `/root`.

Notable system `/tmp` remnants:
- `/tmp/a4_align_first_smoke`: about `1.1G`; contains smoke checkpoints:
  `stage1_align/best.ckpt`, `stage1_align/latest.ckpt`,
  `stage2_condition/best.ckpt`, `stage2_condition/latest.ckpt`.
- `/tmp/last_vla_*`: small dryrun/readiness/cache smoke directories.
- `/tmp/torchinductor_root`: about `29M`.
- `/tmp/pytest-of-root`: about `153M`.

Cleanup guidance:
- `/tmp/a4_align_first_smoke` is a cleanup candidate if we only need the formal
  A4 reports and not the smoke ckpts.
- Do not delete it automatically without an explicit cleanup request, because
  it contains checkpoint files even though they are smoke artifacts.

## Retention Policy

Keep:
- Reports under `reports/`.
- Design/runbook docs under `docs/`.
- Evaluation summaries, `metrics.json`, CSV/TSV summaries, manifests, and
  final reports.
- Scripts that launch, evaluate, summarize, monitor, or rebuild important
  stages.
- A0 official-aligned and A4-V2 milestone records.
- Cache env files and manifests that explain large cache roots.

Usually removable after confirmation:
- Old training weights when the result is already summarized.
- Duplicate cache shards after merge if the merged cache has a manifest and has
  been validated.
- Failed/obsolete temporary logs.
- `/tmp` dryrun/smoke directories.
- Deprecated hard-bottleneck Last-VLA caches and old VLM-summary caches.
- Old `12`-token fallback cache/config artifacts.

Do not use:
- VLM summary-compressed Last-VLA artifacts as the current design.
- Old hard-bottleneck scripts except as historical archive.
- Ambiguous eval results that do not record whether they used `pred_traj`,
  `pred_coarse_traj`, direct text output, or stage2 IL `pred_traj`.

## Useful Lookup Commands

From repo root:

```bash
rg -n "last_vla|A4|A0|residual|LoRA|stage3|DiffGRPO" docs reports scripts
find /mnt/project/VLA-AD/outputs -maxdepth 4 -type f -name "summary.csv" | sort
find /mnt/project/VLA-AD/outputs -maxdepth 4 -type f -name "metrics.json" | sort
find /mnt/project/VLA-AD/cache -maxdepth 4 -type f -name "merge_summary.json" | sort
find /mnt/project/VLA-AD/cache -maxdepth 4 -type f -name "cache.env" | sort
df -h / /mnt/project /mnt/navsim
du -xh --max-depth=2 /tmp | sort -h | tail
```

For direct-text VLM trajectory evals, always check:
- `trajectory_output_key`
- `model_path`
- `lora_adapter_dir`
- `prompt_type`
- `num_parse_ok`
- `PDMS`

For stage2 checkpoint evals, always record:
- checkpoint path
- output trajectory key
- number of navtest samples
- number of valid PDM rows
- metric cache path
- eval precision
