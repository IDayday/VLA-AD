# ReCogDrive Stage3 Algorithm Attempts And Reference Protocol

Date: 2026-06-14 UTC

This is the required ledger for ReCogDrive Stage3 policy-improvement work. Before any new algorithm change or large run, read this file, update the planned attempt, and check the cited paper/open-source implementations for the exact mechanism being borrowed.

The goal is not to prove a simplified toy version. The goal is to implement mature enough versions of each method that a negative result actually means something.

## Mandatory Protocol

Every new Stage3 algorithm change must satisfy this checklist before launch:

1. Past evidence:
   - Identify which previous run, checkpoint, or buffer statistic motivates the change.
   - State the targeted failure mode: weak policy absorption, poor exploration, DDC/TTC regression, invalid candidate leakage, low valid-candidate coverage, learning-rate confound, or inefficient reward evaluation.

2. Reference audit:
   - Search current papers and official/open-source implementations for the method family.
   - Record the sources checked and the exact mechanism copied.
   - Do not use a hand-rolled lightweight approximation as evidence that the method works or fails unless it is explicitly labeled as a smoke test.

3. Mature implementation checklist:
   - Data path: train-only rewards/cache for training; no navtest reward, label, metric cache, or navtest-derived artifact.
   - Policy objective: include the core pieces required by the reference method, such as old policy logprobs, ratio clipping, advantage normalization, multi-epoch minibatch replay, reference model terms, or shared-noise preference comparisons.
   - Diffusion details: use correct denoising timestep treatment, shared noise/timestep where needed, and no target leakage through conditioning tokens.
   - Safety semantics: NC/DAC/TTC/DDC guards must be explicit and logged.
   - Diagnostics: log the quantities that prove the objective is active, not just final PDMS.
   - Controls: separate algorithm effect from LR, effective batch size, optimizer steps, checkpoint epoch, and evaluation seed/sampling differences.

4. Launch requirement:
   - Add a planned-attempt entry to this document before starting the run.
   - Include config, expected diagnostic movement, failure criteria, and references.

5. Result requirement:
   - After each evaluated checkpoint, append the result row.
   - Include PDMS plus NC, DAC, TTC, EP, comfort, DDC, and TLC when available.

## Current Experiment Decisions

| Run / Process | Decision | Reason |
|---|---|---|
| `stage3_grpo_refkl_s16_lr2e4_b4acc2_chunk32_fast_8gpu_20260613T213145Z` | stopped after epoch0 ckpt | High-LR `2e-4` GRPO is not a new mature algorithm mechanism and train reward regressed after epoch0. Keep `epoch_0-step_1330` only as an LR/eval data point. |
| `stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z` | evaluated/keep | Original-LR `1e-4` control with similar effective batch scale. Exact navtest at `epoch_0-step_1290` reached PDMS `0.896794`, clearly above the recent cap05 and v3 early checkpoints. Keep as the current short-run control. |
| `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_log1_diag_zt2_4gpu_20260614T0300Z` | completed diagnostic | Good limited diagnostic. Ratio/KL/clip/advantage logs are active, but this still does not validate a full PPO replay implementation. |
| `stage3_grpo_refkl_s16_lr2e4_b4acc2_chunk32_fast_8gpu_20260613T213145Z/local_eval_epoch0_exact_pool_8x1gpu_shards_20260614T031521Z` | completed eval | Local 8-way independent 1GPU exact navtest shards for the stopped `2e-4` epoch0 checkpoint. PDMS `0.872059`, clearly below the historical strong Stage3 baselines, so do not continue `2e-4` GRPO as a default route. |
| GRPO stable launcher defaults | changed | User clarified that original Stage3 used LR `1e-4`, and original `epoch0-1` PDMS is already around `0.88+`. Stable GRPO launchers now default to `1e-4`; `2e-4` should be treated as an explicit ablation only. |
| zt2 stale checkpoint watchers for old AWAC / old `2e-4` GRPO runs | stopped on 2026-06-14 | These watchers pointed to runs that no longer satisfy the protocol and could auto-consume GPUs when resources become free. |
| zt3 stale checkpoint watchers for stopped buffer-guided `2e-4` and stopped `2e-4` GRPO | stopped on 2026-06-14 | Exact eval of the stopped `2e-4` epoch0 checkpoint is handled by the local shard run; the stale remote watchers were redundant or invalid. |
| `stage3_grpo_buffer_guided_gspo_s16_lr2e4_b2acc8_zt2_4gpu_20260614T012106Z` | stopped | It tested buffer absorption, but Lightning did not log the returned buffer/self-imitation diagnostics. Continuing would not prove whether the auxiliary path was active. |
| `stage3_grpo_buffer_guided_gspo_s16_lr2e4_b2acc8_diagfix_zt2_4gpu_20260614T020706Z` | stopped | It was launched before the PPO/advantage diagnostics patch. It could not report `gspo_approx_kl` or the full advantage-transform diagnostics, so it was stopped before becoming a full run. |
| `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_diag_zt2_4gpu_20260614T023758Z` | failed before training | Hydra struct rejected the new override before the agent YAML was updated. No training occurred. |
| `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_diag_zt2_4gpu_20260614T024536Z` | stop and replace | It entered training and wrote only `lr-AdamW`; with default `log_every_n_steps=50`, it is too low-information for a limited diagnostic. Replace with a shorter `log_every_n_steps=1` diagnostic and a long behavior-policy sync interval. |
| Local queued `stage3_grpo_buffer_guided_selfimit_s16_lr2e4_b2acc4_wait2_20260614T003614Z` launcher | stopped | It was a stale queued run from before this protocol and duplicates the zt2 buffer-guided experiment. |
| `supervise_recogdrive_stage3_experiment.py` auto-supervisor | stopped | It can auto-launch stale next actions without the new reference-audit protocol. Future launches should be manual after updating this file. |
| `stage3_grpo_replay_diag_s4_i2_lr1e4_4gpu_20260614T042519Z` | completed smoke | New PPO replay path completed 40 train batches and wrote `epoch=0-step=200.ckpt`. This validates plumbing and diagnostics only; it is not a PDMS result and not a full algorithm verdict. |
| `stage3_grpo_replay_diag_s16_i2_lr1e4_8gpu_20260614T043345Z` | completed diagnostic | Near-full sampling diagnostic passed the replay gates at `sample_time=16`: valid ratio stayed `1.0`, PPO clip fraction became active, KL stayed finite, and safety submetrics were logged. Promote to controlled 1-epoch training, not full 20-epoch training yet. |
| `stage3_grpo_replay_1epoch_s16_i2_lr1e4_8gpu_20260614T044444Z` | running/watch | Controlled full-navtrain 1-epoch PPO replay run. Replay validity is `1.0` and KL is small; one weak safety/reward batch appeared, so keep but do not promote until epoch eval. |
| `stage3_grpo_replay_1epoch_s16_i1_lr1e4_zt2_8gpu_20260614T050122Z` | running | Controlled zt2 ablation. Same as i2 but `ppo_replay_inner_epochs=1`; currently cleaner ratio/clip/KL behavior and no hard stop signal. |
| `stage3_grpo_dppo_transition_s16_i1_lr1e4_zt2_2gpu_20260614T083005Z` | failed promotion gate | DPPO-style transition replay improved from step300 to epoch0 but stayed far below the original Stage3 `epoch0-1 ~= 0.88+` early gate: step300 `0.744740`, step600 `0.801195`, epoch0-step800 `0.831015`. Do not continue this transition replay implementation without a method-level redesign. |
| `stage3_grpo_rloo_selfimit_s16_lr1e4_b2acc4_8gpu_20260614T101257Z` | stopped on 2026-06-14 | This no-cap run was launched before the self-imitation scene-cap patch and had `checkpoint.every_n_train_steps=0`; given the original Stage3 `epoch0-1 ~= 0.88+` early gate, continuing it would delay a meaningful PDMS decision. |
| `stage3_grpo_rloo_selfimit_cap05_step300_s16_lr1e4_b2acc4_8gpu_20260614T111511Z` | stopped after `step-step_900` checkpoint | Original LR `1e-4`, sample_time `16`, scene cap `0.5`, and step checkpoints every `300` train steps. `step-step_300` exact navtest PDMS is `0.885557`, passing the original Stage3 early `0.88+` gate but not yet a final success. `step-step_600` is `0.885212`: EP improved, but NC/TTC/DDC declined. `step-step_900` later reached `0.887107` by recovering NC/TTC/DDC while EP fell back. This is still far below original 10-epoch `0.9055` / Safe DiffGRPO `0.906184`, so the replacement safety-gated run remains justified. |
| `stage3_grpo_rloo_selfimit_safetygate_step300_s16_lr1e4_b2acc4_8gpu_20260614T145712Z` | stopped/deleted | This auxiliary-target-only safetygate run was stopped before checkpoint because the main GRPO update still allowed TTC/DDC-regressing positive-advantage samples. Do not repeat this exact configuration. |
| `stage3_grpo_rloo_selfimit_mainhardgate_v3_bufferpath_step300_s16_lr1e4_b2acc4_8gpu_20260614T161426Z` | stopped/delete artifacts | Main TTC `0.95` and DDC `0.99` hard gates improved safety but suppressed EP/DAC. Step300 PDMS `0.879395`; step600 PDMS `0.882934`, still below cap05 step900 `0.887107` and far below the zt3 original-LR control step1290 `0.896794`. Do not repeat this main-hardgate + strict self-imitation configuration as a default route. |
| `stage3_grpo_refkl_s16_lr1e4_b2acc4_currentrepo_8gpu_20260614T191227Z` | running/evaluated step300 | This is not a new algorithm. It is a matched original-LR GRPO control on the current repository after AWAC/GRPO code changes: LR `1e-4`, sample_time `16`, BC `0.10->0.05`, ref KL `0.02`, no GSPO ratio, no hard gates, no buffer, no self-imitation. Step300 exact navtest PDMS is `0.881910`, below the stronger zt3 original-LR control step1290 `0.896794`, so keep running as a control but do not treat this checkpoint as a success. |
| `stage3_grpo_buffer_dpo_refctrl_s16_lr1e4_b2acc4_zt2wait_8gpu_20260614T213001Z` | queued on zt2 | Control-aligned Buffer-DPO run launched on `training-vla-zt2` with strict GPU wait instead of preempting the existing zt2 two-expert job. Config matches the current-repo GRPO control shape plus only small train-buffer DPO: LR `1e-4`, sample_time `16`, BC `0.10->0.05`, ref KL `0.02`, no GSPO ratio, no main TTC/DDC hard gate, no reward bonus, no distill, no self-imitation, DPO weight `0.02`. |
| Stage3 next-action gate | patched | `scripts/training/decide_recogdrive_stage3_next_action.py` now tracks the current-repo original-LR control run instead of the stopped `2e-4` run, and its gated next experiment is the control-aligned Buffer-DPO launcher in this repository. After the step300 eval, dry-run result on 2026-06-14 UTC is `launch_buffer_dpo_after_current_checkpoint`, but default policy waits while the current control is still running. |
| Stage3 exact eval infrastructure | patched | Distributed eval timeout is now configurable and defaults to `3600s` for async/exact PDM runners and watcher launchers. Watchers also write global per-checkpoint done markers under `GLOBAL_EVAL_LOCK_DIR` after successful eval so relaxed/strict watchers do not repeat the same checkpoint. These changes only affect orchestration; they do not change trajectory inference, PDM scoring, or submetric calculation. |
| Train-only elite buffer exploration | keep running in background | Continue improving the navtrain-only oracle buffer, but write it in keep-best merge mode so new exploration cannot overwrite a stronger existing record. Use navtest only for checkpoint diagnosis, never for buffer generation or training reward. |
| Train-only elite buffer validation | passed on 2026-06-14 | Full strict-v2 validation of `/mnt/project/VLA-AD/cache/recogdrive_stage3_awac_elite_buffer_train_v2_stage3_awac_iql_dualhost_20260612T182947Z` passed with `85109` records, valid-candidate ratio `0.982056`, has-valid ratio `1.0`, mean best-valid reward `0.973614`, `pct_best_valid_above_gt=0.582477`, and `pct_best_valid_above_il=0.792349`. Summary files: `stage3_awac_keepbest_buffer_zt3_wait_20260614T142658Z/validate_live_summary_20260614T2015Z.json` and `.csv`. This supports using the buffer for future train-only Buffer-DPO/preference absorption; it does not by itself prove sampler improvement. |

## Artifact Retention And Cleanup

Policy:
- Keep summary evidence in this ledger before deleting large artifacts.
- Keep current training runs, current evaluation watchers, train-only elite buffers, and strong reproducible baselines unless explicitly superseded.
- Remove large checkpoints, Lightning logs, and duplicate evaluation shard outputs for failed, invalid, or smoke-only attempts after their result and failure mode are recorded here.
- Do not use deleted navtest artifacts as training data; navtest remains diagnostic only.

Cleanup on 2026-06-14 UTC:
- Freed about `300+ GB` under `/mnt/project/VLA-AD/outputs`, increasing available `/mnt/project` space from about `93 GB` to over `400 GB`.
- Removed obsolete AWAC/IQL outputs whose results are already listed in the completed AWAC ledger: `stage3_awac_iql_dualhost_20260612T182947Z`, `stage3_awac_iql_bc015_20260613T0648Z`, `stage3_awac_iql_pref_rank_ttc_20260613T112709Z`, `stage3_awac_warmup_absddc_blend025_lownoise_online_20260613T183113Z`, `stage3_awac_lownoise_dpo3gpu_20260613T163215Z`, `stage3_awac_blend035_lownoise_dpo8gpu_20260613T164458Z`, `stage3_awac_gapdpo_hybrid4gpu_v2_20260613T155350Z`, and `stage3_awac_gapdpo_lowvalid_2gpu_20260613T155703Z`.
- Removed obsolete DPPO/replay/high-LR diagnostic outputs whose conclusions are already recorded: `stage3_grpo_dppo_transition_s16_i1_lr1e4_zt2_2gpu_20260614T083005Z`, `stage3_dppo_transition_smoke2_20260614T082252Z`, `stage3_grpo_replay_stepmode_stepckpt_s16_i1_lr1e4_zt3_2gpu_20260614T0727Z`, `stage3_grpo_replay_stepckpt_s16_i1_lr1e4_zt3_2gpu_20260614T0610Z`, `stage3_grpo_replay_diag_s4_i2_lr1e4_4gpu_20260614T042519Z`, `stage3_grpo_replay_diag_s16_i2_lr1e4_8gpu_20260614T043345Z`, `stage3_grpo_replay_stepmode_smoke_1gpu_20260614T0530`, `stage3_grpo_replay_smoke_1gpu_nullclip_20260614T041722Z`, `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_diag_zt2_4gpu_20260614T024536Z`, `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_log1_diag_zt2_4gpu_20260614T0300Z`, `stage3_grpo_rloo_selfimit_smoke_20260614T_check`, and `stage3_grpo_refkl_s16_lr2e4_b4acc2_chunk32_fast_8gpu_20260613T213145Z`.
- Preserved the then-current safetygate/main-GRPO successor chain, the zt3 original-LR GRPO control, the zt3 keep-best train-only buffer builder, and the cap05 summaries/checkpoints referenced by the keep-best buffer process.

Second cleanup on 2026-06-14 UTC:
- User approved deleting obsolete results/logs once the failed attempt is recorded here.
- Deleted remaining local dry-run/profile/benchmark residues and low-information launch leftovers, including `exact_pdm_profile_awac_epoch0_32`, `exact_pdm_profile_awac_epoch0_32_retry`, `pdm_exact_benchmarks`, `pdm_bench_exact`, `stage3_rl_dry_run_check`, `stage3_rl_safe_diffgrpo_dry_run_check`, `stage3_safe_diffgrpo_eval_dry_run_check`, and `stage3_rl_2b_safe_diffgrpo_relaunch_dry_run`.
- Deleted failed or superseded Stage3 diagnostic residues whose conclusions are already captured above: `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_diag_zt2_4gpu_20260614T023758Z`, `stage3_grpo_dppo_transition_s16_i1_lr1e4_zt2_2gpu_20260614T083005Z_navtest_exact_shards_step300_20260614T0852Z`, `stage3_grpo_dppo_transition_s16_i1_lr1e4_zt2_2gpu_20260614T083005Z_navtest_exact_shards_step300_20260614T0857Z`, `stage3_grpo_dppo_transition_step600_zt3_eval_root.txt`, `stage3_grpo_replay_diag_s4_i2_lr1e4_4gpu_20260614T042322Z`, `stage3_grpo_refkl_s16_lr2e4_8gpu_20260613T204941Z`, `stage3_grpo_refkl_s16_lr2e4_b4acc2_8gpu_20260613T205949Z`, `stage3_grpo_refkl_s16_lr2e4_b8_chunk32_fast_8gpu_20260613T212548Z`, `stage3_rl_2b_safe_diffgrpo_online_20260613T221859Z`, and `stage3_rl_2b_safe_diffgrpo_online_20260613T224055Z`.
- Deleted `/mnt/project/VLA-AD/outputs/hidden_cache_equiv_navtest2`, which was previously inventoried as old smoke/equivalence output and is not a dependency of current Stage3 training, current Stage3 evaluation, the train-only elite buffer, or the running two-expert job.
- Deleted stale local Stage3 supervisor test logs and status files after confirming the recorded supervisor pid was no longer alive. The auto-supervisor remains disabled by policy because it can relaunch stale actions without the reference-audit protocol.
- Preserved active or decision-critical artifacts: the current main hard-gate v3 GRPO run, the zt3 original-LR GRPO control, the zt3 keep-best train-only elite-buffer generator, the Safe DiffGRPO best-result directory, and the cap05 run summaries/checkpoints used for diagnosis.
- Do not repeat the deleted attempts as default routes. Re-opening any of them requires a fresh planned-attempt entry with a reference audit, a mature implementation checklist, matched-step controls, and explicit NC/DAC/TTC/DDC diagnostics.

Third cleanup on 2026-06-14 UTC:
- User approved deleting obsolete failed-attempt results/logs after recording the conclusion in this ledger.
- Deleted redundant cap05 navtest watcher artifacts, including `stage3_grpo_rloo_selfimit_cap05_step300_s16_lr1e4_b2acc4_8gpu_20260614T111511Z/manual_retry_step300_zt3_2gpu_20260614T130727Z`, `secondary_watch_on_rl_zt3_memfit_4gpu`, and `unique_lock_watch_on_vla_zt2_4gpu`. Their step300/600/900 PDMS and submetric conclusions are preserved in this document and in the compact top-level cap05 analysis files.
- Preserved the cap05 training `step_checkpoints` directory because the zt3 keep-best train-only buffer generator currently references it as an automatic policy-checkpoint source.
- Do not repeat the cap05 broad self-imitation configuration as a default route. Its failure mode is already diagnosed: it can increase EP/high-score bins, but it does not reliably improve mean PDMS because NC/TTC/DDC regressions offset the progress gain. Any future self-imitation run must include main-objective safety gating or a stronger preference objective, and must be judged at matched step/epoch against original Stage3 early PDMS.

Fourth cleanup on 2026-06-14 UTC:
- Stopped the local orphaned v3 `torchrun`/worker process tree after the stable launcher exited but the workers kept 8 GPUs allocated. Confirmed local GPU memory returned to `0 MB` on all 8 cards.
- Deleted the failed v3 run directory `stage3_grpo_rloo_selfimit_mainhardgate_v3_bufferpath_step300_s16_lr1e4_b2acc4_8gpu_20260614T161426Z` after verifying no corresponding watcher was still active on `training-vla-zt2` or `training-rl-zt3`.
- The retained evidence for this failed direction is the row/result summary in this ledger: step300 PDMS `0.879395`, step600 PDMS `0.882934`, and the failure mode that main TTC/DDC hard gates improved safety but suppressed EP/DAC enough to underperform cap05 and original-LR GRPO.
- Do not repeat the v3 main TTC/DDC hard-gate plus strict self-imitation setup as a default route. Any future safety/progress method must preserve progress pressure and be compared against the zt3 original-LR control at matched step/epoch.

Fifth cleanup on 2026-06-14 UTC:
- Removed the final failed-v3 residue after it was recreated by a stale local `watch_recogdrive_stage3_early_gate.sh` process. The stale watcher PID `3706056` had `RUN_ROOT` set to `stage3_grpo_rloo_selfimit_mainhardgate_v3_bufferpath_step300_s16_lr1e4_b2acc4_8gpu_20260614T161426Z` and only wrote `early_gate_latest.json` / `early_gate_watch.log`; it was stopped and the regenerated 12 KB directory was deleted.
- No remote `training-vla-zt2` or `training-rl-zt3` tasks were killed during this cleanup. The only remaining local early-gate watcher is attached to the active current-repo original-LR GRPO control run.
- Keep only this ledger entry as evidence for the failed v3 attempt. Do not restore or re-evaluate deleted v3 logs unless a new planned-attempt entry justifies a different safety/progress design.

Sixth cleanup on 2026-06-14 UTC:
- User approved deleting obsolete results/logs after recording failed or superseded attempts in this ledger.
- Deleted about `46 GB` of old Last-VLA / Stage2 result and eval artifacts under `/mnt/project/VLA-AD/outputs/last_vla_v2`, including obsolete `highcap_no_risk`, `decoupled_highcap_no_risk`, `direct_text_navtest`, `no_residual_stage2_ab`, `vlm_text_residual_anchor`, and `residual_anchor_*` directories. These were superseded by the current two-expert random-HMEF work and are not part of the ReCogDrive Stage3 GRPO / Buffer-DPO plan.
- Preserved `/mnt/project/VLA-AD/outputs/last_vla_v2/navtrain_pdms_val6000_probe_20260611T0315Z`, because the active two-expert watcher still references its token file. Also preserved small audit/VQA residues that are not meaningful space consumers.
- Deleted obsolete Last-VLA hidden-cache subtrees under `/mnt/project/VLA-AD/cache/last_vla_v2`: `highcap_no_risk`, `experiments`, and `decoupled_highcap_no_risk`. No active process referenced this cache. Preserved only `teacher_traj` and `manifests`.
- Result: `/mnt/project` available space increased from about `2.4 TB` to about `4.4 TB`; `/mnt/project/VLA-AD/cache/last_vla_v2` dropped to `24 KB`.
- No active Stage3 training/evaluation, remote zt2/zt3 task, train-only elite buffer, metric cache, or active two-expert run was killed or deleted.
- Do not restore or relaunch the deleted Last-VLA / Stage2 intermediate routes as default experiments. Reusing any of those directions requires a new planned-attempt entry with a current reference audit, matched-step controls, and explicit navtrain/navtest split handling.

Seventh cleanup on 2026-06-14 UTC:
- User approved deleting obsolete results/logs after recording failed attempts here.
- Current-repo original-LR GRPO control step300 exact navtest was successfully evaluated by the local foreground exact-pool path, not by the failed background wrapper attempts:
  - PDMS `0.881910`, NC `0.984800`, DAC `0.969682`, TTC `0.957571`, EP `0.805231`, comfort `0.999918`, DDC `0.965316`.
  - Result root kept: `stage3_grpo_refkl_s16_lr1e4_b2acc4_currentrepo_8gpu_20260614T191227Z/local_foreground_eval_step300_4gpu`.
- The following local eval-launch residues are obsolete and should not be repeated as default evaluation routes: `local_relaxed_watch_shared_4gpu`, `local_direct_eval_step300_shared_4gpu`, `local_direct_eval_step300_flock_4gpu`, `local_direct_eval_step300_execflock_4gpu`, and `local_direct_eval_step300_nolock_4gpu`.
- Failure modes: SIGHUP-prone background wrapper, launcher-wrapper mismatch with eval scripts that end in `exec`, and duplicate hardlinked checkpoint archives. The correct path for this run is either the existing checkpoint watcher with global done markers, or a directly supervised foreground/nohup exact eval whose result directory is retained.
- Updated the global step300 done marker so its archive path points to the real training checkpoint instead of the soon-deleted failed wrapper archive.
- Removed stale cap05 runtime-status residue (`pids`, `status`, and old pid files) after verifying no cap05 training process is alive. Preserved its `train/step_checkpoints` directory because the zt3 keep-best train-only buffer generator still references it, and preserved compact cap05 navtest analysis files.
- Deleted the failed old Safe DiffGRPO relaunch residue `stage3_rl_2b_safe_diffgrpo_online_relaunch_20260609T1944Z` and stale local supervisor latest log/status files. Preserved the actual Safe DiffGRPO historical-best directory `stage3_safe_diffgrpo_ckpt_stream_eval_live_20260610T030355Z`.
- Preserved active training, zt2/zt3 watcher directories, successful foreground eval outputs, the train-only elite buffer, and all strong baseline summaries. No remote task was killed.

## Navtest Diagnostic: Cap05 Self-Imitation Step300 To Step600

Run:
`/mnt/project/VLA-AD/outputs/stage3_grpo_rloo_selfimit_cap05_step300_s16_lr1e4_b2acc4_8gpu_20260614T111511Z`.

This analysis uses navtest evaluation outputs only. It is diagnostic evidence for
algorithm selection and must not be used as a training reward/cache or buffer
source.

Grouped checkpoint summary:

| Checkpoint | Evals | PDMS | NC | DAC | TTC | EP | Comfort | DDC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `step-step_300` | 2 | `0.885432` | `0.981237` | `0.969435` | `0.946326` | `0.826787` | `1.000000` | `0.969023` |
| `step-step_600` | 3 | `0.885497` | `0.976039` | `0.968831` | `0.939941` | `0.835866` | `1.000000` | `0.959068` |
| `step-step_900` | 1 | `0.887107` | `0.984264` | `0.967210` | `0.955841` | `0.823058` | `1.000000` | `0.971742` |

Paired token-level analysis on the zt3 step300 and step600 CSVs:

- Paired valid scenarios: `12139`.
- Mean PDMS delta: `+0.000349`; median delta `0.0`.
- Improved by more than `0.01` PDMS: `3936` scenes (`32.42%`).
- Degraded by more than `0.01` PDMS: `1322` scenes (`10.89%`).
- PDMS-zero scenes increased from `583` to `649`.
- High-score bins improved: `(.95,.99]` increased by `141`, and `(.99,1]` increased by `441`.
- Mid-score bins shrank: `(.8,.88]` decreased by `278`, `(.88,.9]` by `131`, and `(.9,.95]` by `224`.

Safety failure movement under practical gates NC `1.0`, DAC `1.0`,
TTC `0.95`, DDC `0.99`:

| Metric | Fails at step300 | Fails at step600 | New fails | Recovered |
|---|---:|---:|---:|---:|
| NC | `237` | `298` | `125` | `64` |
| DAC | `371` | `380` | `140` | `131` |
| TTC | `664` | `733` | `253` | `184` |
| DDC | `509` | `650` | `226` | `85` |

Interpretation:

- The policy is learning some higher-progress/high-PDMS trajectories: many scenes
  move into the `0.95+` and `0.99+` PDMS bins.
- The mean does not improve because safety regressions also increase. EP rises
  by about `+0.009`, while NC falls `-0.0053`, TTC `-0.0064`, and DDC `-0.0099`.
- This matches the training-path flaw: the self-imitation target selector used
  reward/margin plus broad hard safety, but did not require TTC/DDC-safe targets.
- Next runs must treat NC/DAC/TTC/DDC as target-validity gates for auxiliary
  diffusion regression. Otherwise the auxiliary path can absorb high-progress
  trajectories that hurt the same safety submetrics which dominate navtest PDMS.
- `step-step_900` partially recovered NC/TTC/DDC and improved PDMS to `0.887107`,
  but this came with EP dropping from about `0.8359` to `0.8231`. This indicates
  the run is oscillating between progress and safety rather than producing a
  stable Pareto improvement. It remains only `+0.007107` over the early gate and
  is still `-0.018393` below the original 10-epoch `0.9055` reference.

Artifacts:

- Summary: `/mnt/project/VLA-AD/outputs/stage3_grpo_rloo_selfimit_cap05_step300_s16_lr1e4_b2acc4_8gpu_20260614T111511Z/navtest_pdms_analysis.md`
- Pairwise analysis: `/mnt/project/VLA-AD/outputs/stage3_grpo_rloo_selfimit_cap05_step300_s16_lr1e4_b2acc4_8gpu_20260614T111511Z/navtest_pairwise_step300_step600_analysis.md`
- Worst paired deltas: `/mnt/project/VLA-AD/outputs/stage3_grpo_rloo_selfimit_cap05_step300_s16_lr1e4_b2acc4_8gpu_20260614T111511Z/navtest_pairwise_step300_step600_worst.tsv`

## Current Baselines And Controls

| Run / Reference | Status | Key Config | Best Known PDMS | Notes |
|---|---:|---|---:|---|
| User-reported original local Stage3 RL | historical | original LR `1e-4`, 10 epochs | `0.9055` | Treat this as a same-length/final-training reference, not a 1-epoch diagnostic reference. |
| User-reported original Stage3 early training | historical | original Stage3, `epoch0-1` | `0.88+` | Treat this as the short-run sanity gate. A new Stage3 method that is already clearly below this range at matched early epoch/step should not be promoted without a method-level fix. |
| `stage3_safe_diffgrpo_ckpt_stream_eval_live_20260610T030355Z` | completed | Safe DiffGRPO | `0.906184` at `epoch_12-step_17290` | Strongest confirmed Stage3 result so far. Compare against it only at comparable training length, or label the comparison as short-run diagnostic only. |
| `stage3_grpo_refkl_s16_lr2e4_b4acc2_chunk32_fast_8gpu_20260613T213145Z` | stopped/evaluated | LR `2e-4`, sample_time `16`, BC `0.10->0.05`, ref KL `0.02`, effective batch about `64` | `0.872059` at `epoch_0-step_1330` | Exact 8-shard navtest eval: NC `0.9750`, DAC `0.9619`, TTC `0.9371`, EP `0.8159`, comfort `1.0000`, DDC `0.9459`. This is well below `0.9055/0.906184`, so the higher LR was harmful or at least not sufficient. |
| `stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z` | evaluated | LR `1e-4`, sample_time `16`, BC `0.10->0.05`, ref KL `0.02`, effective batch about `66` | `0.896794` at `epoch_0-step_1290` | zt3 original-LR control. Submetrics: NC `0.987189`, DAC `0.975861`, TTC `0.962597`, EP `0.826516`, comfort `1.000000`, DDC `0.974378`. This is the strongest recent early checkpoint and argues against aggressive safety/self-imitation changes that suppress progress. |
| `stage3_grpo_refkl_s16_lr1e4_b2acc4_currentrepo_8gpu_20260614T191227Z` | running/evaluated step300 | LR `1e-4`, sample_time `16`, BC `0.10->0.05`, ref KL `0.02`, effective batch `64`, `GRPO_USE_GSPO_RATIO=false`, no buffer/self-imitation/hard gates | `0.881910` at `step-step_300` | Purpose: reproduce the strong original-LR GRPO path on the current repository before launching another algorithmic variant. Step300 is in the original early-training band but below the zt3 original-LR control step1290 `0.896794`; do not over-interpret this as an algorithmic improvement. |
| `stage3_grpo_rloo_selfimit_cap05_step300_s16_lr1e4_b2acc4_8gpu_20260614T111511Z` | stopped | LR `1e-4`, sample_time `16`, BC `0.10->0.05`, ref KL `0.02`, GSPO ratio, RLOO self-imitation, max target scene ratio `0.5`, step ckpt every `300` | `0.887107` at `step-step_900` | Step300 passed the early gate (`0.885557`). Step600 was flat and traded EP for safety loss. Step900 recovered NC/TTC/DDC and reached `0.887107`, but EP fell to `0.823058`; this is still `-0.018393` below original 10-epoch `0.9055`. |
| `stage3_grpo_rloo_selfimit_safetygate_step300_s16_lr1e4_b2acc4_8gpu_20260614T145712Z` | stopped/deleted | Same as cap05, with explicit self-imitation target gates NC `1.0`, DAC `1.0`, TTC `0.95`, DDC `0.99`, but no main TTC/DDC hard gate | none | Stopped before checkpoint. This was lower-information than the main hard-gate test because it only filtered auxiliary targets. |
| `stage3_grpo_rloo_selfimit_mainhardgate_v3_bufferpath_step300_s16_lr1e4_b2acc4_8gpu_20260614T161426Z` | stopped/delete artifacts | Same LR/sample/effective batch as cap05, with main TTC `0.95` and DDC `0.99` hard gates plus strict self-imitation targets | `0.882934` at `step-step_600` | Failed matched early comparison. Step600 improved safety over cap05 but EP fell to `0.792348`; PDMS stayed below cap05 step900 and below the zt3 original-LR control. |
| `stage3_grpo_buffer_guided_gspo_s16_lr2e4_b2acc8_zt2_4gpu_20260614T012106Z` | stopped before eval | LR `2e-4`, GSPO ratio, elite-buffer reward bonus/distill/self-imitation | invalid run | Stopped because key diagnostics were not logged, so the run could not validate buffer absorption. |
| `stage3_grpo_buffer_guided_gspo_s16_lr2e4_b2acc8_diagfix_zt2_4gpu_20260614T020706Z` | stopped before eval | LR `2e-4`, GSPO ratio, elite-buffer reward bonus/distill/self-imitation | invalid run | Stopped because it was launched before the full PPO/advantage diagnostics patch. |
| `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_diag_zt2_4gpu_20260614T023758Z` | failed before training | LR `1e-4`, GSPO ratio, batch advantage normalize, fixed advantage clip `3.0`, no buffer | none | Failed at Hydra config parsing because `grpo_normalize_advantage_batch` was missing from `recogdrive_agent.yaml`. |
| `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_diag_zt2_4gpu_20260614T024536Z` | stopped/replaced | LR `1e-4`, GSPO ratio, batch advantage normalize, fixed advantage clip `3.0`, no buffer, `limit_train_batches=80`, `limit_val_batches=0`, default logging | none | Wrote only `lr-AdamW`; insufficient diagnostic density. |
| `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_log1_diag_zt2_4gpu_20260614T0300Z` | completed diagnostic | LR `1e-4`, GSPO ratio, batch advantage normalize, fixed advantage clip `3.0`, no buffer, `limit_train_batches=40`, `limit_val_batches=0`, `log_every_n_steps=1`, behavior sync interval `100000` | no eval | Ratio/KL/clip diagnostics moved after optimizer updates; this validates instrumentation and short-run GSPO activity only. |
| `stage3_grpo_replay_diag_s4_i2_lr1e4_4gpu_20260614T042519Z` | completed smoke | LR `1e-4`, `stage3_objective=grpo_replay`, `sample_time=4`, inner PPO epochs `2`, replay minibatch `4`, 4 GPUs, 40 train batches, no buffer | no eval | End-to-end PPO replay path is active: fixed old logprob replay, manual optimization, PPO ratio diagnostics, reference KL, and BC update ran without NaNs. Too small/conservative to evaluate as a main result. |
| `stage3_grpo_replay_diag_s16_i2_lr1e4_8gpu_20260614T043345Z` | completed diagnostic | LR `1e-4`, `stage3_objective=grpo_replay`, `sample_time=16`, inner PPO epochs `2`, replay minibatch `8`, 8 GPUs, 40 train batches, no buffer | no eval | Near-full sample-time diagnostic passed. Checkpoint `epoch=0-step=360.ckpt`; keep as implementation evidence only, not as a final PDMS result. |
| `stage3_grpo_replay_1epoch_s16_i2_lr1e4_8gpu_20260614T044444Z` | running | LR `1e-4`, `stage3_objective=grpo_replay`, `sample_time=16`, inner PPO epochs `2`, replay minibatch `8`, 8 GPUs, full navtrain, no buffer | pending | Latest logged step `339`: `valid_ratio=1.0`, ratio mean `0.9694`, clip fraction `0.2031`, KL `7.87e-4`; latest rollout safety was weak (`safe_ratio=0.6133`), so this remains a watched diagnostic. |
| `stage3_grpo_replay_1epoch_s16_i1_lr1e4_zt2_8gpu_20260614T050122Z` | running | LR `1e-4`, `stage3_objective=grpo_replay`, `sample_time=16`, inner PPO epochs `1`, replay minibatch `8`, 8 zt2 GPUs, full navtrain, no buffer | pending | Motivation: update-strength ablation from i2. Latest logged step `179`: `valid_ratio=1.0`, ratio mean `0.9992`, clip fraction `0`, KL `2.30e-6`, safe ratio `0.8281`. |

Current-repo original-LR control status on 2026-06-14 19:23 UTC:
- Run root: `/mnt/project/VLA-AD/outputs/stage3_grpo_refkl_s16_lr1e4_b2acc4_currentrepo_8gpu_20260614T191227Z`.
- Training is running on local GPUs `0-7`; monitor reported about `40.9 GB` per GPU and high SM utilization after SceneLoader finished.
- The resolved command confirms no buffer, no DPO, no self-imitation, no hard TTC/DDC gate, and `GRPO_USE_GSPO_RATIO=false`.
- TensorBoard had only `lr-AdamW` at step `0` at this timestamp; no reward/loss scalar and no checkpoint yet. This is expected before the first GRPO/PDM training log window completes.
- zt2 and zt3 checkpoint watchers are attached but only waiting for checkpoint files; they are configured to wait for free GPUs and not preempt existing remote tasks.

Update on 2026-06-14 19:34 UTC:
- The same current-repo control has entered normal training and logged step `49`:
  - `train/reward_step=0.785141`
  - `train/base_reward_step=0.789642`
  - `train/safe_ratio_step=0.914062`
  - `train/mean_ep_step=0.769345`
  - `train/mean_ttc_step=0.898438`
  - `train/mean_ddc_step=0.939453`
  - `train/reference_kl_loss_step=0.007989`
  - `train/bc_coeff_step=0.1`
- No checkpoint or navtest eval row exists yet for this run. Keep it running as the current-code control before launching the buffer-DPO diagnostic.

Update on 2026-06-14 19:51 UTC:
- The run is still alive on local GPUs `0-7`; GPU memory is about `41 GB/card` and utilization is active.
- Latest TensorBoard train scalar is step `99`:
  - `train/reward_step=0.694921`
  - `train/base_reward_step=0.722842`
  - `train/safe_ratio_step=0.875000`
  - `train/mean_ep_step=0.745482`
  - `train/mean_nc_step=0.945312`
  - `train/mean_dac_step=0.937500`
  - `train/mean_ttc_step=0.820312`
  - `train/mean_ddc_step=0.994141`
- Still no `step300` checkpoint and no navtest `checkpoint_eval_submetrics.tsv` row. The zt2/zt3 checkpoint watchers are healthy and repeatedly report `no pending checkpoint evals; training_state=running`.
- Decision unchanged: do not start buffer-DPO until this current-code control has an early exact navtest row.

Update on 2026-06-14 21:20 UTC:
- `step-step_300` exact navtest eval completed through the local foreground exact-pool path:
  - PDMS `0.881910`
  - NC `0.984800`, DAC `0.969682`, TTC `0.957571`, EP `0.805231`, comfort `0.999918`, DDC `0.965316`
  - CSV: `/mnt/project/VLA-AD/outputs/stage3_grpo_refkl_s16_lr1e4_b2acc4_currentrepo_8gpu_20260614T191227Z/local_foreground_eval_step300_4gpu/eval_step-step_300/hydra/stage3_safe_diffgrpo_eval_exact_pool_pdm/2026.06.14.20.44.29/2026.06.14.21.14.37.csv`
- zt2/zt3 watchers saw the global `step-step_300.done` marker and skipped duplicate evaluation. Keep them attached for future checkpoints, but do not repeat the failed local background wrapper attempts listed in the cleanup section.
- Decision gate now recommends the control-aligned Buffer-DPO attempt as the next algorithmic run. The reason is not that the buffer is bad: strict train-only buffer validation is strong. The reason is that the current GRPO-control step300 is below the stronger zt3 original-LR control, so any buffer absorption experiment must be isolated and compared against this control shape.

Update on 2026-06-14 21:33 UTC:
- Current-repo GRPO control is still running locally on GPUs `0-7`; latest summary step is `549` with train reward `0.840181` and safe ratio `0.929688`. No new checkpoint beyond `step-step_300` exists yet.
- zt2 currently has all 8 GPUs occupied by an existing two-expert training job. A Buffer-DPO training launcher was therefore queued with strict wait thresholds rather than starting on top of that job:
  - Run root: `/mnt/project/VLA-AD/outputs/stage3_grpo_buffer_dpo_refctrl_s16_lr1e4_b2acc4_zt2wait_8gpu_20260614T213001Z`.
  - Remote waiting process on `training-vla-zt2`: shell PID `2647910`, launcher PID `2647912`.
  - Training wait gate: target GPUs `0,1,2,3,4,5,6,7`, `TRAIN_GPU_MAX_MEM_USED_MB=2000`, `TRAIN_GPU_MAX_UTIL=5`.
  - Eval watcher wait gate is also strict: `EVAL_GPU_MAX_MEM_USED_MB=2000`, `EVAL_GPU_MAX_UTIL=5`, so navtest eval will not start on busy remote GPUs.
  - Config file: `strict_gspo_launch_config.txt`.
- Immediate strict-v2 buffer validation before launch still passed:
  - `num_records=85109`, `valid_candidate_ratio=0.982654`, `has_valid_candidate_ratio=1.0`.
  - `mean_best_valid_reward=0.974021`, `pct_best_valid_above_gt=0.582594`, `pct_best_valid_above_il=0.793641`.
  - Saved under the run root as `buffer_validate_before_launch.json` and `.csv`.

Train-only keep-best elite buffer status on 2026-06-14 19:23 UTC:
- Buffer path: `/mnt/project/VLA-AD/cache/recogdrive_stage3_awac_elite_buffer_train_v2_stage3_awac_iql_dualhost_20260612T182947Z`.
- Full train-token coverage exists: `85109` `*.pkl.xz` records, about `342 MB`.
- The keep-best zt3 builder is still running and updating records; the latest file mtimes were current at 19:21 UTC.
- A read-only random sample of `5000` records showed:
  - schema version: all v2; missing required fields: `0`; load/shape errors: `0`.
  - mean selected candidates per record: `9.2222`.
  - mean selected valid ratio: `0.980509`; has-valid-candidate ratio: `1.0`.
  - mean GT reward: `0.931496`; mean IL reward: `0.700525`.
  - mean raw best reward: `0.973363`; mean valid best reward: `0.973272`.
  - mean valid best minus GT: `+0.041776`; mean valid best minus IL: `+0.272747`.
  - `pct_best_valid_above_gt=0.587`; `pct_best_valid_above_il=0.802`.
  - valid sources are dominated by structured perturbations: `progress_endpoint`, `progress_speed`, `progress_gamma`, plus policy/lateral/timing candidates.
- Interpretation: the oracle/buffer side is not the main current bottleneck. The buffer contains many valid trajectories that beat GT, so the next algorithmic question remains policy absorption into the diffusion sampler. Any buffer-DPO/preference run must be isolated on top of the original-LR GRPO control shape, not on top of the failed v3 hard-gate configuration.

Update on 2026-06-14 19:51 UTC:
- The keep-best builder is still active on `training-rl-zt3`; shard logs updated at 19:38 and 19:41 UTC.
- Buffer coverage remains `85109` records, about `342 MB`, with latest record mtimes around 19:44 UTC. This means the full train-only buffer is available and still being refreshed; no intervention is needed.

## Planned Attempt: Original-LR GRPO With RLOO Self-Imitation

Motivation:
- User correction: original Stage3 `epoch0-1` is already around `0.88+` PDMS, so the transition replay `epoch0=0.831015` is a failed early gate, not a training-length issue.
- The AWAC elite buffer is good as an oracle (`mean_best_valid_reward=0.971664`, `pct_best_valid_above_gt=0.581537`), but pure weighted denoising did not transfer to sampled trajectories.
- The strongest confirmed family remains on-policy GRPO/Safe DiffGRPO, so the next change must preserve the original GRPO objective and `1e-4` LR.

Reference audit:
- DPPO paper/code: https://arxiv.org/abs/2409.00588 and https://github.com/irom-princeton/dppo. Mechanism checked: diffusion-policy PG should optimize sampled actions with old-policy logprobs/ratio clipping; reward-weighted regression is not enough by itself.
- DDPO paper: https://arxiv.org/abs/2305.13301. Mechanism checked: denoising can be treated as a multi-step decision process, which supports direct policy-gradient optimization over diffusion samples.
- RIPT-VLA code: https://github.com/Ariostgx/ript-vla. Mechanism checked: K-rollout, leave-one-out advantage, dynamic sampling, and no value net are used for VLA post-training.

Implementation:
- Add `group_leave_one_out` to `offline_rl_grpo_self_imitation_baseline_mode`.
- Keep `stage3_objective=grpo`, `LR=1e-4`, `sample_time=16`, BC anneal `0.10 -> 0.05`, reference KL `0.02`, GSPO ratio on.
- Add only a small auxiliary self-imitation loss:
  - `GRPO_SELF_IMITATION_LOSS_WEIGHT=0.01`
  - `GRPO_SELF_IMITATION_LOSS_SCHEDULE=linear_warmup`
  - `GRPO_SELF_IMITATION_WARMUP_EPOCHS=2`
  - `GRPO_SELF_IMITATION_MIN_REWARD=0.88`
  - `GRPO_SELF_IMITATION_MIN_REWARD_MARGIN=0.01`
  - `GRPO_SELF_IMITATION_BASELINE_MODE=group_leave_one_out`
  - `GRPO_SELF_IMITATION_TIMESTEP_SAMPLING=low_noise`
- Do not enable buffer reward bonus or buffer distill in this first run. The offline buffer can remain available for diagnostics, but it must not dominate the update.

Smoke status:
- `stage3_grpo_rloo_selfimit_smoke_20260614T_check` completed 1 train batch on 1 GPU with `sample_time=2`.
- Hydra accepted `offline_rl_grpo_self_imitation_baseline_mode=group_leave_one_out`.
- Logged `grpo_self_imitation_enabled=1.0`, `baseline_from_buffer=0.0`, `use_gspo_ratio=1.0`.
- This smoke batch had `base_reward=0.625077`, below the `0.88` imitation threshold, so `target_ratio=0.0` and `zero_weight_batch=1.0`; this is expected for a low-reward smoke batch and confirms that low-quality samples are not imitated.

Live full-run status:
- Run root: `/mnt/project/VLA-AD/outputs/stage3_grpo_rloo_selfimit_s16_lr1e4_b2acc4_8gpu_20260614T101257Z`.
- Training host/GPU: local `training-vla-zt`, GPUs `0-7`.
- Eval watchers: two `training-vla-zt2` watcher processes are active and waiting for epoch checkpoints.
- Additional eval watcher: `training-rl-zt3` pid `918395`, GPUs `6,7`, strict free-GPU wait, same `global_checkpoint_eval_locks`, added because zt2 GPUs were occupied by an unrelated existing run.
- Config keeps original-LR GRPO/Safe DiffGRPO defaults: `LR=1e-4`, `sample_time=16`, effective batch about `64`, BC `0.10 -> 0.05`, reference KL `0.02`, GSPO ratio on.
- First logged train scalar at `step=49`:
  - `reward=0.797566`, `base_reward=0.792112`, `safe_ratio=0.933594`, `mean_ep=0.773152`, `mean_ttc=0.902344`.
  - `grpo_self_imitation_enabled=1.0`, `weight=0.005` during warmup.
  - `grpo_self_imitation_candidate_ratio=0.429688`, `target_ratio=0.875000`, `target_reward_mean=0.976924`.
- Second logged train scalar at `step=99`:
  - `reward=0.731460`, `base_reward=0.750897`, `safe_ratio=0.890625`, `mean_ep=0.768155`, `mean_ttc=0.820312`.
  - `grpo_self_imitation_candidate_ratio=0.378906`, `target_ratio=0.875000`, `target_reward_mean=0.969151`.
- Third logged train scalar at `step=149`:
  - `reward=0.762590`, `base_reward=0.770046`, `safe_ratio=0.906250`.
  - `grpo_self_imitation_target_ratio=0.937500`, `target_reward_mean=0.977498`.
- Interpretation: the self-imitation path is active and is selecting very high train-reward sampled trajectories rather than imitating low-quality rollouts. However `target_ratio` is `0.875`, `0.875`, then `0.9375` on the first three logged batches, so the auxiliary selection is not sparse. If this run fails navtest despite high target rewards, the first algorithmic fix should be a stricter self-imitation gate or rank/quantile cap, not another LR-only change. This is still only an internal train diagnostic; promotion requires exact navtest PDMS at `epoch0` near or above the original Stage3 early `0.88+` band.

Expected diagnostics:
- `grpo_self_imitation_target_ratio` should be non-zero but sparse; a zero ratio for many steps means the gate is too strict.
- `grpo_self_imitation_target_reward_mean` should be above `0.88`.
- `grpo_self_imitation_zero_weight_batch` should not stay at `1.0` after warmup if the policy finds good samples.
- Main GRPO diagnostics must remain healthy: `gspo_ratio_mean` near `1`, finite small `gspo_approx_kl`, no explosion in `safe_ratio` or hard safety metrics.

Follow-up implementation after live diagnostics:
- Evidence from the first logged full-run batches showed `grpo_self_imitation_target_ratio` was `0.875`, `0.875`, then `0.9375`, while `target_reward_mean` stayed high around `0.969-0.977`.
- This means the on-policy target quality is good, but the self-imitation gate is too broad to behave like sparse elite imitation. It risks turning the auxiliary loss into a broad reward-weighted BC signal, which is exactly the failure mode observed in earlier offline AWAC-style attempts.
- Added configurable batch-level scene cap:
  - `offline_rl_grpo_self_imitation_max_target_scene_ratio` default `1.0` for backward compatibility.
  - `offline_rl_grpo_self_imitation_batch_cap_score` in `{reward, margin}`, default `reward`.
  - Logs `grpo_self_imitation_pre_cap_target_ratio`, `grpo_self_imitation_target_scene_cap_ratio`, and `grpo_self_imitation_target_scene_cap_active`.
- The RLOO launcher default for the next run is `GRPO_SELF_IMITATION_MAX_TARGET_SCENE_RATIO=0.5`; the base GSPO launcher remains `1.0`.
- Synthetic CPU check confirmed a cap of `0.5` reduces `pre_cap_target_ratio=1.0` to `target_ratio=0.5` and keeps the highest-reward scenes.
- Live cap05 run first logged batch on 2026-06-14:
  - `step=49`, reward `0.785556`, base reward `0.784031`, safe ratio `0.925781`, EP `0.772696`, TTC `0.882812`.
  - `grpo_self_imitation_candidate_ratio=0.429688`, `target_ratio=0.500000`, `target_reward_mean=0.983409`, `target_margin_mean=0.162914`.
  - This confirms the scene cap is active and avoids the earlier no-cap target ratios around `0.75-0.94`; promotion still requires step/epoch PDMS.
- Exact navtest gate result on 2026-06-14:
  - `step-step_300` reached PDMS `0.885557`, passing the original Stage3 early `0.88+` sanity gate.
  - Submetrics: NC `0.981134`, DAC `0.969352`, TTC `0.947273`, EP `0.826626`, comfort `1.000000`, DDC `0.968941`.
  - Interpretation: this is the first recent non-original variant to clear the early gate. Continue the run and evaluate `step-step_600`/epoch0 before judging whether the self-imitation cap improves the trend beyond the original Stage3 baseline.
- Exact navtest step600 result on 2026-06-14:
  - `step-step_600` reached PDMS `0.885212`, essentially flat from step300 and still only in the original early-training band.
  - Submetrics: NC `0.975449`, DAC `0.968941`, TTC `0.940105`, EP `0.835322`, comfort `1.000000`, DDC `0.958807`.
  - Delta from step300: EP improved by about `+0.0087`, but NC fell by about `-0.0057`, TTC by about `-0.0072`, and DDC by about `-0.0101`.
  - Interpretation: the auxiliary self-imitation path is likely absorbing higher-progress on-policy samples, but target selection does not explicitly protect TTC/DDC. A further run without fixing target validity would be low-information.

Main hard-gate v3 early result on 2026-06-14:
- Run root: `/mnt/project/VLA-AD/outputs/stage3_grpo_rloo_selfimit_mainhardgate_v3_bufferpath_step300_s16_lr1e4_b2acc4_8gpu_20260614T161426Z`.
- `step-step_300` exact navtest PDMS is `0.879395`, below the `0.88` original early gate but within the configured `0.005` watch margin.
- Submetrics: NC `0.985335`, DAC `0.962679`, TTC `0.961855`, EP `0.803776`, comfort `0.999835`, DDC `0.982699`.
- `step-step_600` exact navtest PDMS is `0.882934`.
- Step600 submetrics: NC `0.990649`, DAC `0.968776`, TTC `0.972730`, EP `0.792348`, comfort `0.999835`, DDC `0.979609`.
- Delta from step300 to step600: NC/TTC/DAC improved, but EP fell by about `-0.0114`, and DDC stayed below the strict target threshold.
- Interpretation: the hard safety gate did what it was asked to do locally, but the objective over-corrected toward safety and did not learn a better progress/safety Pareto point. It underperforms both cap05 step900 (`0.887107`) and the original-LR control at a similar early horizon (`0.896794`), so the run should be stopped and its artifacts deleted after this record. Do not repeat main TTC/DDC hard gating plus strict self-imitation as the default GRPO route.

## Planned Attempt: GRPO With Train-Buffer Diffusion-DPO Absorption

Motivation:
- Past AWAC/IQL attempts show that the train-only elite buffer can discover high-PDMS trajectories, but weighted denoising regression alone did not make the sampler reliably output better trajectories.
- The main hard-gate v3 GRPO run showed that strict hard safety gates can raise NC/TTC while lowering EP enough to reduce PDMS. Therefore buffer preference should not inherit the v3 hard gates by default.
- The next buffer use should therefore not replace GRPO. It should add a small preference signal that tells the diffusion planner: for the same scene/context/noise level, assign lower denoising loss to valid train-buffer winners than to GT/IL behavior losers.

Reference audit:
- Diffusion-DPO paper and official code: https://arxiv.org/abs/2311.12908 and https://github.com/SalesforceAIResearch/DiffusionDPO. Mechanism copied: chosen/rejected samples share the same diffusion timestep and noise; the objective compares current-model loss gap against reference-model loss gap with a sigmoid/DPO classification loss.
- DPPO paper/code: https://arxiv.org/abs/2409.00588 and https://github.com/irom-princeton/dppo. Mechanism retained from the GRPO backbone: keep policy-gradient style optimization over sampled diffusion trajectories rather than relying only on offline regression.
- DGPO code: https://github.com/Luo-Yihong/DGPO. Mechanism borrowed conceptually: group/reward ordering information can be used as direct preference supervision for diffusion models, not only as scalar reward-weighted regression.
- SIPO / SDPO: https://arxiv.org/abs/2505.21893. Caution borrowed: diffusion preference optimization is timestep-sensitive and off-policy biased. The implementation keeps timestep sampling configurable and starts with a small warmup-controlled weight.

Implementation:
- Added GRPO-specific buffer preference-DPO fields under `OfflineRLConfig`, all defaulting to disabled:
  - `grpo_buffer_preference_dpo_loss_weight=0.0`
  - `grpo_buffer_preference_dpo_loss_schedule=linear_warmup`
  - `grpo_buffer_preference_dpo_timestep_sampling=uniform`
  - `grpo_buffer_preference_dpo_beta=8.0`
  - `grpo_buffer_preference_dpo_pair_mode=best_vs_gt_il`
  - `grpo_buffer_preference_dpo_min_reward_gap=0.02`
  - `grpo_buffer_preference_dpo_max_pairs_per_scene=2`
- GRPO buffer guidance now keeps the loaded selected candidates/source codes in memory for the current batch.
- The DPO batch is constructed as:
  - valid buffer target(s) selected from train-only elite buffer are winners;
  - GT trajectory from `action_input.action` is a behavior loser;
  - IL/reference trajectory is loaded from the same buffer when present;
  - GT/IL rows are real loser rows but are not marked valid winners.
- Implementation hardening on 2026-06-14 UTC: old-policy fallback for missing IL losers was removed from the GRPO buffer-DPO path. A sampled fallback IL trajectory would not have a reward recomputed for the exact sampled trajectory and would instead inherit `guidance["il_reward"]` from the buffer record, so it could create mismatched preference pairs. If a future buffer record lacks IL support, that IL loser row is simply not marked real; GT remains available as the behavior loser.
- Smoke coverage added: `scripts/smoke_test_recogdrive_awac_iql.py` now checks a two-row GRPO buffer-DPO target batch where one scene has stored IL support and one scene does not. The missing-IL row must keep the IL column `real_mask=False`, proving the code does not synthesize an unmatched IL loser.
- The existing diffusion-DPO loss is reused, including shared noise/timestep and reference-policy loss gap. Candidate trajectories are denoising targets only; `_prepare_dit_context(..., allow_target_tokens=False)` remains enforced through `_diffusion_per_target_loss_on_targets`.
- Added training logs:
  - `grpo_buffer_preference_dpo_loss`
  - `grpo_buffer_preference_dpo_weight`
  - `grpo_buffer_preference_dpo_pair_count`
  - `grpo_buffer_preference_dpo_active_row_ratio`
  - `grpo_buffer_preference_dpo_reward_gap_mean`
  - `grpo_buffer_preference_dpo_logit_mean`
  - `grpo_buffer_preference_dpo_implicit_accuracy`
  - `grpo_buffer_preference_dpo_timestep_mean/min/max`
- Added dedicated launcher:
  `scripts/training/launch_recogdrive_stage3_grpo_buffer_dpo_2b_local_stable.sh`.
  Its defaults isolate buffer-DPO by setting buffer reward bonus/distill to `0.0`
  and self-imitation to `0.0`; set `GRPO_SELF_IMITATION_LOSS_WEIGHT=0.01`
  only for an explicit v3-plus-buffer-DPO comparison.
- Launcher correction on 2026-06-14 UTC:
  - The buffer-DPO launcher previously still defaulted to a `mainhardgate` run name plus `GRPO_HARD_GATE_TTC=true` and `GRPO_HARD_GATE_DDC=true`, contradicting the post-v3 rule above.
  - It was corrected to default to `stage3_grpo_buffer_dpo_refctrl...`, `GRPO_USE_GSPO_RATIO=false`, `GRPO_HARD_GATE_TTC=false`, `GRPO_HARD_GATE_DDC=false`, and TTC/DDC thresholds `1.0`, matching the active current-repo original-LR GRPO control shape.
  - A `RUN_TRAIN=0` preflight confirmed: `offline_rl_enabled=true`, train-only elite buffer path set, `grpo_buffer_guidance_enabled=true`, `grpo_buffer_preference_dpo_loss_weight=0.02`, and buffer reward bonus/distill/self-imitation weights all `0.0`. The temporary preflight directory was deleted after inspection.
- Buffer-DPO data-path audit on 2026-06-14 UTC:
  - A deterministic random sample of `5000 / 85109` v2 train-only elite-buffer records showed `gt_present_ratio=1.0` and `il_present_ratio=1.0`.
  - `gt_valid_ratio=0.9992`; `il_valid_ratio=0.8314`. The DPO builder can therefore use stored GT/IL support for the intended behavior-loser pairs in normal cases, rather than relying on old-policy IL fallback with potentially mismatched stored `il_reward`.
  - In the same sample, `best_valid_source` was dominated by `gt`, `progress_endpoint`, and `policy`; this keeps the first buffer-DPO run focused on absorbing valid progress/policy improvements rather than the failed AWAC weighted-regression path.

First experiment rule after v3 result:
- Do not launch a buffer-DPO run on top of the failed v3 main-hardgate configuration by default. That would confound buffer absorption with a known EP-suppressing safety gate.
- If buffer-DPO is tested next, match the stronger original-LR control shape instead: LR `1e-4`, `sample_time=16`, BC `0.10->0.05`, reference KL `0.02`, no main TTC/DDC hard gate, no broad self-imitation. Enable only a small train-buffer DPO auxiliary:
  - `GRPO_BUFFER_GUIDANCE_ENABLED=true`
  - `GRPO_BUFFER_REWARD_BONUS_WEIGHT=0.0`
  - `GRPO_BUFFER_DISTILL_LOSS_WEIGHT=0.0`
  - `GRPO_BUFFER_PREFERENCE_DPO_LOSS_WEIGHT=0.02`
  - `GRPO_BUFFER_PREFERENCE_DPO_LOSS_SCHEDULE=linear_warmup`
  - `GRPO_BUFFER_PREFERENCE_DPO_WARMUP_EPOCHS=2`
  - keep `GRPO_SELF_IMITATION_LOSS_WEIGHT=0.0` for the first isolated buffer-DPO diagnostic.
- Success at the diagnostic level requires active nonzero DPO pairs and no deterioration in matched step300/600 exact navtest NC/TTC/DDC/EP compared with the zt3 original-LR control. Final success still requires comparable-length PDMS above the original 10-epoch `0.9055` and Safe DiffGRPO `0.906184` references.

Launch status on 2026-06-14 21:33 UTC:
- The first isolated Buffer-DPO diagnostic is queued on `training-vla-zt2` as
  `stage3_grpo_buffer_dpo_refctrl_s16_lr1e4_b2acc4_zt2wait_8gpu_20260614T213001Z`.
- It is intentionally waiting for all 8 zt2 GPUs to become free; no existing remote task was killed.
- The launch config confirms the intended isolation:
  - `grpo_buffer_preference_dpo_loss_weight=0.02`
  - `grpo_buffer_reward_bonus_weight=0.0`
  - `grpo_buffer_distill_loss_weight=0.0`
  - `grpo_self_imitation_loss_weight=0.0`
  - `grpo_hard_gate_ttc=false`, `grpo_hard_gate_ddc=false`
  - `grpo_use_gspo_ratio=false`, matching the current-repo control.
- The run should be judged first by whether DPO diagnostics become active after training starts:
  `grpo_buffer_preference_dpo_pair_count`, `active_row_ratio`, `reward_gap_mean`,
  `logit_mean`, and `implicit_accuracy`. Do not judge it only from final PDMS.

## Planned Attempt: Safety-Filtered GRPO Self-Imitation Targets

Motivation:
- The cap05 RLOO self-imitation run passed the original Stage3 early `0.88+` gate, but step600 showed EP gains coupled with NC/TTC/DDC degradation.
- Code inspection confirmed that self-imitation target selection used `hard_safe_mask + reward + margin`, while `hard_gate_ttc=False` and `hard_gate_ddc=False` by default. Therefore high-PDMS/high-progress samples with weaker TTC/DDC could become diffusion regression targets.

Reference audit:
- RIPT-VLA / Interactive Post-Training for VLA models: https://arxiv.org/abs/2505.17016 and https://github.com/Ariostgx/ript-vla. Relevant mechanism: K-rollout policy improvement with leave-one-out advantage and PPO-style policy updates, not pure offline weighted regression.
- DPPO: https://arxiv.org/abs/2409.00588 and https://github.com/irom-princeton/dppo. Relevant mechanism: keep direct policy-gradient optimization of sampled diffusion actions as the main objective; auxiliary regression should not replace on-policy improvement.
- The resulting design keeps GRPO as the main update and only filters the auxiliary self-imitation targets more strictly.

Implementation:
- Add explicit GRPO self-imitation target gates:
  - `offline_rl_grpo_self_imitation_require_nc=true`
  - `offline_rl_grpo_self_imitation_require_dac=true`
  - `offline_rl_grpo_self_imitation_require_ttc=true`, threshold `0.95`
  - `offline_rl_grpo_self_imitation_require_ddc=true`, threshold `0.99`
- Add logs:
  - `grpo_self_imitation_safety_candidate_ratio`
  - `grpo_self_imitation_nc/dac/ttc/ddc_pass_ratio`
  - `grpo_self_imitation_target_nc/dac/ttc/ep/comfort/ddc/tlc_mean`
- Keep the main GRPO reward and PDMS scorer unchanged.
- Keep original LR `1e-4`, sample_time `16`, BC `0.10->0.05`, reference KL `0.02`, GSPO ratio, RLOO baseline, and scene cap `0.5` so the algorithmic change is isolated.

Launch record:
- Run root:
  `/mnt/project/VLA-AD/outputs/stage3_grpo_rloo_selfimit_safetygate_step300_s16_lr1e4_b2acc4_8gpu_20260614T145712Z`.
- Local training PID file:
  `/mnt/project/VLA-AD/outputs/stage3_grpo_rloo_selfimit_safetygate_step300_s16_lr1e4_b2acc4_8gpu_20260614T145712Z/pids/stage3_rl_2b.pid`.
- Confirmed resolved command contains:
  - `agent.lr=1e-4`
  - `agent.grpo_sample_time=16`
  - `agent.reference_kl_coeff=0.02`
  - `checkpoint.every_n_train_steps=300`
  - `agent.offline_rl_grpo_self_imitation_require_nc=true`
  - `agent.offline_rl_grpo_self_imitation_require_dac=true`
  - `agent.offline_rl_grpo_self_imitation_require_ttc=true`
  - `agent.offline_rl_grpo_self_imitation_require_ddc=true`
  - thresholds NC `1.0`, DAC `1.0`, TTC `0.95`, DDC `0.99`.
- Eval watchers:
  - zt2: `/mnt/project/VLA-AD/outputs/stage3_grpo_rloo_selfimit_safetygate_step300_s16_lr1e4_b2acc4_8gpu_20260614T145712Z/unique_lock_watch_on_vla_zt2_4gpu`
  - zt3: `/mnt/project/VLA-AD/outputs/stage3_grpo_rloo_selfimit_safetygate_step300_s16_lr1e4_b2acc4_8gpu_20260614T145712Z/secondary_watch_on_rl_zt3_memfit_4gpu`
  - Both watchers use strict free-GPU gates (`EVAL_GPU_MAX_MEM_USED_MB=2000`, `EVAL_GPU_MAX_UTIL=5`) because zt2/zt3 already had existing evaluation/buffer tasks.
- The old cap05 local training was stopped after step900 checkpoint creation.
  One remote step900 eval later completed with PDMS `0.887107`; another duplicate
  eval, if still running, is left to finish naturally.

Expected diagnostics:
- `grpo_self_imitation_safety_candidate_ratio` should be lower than the current broad candidate ratio, but not zero for long stretches.
- Target TTC/DDC means should stay near the configured thresholds or above.
- If early PDMS stays around `0.885` but DDC/TTC no longer decline, the safety filter is doing its intended job; a later run can then test stronger buffer absorption.

Failure criteria:
- If `step-step_300` or epoch0 drops materially below the original early `0.88+` band, do not promote.
- If target ratio collapses to zero for most logged batches, loosen the target safety gate or switch the auxiliary off rather than continuing a no-op run.
- If EP improves but DDC/TTC still decline, the issue is not just target filtering and the next fix should be reward/advantage shaping in the GRPO path or preference ranking, not another self-imitation gate.

## Background Task: Keep-Best Train-Only Buffer Exploration

Motivation:
- The full v2 navtrain buffer already has `mean_best_valid_reward=0.971664` and can still provide a useful oracle for discovering better-than-GT trajectories.
- The pure AWAC training path did not transfer well, but the buffer remains useful for diagnostics, future safety-constrained self-imitation, and preference pairs.

Implementation:
- Added `MERGE_EXISTING_RECORDS=1` support to `build_recogdrive_stage3_awac_elite_buffer.py`.
- New records are merged with existing v2 records per token, deduplicated, and retained by valid top-k selection plus GT/IL support candidates.
- Existing high-quality records are not blindly overwritten by a lower-quality exploration pass.
- Added `scripts/training/launch_recogdrive_stage3_awac_buffer_keepbest_background.sh`:
  - waits for target GPUs instead of killing or preempting existing tasks;
  - loops over navtrain shards;
  - supports `AUTO_POLICY_CHECKPOINT_DIR` so the latest Stage3 policy checkpoint can provide additional sampled candidates;
  - uses only `metric_cache_train_full` for scoring.

Navtest analysis rule:
- Added `scripts/evaluation/analyze_recogdrive_stage3_navtest_pdms.py`.
- It reads existing `checkpoint_eval_submetrics.tsv` files and reports best/latest checkpoint, step deltas, submetric deltas, and deltas to `0.88`, `0.9055`, and `0.906184`.
- `scripts/evaluation/watch_stage3_checkpoints_eval_8gpu.sh` now refreshes `navtest_pdms_analysis.md` and `navtest_pdms_analysis.tsv` under the run root after each successful checkpoint evaluation.
- `AgentLightningModule` now logs the GRPO self-imitation safety-candidate pass ratios and selected-target NC/DAC/TTC/EP/comfort/DDC/TLC means when the planner returns them. This is required to connect navtest regressions back to train-time target selection.
- This is diagnostic only. Navtest summaries must not be used to choose buffer records or train rewards.

Promotion / failure criteria:
- At matched early epoch, exact navtest PDMS must be near or above the original Stage3 early `0.88+` band. If epoch0 is materially below `0.88`, do not run 20 epochs unless diagnostics show the self-imitation path was inactive and the run is effectively a control.
- Compare final training only against the 10-epoch original `0.9055` and Safe DiffGRPO `0.906184` when training length/steps are comparable.

## Completed Attempt Ledger

### 1. AWAC/IQL Offline Elite-Buffer Training

Motivation:
- Treat GT as behavior support, not as optimal.
- Use train-only PDMS as an oracle to discover better trajectories from GT, IL/reference policy, current policy samples, and structured perturbations.
- Train the diffusion planner with advantage-weighted denoising targets.

Implementation work completed:
- Added Stage3 objective switch and AWAC/IQL path.
- Added strict v2 elite buffer with valid masks and submetric guards.
- Added structured perturbation explorer and validator.
- Added missing-submetric safety handling.
- Added DPO/ranking/repulsion options inside the AWAC path.

Oracle/buffer evidence:
- Full v2 train buffer validation over `85109` records:
  - `valid_candidate_ratio`: `0.979694`
  - `has_valid_candidate_ratio`: `1.0`
  - `mean_gt_reward`: `0.932272`
  - `mean_il_reward`: `0.709482`
  - `mean_best_valid_reward`: `0.971664`
  - `pct_best_valid_above_gt`: `0.581537`
  - `pct_best_valid_above_il`: `0.791338`
- The buffer is useful: many train scenes have valid candidates above GT and IL.

Evaluation results:

| Run | Best Ckpt | PDMS | NC | DAC | TTC | EP | Comfort | DDC | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `stage3_awac_iql_dualhost_20260612T182947Z` | `epoch_0-step_665` | `0.860226` | `0.9766` | `0.9439` | `0.9330` | `0.8129` | `0.9998` | `0.9738` | Good buffer, weak policy transfer. |
| `stage3_awac_iql_bc015_20260613T0648Z` | `epoch_0-step_665` | `0.860100` | `0.9768` | `0.9435` | `0.9333` | `0.8127` | `0.9996` | `0.9738` | Similar to dualhost. |
| `stage3_awac_iql_pref_rank_ttc_20260613T112709Z` | `epoch_0-step_665` | `0.850661` | `0.9756` | `0.9341` | `0.9332` | `0.8033` | `0.9997` | `0.9719` | Preference/rank additions did not fix transfer. |
| `stage3_awac_blend035_lownoise_dpo8gpu_20260613T164458Z` | `epoch_0-step_665` | `0.833381` | `0.9682` | `0.9200` | `0.9136` | `0.8032` | `0.9988` | `0.9752` | Lower-noise/blend DPO still below target. |
| `stage3_awac_gapdpo_hybrid4gpu_v2_20260613T155350Z` | `epoch_0-step_665` | `0.798205` | `0.9339` | `0.9219` | `0.8667` | `0.7749` | `0.9977` | `0.9741` | Hybrid was not mature enough to validate DPO. |
| `stage3_awac_gapdpo_lowvalid_2gpu_20260613T155703Z` | `epoch_0-step_665` | `0.784474` | `0.9213` | `0.9208` | `0.8375` | `0.7748` | `0.9997` | `0.9722` | Poor safety transfer. |
| `stage3_awac_lownoise_dpo3gpu_20260613T163215Z` | `epoch_1-step_1184` | `0.780348` | `0.9315` | `0.9072` | `0.8492` | `0.7730` | `0.9908` | `0.9670` | More steps helped this bad config but remained far below GRPO. |
| `stage3_awac_warmup_absddc_blend025_lownoise_online_20260613T183113Z` | `epoch_0-step_665` | `0.862299` | `0.9825` | `0.9449` | `0.9444` | `0.8044` | `0.9996` | `0.9786` | Strongest AWAC online warmup so far; still far below GRPO. |

Lessons:
- The buffer discovery problem is not the main blocker.
- The blocker is policy absorption: high-PDMS candidates did not become high-probability diffusion samples.
- AWAC runs are confounded by LR and effective steps, so they do not prove AWAC/IQL is impossible. They do prove our implementation was not yet a mature, reliable offline-RL post-training implementation.
- Pure weighted denoising toward elite targets can overfit target likelihood without improving final sampling distribution.

Do not resume pure AWAC/IQL until:
- LR/effective-step controls are matched against GRPO.
- DPO/preference diagnostics are complete.
- The implementation is aligned with a mature offline diffusion policy method rather than a simple weighted-regression variant.

### 2. Diffusion-DPO / Preference Attempts Inside AWAC

Reference implementation checked:
- Diffusion-DPO official repo: https://github.com/SalesforceAIResearch/DiffusionDPO

Mechanism to preserve:
- Winner and loser are paired in the same batch order.
- Current and reference model evaluate the same noisy latents/timesteps.
- Loss uses the difference of current MSE gap and reference MSE gap.
- Logs include implicit accuracy and current/reference MSE.

Local status:
- Local `_compute_diffusion_dpo_preference_loss` uses shared noise/timestep and a frozen reference model when configured.
- The loss sign is broadly consistent with Diffusion-DPO's MSE-difference formulation.
- Diagnostics were completed on 2026-06-14: implicit accuracy, winner/loser current MSE, winner/loser reference MSE, current/reference margins, and logit scale are returned and logged.
- DPO currently lives in the AWAC path, not the main GRPO path.

Lesson:
- Existing poor DPO results should not be interpreted as "preference learning does not work." They indicate our DPO use was not yet a fully instrumented, mature preference-training experiment.

Before next preference run:
- Validate active pair ratio and reward-gap distribution.
- Calibrate beta against observed current/reference logratio magnitude.
- Consider preference as a small auxiliary to GRPO only after diagnostics show it is active and correctly ordered.

### 3. Safe DiffGRPO / GRPO Reference-KL Path

Motivation:
- Directly optimize sampled trajectories with train PDMS reward.
- Avoid treating GT as optimal.
- Preserve behavior through BC and reference KL.

Known evidence:
- Historical Safe DiffGRPO reached `0.906184`.
- User-reported original local Stage3 RL reached `0.9055` at 10 epochs with original LR `1e-4`.
- Current `2e-4` run is not a pure reproduction; compare it against the zt3 `1e-4` control.

Implementation notes:
- Current GRPO samples `G=16` trajectories per scene and computes group advantages.
- It uses hard safety masks and asymmetric safe advantage.
- The main `2e-4` run uses reference KL `0.02` and BC anneal `0.10 -> 0.05`.
- The non-GSPO path is effectively single-step trajectory-level policy gradient with current-policy sampling.

Reference gap:
- DDPO, DPPO, and RIPT-VLA all emphasize storing old logprobs/rollouts and applying PPO-style clipped updates over collected samples.
- Our current non-GSPO path does not replay a sampled batch for multiple PPO epochs/minibatches.
- The GSPO variant is closer because it can sample from a behavior policy and use a ratio, but it still needs diagnostics to confirm ratio/clip behavior and policy absorption.

### 3.1 GSPO Advantage-Control Diagnostic

Run:
- `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_log1_diag_zt2_4gpu_20260614T0300Z`

Config:
- LR `1e-4`
- `sample_time=16`
- `batch_size=2`, `accumulate_grad_batches=8`, 4 GPUs
- `limit_train_batches=40`, `limit_val_batches=0`
- `log_every_n_steps=1`
- `grpo_use_gspo_ratio=true`
- `grpo_behavior_policy_sample=true`
- `grpo_behavior_policy_sync_interval=100000`
- `grpo_normalize_advantage_batch=true`
- `grpo_advantage_clip_abs=3.0`
- no buffer guidance, no AWAC, no DPO

Observed step diagnostics over 5 optimizer steps:
- `gspo_ratio_mean`: `1.0000 -> 0.9844`
- `gspo_ratio_min/max` at last step: `0.9649 / 1.0017`
- `gspo_ratio_clip_frac`: max observed `0.2266`, last `0.0156`
- `gspo_log_ratio_std`: last `0.00875`
- `gspo_abs_log_ratio_mean`: last `0.01625`
- `gspo_approx_kl`: last `0.000203`
- `grpo_advantage_batch_normalized`: `1.0`
- `grpo_advantage_std_after_transform`: near `0.96-1.00`
- `grpo_advantage_clip_frac`: `0.0-0.0234`
- `sampled_from_behavior_policy`: `1.0`
- `behavior_policy_synced`: `0.0` after launch, as intended for the long sync interval

Decision:
- Keep these code/logging changes. They prove the GSPO ratio path and advantage transform are active.
- Do not claim this is SOTA PPO yet. It is still a single-pass Lightning training loop, not DDPO/DPPO/RIPT-VLA-style rollout replay with fixed old logprobs over multiple inner epochs.
- The next full run may use these controls as a clean GSPO baseline only if we explicitly label it as "GSPO single-pass", not as full DPPO.

### 3.2 GRPO PPO Replay Smoke Diagnostic

Run:
- `stage3_grpo_replay_diag_s4_i2_lr1e4_4gpu_20260614T042519Z`

Reference alignment:
- DDPO, DPPO, and RIPT-VLA all use collected rollouts or old log-probabilities with PPO-style clipped updates instead of one immediate reward-weighted regression step.
- This run is the first local path that performs fixed-rollout replay in the Lightning training loop with manual optimizer steps.

Config:
- LR `1e-4`
- `stage3_objective=grpo_replay`
- `sample_time=4`
- `batch_size=2`, `accumulate_grad_batches=1`, 4 GPUs
- `limit_train_batches=40`, `limit_val_batches=0`, `log_every_n_steps=1`
- `grpo_use_gspo_ratio=true`
- `grpo_normalize_advantage_batch=true`
- `grpo_advantage_clip_abs=3.0`
- `ppo_replay_inner_epochs=2`
- `ppo_replay_minibatch_size=4`
- `ppo_replay_sync_behavior_each_batch=true`
- `ppo_replay_bc_update=true`

Observed diagnostics:
- Checkpoint written: `epoch=0-step=200.ckpt`.
- `ppo_replay_optimizer_steps`: `5` per batch, matching `2` inner epochs over `8` rollout rows with minibatch `4`, plus one BC trust-region update.
- `ppo_replay_valid_ratio`: epoch mean `0.9852`, last step `0.8750`.
- `ppo_replay_loss_active`: `1.0` for all 40 logged batches.
- `gspo_ratio_mean`: epoch mean `0.9908`, last step `0.9886`.
- `gspo_ratio_min/max`: epoch min/max around `0.9823 / 0.9981`; last step `0.9731 / 0.9985`.
- `gspo_ratio_clip_frac`: `0.0` throughout.
- `gspo_approx_kl`: epoch mean `8.66e-5`, last step `1.13e-4`.
- `grpo_advantage_std_after_transform`: `1.0`, confirming batch advantage normalization.
- `grpo_advantage_zero_ratio`: epoch mean `0.00625`.
- Rollout component means were finite; last-step `mean_nc=0.8438`, `mean_dac=0.8125`, `mean_ttc=0.7188`, `mean_ddc=0.9844`, `safe_ratio=0.6875`.

Decision:
- Keep the implementation. It passes the minimal maturity smoke for PPO replay plumbing.
- Do not evaluate this checkpoint as an algorithm result; `sample_time=4` and 40 batches are intentionally too small.
- Do not launch full training directly. The next meaningful experiment is a controlled 1-epoch PPO replay diagnostic with `sample_time=16`, original LR `1e-4`, and the same safety/BC/reference-KL settings as the GRPO control.

Failure criteria for the next replay run:
- If `ppo_replay_valid_ratio` falls below `0.8` for sustained windows, stop and inspect group sampling.
- If `gspo_ratio_clip_frac` remains exactly `0` while reward does not move, the replay update may be too weak; increase inner epochs or replay minibatch exposure before full training.
- If `gspo_approx_kl` spikes or DDC/TTC means regress, reduce inner epochs or tighten reference/BC before continuing.

### 4. GRPO Buffer-Guided Self-Imitation

Motivation:
- The elite buffer contains useful trajectories, but AWAC did not make the diffusion sampler output them.
- GRPO can only reinforce trajectories it samples. It cannot directly learn unseen buffer trajectories unless buffer knowledge enters through reward shaping, distillation, or preference losses.

Current implementation:
- Buffer reward-neighborhood bonus, only for hard-safe samples.
- Low-noise denoising distillation on valid elite-buffer targets.
- Self-imitation denoising loss on current GRPO samples that are hard-safe and above a baseline.

Invalidated run:
- `stage3_grpo_buffer_guided_gspo_s16_lr2e4_b2acc8_zt2_4gpu_20260614T012106Z` was stopped on 2026-06-14 because buffer/self-imitation diagnostics were not logged by Lightning.

Next planned run:
- `stage3_grpo_buffer_guided_gspo_s16_lr2e4_b2acc8_diagfix_zt2_4gpu_20260614T020706Z`.
- Treat the first 50-100 steps as a go/no-go diagnostic check before letting it reach a full checkpoint.
- Evaluation watchers are disabled during the diagnostic window to avoid using GPUs occupied by unrelated tasks.

Required diagnostics:
- `grpo_buffer_guidance_target_ratio`
- `grpo_buffer_reward_bonus_mean/max`
- `grpo_buffer_target_distance_mean`
- `grpo_buffer_distill_weight_sum`
- `grpo_self_imitation_candidate_ratio`
- `grpo_self_imitation_target_ratio`
- `grpo_self_imitation_target_reward_mean/max`
- `gspo_ratio_mean`
- `gspo_ratio_clip_frac`

Failure criteria:
- If buffer target ratio is high but target distance does not fall, the reward bonus is not shaping exploration.
- If distill/self-imitation weight sum is zero or target ratio is near zero, the absorption channel is inactive.
- If DDC/TTC regress, stop the variant even if raw PDMS or EP improves.

## External Reference Findings

### 2026-06-14 Reference Audit Update

Checked sources:
- DDPO paper: https://arxiv.org/abs/2305.13301
- DPPO official implementation: https://github.com/irom-princeton/dppo
- RIPT-VLA paper and code: https://arxiv.org/abs/2505.17016 and https://github.com/Ariostgx/ript-vla

Concrete requirements borrowed from these references:
- DDPO frames denoising as a multi-step decision-making process, so a mature diffusion RL run should reason about denoising-step likelihoods or stored denoising-chain probabilities, not only final-trajectory weighted regression.
- DPPO's official implementation exposes PPO-specific diffusion code and configs such as clipping coefficient, denoising discount, likelihood standard deviation guards, and a CleanRL-style PPO training loop. A Stage3 run should not be called DPPO-equivalent unless it has fixed old logprobs and replay-style PPO updates.
- RIPT-VLA uses VLA log-prob interfaces, rollout generation, leave-one-out advantages, PPO updates, and dynamic sampling/filtering of zero-advantage groups. This directly matches our observed failure mode: many sampled groups become low-information or unsafe, so group composition and non-zero advantage coverage must be first-class diagnostics.

Decision:
- Do not launch another full single-pass GSPO run as the next main algorithm experiment. The short GSPO diagnostic remains useful, but it is not a mature SOTA implementation.
- The next main implementation target is a full PPO replay path aligned with DDPO/DPPO/RIPT-VLA: rollout collection, fixed old logprobs, advantage normalization/clipping, minibatch replay for multiple inner epochs, ratio clipping, approximate KL, and non-zero-advantage group filtering.

### Diffusion-DPO

Sources:
- Official implementation: https://github.com/SalesforceAIResearch/DiffusionDPO

Core mechanism:
- Shared noise/timestep current-vs-reference MSE gap on winner/loser pairs.
- Implicit accuracy and MSE diagnostics are first-class training signals.

Local gap:
- Add missing diagnostics before using DPO results to make algorithm decisions.

### DDPO

Sources:
- Paper: https://arxiv.org/abs/2305.13301
- Official implementation: https://github.com/kvablack/ddpo-pytorch

Core mechanism:
- Collect sampled trajectories and old denoising logprobs.
- Compute rewards, normalize advantages, and train with PPO ratio clipping over diffusion timesteps.
- Use inner epochs/minibatches over collected samples.
- Log approximate KL and clip fraction.

Local gap:
- Current GRPO mostly updates immediately per batch, without a stored rollout buffer and multi-epoch PPO replay in the non-GSPO path.

### DPPO

Sources:
- Paper: https://arxiv.org/abs/2409.00588
- Official implementation: https://github.com/irom-princeton/dppo

Core mechanism:
- Treat denoising as a diffusion MDP.
- Store chains and old logprobs.
- Optimize PPO clipped loss over denoising steps.
- Normalize and clip advantages.
- Use multi-epoch minibatch updates.

Local gap:
- We use trajectory-level reduced logprob and no critic. That may be acceptable only if justified, but it is not a complete DPPO reproduction.

### RIPT-VLA

Sources:
- Official implementation: https://github.com/Ariostgx/ript-vla

Core mechanism:
- K-rollout sampling per context.
- Leave-one-out advantage estimation.
- Dynamic sampling to avoid uninformative all-success/all-failure groups.
- Multiple PPO epochs over rollout data.
- Explicit valid masks and rollout statistics.

Local gap:
- Our group advantage resembles RLOO, but we do not yet have dynamic sampling or full PPO replay in the main Stage3 loop.

## Next Mature Directions

### Direction A: PPO/GSPO-Complete Stage3 GRPO

Hypothesis:
- A mature PPO-style update over collected diffusion chains may absorb reward gradients more reliably than one immediate policy-gradient step.

Must include:
- Collect chains, trajectories, rewards, components, old logprobs, and group advantages.
- Rebatch and replay for `num_inner_epochs > 1`.
- Ratio clipping, approximate KL, clip fraction, advantage normalization/clipping.
- Denoising-step-aware weighting.
- Reference KL and BC trust region.

Do not launch if:
- It only toggles `use_gspo_ratio` without rollout replay and diagnostics.

Implementation hardening completed on 2026-06-14:
- Added GRPO/GSPO diagnostics required by mature PPO-style methods:
  - `gspo_ratio_min/max`
  - `gspo_abs_log_ratio_mean`
  - `gspo_approx_kl`
  - `gspo_reverse_approx_kl`
  - group-weight and safe-count statistics
  - advantage mean/std/min/max before and after optional transforms
  - advantage positive/zero/clip ratios
- Added optional `grpo_normalize_advantage_batch` and `grpo_advantage_clip_abs`.
- Defaults preserve existing behavior; these controls are for the next controlled run after the current diagnostics are read.

Remaining gap before claiming a SOTA-level PPO implementation:
- The current Lightning path still does not store rollout batches and replay them for multiple PPO epochs/minibatches.
- A full DPPO/DDPO/RIPT-VLA-aligned implementation needs a rollout buffer containing trajectories, denoising chains, old logprobs, rewards, components, masks, and advantages, followed by replay updates with fixed old logprobs.

## Revised Experiment Plan 2026-06-14

### Phase 0: Stop Low-Value Runs And Preserve Evidence

Keep only experiments that answer a specific question:
- Local `2e-4` GRPO: stopped after `epoch_0-step_1330`. Evaluate the checkpoint, but do not continue training.
- zt3 `1e-4` GRPO: keep as the original-LR control.
- zt2 `1e-4` GSPO/advantage-control diagnostic: completed. Use it only as evidence that instrumentation and single-pass GSPO ratio are active.

Stop or avoid:
- Pure AWAC/IQL full training.
- Any preference/DPO run without pair count, active row ratio, logit scale, implicit accuracy, and current/reference winner-loser losses.
- Any GRPO variant that lacks ratio/KL/clip/advantage diagnostics.
- Any launch where LR, effective batch, sample_time, or checkpoint cadence differs from the control without being the explicit variable under test.

### Phase 1: Finish Evidence Collection

Required outputs:
- Full or shard-aggregated navtest eval for local `2e-4` `epoch_0-step_1330`: completed, PDMS `0.872059`.
- First checkpoint/eval for zt3 `1e-4` GRPO.
- Store zt2 diagnostic scalars in this ledger and compare with any future GSPO run.
- Store local PPO replay smoke scalars in this ledger: completed.

Go/no-go:
- Local `2e-4` is below Safe DiffGRPO by about `0.0341` PDMS at epoch0; do not revisit `2e-4` as a default LR.
- If the zt3 `1e-4` control is competitive with Safe DiffGRPO, use `1e-4` as the default for algorithm tests.
- If zt3 `1e-4` is also weak, inspect data/eval parity before modifying algorithms.

Diagnostic launch rule:
- Limited diagnostics must set `LOG_EVERY_N_STEPS=1`.
- If testing GSPO ratio behavior, use a behavior-policy sync interval larger than the diagnostic window, e.g. `GRPO_BEHAVIOR_POLICY_SYNC_INTERVAL=100000`, because the first forward syncs the behavior policy and ratio is expected to be near `1.0` before the first optimizer update.
- Use at least `2 * accumulate_grad_batches` train batches so there are logged batches after one optimizer step.

### Phase 2: Do Not Promote Single-Pass GSPO To A Main Full Run

Allowed use:
- GSPO/GRPO with original LR `1e-4`, `sample_time=16`, same effective batch as the control.
- Enable batch advantage normalization and conservative fixed clipping:
  - `GRPO_NORMALIZE_ADVANTAGE_BATCH=true`
  - `GRPO_ADVANTAGE_CLIP_ABS=3.0`
- Keep BC anneal `0.10 -> 0.05`, reference KL `0.02`, scheduler `20` epochs to `1e-5`.
- Do not add buffer guidance in this run. This isolates the PPO/advantage-control effect.
- Limit this to a short diagnostic or ablation unless Phase 1 evidence shows the current implementation is already competitive and stable.

Rationale:
- DDPO/DPPO/RIPT-VLA-style methods all rely on stable old-policy ratio updates and controlled advantages.
- The completed short GSPO diagnostic proves the ratio path and advantage transforms are active, but it does not prove that the training loop has enough policy-improvement structure to absorb high-reward trajectories.

Interpretation limit:
- This run can validate whether single-pass GSPO improves over current GRPO.
- It cannot validate full DPPO/DDPO/RIPT-VLA until rollout replay exists.

### Phase 3: Prove PPO Replay At Near-Full GRPO Sampling Before Heavy Buffer Absorption

Status:
- A trajectory-level PPO replay path now exists and passed a `sample_time=4` smoke test.
- It stores sampled trajectories, denoising chains, old reduced logprobs, rewards, components, safety masks, group ids, group advantages, and optional old-policy BC chains.
- It replays fixed rollouts for multiple inner epochs/minibatches with PPO clipping, reference KL, and manual optimizer steps.

Required before the next main full-training launch:
- Run a near-full-sampling diagnostic with `sample_time=16`, LR `1e-4`, short duration, and `LOG_EVERY_N_STEPS=1`.
- Compare replay diagnostics against zt3 original-LR GRPO control:
  - reward/base_reward trend,
  - valid rollout ratio,
  - mixed/all-safe/all-unsafe group ratios,
  - ratio mean/min/max,
  - clip fraction,
  - approximate KL,
  - NC/DAC/TTC/DDC means.
- Add dynamic group filtering or resampling only if the diagnostic shows too many all-safe/all-unsafe or near-zero-advantage groups.
- Add per-denoising-step PPO only if trajectory-level replay shows a useful signal but insufficient policy absorption; do not add it preemptively because it increases memory and implementation risk.

This is the first implementation level that can reasonably be compared to DDPO/DPPO/RIPT-VLA.

Implementation shape in this codebase:
- Keep the VLM backbone and DiT architecture unchanged.
- Add a new Stage3 objective/path rather than mutating the current control path in place, e.g. `stage3_objective=grpo_replay` or an explicit `grpo_ppo_replay_enabled=true`.
- Reuse existing `sample_chain`, `get_logprobs`, `_reduce_chain_logprobs`, `_compose_stage3_reward`, and `_compute_stage3_advantages` so reward and logprob semantics stay identical to the current code.
- Move the extra PPO update logic into the Lightning training layer only if multiple optimizer steps over the same fixed rollout are required. Summing several replay losses inside a single forward pass is not equivalent to PPO replay because parameters do not change between inner epochs.
- Start with trajectory-level reduced logprob replay, because current code already supports chain logprob reduction. Only add per-step denoising PPO if the trajectory-level replay diagnostic is stable and if memory allows storing enough denoising-chain state.

PPO replay smoke-test acceptance before a full run:
- Can collect a rollout mini-buffer with shapes `[B, G, K, H, D]` or an explicitly documented equivalent: passed for trajectory-level replay.
- Old logprob stays fixed across replay epochs; new logprob changes after optimizer steps: passed by finite ratio/KL movement.
- Ratio mean begins near `1.0`, clip fraction becomes non-zero only after updates, and approximate KL remains finite: partially passed. KL is finite, but clip fraction stayed `0.0`, so the first full-sampling diagnostic should test whether the update is too conservative.
- Advantage rows with no useful group signal are filtered or down-weighted and logged: partially passed. Zero-advantage ratio is logged and low in the smoke, but dynamic resampling is not implemented.
- NC/DAC/TTC/DDC component means are logged for rollout and replay batches: passed for rollout components.
- A diagnostic run with `limit_train_batches<=40` logs all replay diagnostics at `LOG_EVERY_N_STEPS=1`: passed.

Only after this smoke test passes:
- Launch a limited 1-epoch replay run with checkpoint eval.
- Launch full 20-epoch replay only if the 1-epoch eval is not below the `1e-4` control by more than normal eval noise and does not degrade DDC/TTC.

Planned near-full diagnostic:
- Run name pattern: `stage3_grpo_replay_diag_s16_i2_lr1e4_8gpu_*`.
- Purpose: test the PPO replay implementation under the same group sampling scale as the historical GRPO runs before committing to full training.
- Config:
  - LR `1e-4`
  - `stage3_objective=grpo_replay`
  - `sample_time=16`
  - `batch_size=2`
  - 8 local GPUs
  - `max_epochs=1`
  - `limit_train_batches=40`
  - `limit_val_batches=0`
  - `log_every_n_steps=1`
  - BC anneal `0.10 -> 0.05`
  - reference KL `0.02`
  - scheduler horizon `20` epochs, min LR `1e-5`
  - PPO replay inner epochs `2`
  - PPO replay minibatch size `8`
  - max grad norm `1.0`
  - no elite buffer, no AWAC, no DPO
- Expected signs:
  - `ppo_replay_valid_ratio >= 0.8`.
  - `gspo_approx_kl` finite and small.
  - `gspo_ratio_mean` near `1.0` but not exactly constant.
  - `gspo_ratio_clip_frac` may become non-zero after several replay updates; if it stays zero, the update is conservative and should be tuned before full training.
  - NC/DAC/TTC/DDC component means must remain comparable to the GRPO control windows.
- Stop criteria:
  - NaN/inf in loss, ratio, KL, or reward.
  - Sustained `ppo_replay_loss_active=0`.
  - Severe DDC/TTC regression in rollout components.
  - Out-of-memory at `sample_time=16`; retry with the same algorithm and smaller per-GPU batch rather than changing the objective.

Near-full diagnostic result:
- Run: `stage3_grpo_replay_diag_s16_i2_lr1e4_8gpu_20260614T043345Z`.
- Status: completed 40/40 train batches in about 3 minutes and wrote `epoch=0-step=360.ckpt`.
- `ppo_replay_optimizer_steps`: `9` per batch, matching `2` inner epochs over 32 rollout rows with minibatch `8`, plus one BC update.
- `ppo_replay_valid_ratio`: `1.0` for all logged batches.
- `ppo_replay_loss_active`: `1.0` for all logged batches.
- `gspo_ratio_mean`: epoch `0.9746`; step range `0.9436 - 0.9890`.
- `gspo_ratio_clip_frac`: epoch `0.1020`; step range `0.0 - 0.5`.
- `gspo_approx_kl`: epoch `5.38e-4`; step max `0.00223`.
- Advantage normalization/clipping was active: epoch `grpo_advantage_std_after_transform=0.9494`, `grpo_advantage_clip_frac=0.0135`, `zero_ratio=0.0`.
- Group composition was informative: epoch `mixed_group_ratio=0.6469`, `all_safe_group_ratio=0.3438`, `all_unsafe_group_ratio=0.0094`.
- Safety/submetric rollout means: `NC=0.9312`, `DAC=0.8471`, `TTC=0.8032`, `DDC=0.9555`, `safe_ratio=0.7863`.
- Decision: launch a controlled 1-epoch PPO replay training run with the same algorithm settings and epoch-end PDMS evaluation. Do not launch full 20-epoch replay until the 1-epoch checkpoint evaluation is competitive and does not regress DDC/TTC.

Buffer absorption should then be added as a controlled auxiliary:
1. reward-neighborhood bonus only
2. buffer low-noise distillation only
3. self-imitation from high-reward sampled trajectories only
4. preference/DPO auxiliary only after active-pair diagnostics pass
5. combined, only if individual terms are active and non-regressive

Buffer absorption success criteria:
- `grpo_buffer_target_distance_mean` decreases or remains low while reward improves.
- self-imitation target reward is above group/buffer baseline.
- NC/DAC/TTC/DDC do not regress.
- Evaluation PDMS improves over both `1e-4` GRPO control and Safe DiffGRPO `0.906184`.

### Phase 4: Preference/DPO Only As A Mature Auxiliary

Do not run preference/DPO as a standalone replacement again until:
- winner/loser active pair ratio is high enough,
- reward-gap distribution is logged,
- current/reference winner-loser MSE margins are logged,
- implicit accuracy is not saturated or random,
- beta/logit scale is calibrated from observed logratio magnitude,
- shared timestep/noise and reference model parity are verified.

Preferred use:
- Add small preference auxiliary to a stable GRPO/GSPO run, not to pure AWAC.
- Pair buffer valid-best against GT/IL/current low-reward samples only when DDC/TTC/NC/DAC guards pass.

### Direction B: GRPO + Buffer Absorption With Proof Of Activity

Hypothesis:
- Buffer trajectories help only when sampled trajectories are close enough for reward bonus or distillation to be active.

Must include:
- Target coverage and target distance diagnostics.
- Hard-safe gating.
- Low-noise denoising target schedule.
- Ablation against same LR/effective steps without buffer.

Do not increase weights until:
- `grpo_buffer_target_distance_mean`, target ratios, and self-imitation ratios show the auxiliary loss is active.

### Direction C: Proper Diffusion-DPO Auxiliary

Hypothesis:
- Preference learning can help rank valid high-PDMS buffer trajectories above GT/IL/current low-PDMS samples, but only if the implementation matches Diffusion-DPO mechanics.

Must include:
- Shared noise/timestep current/reference loss.
- Implicit accuracy.
- Winner/loser current and reference MSE.
- Pair-gap distribution and active row ratio.
- Beta calibration from observed logratio scale.
- Evaluation as an auxiliary to GRPO first, not as a replacement.

Do not use as evidence if:
- Pair count is low, implicit accuracy is unlogged, or beta/logits saturate.

### Direction D: Revisit AWAC/IQL Only As A Full Offline Diffusion RL Method

Hypothesis:
- AWAC/IQL may still work, but our previous version was closer to weighted target regression than a mature offline diffusion RL implementation.

Must include:
- Matched LR/effective optimizer steps.
- Explicit behavior support/trust-region checks.
- Diagnostics proving selected targets become more likely under the diffusion sampler.
- No invalid candidate training by default.

Do not prioritize this over GRPO until:
- Current GRPO LR control and buffer-guided run are evaluated.

## Active Questions

- Does `2e-4` improve GRPO over original `1e-4`, or only change early training dynamics?
- Does GSPO ratio/clipping improve stability compared with current-policy single-step GRPO?
- Do buffer-guided auxiliary terms actually activate, or are they mostly zero/too weak?
- Does the model's denoising likelihood on elite targets improve without navtest PDMS improving? If yes, the issue is sampler distribution or target mismatch.
- Are DDC/TTC protected during progress improvements, or are gains coming from reward-hacking side effects?

## Current Run Decisions

As of 2026-06-14 05:18 UTC:

- Keep running: `stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z`.
  - Reason: this is the needed original-LR GRPO control. It isolates algorithm effects from the failed/weak `2e-4` route.
  - Latest status: still running, no checkpoint yet.
  - Evaluation watchers already exist; do not start another duplicate watcher:
    - primary zt3 2GPU watcher: `/mnt/project/VLA-AD/outputs/stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z/primary_watch_on_rl_zt3_2gpu`
    - secondary zt2 4GPU watcher: `/mnt/project/VLA-AD/outputs/stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z/secondary_watch_on_vla_zt2_4gpu`
  - Latest logged scalar window: step `49 -> 349`, reward `0.50699 -> 0.88152`, base reward `0.61353 -> 0.84410`, TTC `0.80208 -> 0.88542`, safe ratio `0.72917 -> 0.98958`, GSPO ratio stays `1.0` because this is the non-replay control path.
  - Decision: keep it. It is a meaningful control, not a stale or weakly configured experiment.
- Completed and keep as smoke evidence: `stage3_grpo_replay_diag_s4_i2_lr1e4_4gpu_20260614T042519Z`.
  - Reason: it validates the new PPO replay implementation path, but is intentionally too small for PDMS conclusions.
- Completed and promote to controlled 1-epoch training: `stage3_grpo_replay_diag_s16_i2_lr1e4_8gpu_20260614T043345Z`.
  - Reason: the near-full sample-time diagnostic passed replay, ratio/KL, and safety logging gates.
- Running: `stage3_grpo_replay_1epoch_s16_i2_lr1e4_8gpu_20260614T044444Z`.
  - Reason: this is the first controlled full-navtrain 1-epoch PPO replay run. It keeps LR, `sample_time`, BC, reference KL, and scheduler semantics aligned with the GRPO control, and changes only the replay update path.
  - Latest status: still running, no checkpoint yet.
  - Latest logged scalar window: step `19 -> 339`, `ppo_replay_valid_ratio=1.0` throughout, ratio mean `0.9282 - 0.9951`, latest ratio mean `0.96936`, latest clip fraction `0.2031`, max clip fraction `0.7344`, latest approx KL `7.87e-4`, max approx KL `0.00336`.
  - Latest rollout reward/submetrics include one weak batch: reward `0.34987`, base reward `0.52296`, NC `0.73438`, TTC `0.625`, DDC `0.77148`, safe ratio `0.61328`.
  - Decision: keep for now, but watch closely. The PPO ratio/KL are still within a small range, so this is not an implementation failure. If weak safety batches persist rather than fluctuate, stop before promoting this setting to a 20-epoch run.
  - Evaluation watcher: `stage3_grpo_replay_1epoch_s16_i2_lr1e4_8gpu_20260614T044444Z_navtest_exact_eval`.
- Running on zt2: `stage3_grpo_replay_1epoch_s16_i1_lr1e4_zt2_8gpu_20260614T050122Z`.
  - Reason: same algorithm and data path as the i2 run, but half the replay inner epochs. This isolates whether replay update strength is too aggressive.
  - zt2 GPUs were free; existing zt2 watchers/supervisors were not stopped.
  - Latest status: still running, no checkpoint yet.
  - Latest logged scalar window: step `19 -> 179`, `ppo_replay_valid_ratio=1.0` throughout, latest ratio mean `0.99916`, clip fraction `0.0`, approx KL `2.30e-6`, reward `0.63984`, base reward `0.69427`, safe ratio `0.82813`, DDC `0.91992`.
  - Decision: keep. This is the cleaner conservative replay-strength ablation and may be preferable if i2 continues to show weak safety windows.
  - Evaluation watcher: `stage3_grpo_replay_1epoch_s16_i1_lr1e4_zt2_8gpu_20260614T050122Z_navtest_exact_eval`.
- Do not resume: pure AWAC/IQL and AWAC+DPO variants listed above.
  - Reason: buffer quality is good, but policy absorption failed; current variants are not mature enough to justify more full-scale compute.
- Do not resume: `2e-4` GRPO/buffer-guided variants as main evidence.
  - Reason: the exact epoch-0 evaluation from the `2e-4` path was low (`PDMS 0.872059`), and the LR confound is already known.
- Treat any new GRPO replay run as diagnostic until it has passed smoke, short-run training diagnostics, and at least one controlled checkpoint evaluation.

## 2026-06-14 Attempt: Stage3 GRPO PPO Replay Diagnostic

Motivation:
- Previous evidence: original/Safe DiffGRPO is the strongest confirmed family (`0.9055` historical, `0.906184` confirmed), while AWAC/IQL discovered better train candidates but did not make the diffusion sampler emit better trajectories.
- Failure mode targeted: weak policy absorption and single-update GRPO instability.
- Reference sources checked:
  - DDPO paper/code: https://arxiv.org/abs/2305.13301 and https://github.com/kvablack/ddpo-pytorch
  - DPPO paper/code: https://arxiv.org/abs/2409.00588 and https://github.com/irom-princeton/dppo
  - RIPT-VLA code: https://github.com/Ariostgx/ript-vla
  - Diffusion-DPO code for preference mechanics, not used as the main path here: https://github.com/SalesforceAIResearch/DiffusionDPO
- Mechanism copied: collect a fixed rollout, store old trajectory logprob, compute group/leave-one-out style advantages, then replay the fixed rollout with clipped PPO ratios over multiple inner minibatch updates.

Implementation completeness:
- Required pieces present:
  - `stage3_objective=grpo_replay` objective switch.
  - Fixed rollout collection with sampled chains, trajectories, rewards, PDM submetrics, old trajectory logprob, and transformed advantages.
  - PPO clipped ratio loss against old logprob.
  - Multi-inner-epoch/minibatch manual optimization in Lightning.
  - Manual AMP-aware gradient clipping through `self.clip_gradients`; launcher sets `trainer.params.gradient_clip_val=null` for replay.
  - BC trust-region update after replay when `grpo_ppo_replay_bc_update=true`.
  - Logs for `ppo_replay_*`, `gspo_ratio_*`, `gspo_approx_kl`, NC/DAC/TTC/DDC/TLC, and advantage statistics.
- Known simplifications:
  - Uses trajectory-level reduced diffusion logprob, not per-denoising-step PPO yet.
  - No critic/value function.
  - No dynamic re-sampling for all-success/all-failure groups yet; current implementation filters near-zero advantages.
- Why acceptable now:
  - This is the minimum diagnostic needed before a full DPPO-style implementation. A bad result here cannot rule out DPPO/RIPT-VLA, but it can tell whether replay and old-logprob ratio mechanics are active in this codebase.

Smoke result:
- Command family: `STAGE3_OBJECTIVE=grpo_replay`, `LR=1e-4`, `GRPO_SAMPLE_TIME=2`, `GRPO_USE_GSPO_RATIO=true`, `GRPO_NORMALIZE_ADVANTAGE_BATCH=true`, `GRPO_ADVANTAGE_CLIP_ABS=3.0`, `GRPO_PPO_REPLAY_INNER_EPOCHS=2`, `GRPO_PPO_REPLAY_MINIBATCH_SIZE=1`, `MAX_SCENES=8`, `LIMIT_TRAIN_BATCHES=1`.
- Output root: `/mnt/project/VLA-AD/outputs/stage3_grpo_replay_smoke_1gpu_nullclip_20260614T041722Z`.
- Result: passed one real training batch and saved `epoch=0-step=5.ckpt`.
- Key diagnostics:
  - `train/ppo_replay_optimizer_steps_step = 5.0` (2 samples x 2 inner epochs + 1 BC update)
  - `train/ppo_replay_valid_ratio_step = 1.0`
  - `train/ppo_replay_valid_count_step = 2.0`
  - `train/ppo_replay_minibatch_valid_ratio_step = 1.0`
  - `train/ppo_replay_loss_active_step = 1.0`
  - `train/gspo_ratio_mean_step = 0.979439`
  - `train/gspo_ratio_clip_frac_step = 0.0`
  - `train/gspo_approx_kl_step = 0.000214`
  - `train/grpo_advantage_std_after_transform_step = 1.0`
  - `train/mean_nc_step = 1.0`, `train/mean_dac_step = 1.0`, `train/mean_ttc_step = 1.0`
- Decision:
  - Implementation smoke is now meaningful enough for a short diagnostic run.
  - Do not launch full training yet. First run a short 1-epoch/limited-batch diagnostic and inspect replay ratio, KL, advantage activity, safe submetrics, and optimizer-step count.

## Revised Experiment Plan

This plan replaces "try another variant" with gated experiments. A run is meaningful only if it isolates one variable, has mature implementation pieces for the method family, and can be stopped on objective diagnostics before spending a full training cycle.

### Comparison Protocol

Do not compare checkpoints across unmatched training lengths as final evidence.

Rules:
- Short-run replay checkpoints, such as a 1-epoch run, may only be compared against same-epoch or same-step controls.
- The historical `0.9055` original Stage3 result was trained for 10 epochs. It is a final-training target, not a fair comparator for a 1-epoch replay diagnostic.
- The confirmed `0.906184` Safe DiffGRPO result is `epoch_12-step_17290`; it is also a long-run target, not a short-run comparator.
- The user-reported original Stage3 early-training reference is already around `0.88+` PDMS at `epoch0-1`; use that as the short-run sanity gate until a locally aligned checkpoint/control is available.
- If an early replay/preference/offline-RL checkpoint is materially below `0.88` PDMS at a comparable epoch or step, treat it as a failed promotion gate unless another diagnostic proves the method needs more burn-in for a principled reason.
- A 1-epoch replay run can pass the promotion gate if it is competitive with the zt3 original-LR GRPO control at the same completed epoch and does not regress DDC/TTC.
- A method can only claim "better than original/Safe DiffGRPO" after running for comparable epochs/steps with the same navtest evaluation protocol.
- Because `run_training_recogdrive_rl.py` saves checkpoints every epoch via `ModelCheckpoint(save_top_k=-1, every_n_epochs=1, save_on_train_epoch_end=True)`, current active runs can be aligned by completed epoch once their first checkpoints appear.
- For replay objectives, same epoch is not always the same optimizer-step budget because inner PPO replay and BC updates increase Lightning `global_step`. Future replay/control launches should enable optional train-step checkpoints and compare both epoch-aligned and step-aligned PDMS.

### Checkpointing For Step-Aligned Comparisons

Patch added on 2026-06-14:
- `navsim/planning/script/run_training_recogdrive_rl.py` now supports optional extra step checkpoints through `checkpoint.every_n_train_steps`.
- `scripts/training/run_recogdrive_stage3_rl_2b_local.sh` exposes this as `CHECKPOINT_EVERY_N_TRAIN_STEPS`, while preserving the default epoch checkpoint path with `CHECKPOINT_EVERY_N_EPOCHS=1`.
- Step checkpoints are written under `step_checkpoints/` so existing epoch checkpoint watchers remain compatible.
- Default behavior is unchanged for the base launcher: no step checkpoints unless `CHECKPOINT_EVERY_N_TRAIN_STEPS > 0`.
- RLOO self-imitation launches now default to `CHECKPOINT_EVERY_N_TRAIN_STEPS=300` so new algorithm attempts can be screened before epoch0 when the original Stage3 early reference is already `0.88+` PDMS.
- `scripts/training/gate_recogdrive_stage3_early_pdms.py` reads watcher `checkpoint_eval_submetrics.tsv` files and returns a read-only `wait/continue/watch/stop` decision. Use it after the first step checkpoint eval; it does not kill processes automatically.

### Phase 0: Keep Only Clean Controls And Mature Diagnostics

Current keep list:
- zt3 original-LR GRPO control: `stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z`.
- local PPO replay i2 one-epoch run: `stage3_grpo_replay_1epoch_s16_i2_lr1e4_8gpu_20260614T044444Z`.
- zt2 PPO replay i1 one-epoch run: `stage3_grpo_replay_1epoch_s16_i1_lr1e4_zt2_8gpu_20260614T050122Z`.
- zt3 PPO replay i1 step-checkpoint diagnostic: `stage3_grpo_replay_stepckpt_s16_i1_lr1e4_zt3_2gpu_20260614T0610Z`.
- Existing zt3 control eval watchers: keep the already-running primary/secondary watchers; do not launch a third watcher for the same checkpoint stream.

Current stop/avoid list:
- Pure AWAC/IQL, AWAC+DPO, and simplified preference variants until implementation is upgraded to a mature offline/preference diffusion policy method.
- `2e-4` GRPO and buffer-guided variants as a default route, because the exact epoch-0 eval was `PDMS 0.872059`, far below the known Stage3/Safe DiffGRPO level.
- Any run without ratio/KL/clip/advantage/safety diagnostics.

No new full training should start until at least one of the current one-epoch replay runs has an exact navtest checkpoint evaluation.

### Phase 1: Finish The Three Active Evidence Runs

Expected outputs:
- zt3 original-LR GRPO control checkpoint and exact PDMS/submetrics.
- local replay i2 epoch checkpoint and exact PDMS/submetrics.
- zt2 replay i1 epoch checkpoint and exact PDMS/submetrics.

Interpretation:
- If zt3 `1e-4` is near the historical `0.9055` / Safe DiffGRPO `0.906184`, use it as the primary control for all future algorithm comparisons.
- If replay i1/i2 beats or matches zt3 while preserving DDC/TTC, promote replay to the next multi-epoch candidate.
- If replay i2 is worse but i1 is stable, reduce replay update strength rather than abandoning PPO replay.
- If both replay variants are clearly below the original-LR GRPO control, do not launch 20-epoch replay. Inspect whether trajectory-level reduced logprob is too coarse and move to denoising-step-level PPO before spending more full-run compute.

Hard stop criteria before epoch end:
- NaN/inf in loss, reward, logprob ratio, or KL.
- Sustained `ppo_replay_valid_ratio < 0.8`.
- Sustained high clip saturation, not just isolated windows.
- Repeated rollout safety collapse: NC/DAC/TTC/DDC or safe ratio stays far below control windows across several logs.

Current read:
- None of the three active runs meets a hard stop criterion.
- i2 is higher-update-strength and should be treated as riskier.
- i1 is the cleaner conservative ablation and should remain running even if i2 later looks weak.

### Phase 2: Decide Whether PPO Replay Merits Multi-Epoch Training

Only launch a 20-epoch PPO replay run if a one-epoch replay checkpoint is competitive enough to be informative:
- Not more than normal eval noise below zt3 original-LR GRPO.
- Does not regress DDC/TTC relative to the GRPO control.
- Replay diagnostics show active but controlled ratio movement: finite small KL, non-saturated clip fraction, and nonzero useful advantage coverage.

Default candidate if Phase 1 passes:
- LR `1e-4`, scheduler horizon `20` epochs down to `1e-5`.
- `sample_time=16`.
- BC anneal `0.10 -> 0.05`.
- reference KL `0.02`.
- `stage3_objective=grpo_replay`.
- Start from the better of i1 vs i2:
  - i1 if i2 has unstable safety/clip windows;
  - i2 only if it improves PDMS without DDC/TTC regression.

Do not change LR, sample_time, buffer loss, and replay strength simultaneously. One new variable per experiment.

### Phase 3: Upgrade Replay Toward Full DPPO/RIPT-VLA Before Calling It SOTA-Level

Trajectory-level replay is a necessary intermediate, not the final mature implementation. If Phase 1 shows promise but not enough PDMS:
- Add denoising-step-level old logprobs and PPO ratios instead of only reduced trajectory logprob.
- Add dynamic group filtering/resampling for all-success/all-failure or near-zero-advantage groups, matching the RIPT-VLA style failure mode.
- Consider a value/critic baseline only after advantage-normalized PPO replay still shows high variance.

This phase is implementation work first, not a blind full run. It needs smoke tests proving:
- old per-step logprobs stay fixed across replay epochs;
- current per-step logprobs change after optimizer steps;
- KL/clip are computed over the same denoising timesteps;
- memory fits at `sample_time=16` or a documented equivalent.

### Phase 4: Add Buffer Knowledge Only Through Proven Absorption Channels

The elite buffer is useful, but previous AWAC showed that good targets alone do not guarantee good sampled trajectories. Buffer work resumes only after the base replay path is stable.

Allowed buffer auxiliaries, one at a time:
- Reward-neighborhood bonus for sampled trajectories close to high-PDMS valid buffer targets.
- Low-noise denoising distillation toward valid elite targets.
- Self-imitation on current sampled trajectories that are hard-safe and above baseline.
- Preference/DPO auxiliary using shared noise/timestep and a frozen reference, only when active-pair diagnostics pass.

Required diagnostics:
- target coverage and target distance;
- distill/self-imitation weight sums;
- winner/loser active-pair ratio for DPO;
- current/reference winner-loser loss margins and implicit accuracy;
- DDC/TTC/NC/DAC protection.

Stop buffer variants if:
- target ratios are near zero;
- target distance does not improve;
- DDC/TTC regress;
- PDMS gains come only from EP while safety declines.

### Phase 5: Revisit AWAC/IQL Only As A Mature Offline Diffusion RL Method

Do not continue the existing AWAC/IQL line as-is. A future AWAC/IQL revisit must include:
- matched LR/effective optimizer steps against GRPO;
- diagnostics proving elite-target likelihood increases under the sampler;
- behavior-support/trust-region checks;
- strict valid-mask persistence and no invalid candidate training by default;
- preference or Q/value structure aligned with a reference implementation, not only weighted denoising regression.

Until then, AWAC results are evidence of a policy-absorption failure in our implementation, not evidence that offline RL or preference learning is inherently unsuitable.

## 2026-06-14 06:09 UTC Active Run Snapshot

No active run has produced a checkpoint yet, so there is no new PDMS result and no basis to claim a better method.

Current diagnostics:
- Local replay i2 `stage3_grpo_replay_1epoch_s16_i2_lr1e4_8gpu_20260614T044444Z`:
  - Still running, no checkpoint.
  - Latest logged step `799`: reward `0.59714`, base reward `0.67684`, safe ratio `0.78516`, NC `0.84180`, DAC `0.95703`, TTC `0.67578`, DDC `0.95898`.
  - Replay mechanics remain active: `ppo_replay_valid_ratio=1.0`, `optimizer_steps=9`, ratio mean `0.96882`, clip fraction `0.04688`, approx KL `6.05e-4`.
  - Decision: keep running but do not promote i2 until epoch eval. Recent low reward/safety windows make it a watched diagnostic.
- zt2 replay i1 `stage3_grpo_replay_1epoch_s16_i1_lr1e4_zt2_8gpu_20260614T050122Z`:
  - Still running, no checkpoint.
  - Latest logged step `759`: reward `0.87667`, base reward `0.85840`, safe ratio `0.94922`, NC `0.98828`, DAC `0.96094`, TTC `0.92188`, DDC `0.95508`.
  - Conservative replay remains very stable: `ppo_replay_valid_ratio=1.0`, `optimizer_steps=5`, ratio mean `0.99749`, clip fraction `0.0`, approx KL `7.51e-6`.
  - Decision: keep running. This remains the cleaner replay-strength ablation.
- zt3 original-LR GRPO control `stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z`:
  - Still running, no checkpoint.
  - Latest logged step `399`: reward `0.83706`, base reward `0.83057`, TTC `0.84375`, safe ratio `0.92708`.
  - Decision: keep running. It is slow but progressing and is still the necessary LR/control baseline.

Decision:
- Do not stop any of these three runs.
- Do not launch a new full training run yet.
- The step-level PPO replay implementation is ready for a short diagnostic, but it should wait until at least one current one-epoch replay checkpoint has exact PDMS unless an active run fails.

## 2026-06-14 06:14 UTC Step-Checkpoint Diagnostic Launch

Reason:
- Existing replay i1/i2 jobs were launched before step checkpoint support. They are healthy but will only produce checkpoints at epoch end.
- Replay objectives increment Lightning `global_step` by inner PPO/BC optimizer updates, so waiting for epoch-only checkpoints delays same-step diagnostics.
- The original Stage3 early reference is already around `0.88+` PDMS at `epoch0-1`; replay should be screened earlier before committing to long multi-epoch runs.

Launched on zt3:
- Run: `stage3_grpo_replay_stepckpt_s16_i1_lr1e4_zt3_2gpu_20260614T0610Z`.
- GPUs: `6,7`; existing zt3 tasks were not stopped.
- Config: LR `1e-4`, `stage3_objective=grpo_replay`, `sample_time=16`, inner PPO replay epochs `1`, replay minibatch `8`, batch size `1`, 2 GPUs, `LIMIT_TRAIN_BATCHES=200`, `LIMIT_VAL_BATCHES=0`.
- Checkpointing: `CHECKPOINT_EVERY_N_TRAIN_STEPS=300`, `CHECKPOINT_EVERY_N_EPOCHS=1`.
- Eval watcher: `/mnt/project/VLA-AD/outputs/stage3_grpo_replay_stepckpt_s16_i1_lr1e4_zt3_2gpu_20260614T0610Z_navtest_step_eval`, configured for recursive checkpoint discovery under the training Hydra root.

Expected use:
- Use the first step checkpoint as a quick PDMS sanity check only.
- Do not compare this limited-batch diagnostic as final evidence against original `epoch10` or Safe DiffGRPO `epoch12`.
- If the step checkpoint is materially below the original early `0.88+` band, do not promote replay i1/i2 without fixing the method.
- If it is plausible, wait for the full one-epoch replay/control checkpoints for the real gate.

## 2026-06-14 06:24 UTC Active Run Snapshot

No PDMS result is available yet.

Current status:
- Local replay i2 `stage3_grpo_replay_1epoch_s16_i2_lr1e4_8gpu_20260614T044444Z`:
  - Still running, no checkpoint.
  - Latest logged step `1079`: reward `0.70439`, base reward `0.72044`, safe ratio `0.89844`, NC `0.91406`, DAC `0.98438`, TTC `0.82422`, DDC `0.85938`.
  - Replay diagnostics: valid ratio `1.0`, optimizer steps `9`, ratio clip fraction `0.15625`, approx KL `7.33e-4`.
- zt2 replay i1 `stage3_grpo_replay_1epoch_s16_i1_lr1e4_zt2_8gpu_20260614T050122Z`:
  - Still running, no checkpoint.
  - Latest logged step `1119`: reward `0.80203`, base reward `0.80814`, safe ratio `0.90625`, NC `0.96680`, DAC `0.94141`, TTC `0.89062`, DDC `1.0`.
  - Replay diagnostics: valid ratio `1.0`, optimizer steps `5`, ratio clip fraction `0.0`, approx KL `8.13e-6`.
- zt3 original-LR GRPO control `stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z`:
  - Still running, no checkpoint.
  - Latest logged step `449`: reward `0.89311`, base reward `0.86898`, safe ratio `0.95833`, TTC `0.93750`.
- zt3 step-checkpoint replay diagnostic `stage3_grpo_replay_stepckpt_s16_i1_lr1e4_zt3_2gpu_20260614T0610Z`:
  - `step-step=300.ckpt` was written at `2026-06-14T06:20:12Z`.
  - Watcher archived it as `step-step_300.ckpt` at `2026-06-14T06:22:59Z`.
  - Training completed successfully with return code `0` at `2026-06-14T06:24:39Z`.
  - Additional checkpoints were produced: `step-step=600.ckpt` and `epoch=0-step=600.ckpt`.
  - Eval of `step-step_300` started at `2026-06-14T06:24:59Z`; latest eval log shows progress through roughly `100 / 6069` navtest scenarios.
  - Latest logged step `79`: reward `0.22160`, base reward `0.44747`, safe ratio `0.5`, NC `0.95312`, DAC `0.5`, TTC `0.875`, DDC `0.76562`, replay valid ratio `0.5`.

Decision:
- Do not stop the full replay i1/i2 jobs before their first checkpoint.
- Do not judge the step diagnostic from its noisy training scalar windows; wait for exact navtest PDMS on `step-step_300.ckpt`.
- Let the watcher evaluate queued checkpoints in order; do not start duplicate evals for the same ckpts.
- If `step-step_300` is far below the original early `0.88+` PDMS band, replay needs method changes before promotion.

## 2026-06-14 06:46 UTC Active Run Snapshot

No new PDMS result is available yet.

Current status:
- Local replay i2 `stage3_grpo_replay_1epoch_s16_i2_lr1e4_8gpu_20260614T044444Z`:
  - Still running, no checkpoint.
  - Latest TensorBoard scalar step `1379`: reward `0.69689`, base reward `0.74012`, safe ratio `0.83984`, NC `0.97070`, DAC `0.86719`, TTC `0.91016`, DDC `0.98047`.
  - Replay diagnostics: valid ratio `1.0`, optimizer steps `9`, ratio clip fraction `0.0`, approx KL `1.54e-4`, reference KL loss `0.03040`, BC coeff `0.10`.
- zt2 replay i1 `stage3_grpo_replay_1epoch_s16_i1_lr1e4_zt2_8gpu_20260614T050122Z`:
  - Still running, no checkpoint.
  - Latest TensorBoard scalar step `1519`: reward `0.68934`, base reward `0.73539`, safe ratio `0.87109`, NC `0.94141`, DAC `0.95312`, TTC `0.90234`, DDC `0.90234`.
  - Replay diagnostics: valid ratio `1.0`, optimizer steps `5`, ratio clip fraction `0.0`, approx KL `1.20e-5`, reference KL loss `0.04611`, BC coeff `0.10`.
- zt3 original-LR GRPO control `stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z`:
  - Still running, no checkpoint.
  - Latest TensorBoard scalar step `499`: reward `0.85184`, base reward `0.83066`, safe ratio `0.95833`, TTC `0.84375`, reference KL loss `0.00900`, BC coeff `0.10`.
- zt3 step-checkpoint replay diagnostic `stage3_grpo_replay_stepckpt_s16_i1_lr1e4_zt3_2gpu_20260614T0610Z`:
  - `step-step_300` exact navtest eval is still running.
  - Latest eval log around `2026-06-14T06:43Z` shows progress through roughly `2067 / 6069` navtest scenarios.
  - `step-step_600` remains queued by the watcher.

Resource state:
- Local 8 A800 GPUs are occupied by replay i2 training.
- zt2 8 A800 GPUs are occupied by replay i1 training.
- zt3 is running the original-LR control plus the step-checkpoint eval; do not start duplicate evals or kill existing remote tasks.

Decision:
- Continue waiting for exact navtest PDMS. The replay training diagnostics show the objective is active and numerically stable, but recent safety/reward batches are not strong enough to promote without exact evaluation.
- Apply the `0.88+` original Stage3 `epoch0-1` sanity gate to the first comparable replay checkpoints.

## 2026-06-14 07:18 UTC Trajectory-Level Replay Gate Result

Exact navtest result:
- Run: `stage3_grpo_replay_stepckpt_s16_i1_lr1e4_zt3_2gpu_20260614T0610Z`.
- Checkpoint: `step-step_300`.
- Eval completed at `2026-06-14T07:16:11Z` with return code `0`.
- Valid rows: `12138`.
- PDMS: `0.816440`.
- Submetrics: NC `0.966263`, DAC `0.914483`, TTC `0.908304`, EP `0.775452`, comfort `0.998517`, DDC `0.951598`, TLC not reported by this summary.

Interpretation:
- This is materially below the user-reported original Stage3 early `epoch0-1` sanity band of `0.88+`.
- It is also below the stopped high-LR `2e-4` epoch0 result `0.872059`.
- The limited-batch step checkpoint is not a final same-epoch comparison, but the gap is too large to justify promoting the trajectory-level replay implementation.

Action taken:
- Stopped the trajectory-level replay long jobs:
  - local i2 `stage3_grpo_replay_1epoch_s16_i2_lr1e4_8gpu_20260614T044444Z`.
  - zt2 i1 `stage3_grpo_replay_1epoch_s16_i1_lr1e4_zt2_8gpu_20260614T050122Z`.
- Stopped the failed replay checkpoint watcher/eval streams:
  - local i2 watcher.
  - zt2 i1 watcher.
  - zt3 replay step-checkpoint watcher and the queued `step-step_600` eval.
- Kept the zt3 original-LR GRPO control and its zt2/zt3 watchers running.

Decision:
- Do not run or promote more trajectory-level PPO replay as-is.
- Treat the failure as a method/objective issue, not a simple LR or epoch-count issue.
- Next diagnostic should use the already implemented denoising-step-level PPO replay mode, because DDPO/DPPO operate on denoising transitions rather than one reduced trajectory logprob.
- The next run must be an isolated same-scale diagnostic against the failed trajectory-level setup: same LR `1e-4`, `sample_time=16`, `LIMIT_TRAIN_BATCHES=200`, 2 GPUs, but `GRPO_PPO_REPLAY_LOGPROB_MODE=step`.

## 2026-06-14 Attempt: Step-Level PPO Replay Implementation

Motivation:
- Previous evidence: AWAC/IQL found high-PDMS train candidates but did not make the diffusion sampler output better trajectories. Trajectory-level PPO replay is now running, but it still reduces the full denoising chain into one scalar logprob.
- Targeted failure mode: weak policy absorption caused by a coarse trajectory-level objective. DDPO/DPPO treat denoising transitions as the RL action sequence, so the PPO ratio should be able to operate at the denoising-step level.
- Reference sources:
  - DDPO paper/code: https://arxiv.org/abs/2305.13301 and https://github.com/kvablack/ddpo-pytorch
  - DPPO paper/code: https://arxiv.org/abs/2409.00588 and https://github.com/irom-princeton/dppo
  - RIPT-VLA code: https://github.com/Ariostgx/ript-vla

Implementation completed:
- Added optional `grpo_ppo_replay_logprob_mode` / `ppo_replay_logprob_mode`.
- Default is `trajectory`, preserving existing replay experiments.
- New `step` mode stores old per-denoising-step logprobs in the rollout and computes PPO ratios per denoising transition.
- Step-level surrogate is weighted by the existing normalized `gamma_denoising` discount.
- Logs `ppo_replay_step_logprob_mode` so runs can prove which path was active.
- No VLM, DiT architecture, trainable module, reward, or navtest-training path change.

Smoke test:
- Run: `stage3_grpo_replay_stepmode_smoke_1gpu_20260614T0530`
- Host/GPU: `training-rl-zt3`, `CUDA_VISIBLE_DEVICES=6`.
- Config: `MAX_SCENES=8`, `LIMIT_TRAIN_BATCHES=1`, `sample_time=2`, `BATCH_SIZE=1`, `GRPO_PPO_REPLAY_LOGPROB_MODE=step`, `GRPO_PPO_REPLAY_INNER_EPOCHS=1`, `GRPO_PPO_REPLAY_MINIBATCH_SIZE=2`.
- Result: completed one training batch without NaN/OOM.
- Key scalars:
  - `train/ppo_replay_step_logprob_mode_step = 1.0`
  - `train/ppo_replay_valid_ratio_step = 1.0`
  - `train/ppo_replay_optimizer_steps_step = 2.0`
  - `train/gspo_ratio_mean_step = 1.0`
  - `train/gspo_approx_kl_step = 0.0`
  - `train/reward_step = 0.91329`
  - `train/safe_ratio_step = 1.0`

Decision:
- This is an implementation maturity improvement, not a better-PDMS result yet.
- Do not launch a full step-level run immediately.
- First wait for the current trajectory-level replay i1/i2 one-epoch PDMS results and zt3 original-GRPO control.
- If trajectory-level replay is promising but not clearly better, run a short step-level diagnostic before any 20-epoch full run.
- If trajectory-level replay is already worse than the control, step-level PPO becomes the next implementation-focused diagnostic rather than another hyperparameter tweak.

### 2026-06-14 07:27 UTC Step-Level PPO Diagnostic Launch Plan

Reason:
- The trajectory-level replay gate failed with PDMS `0.816440` at `step-step_300`, which is far below the original Stage3 early `0.88+` band.
- This points to a coarse-objective absorption issue: reducing a diffusion rollout to one trajectory logprob is not aligned with DDPO/DPPO, which optimize denoising transitions.

Implementation fix before launch:
- `scripts/training/launch_recogdrive_stage3_rl_2b_local_stable.sh` now forwards:
  - `GRPO_PPO_REPLAY_LOGPROB_MODE`
  - `GPUS`, `GPUS_PER_NODE`
  - `CHECKPOINT_EVERY_N_EPOCHS`, `CHECKPOINT_EVERY_N_TRAIN_STEPS`
  - `TRAINER_GRADIENT_CLIP_VAL`
- Dry-run verified that `agent.grpo_ppo_replay_logprob_mode=step`, `trainer.params.devices=2`, and `checkpoint.every_n_train_steps=300` reach the final Hydra command.

Planned isolated diagnostic:
- Run name: `stage3_grpo_replay_stepmode_stepckpt_s16_i1_lr1e4_zt3_2gpu_20260614T0727Z`.
- Training host/GPU: zt3 GPUs `6,7`.
- Eval watcher: local 2 GPUs watching the shared output directory.
- Config held equal to the failed trajectory-level diagnostic where possible:
  - LR `1e-4`
  - `stage3_objective=grpo_replay`
  - `sample_time=16`
  - `GRPO_PPO_REPLAY_INNER_EPOCHS=1`
  - `GRPO_PPO_REPLAY_MINIBATCH_SIZE=8`
  - `GRPO_USE_GSPO_RATIO=true`
  - `GRPO_NORMALIZE_ADVANTAGE_BATCH=true`
  - `GRPO_ADVANTAGE_CLIP_ABS=3.0`
  - `REFERENCE_KL_COEFF=0.02`
  - `REFERENCE_KL_CHUNK_SIZE=16`
  - `BATCH_SIZE=1`, `GPUS_PER_NODE=2`
  - `LIMIT_TRAIN_BATCHES=200`, `LIMIT_VAL_BATCHES=0`
  - step checkpoints every `300` train steps.
- Single intended variable: `GRPO_PPO_REPLAY_LOGPROB_MODE=step` instead of `trajectory`.

Promotion gate:
- User clarification on `2026-06-14`: original Stage3 is already around `0.88+` PDMS at `epoch0-1`; use that as the short-run comparator, not the 10-epoch `0.9055` result.
- If `step-step_300` remains far below `0.88`, step-level replay alone is not enough and we should inspect reward/action-distribution mismatch or return to original GRPO with better reward shaping.
- If `step-step_300` materially improves over `0.816440`, continue evaluating queued `step-step_600` / epoch checkpoint and consider a larger run.

Launch status:
- Training launched on zt3 at `2026-06-14T07:29Z`.
- Output root: `/mnt/project/VLA-AD/outputs/stage3_grpo_replay_stepmode_stepckpt_s16_i1_lr1e4_zt3_2gpu_20260614T0727Z`.
- Verified resolved command contains `agent.grpo_ppo_replay_logprob_mode=step`, `trainer.params.devices=2`, and `checkpoint.every_n_train_steps=300`.
- Local watcher launched with output root `/mnt/project/VLA-AD/outputs/stage3_grpo_replay_stepmode_stepckpt_s16_i1_lr1e4_zt3_2gpu_20260614T0727Z_navtest_step_eval`.
- Initial watcher state: no pending checkpoint, training state running.
- Training finished cleanly at `2026-06-14T07:40:13Z` with return code `0`.
- Produced checkpoints:
  - `step-step=300.ckpt` at `2026-06-14T07:35:47Z`
  - `step-step=600.ckpt` at `2026-06-14T07:39:29Z`
  - `epoch=0-step=600.ckpt` at `2026-06-14T07:39:42Z`
- The local watcher archived `step-step=300` and started exact navtest evaluation at `2026-06-14T07:39:45Z`; PDMS is pending.
- A second watcher was launched on `training-vla-zt2` at `2026-06-14T07:45:23Z` using GPUs `0-7`.
  - Eval root: `/mnt/project/VLA-AD/outputs/stage3_grpo_replay_stepmode_stepckpt_s16_i1_lr1e4_zt3_2gpu_20260614T0727Z_zt2_8gpu_step600_eval`.
  - It uses the local watcher summary as `EXTERNAL_SUMMARY_TSV`, so `step-step_300` is skipped there and `step-step_600` is evaluated first.
  - This keeps the early gate aligned: local `step300` checks the first update point; zt2 `step600` checks whether the method recovers by the end of the limited diagnostic.
- zt2 `step-step_600` exact navtest completed at `2026-06-14T07:59:19Z`, return code `0`.
  - PDMS `0.824322`, NC `0.975943`, DAC `0.913330`, TTC `0.927748`, EP `0.770388`, comfort `0.999753`, DDC `0.979980`, valid rows `12138`.
  - This is still far below the original Stage3 `epoch0-1` sanity gate of `0.88+`, so the current simplified step-level replay implementation fails the promotion gate.
  - zt2 then started the duplicate `epoch_0-step_600` eval automatically; it was stopped intentionally because the same update point was already evaluated.
- Local `step-step_300` exact navtest completed at `2026-06-14T08:27:05Z`, return code `0`.
  - PDMS `0.787823`, NC `0.958436`, DAC `0.900066`, TTC `0.893557`, EP `0.742667`, comfort `1.000000`, DDC `0.947067`, valid rows `12138`.
  - This is worse than both the failed trajectory-level replay and the later all-step `step600` checkpoint, so the simplified step-level replay is rejected.
  - The local watcher then started a duplicate `step-step_600` eval; it was stopped because zt2 had already completed `step-step_600`.

Implementation audit while evals are running:
- DPPO official PPO diffusion code clamps both old/new logprobs, normalizes/clips advantages, discounts by denoising step, and samples PPO minibatches across `(environment step, denoising step)` rather than only across trajectory rows.
- DPPO also uses step-dependent PPO clipping and an optional value/critic loss. We currently use a fixed clip range and no critic.
- Our `step` mode is therefore a meaningful improvement over trajectory-level replay, but it is still not a complete DPPO/RIPT-VLA-equivalent implementation.
- Since `step600` remains below the `0.88+` early gate, the next implementation target should be mature DPPO-style transition sampling and step-dependent clipping before any new LR or buffer-tuning run.

### 2026-06-14 Attempt: DPPO-Style Transition Replay Implementation

Motivation:
- The failed trajectory-level replay (`step300` PDMS `0.816440`) and simplified all-step replay (`step600` PDMS `0.824322`) both underperform the original Stage3 early `0.88+` PDMS band.
- The failure pattern suggests the issue is not just LR or epoch count: diffusion policy improvement needs to optimize denoising transitions in a way closer to DDPO/DPPO/RIPT-VLA instead of treating the whole sampled trajectory as one coarse action.

Reference sources:
- DDPO optimizes diffusion sampling trajectories using PPO-style likelihood ratios over denoising steps.
- DPPO samples replay minibatches over diffusion transitions, clips old/new logprobs, applies denoising-step discounting, and uses step-dependent PPO clipping.
- RIPT-VLA applies RL post-training to VLA/action-generation policies with PPO-style rollout replay rather than plain behavior cloning onto discovered trajectories.

Implementation completeness:
- Added `GRPO_PPO_REPLAY_STEP_MINIBATCH_MODE=transition`, which changes the manual PPO replay minibatch unit from sampled trajectory rows to `(sample, denoising_step)` transitions when `GRPO_PPO_REPLAY_LOGPROB_MODE=step`.
- Added configurable step logprob clamp:
  - `GRPO_PPO_REPLAY_LOGPROB_CLAMP_MIN=-5.0`
  - `GRPO_PPO_REPLAY_LOGPROB_CLAMP_MAX=2.0`
- Added DPPO-style denoising-step clip schedule:
  - `GRPO_PPO_REPLAY_STEP_CLIP_SCHEDULE=dppo_exp`
  - `GRPO_PPO_REPLAY_STEP_CLIP_BASE=0.001`
  - `GRPO_PPO_REPLAY_STEP_CLIP_RATE=3.0`
- Kept backward compatibility:
  - default `GRPO_PPO_REPLAY_STEP_MINIBATCH_MODE=trajectory_all_steps`
  - default `GRPO_PPO_REPLAY_STEP_CLIP_SCHEDULE=constant`
  - old trajectory and all-step replay behavior remain available.

Known simplifications:
- No critic/value head has been added.
- Reference KL still computes on selected rollout rows rather than only selected denoising transitions.
- This implementation is closer to DPPO than the previous replay, but still not a full DPPO reproduction.

Smoke test:
- Run root: `/mnt/project/VLA-AD/outputs/stage3_dppo_transition_smoke2_20260614T082252Z`.
- Config: `MAX_SCENES=8`, `LIMIT_TRAIN_BATCHES=1`, `sample_time=2`, `BATCH_SIZE=1`, `LR=1e-4`, `REFERENCE_KL_COEFF=0`, transition minibatch size `4`.
- Result: completed one training batch without NaN/OOM/autograd error.
- Key TensorBoard scalars:
  - `train/ppo_replay_step_logprob_mode_step = 1.0`
  - `train/ppo_replay_step_minibatch_mode_step = 1.0`
  - `train/ppo_replay_transition_mode_step = 1.0`
  - `train/ppo_replay_optimizer_steps_step = 4.0`
  - `train/ppo_replay_step_clip_min_step = 0.003868`
  - `train/ppo_replay_step_clip_mean_step = 0.013330`
  - `train/ppo_replay_step_clip_max_step = 0.022791`
  - `train/ppo_replay_logprob_clamped_frac_step = 0.0`

Next diagnostic:
- Run a short 2-GPU transition replay diagnostic only after the smoke-tested code is committed.
- Use the original Stage3 `epoch0-1 ~= 0.88+` as the short-run gate.
- If the new transition replay still scores far below `0.88`, do not proceed to full training. Next step should be either:
  - complete DPPO with critic/value or stronger KL/ratio controls, or
  - revert to original GRPO baseline and add buffer-derived preference/self-imitation only after verifying the policy can absorb the signal.

Launch status:
- Code committed and pushed as `06c79ad Add DPPO-style transition replay for Stage3 GRPO`.
- Run name: `stage3_grpo_dppo_transition_s16_i1_lr1e4_zt2_2gpu_20260614T083005Z`.
- Training host/GPU: `training-vla-zt2`, GPUs `0,1`.
- Eval watcher: local 8-GPU watcher at `/mnt/project/VLA-AD/outputs/stage3_grpo_dppo_transition_s16_i1_lr1e4_zt2_2gpu_20260614T083005Z_navtest_eval`.
- Config:
  - `MAX_EPOCHS=1`, `LIMIT_TRAIN_BATCHES=200`
  - `LR=1e-4`, `sample_time=16`
  - `GRPO_PPO_REPLAY_LOGPROB_MODE=step`
  - `GRPO_PPO_REPLAY_STEP_MINIBATCH_MODE=transition`
  - `GRPO_PPO_REPLAY_MINIBATCH_SIZE=32`
  - `GRPO_PPO_REPLAY_STEP_CLIP_SCHEDULE=dppo_exp`
  - `REFERENCE_KL_COEFF=0.02`, `REFERENCE_KL_CHUNK_SIZE=16`
  - `GRPO_USE_GSPO_RATIO=true`
  - `GRPO_NORMALIZE_ADVANTAGE_BATCH=true`
  - `GRPO_ADVANTAGE_CLIP_ABS=3.0`
  - step checkpoints every `300` train steps.
- Fairness note:
  - `num_denoising_steps` is `4` in the smoke run. With `sample_time=16` and transition minibatch `32`, PPO replay should perform about two transition minibatches plus one BC update per train batch, matching the previous simplified step replay's optimizer-step cadence more closely than a smaller transition minibatch would.

Evaluation result:
- The original torchrun watcher launched for `step-step_300` lost its parent and left orphan ranks before scoring. The final reported numbers below use independent exact-PDMS token shards with no scoring formula changes:
  - `step-step_300`: local 8 independent 1GPU shards.
  - `step-step_600`: zt3 2 independent 1GPU shards on idle GPUs 6 and 7.
  - `epoch=0-step=800`: local 8 independent 1GPU shards.
- Full navtest exact PDMS results:

| Checkpoint | Eval root | PDMS | NC | DAC | TTC | EP | Comfort | DDC | TLC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `step-step_300` | `/mnt/project/VLA-AD/outputs/stage3_grpo_dppo_transition_s16_i1_lr1e4_zt2_2gpu_20260614T083005Z_navtest_exact_shards_step300_20260614T0900Z` | `0.744740` | `0.954894` | `0.848163` | `0.878234` | `0.719376` | `0.996375` | `0.942289` |  |
| `step-step_600` | `/mnt/project/VLA-AD/outputs/stage3_grpo_dppo_transition_s16_i1_lr1e4_zt2_2gpu_20260614T083005Z_navtest_exact_shards_step600_zt3_2gpu_20260614T0858Z` | `0.801195` | `0.972524` | `0.891251` | `0.923875` | `0.752611` | `0.999423` | `0.947767` |  |
| `epoch=0-step=800` | `/mnt/project/VLA-AD/outputs/stage3_grpo_dppo_transition_s16_i1_lr1e4_zt2_2gpu_20260614T083005Z_navtest_exact_shards_epoch0_step800_20260614T0912Z` | `0.831015` | `0.978003` | `0.912836` | `0.927583` | `0.786319` | `0.999423` | `0.961691` |  |

Conclusion:
- This run fails the user-reported original Stage3 early gate of `0.88+` at the matched epoch0 point.
- The trajectory quality recovers over the limited 1-epoch run, but the recovery is not close enough; the gap at epoch0 is about `-0.049` PDMS versus the early gate.
- Main regressions versus the expected early Stage3 band are DAC and EP, with TTC/DDC also below a healthy original Stage3 trajectory distribution.
- Do not promote this DPPO-style transition replay variant to full training. Any next GRPO redesign should revisit the objective itself, not only tune LR or minibatch count.

### 2026-06-14 Protocol Update: early Stage3 gate

Baseline:
- The original ReCogDrive Stage3 run is reported to reach roughly `0.88+` navtest PDMS by epoch0-1.
- Therefore new Stage3 variants must be compared at matched early checkpoints, not only at epoch10 or later.

Operational rule:
- Use `scripts/training/gate_recogdrive_stage3_early_pdms.py` with `--threshold 0.88 --margin 0.005`.
- Treat `PDMS < 0.875` at the selected early checkpoint as a stop signal.
- Treat `0.875 <= PDMS < 0.88` as a watch zone: allow at most the next early checkpoint, but do not promote to full training.
- Treat `PDMS >= 0.88` as the minimum condition to continue; it is not success by itself because the final target remains beating the original Stage3/Safe DiffGRPO best band.

Automation:
- `scripts/training/watch_recogdrive_stage3_early_gate.sh` polls the gate output for a run.
- With `STOP_ON_FAIL=0`, it reports the decision and leaves training untouched.
- With `STOP_ON_FAIL=1`, it only terminates the local training process group recorded by the gate JSON. It does not terminate remote evaluation watchers or unrelated remote tasks.
- For new stable launches, set `START_EARLY_GATE_WATCHER=1 EARLY_GATE_THRESHOLD=0.88 EARLY_GATE_MARGIN=0.005`.
- Use `EARLY_GATE_STOP_ON_FAIL=1` only when the run root is local and the training status JSON records the correct process group.
- The GRPO/GSPO launcher now defaults to `CHECKPOINT_EVERY_N_TRAIN_STEPS=300` and starts the early gate watcher.
- The buffer-guided GRPO wrapper defaults to `CHECKPOINT_EVERY_N_TRAIN_STEPS=300`, `START_EARLY_GATE_WATCHER=1`, and `EARLY_GATE_STOP_ON_FAIL=1`, so any future buffer-absorption run must clear the original early Stage3 gate before becoming a long run.

### 2026-06-14 Attempt: GRPO self-imitation safetygate stopped before checkpoint

Motivation:
- The previous cap05 run reached `0.887107` at step 900 but improved EP while regressing NC/TTC/DDC.
- The next test added stricter self-imitation target gates so only high-reward, NC/DAC/TTC/DDC-safe generated samples could become distillation targets.

Config:
- Run name: `stage3_grpo_rloo_selfimit_safetygate_step300_s16_lr1e4_b2acc4_8gpu_20260614T145712Z`
- LR: `1e-4`.
- `sample_time=16`, batch size `2`, accumulation `4`, 8 local GPUs.
- Self-imitation target gates: NC `1.0`, DAC `1.0`, TTC `0.95`, DDC `0.99`.
- Main GRPO group filtering still only hard-gated NC/DAC, not TTC/DDC.

Observed training diagnostics before stopping:
- Last logged step: `249`.
- `train/reward_step`: `0.766856`.
- `train/base_reward_step`: `0.778479`.
- `train/safe_ratio_step` / `hard_safe_ratio`: `0.898438`.
- `train/mean_ep_step`: `0.783179`.
- `train/mean_ttc_step`: `0.886719`.
- Self-imitation was active: target scene ratio `0.5`, target reward mean `0.987467`.

Decision:
- Stop before the first checkpoint and navtest evaluation.
- Reason: this run only constrained auxiliary self-imitation targets, while the primary GRPO update could still accept TTC/DDC-regressing positive-advantage samples. That makes it a lower-information variant than a main-objective hard-gate run.
- Do not repeat this exact configuration.

### 2026-06-14 Attempt: main GRPO TTC/DDC hard gate first launch

Motivation:
- Step300 to step900 diagnostics from the best cap05 run indicate PDMS gains were dominated by EP, while TTC/DDC and NC regressed.
- The targeted next change is to hard-mask the main GRPO advantage for samples that fail TTC/DDC, so progress-improving but direction/timing-unsafe samples do not contribute positive policy gradient.

Implementation:
- Added agent-level controls:
  - `grpo_hard_gate_ttc`
  - `grpo_hard_gate_ddc`
  - `grpo_ttc_safe_threshold`
  - `grpo_ddc_safe_threshold`
- Wired launcher env/Hydra passthrough and `GRPOConfig` assignment.

Launch issue:
- First launch name: `stage3_grpo_rloo_selfimit_mainhardgate_step300_s16_lr1e4_b2acc4_8gpu_20260614T160328Z`.
- Hydra failed before training:
  - `Could not override 'agent.grpo_hard_gate_ttc'`
  - Root cause: the Python constructor and scripts were patched, but the structured agent YAML lacked default keys.

Decision:
- This is a configuration-wiring failure, not an algorithm result.
- Fix by adding the four default fields to `navsim/planning/script/config/common/agent/recogdrive_agent.yaml`, then relaunch under a new run name.

### 2026-06-14 Attempt: main GRPO hard gate v2 launch wiring

Launch issue:
- Run name: `stage3_grpo_rloo_selfimit_mainhardgate_v2_step300_s16_lr1e4_b2acc4_8gpu_20260614T161141Z`.
- Hydra parsing succeeded after adding the YAML defaults.
- The shell preflight failed before training:
  - `ELITE_BUFFER_DIR must exist when OFFLINE_RL_ENABLED/GRPO_BUFFER_GUIDANCE_ENABLED is true`.

Root cause:
- Self-imitation in `forward_grpo` is gated by `offline_cfg.enabled`.
- Therefore `OFFLINE_RL_ENABLED=true` is needed for this specific self-imitation variant, even when buffer guidance and buffer distillation are disabled.
- The launcher safety check still requires a real train-only `ELITE_BUFFER_DIR` whenever offline RL is enabled.

Decision:
- This is a launch-contract error, not an algorithm result.
- Do not repeat this exact startup command.
- Correct startup for self-imitation GRPO hard-gate runs:
  - `OFFLINE_RL_ENABLED=true`
  - pass a real train-only elite buffer directory
  - keep `GRPO_BUFFER_GUIDANCE_ENABLED=false`, `GRPO_BUFFER_DISTILL_LOSS_WEIGHT=0.0`, and `GRPO_BUFFER_REWARD_BONUS_WEIGHT=0.0` if the experiment should isolate online GRPO + self-imitation.

### 2026-06-14 Attempt: main GRPO TTC/DDC hard gate v3 running

Motivation:
- Preserve the cap05 run's EP upside while preventing positive policy-gradient updates from TTC/DDC-regressing samples.
- Keep the online self-imitation auxiliary loss, but gate targets by NC/DAC/TTC/DDC and cap target scenes at `0.5`.

Config:
- Run name: `stage3_grpo_rloo_selfimit_mainhardgate_v3_bufferpath_step300_s16_lr1e4_b2acc4_8gpu_20260614T161426Z`.
- LR: `1e-4`, scheduler epochs `20`, min LR `1e-5`.
- `sample_time=16`, batch size `2`, accumulation `4`, 8 local GPUs.
- Checkpoint every `300` train steps and every epoch.
- Main GRPO hard gates:
  - NC/DAC existing hard safe mask.
  - TTC threshold `0.95`.
  - DDC threshold `0.99`.
- Self-imitation:
  - weight `0.01`, linear warmup over 2 epochs.
  - top-k `1`, min reward `0.88`, min reward margin `0.01`.
  - target scene cap `0.5`.
  - baseline mode `group_leave_one_out`.
  - low-noise diffusion timestep regression.
  - target gates NC `1.0`, DAC `1.0`, TTC `0.95`, DDC `0.99`.
- Offline buffer path passed only to satisfy `offline_cfg.enabled` safety contract:
  - `/mnt/project/VLA-AD/cache/recogdrive_stage3_awac_elite_buffer_train_v2_stage3_awac_iql_dualhost_20260612T182947Z`
  - buffer guidance disabled.
  - buffer distillation disabled.
  - buffer reward bonus disabled.
- Eval watchers:
  - zt2: `unique_lock_watch_on_vla_zt2_4gpu`, pid `2446350`, GPU list `4,5,6,7`, strict wait `GPU_MAX_MEM_USED_MB=2000`, `GPU_MAX_UTIL=5`.
  - zt3: `secondary_watch_on_rl_zt3_memfit_4gpu`, pid `1015107`, GPU list `0,1,2,4`, strict wait `GPU_MAX_MEM_USED_MB=2000`, `GPU_MAX_UTIL=5`.

Startup verification:
- Training entered Lightning train loop at `2026-06-14 16:17:48`.
- Local GPU utilization after entering training: roughly `56-95%`.
- Dataset sizes: train `85109`, validation `18179`.
- First checkpoint and navtest PDMS are pending.
- First available training scalar at step `49`:
  - `reward_step=0.645440`
  - `base_reward_step=0.790706`
  - `hard_safe_ratio_step=0.816406`
  - `mean_ttc_step=0.902344`
  - self-imitation target ratio `0.4375`
  - self-imitation target reward mean `0.867652`
- Diagnostic gap found: main GRPO logging had `mean_ttc` but not main `mean_ddc`/NC/DAC/TLC pass-ratio fields. A follow-up logging-only patch adds those fields for future runs; it does not affect this already-running v3 process.

Expected early gate:
- Stop if the first evaluated early checkpoint is below `0.875` PDMS.
- Watch for one more early checkpoint if within `[0.875, 0.88)`.
- Continue only if the run clears `0.88` early PDMS, then compare against cap05 `0.887107`, Safe DiffGRPO `0.906184`, and original Stage3 `0.9055`.

Artifact cleanup:
- Deleted no-checkpoint failure directories after recording the causes:
  - `stage3_grpo_rloo_selfimit_safetygate_step300_s16_lr1e4_b2acc4_8gpu_20260614T145712Z`
  - `stage3_grpo_rloo_selfimit_mainhardgate_step300_s16_lr1e4_b2acc4_8gpu_20260614T160328Z`
  - `stage3_grpo_rloo_selfimit_mainhardgate_v2_step300_s16_lr1e4_b2acc4_8gpu_20260614T161141Z`

### 2026-06-14 Reference audit: diffusion RL and preference training

Purpose:
- Do not keep iterating only by local hyperparameter tweaks.
- Before another algorithmic change, compare our implementation against mature diffusion-policy and diffusion-preference implementations.

References inspected:
- DPPO: Diffusion Policy Policy Optimization.
  - Paper/project/code: `https://diffusion-ppo.github.io/`, `https://github.com/irom-princeton/dppo`.
  - Local audit clone: `/tmp/dppo_ref`.
  - Key files inspected:
    - `/tmp/dppo_ref/model/diffusion/diffusion_ppo.py`
    - `/tmp/dppo_ref/agent/finetune/train_ppo_diffusion_agent.py`
    - `/tmp/dppo_ref/README.md`
- Diffusion-DPO.
  - Paper/code: `https://arxiv.org/abs/2311.12908`, `https://github.com/SalesforceAIResearch/DiffusionDPO`.
  - Local audit clone: `/tmp/DiffusionDPO_ref`.
  - Key files inspected:
    - `/tmp/DiffusionDPO_ref/train.py`
    - `/tmp/DiffusionDPO_ref/README.md`
- SimpleVLA-RL repository existence confirmed:
  - `https://github.com/PRIME-RL/SimpleVLA-RL`
  - A shallow clone was attempted but cancelled because the transfer was slow. Do a focused follow-up audit before implementing a VLA-specific PPO/DPO variant.

DPPO implementation takeaways:
- It treats each denoising transition as a PPO sample:
  - stores `chains_prev`, `chains_next`, old logprobs, values, returns, and advantages.
  - samples minibatches over `(environment step, denoising step)`.
- It uses true PPO clipping against old logprobs:
  - `ratio = exp(newlogprob - oldlogprob)`.
  - `max(-A * ratio, -A * clipped_ratio)`.
- It schedules clipping by denoising step:
  - small base clip at early denoising steps, larger clip later.
- It has advantage normalization and quantile clipping.
- It supports KL diagnostics and early stop when approximate KL exceeds `target_kl`.
- It uses reward scaling / GAE / critic in online environments.
- It keeps a BC regularizer to the base policy as a trust region.

Diffusion-DPO implementation takeaways:
- Preference learning is not plain regression to the preferred sample.
- The chosen/rejected pair shares the same diffusion timestep and noise.
- Loss is based on the difference between current-model diffusion MSE gap and reference-model diffusion MSE gap:
  - `model_diff = loss_w - loss_l`
  - `ref_diff = ref_loss_w - ref_loss_l`
  - `loss = -logsigmoid(-0.5 * beta * (model_diff - ref_diff))`
- This is important for our buffer setting: if we use high-PDMS buffer trajectories for preference learning, the preferred and rejected trajectories should be compared under matched diffusion noise/timestep and against the IL/reference policy, not just distilled with MSE.

Gap versus current ReCogDrive implementation:
- Current v3 run is a trajectory-level GRPO/GSPO variant with online PDM rewards and hard safety masks.
- We have GSPO ratio, reference KL, BC anneal, safety-shaped reward, and self-imitation.
- We do not yet have a full DPPO-style transition replay path where every denoising transition from sampled chains becomes a PPO sample with old logprob, denoising-step clip schedule, KL early stop, and multiple minibatch epochs.
- The earlier DPPO-style smoke in this repo was intentionally simplified and did not clear the early Stage3 gate; it should not be treated as a decisive test of mature DPPO.
- The earlier AWAC/IQL path was also not a decisive test of preference-style learning, because it mainly used weighted diffusion regression. Diffusion-DPO suggests a more appropriate pairwise diffusion objective for buffer knowledge absorption.

Actionable next algorithm directions after v3 early gate:
- If v3 clears early PDMS and improves safety submetrics:
  - Continue to step600/900 and compare against cap05.
  - Consider adding a mature DPPO transition replay branch only after confirming the hard safety gate helps.
- If v3 fails early PDMS because reward drops too much:
  - Try a less restrictive main mask or soft penalty schedule for TTC/DDC, but keep self-imitation target gates strict.
  - Do not remove DDC/TTC diagnostics.
- If buffer knowledge still needs absorption:
  - Implement a diffusion-DPO style pairwise loss over high-PDMS buffer candidate vs GT/IL/rejected candidate.
  - Pair samples must share the same diffusion timestep and noise.
  - Use IL/reference model losses as the DPO reference term.
  - Keep BC/reference KL small but present.
- If adopting DPPO:
  - Implement old-logprob storage for denoising transitions.
  - Minibatch over `(scene, sample, denoising step)`, not only final trajectory.
  - Add denoising-step clip schedule, target KL, advantage normalization, and diagnostics.
  - Validate on a small run with matched update steps before full training.

### 2026-06-14 Cleanup: Remove Obsolete Stage3 Artifacts

Reason:
- `/mnt/project/VLA-AD/outputs` had grown to roughly `252G`.
- The user explicitly allowed outdated failed results/logs to be deleted after recording the failed attempt in this summary.
- The cleanup must not affect active training/evaluation, train-only elite buffers, or the strongest retained checkpoints.

Kept:
- Current active v3 run:
  - `stage3_grpo_rloo_selfimit_mainhardgate_v3_bufferpath_step300_s16_lr1e4_b2acc4_8gpu_20260614T161426Z`
- Active zt3 original-LR control:
  - `stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z`
- Active keep-best buffer generation:
  - `stage3_awac_keepbest_buffer_zt3_wait_20260614T142658Z`
- Train-only AWAC elite buffer cache:
  - `recogdrive_stage3_awac_elite_buffer_train_v2_stage3_awac_iql_dualhost_20260612T182947Z`
- Safe DiffGRPO best checkpoint only:
  - `stage3_safe_diffgrpo_ckpt_stream_eval_live_20260610T030355Z/checkpoint_archive/epoch_12-step_17290.ckpt`
- Best cap05 checkpoint only:
  - `stage3_grpo_rloo_selfimit_cap05_step300_s16_lr1e4_b2acc4_8gpu_20260614T111511Z/train/hydra/training_recogdrive_agent/2026.06.14.11.15.51/step_checkpoints/step-step=900.ckpt`

Deleted or pruned:
- Dry-run/smoke/debug/preflight/probe directories, including AWAC preflights, GRPO launcher dry-runs, replay smoke tests, DPPO smoke tests, and evaluation launcher dry-runs.
- Failed or rejected runs whose conclusions are already recorded above:
  - trajectory-level replay one-epoch runs and their empty navtest watcher directories;
  - stopped `2e-4` buffer-guided GRPO variants;
  - no-cap self-imitation run;
  - stale queued self-imitation wait directories.
- Repeated historical checkpoints:
  - Safe DiffGRPO archive checkpoints except `epoch_12-step_17290.ckpt`;
  - cap05 duplicate watcher archive checkpoints and train checkpoints except `step-step=900.ckpt`;
  - old Safe DiffGRPO relaunch checkpoint directory;
  - old lrfix/bcanneal train checkpoint directory;
  - old lrfix/bcanneal eval checkpoint archive.

Result:
- Deleted `69` dry-run/smoke/debug/preflight/probe directories.
- Reduced `/mnt/project/VLA-AD/outputs` from roughly `252G` to roughly `170G`.
- Did not kill active local or remote tasks.

Do not repeat:
- Do not resume the deleted dry-run/preflight directories.
- Do not rerun the rejected trajectory-level replay or simplified step-level replay path as a full experiment without a mature DPPO-style implementation.
- Do not treat AWAC weighted regression failures as proof that buffer/preference learning is useless; use mature Diffusion-DPO or DPPO-style mechanics if revisiting buffer absorption.

### 2026-06-14 Cleanup: Remove Remaining Obsolete Logs And Probe Caches

Reason:
- The user approved deleting outdated results/logs after the failed attempts are recorded in this ledger.
- The cleanup target is stale artifacts that can confuse future experiment selection or waste space, not active training/evaluation state.

Deleted on 2026-06-14 UTC:
- Stale Stage3 metric-cache launcher/watcher wrappers:
  - `stage3_rl_after_metric_cache_watch_20260608T145806Z`
  - `metric_cache_navtrain_full_20260608T145628Z`
  - `metric_cache_navtrain_full_safe_diffgrpo_20260609T175818Z`
- Obsolete non-best Safe DiffGRPO / lrfix outputs:
  - `stage3_rl_2b_safe_diffgrpo_online_20260609T175818Z`
  - `stage3_safe_diffgrpo_navtest_eval_after_relaunch_20260609T1944Z`
  - `stage3_rl_2b_safe_diffgrpo_lrfix_g16_20260612T082041Z`
  - `stage3_safe_diffgrpo_lrfix_g16_ckpt_stream_eval_20260612T082041Z`
  - `stage3_rl_2b_safe_diffgrpo_lrfix_g16_bcanneal_20260612T082634Z`
  - `stage3_safe_diffgrpo_lrfix_g16_bcanneal_ckpt_stream_eval_20260612T082634Z`
  - `stage3_rl_2b_safe_diffgrpo_lrfix_g16_bcanneal_b4a2_20260612T084307Z`
  - `stage3_safe_diffgrpo_lrfix_g16_bcanneal_b4a2_ckpt_stream_eval_20260612T084307Z`
- Stale metric-cache smoke/probe artifacts:
  - `fast_metric_cache_navtest_mirror_20260613T011857Z`
  - `metric_cache_navtest_first1024`
  - `metric_cache_navtest_smoke16`
  - `metric_cache_train_smoke_stage3_rl`
- Early AWAC elite-buffer probe caches, superseded by the full train-only v2 elite buffer:
  - `stage3_awac_iql_builder_b32_probe_20260612T191838Z_elite_buffer`
  - `stage3_awac_iql_builder_b4_probe_20260612T184659Z_elite_buffer`
  - `stage3_awac_iql_builder_b4_probe_fg_20260612T184819Z_elite_buffer`
  - `stage3_awac_iql_builder_b8_probe_fg_20260612T185232Z_elite_buffer`
- Stale two-expert Stage2 intermediate output dirs whose conclusions are already in `reports/two_expert_slot/` and which have no live pid:
  - `two_expert_slot_stage2_full_dit_sft_A0init_clonefix_20260613T133206Z`
  - `two_expert_slot_stage2_full_dit_sft_A0init_indexfix_20260613T195319Z`
  - `two_expert_slot_stage2_full_dit_sft_A0init_indexfix_4gpu_eqbs128_20260613T205724Z`
  - `two_expert_slot_stage2_full_dit_sft_A0init_collatefix_4gpu_eqbs128_20260613T210233Z`

Explicitly kept:
- `a0_stage2_repro_20260531_003029`, because highcap/two-expert scripts still use its A0 checkpoint path as a default.
- `recogdrive_stage2_residual_anchor_base2b_20260610T034055Z_setsid_freezecot`, because residual-anchor scripts still reference it by default; delete only after updating or retiring those scripts.
- `stage3_grpo_rloo_selfimit_cap05_step300_s16_lr1e4_b2acc4_8gpu_20260614T111511Z`, because the zt3 keep-best train-only buffer generator still points at its `step_checkpoints` directory.
- Current active v3 run, zt3 original-LR control, active keep-best buffer generator, full train-only elite buffer, and Safe DiffGRPO best checkpoint.

Do not repeat as default routes:
- Do not relaunch old lrfix/bcanneal Safe DiffGRPO outputs. The retained baseline is the already-evaluated Safe DiffGRPO best checkpoint, not every intermediate rerun.
- Do not use the tiny AWAC probe buffers as training inputs. Use the full train-only v2 elite buffer or rebuild a new validated full buffer.
- Do not restart stale two-expert Stage2 intermediate jobs; the current tracked two-expert work is the random-HMEF val6000/navtest run.
- Do not rerun auxiliary-target-only safetygate self-imitation. If GRPO safety gating is tested, TTC/DDC must be in the main GRPO hard-safe mask and reported at matched checkpoints.

### 2026-06-14 Eval Scheduling Update: relaxed checkpoint watchers

Reason:
- The v3 main hard-gate run had already produced `step-step_300.ckpt`, but both strict remote watchers were blocked by a conservative `GPU_MAX_MEM_USED_MB=2000` threshold.
- zt2 GPUs had enough remaining 80GB-card memory for exact navtest eval while existing remote tasks continued. The user explicitly allowed using remote resources, with the constraint that existing remote tasks must not be killed.
- The zt3 original-LR control also had an `epoch_0-step_1290.ckpt` waiting for eval. Its old watcher config pointed to `/mnt/project/VLA-AD_stage3_algo_clean_4f3eb73`, so a current-repo relaxed watcher was launched to avoid stale script risk.

Actions:
- Launched v3 relaxed zt2 watcher without killing any remote process:
  - Host: `training-vla-zt2`
  - Dir: `/mnt/project/VLA-AD/outputs/stage3_grpo_rloo_selfimit_mainhardgate_v3_bufferpath_step300_s16_lr1e4_b2acc4_8gpu_20260614T161426Z/relaxed_watch_on_vla_zt2_4gpu`
  - PID: `2494061`
  - GPU list: `4,5,6,7`
  - `GPUS_PER_NODE=4`, `GPU_MAX_MEM_USED_MB=10000`, `GPU_MAX_UTIL=100`
  - Eval script remains `run_recogdrive_stage3_safe_diffgrpo_eval_8gpu_exact_pool_pdm.sh`.
  - PDM runner remains `exact_pool`; navtest full metric cache remains unchanged.
  - Started evaluating `step-step_300` at `2026-06-14T17:37:17Z`.
- Launched original-LR control relaxed zt3 watcher without killing any remote process:
  - Host: `training-rl-zt3`
  - Dir: `/mnt/project/VLA-AD/outputs/stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z/relaxed_watch_on_rl_zt3_2gpu`
  - PID: `1043991`
  - GPU list: `6,7`
  - `GPUS_PER_NODE=2`, `GPU_MAX_MEM_USED_MB=20000`, `GPU_MAX_UTIL=100`
  - Eval script uses the current repo path under `/mnt/project/VLA-AD_last_vla_dev`.
  - Started evaluating `epoch_0-step_1290` at `2026-06-14T17:42:54Z`.

Interpretation rules:
- These relaxed watchers only change scheduling thresholds. They do not change inference, PDM scorer, metric cache, or submetric calculation.
- Use their `checkpoint_eval_submetrics.tsv` rows as authoritative navtest results once completed.
- If the strict original watchers later wake up, avoid duplicate conclusions by preferring the first completed exact full-navtest row and checking checkpoint ids.

## Update Template

Append a new section for every algorithm run:

```text
### YYYY-MM-DD Attempt: <short name>

Motivation:
- Previous evidence:
- Reference sources:
- Mechanism copied:

Implementation completeness:
- Required pieces present:
- Known simplifications:
- Why simplifications are acceptable, or why this is only a smoke test:

Config:
- Run name:
- LR / schedule:
- Effective batch / sample_time / steps per epoch:
- Reward/cache split:
- Safety guards:

Expected diagnostics:
- Should increase:
- Should stay stable:
- Stop if:

Results:
- Checkpoint:
- PDMS / NC / DAC / TTC / EP / comfort / DDC / TLC:
- Training diagnostics:
- Decision:
```
