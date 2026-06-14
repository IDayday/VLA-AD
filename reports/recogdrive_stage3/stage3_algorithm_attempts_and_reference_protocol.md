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
| `stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z` | keep | Original-LR `1e-4` control with similar effective batch scale. Latest logged window is healthy; no checkpoint yet. |
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
| `stage3_grpo_rloo_selfimit_s16_lr1e4_b2acc4_*` | planned | Return to original-LR on-policy GRPO/GSPO and add only a small low-noise self-imitation term for safe on-policy samples above a leave-one-out same-token baseline. This targets policy absorption without replacing the proven GRPO objective. |

## Current Baselines And Controls

| Run / Reference | Status | Key Config | Best Known PDMS | Notes |
|---|---:|---|---:|---|
| User-reported original local Stage3 RL | historical | original LR `1e-4`, 10 epochs | `0.9055` | Treat this as a same-length/final-training reference, not a 1-epoch diagnostic reference. |
| User-reported original Stage3 early training | historical | original Stage3, `epoch0-1` | `0.88+` | Treat this as the short-run sanity gate. A new Stage3 method that is already clearly below this range at matched early epoch/step should not be promoted without a method-level fix. |
| `stage3_safe_diffgrpo_ckpt_stream_eval_live_20260610T030355Z` | completed | Safe DiffGRPO | `0.906184` at `epoch_12-step_17290` | Strongest confirmed Stage3 result so far. Compare against it only at comparable training length, or label the comparison as short-run diagnostic only. |
| `stage3_grpo_refkl_s16_lr2e4_b4acc2_chunk32_fast_8gpu_20260613T213145Z` | stopped/evaluated | LR `2e-4`, sample_time `16`, BC `0.10->0.05`, ref KL `0.02`, effective batch about `64` | `0.872059` at `epoch_0-step_1330` | Exact 8-shard navtest eval: NC `0.9750`, DAC `0.9619`, TTC `0.9371`, EP `0.8159`, comfort `1.0000`, DDC `0.9459`. This is well below `0.9055/0.906184`, so the higher LR was harmful or at least not sufficient. |
| `stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z` | running | LR `1e-4`, sample_time `16`, BC `0.10->0.05`, ref KL `0.02`, effective batch about `66` | pending | zt3 original-LR control. |
| `stage3_grpo_buffer_guided_gspo_s16_lr2e4_b2acc8_zt2_4gpu_20260614T012106Z` | stopped before eval | LR `2e-4`, GSPO ratio, elite-buffer reward bonus/distill/self-imitation | invalid run | Stopped because key diagnostics were not logged, so the run could not validate buffer absorption. |
| `stage3_grpo_buffer_guided_gspo_s16_lr2e4_b2acc8_diagfix_zt2_4gpu_20260614T020706Z` | stopped before eval | LR `2e-4`, GSPO ratio, elite-buffer reward bonus/distill/self-imitation | invalid run | Stopped because it was launched before the full PPO/advantage diagnostics patch. |
| `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_diag_zt2_4gpu_20260614T023758Z` | failed before training | LR `1e-4`, GSPO ratio, batch advantage normalize, fixed advantage clip `3.0`, no buffer | none | Failed at Hydra config parsing because `grpo_normalize_advantage_batch` was missing from `recogdrive_agent.yaml`. |
| `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_diag_zt2_4gpu_20260614T024536Z` | stopped/replaced | LR `1e-4`, GSPO ratio, batch advantage normalize, fixed advantage clip `3.0`, no buffer, `limit_train_batches=80`, `limit_val_batches=0`, default logging | none | Wrote only `lr-AdamW`; insufficient diagnostic density. |
| `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_log1_diag_zt2_4gpu_20260614T0300Z` | completed diagnostic | LR `1e-4`, GSPO ratio, batch advantage normalize, fixed advantage clip `3.0`, no buffer, `limit_train_batches=40`, `limit_val_batches=0`, `log_every_n_steps=1`, behavior sync interval `100000` | no eval | Ratio/KL/clip diagnostics moved after optimizer updates; this validates instrumentation and short-run GSPO activity only. |
| `stage3_grpo_replay_diag_s4_i2_lr1e4_4gpu_20260614T042519Z` | completed smoke | LR `1e-4`, `stage3_objective=grpo_replay`, `sample_time=4`, inner PPO epochs `2`, replay minibatch `4`, 4 GPUs, 40 train batches, no buffer | no eval | End-to-end PPO replay path is active: fixed old logprob replay, manual optimization, PPO ratio diagnostics, reference KL, and BC update ran without NaNs. Too small/conservative to evaluate as a main result. |
| `stage3_grpo_replay_diag_s16_i2_lr1e4_8gpu_20260614T043345Z` | completed diagnostic | LR `1e-4`, `stage3_objective=grpo_replay`, `sample_time=16`, inner PPO epochs `2`, replay minibatch `8`, 8 GPUs, 40 train batches, no buffer | no eval | Near-full sample-time diagnostic passed. Checkpoint `epoch=0-step=360.ckpt`; keep as implementation evidence only, not as a final PDMS result. |
| `stage3_grpo_replay_1epoch_s16_i2_lr1e4_8gpu_20260614T044444Z` | running | LR `1e-4`, `stage3_objective=grpo_replay`, `sample_time=16`, inner PPO epochs `2`, replay minibatch `8`, 8 GPUs, full navtrain, no buffer | pending | Latest logged step `339`: `valid_ratio=1.0`, ratio mean `0.9694`, clip fraction `0.2031`, KL `7.87e-4`; latest rollout safety was weak (`safe_ratio=0.6133`), so this remains a watched diagnostic. |
| `stage3_grpo_replay_1epoch_s16_i1_lr1e4_zt2_8gpu_20260614T050122Z` | running | LR `1e-4`, `stage3_objective=grpo_replay`, `sample_time=16`, inner PPO epochs `1`, replay minibatch `8`, 8 zt2 GPUs, full navtrain, no buffer | pending | Motivation: update-strength ablation from i2. Latest logged step `179`: `valid_ratio=1.0`, ratio mean `0.9992`, clip fraction `0`, KL `2.30e-6`, safe ratio `0.8281`. |

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
- Default behavior is unchanged: no step checkpoints unless `CHECKPOINT_EVERY_N_TRAIN_STEPS > 0`.

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
