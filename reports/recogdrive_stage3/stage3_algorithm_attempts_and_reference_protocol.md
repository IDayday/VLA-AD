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
| `stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z` | keep | Original-LR `1e-4` control with similar effective batch scale. Needed to separate algorithm effect from LR effect. |
| `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_log1_diag_zt2_4gpu_20260614T0300Z` | completed diagnostic | Good limited diagnostic. Ratio/KL/clip/advantage logs are active, but this still does not validate a full PPO replay implementation. |
| `stage3_grpo_refkl_s16_lr2e4_b4acc2_chunk32_fast_8gpu_20260613T213145Z/local_eval_epoch0_exact_pool_8x1gpu_shards_20260614T031521Z` | running eval | Local 8-way independent 1GPU exact navtest shards for the stopped `2e-4` epoch0 checkpoint. This avoids the fragile 8GPU DDP broadcast path while preserving exact scalar PDMS per token. |
| `stage3_grpo_buffer_guided_gspo_s16_lr2e4_b2acc8_zt2_4gpu_20260614T012106Z` | stopped | It tested buffer absorption, but Lightning did not log the returned buffer/self-imitation diagnostics. Continuing would not prove whether the auxiliary path was active. |
| `stage3_grpo_buffer_guided_gspo_s16_lr2e4_b2acc8_diagfix_zt2_4gpu_20260614T020706Z` | stopped | It was launched before the PPO/advantage diagnostics patch. It could not report `gspo_approx_kl` or the full advantage-transform diagnostics, so it was stopped before becoming a full run. |
| `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_diag_zt2_4gpu_20260614T023758Z` | failed before training | Hydra struct rejected the new override before the agent YAML was updated. No training occurred. |
| `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_diag_zt2_4gpu_20260614T024536Z` | stop and replace | It entered training and wrote only `lr-AdamW`; with default `log_every_n_steps=50`, it is too low-information for a limited diagnostic. Replace with a shorter `log_every_n_steps=1` diagnostic and a long behavior-policy sync interval. |
| Local queued `stage3_grpo_buffer_guided_selfimit_s16_lr2e4_b2acc4_wait2_20260614T003614Z` launcher | stopped | It was a stale queued run from before this protocol and duplicates the zt2 buffer-guided experiment. |
| `supervise_recogdrive_stage3_experiment.py` auto-supervisor | stopped | It can auto-launch stale next actions without the new reference-audit protocol. Future launches should be manual after updating this file. |

## Current Baselines And Controls

| Run / Reference | Status | Key Config | Best Known PDMS | Notes |
|---|---:|---|---:|---|
| User-reported original local Stage3 RL | historical | original LR `1e-4`, 10 epochs | `0.9055` | Treat this as the original GRPO/Stage3 reference. Do not compare new `2e-4` runs to it without LR caveat. |
| `stage3_safe_diffgrpo_ckpt_stream_eval_live_20260610T030355Z` | completed | Safe DiffGRPO | `0.906184` at `epoch_12-step_17290` | Strongest confirmed Stage3 result so far. |
| `stage3_grpo_refkl_s16_lr2e4_b4acc2_chunk32_fast_8gpu_20260613T213145Z` | stopped | LR `2e-4`, sample_time `16`, BC `0.10->0.05`, ref KL `0.02`, effective batch about `64` | pending eval for `epoch_0-step_1330` | Stopped at 2026-06-14 03:10 UTC after epoch0 checkpoint. Last read train reward at step `1499` was `0.7552`, below earlier windows, so do not continue full training. |
| `stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z` | running | LR `1e-4`, sample_time `16`, BC `0.10->0.05`, ref KL `0.02`, effective batch about `66` | pending | zt3 original-LR control. |
| `stage3_grpo_buffer_guided_gspo_s16_lr2e4_b2acc8_zt2_4gpu_20260614T012106Z` | stopped before eval | LR `2e-4`, GSPO ratio, elite-buffer reward bonus/distill/self-imitation | invalid run | Stopped because key diagnostics were not logged, so the run could not validate buffer absorption. |
| `stage3_grpo_buffer_guided_gspo_s16_lr2e4_b2acc8_diagfix_zt2_4gpu_20260614T020706Z` | stopped before eval | LR `2e-4`, GSPO ratio, elite-buffer reward bonus/distill/self-imitation | invalid run | Stopped because it was launched before the full PPO/advantage diagnostics patch. |
| `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_diag_zt2_4gpu_20260614T023758Z` | failed before training | LR `1e-4`, GSPO ratio, batch advantage normalize, fixed advantage clip `3.0`, no buffer | none | Failed at Hydra config parsing because `grpo_normalize_advantage_batch` was missing from `recogdrive_agent.yaml`. |
| `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_diag_zt2_4gpu_20260614T024536Z` | stopped/replaced | LR `1e-4`, GSPO ratio, batch advantage normalize, fixed advantage clip `3.0`, no buffer, `limit_train_batches=80`, `limit_val_batches=0`, default logging | none | Wrote only `lr-AdamW`; insufficient diagnostic density. |
| `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_log1_diag_zt2_4gpu_20260614T0300Z` | completed diagnostic | LR `1e-4`, GSPO ratio, batch advantage normalize, fixed advantage clip `3.0`, no buffer, `limit_train_batches=40`, `limit_val_batches=0`, `log_every_n_steps=1`, behavior sync interval `100000` | no eval | Ratio/KL/clip diagnostics moved after optimizer updates; this validates instrumentation and short-run GSPO activity only. |

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
- Full or shard-aggregated navtest eval for local `2e-4` `epoch_0-step_1330`.
- First checkpoint/eval for zt3 `1e-4` GRPO.
- Store zt2 diagnostic scalars in this ledger and compare with any future GSPO run.

Go/no-go:
- If the local `2e-4` checkpoint is below the `1e-4` control or Safe DiffGRPO, do not revisit `2e-4` as a default LR.
- If the zt3 `1e-4` control is competitive with Safe DiffGRPO, use `1e-4` as the default for algorithm tests.
- If zt3 `1e-4` is also weak, inspect data/eval parity before modifying algorithms.

Diagnostic launch rule:
- Limited diagnostics must set `LOG_EVERY_N_STEPS=1`.
- If testing GSPO ratio behavior, use a behavior-policy sync interval larger than the diagnostic window, e.g. `GRPO_BEHAVIOR_POLICY_SYNC_INTERVAL=100000`, because the first forward syncs the behavior policy and ratio is expected to be near `1.0` before the first optimizer update.
- Use at least `2 * accumulate_grad_batches` train batches so there are logged batches after one optimizer step.

### Phase 2: Next Allowed Algorithm Run: Clean Single-Pass GSPO

Only launch after Phase 1 results are read.

Preferred next run:
- GSPO/GRPO with original LR `1e-4`, `sample_time=16`, same effective batch as the control.
- Enable batch advantage normalization and conservative fixed clipping:
  - `GRPO_NORMALIZE_ADVANTAGE_BATCH=true`
  - `GRPO_ADVANTAGE_CLIP_ABS=3.0`
- Keep BC anneal `0.10 -> 0.05`, reference KL `0.02`, scheduler `20` epochs to `1e-5`.
- Do not add buffer guidance in this run. This isolates the PPO/advantage-control effect.

Rationale:
- DDPO/DPPO/RIPT-VLA-style methods all rely on stable old-policy ratio updates and controlled advantages.
- Before adding buffer absorption again, we need a clean PPO-control run that separates LR and advantage-scale effects.

Interpretation limit:
- This run can validate whether single-pass GSPO improves over current GRPO.
- It cannot validate full DPPO/DDPO/RIPT-VLA until rollout replay exists.

### Phase 3: Full PPO Replay Before Heavy Buffer Absorption

If Phase 2 is stable but does not improve eval:
- Implement a real rollout replay buffer before another expensive full run.
- Store sampled denoising chains or sufficient denoising state, old logprobs, rewards, components, safety masks, group ids, and advantages.
- Replay fixed rollouts for multiple inner epochs/minibatches with PPO clipping.
- Add per-denoising-step KL/ratio diagnostics if chain-level data is available.
- This is the first implementation level that can reasonably be compared to DDPO/DPPO/RIPT-VLA.

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
