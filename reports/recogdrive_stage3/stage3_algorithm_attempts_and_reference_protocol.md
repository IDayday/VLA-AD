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
| `stage3_grpo_refkl_s16_lr2e4_b4acc2_chunk32_fast_8gpu_20260613T213145Z/local_eval_epoch0_exact_pool_8x1gpu_shards_20260614T031521Z` | completed eval | Local 8-way independent 1GPU exact navtest shards for the stopped `2e-4` epoch0 checkpoint. PDMS `0.872059`, clearly below the historical strong Stage3 baselines, so do not continue `2e-4` GRPO as a default route. |
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
| `stage3_grpo_replay_1epoch_s16_i2_lr1e4_8gpu_20260614T044444Z` | running | Controlled full-navtrain 1-epoch PPO replay run. First logged replay batch is healthy. Epoch-end checkpoint will be evaluated by `stage3_grpo_replay_1epoch_s16_i2_lr1e4_8gpu_20260614T044444Z_navtest_exact_eval`. |

## Current Baselines And Controls

| Run / Reference | Status | Key Config | Best Known PDMS | Notes |
|---|---:|---|---:|---|
| User-reported original local Stage3 RL | historical | original LR `1e-4`, 10 epochs | `0.9055` | Treat this as the original GRPO/Stage3 reference. Do not compare new `2e-4` runs to it without LR caveat. |
| `stage3_safe_diffgrpo_ckpt_stream_eval_live_20260610T030355Z` | completed | Safe DiffGRPO | `0.906184` at `epoch_12-step_17290` | Strongest confirmed Stage3 result so far. |
| `stage3_grpo_refkl_s16_lr2e4_b4acc2_chunk32_fast_8gpu_20260613T213145Z` | stopped/evaluated | LR `2e-4`, sample_time `16`, BC `0.10->0.05`, ref KL `0.02`, effective batch about `64` | `0.872059` at `epoch_0-step_1330` | Exact 8-shard navtest eval: NC `0.9750`, DAC `0.9619`, TTC `0.9371`, EP `0.8159`, comfort `1.0000`, DDC `0.9459`. This is well below `0.9055/0.906184`, so the higher LR was harmful or at least not sufficient. |
| `stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z` | running | LR `1e-4`, sample_time `16`, BC `0.10->0.05`, ref KL `0.02`, effective batch about `66` | pending | zt3 original-LR control. |
| `stage3_grpo_buffer_guided_gspo_s16_lr2e4_b2acc8_zt2_4gpu_20260614T012106Z` | stopped before eval | LR `2e-4`, GSPO ratio, elite-buffer reward bonus/distill/self-imitation | invalid run | Stopped because key diagnostics were not logged, so the run could not validate buffer absorption. |
| `stage3_grpo_buffer_guided_gspo_s16_lr2e4_b2acc8_diagfix_zt2_4gpu_20260614T020706Z` | stopped before eval | LR `2e-4`, GSPO ratio, elite-buffer reward bonus/distill/self-imitation | invalid run | Stopped because it was launched before the full PPO/advantage diagnostics patch. |
| `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_diag_zt2_4gpu_20260614T023758Z` | failed before training | LR `1e-4`, GSPO ratio, batch advantage normalize, fixed advantage clip `3.0`, no buffer | none | Failed at Hydra config parsing because `grpo_normalize_advantage_batch` was missing from `recogdrive_agent.yaml`. |
| `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_diag_zt2_4gpu_20260614T024536Z` | stopped/replaced | LR `1e-4`, GSPO ratio, batch advantage normalize, fixed advantage clip `3.0`, no buffer, `limit_train_batches=80`, `limit_val_batches=0`, default logging | none | Wrote only `lr-AdamW`; insufficient diagnostic density. |
| `stage3_grpo_gspo_advnorm_lr1e4_b2acc8_log1_diag_zt2_4gpu_20260614T0300Z` | completed diagnostic | LR `1e-4`, GSPO ratio, batch advantage normalize, fixed advantage clip `3.0`, no buffer, `limit_train_batches=40`, `limit_val_batches=0`, `log_every_n_steps=1`, behavior sync interval `100000` | no eval | Ratio/KL/clip diagnostics moved after optimizer updates; this validates instrumentation and short-run GSPO activity only. |
| `stage3_grpo_replay_diag_s4_i2_lr1e4_4gpu_20260614T042519Z` | completed smoke | LR `1e-4`, `stage3_objective=grpo_replay`, `sample_time=4`, inner PPO epochs `2`, replay minibatch `4`, 4 GPUs, 40 train batches, no buffer | no eval | End-to-end PPO replay path is active: fixed old logprob replay, manual optimization, PPO ratio diagnostics, reference KL, and BC update ran without NaNs. Too small/conservative to evaluate as a main result. |
| `stage3_grpo_replay_diag_s16_i2_lr1e4_8gpu_20260614T043345Z` | completed diagnostic | LR `1e-4`, `stage3_objective=grpo_replay`, `sample_time=16`, inner PPO epochs `2`, replay minibatch `8`, 8 GPUs, 40 train batches, no buffer | no eval | Near-full sample-time diagnostic passed. Checkpoint `epoch=0-step=360.ckpt`; keep as implementation evidence only, not as a final PDMS result. |
| `stage3_grpo_replay_1epoch_s16_i2_lr1e4_8gpu_20260614T044444Z` | running | LR `1e-4`, `stage3_objective=grpo_replay`, `sample_time=16`, inner PPO epochs `2`, replay minibatch `8`, 8 GPUs, full navtrain, no buffer | pending | First logged batch at step 19: `ppo_replay_valid_ratio=1.0`, `optimizer_steps=9`, `gspo_ratio_mean=0.9757`, `clip_frac=0.03125`, `gspo_approx_kl=3.60e-4`, `safe_ratio=0.7695`. |

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

As of 2026-06-14 04:35 UTC:

- Keep running: `stage3_grpo_refkl_s16_lr1e4_b2acc11_zt3_3gpu_20260614T013856Z`.
  - Reason: this is the needed original-LR GRPO control. It isolates algorithm effects from the failed/weak `2e-4` route.
  - Do not kill it unless it crashes, produces invalid logs, or the user explicitly cancels it.
- Completed and keep as smoke evidence: `stage3_grpo_replay_diag_s4_i2_lr1e4_4gpu_20260614T042519Z`.
  - Reason: it validates the new PPO replay implementation path, but is intentionally too small for PDMS conclusions.
- Completed and promote to controlled 1-epoch training: `stage3_grpo_replay_diag_s16_i2_lr1e4_8gpu_20260614T043345Z`.
  - Reason: the near-full sample-time diagnostic passed replay, ratio/KL, and safety logging gates.
- Running: `stage3_grpo_replay_1epoch_s16_i2_lr1e4_8gpu_20260614T044444Z`.
  - Reason: this is the first controlled full-navtrain 1-epoch PPO replay run. It keeps LR, `sample_time`, BC, reference KL, and scheduler semantics aligned with the GRPO control, and changes only the replay update path.
  - First logged replay metrics are healthy. Let it run to epoch end unless replay validity drops, KL/ratio becomes unstable, or safety submetrics collapse.
  - Evaluation watcher: `stage3_grpo_replay_1epoch_s16_i2_lr1e4_8gpu_20260614T044444Z_navtest_exact_eval`.
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

Phase 0: keep controls clean.
- Keep the zt3 original-LR GRPO control alive.
- Do not restart AWAC/IQL or `2e-4` buffer-guided runs.
- Do not use navtest reward/cache for training; navtest remains evaluation only.

Phase 1: PPO replay short diagnostic.
- Use `stage3_objective=grpo_replay`.
- Start from original LR `1e-4`; no `2e-4` until the original-LR control is understood.
- Use small but non-toy rollout settings first: `GRPO_SAMPLE_TIME=4` or `8`, `GRPO_PPO_REPLAY_INNER_EPOCHS=2`, minibatch equal to a small divisor of `B * G`.
- Effective optimizer steps must be logged and compared against standard GRPO.
- Stop if `ppo_replay_loss_active` is frequently 0, `gspo_ratio_clip_frac` saturates, `gspo_approx_kl` explodes, or NC/DAC/TTC/DDC regress in train diagnostics.

Phase 2: controlled checkpoint evaluation.
- Only after Phase 1 logs are sane, run a limited checkpoint-producing diagnostic.
- Evaluate every produced checkpoint with exact PDMS using the existing async/exact pool scripts.
- Compare against:
  - Stage2 IL
  - historical original Stage3 `0.9055`
  - Safe DiffGRPO `0.906184`
  - zt3 `1e-4` control when its checkpoints are available

Phase 3: mature GRPO-buffer absorption only after replay is stable.
- Add elite-buffer knowledge to GRPO only as an auxiliary after proving replay works.
- Required before launch:
  - target coverage and target-distance logs are active;
  - self-imitation candidate/target ratios are nonzero;
  - DDC/TTC protection is explicit;
  - same LR/effective-step ablation exists.

Phase 4: full DPPO/RIPT-VLA alignment if Phase 1 helps.
- Move from trajectory-level ratio to denoising-step-level PPO replay.
- Consider dynamic sampling for uninformative groups.
- Add value/critic only if trajectory-level replay shows a clear variance problem that advantage normalization cannot handle.

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
