# ReCogDrive Stage3 GRPO Buffer-Guided Self-Imitation Rationale

Date: 2026-06-14

## Empirical Starting Point

- The strongest evaluated Stage3 result so far is still the Safe DiffGRPO run
  `stage3_safe_diffgrpo_ckpt_stream_eval_live_20260610T030355Z`, checkpoint
  `epoch_12-step_17290`, with navtest PDMS about 0.9062.
- AWAC/IQL and AWAC+DPO variants were useful for building and validating elite
  candidate buffers, but their evaluated PDMS lagged the GRPO path. This suggests
  that offline elite discovery alone is not enough.
- The practical gap is policy absorption: high-PDMS candidates or samples must
  become high-probability outputs of the diffusion planner, not only appear in a
  buffer or in reward logs.

## Literature Signals Checked

- Diffusion Policy shows that conditional denoising diffusion is a strong and
  stable action policy class for multimodal robot actions.
  Source: https://arxiv.org/abs/2303.04137
- DPPO reports that policy-gradient fine-tuning can work well for diffusion
  policies because diffusion sampling keeps exploration structured and on
  manifold.
  Source: https://arxiv.org/abs/2409.00588
- DDPO frames denoising as a multi-step decision process and reports that direct
  diffusion policy optimization can beat reward-weighted regression for
  downstream objectives, which supports keeping GRPO as the primary signal rather
  than replacing it with pure reward-weighted BC.
  Source: https://arxiv.org/abs/2305.13301
- Diffusion-QL couples behavior cloning with policy improvement inside the
  diffusion training objective, which supports adding a small denoising
  regression term on better actions while keeping a trust-region style anchor.
  Source: https://arxiv.org/abs/2208.06193
- Self-Imitation Learning trains on the agent's own past high-return behavior
  only when it has positive advantage. This maps well to current GRPO samples:
  only hard-safe, above-baseline samples should be imitated.
  Source: https://arxiv.org/abs/1806.05635
- Recent VLA post-training work such as RIPT-VLA and TGRPO emphasizes
  trajectory-level RL, group/leave-one-out advantages, and stable post-SFT
  updates, supporting the direction of trajectory-wise GRPO with conservative
  auxiliary imitation instead of a wholesale algorithm switch.
  Sources: https://openreview.net/forum?id=oXYZHg7HiZ and
  https://arxiv.org/html/2506.08440v3

## Design Decision

Keep GRPO/GSPO as the main Stage3 objective, then add two small absorption
channels:

1. Buffer-guided reward-neighborhood bonus:
   - Uses train-only elite buffer candidates.
   - Rewards current samples only when they are hard-safe and close to valid
     elite targets.
   - This nudges exploration toward known good trajectory neighborhoods without
     giving unsafe samples a shortcut.

2. GRPO self-imitation diffusion loss:
   - Selects current on-policy GRPO samples that are hard-safe, exceed a minimum
     PDMS threshold, and have positive margin over a baseline.
   - Uses those trajectories only as detached denoising targets with
     `allow_target_tokens=False`.
   - Uses a small, warm-started loss weight so GRPO remains the primary
     optimizer.

## Intended Diagnostics

- `grpo_self_imitation_candidate_ratio`: whether current sampling can discover
  trainable high-quality trajectories.
- `grpo_self_imitation_target_ratio`: scene-level coverage of self-imitation
  targets.
- `grpo_self_imitation_target_reward_mean/max`: quality of selected targets.
- `grpo_self_imitation_target_margin_mean`: whether targets are genuinely above
  baseline.
- `grpo_self_imitation_weight_sum` and
  `grpo_self_imitation_zero_weight_batch`: whether the auxiliary loss is active.
- `grpo_buffer_target_distance_mean`: whether generated samples are moving
  closer to elite-buffer neighborhoods.

## Rejection Criteria

- If candidate ratio stays near zero, the model is not sampling enough good
  trajectories; increase sample diversity or improve proposal generation before
  raising imitation weight.
- If candidate ratio is healthy but target distance and eval PDMS do not improve,
  the diffusion loss is not absorbing the targets; consider timestep scheduling
  or a stronger but still warm-started distillation weight.
- If safety submetrics degrade, reduce or disable reward bonus/self-imitation and
  tighten hard-safe eligibility instead of chasing raw PDMS.

## Change Admission Rule

Every later Stage3 algorithm change should be admitted only after recording:

1. Empirical lesson:
   - Which previous run/checkpoint/buffer statistic motivates the change.
   - Which failure mode it targets, such as weak policy absorption, unsafe
     candidate leakage, poor exploration, low valid-candidate coverage, or
     degraded DDC/TTC.
2. Literature basis:
   - Relevant papers or technical reports checked online before the change.
   - The specific mechanism being borrowed, not just the paper title.
3. Testable hypothesis:
   - Expected movement in PDMS and submetrics.
   - Expected movement in training diagnostics.
4. Risk guard:
   - What would count as regression, especially for NC, DAC, TTC, DDC, and
     target leakage.
5. Minimal implementation and validation:
   - Keep the first patch scoped.
   - Compile/smoke-test the code path.
   - Evaluate checkpoint PDMS and submetrics before treating the change as
     successful.

This rule is meant to prevent blind parameter search. A single navtest score is
not enough evidence for a new algorithm direction unless it is supported by the
corresponding training diagnostics and submetric behavior.
