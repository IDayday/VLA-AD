# ReCogDrive Stage3 Core-Pareto GRPO v2 Plan

Date: 2026-06-16 UTC

## Motivation

- Current GRPO/Safe-DiffGRPO-style tricks can improve NC/DAC/TTC/DDC, but repeated local evidence shows EP can fall enough to reduce final PDMS.
- The navtest PDMS formula weights EP and TTC equally inside the feasible set:
  `PDMS = NC * DAC * (5*EP + 5*TTC + 2*comfort) / 12`.
- DDC has zero positive coefficient in the PDMS formula, so it should be a direction/legal guard only, not a reward maximization target.
- Pure AWAC/IQL offline regression, lightweight Buffer-DPO, static buffer distillation, and hard TTC/DDC gates are not the primary path for this attempt. They either transferred weakly into sampled diffusion policy outputs or suppressed progress.
- The active buffer-bonus run shows the same failure mode: step600 improved EP/DDC/DAC and reached PDMS `0.888722`, but step900 recovered safety while EP dropped from `0.832373` to `0.808524`, lowering PDMS to `0.884053`.

References checked:

- DeepSeekMath GRPO: critic-free group-relative advantage estimation; use same-scene sample groups instead of a learned value critic.
- Diffusion-DPO: preference-style diffusion alignment requires careful diffusion likelihood/noise/timestep treatment; the current E1 therefore does not re-enable lightweight DPO.
- Diffusion-QL / diffusion offline RL references: weighted regression/offline policy improvement can be useful, but our AWAC/IQL results show target absorption into this DiT sampler is the bottleneck, so E1 remains online GRPO-primary.

## Mechanism

- NC and DAC are hard feasibility constraints.
- DDC is a guard only: candidates can be invalidated for DDC drop, but DDC is not added as a positive reward term.
- EP floor is relative to GT/IL reference progress to prevent conservative collapse.
- Core optimization uses `(5*EP + 5*TTC + 2*comfort) / 12` as the main within-feasible-set signal.
- TTC is handled by Core and a soft EP/TTC tradeoff penalty, not by a hard default gate.
- Pareto front over EP/TTC/comfort receives a small advantage bonus to avoid selecting only conservative TTC-heavy samples.
- Group-state aware advantages handle all-valid, mixed, all-invalid, and all-slow groups without NaNs.
- GRPO remains the main optimizer. No AWAC/IQL, Buffer-DPO, buffer distill, or self-imitation is enabled in E1.

## Config

- Run name: `stage3_grpo_core_pareto_s16_lr1e4_b2acc4_20e_local8_<timestamp>`
- Stage3 objective: `grpo`
- LR: `1e-4`
- Scheduler: 20 epochs, min LR `1e-5`, no zero final LR.
- Epochs: `20`
- GPUs: local 8 GPU when available without OOM.
- Batch: per-GPU `2`, grad accumulation `4`, effective batch `64`.
- GRPO sample time: `16`
- BC anneal: `0.10 -> 0.05` over 5 epochs.
- Reference KL: `0.02`.
- GSPO ratio: disabled for E1.
- Advantage batch normalization: enabled.
- Advantage clip abs: `5.0`.
- Core-Pareto:
  - `use_core_pareto_grpo=true`
  - reference mode: `max_gt_il`
  - Core weights: EP `5`, TTC `5`, comfort `2`, normalizer `12`
  - NC/DAC required
  - DDC guard enabled, drop tolerance `0.01`, min absolute `0.95`
  - EP floor enabled, tolerance `0.02`, slow penalty weight `0.5`
  - TTC tradeoff penalty enabled, tolerance `0.01`, weight `0.2`
  - Pareto front enabled, bonus `0.2`, dominated positive advantage cap `0.0`
  - phenotype buckets disabled for E1
  - adaptive dual disabled for E1
  - buffer-neighborhood bonus disabled for E1
- Disabled:
  - `offline_rl_enabled=false`
  - `grpo_buffer_guidance_enabled=false`
  - `grpo_buffer_distill_loss_weight=0.0`
  - `grpo_buffer_preference_dpo_loss_weight=0.0`
  - `grpo_self_imitation_loss_weight=0.0`

## Expected Diagnostics

- `core_pareto_enabled = 1`
- `pdms_core` and `core_pareto_score_mean` finite.
- `positive_advantage_slow_fail_ratio` near `0`.
- `ep_floor_pass_ratio` should not collapse to `0`.
- `pareto_front_ratio` should be finite and nonzero.
- `core_pareto_valid_ratio` should be finite; NC/DAC should not collapse.
- DDC should stay guarded against major drop, but should not be optimized upward as reward.
- Compared with matched clean GRPO and buffer-bonus checkpoints, EP should degrade less while TTC/NC/DAC stay competitive.

## Failure Criteria

- NaN/Inf loss, reward, advantage, or policy logprob.
- reward_fn or submetric extraction failure.
- Catastrophic safety collapse: NC/DAC collapse or all-invalid groups dominate for sustained checkpoints.
- EP floor collapse: `ep_floor_pass_ratio` near zero for sustained checkpoints.
- Do not stop solely because step300/600 is below another run unless there is catastrophic safety/loss failure. First real verdict is step3000-5000 or epoch3-4.

## Result Template

| Checkpoint | PDMS | NC | DAC | TTC | EP | Comfort | DDC | TLC | Core | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| step300 | | | | | | | | | | launch health only |
| step600 | | | | | | | | | | diagnostic |
| step900 | | | | | | | | | | diagnostic |
| step3000-5000 / epoch3-4 | | | | | | | | | | first verdict |
| epoch9+ | | | | | | | | | | long-horizon comparison |

## Launch Record

Date: 2026-06-16 UTC

Resource decision:
- Stopped the local `stage3_grpo_buffer_bonusonly_s16_lr1e4_b2acc4_20e_local8_20260615T215506Z` training run to free all 8 local GPUs.
- Rationale: the run peaked at step600 PDMS `0.888722`, then step900 fell to `0.884053` mainly from EP dropping `0.832373 -> 0.808524`; this is the failure mode Core-Pareto E1 is designed to address.
- Existing checkpoints/logs were left in place.

Active run:
- Run name: `stage3_grpo_core_pareto_s16_lr1e4_b2acc4_20e_local8_20260616T020136Z`
- Run root: `/mnt/project/VLA-AD/outputs/stage3_grpo_core_pareto_s16_lr1e4_b2acc4_20e_local8_20260616T020136Z`
- Training status at launch check: running, 8 local GPUs active, no OOM/Traceback observed.
- zt2 checkpoint watcher: started at `/mnt/project/VLA-AD/outputs/stage3_grpo_core_pareto_s16_lr1e4_b2acc4_20e_local8_20260616T020136Z/unique_lock_watch_on_vla_zt2_4gpu`.
- Watcher policy: evaluate checkpoints from step300, every 300 steps, while waiting for remote GPUs to be available; do not kill remote tasks.

Confirmed overrides:
- `agent.stage3_objective=grpo`
- `agent.grpo_use_core_pareto=true`
- `agent.grpo_core_pareto_use_ep_floor=true`
- `agent.grpo_core_pareto_use_ddc_guard=true`
- `agent.grpo_core_pareto_use_ttc_tradeoff_penalty=true`
- `agent.grpo_core_pareto_use_pareto_front=true`
- `agent.grpo_hard_gate_ttc=false`
- `agent.grpo_hard_gate_ddc=false`
- `agent.offline_rl_enabled=false`
- Buffer distill, Buffer-DPO, and self-imitation weights are `0.0`.

Next readout:
- step300 is launch health only.
- Continue unless there is OOM, NaN/loss collapse, reward/submetric failure, or catastrophic safety collapse.
- First algorithm verdict remains step3000-5000 / epoch3-4.
