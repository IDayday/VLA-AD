# ReCogDrive Stage3 Core-Pareto GRPO v2 Summary - 2026-06-16

Note: this file is an early-run snapshot. For the latest evaluated results up
to `step-step_24300`, use:
`reports/recogdrive_stage3/core_pareto_grpo_v2_results_summary_20260621.md`.

## Scope

This document summarizes the current Stage3 run:

`stage3_grpo_core_pareto_s16_lr1e4_b2acc4_20e_local8_20260616T020136Z`

The comparison target is the local historical original ReCogDrive Stage3 RL
reference recorded in the Stage3 ledger:

- Original long-horizon reference: `epoch9-step13300`, navtest PDMS `0.9055`.
- Original early sanity band: `epoch0-1` around `0.88+` PDMS.
- Current run should not be judged as final before comparable training length.

This version is Stage3-only. It keeps the ReCogDrive VLM backbone and DiT
architecture unchanged, and does not add trainable modules.

## Design Summary

Core-Pareto GRPO v2 keeps GRPO as the optimizer, but changes the Stage3
advantage construction to better match the PDMS formula:

```text
PDMS = NC * DAC * (5 * EP + 5 * TTC + 2 * Comfort + 0 * DDC) / 12
Core = (5 * EP + 5 * TTC + 2 * Comfort) / 12
```

The main design change is to treat `NC` and `DAC` as feasibility constraints,
use `DDC` as a guard rather than a positive reward term, and optimize the
within-feasible-set Core through GRPO group advantages. This is intended to
avoid the previous failure mode where hard safety/TTC/DDC shaping improved
some safety submetrics but suppressed progress.

## Changes Compared With Original Stage3

| Area | Original ReCogDrive Stage3 / local historical baseline | Current Core-Pareto GRPO v2 | Purpose | Current evidence / risk |
|---|---|---|---|---|
| Stage3 objective path | GRPO-style Stage3 RL baseline. No Core-Pareto objective. | `stage3_objective=grpo`, `grpo_use_core_pareto=true`. | Keep the proven GRPO training spine while changing the advantage semantics. | Early navtest best so far is `0.892644` at `epoch0-step1330`, above the `0.88+` early sanity band but below long-horizon `0.9055`. |
| Reward decomposition | Original path does not explicitly align advantage with `Core=(5EP+5TTC+2Comfort)/12`. | Adds PDMS Core computation and logs Core, Core-vs-reference deltas, EP/TTC/DDC deltas. | Make optimization target match the actual PDMS weighting. | Core improves at `step1200/epoch0` versus early steps, but `step1500` regresses, so stability is not proven. |
| NC / DAC handling | Treated through original reward/safety path. | `NC` and `DAC` are hard feasibility constraints inside Core-Pareto advantage. | Avoid giving positive advantage to high-Core trajectories that are infeasible. | `epoch0-step1330`: NC `0.983852`, DAC `0.966881`; feasible metrics are healthy but still fluctuate. |
| DDC handling | Prior safe variants could overemphasize DDC/TTC as safety reward terms. | DDC has zero positive reward coefficient; it is used as a guard only: `DDC >= max(0.95, ref_ddc - 0.01)`. | Prevent optimizing DDC at the expense of EP, while still avoiding large direction/legal degradation. | DDC drops from `0.979033` at step600 to `0.968034` at epoch0; guard prevents collapse but does not force DDC upward. |
| EP handling | No explicit Core-Pareto EP anti-conservative floor. | Adds EP floor relative to max(GT, IL) reference: tolerance `0.02`, slow penalty weight `0.5`. | Prevent conservative trajectories from receiving positive advantage only because TTC/safety is high. | EP improves from `0.830495` step300 to `0.836308` epoch0, but step1500 safety drops despite EP staying higher. |
| TTC handling | Previous hard TTC gating could suppress progress. | Hard TTC/DDC gates disabled in launcher; TTC enters Core and a soft EP/TTC tradeoff penalty. | Preserve TTC importance without making it an overly hard gate. | TTC peaks at step300 `0.956747`, epoch0 `0.955924`, then falls at step1500 `0.943566`. |
| Pareto logic | No Pareto-front treatment over EP/TTC/Comfort. | Adds Pareto-front bonus over EP, TTC, Comfort and caps positive advantage for dominated candidates. | Prefer non-dominated tradeoffs instead of a single scalar that can favor conservative modes. | Mechanism is active; early best improved to epoch0, but later regression means more checkpoints are needed. |
| Reference baseline | Original Stage3 does not use Core-Pareto max-GT/IL reference for advantage floors/guards. | Scores GT and IL/reference policy, uses `max_gt_il` reference for Core, EP, TTC, DDC. | Compare sampled trajectories against behavior/reference support without treating GT as optimal. | Reference is used only for reward/advantage, not as inference conditioning. |
| Candidate group size | Historical/default configs varied; original local final reference is the long-horizon Stage3 run. | `grpo_sample_time=16`: each scene samples 16 trajectories for group advantage. | Increase within-scene comparison resolution for GRPO. | Per micro-step: `2 scenes/GPU * 16 samples`, 8 GPUs -> 256 sampled trajectories; per optimizer step with grad-accum 4 -> 1024 sampled trajectories. |
| BC / trust region | Original Stage3 uses RL plus behavior support according to its training path. | Keeps BC anneal `0.10 -> 0.05` over 5 epochs and reference KL `0.02`. | Keep policy near the IL/reference region while allowing RL improvement. | Stability is mixed: epoch0 improves, step1500 regresses. |
| LR schedule / epoch budget | Local historical original reference: LR `1e-4`, 10 epochs, final result at `epoch9-step13300`; earlier original schedules may decay to zero. | LR `1e-4`, `20` epochs, GRPO cosine schedule to min LR `1e-5`, no warmup. | Keep original initial LR but avoid zero LR and allow longer training. | This changes training length; compare final only at matched or clearly labeled exposure. |
| Offline buffer / AWAC / DPO | Not part of original Stage3 baseline. | Disabled in this run: `offline_rl_enabled=false`, buffer guidance/distill/DPO/self-imitation weights all `0`. | Isolate the Core-Pareto GRPO change. | Clean ablation: current result is not caused by buffer or AWAC. |
| Architecture | Original ReCogDrive VLM + diffusion planner. | VLM backbone unchanged; DiT architecture unchanged; no new trainable critic/value module. | Attribute changes to Stage3 objective, not architecture. | Satisfies Stage3-only constraint. |
| Evaluation workflow | Historical local results came from checkpoint/navtest evaluation. | Continuous navtest evaluation for step checkpoints and epoch checkpoints; navtest is reporting only, not training reward. | Track early trend without using navtest for optimization. | Current best is epoch0; step1500 regression requires continued monitoring. |

## Current Run Configuration

| Field | Value |
|---|---|
| Run root | `/mnt/project/VLA-AD/outputs/stage3_grpo_core_pareto_s16_lr1e4_b2acc4_20e_local8_20260616T020136Z` |
| Branch | `feature/recogdrive-last-vla-v2` |
| Stage3 objective | `grpo` |
| Core-Pareto | enabled |
| Offline RL / AWAC | disabled |
| Buffer guidance / distill / DPO / self-imitation | disabled |
| GPUs | local 8 GPU training |
| Per-GPU batch | `2` |
| Gradient accumulation | `4` |
| Effective scene batch | `64` |
| GRPO samples per scene | `16` |
| Sampled trajectories per optimizer step | about `1024` |
| LR | `1e-4` |
| Max epochs | `20` |
| Scheduler | GRPO WarmupCosLR, warmup `0`, min LR `1e-5` |
| BC schedule | `0.10 -> 0.05` over 5 epochs |
| Reference KL | `0.02` |
| Reference mode | `max_gt_il` |
| Hard TTC gate | disabled |
| Hard DDC gate | disabled |

## Current Navtest Results

These are evaluation results only. They must not be used as training rewards or
for train-time checkpoint selection.

| Checkpoint | PDMS | Core | NC | DAC | TTC | EP | Comfort | DDC | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `step-step_300` | `0.887645` | `0.911337` | `0.985706` | `0.961031` | `0.956747` | `0.830495` | `0.999918` | `0.974172` | Healthy launch check. |
| `step-step_600` | `0.888936` | `0.910401` | `0.985088` | `0.965398` | `0.954193` | `0.830769` | `1.000000` | `0.979033` | Small PDMS gain, mostly DAC/DDC. |
| `step-step_900` | `0.885056` | `0.910578` | `0.983523` | `0.956912` | `0.946779` | `0.838608` | `1.000000` | `0.978621` | Regression in DAC/TTC despite higher EP. |
| `step-step_1200` | `0.891423` | `0.913477` | `0.981422` | `0.965810` | `0.950816` | `0.841530` | `1.000000` | `0.970671` | Best step checkpoint so far. |
| `epoch_0-step_1330` | `0.892644` | `0.913430` | `0.983852` | `0.966881` | `0.955924` | `0.836308` | `1.000000` | `0.968034` | Current best overall. |
| `step-step_1500` | `0.881128` | `0.909725` | `0.978744` | `0.952051` | `0.943566` | `0.839806` | `0.999918` | `0.976891` | Sharp regression; continue monitoring before verdict. |

## Interim Interpretation

The current Core-Pareto GRPO v2 run is a clean GRPO-primary ablation. It
successfully avoids mixing in AWAC, buffer distillation, DPO, or self-imitation,
so the observed behavior reflects the new advantage design and schedule.

Early evidence is mixed:

- Positive: best early checkpoint reaches PDMS `0.892644`, clearing the
  original Stage3 early sanity band of `0.88+`.
- Positive: EP is generally higher than the earliest checkpoints, which matches
  the goal of avoiding conservative collapse.
- Negative: step1500 drops sharply through NC/DAC/TTC, showing the current
  advantage can still create unstable safety/feasibility oscillation.
- Not yet a final success: the long-horizon references remain original Stage3
  `0.9055` at `epoch9-step13300` and Safe DiffGRPO `0.906184` at
  `epoch_12-step_17290`.

## Next Checks

- Continue automatic eval for later checkpoints, especially `step3000-5000` and
  epochs 3-4, before making an algorithm verdict.
- Watch whether the step1500 regression is transient or the start of sustained
  NC/DAC/TTC degradation.
- If regression persists, the likely next ablation should be a more conservative
  Core-Pareto safety/group-weight design, not AWAC or buffer distillation by
  default.
