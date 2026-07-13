# Server A Low-PDMS Diagnosis

Generated: 2026-06-06 UTC

## Scope

This note diagnoses why `serverA_frozen_highcap_no_risk` produced low full-navtest PDMS in the Last-VLA v2 high-cap/no-risk round.

Primary result source:

`/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_navtest_eval_20260606T034658Z/eval/serverA_frozen_highcap_no_risk`

Training source:

`/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_full_sft/train_local_retry_20260605T050055Z_vggt12/serverA_frozen_highcap_no_risk`

## Bottom Line

The low Server A PDMS is not an evaluation-launch failure. The eval ran on the expected navtest split, had `0` PDM runtime failures, and only had the known `8` missing metric-cache samples out of `12,146`.

The observed failure is model behavior: Server A's predicted trajectories are far from the A0 action prior and degrade multiple PDM components at once. The most likely root cause is that the frozen-VLM high-cap Last-VLA branch changes the action model without a strong policy-KD/A0 anchor, while the VLM hidden states themselves are not adapted to planning. The progressive stage then continues training past the early best checkpoint and weakens navtest PDMS.

## Hard Evidence

### A0 vs Server A

| Run | Checkpoint | PDMS | Traj L1 | NC | DAC | TTC | Comfort | EP | DDC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 official-aligned | `step_00100000` | 0.864891 | 0.260491 | 0.981340 | 0.947273 | 0.942742 | 0.999506 | 0.808921 | 0.978333 |
| Server A best | `top_step_00050000` | 0.471093 | 2.668526 | 0.754614 | 0.760092 | 0.645905 | 0.489290 | 0.533000 | 0.898336 |
| Server A latest | `top_latest` | 0.413652 | 2.803994 | 0.735871 | 0.693689 | 0.616411 | 0.513264 | 0.466733 | 0.836505 |

Relative to A0 best, Server A best has:

- `PDMS` down by `0.393798`.
- `trajectory_l1` roughly `10.24x` larger.
- `comfort` down by `0.510216`.
- `ego_progress` down by `0.275921`.
- `TTC` down by `0.296836`.

This is a broad trajectory-quality regression, not a single metric artifact.

### Best Checkpoint Still Has Many Zero-Score Scenes

For `top_step_00050000`:

- Valid PDM rows: `12,138`
- `score` zero fraction: `44.54%`
- `ego_progress` zero fraction: `44.54%`
- `comfort` zero fraction: `51.07%`
- `TTC` zero fraction: `35.41%`
- `NC` zero fraction: `23.55%`
- `DAC` zero fraction: `23.99%`
- At least one zero PDM component: `73.92%`
- At least two zero PDM components: `50.02%`

This pattern is consistent with poor trajectory/action prior quality. It is not consistent with a mostly correct planner losing only one auxiliary score.

### Server A Peaks Early

Server A checkpoint ranking:

| Rank | Checkpoint | PDMS |
|---:|---|---:|
| 1 | `top_step_00050000` | 0.471093 |
| 2 | `top_step_00080000` | 0.467799 |
| 3 | `lightning_version_0_epoch=86-step=57768` | 0.448593 |
| 10 | `top_latest` | 0.413652 |
| 12 | `lightning_version_0_epoch=94-step=63080` | 0.406420 |

The best checkpoint is the first saved top checkpoint at `50k`. Later checkpoints do not improve navtest PDMS, even though training/validation losses continue moving down.

### Training Loss Was Not Enough To Predict PDMS

CoT alignment:

- `train/loss_epoch`: `0.003437` -> `0.000208`
- `val/loss_epoch`: `0.002095` -> `0.000643`

Progressive stage:

- `train/loss_epoch`: `0.010552` -> `0.006237`
- `val/loss_epoch`: `0.926760` -> `0.010957`

These curves prove the optimizer learned the configured loss, but they did not predict full-navtest PDMS. The validation loss is computed on the same high-cap cache objective, not on PDM behavior, so the falling validation loss does not guarantee better closed-loop PDM.

## Configuration-Level Findings

### 1. Server A Does Not Adapt The VLM Hidden Space

Server A is the frozen-VLM line. It uses the high-cap Last-VLA bottleneck but does not train VLM LoRA. The VLM hidden representation is therefore assumed to already carry the right planning signal.

The B line was created to test this exact risk by training VLM LoRA and monitoring:

- `hidden_drift_cosine`
- `hidden_anchor_loss`
- `last_vla_geometry_loss`
- `last_vla_dynamic_loss`
- `last_vla_coarse_traj_l1`
- `lora_trainable_param_count`
- `lora_matched_module_count`

Server A cannot answer whether the VLM hidden state was actually aligned to planning; it only trains downstream adapters/action layers against frozen hidden features.

### 2. Progressive Stage Has No A0 Policy Anchor

Server A starts from the strong A0 checkpoint, but the progressive config has:

- `policy_kd_loss_weight: 0.0`
- `policy_kd_mode: none`
- no active A0 reference policy loss
- `train_expert_only: false`
- `freeze_base_action_head: false`

This allows the progressive stage to move the action base and Last-VLA CoT stack away from the strong A0 action prior. The navtest ranking supports that interpretation: early checkpoints are best, and longer training degrades PDMS.

### 3. The Auxiliary Target Is Not The Final Metric

The CoT alignment stage optimizes auxiliary Last-VLA losses with `diffusion_loss_weight=0.0`. The progressive stage optimizes residual diffusion plus auxiliary losses, but the actual eval metric is PDM. Because no policy-KD/PDM-proxy anchor was active, optimizing the auxiliary high-cap objective can reduce training loss while producing worse planner rollouts.

### 4. Risk Head Was Disabled

The no-risk setup intentionally sets:

- `num_risk_tokens=0`
- `last_vla_use_risk_head=false`
- `last_vla_risk_loss_weight=0.0`

This does not alone explain the whole gap, but it removes an explicit collision/TTC/comfort supervisory path. Server A best has weak `TTC=0.645905`, `comfort=0.489290`, and `NC=0.754614`, so the missing risk supervision is a plausible contributor.

### 5. The Completed A Run Lacked Detailed Last-VLA Component Logs

The A run only recorded total losses. It did not record per-component `last_vla_*` or hidden-drift TensorBoard tags. That makes post-hoc diagnosis weaker than the current B run.

This is why the code change to `navsim/planning/training/agent_lightning_module.py` is important: future runs will log the component metrics needed to distinguish:

- CoT action prior not learned.
- Geometry/dynamic alignment failed.
- Hidden states drifted too far.
- LoRA parameter/module matching was wrong.

## Most Likely Root Cause

The most likely cause is not a data/eval pipeline bug. It is the modeling/training recipe:

1. The A line adds a high-cap Last-VLA bottleneck on top of a strong A0 planner.
2. It keeps VLM hidden states frozen, so the representation may not be planning-aligned.
3. It trains the downstream policy for 200 epochs with no A0 policy-KD anchor.
4. The training objective falls, but the action prior becomes worse on navtest.
5. PDM drops broadly through trajectory L1, comfort, progress, TTC, NC, and DAC.

In short: Server A appears to overwrite a strong A0 action prior with a frozen-hidden Last-VLA bottleneck that is not sufficiently aligned to downstream planning.

## Implications For B

B should not be judged only by throughput. It must show that LoRA learns useful alignment without corrupting hidden states:

- `last_vla_geometry_loss` and `last_vla_dynamic_loss` should drop in the first 1-2 epochs.
- `last_vla_coarse_traj_l1` should drop.
- `hidden_drift_cosine` should remain safely above `0.95`.
- `hidden_anchor_loss` should stay small/stable.
- `lora_trainable_param_count` should remain about `36.93M`.
- `lora_matched_module_count` should remain `196`.

The fresh B run already shows the early desired pattern through step 149:

| Metric | Step 49 | Step 149 |
|---|---:|---:|
| `last_vla_geometry_loss` | 0.003112 | 0.001492 |
| `last_vla_dynamic_loss` | 0.000454 | 0.000368 |
| `last_vla_coarse_traj_l1` | 0.012756 | 0.002979 |
| `hidden_anchor_loss` | 0.002852 | 0.003640 |
| `hidden_drift_cosine` | 0.997148 | 0.996360 |
| `lora_trainable_param_count` | 36,929,536 | 36,929,536 |
| `lora_matched_module_count` | 196 | 196 |

This does not prove B will beat A0, but it does mean B is testing a real hypothesis that Server A could not test.

## Recommended Next Experiments

1. Keep Server A best as `top_step_00050000`, not latest.
2. For any future A-style run, add an A0 policy-KD anchor during progressive training.
3. Shorten A progressive or early-stop using navtest-like proxy; the first saved checkpoint was already best.
4. Keep detailed Last-VLA component logging enabled for all new runs.
5. Compare B against A using full navtest only after B's LoRA-regenerated hidden cache is used for both training and eval.
6. Consider a risk-supervised variant after the no-risk A/B comparison finishes, because Server A's weak TTC/comfort/NC suggests safety-related supervision is missing.
