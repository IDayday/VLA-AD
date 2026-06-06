# Last-VLA v2 HighCap No-Risk Server A Navtest Results

Generated: 2026-06-06 UTC

## Summary

Server A is the frozen-VLM high-cap/no-risk line:

- Line: `serverA_frozen_highcap_no_risk`
- Training root: `/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_full_sft/train_local_retry_20260605T050055Z_vggt12/serverA_frozen_highcap_no_risk`
- Navtest eval root: `/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_navtest_eval_20260606T034658Z/eval/serverA_frozen_highcap_no_risk`
- Base checkpoint: `/mnt/project/VLA-AD/outputs/a0_stage2_repro_20260531_003029/a0_official_aligned_a0complete/step_00100000.ckpt`
- Train cache: `/mnt/project/VLA-AD/cache/last_vla_v2/highcap_no_risk/train_full_highcap_chunks`
- Code branch at report generation: `feature/recogdrive-last-vla-v2`
- Code commit at report generation: `94d5ad0c0995e079d11a5a7f75ee1d777aef2d31`

Best A result is `top_step_00050000`:

- PDMS: `0.471093`
- NC: `0.754614`
- DAC: `0.760092`
- TTC: `0.645905`
- Comfort: `0.489290`
- Trajectory L1: `2.668526`

The previous A0 official-aligned reference was `PDMS=0.864891`. Server A best is lower by `0.393798`, so this A line is not competitive with the A0 baseline. Within Server A itself, the best checkpoint is early (`50k`), and the final/latest checkpoints are weaker.

## Training Setup

Server A used two stages.

### Stage A1: CoT Alignment

- Experiment: `last_vla_cot_alignment_highcap_no_risk`
- Output: `/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_full_sft/train_local_retry_20260605T050055Z_vggt12/serverA_frozen_highcap_no_risk/cot_alignment`
- Start command timestamp: `2026-06-05T05:00:55+00:00`
- GPUs: `8`
- Precision: `16-mixed`
- Strategy: `ddp_find_unused_parameters_true`
- Max epochs: `20`
- Batch size: `16`
- Validation: enabled every epoch
- LR: `0.0001`
- Trainable params: `33,674,079`
- Trainable groups: Last-VLA CoT only; action base frozen
- Key flags: `train_expert_only=true`, `freeze_base_action_head=true`, `last_vla_stage=cot_alignment`
- Latest checkpoint: `cot_alignment/latest.ckpt` (`470,841,437` bytes), written at `2026-06-05 06:52:18 UTC`

CoT alignment loss summary from TensorBoard:

| Metric | First Logged Point | Last Logged Point |
|---|---:|---:|
| `train/loss_step` | step 49: `0.009553` | step 13249: `0.000200` |
| `train/loss_epoch` | step 663: `0.003437` | step 13279: `0.000208` |
| `val/loss_epoch` | step 663: `0.002095` | step 13279: `0.000643` |

### Stage A3: Progressive Bottleneck SFT

- Experiment: `last_vla_progressive_bottleneck_highcap_no_risk`
- Output: `/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_full_sft/train_local_retry_20260605T050055Z_vggt12/serverA_frozen_highcap_no_risk/progressive_bottleneck`
- Stage adapter input: `cot_alignment/latest.ckpt`
- GPUs: `8`
- Precision: `16-mixed`
- Strategy: `ddp_find_unused_parameters_true`
- Max epochs: `200`
- Batch size: `16`
- Validation: enabled every epoch
- LR: `0.0001`
- Trainable params: `68,003,298`
- Trainable groups: action base + Last-VLA CoT
- Key flags: `train_expert_only=false`, `freeze_base_action_head=false`, `last_vla_stage=progressive_sft_bottleneck`
- Latest checkpoint: `progressive_bottleneck/latest.ckpt` (`799,446,386` bytes), written at `2026-06-06 02:59:10 UTC`

Progressive loss summary from TensorBoard:

| Metric | First Logged Point | Last Logged Point |
|---|---:|---:|
| `train/loss_step` | step 49: `0.013943` | step 132799: `0.000440` |
| `train/loss_epoch` | step 663: `0.010552` | step 132799: `0.006237` |
| `val/loss_epoch` | step 663: `0.926760` | step 132799: `0.010957` |

Important logging caveat: this completed A run did not record the detailed `last_vla_*` or hidden-drift TensorBoard tags. The event files contain total train/validation losses, but not per-component Last-VLA curves such as `last_vla_geometry_loss`, `last_vla_dynamic_loss`, or `hidden_drift_cosine`.

## Data Coverage

Training cache report:

- Loader mode: `official-aligned-local-loader-log-split`
- Train records: `84,918`
- Validation records: `18,118`
- Train/validation overlap: `0`
- Unique train sample tokens: `84,918`
- Duplicate sample tokens: `0`
- Number of log names: `978`
- JEPA tokens: `128`
- Dynamic tokens: `128`
- VGGT geometry tokens: `192`
- VGGT summary tokens: `12`
- Risk tokens: `0`
- Strict full geometry required: `true`
- Patch geometry fallback: `false`

Navtest evaluation coverage:

- Split: `navtest`
- Samples: `12,146`
- Valid PDM samples: `12,138`
- PDM failures: `0`
- Missing metric-cache samples: `8`
- Feature source: `chunk`
- Precision: `fp32`
- Target teacher tokens disabled in eval: `true`
- Metric cache: `/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1`
- Frozen A navtest chunk cache: `/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_navtest_eval_20260606T034658Z/cache/serverA_navtest_highcap`

## Evaluated Checkpoints

The eval sweep produced metrics for 12 checkpoint aliases/directories. All rows used the same navtest split and valid-sample count.

| Rank | Checkpoint | PDMS | NC | DAC | TTC | Comfort | DDC | EP | Traj L1 | Valid / Samples | Missing Cache |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `top_step_00050000` | 0.471093 | 0.754614 | 0.760092 | 0.645905 | 0.489290 | 0.898336 | 0.533000 | 2.668526 | 12138 / 12146 | 8 |
| 2 | `top_step_00080000` | 0.467799 | 0.764459 | 0.738919 | 0.639726 | 0.604877 | 0.861715 | 0.518929 | 3.019287 | 12138 / 12146 | 8 |
| 3 | `lightning_version_0_epoch=86-step=57768` | 0.448593 | 0.740072 | 0.744027 | 0.624732 | 0.525952 | 0.848657 | 0.505544 | 2.992319 | 12138 / 12146 | 8 |
| 4 | `top_step_00120000` | 0.432398 | 0.753749 | 0.702834 | 0.629428 | 0.523233 | 0.840624 | 0.487989 | 2.755363 | 12138 / 12146 | 8 |
| 5 | `lightning_version_0_epoch=191-step=127488` | 0.425938 | 0.745757 | 0.703493 | 0.624815 | 0.520267 | 0.845526 | 0.481338 | 2.776274 | 12138 / 12146 | 8 |
| 6 | `top_step_00100000` | 0.425338 | 0.740567 | 0.712803 | 0.633548 | 0.440435 | 0.840254 | 0.484123 | 2.707942 | 12138 / 12146 | 8 |
| 7 | `lightning_version_0_last` | 0.425141 | 0.744480 | 0.705306 | 0.621766 | 0.517878 | 0.845732 | 0.481045 | 2.777327 | 12138 / 12146 | 8 |
| 8 | `top_step_00060000` | 0.422182 | 0.720506 | 0.722524 | 0.611962 | 0.484759 | 0.830120 | 0.473401 | 3.121769 | 12138 / 12146 | 8 |
| 9 | `lightning_version_0_epoch=85-step=57104` | 0.415857 | 0.728744 | 0.697726 | 0.610150 | 0.530730 | 0.833786 | 0.469423 | 3.070910 | 12138 / 12146 | 8 |
| 10 | `top_latest` | 0.413652 | 0.735871 | 0.693689 | 0.616411 | 0.513264 | 0.836505 | 0.466733 | 2.803994 | 12138 / 12146 | 8 |
| 11 | `lightning_version_0_epoch=110-step=73704` | 0.412849 | 0.710372 | 0.717499 | 0.596474 | 0.501153 | 0.820316 | 0.465654 | 3.047024 | 12138 / 12146 | 8 |
| 12 | `lightning_version_0_epoch=94-step=63080` | 0.406420 | 0.717458 | 0.706377 | 0.594414 | 0.474955 | 0.818792 | 0.463503 | 3.172726 | 12138 / 12146 | 8 |

Aggregate across the 12 evaluated aliases:

- Best PDMS: `0.471093`
- Worst PDMS: `0.406420`
- Mean PDMS: `0.430605`
- Median PDMS: `0.425239`

## Interpretation

The A line learns the training objective, but the full navtest result is weak compared with the A0 official-aligned checkpoint. The strongest A checkpoint is `top_step_00050000`, not the final checkpoint. The ranking suggests that continuing the progressive training past the early saved checkpoints did not improve navtest PDMS for this frozen-VLM high-cap/no-risk setup.

For any A-versus-B comparison in this round, use `top_step_00050000` as Server A's best checkpoint and keep `top_latest`/`lightning_version_0_last` only as final-state references. Do not report Server A by latest checkpoint alone, because that would understate the best achieved A performance and obscure the early-peak behavior.

## Artifact Pointers

Best metric file:

`/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_navtest_eval_20260606T034658Z/eval/serverA_frozen_highcap_no_risk/top_step_00050000/metrics.json`

Best source checkpoint:

`/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_full_sft/train_local_retry_20260605T050055Z_vggt12/serverA_frozen_highcap_no_risk/progressive_bottleneck/step_00050000.ckpt`

Best eval snapshot checkpoint:

`/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_navtest_eval_20260606T034658Z/checkpoint_snapshot/serverA_frozen_highcap_no_risk/top_step_00050000.ckpt`

Training command log:

`/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_full_sft/train_local_retry_20260605T050055Z_vggt12/serverA_frozen_highcap_no_risk/commands.log`

Progressive training log:

`/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_full_sft/train_local_retry_20260605T050055Z_vggt12/serverA_frozen_highcap_no_risk/logs/progressive_bottleneck.log`
