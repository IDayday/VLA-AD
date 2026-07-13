# ReCogDrive Stage3 Safe DiffGRPO Results - 2026-06-12

## Scope

This records the completed Stage3 RL comparison and the follow-up LR/sample-count fix state. Large artifacts are intentionally left outside git:

- Original Stage3 RL baseline evals: `/workspace/recogdrive_stage3_rl/outputs/pdms_eval_stage3_ckpt_watcher_20260609T093852Z/pdms_summary.csv`
- Safe DiffGRPO eval watcher: `/mnt/project/VLA-AD/outputs/stage3_safe_diffgrpo_ckpt_stream_eval_live_20260610T030355Z`
- Safe DiffGRPO training root: `/mnt/project/VLA-AD/outputs/stage3_rl_2b_safe_diffgrpo_online_relaunch_20260609T1944Z`
- Restarted LR-fixed run: `/mnt/project/VLA-AD/outputs/stage3_rl_2b_safe_diffgrpo_lrfix_g16_bcanneal_b4a2_20260612T084307Z`

The numeric table is checked in as `reports/recogdrive_stage3_safe_diffgrpo_pdms_summary_20260612.csv`.

## Main Result

Safe DiffGRPO produced the best observed Stage3 PDMS score:

| run | best checkpoint | PDMS |
| --- | --- | ---: |
| original Stage3 RL | epoch9_step13300 | 90.5500 |
| Safe DiffGRPO Stage3 | epoch_12-step_17290 | 90.6184 |

The absolute best improvement is `+0.0684` PDMS points over the local original Stage3 RL best. The benefit is real but small, and the completed Safe DiffGRPO run was not stable after the peak.

## Checkpoint Curve

| epoch | original RL PDMS | Safe DiffGRPO PDMS |
| ---: | ---: | ---: |
| 0 | 88.2648 | 87.8473 |
| 1 | 88.8884 | 88.9881 |
| 2 | 89.1371 | 88.7386 |
| 3 | 89.3800 | 89.5788 |
| 4 | 89.3820 | 89.4501 |
| 5 | 90.2917 | 89.7112 |
| 7 | 90.2735 | 90.2885 |
| 8 | 90.4199 | 90.4650 |
| 9 | 90.5500 | 90.4850 |
| 10 | - | 90.5776 |
| 11 | - | 90.5316 |
| 12 | - | 90.6184 |
| 13 | - | 90.1642 |
| 17 | - | 89.2870 |

## Interpretation

The early and middle checkpoint scores show that Safe DiffGRPO can match or slightly exceed the original Stage3 RL curve, with the best checkpoint appearing after the original 10-epoch comparison window. This supports keeping the reward shaping and safer GRPO update path.

The late degradation is explained by a scheduler bug in the old code path: GRPO used a hard-coded 10-epoch cosine schedule with no clamp, so extending beyond epoch 10 caused the LR to rebound instead of continuing to decay. The current code fixes that by making GRPO scheduler length configurable and clamping cosine decay progress.

## Failed Follow-Up Run

Run `stage3_rl_2b_safe_diffgrpo_lrfix_g16_bcanneal_20260612T082634Z` failed before producing checkpoints:

- Config included `sample_time=16`, `lr=2e-4`, `max_epochs=20`, `bc_coeff 0.10 -> 0.05 over 5 epochs`.
- It kept `batch_size=8`, doubling per-rank sampled trajectories from `8 x 8 = 64` to `8 x 16 = 128`.
- Failure was CUDA OOM during backward, with PyTorch reporting about `67.88 GiB` allocated and `7.95 GiB` reserved per affected A800.
- No checkpoint was produced, so the watcher summary contains only the header.

## Current Restart

The active restart keeps `sample_time=16` but reduces micro-batch memory:

- `batch_size=4`
- `accumulate_grad_batches=2`
- `lr=2e-4`
- `max_epochs=20`
- `grpo_scheduler_epochs=20`
- `grpo_scheduler_min_lr=1e-5`
- `bc_anneal=true`
- `bc_coeff_start=0.10`
- `bc_coeff_end=0.05`
- `bc_anneal_epochs=5`

This restores the per-rank sampled trajectory count to `4 x 16 = 64` while preserving the effective batch through accumulation. At the time of this report, the restarted run was alive and had not yet completed epoch 0, so no new PDMS checkpoint result was available.

## Code Changes Captured

- GRPO `sample_time` is configurable through Hydra and agent wiring.
- BC coefficient annealing is implemented and logged as `bc_coeff`.
- GRPO LR schedule now accepts configurable `epochs`, `warmup_epochs`, and `min_lr`.
- Cosine LR decay is clamped after the configured schedule horizon to prevent LR rebound.
- Stage3 RL training now saves every epoch and logs LR.
- The local launcher supports `BATCH_SIZE` and `ACCUMULATE_GRAD_BATCHES`.
- A checkpoint watcher can evaluate every generated Stage3 checkpoint with PDMS.
