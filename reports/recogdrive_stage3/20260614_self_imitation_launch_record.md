# Stage3 GRPO Buffer-Guided Self-Imitation Launch Record

Date: 2026-06-14 UTC

## Why This Run Exists

Past experiments showed that the strongest evaluated Stage3 path is still
Safe DiffGRPO, while AWAC/IQL-style buffer training did not match its PDMS.
The practical lesson is that finding better-than-GT trajectories is necessary
but not sufficient. The diffusion planner must assign higher likelihood to
high-quality trajectories at inference time.

This run therefore keeps GRPO/GSPO as the main objective and adds small,
guarded absorption terms:

- elite-buffer neighborhood reward bonus, hard-safe samples only
- low-noise denoising distillation on valid elite-buffer targets
- self-imitation denoising loss on current GRPO samples that are hard-safe and
  have positive reward margin

The design rationale and paper references are recorded in
`reports/recogdrive_stage3/grpo_buffer_guided_self_imitation_rationale.md`.

## Code State

- `3ae531e Add GRPO self-imitation absorption for Stage3`
- `6baf5f4 Harden Stage3 GRPO launch resource guards`

The second commit fixes two operational issues:

- local Stage3 scripts now choose an available torchrun master port by default
- buffer-guided launcher waits for local target GPUs to be free before starting
  training, so it does not collide with the currently running experiment

## Queued Experiment

- run name:
  `stage3_grpo_buffer_guided_selfimit_s16_lr2e4_b2acc4_wait_20260614T003355Z`
- output root:
  `/mnt/project/VLA-AD/outputs/stage3_grpo_buffer_guided_selfimit_s16_lr2e4_b2acc4_wait_20260614T003355Z`
- queued launcher pid:
  `3395964`
- state at launch:
  waiting for local GPUs `0,1,2,3,4,5,6,7`

Key config:

- LR `2e-4`
- max epochs `20`
- scheduler min LR `1e-5`
- batch size `2`
- accumulate grad batches `4`
- GRPO sample time `16`
- reference KL coeff `0.02`
- buffer reward bonus weight `0.03`
- buffer distill loss weight `0.03`
- self-imitation loss weight `0.02`
- self-imitation min reward `0.86`
- self-imitation min reward margin `0.01`
- self-imitation timestep sampling `low_noise`

## Current Blocking Resource State

At queue time, the original run
`stage3_grpo_refkl_s16_lr2e4_b4acc2_chunk32_fast_8gpu_20260613T213145Z`
was still occupying all local GPUs and had no checkpoint yet. The new run is
therefore queued instead of launched immediately.

Remote servers `training-vla-zt2` and `training-rl-zt3` also had existing
training/evaluation tasks. No remote tasks were killed or preempted.

## Monitoring

Current run:

```bash
python scripts/training/monitor_recogdrive_stage3_grpo_run.py \
  --run-name stage3_grpo_refkl_s16_lr2e4_b4acc2_chunk32_fast_8gpu_20260613T213145Z \
  --watcher-log-lines 8 --no-gpu-snapshot
```

Queued run:

```bash
tail -f /mnt/project/VLA-AD/outputs/stage3_grpo_buffer_guided_selfimit_s16_lr2e4_b2acc4_wait_20260614T003355Z/train_gpu_wait.log
```

After the queued run starts:

```bash
python scripts/training/monitor_recogdrive_stage3_grpo_run.py \
  --run-name stage3_grpo_buffer_guided_selfimit_s16_lr2e4_b2acc4_wait_20260614T003355Z \
  --watcher-log-lines 8
```
