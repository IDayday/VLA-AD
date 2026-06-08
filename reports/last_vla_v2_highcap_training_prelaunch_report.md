# Last-VLA v2 High-cap Training Prelaunch Report

Date: 2026-06-04

## Code Status

- Branch: `feature/recogdrive-last-vla-v2`
- Commit after `git pull --ff-only`: `97415d098097fd3ce51d8ba0a239f4a25f6e08c0`
- Git status before report generation: clean
- Remote sync: `git fetch origin` and `git pull --ff-only origin feature/recogdrive-last-vla-v2` completed; branch was already up to date.

## Required Path Check

The required environment variables were not explicitly exported in the shell.

Missing local environment variables:

- `PROJECT_ROOT`
- `OUT_ROOT`
- `BASE_CHUNK_ROOT`
- `A0_INIT_CHECKPOINT`
- `VGGT_MODEL_PATH`
- one of `VJEPA_MODEL_PATH` or `JEPA_DENSE_CACHE_ROOT`

Missing remote variables:

- `REMOTE_HOST`
- `REMOTE_PROJECT_ROOT`
- `REMOTE_OUT_ROOT`
- `REMOTE_FULL_HIGHCAP_TRAIN_CHUNK_ROOT`
- `REMOTE_A0_INIT_CHECKPOINT`
- `REMOTE_VLM_PATH`
- `REMOTE_NAVSIM_LOG_PATH`
- `REMOTE_SENSOR_BLOBS_PATH`

Local filesystem discovery found usable candidate paths:

- `PROJECT_ROOT=/mnt/project/VLA-AD_last_vla_dev`
- `BASE_CHUNK_ROOT=/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1`
- `A0_INIT_CHECKPOINT=/mnt/project/VLA-AD/outputs/a0_stage2_repro_20260531_003029/a0_official_aligned_a0complete/step_00100000.ckpt`
- `VGGT_MODEL_PATH=/mnt/project/VLA-AD/checkpoints/teachers/VGGT-1B`
- `VJEPA_MODEL_PATH=/mnt/project/VLA-AD/checkpoints/teachers/vjepa2-vitl-fpc64-256`

Existing cache inspection found `/mnt/project/VLA-AD/cache/last_vla_v2/full_geometry/full_geometry_chunks_512_20260603_035727`, but it is not strict high-cap:

- `jepa_context_tokens=[12,1024]`
- `jepa_target_tokens=[12,1024]`
- `vggt_geometry_tokens=[12,512]`

The required high-cap cache is missing:

- `/mnt/project/VLA-AD/cache/last_vla_v2/highcap_no_risk/train_full_highcap_chunks`

Operational blockers:

- Local GPUs are occupied by an existing Last-VLA process using the old cache.
- Remote default host `training-vla-zt-peer` is reachable, but `/mnt/project/VLA-AD` is on branch `research/bit-drive-left-tail` with local modifications.

## Decision

Status: `NOT READY`

Cache generation dry-run commands were written, but production cache generation was not launched because GPUs are occupied by an existing old-cache training job. Strict readiness gate was run and reported `NOT READY`. Local training, remote training, and full eval were not launched.

Baseline remains A0-official-aligned full navtest PDMS = `0.864891`.
