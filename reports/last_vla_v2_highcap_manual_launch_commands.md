# Last-VLA v2 High-cap Manual Launch Commands

Baseline: A0-official-aligned full navtest PDMS = `0.864891`.

Current status: `NOT READY`. Do not launch training until the strict readiness report contains `Status: READY`.

## Current Blockers

- High-cap strict train cache does not exist at `/mnt/project/VLA-AD/cache/last_vla_v2/highcap_no_risk/train_full_highcap_chunks`.
- Existing cache `/mnt/project/VLA-AD/cache/last_vla_v2/full_geometry/full_geometry_chunks_512_20260603_035727` is old 12-token cache: sampled `jepa_context_tokens=[12,1024]`, `jepa_target_tokens=[12,1024]`, `vggt_geometry_tokens=[12,512]`.
- Local GPUs are occupied by an existing Last-VLA process using the old cache: PID `729208`, running for more than 17 hours.
- Remote default host `training-vla-zt-peer` is reachable, but `/mnt/project/VLA-AD` is on branch `research/bit-drive-left-tail` with local modifications, so automatic checkout/pull is blocked.
- No production cache generation, training, or full eval was launched from this prelaunch run.

## Discovered Paths

```bash
PROJECT_ROOT=/mnt/project/VLA-AD_last_vla_dev
BASE_CHUNK_ROOT=/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1
OUT_ROOT=/mnt/project/VLA-AD/cache/last_vla_v2
A0_INIT_CHECKPOINT=/mnt/project/VLA-AD/outputs/a0_stage2_repro_20260531_003029/a0_official_aligned_a0complete/step_00100000.ckpt
VGGT_MODEL_PATH=/mnt/project/VLA-AD/checkpoints/teachers/VGGT-1B
VJEPA_MODEL_PATH=/mnt/project/VLA-AD/checkpoints/teachers/vjepa2-vitl-fpc64-256
REMOTE_HOST=training-vla-zt-peer
REMOTE_PROJECT_ROOT=/mnt/project/VLA-AD
REMOTE_A0_INIT_CHECKPOINT=/mnt/project/VLA-AD/outputs/a0_stage2_repro_20260531_003029/a0_official_aligned_a0complete/step_00100000.ckpt
REMOTE_VLM_PATH=/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B
REMOTE_NAVSIM_LOG_PATH=/mnt/navsim/trainval_navsim_logs/trainval
REMOTE_SENSOR_BLOBS_PATH=/mnt/navsim/trainval_sensor_blobs/trainval
```

## 1. Optional Cache Cleanup Dry-run

Do not delete the old cache while any running process still references it.

```bash
OUT_ROOT=/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_cleanup \
OLD_LAST_VLA_CACHE_ROOTS=/mnt/project/VLA-AD/cache/last_vla_v2/full_geometry/full_geometry_chunks_512_20260603_035727 \
scripts/last_vla_v2/highcap_no_risk/cleanup_old_last_vla_caches.sh
```

Actual deletion remains forbidden unless both are explicitly set:

```bash
CLEAR_OLD_CACHE=1
CONFIRM_DELETE_LAST_VLA_CACHE=delete-last-vla-old-caches
```

## 2. High-cap No-risk Train Cache Preparation

Production cache generation, after GPUs are free:

```bash
RUN_CACHE=1 \
BASE_CHUNK_ROOT=/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1 \
OUTPUT_ROOT=/mnt/project/VLA-AD/cache/last_vla_v2 \
VGGT_MODEL_PATH=/mnt/project/VLA-AD/checkpoints/teachers/VGGT-1B \
VJEPA_MODEL_PATH=/mnt/project/VLA-AD/checkpoints/teachers/vjepa2-vitl-fpc64-256 \
TRAIN_CHUNK_NAME_PATTERN='train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*' \
PRECISION=bf16 \
DEVICE=cuda \
ALLOW_FULL_CACHE_WITHOUT_MAX=1 \
CHUNK_INDEX=0 \
CHUNK_SIZE=103288 \
scripts/last_vla_v2/highcap_no_risk/prepare_highcap_no_risk_data.sh
```

Dry-run command log already written:

```text
/mnt/project/VLA-AD/cache/last_vla_v2/highcap_no_risk/commands.log
```

Expected output:

```text
/mnt/project/VLA-AD/cache/last_vla_v2/highcap_no_risk/train_full_highcap_chunks
```

## 3. Strict Readiness Gate

```bash
FULL_HIGHCAP_TRAIN_CHUNK_ROOT=/mnt/project/VLA-AD/cache/last_vla_v2/highcap_no_risk/train_full_highcap_chunks \
A0_INIT_CHECKPOINT=/mnt/project/VLA-AD/outputs/a0_stage2_repro_20260531_003029/a0_official_aligned_a0complete/step_00100000.ckpt \
OUT_ROOT=/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_prelaunch_20260604_utc \
PYTHON_BIN=/root/miniconda3/envs/navsim/bin/python \
scripts/last_vla_v2/highcap_no_risk/run_final_readiness_gate.sh
```

Required report:

```text
/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_prelaunch_20260604_utc/readiness/final_readiness_report.md
```

## 4. Remote Path And Git Check

Run only after the remote worktree is clean or after moving to a clean remote clone.

```bash
ssh training-vla-zt-peer "
  cd /mnt/project/VLA-AD &&
  git fetch origin &&
  git checkout feature/recogdrive-last-vla-v2 &&
  git pull --ff-only origin feature/recogdrive-last-vla-v2 &&
  git status --short &&
  git rev-parse HEAD &&
  test -d /mnt/project/VLA-AD/cache/last_vla_v2/highcap_no_risk/train_full_highcap_chunks &&
  test -e /mnt/project/VLA-AD/outputs/a0_stage2_repro_20260531_003029/a0_official_aligned_a0complete/step_00100000.ckpt &&
  test -e /mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B &&
  test -d /mnt/navsim/trainval_navsim_logs/trainval &&
  test -d /mnt/navsim/trainval_sensor_blobs/trainval
"
```

## 5. Local Server A And Remote Server B Training Launch

```bash
RUN_TRAIN=1 \
PROJECT_ROOT=/mnt/project/VLA-AD_last_vla_dev \
OUT_ROOT=/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_full_sft \
READINESS_REPORT=/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_prelaunch_20260604_utc/readiness/final_readiness_report.md \
FULL_HIGHCAP_TRAIN_CHUNK_ROOT=/mnt/project/VLA-AD/cache/last_vla_v2/highcap_no_risk/train_full_highcap_chunks \
A0_INIT_CHECKPOINT=/mnt/project/VLA-AD/outputs/a0_stage2_repro_20260531_003029/a0_official_aligned_a0complete/step_00100000.ckpt \
REMOTE_HOST=training-vla-zt-peer \
REMOTE_PROJECT_ROOT=/mnt/project/VLA-AD \
REMOTE_OUT_ROOT=/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_full_sft \
REMOTE_FULL_HIGHCAP_TRAIN_CHUNK_ROOT=/mnt/project/VLA-AD/cache/last_vla_v2/highcap_no_risk/train_full_highcap_chunks \
REMOTE_A0_INIT_CHECKPOINT=/mnt/project/VLA-AD/outputs/a0_stage2_repro_20260531_003029/a0_official_aligned_a0complete/step_00100000.ckpt \
REMOTE_VLM_PATH=/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B \
REMOTE_NAVSIM_LOG_PATH=/mnt/navsim/trainval_navsim_logs/trainval \
REMOTE_SENSOR_BLOBS_PATH=/mnt/navsim/trainval_sensor_blobs/trainval \
MASTER_PORT_LOCAL=29531 \
MASTER_PORT_REMOTE=29541 \
LORA_PRESET=attention_mlp \
LORA_SCOPE=llm \
LORA_R=32 \
LORA_ALPHA=64 \
LORA_DROPOUT=0.05 \
LORA_USE_RSLORA=true \
LORA_USE_DORA=false \
LORA_TARGET_MODULES='' \
LORA_HIDDEN_ANCHOR_EVERY_N_STEPS=4 \
scripts/last_vla_v2/highcap_no_risk/launch_local_remote_full_training.sh
```

Expected launch report:

```text
/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_full_sft/reports/last_vla_v2_highcap_training_launch_report.json
```

## 6. Eval Prep Only

Do not run full navtest eval here.

```bash
OUT_ROOT=/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_full_sft \
NAVTEST_HIGHCAP_CHUNK_ROOT=/path/to/frozen_highcap_navtest_chunks \
VLM_PATH=/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B \
PDM_METRIC_CACHE_DIR=/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1 \
scripts/last_vla_v2/highcap_no_risk/prepare_eval_after_training.sh
```

Line B eval must use LoRA-regenerated navtest hidden cache, not original navtest hidden cache.
