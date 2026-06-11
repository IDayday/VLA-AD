# Last-VLA v2 Decoupled HighCap Runbook

Formal path: `ReCogDrive-LaST-v2 Decoupled HighCap NoRisk`.

Baseline: A0-official-aligned `step_00100000`, full navtest PDMS `0.864891`.

Do not set `RUN_CACHE=1`, `RUN_TRAIN=1`, or `RUN_EVAL=1` until paths are reviewed.

Deprecated hard-bottleneck / summary replacement configs and launch wrappers are archived under:

- `configs/last_vla_v2/archive/hard_bottleneck_legacy/`
- `navsim/planning/script/config/experiment/archive/hard_bottleneck_legacy/`
- `scripts/last_vla_v2/archive/hard_bottleneck_legacy/`

Formal configs live only under `configs/last_vla_v2/decoupled_highcap_no_risk/`.

## Configs

Agent configs:

- `configs/last_vla_v2/decoupled_highcap_no_risk/base.yaml`
- `configs/last_vla_v2/decoupled_highcap_no_risk/cot_alignment.yaml`
- `configs/last_vla_v2/decoupled_highcap_no_risk/progressive_sft.yaml`
- `configs/last_vla_v2/decoupled_highcap_no_risk/progressive_sft_eval.yaml`
- `configs/last_vla_v2/decoupled_highcap_no_risk/vlm_lora_cot_alignment.yaml`

Hydra experiments:

- `last_vla_decoupled_cot_alignment_highcap_no_risk`
- `last_vla_decoupled_progressive_highcap_no_risk`
- `last_vla_decoupled_vlm_lora_cot_alignment_highcap_no_risk`

## Organized Cache Entrypoint

Use this stable cache entrypoint for preflight, Line A, and Line B:

```bash
source /mnt/project/VLA-AD/cache/last_vla_v2/experiments/decoupled_highcap_no_risk/cache.env
```

It exposes `BASE_CHUNK_ROOT`, `JEPA_DENSE_CACHE_ROOT`, `GEOMETRY_CACHE_ROOT`,
`FULL_HIGHCAP_TRAIN_CHUNK_ROOT`, `NAVSIM_DATA_ROOT=/mnt/navsim`, and
`PYTHON_BIN=/root/miniconda3/envs/navsim/bin/python`.

The directory contains symlinks only; large cache payloads stay in their
generation directories.

Current concrete cache and adapter paths:

- Line A stage2 train cache:
  `/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/train_full_highcap_chunks`
- Line A CoT adapter for stage2:
  `/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_ab_cacheenv_setsid_20260607T022706Z/A/serverA_frozen_vlm_decoupled_highcap_no_risk/cot_alignment/latest.ckpt`
- Line B LoRA-regenerated train hidden cache:
  `/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_ab_cacheenv_setsid_20260607T022706Z/B/serverB_vlm_lora_decoupled_highcap_no_risk/lora_regenerated_train_hidden_cache`
- Line B extracted CoT adapter:
  `/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_ab_cacheenv_setsid_20260607T022706Z/B/serverB_vlm_lora_decoupled_highcap_no_risk/vlm_lora_cot_alignment/adapters/last_vla_cot_adapter.pt`
- Line B extracted VLM LoRA adapter:
  `/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_ab_cacheenv_setsid_20260607T022706Z/B/serverB_vlm_lora_decoupled_highcap_no_risk/vlm_lora_cot_alignment/adapters/vlm_lora`
- Frozen-VLM navtest cache for Line A eval:
  `/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/navtest_full_highcap_chunks`
- NAVTEST metric cache:
  `/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1`

Do not use old VLM-summary caches or old 12-token target caches for this path.
B stage2 training reads the LoRA-regenerated train hidden cache above. For B
navtest eval, prefer direct-online VLM+LoRA evaluation unless a dedicated
LoRA-regenerated navtest hidden cache has been explicitly generated.

## Cache Contract

- full raw VLM hidden state preserved
- JEPA context/target `128 x 1024`
- VGGT geometry `192 x 512`
- `vggt_geometry_mode_code=2`
- no risk labels required
- no summary cache
- no patch fallback

For the current A/B no-residual stage2 experiments, `vlm_text_trajectory` is
not required and is ignored by the chunk loader. For the separate VLM-text
anchor residual experiment, each training and navtest sample must additionally
contain either `vlm_text_trajectory_norm` or `vlm_text_trajectory`; these fields
come from the frozen 2B-base direct text `[PT,...]` trajectory output.

## VLM-Text Anchor Residual Variant

Current A/B stage2 no-residual config remains:

`+experiment=last_vla_decoupled_progressive_highcap_no_risk`

The residual-anchor variant is separate:

`+experiment=last_vla_decoupled_progressive_highcap_no_risk_vlm_text_residual`

Generate fixed 2B-base direct-text trajectory anchors:

```bash
python scripts/build_vlm_text_traj_anchor_cache.py \
  --model-path /mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B \
  --navsim-log-path /mnt/navsim/train_navsim_logs/train \
  --sensor-blobs-path /mnt/navsim/train_sensor_blobs/train \
  --scene-filter-yaml navsim/planning/script/config/common/train_test_split/scene_filter/trainval.yaml \
  --base-chunk-root /path/to/train_full_highcap_chunks \
  --output-dir /path/to/vlm_text_anchor_cache \
  --num-shards 8 --shard-index 0 \
  --precision bf16 --allow-tolerant-parse
```

Merge anchors into the stage2 chunk cache:

```bash
python scripts/merge_vlm_text_traj_anchor_cache_into_chunks.py \
  --base-chunk-root /path/to/train_full_highcap_chunks \
  --anchor-cache-root /path/to/vlm_text_anchor_cache \
  --output-chunk-root /path/to/train_full_highcap_chunks_vlm_text_anchor
```

Use the merged cache for residual-anchor stage2 training. Build and merge the
same anchor fields for navtest before evaluating a residual-anchor checkpoint.
No-residual checkpoints can continue to use the existing train/navtest caches.

## Generate Train Cache

Dry-run:

```bash
BASE_CHUNK_ROOT=/path/to/base_chunks \
OUTPUT_ROOT=/path/to/out \
VGGT_MODEL_PATH=/path/to/VGGT-1B \
VJEPA_MODEL_PATH=/path/to/vjepa2 \
NAVSIM_DATA_ROOT=/mnt/navsim \
TRAIN_SPLIT=navtrain \
scripts/last_vla_v2/decoupled_highcap_no_risk/prepare_decoupled_highcap_no_risk_data.sh
```

Full run:

```bash
RUN_CACHE=1 \
BASE_CHUNK_ROOT=/path/to/base_chunks \
OUTPUT_ROOT=/path/to/out \
VGGT_MODEL_PATH=/path/to/VGGT-1B \
VJEPA_MODEL_PATH=/path/to/vjepa2 \
NAVSIM_DATA_ROOT=/mnt/navsim \
TRAIN_SPLIT=navtrain \
ALLOW_FULL_CACHE_WITHOUT_MAX=1 \
scripts/last_vla_v2/decoupled_highcap_no_risk/prepare_decoupled_highcap_no_risk_data.sh
```

Output:

`$OUTPUT_ROOT/decoupled_highcap_no_risk/train_full_highcap_chunks`

## Two-Machine Sharded Cache

Each shard writes an isolated overlay directory:

- JEPA: `$OUTPUT_ROOT/decoupled_highcap_no_risk/train_jepa128_overlay_raw/shards/shard_XXXXX/`
- geometry: `$OUTPUT_ROOT/decoupled_highcap_no_risk/train_geometry192_overlay_raw/shards/shard_XXXXX/`

Server 0:

```bash
RUN_CACHE=1 \
BASE_CHUNK_ROOT=/path/to/base_chunks \
OUTPUT_ROOT=/path/to/out \
VGGT_MODEL_PATH=/path/to/VGGT-1B \
VJEPA_MODEL_PATH=/path/to/vjepa2 \
NAVSIM_DATA_ROOT=/mnt/navsim \
TRAIN_SPLIT=navtrain \
NUM_SHARDS=2 \
SHARD_INDEX=0 \
MERGE_SHARDS=0 \
ALLOW_FULL_CACHE_WITHOUT_MAX=1 \
scripts/last_vla_v2/decoupled_highcap_no_risk/prepare_decoupled_highcap_no_risk_data.sh
```

Server 1:

```bash
RUN_CACHE=1 \
BASE_CHUNK_ROOT=/path/to/base_chunks \
OUTPUT_ROOT=/path/to/out \
VGGT_MODEL_PATH=/path/to/VGGT-1B \
VJEPA_MODEL_PATH=/path/to/vjepa2 \
NAVSIM_DATA_ROOT=/mnt/navsim \
TRAIN_SPLIT=navtrain \
NUM_SHARDS=2 \
SHARD_INDEX=1 \
MERGE_SHARDS=0 \
ALLOW_FULL_CACHE_WITHOUT_MAX=1 \
scripts/last_vla_v2/decoupled_highcap_no_risk/prepare_decoupled_highcap_no_risk_data.sh
```

Merge and audit from one server after both shard directories exist:

```bash
RUN_CACHE=1 \
BASE_CHUNK_ROOT=/path/to/base_chunks \
OUTPUT_ROOT=/path/to/out \
NUM_SHARDS=2 \
EXPECTED_NUM_SHARDS=2 \
SKIP_BUILD_SHARDS=1 \
MERGE_SHARDS=1 \
ALLOW_FULL_CACHE_WITHOUT_MAX=1 \
scripts/last_vla_v2/decoupled_highcap_no_risk/prepare_decoupled_highcap_no_risk_data.sh
```

## Preflight

```bash
RUN_PREFLIGHT=1 \
FULL_HIGHCAP_TRAIN_CHUNK_ROOT=/path/to/train_full_highcap_chunks \
OUT_ROOT=/path/to/out \
scripts/last_vla_v2/decoupled_highcap_no_risk/run_strict_preflight_decoupled.sh
```

The preflight report is written to:

`$OUT_ROOT/decoupled_highcap_no_risk_preflight/readiness.md`

## Line A

Frozen VLM hidden cache:

```bash
RUN_TRAIN=1 \
FULL_HIGHCAP_TRAIN_CHUNK_ROOT=/path/to/train_full_highcap_chunks \
OUT_ROOT=/path/to/out \
MASTER_PORT=29601 \
scripts/last_vla_v2/decoupled_highcap_no_risk/serverA_frozen_vlm_decoupled_highcap_no_risk.sh
```

No A0 checkpoint is passed by default. `PERFORMANCE_ROUTE=1` is the only path that allows `agent.checkpoint_path`.

## Line B

VLM-LoRA alignment, adapter extraction, hidden cache regeneration, then the same progressive SFT as Line A:

```bash
RUN_TRAIN=1 \
FULL_HIGHCAP_TRAIN_CHUNK_ROOT=/path/to/train_full_highcap_chunks \
VLM_PATH=/path/to/vlm \
NAVSIM_LOG_PATH=/path/to/navsim_logs \
SENSOR_BLOBS_PATH=/path/to/sensor_blobs \
OUT_ROOT=/path/to/out \
MASTER_PORT=29701 \
scripts/last_vla_v2/decoupled_highcap_no_risk/serverB_vlm_lora_decoupled_highcap_no_risk.sh
```

Line B navtest eval requires a LoRA-regenerated navtest hidden cache:

```bash
RUN_CACHE=1 \
NAVTEST_CHUNK_CACHE_ROOT=/path/to/navtest_original_hidden_cache \
VLM_PATH=/path/to/vlm \
VLM_LORA_ADAPTER_DIR=/path/to/serverB/adapters/vlm_lora \
OUTPUT_CHUNK_ROOT=/path/to/lora_navtest_hidden_cache \
scripts/last_vla_v2/decoupled_highcap_no_risk/build_lora_navtest_cache_decoupled.sh
```

## Eval

Checkpoint sweep, including original ReCogDrive-style Lightning top-5
validation checkpoints:

```bash
OUT_ROOT=/path/to/out \
NAVTEST_CHUNK_CACHE_ROOT=/path/to/navtest_cache \
LORA_NAVTEST_CHUNK_CACHE_ROOT=/path/to/lora_navtest_cache \
METRIC_CACHE_DIR=/path/to/navtest_metric_cache \
scripts/last_vla_v2/decoupled_highcap_no_risk/eval_checkpoint_sweep_decoupled.sh
```

For the current A/B no-residual stage2 runs, keep validation enabled during
training. Do not pass `trainer.params.limit_val_batches=0`,
`trainer.params.check_val_every_n_epoch=999999`, or `SKIP_VALIDATION=1` when
top-5 eval is required. The training script already uses
`ModelCheckpoint(save_top_k=5, monitor="val/loss_epoch")`.

The reusable no-residual A/B stage2 launcher starts both training jobs and
attaches a watcher that waits for training completion before running the top-5
NAVTEST sweep:

```bash
RUN_TRAIN=1 \
PROJECT_ROOT=/mnt/project/VLA-AD_last_vla_dev \
REMOTE_HOST=training-rl-zt2 \
scripts/last_vla_v2/decoupled_highcap_no_risk/launch_no_residual_stage2_ab_top5val.sh
```

B top-5 eval can avoid building a LoRA navtest cache by using direct-online VLM
hidden computation:

```bash
OUT_ROOT=/path/to/no_residual_stage2_ab_top5val_run \
NAVTEST_CHUNK_CACHE_ROOT=/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/navtest_full_highcap_chunks \
METRIC_CACHE_DIR=/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1 \
B_ONLINE_VLM_PATH=/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B \
B_ONLINE_VLM_LORA_ADAPTER_DIR=/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_ab_cacheenv_setsid_20260607T022706Z/B/serverB_vlm_lora_decoupled_highcap_no_risk/vlm_lora_cot_alignment/adapters/vlm_lora \
INCLUDE_TOPK_CHECKPOINTS=1 \
INCLUDE_CHECKPOINTS_TO_EVAL=0 \
scripts/last_vla_v2/decoupled_highcap_no_risk/eval_checkpoint_sweep_decoupled.sh
```

Training and eval progress monitor:

```bash
python scripts/monitor_last_vla_stage2_ab_progress.py \
  --out-root /path/to/no_residual_stage2_ab_top5val_run \
  --watch 30
```

CoT corruption:

```bash
OUT_ROOT=/path/to/out \
CHECKPOINT=/path/to/checkpoint.ckpt \
NAVTEST_CHUNK_CACHE_ROOT=/path/to/navtest_cache \
METRIC_CACHE_DIR=/path/to/navtest_metric_cache \
scripts/last_vla_v2/decoupled_highcap_no_risk/eval_cot_corruption_decoupled.sh
```
