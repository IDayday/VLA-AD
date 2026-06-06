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

## Cache Contract

- full raw VLM hidden state preserved
- JEPA context/target `128 x 1024`
- VGGT geometry `192 x 512`
- `vggt_geometry_mode_code=2`
- no risk labels required
- no summary cache
- no patch fallback

## Generate Train Cache

Dry-run:

```bash
BASE_CHUNK_ROOT=/path/to/base_chunks \
OUTPUT_ROOT=/path/to/out \
VGGT_MODEL_PATH=/path/to/VGGT-1B \
VJEPA_MODEL_PATH=/path/to/vjepa2 \
scripts/last_vla_v2/decoupled_highcap_no_risk/prepare_decoupled_highcap_no_risk_data.sh
```

Full run:

```bash
RUN_CACHE=1 \
BASE_CHUNK_ROOT=/path/to/base_chunks \
OUTPUT_ROOT=/path/to/out \
VGGT_MODEL_PATH=/path/to/VGGT-1B \
VJEPA_MODEL_PATH=/path/to/vjepa2 \
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

Checkpoint sweep:

```bash
OUT_ROOT=/path/to/out \
NAVTEST_CHUNK_CACHE_ROOT=/path/to/navtest_cache \
LORA_NAVTEST_CHUNK_CACHE_ROOT=/path/to/lora_navtest_cache \
METRIC_CACHE_DIR=/path/to/navtest_metric_cache \
scripts/last_vla_v2/decoupled_highcap_no_risk/eval_checkpoint_sweep_decoupled.sh
```

CoT corruption:

```bash
OUT_ROOT=/path/to/out \
CHECKPOINT=/path/to/checkpoint.ckpt \
NAVTEST_CHUNK_CACHE_ROOT=/path/to/navtest_cache \
METRIC_CACHE_DIR=/path/to/navtest_metric_cache \
scripts/last_vla_v2/decoupled_highcap_no_risk/eval_cot_corruption_decoupled.sh
```
