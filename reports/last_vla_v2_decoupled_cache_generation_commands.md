# Last-VLA v2 Decoupled Cache And Launch Commands

Baseline for reporting: A0-official-aligned `step_00100000`, full navtest PDMS `0.864891`.

Formal path: `ReCogDrive-LaST-v2 Decoupled HighCap NoRisk`.

Deprecated hard-bottleneck / summary replacement configs, Hydra experiments, and launch wrappers are archived. Formal configs live only under `configs/last_vla_v2/decoupled_highcap_no_risk/`.

## 1. Single-Machine Full Cache Generation

```bash
RUN_CACHE=1 \
BASE_CHUNK_ROOT=/path/to/base_train_chunks \
OUTPUT_ROOT=/path/to/last_vla_v2_outputs \
VGGT_MODEL_PATH=/path/to/VGGT-1B \
VJEPA_MODEL_PATH=/path/to/vjepa2 \
NAVSIM_DATA_ROOT=/mnt/navsim \
TRAIN_SPLIT=navtrain \
ALLOW_FULL_CACHE_WITHOUT_MAX=1 \
scripts/last_vla_v2/decoupled_highcap_no_risk/prepare_decoupled_highcap_no_risk_data.sh
```

Output:

```text
$OUTPUT_ROOT/decoupled_highcap_no_risk/train_full_highcap_chunks
```

## 2. Two-Machine Sharded Cache Generation

Server 0:

```bash
RUN_CACHE=1 \
BASE_CHUNK_ROOT=/path/to/base_train_chunks \
OUTPUT_ROOT=/path/to/last_vla_v2_outputs \
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
BASE_CHUNK_ROOT=/path/to/base_train_chunks \
OUTPUT_ROOT=/path/to/last_vla_v2_outputs \
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

Merge/audit after both shard directories exist:

```bash
RUN_CACHE=1 \
BASE_CHUNK_ROOT=/path/to/base_train_chunks \
OUTPUT_ROOT=/path/to/last_vla_v2_outputs \
NUM_SHARDS=2 \
EXPECTED_NUM_SHARDS=2 \
SKIP_BUILD_SHARDS=1 \
MERGE_SHARDS=1 \
ALLOW_FULL_CACHE_WITHOUT_MAX=1 \
scripts/last_vla_v2/decoupled_highcap_no_risk/prepare_decoupled_highcap_no_risk_data.sh
```

## 3. Readiness Gate

```bash
RUN_PREFLIGHT=1 \
FULL_HIGHCAP_TRAIN_CHUNK_ROOT=/path/to/last_vla_v2_outputs/decoupled_highcap_no_risk/train_full_highcap_chunks \
OUT_ROOT=/path/to/last_vla_v2_outputs \
scripts/last_vla_v2/decoupled_highcap_no_risk/run_strict_preflight_decoupled.sh
```

Readiness report:

```text
$OUT_ROOT/decoupled_highcap_no_risk_preflight/readiness.md
```

## 4. Launch A

```bash
RUN_TRAIN=1 \
FULL_HIGHCAP_TRAIN_CHUNK_ROOT=/path/to/last_vla_v2_outputs/decoupled_highcap_no_risk/train_full_highcap_chunks \
OUT_ROOT=/path/to/last_vla_v2_outputs \
MASTER_PORT=29601 \
scripts/last_vla_v2/decoupled_highcap_no_risk/serverA_frozen_vlm_decoupled_highcap_no_risk.sh
```

## 5. Launch B

```bash
RUN_TRAIN=1 \
FULL_HIGHCAP_TRAIN_CHUNK_ROOT=/path/to/last_vla_v2_outputs/decoupled_highcap_no_risk/train_full_highcap_chunks \
VLM_PATH=/path/to/recogdrive_vlm \
NAVSIM_LOG_PATH=/path/to/navsim_logs \
SENSOR_BLOBS_PATH=/path/to/sensor_blobs \
OUT_ROOT=/path/to/last_vla_v2_outputs \
MASTER_PORT=29701 \
scripts/last_vla_v2/decoupled_highcap_no_risk/serverB_vlm_lora_decoupled_highcap_no_risk.sh
```

## Optional Line B Navtest Hidden Cache

```bash
RUN_CACHE=1 \
NAVTEST_CHUNK_CACHE_ROOT=/path/to/frozen_vlm_navtest_cache \
VLM_PATH=/path/to/recogdrive_vlm \
VLM_LORA_ADAPTER_DIR=/path/to/serverB_vlm_lora_decoupled_highcap_no_risk/vlm_lora_cot_alignment/adapters/vlm_lora \
OUTPUT_CHUNK_ROOT=/path/to/lora_regenerated_navtest_cache \
scripts/last_vla_v2/decoupled_highcap_no_risk/build_lora_navtest_cache_decoupled.sh
```
