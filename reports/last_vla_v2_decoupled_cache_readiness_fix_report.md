# Last-VLA v2 Decoupled Cache Readiness Fix Report

Date: 2026-06-06

Formal path: `ReCogDrive-LaST-v2 Decoupled HighCap NoRisk`.

Baseline: A0-official-aligned `step_00100000`, full navtest PDMS `0.864891`.

No training launched. No full eval launched. No production cache generated.

## Changed Files

- `scripts/build_recogdrive_jepa_overlay_from_chunks.py`
- `scripts/build_last_vla_full_geometry_cache.py`
- `scripts/merge_last_vla_overlay_shards.py`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/prepare_decoupled_highcap_no_risk_data.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/run_strict_preflight_decoupled.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/serverB_vlm_lora_decoupled_highcap_no_risk.sh`
- `tests/test_last_vla_decoupled_cache_sharding.py`
- `tests/test_last_vla_highcap_launchers_dryrun.py`
- `tests/test_last_vla_legacy_paths.py`
- `tests/test_last_vla_round2_launchers_dryrun.py`
- `docs/Last_VLA_v2_Decoupled_CoT_Design.md`
- `docs/Last_VLA_v2_Decoupled_HighCap_Runbook.md`
- `reports/last_vla_v2_decoupled_cache_generation_commands.md`
- `reports/last_vla_v2_decoupled_highcap_readiness_report.md`

## Archived Old Files

Configs:

- `configs/last_vla_v2/archive/hard_bottleneck_legacy/last_vla_cot_alignment.yaml`
- `configs/last_vla_v2/archive/hard_bottleneck_legacy/last_vla_cot_alignment_highcap_no_risk.yaml`
- `configs/last_vla_v2/archive/hard_bottleneck_legacy/last_vla_highcap_no_risk_base.yaml`
- `configs/last_vla_v2/archive/hard_bottleneck_legacy/last_vla_progressive_bottleneck.yaml`
- `configs/last_vla_v2/archive/hard_bottleneck_legacy/last_vla_progressive_bottleneck_eval.yaml`
- `configs/last_vla_v2/archive/hard_bottleneck_legacy/last_vla_progressive_bottleneck_highcap_no_risk.yaml`
- `configs/last_vla_v2/archive/hard_bottleneck_legacy/last_vla_progressive_bottleneck_highcap_no_risk_eval.yaml`
- `configs/last_vla_v2/archive/hard_bottleneck_legacy/last_vla_vlm_lora_cot_alignment.yaml`
- `configs/last_vla_v2/archive/hard_bottleneck_legacy/last_vla_vlm_lora_cot_alignment_highcap_no_risk.yaml`

Hydra experiments:

- `navsim/planning/script/config/experiment/archive/hard_bottleneck_legacy/last_vla_cot_alignment.yaml`
- `navsim/planning/script/config/experiment/archive/hard_bottleneck_legacy/last_vla_cot_alignment_highcap_no_risk.yaml`
- `navsim/planning/script/config/experiment/archive/hard_bottleneck_legacy/last_vla_progressive_bottleneck.yaml`
- `navsim/planning/script/config/experiment/archive/hard_bottleneck_legacy/last_vla_progressive_bottleneck_highcap_no_risk.yaml`
- `navsim/planning/script/config/experiment/archive/hard_bottleneck_legacy/last_vla_vlm_lora_cot_alignment_highcap_no_risk.yaml`

Scripts:

- `scripts/last_vla_v2/archive/hard_bottleneck_legacy/serverA_frozen_vlm_highcap_no_risk.sh`
- `scripts/last_vla_v2/archive/hard_bottleneck_legacy/serverB_vlm_lora_highcap_no_risk.sh`
- `scripts/last_vla_v2/archive/hard_bottleneck_legacy/eval_checkpoint_sweep_highcap.sh`
- `scripts/last_vla_v2/archive/hard_bottleneck_legacy/eval_cot_corruption_highcap.sh`
- `scripts/last_vla_v2/archive/hard_bottleneck_legacy/run_strict_preflight_highcap.sh`

Formal configs live only under `configs/last_vla_v2/decoupled_highcap_no_risk/`.

## Sharded Cache Fix

JEPA and geometry overlay builders now support safe parallel sharding:

- `--output-shard-subdir`
- `--write-shard-manifest`

With `--num-shards > 1`, builders write only to:

```text
output_root/shards/shard_XXXXX/
```

Each shard contains `samples/`, `index.jsonl`, `metadata.json`, and optional `shard_manifest.json`. Shard metadata records `shard_index`, `num_shards`, `num_written`, `num_skipped_by_shard`, token hash summary, and `overlay_type`.

`scripts/merge_last_vla_overlay_shards.py` merges sharded overlays, checks duplicate `sample_token`, checks missing/extra shards in strict mode, hardlinks or copies samples, and writes a merged `index.jsonl` plus `metadata.json`.

## Prepare Script Fix

`prepare_decoupled_highcap_no_risk_data.sh` now supports:

- `SKIP_BUILD_SHARDS=0/1`
- `MERGE_SHARDS=0/1`
- `EXPECTED_NUM_SHARDS`
- `COPY_MODE=hardlink/copy`

For `NUM_SHARDS > 1`, shard builders write to raw overlay roots and exit without final merge/audit unless `MERGE_SHARDS=1`.

## Preflight Fix

`run_strict_preflight_decoupled.sh` writes:

```text
$OUT_ROOT/decoupled_highcap_no_risk_preflight/readiness.md
```

The report contains `Status: READY` only when manifest audit and preflight tests pass. Failures write `Status: NOT READY`, failed command, log path, blocker list, and nonzero exit code.

## LoRA Target Modules Fix

`serverB_vlm_lora_decoupled_highcap_no_risk.sh` now passes `agent.last_vla_vlm_lora_target_modules=...` only when `LORA_TARGET_MODULES` is nonempty. Empty value uses the config default and no longer sends a quoted empty Hydra override.

## Validation

Passed:

```bash
python -m py_compile \
  scripts/build_recogdrive_jepa_overlay_from_chunks.py \
  scripts/build_last_vla_full_geometry_cache.py \
  scripts/merge_last_vla_overlay_shards.py \
  scripts/merge_last_vla_geometry_cache_into_chunks.py \
  scripts/audit_last_vla_cache_manifest.py
```

Passed:

```bash
bash -n \
  scripts/last_vla_v2/decoupled_highcap_no_risk/prepare_decoupled_highcap_no_risk_data.sh \
  scripts/last_vla_v2/decoupled_highcap_no_risk/run_strict_preflight_decoupled.sh \
  scripts/last_vla_v2/decoupled_highcap_no_risk/serverA_frozen_vlm_decoupled_highcap_no_risk.sh \
  scripts/last_vla_v2/decoupled_highcap_no_risk/serverB_vlm_lora_decoupled_highcap_no_risk.sh
```

Passed:

```bash
pytest -q \
  tests/test_last_vla_decoupled_cache_sharding.py \
  tests/test_last_vla_decoupled_*.py \
  tests/test_dit_decoupled_cot_branch_zero_init.py \
  tests/test_last_vla_*.py \
  tests/test_last_rd_*.py \
  tests/test_expert_*.py \
  tests/test_no_future_leakage.py
```

Result: `121 passed, 3 skipped`.

Passed dry-run:

```bash
RUN_CACHE=0 \
BASE_CHUNK_ROOT=/tmp/base \
OUTPUT_ROOT=/tmp/out \
VGGT_MODEL_PATH=/tmp/vggt \
VJEPA_MODEL_PATH=/tmp/vjepa \
NUM_SHARDS=2 \
SHARD_INDEX=0 \
scripts/last_vla_v2/decoupled_highcap_no_risk/prepare_decoupled_highcap_no_risk_data.sh
```

The dry-run command log points to shard dirs and does not call final merge/audit.

Also passed:

- `git diff --check`
- merge-only dry-run with `SKIP_BUILD_SHARDS=1 MERGE_SHARDS=1` and no VGGT/VJEPA paths
- preflight dry-run writing `Status: NOT READY`

## Next Commands

Single-machine cache generation:

```bash
RUN_CACHE=1 \
BASE_CHUNK_ROOT=/path/to/base_train_chunks \
OUTPUT_ROOT=/path/to/last_vla_v2_outputs \
VGGT_MODEL_PATH=/path/to/VGGT-1B \
VJEPA_MODEL_PATH=/path/to/vjepa2 \
ALLOW_FULL_CACHE_WITHOUT_MAX=1 \
scripts/last_vla_v2/decoupled_highcap_no_risk/prepare_decoupled_highcap_no_risk_data.sh
```

Two-machine sharded cache generation, server 0:

```bash
RUN_CACHE=1 \
BASE_CHUNK_ROOT=/path/to/base_train_chunks \
OUTPUT_ROOT=/path/to/last_vla_v2_outputs \
VGGT_MODEL_PATH=/path/to/VGGT-1B \
VJEPA_MODEL_PATH=/path/to/vjepa2 \
NUM_SHARDS=2 \
SHARD_INDEX=0 \
MERGE_SHARDS=0 \
ALLOW_FULL_CACHE_WITHOUT_MAX=1 \
scripts/last_vla_v2/decoupled_highcap_no_risk/prepare_decoupled_highcap_no_risk_data.sh
```

Two-machine sharded cache generation, server 1:

```bash
RUN_CACHE=1 \
BASE_CHUNK_ROOT=/path/to/base_train_chunks \
OUTPUT_ROOT=/path/to/last_vla_v2_outputs \
VGGT_MODEL_PATH=/path/to/VGGT-1B \
VJEPA_MODEL_PATH=/path/to/vjepa2 \
NUM_SHARDS=2 \
SHARD_INDEX=1 \
MERGE_SHARDS=0 \
ALLOW_FULL_CACHE_WITHOUT_MAX=1 \
scripts/last_vla_v2/decoupled_highcap_no_risk/prepare_decoupled_highcap_no_risk_data.sh
```

Merge/audit:

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

Readiness:

```bash
RUN_PREFLIGHT=1 \
FULL_HIGHCAP_TRAIN_CHUNK_ROOT=/path/to/last_vla_v2_outputs/decoupled_highcap_no_risk/train_full_highcap_chunks \
OUT_ROOT=/path/to/last_vla_v2_outputs \
scripts/last_vla_v2/decoupled_highcap_no_risk/run_strict_preflight_decoupled.sh
```

A launch:

```bash
RUN_TRAIN=1 \
FULL_HIGHCAP_TRAIN_CHUNK_ROOT=/path/to/last_vla_v2_outputs/decoupled_highcap_no_risk/train_full_highcap_chunks \
OUT_ROOT=/path/to/last_vla_v2_outputs \
MASTER_PORT=29601 \
scripts/last_vla_v2/decoupled_highcap_no_risk/serverA_frozen_vlm_decoupled_highcap_no_risk.sh
```

B launch:

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
