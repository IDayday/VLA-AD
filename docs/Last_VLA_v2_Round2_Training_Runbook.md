# Last-VLA v2 Round2 Training Runbook

Baseline: A0-official-aligned `step_00100000`, full navtest PDMS `0.864891`, eval precision `fp32`.

This runbook prepares two full SFT lines. Do not set `RUN_TRAIN=1` or `RUN_EVAL=1` until preflight passes.

## Current Status

The old Round2/minimal launchers and teacher-trajectory/PDM-reranked Last-VLA configs have been moved to `scripts/last_vla_v2/archive/` and `configs/last_vla_v2/archive/`. They are retained for historical inspection only and are not production training entrypoints.

Formal Last-VLA v2 training now uses only the decoupled high-capacity no-risk path:

- `scripts/last_vla_v2/decoupled_highcap_no_risk/serverA_frozen_vlm_decoupled_highcap_no_risk.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/serverB_vlm_lora_decoupled_highcap_no_risk.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/run_strict_preflight_decoupled.sh`
- `scripts/last_vla_v2/decoupled_highcap_no_risk/prepare_decoupled_highcap_no_risk_data.sh`

Old hard-bottleneck configs are not compatible with the current code path. Minimal `4/32/12/12`, summary replacement, and `patch_fallback` configs must not be used as formal training.

## High-capacity No-risk Official Config

The minimal Last-VLA configs are smoke/debug configs only. The formal Last-VLA v2 training config is the high-capacity no-risk line in `docs/Last_VLA_v2_HighCap_NoRisk_Runbook.md`.

Official values:

- raw VLM tokens preserved as base DiT context
- CoT `192`
- JEPA context/target `128 x 1024`
- VGGT full geometry `192 x 512`
- geometry grid `12 x 16`
- risk disabled
- CoT condition tokens `192` enter through a zero-init residual branch

VLM summary tokens are not generated or consumed. The high-cap line requires regenerated full geometry cache and regenerated 128-token JEPA cache; old 12-token JEPA cache is not acceptable for strict high-cap training. Line B VLM-LoRA requires regenerated train and navtest hidden caches before frozen-cache training/eval.

## Full Geometry Cache

Build strict full VGGT geometry overlay:

```bash
CHUNK_CACHE_ROOT=/path/to/original_train_chunks \
OUTPUT_CACHE_ROOT=$OUT_ROOT/last_vla_v2/full_geometry_overlay \
VGGT_MODEL_PATH=/path/to/VGGT-1B \
GEOMETRY_TEACHER_DIM=512 \
scripts/last_vla_v2/generate_full_geometry_overlay_cache.sh
```

Merge overlay into train chunks:

```bash
BASE_CHUNK_ROOT=/path/to/original_train_chunks \
GEOMETRY_CACHE_ROOT=$OUT_ROOT/last_vla_v2/full_geometry_overlay \
OUTPUT_CHUNK_ROOT=$OUT_ROOT/last_vla_v2/full_geometry_chunks \
STRICT_COVERAGE=1 \
GEOMETRY_TEACHER_DIM=512 \
scripts/last_vla_v2/merge_full_geometry_overlay_cache.sh
```

Audit strict full geometry:

```bash
python scripts/audit_last_vla_cache_manifest.py \
  --cache-root $OUT_ROOT/last_vla_v2/full_geometry_chunks \
  --strict-full-geometry \
  --min-full-geometry-coverage 0.99 \
  --geometry-teacher-dim 512 \
  --output $OUT_ROOT/preflight/full_geometry_manifest.json
```

## Preflight

```bash
TRAIN_CHUNK_CACHE_ROOT=/path/to/original_train_chunks \
FULL_GEOMETRY_CHUNK_ROOT=$OUT_ROOT/last_vla_v2/full_geometry_chunks \
A0_INIT_CHECKPOINT=/path/to/step_00100000.ckpt \
OUT_ROOT=$OUT_ROOT/last_vla_v2/round2 \
scripts/last_vla_v2/round2/run_strict_preflight.sh
```

## Server A: Frozen VLM

Dry-run:

```bash
FULL_GEOMETRY_CHUNK_ROOT=$OUT_ROOT/last_vla_v2/full_geometry_chunks \
A0_INIT_CHECKPOINT=/path/to/step_00100000.ckpt \
OUT_ROOT=$OUT_ROOT/last_vla_v2/round2 \
MASTER_PORT=29531 \
scripts/last_vla_v2/round2/serverA_frozen_vlm_full_sft.sh
```

Actual training:

```bash
RUN_TRAIN=1 \
FULL_GEOMETRY_CHUNK_ROOT=$OUT_ROOT/last_vla_v2/full_geometry_chunks \
A0_INIT_CHECKPOINT=/path/to/step_00100000.ckpt \
OUT_ROOT=$OUT_ROOT/last_vla_v2/round2 \
MASTER_PORT=29531 \
scripts/last_vla_v2/round2/serverA_frozen_vlm_full_sft.sh
```

## Server B: VLM-LoRA

The Server B launchers pass `NAVSIM_LOG_PATH` and `SENSOR_BLOBS_PATH` through to Hydra as
`navsim_log_path=${NAVSIM_LOG_PATH}` and `sensor_blobs_path=${SENSOR_BLOBS_PATH}`. Check
`commands.log` before setting `RUN_TRAIN=1`.

The formal Line B LoRA setting is now `attention_mlp` / `llm` / `r=32` / `alpha=64` / `dropout=0.05` / `rsLoRA=true`. The launcher writes `lora_target_report.json`, `trainable_parameter_counts.json`, `lora_training_config.json`, and `lora_runtime_report.json` during startup. Adapter extraction writes both the Last-VLA CoT adapter and a `vlm_lora/` directory containing LoRA state, config, metadata, and target audit. Hidden-cache regeneration should use `--vlm-lora-adapter-dir`, not manually retyped rank/alpha/target modules.

Dry-run:

```bash
FULL_GEOMETRY_CHUNK_ROOT=$OUT_ROOT/last_vla_v2/full_geometry_chunks \
A0_INIT_CHECKPOINT=/path/to/step_00100000.ckpt \
VLM_PATH=/path/to/base_vlm \
NAVSIM_LOG_PATH=/path/to/navsim/logs \
SENSOR_BLOBS_PATH=/path/to/sensor_blobs \
OUT_ROOT=$OUT_ROOT/last_vla_v2/round2 \
MASTER_PORT=29541 \
scripts/last_vla_v2/round2/serverB_vlm_lora_full_sft.sh
```

Actual training:

```bash
RUN_TRAIN=1 \
FULL_GEOMETRY_CHUNK_ROOT=$OUT_ROOT/last_vla_v2/full_geometry_chunks \
A0_INIT_CHECKPOINT=/path/to/step_00100000.ckpt \
VLM_PATH=/path/to/base_vlm \
NAVSIM_LOG_PATH=/path/to/navsim/logs \
SENSOR_BLOBS_PATH=/path/to/sensor_blobs \
OUT_ROOT=$OUT_ROOT/last_vla_v2/round2 \
MASTER_PORT=29541 \
scripts/last_vla_v2/round2/serverB_vlm_lora_full_sft.sh
```

Single-stage VLM-LoRA CoT alignment can also be dry-run directly:

```bash
NAVSIM_LOG_PATH=/path/to/navsim/logs \
SENSOR_BLOBS_PATH=/path/to/sensor_blobs \
TRAIN_CHUNK_CACHE_ROOT=$OUT_ROOT/last_vla_v2/full_geometry_chunks \
VLM_PATH=/path/to/base_vlm \
OUTPUT_DIR=$OUT_ROOT/last_vla_v2/round2/serverB_lora/vlm_lora_cot_alignment \
MASTER_PORT=29541 \
scripts/last_vla_v2/run_vlm_lora_cot_alignment_8gpu.sh
```

## LoRA Hidden Cache

Regenerate train hidden cache after B1:

```bash
python scripts/build_recogdrive_hidden_cache_with_lora.py \
  --base-chunk-root $OUT_ROOT/last_vla_v2/full_geometry_chunks \
  --output-chunk-root $OUT_ROOT/last_vla_v2/round2/serverB_lora/lora_hidden_cache \
  --vlm-path /path/to/base_vlm \
  --vlm-lora-adapter $OUT_ROOT/last_vla_v2/round2/serverB_lora/vlm_lora_cot_alignment/adapters/vlm_lora_adapter_state.pt \
  --precision bf16
```

Regenerate navtest hidden cache for Line B eval:

```bash
NAVTEST_CHUNK_CACHE_ROOT=/path/to/navtest_full_geometry_chunks \
VLM_PATH=/path/to/base_vlm \
VLM_LORA_ADAPTER=$OUT_ROOT/last_vla_v2/round2/serverB_lora/vlm_lora_cot_alignment/adapters/vlm_lora_adapter_state.pt \
OUTPUT_CHUNK_ROOT=$OUT_ROOT/last_vla_v2/round2/serverB_lora/lora_navtest_cache \
scripts/last_vla_v2/round2/build_lora_navtest_cache.sh
```

Set `RUN_CACHE=1` to execute cache regeneration.

## Checkpoint Sweep Eval

Dry-run:

```bash
OUT_ROOT=$OUT_ROOT/last_vla_v2/round2 \
NAVTEST_CHUNK_CACHE_ROOT=/path/to/navtest_full_geometry_chunks \
LORA_NAVTEST_CHUNK_CACHE_ROOT=$OUT_ROOT/last_vla_v2/round2/serverB_lora/lora_navtest_cache \
METRIC_CACHE_DIR=/path/to/navtest_metric_cache \
scripts/last_vla_v2/round2/eval_round2_checkpoint_sweep.sh
```

Set `RUN_EVAL=1` only when ready for full eval.

## CoT Corruption Eval

```bash
OUT_ROOT=$OUT_ROOT/last_vla_v2/round2 \
CHECKPOINT=/path/to/best_progressive.ckpt \
NAVTEST_CHUNK_CACHE_ROOT=/path/to/navtest_full_geometry_chunks \
METRIC_CACHE_DIR=/path/to/navtest_metric_cache \
MAX_SAMPLES=1000 \
scripts/last_vla_v2/round2/eval_round2_cot_corruption.sh
```

Set `RUN_EVAL=1` to execute and `FULL_CORRUPTION=1` to run full corruption eval.

## Summary

```bash
python scripts/last_vla_v2/round2/summarize_round2.py \
  --out-root $OUT_ROOT/last_vla_v2/round2
```

Success thresholds:

- minimum: PDMS >= `0.870`
- meaningful: PDMS >= `0.880`
- SFT target: PDMS >= `0.895`
- CoT dependency: normal - zero_all_cot >= `0.01` full or `0.008` on 1k
