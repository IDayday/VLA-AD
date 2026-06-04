# Last-VLA v2 High-cap No-risk Runbook

Baseline: A0-official-aligned `step_00100000`, full navtest PDMS `0.864891`, eval precision `fp32`.

This is the formal Last-VLA v2 SFT path. Minimal configs are smoke only. Do not set `RUN_CACHE=1`, `RUN_TRAIN=1`, or `RUN_EVAL=1` until the commands have been reviewed.

## Token Contract

- VLM summary: `64`
- latent CoT: `192`
- JEPA context/target: `[128, 1024]`
- dynamic teacher: `[128, 1024]`
- VGGT full geometry: `[192, 512]`
- geometry grid: `[12, 16]`
- risk: disabled, no risk labels required
- DiT context in eval: `192 + 64 = 256`

Full raw VLM hidden tokens do not enter DiT in bottleneck mode. Patch fallback is not allowed. Old 12-token JEPA caches are invalid for strict high-cap.

## 1. Generate High-cap Expert/JEPA Chunk Cache

Use dense JEPA outputs and strict high-cap pooling:

```bash
python scripts/build_recogdrive_chunk_cache.py \
  --split navtrain \
  --chunk-index 0 \
  --chunk-size 1024 \
  --output-dir /path/to/highcap_chunks/train_full_chunk_000000 \
  --build-vlm-hidden \
  --build-jepa \
  --build-vggt \
  --require-vggt-geometry \
  --recogdrive-vlm-path /path/to/base_vlm \
  --jepa-model-path /path/to/vjepa2 \
  --vggt-model-path /path/to/VGGT-1B \
  --num-jepa-tokens 128 \
  --num-vggt-tokens 128 \
  --num-geometry-tokens 192 \
  --geometry-grid-rows 12 \
  --geometry-grid-cols 16 \
  --vggt-geometry-teacher-dim 512 \
  --strict-highcap-jepa \
  --precision bf16
```

For smoke only, add `--max-samples N`. Production cache generation should be scheduled explicitly outside this task.

## 2. Generate Full Geometry Overlay

Dry-run:

```bash
CHUNK_CACHE_ROOT=/path/to/highcap_or_base_chunks \
OUTPUT_CACHE_ROOT=$OUT_ROOT/highcap_no_risk/full_geometry_overlay \
VGGT_MODEL_PATH=/path/to/VGGT-1B \
scripts/last_vla_v2/highcap_no_risk/generate_full_geometry_overlay_cache_highcap.sh
```

Small sample execution:

```bash
RUN_CACHE=1 \
MAX_SAMPLES=16 \
CHUNK_CACHE_ROOT=/path/to/highcap_or_base_chunks \
OUTPUT_CACHE_ROOT=$OUT_ROOT/highcap_no_risk/full_geometry_overlay \
VGGT_MODEL_PATH=/path/to/VGGT-1B \
scripts/last_vla_v2/highcap_no_risk/generate_full_geometry_overlay_cache_highcap.sh
```

## 3. Merge Geometry Overlay

```bash
BASE_CHUNK_ROOT=/path/to/highcap_base_chunks \
GEOMETRY_CACHE_ROOT=$OUT_ROOT/highcap_no_risk/full_geometry_overlay \
OUTPUT_CHUNK_ROOT=$OUT_ROOT/highcap_no_risk/full_geometry_chunks \
scripts/last_vla_v2/highcap_no_risk/merge_full_geometry_overlay_cache_highcap.sh
```

Set `RUN_CACHE=1` only when ready to write the merged cache.

## 4. Audit High-cap Cache

```bash
python scripts/audit_last_vla_cache_manifest.py \
  --cache-root $OUT_ROOT/highcap_no_risk/full_geometry_chunks \
  --strict-full-geometry \
  --strict-no-risk \
  --min-full-geometry-coverage 0.99 \
  --expected-jepa-tokens 128 \
  --expected-geometry-tokens 192 \
  --geometry-teacher-dim 512 \
  --output $OUT_ROOT/highcap_no_risk/preflight/highcap_manifest.json
```

Readiness requires base key coverage `1.0`, JEPA context/target coverage at least `0.99`, full geometry coverage at least `0.99`, patch fallback count `0`, duplicate sample tokens `0`, and risk labels ignored or absent.

## 5. Server A Frozen VLM Train

Dry-run:

```bash
FULL_GEOMETRY_CHUNK_ROOT=$OUT_ROOT/highcap_no_risk/full_geometry_chunks \
A0_INIT_CHECKPOINT=/path/to/step_00100000.ckpt \
OUT_ROOT=$OUT_ROOT/highcap_no_risk \
MASTER_PORT=29531 \
scripts/last_vla_v2/highcap_no_risk/serverA_frozen_vlm_highcap_no_risk.sh
```

Actual training:

```bash
RUN_TRAIN=1 \
FULL_GEOMETRY_CHUNK_ROOT=$OUT_ROOT/highcap_no_risk/full_geometry_chunks \
A0_INIT_CHECKPOINT=/path/to/step_00100000.ckpt \
OUT_ROOT=$OUT_ROOT/highcap_no_risk \
MASTER_PORT=29531 \
scripts/last_vla_v2/highcap_no_risk/serverA_frozen_vlm_highcap_no_risk.sh
```

## 6. Server B VLM-LoRA Train and Hidden Regeneration

Dry-run:

```bash
FULL_GEOMETRY_CHUNK_ROOT=$OUT_ROOT/highcap_no_risk/full_geometry_chunks \
A0_INIT_CHECKPOINT=/path/to/step_00100000.ckpt \
VLM_PATH=/path/to/base_vlm \
NAVSIM_LOG_PATH=/path/to/navsim_logs \
SENSOR_BLOBS_PATH=/path/to/sensor_blobs \
OUT_ROOT=$OUT_ROOT/highcap_no_risk \
MASTER_PORT=29541 \
scripts/last_vla_v2/highcap_no_risk/serverB_vlm_lora_highcap_no_risk.sh
```

The Server B launcher writes both `last_vla_cot_adapter.pt` and `vlm_lora_adapter_state.pt` extraction commands, regenerates the train hidden cache with `--cache-variant highcap_no_risk`, and then launches progressive bottleneck SFT on the regenerated cache when `RUN_TRAIN=1`.

Regenerate Line B navtest hidden cache before eval:

```bash
NAVTEST_CHUNK_CACHE_ROOT=/path/to/navtest_highcap_full_geometry_chunks \
VLM_PATH=/path/to/base_vlm \
VLM_LORA_ADAPTER=$OUT_ROOT/highcap_no_risk/serverB_lora_highcap_no_risk/vlm_lora_cot_alignment/adapters/vlm_lora_adapter_state.pt \
OUTPUT_CHUNK_ROOT=$OUT_ROOT/highcap_no_risk/serverB_lora_highcap_no_risk/lora_navtest_cache \
scripts/last_vla_v2/highcap_no_risk/build_lora_navtest_cache_highcap.sh
```

Set `RUN_CACHE=1` only when ready. Line B eval must use this LoRA-regenerated navtest hidden cache.

## 7. Checkpoint Sweep

```bash
OUT_ROOT=$OUT_ROOT/highcap_no_risk \
NAVTEST_CHUNK_CACHE_ROOT=/path/to/navtest_highcap_full_geometry_chunks \
LORA_NAVTEST_CHUNK_CACHE_ROOT=$OUT_ROOT/highcap_no_risk/serverB_lora_highcap_no_risk/lora_navtest_cache \
METRIC_CACHE_DIR=/path/to/navtest_metric_cache \
scripts/last_vla_v2/highcap_no_risk/eval_checkpoint_sweep_highcap.sh
```

Set `RUN_EVAL=1` only for full eval.

## 8. CoT Corruption Eval

```bash
OUT_ROOT=$OUT_ROOT/highcap_no_risk \
CHECKPOINT=/path/to/best_progressive.ckpt \
NAVTEST_CHUNK_CACHE_ROOT=/path/to/navtest_highcap_full_geometry_chunks \
METRIC_CACHE_DIR=/path/to/navtest_metric_cache \
MAX_SAMPLES=1000 \
scripts/last_vla_v2/highcap_no_risk/eval_cot_corruption_highcap.sh
```

Set `RUN_EVAL=1` to execute and `FULL_CORRUPTION=1` to run full corruption eval.

## 9. Summarize

```bash
python scripts/last_vla_v2/highcap_no_risk/summarize_highcap_no_risk.py \
  --out-root $OUT_ROOT/highcap_no_risk
```

No performance should be claimed before full training and full PDM eval artifacts exist.
