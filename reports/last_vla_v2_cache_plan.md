# Last-VLA v2 Cache Plan

Updated: 2026-06-02

## Active Training Cache

- Root: `/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1`
- Train chunk pattern: `train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*`
- Train-side records indexed: `103036`
- Current use: active `cot_alignment` training on local seed0 and remote seed1.
- Readiness manifest: `/mnt/project/VLA-AD/cache/last_vla_v2/manifests/full_v1_teacher_traj_readiness_2048.json`
- Teacher trajectory coverage: `0.0`, so this root is not ready for `teacher_traj_sft`.

## Strict Geometry Overlay

Purpose: replace patch-fallback geometry supervision with true VGGT full-geometry tokens for SOTA-oriented Last-VLA configs.

Current finding:

- bf16 strict VGGT geometry probe failed.
- fp32 strict VGGT geometry probe succeeded after aligning VGGT image preprocessing to `[0, 1]`.
- Probe root: `/mnt/project/VLA-AD/cache/last_vla_v2/full_geometry/probe_fp32_20260602_163602`
- Probe keys include `vggt_geometry_tokens`, `vggt_geometry_target_tokens`, `vggt_context_tokens`, `vggt_target_tokens`, and `vggt_geometry_mode=full_geometry`.
- Merge smoke root: `/mnt/project/VLA-AD/cache/last_vla_v2/full_geometry/merge_probe_fp32_20260602_164948`
- Active pilot root: `/mnt/project/VLA-AD/cache/last_vla_v2/full_geometry/raw_fp32_pilot_20260602_165103`

Production layout:

- Raw overlay chunks: `/mnt/project/VLA-AD/cache/last_vla_v2/full_geometry/raw_fp32_v1`
- Merged full-v1 chunks: `/mnt/project/VLA-AD/cache/last_vla_v2/full_geometry/merged_full_v1_fp32_v1`
- Merge summary: `<merged_root>/last_vla_geometry_merge_summary.json`

Generation command template:

```bash
OUTPUT_CACHE_ROOT=/mnt/project/VLA-AD/cache/last_vla_v2/full_geometry/raw_fp32_v1 \
VGGT_MODEL_PATH=/mnt/project/VLA-AD/checkpoints/teachers/VGGT-1B \
CHUNK_INDEX=0 \
CHUNK_START=0 \
CHUNK_SIZE=1024 \
PRECISION=fp32 \
CUDA_VISIBLE_DEVICES=0 \
scripts/last_vla_v2/generate_full_geometry_overlay_cache.sh
```

Merge command template:

```bash
BASE_CHUNK_ROOT=/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1 \
GEOMETRY_CACHE_ROOT=/mnt/project/VLA-AD/cache/last_vla_v2/full_geometry/raw_fp32_v1 \
OUTPUT_CHUNK_ROOT=/mnt/project/VLA-AD/cache/last_vla_v2/full_geometry/merged_full_v1_fp32_v1 \
STRICT_COVERAGE=1 \
scripts/last_vla_v2/merge_full_geometry_overlay_cache.sh
```

Operational note: full-geometry generation is GPU-heavy and should be chunked. It can run as a background cache job, but a full 103k-sample pass will compete with active 8-GPU training if launched immediately.

## Teacher Trajectory Cache

Purpose: support `last_vla_stage=teacher_traj_sft` using best-of-K or reranked trajectory targets.

Dependency:

- Needs a usable checkpoint from `progressive_sft_bottleneck`, not the current `cot_alignment` checkpoint.
- Best-of-K oracle should be run before SFT. If oracle best-of-8 does not beat deterministic by at least 1 PDMS point, teacher trajectory SFT is not expected to recover a 3-point gain.

Production layout:

- Oracle reports: `/mnt/project/VLA-AD/outputs/last_vla_v2/oracle/<checkpoint_tag>`
- Raw teacher cache: `/mnt/project/VLA-AD/cache/last_vla_v2/teacher_traj/raw/<checkpoint_tag>`
- Merged teacher chunks: `/mnt/project/VLA-AD/cache/last_vla_v2/teacher_traj/merged_full_v1/<checkpoint_tag>`

Current scoring availability:

- Found metric caches only for navtest: `/mnt/project/VLA-AD/cache/metric_cache_navtest_*`
- No train split PDM metric cache found yet, so teacher trajectory generation currently falls back to proxy scoring unless train metric cache is produced.

Generation command template:

```bash
CONFIG=configs/last_vla_v2/last_vla_progressive_bottleneck_eval.yaml \
CHECKPOINT=/path/to/progressive/latest.ckpt \
TRAIN_CHUNK_CACHE_ROOT=/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1 \
OUTPUT_CACHE_ROOT=/mnt/project/VLA-AD/cache/last_vla_v2/teacher_traj/raw/<checkpoint_tag> \
NUM_CANDIDATES=8 \
SCORE_MODE=proxy \
scripts/last_vla_v2/generate_teacher_traj_cache.sh
```

Merge command:

```bash
python scripts/merge_last_vla_teacher_cache_into_chunks.py \
  --base-chunk-root /mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1 \
  --teacher-cache-root /mnt/project/VLA-AD/cache/last_vla_v2/teacher_traj/raw/<checkpoint_tag> \
  --output-chunk-root /mnt/project/VLA-AD/cache/last_vla_v2/teacher_traj/merged_full_v1/<checkpoint_tag> \
  --chunk-name-pattern 'train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*'
```

## Risk Labels

No risk label cache was found or generated. Keep `last_vla_risk_loss_weight=0.0` unless a future cache provides `risk_labels` with validated `[B,H,C]` semantics.

## Immediate Action

- Current training continues on the original `full_v1` cache with patch geometry fallback.
- A 32-sample strict geometry fp32 pilot is running on local GPU0. Do not expand to full production until this pilot writes valid samples/metadata.
- Strict geometry overlay can then be produced in fp32 chunk jobs and merged into a separate root for later strict-geometry experiments.
- Teacher trajectory cache should wait until progressive bottleneck training produces a checkpoint; generation paths and scripts are ready.
