# Last-VLA v2 Implementation Report

## Status

Implementation and targeted verification completed. Training dispatch started after verification per operator request. No full eval launched.

A0-official baseline is 0.864891.

## Changed Files

- `navsim/agents/recogdrive/recogdrive_diffusion_planner.py`
- `navsim/agents/recogdrive/recogdrive_agent.py`
- `navsim/agents/recogdrive/expert_backends.py`
- `navsim/agents/recogdrive/recogdrive_features.py`
- `navsim/agents/recogdrive/expert_cache.py`
- `navsim/planning/script/run_training_recogdrive.py`
- `navsim/planning/script/config/common/agent/recogdrive_agent.yaml`

## New Files

- `navsim/agents/recogdrive/last_vla_cot_planning.py`
- `configs/last_vla_v2/*.yaml`
- `navsim/planning/script/config/experiment/last_vla_*.yaml`
- `scripts/eval_last_vla_best_of_k_oracle.py`
- `scripts/generate_last_vla_teacher_trajectory_cache.py`
- `scripts/merge_last_vla_teacher_cache_into_chunks.py`
- `scripts/merge_last_vla_geometry_cache_into_chunks.py`
- `scripts/audit_last_vla_cache_manifest.py`
- `scripts/eval_last_vla_cot_corruption_pdm.py`
- `scripts/last_vla_v2/*.sh`
- `tests/test_last_vla_*.py`
- `docs/LaST_VLA_ReCogDrive_v2_Design.md`

## Tests Run

- `python -m py_compile navsim/agents/recogdrive/last_vla_cot_planning.py navsim/agents/recogdrive/recogdrive_diffusion_planner.py navsim/agents/recogdrive/recogdrive_agent.py navsim/agents/recogdrive/expert_backends.py navsim/agents/recogdrive/recogdrive_features.py navsim/agents/recogdrive/expert_cache.py navsim/planning/script/run_training_recogdrive.py scripts/eval_last_vla_best_of_k_oracle.py scripts/generate_last_vla_teacher_trajectory_cache.py scripts/merge_last_vla_teacher_cache_into_chunks.py scripts/audit_last_vla_cache_manifest.py scripts/eval_last_vla_cot_corruption_pdm.py`: pass
- `pytest -q tests/test_last_vla_*.py`: pass, 13 passed
- `pytest -q tests/test_last_rd_*.py tests/test_expert_*.py tests/test_no_future_leakage.py`: pass, 18 passed, 2 skipped

## Pending Verification

- Monitor full-v1 CoT alignment to completion and select the best `latest.ckpt` for progressive bottleneck SFT.
- Generate/merge teacher trajectory cache after a progressive checkpoint exists.

## Training Dispatch

- A0 initialization checkpoint: `/mnt/project/VLA-AD/outputs/a0_stage2_repro_20260531_003029/a0_official_aligned_a0complete/latest.ckpt`
- A0-official baseline for all later reports: `0.864891`
- Full training cache: `/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1`
- Full-v1 cache preflight: `27` train chunks, `103036` records sampled without missing required VLM/JEPA/VGGT keys; teacher trajectory coverage is `0.0`.
- Full-geometry VGGT probe: initial bf16 strict probe failed, but fp32 strict probe succeeded after aligning VGGT preprocessing to `[0, 1]`. Probe root: `/mnt/project/VLA-AD/cache/last_vla_v2/full_geometry/probe_fp32_20260602_163602`. Current active training still uses explicit patch-fallback geometry-lite (`last_vla_allow_patch_geometry_fallback=true`, `last_vla_require_full_geometry=false`).
- Cache plan: `reports/last_vla_v2_cache_plan.md`
- Full-geometry merge smoke: `/mnt/project/VLA-AD/cache/last_vla_v2/full_geometry/merge_probe_fp32_20260602_164948`, merged `1` fp32 probe sample into a `4096`-record backfill chunk while preserving `last_hidden_state`, JEPA, and VGGT context keys.
- Active cache pilot: `/mnt/project/VLA-AD/cache/last_vla_v2/full_geometry/raw_fp32_pilot_20260602_165103`, local GPU0, `MAX_SAMPLES=32`, fp32 strict VGGT overlay. It is a pilot only; not a full production geometry cache.
- Completed small A0 initialized CoT alignment smoke runs:
  - `/mnt/project/VLA-AD/outputs/last_vla_v2/cot_alignment_a0full_local_seed0_20260602_161310/latest.ckpt`
  - `/mnt/project/VLA-AD/outputs/last_vla_v2/cot_alignment_a0full_remote_seed1_20260602_161310/latest.ckpt`
- Active full-v1 8-GPU CoT alignment runs:
  - local seed0: `/mnt/project/VLA-AD/outputs/last_vla_v2/cot_alignment_fullv1_a0_local_seed0_20260602_161736`
  - remote seed1: `/mnt/project/VLA-AD/outputs/last_vla_v2/cot_alignment_fullv1_a0_remote_seed1_20260602_161736`

## Unimplemented TODOs

- Real PDM scoring in teacher cache generation currently has a proxy fallback path; NAVSIM metric-cache integration should be validated on real cache before production use.
- Full geometry teacher cache is not yet available for all train chunks; fp32 overlay generation and merge tooling is ready. SOTA configs should enable `last_vla_require_full_geometry=true` only after strict full-geometry cache coverage is generated and audited.

## Commands

Best-of-K oracle:

```bash
CONFIG=configs/last_vla_v2/last_vla_progressive_bottleneck_eval.yaml \
CHECKPOINT=/path/to/checkpoint.ckpt \
TRAIN_CHUNK_CACHE_ROOT=/path/to/navtrain_chunks \
OUTPUT_DIR=$OUT_ROOT/last_vla_v2/oracle \
scripts/last_vla_v2/run_best_of_k_oracle.sh
```

Teacher trajectory cache generation:

```bash
CONFIG=configs/last_vla_v2/last_vla_progressive_bottleneck_eval.yaml \
CHECKPOINT=/path/to/checkpoint.ckpt \
TRAIN_CHUNK_CACHE_ROOT=/path/to/navtrain_chunks \
OUTPUT_CACHE_ROOT=$OUT_ROOT/last_vla_v2/teacher_cache \
scripts/last_vla_v2/generate_teacher_traj_cache.sh
```

CoT alignment training:

```bash
TRAIN_CHUNK_CACHE_ROOT=/path/to/navtrain_chunks \
OUTPUT_DIR=$OUT_ROOT/last_vla_v2/cot_alignment \
MASTER_PORT=29521 \
scripts/last_vla_v2/run_cot_alignment_8gpu.sh
```

Progressive bottleneck SFT:

```bash
TRAIN_CHUNK_CACHE_ROOT=/path/to/navtrain_chunks \
COT_ALIGNMENT_CHECKPOINT=/path/to/cot_alignment.ckpt \
OUTPUT_DIR=$OUT_ROOT/last_vla_v2/progressive_bottleneck \
MASTER_PORT=29522 \
scripts/last_vla_v2/run_progressive_bottleneck_8gpu.sh
```

Teacher trajectory SFT:

```bash
TEACHER_TRAJ_CHUNK_ROOT=/path/to/merged_teacher_chunks \
PROGRESSIVE_CHECKPOINT=/path/to/progressive.ckpt \
OUTPUT_DIR=$OUT_ROOT/last_vla_v2/teacher_traj_sft \
MASTER_PORT=29523 \
scripts/last_vla_v2/run_teacher_traj_sft_8gpu.sh
```

Corruption eval:

```bash
python scripts/eval_last_vla_cot_corruption_pdm.py \
  --config configs/last_vla_v2/last_vla_progressive_bottleneck_eval.yaml \
  --checkpoint /path/to/checkpoint.ckpt \
  --chunk-cache-root /path/to/navtest_chunks \
  --output-dir $OUT_ROOT/last_vla_v2/corruption/zero_all \
  --zero-all-cot
```
