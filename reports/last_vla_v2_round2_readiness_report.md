# Last-VLA v2 Round2 Readiness Report

## Baseline

A0-official-aligned best checkpoint: `step_00100000`.
Full navtest PDMS baseline: `0.864891`.
Eval precision baseline: `fp32`.

## Status

Round2 SFT readiness code is implemented for two experiment lines:

- Line A: frozen VLM, original ReCogDrive Stage1 hidden cache, strict full-geometry teacher cache.
- Line B: online VLM-LoRA CoT alignment, LoRA hidden-cache regeneration, then Last-VLA v2 progressive SFT on regenerated hidden cache.

No training launched.
No full navtest eval launched.
No performance is claimed before training and full PDM eval.

## Changed Files

- `navsim/agents/recogdrive/last_vla_cot_planning.py`
- `navsim/agents/recogdrive/recogdrive_diffusion_planner.py`
- `navsim/agents/recogdrive/recogdrive_agent.py`
- `navsim/agents/recogdrive/expert_extractors/vggt_extractor.py`
- `navsim/planning/script/run_training_recogdrive.py`
- `navsim/planning/script/config/common/agent/recogdrive_agent.yaml`
- `navsim/planning/script/config/experiment/last_vla_cot_alignment.yaml`
- `navsim/planning/script/config/experiment/last_vla_progressive_bottleneck.yaml`
- `configs/last_vla_v2/last_vla_cot_alignment.yaml`
- `configs/last_vla_v2/last_vla_progressive_bottleneck.yaml`
- `configs/last_vla_v2/last_vla_progressive_bottleneck_eval.yaml`
- `scripts/audit_last_vla_cache_manifest.py`
- `scripts/build_recogdrive_chunk_cache.py`
- `scripts/eval_last_vla_cot_corruption_pdm.py`
- `scripts/merge_last_vla_geometry_cache_into_chunks.py`
- `scripts/last_vla_v2/generate_full_geometry_overlay_cache.sh`
- `scripts/last_vla_v2/merge_full_geometry_overlay_cache.sh`
- `docs/LaST_VLA_ReCogDrive_v2_Design.md`
- `tests/test_last_vla_config_compose.py`

## New Files

- `navsim/agents/recogdrive/geometry_tokenizer.py`
- `scripts/build_last_vla_full_geometry_cache.py`
- `scripts/build_recogdrive_hidden_cache_with_lora.py`
- `scripts/last_vla_v2/extract_vlm_lora_and_cot_adapters.py`
- `scripts/last_vla_v2/run_vlm_lora_cot_alignment_8gpu.sh`
- `scripts/last_vla_v2/round2/run_strict_preflight.sh`
- `scripts/last_vla_v2/round2/serverA_frozen_vlm_full_sft.sh`
- `scripts/last_vla_v2/round2/serverB_vlm_lora_full_sft.sh`
- `scripts/last_vla_v2/round2/build_lora_navtest_cache.sh`
- `scripts/last_vla_v2/round2/eval_round2_checkpoint_sweep.sh`
- `scripts/last_vla_v2/round2/eval_round2_cot_corruption.sh`
- `scripts/last_vla_v2/round2/summarize_round2.py`
- `configs/last_vla_v2/last_vla_cot_alignment_geometry_lite.yaml`
- `configs/last_vla_v2/last_vla_vlm_lora_cot_alignment.yaml`
- `navsim/planning/script/config/experiment/last_vla_vlm_lora_cot_alignment.yaml`
- `docs/Last_VLA_v2_Round2_Training_Runbook.md`
- `tests/test_last_vla_geometry_token_packer.py`
- `tests/test_last_vla_geometry_manifest.py`
- `tests/test_last_vla_residual_alpha_schedule.py`
- `tests/test_last_vla_vlm_summary_schedule.py`
- `tests/test_last_vla_vlm_lora_guard.py`
- `tests/test_last_vla_vlm_lora_trainable_scope.py`
- `tests/test_last_vla_lora_hidden_cache_schema.py`
- `tests/test_last_vla_round2_launchers_dryrun.py`

## Implemented Readiness Items

- Real PDM paths remain strict: `--score-mode pdm` requires NAVSIM metric cache and does not fall back to proxy. Proxy mode is explicit and smoke/debug only.
- Corruption eval now uses runtime corruption flags before downstream CoT stages, so geometry/dynamics/ego/action-refine/coarse-prior corruptions affect planner context or residual inference.
- Full geometry mode is explicit via `vggt_geometry_mode` / `vggt_geometry_mode_code`; `patch_fallback` cannot be labeled as `full_geometry`.
- Geometry teacher dim is configurable through `last_vla_geometry_teacher_dim`, default strict SFT value `512`.
- `GeometryTokenPacker` produces deterministic full-geometry teacher tokens `[12, 512]` from VGGT depth, point map, camera, and optional tracks.
- Strict full-geometry overlay build, merge, and audit scripts are ready.
- Progressive residual alpha schedule is implemented: training transitions from full-action diffusion target to full residual target; inference uses alpha `1.0`.
- VLM summary keep schedule is implemented without reintroducing full raw VLM tokens in bottleneck mode.
- Aux loss decay uses `last_vla_aux_decay_epochs` and reports effective weights.
- VLM-LoRA trainable scope keeps only LoRA and `last_vla_cot` trainable while freezing base backbone/action modules.
- Line B online VLM launchers pass `NAVSIM_LOG_PATH` and `SENSOR_BLOBS_PATH` through as Hydra overrides:
  `navsim_log_path=...` and `sensor_blobs_path=...`.
- LoRA hidden-cache regeneration script preserves teacher keys and supports sharding.
- Round2 launchers are dry-run by default and require `RUN_TRAIN=1`, `RUN_EVAL=1`, or `RUN_CACHE=1` for expensive execution.

## Data Dependencies

- A0 init checkpoint: A0-official-aligned `step_00100000`.
- Full train chunk cache with original ReCogDrive hidden states.
- Strict full VGGT geometry cache merged into train chunks.
- JEPA context and future JEPA target coverage for alignment losses.
- VGGT context and strict full geometry tokens with `vggt_geometry_mode_code=2`.
- Metric cache for any real PDM oracle, corruption, or checkpoint sweep.
- For Line B: base VLM path, online NAVSIM data paths, VLM-LoRA adapter after B1, regenerated train hidden cache, regenerated navtest hidden cache for eval.

## Full Geometry Cache Status

Tooling is ready, but no production full-geometry cache was generated in this task.
Strict SFT should not start until:

- full-geometry coverage is at least `0.99`,
- geometry token shapes match `[12, 512]`,
- patch fallback count is zero under strict mode,
- manifest audit exits successfully.

The requested synthetic manifest fixture `tests/fixtures/synthetic_last_vla_cache` is absent, so that smoke was recorded as skipped.

## VLM-LoRA Readiness

- Online VLM-LoRA CoT alignment config added.
- Guard remains active: `cache_hidden_state=true` with `last_vla_train_vlm_lora=true` raises.
- Trainable scope test verifies only LoRA and Last-VLA CoT are trainable.
- Adapter extraction script writes `last_vla_cot_adapter.pt` and `vlm_lora_adapter_state.pt`.
- Hidden-cache regeneration supports train/navtest chunks and sharding.
- Line B eval must use LoRA-regenerated navtest hidden cache; evaluating LoRA checkpoints on original hidden cache should be treated as mismatch.

## Tests Run

- `python -m py_compile ...`: pass.
- `bash -n ...`: pass.
- `pytest -q tests/test_last_vla_*.py tests/test_last_rd_*.py tests/test_expert_*.py tests/test_no_future_leakage.py`: pass, `54 passed, 2 skipped`.
- `python scripts/eval_last_vla_best_of_k_oracle.py --synthetic-smoke --score-mode proxy --num-candidates 4 --max-samples 4 --output-dir reports/last_vla_v2_oracle_synthetic_smoke`: pass.
- `python scripts/audit_last_vla_cache_manifest.py --cache-root tests/fixtures/synthetic_last_vla_cache --output reports/last_vla_v2_manifest_synthetic.json || true`: skipped because fixture is absent.
- `RUN_TRAIN=0 ... scripts/last_vla_v2/round2/serverB_vlm_lora_full_sft.sh`: pass; `commands.log` contains `navsim_log_path=...` and `sensor_blobs_path=...`.
- `git diff --check`: pass.

## Exact Commands

Build full geometry cache:

```bash
CHUNK_CACHE_ROOT=/path/to/original_train_chunks \
OUTPUT_CACHE_ROOT=$OUT_ROOT/last_vla_v2/full_geometry_overlay \
VGGT_MODEL_PATH=/path/to/VGGT-1B \
GEOMETRY_TEACHER_DIM=512 \
scripts/last_vla_v2/generate_full_geometry_overlay_cache.sh
```

Merge full geometry cache:

```bash
BASE_CHUNK_ROOT=/path/to/original_train_chunks \
GEOMETRY_CACHE_ROOT=$OUT_ROOT/last_vla_v2/full_geometry_overlay \
OUTPUT_CHUNK_ROOT=$OUT_ROOT/last_vla_v2/full_geometry_chunks \
STRICT_COVERAGE=1 \
GEOMETRY_TEACHER_DIM=512 \
scripts/last_vla_v2/merge_full_geometry_overlay_cache.sh
```

Audit full geometry cache:

```bash
python scripts/audit_last_vla_cache_manifest.py \
  --cache-root $OUT_ROOT/last_vla_v2/full_geometry_chunks \
  --strict-full-geometry \
  --min-full-geometry-coverage 0.99 \
  --geometry-teacher-dim 512 \
  --output $OUT_ROOT/preflight/full_geometry_manifest.json
```

Run strict preflight:

```bash
TRAIN_CHUNK_CACHE_ROOT=/path/to/original_train_chunks \
FULL_GEOMETRY_CHUNK_ROOT=$OUT_ROOT/last_vla_v2/full_geometry_chunks \
A0_INIT_CHECKPOINT=/path/to/step_00100000.ckpt \
OUT_ROOT=$OUT_ROOT/last_vla_v2/round2 \
scripts/last_vla_v2/round2/run_strict_preflight.sh
```

Server A frozen VLM:

```bash
RUN_TRAIN=1 \
FULL_GEOMETRY_CHUNK_ROOT=$OUT_ROOT/last_vla_v2/full_geometry_chunks \
A0_INIT_CHECKPOINT=/path/to/step_00100000.ckpt \
OUT_ROOT=$OUT_ROOT/last_vla_v2/round2 \
MASTER_PORT=29531 \
scripts/last_vla_v2/round2/serverA_frozen_vlm_full_sft.sh
```

Server B VLM-LoRA:

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

Single-stage VLM-LoRA CoT alignment:

```bash
RUN_TRAIN=1 \
NAVSIM_LOG_PATH=/path/to/navsim/logs \
SENSOR_BLOBS_PATH=/path/to/sensor_blobs \
TRAIN_CHUNK_CACHE_ROOT=$OUT_ROOT/last_vla_v2/full_geometry_chunks \
VLM_PATH=/path/to/base_vlm \
OUTPUT_DIR=$OUT_ROOT/last_vla_v2/round2/serverB_lora/vlm_lora_cot_alignment \
MASTER_PORT=29541 \
scripts/last_vla_v2/run_vlm_lora_cot_alignment_8gpu.sh
```

Regenerate LoRA hidden cache:

```bash
python scripts/build_recogdrive_hidden_cache_with_lora.py \
  --base-chunk-root $OUT_ROOT/last_vla_v2/full_geometry_chunks \
  --output-chunk-root $OUT_ROOT/last_vla_v2/round2/serverB_lora/lora_hidden_cache \
  --vlm-path /path/to/base_vlm \
  --vlm-lora-adapter $OUT_ROOT/last_vla_v2/round2/serverB_lora/vlm_lora_cot_alignment/adapters/vlm_lora_adapter_state.pt \
  --precision bf16
```

Regenerate LoRA navtest cache:

```bash
RUN_CACHE=1 \
NAVTEST_CHUNK_CACHE_ROOT=/path/to/navtest_full_geometry_chunks \
VLM_PATH=/path/to/base_vlm \
VLM_LORA_ADAPTER=$OUT_ROOT/last_vla_v2/round2/serverB_lora/vlm_lora_cot_alignment/adapters/vlm_lora_adapter_state.pt \
OUTPUT_CHUNK_ROOT=$OUT_ROOT/last_vla_v2/round2/serverB_lora/lora_navtest_cache \
scripts/last_vla_v2/round2/build_lora_navtest_cache.sh
```

Checkpoint sweep eval:

```bash
RUN_EVAL=1 \
OUT_ROOT=$OUT_ROOT/last_vla_v2/round2 \
NAVTEST_CHUNK_CACHE_ROOT=/path/to/navtest_full_geometry_chunks \
LORA_NAVTEST_CHUNK_CACHE_ROOT=$OUT_ROOT/last_vla_v2/round2/serverB_lora/lora_navtest_cache \
METRIC_CACHE_DIR=/path/to/navtest_metric_cache \
scripts/last_vla_v2/round2/eval_round2_checkpoint_sweep.sh
```

Corruption eval:

```bash
RUN_EVAL=1 \
OUT_ROOT=$OUT_ROOT/last_vla_v2/round2 \
CHECKPOINT=/path/to/best_progressive.ckpt \
NAVTEST_CHUNK_CACHE_ROOT=/path/to/navtest_full_geometry_chunks \
METRIC_CACHE_DIR=/path/to/navtest_metric_cache \
MAX_SAMPLES=1000 \
scripts/last_vla_v2/round2/eval_round2_cot_corruption.sh
```

Summarize:

```bash
python scripts/last_vla_v2/round2/summarize_round2.py \
  --out-root $OUT_ROOT/last_vla_v2/round2
```

## Remaining Risks

- Real PDM scoring depends on valid NAVSIM metric-cache coverage and must be verified on the target split before any SOTA claim.
- Full VGGT extraction depends on the installed VGGT API exposing depth, point map, and camera outputs; strict cache build fails if full outputs are unavailable.
- LoRA hidden-cache regeneration real path depends on the online backbone input schema and should be tiny-real-smoked before full cache generation.
- Line B navtest eval requires LoRA-regenerated navtest hidden cache.

## Statements

- No training launched.
- No full eval launched.
- Proxy scoring is smoke/debug only.
- PDM scoring is required for real oracle, corruption, and checkpoint sweep reports.
- A0-official-aligned PDMS `0.864891` remains the primary comparison baseline.
