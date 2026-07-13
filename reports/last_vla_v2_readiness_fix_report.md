# Last-VLA v2 Readiness Fix Report

## Baseline

A0-official-aligned best full navtest PDMS = `0.864891`.

## Blocker Fix Status

- Real PDM scoring: fixed. `--score-mode=pdm` now uses NAVSIM metric-cache PDM through a shared `TrajectoryScorer`; missing metric cache fails fast. Proxy scoring is only active under `--score-mode=proxy`.
- Corruption PDM eval: fixed. `eval_last_vla_cot_corruption_pdm.py` now reports PDMS/submetrics in PDM mode and only reports proxy metrics when explicitly allowed.
- Geometry mode strictness: fixed. `vggt_geometry_mode(_code)` is explicit and preserved; `patch_fallback` can no longer be treated as `full_geometry`.
- Hydra launcher override: fixed. Launchers now use `agent.last_vla_adapter_checkpoint=...`, with a dry-run checker asserting the old `++agent...` override is absent.

## Changed Files

- `navsim/agents/recogdrive/expert_backends.py`
- `navsim/agents/recogdrive/expert_extractors/vggt_extractor.py`
- `navsim/agents/recogdrive/last_vla_cot_planning.py`
- `navsim/agents/recogdrive/recogdrive_diffusion_planner.py`
- `navsim/agents/recogdrive/recogdrive_features.py`
- `navsim/planning/script/run_training_recogdrive.py`
- `scripts/audit_last_vla_cache_manifest.py`
- `scripts/build_recogdrive_chunk_cache.py`
- `scripts/eval_last_vla_best_of_k_oracle.py`
- `scripts/eval_last_vla_cot_corruption_pdm.py`
- `scripts/eval_recogdrive_expert_pdm.py`
- `scripts/generate_last_vla_teacher_trajectory_cache.py`
- `scripts/merge_last_vla_teacher_cache_into_chunks.py`
- `scripts/last_vla_v2/run_best_of_k_oracle.sh`
- `scripts/last_vla_v2/generate_teacher_traj_cache.sh`
- `scripts/last_vla_v2/run_progressive_bottleneck_8gpu.sh`
- `scripts/last_vla_v2/run_teacher_traj_sft_8gpu.sh`
- `docs/LaST_VLA_ReCogDrive_v2_Design.md`

## New Files

- `scripts/last_vla_v2/pdm_scoring_utils.py`
- `scripts/last_vla_v2/check_launchers_dryrun.sh`
- `reports/last_vla_v2_oracle_synthetic_smoke/metrics.json`
- `reports/last_vla_v2_oracle_synthetic_smoke/rows.json`
- `tests/test_last_vla_corruption_hook_effect.py`
- `tests/test_last_vla_corruption_metrics_schema.py`
- `tests/test_last_vla_geometry_mode_strict.py`
- `tests/test_last_vla_oracle_proxy_schema.py`
- `tests/test_last_vla_pdm_scorer_mock.py`
- `tests/test_last_vla_pdm_scorer_proxy.py`
- `tests/test_last_vla_teacher_cache_proxy_schema.py`

## Tests Run

- Targeted readiness tests: pass, `14 passed`.
- `python -m py_compile ...`: pass.
- `bash -n scripts/last_vla_v2/run_best_of_k_oracle.sh scripts/last_vla_v2/generate_teacher_traj_cache.sh scripts/last_vla_v2/run_progressive_bottleneck_8gpu.sh scripts/last_vla_v2/run_teacher_traj_sft_8gpu.sh scripts/last_vla_v2/check_launchers_dryrun.sh`: pass.
- `scripts/last_vla_v2/check_launchers_dryrun.sh`: pass; no training launched.
- `pytest -q tests/test_last_vla_*.py tests/test_last_rd_*.py tests/test_expert_*.py tests/test_no_future_leakage.py`: pass, `43 passed, 2 skipped`.
- `python scripts/eval_last_vla_best_of_k_oracle.py --synthetic-smoke --num-candidates 4 --max-samples 4 --output-dir reports/last_vla_v2_oracle_synthetic_smoke`: pass.
- Proxy teacher cache schema smoke is covered by `tests/test_last_vla_teacher_cache_proxy_schema.py`.

## Remaining Risks

- Real PDM scoring requires valid NAVSIM metric cache coverage for the evaluated split.
- Proxy mode remains available only for smoke/debug and must not be used for SOTA claims.
- Production teacher trajectory SFT should wait for positive best-of-K oracle PDM gain, at least 99% teacher trajectory cache coverage, and audited geometry mode coverage.

## Commands

Best-of-K oracle:

```bash
python scripts/eval_last_vla_best_of_k_oracle.py \
  --config configs/last_vla_v2/last_vla_progressive_bottleneck_eval.yaml \
  --checkpoint /path/to/checkpoint.ckpt \
  --chunk-cache-root /path/to/navtrain_chunks \
  --metric-cache-dir /path/to/metric_cache \
  --score-mode pdm \
  --num-candidates 8 \
  --output-dir $OUT_ROOT/last_vla_v2/oracle
```

Teacher trajectory cache generation:

```bash
python scripts/generate_last_vla_teacher_trajectory_cache.py \
  --config configs/last_vla_v2/last_vla_progressive_bottleneck_eval.yaml \
  --checkpoint /path/to/checkpoint.ckpt \
  --chunk-cache-root /path/to/navtrain_chunks \
  --metric-cache-dir /path/to/metric_cache \
  --output-cache-root $OUT_ROOT/last_vla_v2/teacher_cache \
  --score-mode pdm \
  --num-candidates 8
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
  --metric-cache-dir /path/to/metric_cache \
  --score-mode pdm \
  --output-dir $OUT_ROOT/last_vla_v2/corruption/zero_all \
  --zero-all-cot
```

## Statements

- No training launched by this readiness fix.
- No full eval launched by this readiness fix.
- Do not claim performance before proper PDM eval against the A0-official baseline `0.864891`.
