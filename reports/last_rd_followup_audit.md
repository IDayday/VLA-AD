# LaST-RD Follow-up Audit

Date: 2026-06-01

## Scope

This follow-up only changed code/config/scripts/tests/docs. No 8-GPU training was launched and no full eval was run.

## Changed Files

- `navsim/planning/script/run_training_recogdrive.py`
- `navsim/agents/recogdrive/latent_spatiotemporal_planning.py`
- `navsim/agents/recogdrive/recogdrive_diffusion_planner.py`
- `navsim/agents/recogdrive/recogdrive_agent.py`
- `navsim/planning/script/config/training/default_training.yaml`
- `configs/last_rd/last_rd_stage1_5.yaml`
- `configs/last_rd/last_rd_progressive_sft.yaml`
- `configs/last_rd/last_rd_performance_route.yaml`
- `configs/last_rd/last_rd_fair_route.yaml`
- `scripts/run_last_rd_stage1_5_8gpu.sh`
- `scripts/run_last_rd_progressive_sft_8gpu.sh`
- `scripts/run_last_rd_smoke_forward.py`
- `scripts/eval_recogdrive_last_rd_corruption_pdm.py`
- `scripts/testing/smoke_recogdrive_expert_planner.py`
- `tests/test_last_rd_shapes.py`
- `tests/test_last_rd_progressive_sft_loss.py`
- `docs/LaST_RD_Implementation.md`

## New Files

- `navsim/planning/script/config/experiment/__init__.py`
- `navsim/planning/script/config/experiment/last_rd_stage1_5.yaml`
- `navsim/planning/script/config/experiment/last_rd_progressive_sft.yaml`
- `scripts/audit_last_rd_cache_manifest.py`
- `scripts/check_last_rd_hydra_config.py`
- `scripts/run_last_rd_real_batch_smoke.py`
- `tests/test_last_rd_stage1_5_trainable_scope.py`
- `tests/test_last_rd_no_lazy_modules.py`
- `tests/test_last_rd_adapter_extract_load.py`
- `tests/test_last_rd_cache_manifest_synthetic.py`
- `reports/last_rd_hydra_config_check_stage1_5.json`
- `reports/last_rd_hydra_config_check_progressive_sft.json`
- `reports/last_rd_smoke_forward.json`
- `reports/last_rd_followup_audit.md`

## P0 Status

- high_command_one_hot shape fixed: `ChunkCacheDataset` now expects `(3,)`; `(4,)` fails fast with the NAVSIM left/straight/right explanation. `data_report.json` now includes `high_command_one_hot_shape_distribution`.
- Stage1.5 trainable scope fixed: launcher disables legacy A4 expert fusion/alignment, enables Last-RD, sets diffusion loss to 0, and trains only `action_head.last_rd.*`. Adapter extraction saves only Last-RD keys by default.
- Progressive SFT KD fixed: launcher and planner both fail fast when KD weight/mode are enabled without `A0_REFERENCE_CHECKPOINT` / `reference_a0_checkpoint`. KD can be disabled by setting `LAST_RD_POLICY_KD_WEIGHT=0`.
- Config reproducibility fixed: `configs/last_rd/*.yaml` no longer depend on `defaults`; Hydra experiment configs were added separately for official-aligned training.

## P1 Status

- VGGT geometry mode manifest added through `scripts/audit_last_rd_cache_manifest.py`; patch fallback is reported as `patch_fallback` and not treated as full geometry.
- Risk loss defaults changed to 0.0 in Stage1.5 and Progressive SFT configs/launchers. Supervised risk loss now requires labels or explicit override.
- `nn.LazyLinear` removed from `VGGTGeometryEncoder`; geometry tokens are matched to `vggt_dim` and projected with explicit LayerNorm + Linear.
- Corruption eval shuffle semantics are recorded in `rows.json` and `metrics.json`: `scene_batch_shuffle` for batch size > 1, `token_order_reverse` for per-sample eval.
- Added synthetic/adapter/trainable-scope/no-lazy/cache-manifest tests.

## Checks Run

- `python -m py_compile navsim/agents/recogdrive/latent_spatiotemporal_planning.py navsim/agents/recogdrive/recogdrive_diffusion_planner.py navsim/agents/recogdrive/recogdrive_agent.py navsim/planning/script/run_training_recogdrive.py scripts/audit_last_rd_cache_manifest.py scripts/check_last_rd_hydra_config.py scripts/run_last_rd_real_batch_smoke.py scripts/run_last_rd_smoke_forward.py scripts/eval_recogdrive_last_rd_corruption_pdm.py scripts/testing/smoke_recogdrive_expert_planner.py`
  - Status: passed
- `bash -n scripts/run_last_rd_stage1_5_8gpu.sh scripts/run_last_rd_progressive_sft_8gpu.sh`
  - Status: passed
- `pytest -q tests/test_last_rd_shapes.py tests/test_last_rd_no_future_leakage.py tests/test_last_rd_stage1_5_loss.py tests/test_last_rd_progressive_sft_loss.py tests/test_last_rd_stage1_5_trainable_scope.py tests/test_last_rd_no_lazy_modules.py tests/test_last_rd_adapter_extract_load.py tests/test_last_rd_cache_manifest_synthetic.py`
  - Status: passed, `9 passed, 1 skipped`
- `python scripts/check_last_rd_hydra_config.py --experiment last_rd_stage1_5 --output reports/last_rd_hydra_config_check_stage1_5.json`
  - Status: passed
- `python scripts/check_last_rd_hydra_config.py --experiment last_rd_progressive_sft --output reports/last_rd_hydra_config_check_progressive_sft.json`
  - Status: passed
- `python scripts/run_last_rd_smoke_forward.py`
  - Status: passed; wrote `reports/last_rd_smoke_forward.json`
- Direct `yaml.safe_load` check for `configs/last_rd/*.yaml`
  - Status: passed; all direct YAML files expose `use_last_rd` and `last_rd_stage`

## Skipped

- Real chunk cache audit and real-batch smoke were not run because `TRAIN_CHUNK_CACHE_ROOT` was not set.
- Full eval was not run.
- 8-GPU Stage1.5 / Progressive SFT training was not run.

## Remaining TODOs

- Run cache manifest on the real official-aligned chunk cache before training.
- Run real-batch smoke on the real chunk cache before training.
- Full scene-level corruption shuffle for per-sample eval still needs a `--permutation-file` implementation; current per-sample shuffle intentionally records `token_order_reverse`.
- Full VGGT geometry quality depends on the cache/extractor producing `full_geometry`; patch fallback must remain labeled as `patch_fallback`.
- Do not claim risk/safety supervision unless risk labels or PDM proxy labels are present.

## Exact Smoke Commands

```bash
python scripts/check_last_rd_hydra_config.py --experiment last_rd_stage1_5 --output reports/last_rd_hydra_config_check_stage1_5.json
python scripts/check_last_rd_hydra_config.py --experiment last_rd_progressive_sft --output reports/last_rd_hydra_config_check_progressive_sft.json
python scripts/run_last_rd_smoke_forward.py
```

With a real cache:

```bash
python scripts/audit_last_rd_cache_manifest.py \
  --cache-root "${TRAIN_CHUNK_CACHE_ROOT}" \
  --max-samples 128 \
  --output reports/last_rd_cache_manifest_smoke.json

python scripts/run_last_rd_real_batch_smoke.py \
  --cache-root "${TRAIN_CHUNK_CACHE_ROOT}" \
  --max-samples 2 \
  --output reports/last_rd_real_batch_smoke.json
```

## Future Stage1.5 Command

Do not run automatically:

```bash
BASE_CONFIG=last_rd_stage1_5 \
TRAIN_CHUNK_CACHE_ROOT=/path/to/official_aligned_train_chunks \
TRAIN_CHUNK_NAME_PATTERN='navtrain_chunk_*' \
OUTPUT_DIR=/path/to/last_rd_stage1_5_output \
MASTER_PORT=29531 \
bash scripts/run_last_rd_stage1_5_8gpu.sh
```

## Future Progressive SFT Command

Do not run automatically:

```bash
BASE_CONFIG=last_rd_progressive_sft \
STAGE1_5_CHECKPOINT=/path/to/last_rd_stage1_5_output/last_rd_adapter.pt \
A0_REFERENCE_CHECKPOINT=/path/to/A0-official-aligned/step_00100000 \
TRAIN_CHUNK_CACHE_ROOT=/path/to/official_aligned_train_chunks \
OUTPUT_DIR=/path/to/last_rd_progressive_sft_output \
MASTER_PORT=29532 \
bash scripts/run_last_rd_progressive_sft_8gpu.sh
```
