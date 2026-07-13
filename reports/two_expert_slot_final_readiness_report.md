# Two-Expert Slot Final Readiness Report

- Base commit: `33d91ce`
- Branch: `feature/recogdrive-last-vla-v2`
- Baseline reference: A0 official-aligned ReCogDrive Stage2 `step_00100000`, full navtest PDMS `0.864891`

## Readiness Fixes

- Index path resolver fixed:
  - Added `scripts/last_vla_v2/two_expert_slot/two_expert_cache_utils.py`.
  - Replaced direct `Path(record["path"])` use in two-expert builders, preflight, and audits.
  - Relative index paths like `samples/a.pt`, absolute paths, and shard merge paths are covered by tests.
- Stage1 lazy teacher loading:
  - `run_two_expert_vlm_sft.py` now builds `sample_token -> path` indexes only.
  - Teacher tensors are loaded in `Dataset.__getitem__`, with optional LRU caching.
- VGGT Feature(23) extractor:
  - `build_vggt_feature23_cache.py` now supports real VGGT model extraction via `--vggt-model-path`.
  - The extractor hooks `aggregator.global_blocks.23`, captures features, and packs to `[12, D]`.
  - Added `smoke_vggt_feature23_extractor.py`.
  - Existing repack remains only for non-strict/dev fallback.
- JEPA strict policy:
  - JEPA model extraction is still not implemented.
  - Stage1 and JEPA launcher default to strict production teacher tokens.
  - Legacy JEPA downsample is explicitly `dev_fallback_not_for_final`.
- Real InternVL smoke:
  - `smoke_internvl_soft_slots.py` now accepts `VLM_PATH`, `IMAGE_PATH`, or `BASE_CACHE_ROOT`.
  - Added `run_real_internvl_soft_slot_smoke.sh`.
  - Launcher writes `reports/two_expert_slot/internvl_soft_slot_smoke.json`.
- Stage1 optimizer:
  - Separate parameter groups: `lr_vlm` and `lr_slots_adapters`.
  - LoRA is the formal default train mode.
  - Grad accumulation now flushes leftover gradients at epoch end.
- Hidden cache launcher passthrough:
  - `STAGE1_TRAIN_MODE` and `VLM_LORA_ADAPTER_DIR` are passed through.
  - `STAGE1_CHECKPOINT` is required for non-synthetic cache generation when `RUN_CACHE=1`.

## Validation

- `python -m py_compile` on updated two-expert scripts and audits: passed.
- `bash -n` on two-expert launchers: passed.
- `pytest -q tests/test_two_expert_*.py`: `34 passed`.
- `python scripts/last_vla_v2/two_expert_slot/smoke_stage1_two_expert_sft.py`: passed.
- `python scripts/last_vla_v2/two_expert_slot/smoke_stage2_two_expert_batch.py`: passed.
- `python scripts/last_vla_v2/two_expert_slot/run_two_expert_vlm_sft.py ...`: dry-run passed; no training launched.
- `python scripts/last_vla_v2/two_expert_slot/smoke_internvl_soft_slots.py`: dry-run guard passed.
- `python scripts/last_vla_v2/two_expert_slot/smoke_vggt_feature23_extractor.py`: dry-run guard passed.

## Not Run

- Real InternVL soft-slot smoke was not run because `VLM_PATH`, image path, and base cache sample env vars were not available.
- Real VGGT Feature(23) smoke was not run because `VGGT_MODEL_PATH` and image/base sample env vars were not available.
- No full Stage1/Stage2 training was launched.
- No production cache was generated.
- No full navtest eval was launched.

## Remaining Blockers

- Production JEPA model extraction remains external to this route; strict `jepa_dynamic_teacher_tokens [3,12,1024]` must already exist.
- Before full training, run real InternVL smoke with `RUN_SMOKE=1`.
- Before full training, run VGGT Feature(23) smoke with `RUN_SMOKE=1` and the production VGGT checkpoint.
