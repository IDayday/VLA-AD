# Two-Expert Slot Readiness Fix Report

- Current HEAD before this fix: `33069a0`
- Branch: `feature/recogdrive-last-vla-v2`
- Baseline reference: A0 official-aligned ReCogDrive Stage2 `step_00100000`, full navtest PDMS `0.864891`

## Fix Summary

- Added a dedicated Stage1 entrypoint: `scripts/last_vla_v2/two_expert_slot/run_two_expert_vlm_sft.py`.
- Updated the Stage1 launcher to call the dedicated entrypoint instead of the generic ReCogDrive trainer.
- Made Stage1 default train mode `top_layers`; LoRA is no longer a placeholder default.
- Required `vggt_feature_dim` to be resolved from teacher metadata or passed explicitly; it is no longer hardcoded to `512`.
- Added synthetic Stage1 and Stage2 smoke scripts.
- Added a real InternVL soft-slot compatibility smoke script with `RUN_SMOKE=1` guard.
- Added teacher-cache preflight for strict production-vs-fallback status.
- Tightened JEPA/VGGT teacher cache builders so `--jepa-model-path` / `--vggt-model-path` fail fast because production model extraction is not implemented in these repack builders.
- Extended hidden cache Stage1 loading metadata for LoRA/top-layer state contract visibility.

## Status

- Planner structure: `use_two_expert_slots` is a separate route and remains mutually exclusive with Last-VLA, Last-RD, and A4 direct expert features.
- Stage2 target: GT normalized trajectory; residual diffusion and coarse+residual output remain disabled.
- Stage1 module: `TwoExpertVLMSFTModule` is instantiated by the new entrypoint.
- VGGT Feature(23) dimension: resolved from cache metadata/sample tensors or explicit `--vggt-feature-dim`.
- Teacher builders: current scripts are explicit existing-feature repack/validation tools, not production JEPA/VGGT extractors.
- Real InternVL smoke: script exists, but actual model load was not run in this environment because `RUN_SMOKE=1` and `VLM_PATH` were not provided.

## Validation

- `python -m py_compile` on core two-expert route files and new scripts: passed.
- `bash -n` on two-expert launchers: passed.
- `python scripts/last_vla_v2/two_expert_slot/smoke_stage1_two_expert_sft.py`: passed.
- `python scripts/last_vla_v2/two_expert_slot/smoke_stage2_two_expert_batch.py`: passed.
- `python scripts/last_vla_v2/two_expert_slot/smoke_internvl_soft_slots.py`: dry-run guard passed; real smoke not run.
- `python scripts/last_vla_v2/two_expert_slot/run_two_expert_vlm_sft.py ...`: dry-run guard passed; no training launched.
- `pytest -q tests/test_two_expert_*.py tests/test_last_vla_*.py tests/test_last_rd_*.py tests/test_expert_*.py tests/test_no_future_leakage.py`: `146 passed, 4 skipped`.

## Remaining Blockers

- Production JEPA dynamic extraction from model is still not implemented in the cache builder.
- Production VGGT Feature(23) extraction via model hook is still not implemented in the cache builder.
- Real InternVL soft-slot smoke must be run with an actual `VLM_PATH` and `RUN_SMOKE=1` before claiming production readiness.
- No Stage1/Stage2 training was launched.
- No production cache was generated.
- No full navtest eval was launched.
