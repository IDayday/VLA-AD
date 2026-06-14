# Two-Expert Coverage and Stage3 Isolation Report

- Base remote HEAD: `930f30a`
- Branch: `feature/recogdrive-last-vla-v2`
- Baseline reference: A0 official-aligned ReCogDrive Stage2 `step_00100000`, full navtest PDMS `0.864891`

## Coverage Preflight

- Added `scripts/last_vla_v2/two_expert_slot/preflight_two_expert_coverage.py`.
- The preflight uses the shared two-expert index resolver from `two_expert_cache_utils.py`.
- It reports:
  - `base_count`
  - `jepa_count`
  - `vggt_count`
  - `intersection_count`
  - `jepa_base_coverage`
  - `vggt_base_coverage`
  - `all_teacher_coverage`
- It fails when `all_teacher_coverage < min_train_coverage`.

## Readiness Gate

- `final_two_expert_readiness_gate.py` now accepts `--coverage-json` and `--min-coverage`.
- Gate output records `coverage_ok` and `all_teacher_coverage`.
- Stage1 and Stage2 full-training launchers require `coverage_ok=true` in `READINESS_GATE_JSON` unless explicitly overridden with `ALLOW_LOW_COVERAGE=1`.
- This prevents full Stage1 from silently training on a small teacher/base intersection.

## Stage3 Isolation

- Stage3 AWAC/IQL run script now defaults to dry-run.
- Stage3 AWAC/IQL stable launcher now writes `jobs.tsv` and exits unless `RUN_STAGE3=1` or `RUN_TRAIN=1`.
- Stage3 AWAC elite-buffer builder exits before loading model/data unless `RUN_STAGE3=1` or `RUN_TRAIN=1`.
- No two-expert launcher calls Stage3 AWAC/IQL or `run_training_recogdrive_rl.py`.
- Added `docs/Route_Status_And_Pruning_Plan.md` to clarify:
  - active route is `two_expert_slot` Stage1/Stage2,
  - Stage3 AWAC/IQL is downstream experimental and disabled by default,
  - Stage3 results must not be mixed into two-expert Stage1/Stage2 interpretation.

## Validation

- `python -m py_compile scripts/last_vla_v2/two_expert_slot/preflight_two_expert_coverage.py scripts/last_vla_v2/two_expert_slot/final_two_expert_readiness_gate.py`: passed.
- `python -m py_compile scripts/training/build_recogdrive_stage3_awac_elite_buffer.py`: passed.
- `bash -n` on Stage3 AWAC/IQL and two-expert launchers: passed.
- `pytest -q tests/test_two_expert_*.py tests/test_no_future_leakage.py`: `47 passed`.
- Stage3 AWAC/IQL dry-run guard was checked; no training command was executed.

## Not Run

- No full training launched.
- No production cache generated.
- No full navtest eval launched.
