# Last-VLA v2 No-Residual Diffusion Review Fix

## Summary

- Fixed the `sample_chain()` CoT conditioning bug in `navsim/agents/recogdrive/recogdrive_diffusion_planner.py`.
- `sample_chain()` now initializes `cot_condition_tokens` from the initial DIT context.
- The flow branch now refreshes `cot_condition_tokens` after each `_prepare_dit_context()` call.
- `get_logprobs()` now passes `cot_condition_tokens` through `p_mean_variance()` so GRPO chain log-prob computation uses the same decoupled CoT condition as chain generation.

## No-Residual Diffusion Semantics

- Last-VLA diffusion target remains GT / `selected_target_norm`.
- `_last_vla_diffusion_target(selected_target_norm, training=...)` returns `(selected_target_norm, 0.0)`.
- Final `pred_traj` remains `denorm(current_actions)`.
- `pred_traj` is not reconstructed as `coarse_traj + residual`.
- `pred_coarse_traj` remains a diagnostic / auxiliary CoT output only.
- `pred_residual_norm` is not emitted.

## Tests

Commands run:

```bash
/root/miniconda3/envs/navsim/bin/python -m py_compile \
  navsim/agents/recogdrive/recogdrive_diffusion_planner.py

/root/miniconda3/envs/navsim/bin/python -m pytest -q \
  tests/test_last_vla_diffusion_target_no_residual.py \
  tests/test_last_vla_sample_chain_cot_condition.py

/root/miniconda3/envs/navsim/bin/python -m pytest -q \
  tests/test_last_vla_diffusion_target_no_residual.py \
  tests/test_last_vla_sample_chain_cot_condition.py \
  tests/test_last_vla_*.py \
  tests/test_last_rd_*.py \
  tests/test_expert_*.py \
  tests/test_no_future_leakage.py
```

Results:

- Focused tests: `3 passed`.
- Requested pytest set: `114 passed, 3 skipped`.

## Execution Scope

- No training launched.
- No `torchrun` launched.
- No full eval launched.

## Baseline

- Baseline A0-official-aligned PDMS: `0.864891`.
