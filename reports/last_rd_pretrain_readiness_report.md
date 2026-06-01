# LaST-RD Pretrain Readiness Report

Date: 2026-06-01

## Summary

Status: READY for Stage1.5 adapter pretraining only.

The code/config readiness gate passed, and the real project cache passed a 128-sample strict manifest plus 2-sample real-batch smoke. Progressive SFT is not ready to launch until the Stage1.5 adapter checkpoint exists.

Resolved training cache root:

```text
/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1
```

The cache stores legacy raw NAVSIM `high_command_one_hot` as `[4]`. This is compatible after the explicit loader repair added in this pass: the loader verifies the fourth slot is zero, returns `[3]` left/straight/right to LaST-RD, and leaves the 8D `status_feature` unchanged for A0/Stage2 compatibility.

No Stage1.5 training, Progressive SFT training, or full navtest eval was launched.

## Baseline

- A0-official-aligned best checkpoint: `step_00100000`
- A0-official-aligned full navtest PDMS: `0.864891`
- A0-local-fixed is not the standard comparison target.

LaST-RD must exceed `0.864891` full navtest PDMS to count as a main-metric success. Corruption eval must also show that the planner actually uses LaST-RD tokens.

## Code Checks

- `py_compile`: PASS
  - Included LaST-RD planner modules, agent, official training script, cache manifest, Hydra checker, eval config checker, real-batch smoke, synthetic smoke, corruption eval, adapter audit, and the shared expert PDM eval batch builder.
- `pytest`: PASS
  - `11 passed, 1 skipped`
  - Covered planner shapes, no-future-leakage, Stage1.5 loss, Progressive SFT loss, trainable scope, no lazy modules, adapter load, cache manifest synthetic, adapter audit synthetic, and official-agent no-future-leakage.
- Hydra config check: PASS
  - `last_rd_stage1_5`: PASS
  - `last_rd_progressive_sft`: PASS
- Eval config check: PASS
  - `last_rd_progressive_sft_hybrid_eval.yaml`: PASS
  - `last_rd_progressive_sft_lastrd_only_eval.yaml`: PASS
- Launcher syntax/dry-run: PASS
  - `bash -n` passed.
  - Stage1.5 dry-run writes the recommended manifest command to `commands.log`.
  - Progressive launcher fails fast when KD is enabled without `A0_REFERENCE_CHECKPOINT`.
  - Progressive launcher dry-runs successfully with `LAST_RD_POLICY_KD_WEIGHT=0`.

## Data Checks

- Cache manifest smoke: PASS
  - Cache root: `/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1`
  - Pattern: `train_*chunk_*`
  - Samples scanned: `128`
- Full manifest: NOT RUN
- JEPA context/target coverage: `1.0 / 1.0`
- VGGT context/target coverage: `1.0 / 1.0`
- Geometry mode distribution: `patch_fallback=128`
- raw `high_command_one_hot` shape distribution: `(4,)=128`
- normalized `high_command_one_hot` shape distribution: `(3,)=128`
- legacy command repair: `legacy_4d_repairable=128`, `legacy_4d_unrepairable=0`
- Risk label coverage: all risk label keys `0.0`; risk loss remains disabled.
- Duplicate sample tokens: `0`

The strict manifest gate now enforces:

- no shape/dtype/finiteness errors
- `high_command_one_hot` must normalize to `(3,)`; native `(3,)` and legacy repairable `(4,)` with zero fourth slot are accepted.
- required base key coverage `1.0`
- JEPA/VGGT context and target coverage at least `0.99`
- zero duplicate `sample_token`
- risk labels present when risk loss is enabled
- full VGGT geometry coverage at least `0.99` only when `require_vggt_geometry=true`

Patch fallback is not a blocker unless full geometry is required, but if full geometry coverage is below `0.5`, the manifest emits: `VGGT geometry is mostly patch_fallback; do not claim full geometry distillation.`

## Real-Batch Smoke

- Real-batch smoke: PASS
  - Cache root: `/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1`
  - Samples: `2`
- Stage1.5 loss finite: PASS
- Progressive loss finite: PASS
- `get_action` no target pass: PASS
- Stage1.5 trainable scope pass: PASS
- batch `high_command_one_hot` shape: `[2, 3]`
- batch `status_feature` shape: `[2, 8]`

The real-batch smoke script now reports:

- target keys present in the training batch
- target keys present in the `get_action` batch
- `use_expert_features`
- `use_last_rd`
- geometry mode distribution
- `high_command_one_hot` shape
- Stage1.5 trainable-scope pass/fail
- `get_action` no-future-target pass/fail

It exits nonzero if Stage1.5 trainable scope fails.

## Synthetic Smoke

- `python scripts/run_last_rd_smoke_forward.py`: PASS
- Stage1.5 synthetic loss: finite
- Progressive synthetic loss: finite
- `get_action` output shape: `[1, 8, 3]`
- target tokens passed to `get_action`: false

## Eval-Safe Configs

- `configs/last_rd/last_rd_progressive_sft_hybrid_eval.yaml`
  - `use_expert_features: true`
  - `use_last_rd: true`
  - `policy_kd_loss_weight: 0.0`
  - `policy_kd_mode: none`
  - `allow_expert_target_features: false`
  - `allow_future_targets_in_inference: false`
- `configs/last_rd/last_rd_progressive_sft_lastrd_only_eval.yaml`
  - `use_expert_features: false`
  - `use_last_rd: true`
  - `policy_kd_loss_weight: 0.0`
  - `policy_kd_mode: none`
  - `allow_expert_target_features: false`
  - `allow_future_targets_in_inference: false`

## Remaining Risks

- Full cache manifest was not run; a 128-sample strict smoke manifest passed. Run full manifest if you want a complete preflight audit before consuming the whole cache.
- VGGT geometry is `patch_fallback` in the smoke manifest; do not describe patch fallback as full VGGT geometry.
- Risk loss remains default-off because risk label coverage is unknown. Do not claim safety/risk supervision without labels or proxy labels.
- Full scene-level corruption shuffle still needs a permutation-file implementation. Current per-sample shuffle semantics are recorded as `token_order_reverse`.
- One unit test branch was skipped because it only runs when a tiny real cache fixture exists.

## Commands Run

```bash
python -m py_compile \
  navsim/agents/recogdrive/latent_spatiotemporal_planning.py \
  navsim/agents/recogdrive/recogdrive_diffusion_planner.py \
  navsim/agents/recogdrive/recogdrive_agent.py \
  navsim/planning/script/run_training_recogdrive.py \
  scripts/audit_last_rd_cache_manifest.py \
  scripts/check_last_rd_hydra_config.py \
  scripts/check_last_rd_eval_configs.py \
  scripts/run_last_rd_real_batch_smoke.py \
  scripts/run_last_rd_smoke_forward.py \
  scripts/eval_recogdrive_last_rd_corruption_pdm.py \
  scripts/audit_last_rd_adapter_checkpoint.py \
  scripts/eval_recogdrive_expert_pdm.py

bash -n \
  scripts/run_last_rd_stage1_5_8gpu.sh \
  scripts/run_last_rd_progressive_sft_8gpu.sh \
  scripts/check_last_rd_launchers.sh

bash scripts/check_last_rd_launchers.sh

pytest -q \
  tests/test_last_rd_shapes.py \
  tests/test_last_rd_no_future_leakage.py \
  tests/test_last_rd_stage1_5_loss.py \
  tests/test_last_rd_progressive_sft_loss.py \
  tests/test_last_rd_stage1_5_trainable_scope.py \
  tests/test_last_rd_no_lazy_modules.py \
  tests/test_last_rd_adapter_extract_load.py \
  tests/test_last_rd_cache_manifest_synthetic.py \
  tests/test_last_rd_adapter_audit_synthetic.py \
  tests/test_last_rd_agent_no_future_leakage.py

python scripts/check_last_rd_hydra_config.py \
  --experiment last_rd_stage1_5 \
  --output reports/last_rd_hydra_config_check_stage1_5.json

python scripts/check_last_rd_hydra_config.py \
  --experiment last_rd_progressive_sft \
  --output reports/last_rd_hydra_config_check_progressive_sft.json

python scripts/check_last_rd_eval_configs.py \
  --output reports/last_rd_eval_config_check.json

python scripts/run_last_rd_smoke_forward.py

python scripts/audit_last_rd_cache_manifest.py \
  --cache-root /mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1 \
  --chunk-name-pattern 'train_*chunk_*' \
  --max-samples 128 \
  --future-jepa-loss-weight 0.30 \
  --risk-loss-weight 0.0 \
  --strict \
  --output reports/last_rd_cache_manifest_smoke.json

python scripts/run_last_rd_real_batch_smoke.py \
  --cache-root /mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1 \
  --max-samples 2 \
  --stage both \
  --output reports/last_rd_real_batch_smoke.json
```

## Required Preflight With Real Cache

Run before Stage1.5:

```bash
python scripts/audit_last_rd_cache_manifest.py \
  --cache-root "${TRAIN_CHUNK_CACHE_ROOT}" \
  --max-samples 128 \
  --future-jepa-loss-weight 0.30 \
  --risk-loss-weight 0.0 \
  --strict \
  --output reports/last_rd_cache_manifest_smoke.json

python scripts/run_last_rd_real_batch_smoke.py \
  --cache-root "${TRAIN_CHUNK_CACHE_ROOT}" \
  --max-samples 2 \
  --stage both \
  --output reports/last_rd_real_batch_smoke.json
```

If the smoke manifest passes, run the full manifest:

```bash
python scripts/audit_last_rd_cache_manifest.py \
  --cache-root "${TRAIN_CHUNK_CACHE_ROOT}" \
  --future-jepa-loss-weight 0.30 \
  --risk-loss-weight 0.0 \
  --strict \
  --output reports/last_rd_cache_manifest_full.json
```

## Stage1.5 Training Command

DO NOT RUN AUTOMATICALLY:

```bash
BASE_CONFIG=last_rd_stage1_5 \
TRAIN_CHUNK_CACHE_ROOT=/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1 \
TRAIN_CHUNK_NAME_PATTERN='train_*chunk_*' \
TRAIN_TEST_SPLIT=navtrain \
OUTPUT_DIR=/path/to/outputs/last_rd_stage1_5 \
MASTER_PORT=29531 \
bash scripts/run_last_rd_stage1_5_8gpu.sh
```

## Progressive SFT Training Command

DO NOT RUN AUTOMATICALLY:

```bash
BASE_CONFIG=last_rd_progressive_sft \
STAGE1_5_CHECKPOINT=/path/to/outputs/last_rd_stage1_5/last_rd_adapter.pt \
A0_INIT_CHECKPOINT=/path/to/a0_official_aligned/step_00100000.ckpt \
A0_REFERENCE_CHECKPOINT=/path/to/a0_official_aligned/step_00100000.ckpt \
TRAIN_CHUNK_CACHE_ROOT=/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1 \
TRAIN_TEST_SPLIT=navtrain \
OUTPUT_DIR=/path/to/outputs/last_rd_progressive_sft \
MASTER_PORT=29532 \
LAST_RD_POLICY_KD_WEIGHT=0.05 \
LAST_RD_POLICY_KD_MODE=noise \
LAST_RD_RISK_LOSS_WEIGHT=0.0 \
bash scripts/run_last_rd_progressive_sft_8gpu.sh
```
