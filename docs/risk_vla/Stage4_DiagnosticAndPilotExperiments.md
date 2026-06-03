# Stage 4 Diagnostic And Pilot Experiments

Stage 4 moves RISK-VLA from synthetic forward checks to controlled, reproducible diagnostics on real cached data. The target questions are:

1. Can the VLA latent state predict risk types for low-score risk scenarios / critical-risk subsets using train/val labels?
2. Do oracle and predicted risk routing activate the intended strategy modules without breaking the baseline diffusion-planning path?

This is not final PDM optimization. Do not run long training jobs, full NAVSIM PDM evaluation, GRPO, hard negative mining, or online JEPA/VGGT integration in this stage.

Use the terminology `低分风险场景 / low-score risk scenarios / critical-risk subset`. Avoid framing the work as “left-tail failure.”

BiT is only a path/terminal intent strategy inside the RISK-VLA strategy bank. The core method remains:

```text
risk state -> strategy routing/modulation -> diffusion planning
```

Oracle-router is analysis-only. It must not be enabled in standard training configs. Navtest/test labels must never be used for training.

## Data Flow

```text
PDM CSVs -> matched token table -> risk labels -> labeled chunk-cache overlay
       -> risk-head diagnostic train/eval
       -> risk prediction aggregation
       -> oracle-router pilot
       -> predicted-router pilot
       -> round report
```

## Default Paths

The runners resolve paths from environment variables and write new outputs under isolated experiment roots when available:

```bash
BIT_WORK_ROOT=/mnt/project/bit_drive_left_tail
BIT_EXP_ROOT=/mnt/project/bit_drive_left_tail/experiments/risk_vla
BIT_CACHE_ROOT=/mnt/project/bit_drive_left_tail/cache
SHARED_CHUNK_CACHE_ROOT=/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1
CHECKPOINT_ROOT=/mnt/project/VLA-AD/checkpoints
```

Shared multi-TB caches are read-only inputs. Use `create_labeled_chunk_cache_overlay.py` to create a symlink or copied overlay before adding labels.

## Round 1 Modes

- `A0_base`: baseline ReCogDrive/BiT branch behavior without RISK-VLA.
- `B3_direct_bit`: direct BiT path/terminal intent output formulation as a strategy baseline.
- `R0_risk_head_diagnostic`: RISK-VLA risk head only; strategy token and horizon residual scales are zero.
- `R1_oracle_router_pilot`: analysis-only upper bound with labels used for routing.
- `R2_predicted_router_pilot`: predicted-risk routing with small conditioning scales.

Risk diagnostic, oracle-router, and predicted-router results must be reported separately.

## Safe Execution

Round scripts in `scripts/risk_vla/round1/` default to `DRY_RUN=1` and do not execute expensive work unless `EXECUTE=1` is set. Python command builders also default to dry-run and require explicit `--execute`.

Use `--max-samples`, `MAX_SAMPLES`, or equivalent overrides for pilot commands. Do not use navtest/test labels for train-time risk supervision.
