# Stage 5 Round 1 Small-Scale Execution

Stage 5 produces the first trustworthy small-scale Round 1 evidence for RISK-VLA on real cached data where available. The goal is diagnostic evidence, not final performance claims.

Use `低分风险场景 / low-score risk scenarios / critical-risk subset`. BiT is a path/terminal intent strategy in the RISK-VLA strategy bank, not the core method. The core method remains:

```text
risk state -> strategy routing/modulation -> diffusion planning
```

## Non-Goals

- No full-scale training by default.
- No full NAVSIM PDM evaluation by default.
- No final PDM metric claims from small pilots.
- No GRPO.
- No hard negative mining.
- No online JEPA/VGGT integration.
- No navtest/test labels for training.

## Required Inputs

The Stage 5 scripts resolve inputs from explicit arguments first, then from environment variables:

```bash
BIT_WORK_ROOT=/mnt/project/bit_drive_left_tail
BIT_EXP_ROOT=/mnt/project/bit_drive_left_tail/experiments/risk_vla
BIT_CACHE_ROOT=/mnt/project/bit_drive_left_tail/cache
SHARED_CHUNK_CACHE_ROOT=/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1
CHECKPOINT_ROOT=/mnt/project/VLA-AD/checkpoints
```

Required Round 1 inputs:

- A0/Base PDM CSV.
- B3/BiT PDM CSV.
- Train/val chunk cache or labeled overlay source.
- Base IL or pilot checkpoint directory.

If required inputs are missing, use `discover_round1_inputs.py` to write an input-gap report. Do not guess private paths.

## Execution Order

```text
input discovery
  -> matched A0/B3 PDM table
  -> train/val risk labels
  -> leakage guard
  -> labeled chunk-cache overlay
  -> R0 risk-head diagnostic short run
  -> risk prediction export
  -> risk diagnostic aggregation
  -> R1 oracle-router pilot command/result
  -> R2 predicted-router pilot command/result
  -> Round 1 evidence report
```

## Success / Failure Gates

R0 risk-head diagnostic is useful only if:

- risk labels are train/val only,
- prediction export evaluates at least 100 labeled tokens,
- path/DAC enrichment@100 is at least 1.5,
- low-score enrichment@100 is at least 1.2,
- interaction NC or TTC enrichment@100 is at least 1.2.

If these fail, do not interpret R2 predicted routing as a method result. Improve labels, calibration, or export coverage first. R1 oracle-router can still proceed as analysis-only.

## Interpreting R0/R1/R2

- `R0_risk_head_diagnostic`: tests whether the VLA latent state predicts risk types. Strategy token and horizon residual scales are zero.
- `R1_oracle_router_pilot`: analysis-only upper bound for strategy routing from labels. Do not use test labels for training claims.
- `R2_predicted_router_pilot`: predicted-risk router pilot. Run only after R0 diagnostic evidence is strong enough.

Risk diagnostic, oracle-router, and predicted-router outputs must be reported separately.
