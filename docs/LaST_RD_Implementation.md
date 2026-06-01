# LaST-RD Implementation

## Motivation

A0 official-aligned is the standard ReCogDrive Stage2 baseline for later comparisons. The recorded full navtest best PDMS baseline is 0.864891, and A0-local-fixed is not treated as the comparison target.

A4-V2 mainly concatenates JEPA/VGGT tokens into the DiT context. That leaves several gaps: no predicted future dynamic latent at inference, no explicit geometry token path beyond pooled VGGT patches, static horizon residuals, global scalar expert gates, and no coarse trajectory/risk token supervision. LaST-RD adds a trainable Stage1.5 adapter and progressive SFT path without retraining the frozen Stage1/VLM.

## Architecture

`navsim/agents/recogdrive/latent_spatiotemporal_planning.py` contains the LaST-RD modules:

- `FutureJEPAPredictor`: learns dynamic query tokens that cross-attend to VLM tokens, JEPA context tokens, ego state, command, and history. It predicts future JEPA teacher tokens for train-only alignment and emits planner-dimension dynamic tokens for inference.
- `VGGTGeometryEncoder`: builds geometry tokens from explicit VGGT geometry/depth/pointmap/camera tokens when available. If full geometry is not required, old patch-token caches run through `patch_fallback` mode and log the mode explicitly.
- `CoarseTrajectoryHead`: predicts a normalized coarse trajectory in the same action space as `ReCogDriveDiffusionPlanner.norm_odo` and encodes it into ego trajectory tokens.
- `RiskTokenHead`: emits risk tokens and optional horizon risk logits for generic, drivable/offroad, TTC/collision, and comfort risks. Missing labels skip risk loss unless required.
- `SceneAwareExpertGate` and `TimestepAwareExpertGate`: produce group weights for `[vlm, dynamic, geometry, ego, risk]`. Training denoising and sampling can pass the current noisy trajectory and diffusion timestep into the timestep-aware gate.

`LatentSpatioTemporalReasoner` orchestrates the modules, combines generated planning tokens, returns horizon conditioning for the DiT input, and exposes auxiliary losses and diagnostics.

## Training

Stage1.5 adapter pretraining uses `last_rd_stage=stage1_5` and `diffusion_loss_weight=0.0`. It computes future JEPA, VGGT geometry/token, coarse trajectory, heading, and risk auxiliary losses without requiring the diffusion denoising path.

Progressive SFT uses `last_rd_stage=progressive_sft` with the diffusion loss enabled. Auxiliary weights decay toward configured floors through `set_training_progress(epoch, total_epochs)`, while diffusion trajectory generation remains the main objective.

The performance route can initialize from the A0-official-aligned best checkpoint and optionally use a frozen A0 reference for noise/x0 policy KD. The fair route keeps the same data and schedule as A0 without A0 policy initialization.

## No Future Leakage

Future teacher targets are train-only. `jepa_target_tokens`, `vggt_target_tokens`, and `vggt_geometry_target_tokens` are read only for training or validation loss mode. `get_action`, `sample_chain`, and eval paths ignore target keys and warn if they are present.

Inference uses only context-side VLM, JEPA, VGGT, ego, command, and history signals plus LaST-RD predicted latent tokens. It never conditions on future JEPA/VGGT targets.

## Cache Requirements

Required base fields:

- `history_trajectory`
- `high_command_one_hot`
- `last_hidden_state`
- `status_feature`
- `trajectory`

Required LaST-RD context fields when enabled:

- `jepa_context_tokens`
- `vggt_context_tokens` when geometry tokens or A4 expert context are enabled

Train/validation supervision may include:

- `jepa_target_tokens`
- `vggt_target_tokens`
- `vggt_geometry_tokens`
- `vggt_geometry_target_tokens`
- `vggt_depth_tokens`
- `vggt_pointmap_tokens`
- `risk_labels`

Old caches remain usable with `allow_patch_geometry_fallback=true`; diagnostics record `patch_fallback`. Set `require_vggt_geometry=true` to fail fast when full geometry tokens are absent.

## Commands

These commands are for later use and are not run automatically:

```bash
python scripts/run_last_rd_smoke_forward.py
```

```bash
BASE_CONFIG=recogdrive_agent_a4_v2 \
TRAIN_CHUNK_CACHE_ROOT=/path/to/train_chunks \
TRAIN_CHUNK_NAME_PATTERN='navtrain_chunk_*' \
OUTPUT_DIR=/path/to/last_rd_stage1_5 \
MASTER_PORT=29571 \
bash scripts/run_last_rd_stage1_5_8gpu.sh
```

```bash
STAGE1_5_CHECKPOINT=/path/to/last_rd_stage1_5/last_rd_adapter.pt \
TRAIN_CHUNK_CACHE_ROOT=/path/to/train_chunks \
OUTPUT_DIR=/path/to/last_rd_progressive_sft \
MASTER_PORT=29572 \
A0_INIT_CHECKPOINT=/path/to/a0_official_best.ckpt \
A0_REFERENCE_CHECKPOINT=/path/to/a0_official_best.ckpt \
bash scripts/run_last_rd_progressive_sft_8gpu.sh
```

## Tests And Acceptance

Target checks:

```bash
python -m py_compile \
  navsim/agents/recogdrive/latent_spatiotemporal_planning.py \
  navsim/agents/recogdrive/recogdrive_diffusion_planner.py \
  navsim/agents/recogdrive/recogdrive_agent.py \
  scripts/run_last_rd_smoke_forward.py \
  scripts/eval_recogdrive_last_rd_corruption_pdm.py
```

```bash
pytest -q tests/test_last_rd_shapes.py \
          tests/test_last_rd_no_future_leakage.py \
          tests/test_last_rd_stage1_5_loss.py \
          tests/test_last_rd_progressive_sft_loss.py
```

## Difference From A4-V2

A4-V2 adds JEPA/VGGT context tokens and static horizon residuals. LaST-RD predicts future dynamic JEPA latents for inference, encodes VGGT geometry with explicit fallback diagnostics, learns normalized coarse trajectory and risk tokens, and routes expert groups with scene/timestep-aware gates.

## Difference From Full Stage1 Pretraining

This implementation does not retrain the VLM/Stage1. The Stage1/VLM remains frozen unless a later LoRA path is added. LaST-RD learns a Stage1.5 latent adapter from cached JEPA/VGGT teachers and then integrates it into Stage2 diffusion SFT.
