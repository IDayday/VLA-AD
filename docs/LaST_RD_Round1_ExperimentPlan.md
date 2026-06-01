# LaST-RD Round1 Experiment Plan

## Baseline

- Standard baseline: A0-official-aligned.
- Best checkpoint: `step_00100000`.
- Full navtest PDMS: `0.864891`.
- Eval protocol: full navtest, `num_samples=12146`, `num_pdm_valid=12138`, fp32.
- A0-local-fixed is not the main comparison target.

All LaST-RD Round1 claims must compare against `0.864891`. Only a full navtest best PDMS above `0.864891` counts as the main metric beating baseline.

## Wave 0: Preflight

Run preflight before any 8-GPU job:

- Hydra config check for Stage1.5 and Progressive SFT.
- Eval-safe config check.
- Strict cache manifest smoke, and full manifest when requested.
- Real-batch smoke with Stage1.5 forward/backward, Progressive SFT forward/backward, and `get_action` without target tokens.
- Readiness gate summary.

The known training cache root is:

```text
/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1
```

This cache stores legacy raw NAVSIM `high_command_one_hot` as `[4]`; the loader verifies the fourth slot is zero, passes `[3]` left/straight/right to LaST-RD, and keeps the 8D `status_feature` unchanged for A0/Stage2 compatibility.

## Wave 1: Stage1.5 Adapter Pretraining

Server 1: `last_rd_stage1_5_full`

- Full Last-RD adapter.
- JEPA dynamic future prediction.
- VGGT geometry-lite/full token path.
- Coarse trajectory token learning.
- `use_expert_features=false`, `use_last_rd=true`.
- Trainable scope: `action_head.last_rd.*` only.

Server 2: `last_rd_stage1_5_jepa_only`

- Backup/ablation adapter.
- JEPA dynamic + ego/coarse trajectory only.
- `use_vggt=false`, `use_vggt_geometry_tokens=false`.
- No VGGT geometry loss.
- No risk loss.
- Trainable scope: `action_head.last_rd.*` only.

Both jobs use 8 GPUs, per-GPU batch size 16, grad accumulation 1, effective batch size 128, and 20 epochs.

## Wave 2: Progressive SFT

Both Progressive SFT jobs initialize from the Server 1 full Stage1.5 adapter.

Server 1: `last_rd_progressive_lastrd_only`

- `use_last_rd=true`
- `use_expert_features=false`
- A0 official checkpoint initialization.
- Frozen A0 reference for noise KD.
- 200 epochs, 8 GPUs, batch size 16 per GPU.

Server 2: `last_rd_progressive_hybrid`

- `use_last_rd=true`
- `use_expert_features=true`
- A0 official checkpoint initialization.
- Frozen A0 reference for noise KD.
- 200 epochs, 8 GPUs, batch size 16 per GPU.

## Wave 3: Checkpoint Sweep Full Eval

Evaluate these checkpoints when present:

- `step_00050000`
- `step_00060000`
- `step_00080000`
- `step_00100000`
- `step_00120000`
- Lightning top-k validation/loss checkpoints
- `latest` / final checkpoints

Eval settings:

- Precision: fp32.
- Split: full navtest.
- Fixed metric cache.
- Hybrid eval config: `configs/last_rd/last_rd_progressive_sft_hybrid_eval.yaml`.
- LastRD-only eval config: `configs/last_rd/last_rd_progressive_sft_lastrd_only_eval.yaml`.

## Wave 4: Corruption Eval

Run first with `max_samples=1000`. Run full corruption only if the 1k sample result shows meaningful token dependency.

Modes:

- normal
- zero-all-last-rd
- zero-jepa-dynamic
- shuffle-jepa-dynamic
- zero-vggt-geometry
- shuffle-vggt-geometry
- zero-ego-tokens
- optional zero-risk-tokens if a supervised/proxy risk branch is used

For batch size 1, shuffle modes are token-order reverse, not cross-scene shuffle. Do not over-interpret them as scene-level permutation until a permutation-file workflow is added.

## Success Criteria

- Main success: best full navtest PDMS > `0.864891`.
- Strong signal: PDMS >= `0.867`.
- Attribution signal: `normal PDMS - zero_all_last_rd PDMS >= 0.003`.
- If corruption has no effect, do not claim latent reasoning even if PDMS improves.

## Interpretation

- `hybrid > lastrd_only`: legacy A4 expert context helps.
- `lastrd_only ~= hybrid`: LastRD planning tokens are the main driver.
- `lastrd_only > hybrid`: legacy A4 concat context may interfere.
- Both below baseline: inspect loss curves, group weights, geometry mode, and corruption impact.
