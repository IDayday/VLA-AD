# PTA-FS-DiT Stage2 Implementation

## Repository Baseline

- Branch: `feature/recogdrive-last-vla-v2`
- Base commit: `36e221858b30b973d9995f5853bbf7b16464245d`
- Pre-existing untracked training artifacts were not modified or added.

## Static Audit

The pre-change Stage2 target chain was:

1. SG-FPS selected supports were loaded by token.
2. DPSI built support/anchor weights and ASMI sampled a small target set.
3. The sampled raw trajectories were encoded into legacy or FS representation.
4. Epsilon MSE was computed per target and reduced with the original scene-normalized ASMI weights.

The ordinary and Last-VLA calls to `_compute_x0_geo_aux_losses()` used
`action_input.action`, even when diffusion used a teacher-selected or residual target. The
DPSI per-target path used sampled supports correctly. This mismatch is fixed by the
`TrainingTarget` contract.

Audited compatibility/config fields:

| Field | Pre-change status | Current status |
|---|---|---|
| `cot_condition_tokens` | Last-VLA output and DiT condition key | Kept as compatibility alias; internal code prefers `planning_condition_tokens` |
| `last_vla_cot_condition_layers` | Declared and passed, but ignored by DiT | Active deprecated alias controlling Last-VLA layer injection |
| `last_vla_vlm_context_dropout_start/end` | Declared, no forward effect | Non-default values emit `DeprecationWarning` |
| `last_vla_cot_num_steps` | Declared, no forward effect | Non-default values emit `DeprecationWarning` |
| `last_vla_fusion_tokens` | Declared, actual count followed CoT tokens | Non-default values emit `DeprecationWarning` |
| `use_fs_norm` | Selected FS encode/decode | Retained, with explicit clipping and v2 version guard |
| `fs_norm_target_clip` | Could implicitly clip diffusion targets | Deprecated and ignored; non-default positive values warn |
| `fs_norm_output_clip_mode` | `stats_bounds` did not cover every reverse path | Unified through `_bound_output_representation()` |
| `x0_aux_weight` | Legacy low-noise raw x0 auxiliary | Retained for old configs; main PTA config sets zero |
| `delta_aux_weight` | Legacy low-noise delta auxiliary | Retained for old configs; main PTA config sets zero |
| `geo_aux_weight` | Legacy feasibility bundle | Retained for old configs; main PTA config sets zero |

## Planning Token Data Flow

The standalone adapter reads only encoded VLM tokens, status, command, and history. It
builds one static `[B, 16, 384]` condition per scene. Raw VLM context remains the DiT
cross-attention memory.

```text
vl_features -> feature_encoder -> raw VLM tokens -----------------> DiT context
                                  |                                  |
status + command + history -------+-> PlanningTokenAdapter ----------+-> gated planning residual
                                  |
                                  +-> gated mean residual -> fusion context mean
```

The adapter runs once before a DDPM/DDIM chain. Its tokens are passed unchanged into each
reverse step. In interleaved LightningDiT, the default `cross_attention` mode injects only
into odd blocks. Each block uses its retained `cot_cross_attn`/`cot_out_proj` keys with a
new sigmoid gate initialized to `0.05`. Last-VLA can retain the old zero-init and
straight-through gradient path through `planning_legacy_cot_gradient_path=True`.

## FS-Norm v2

v1 statistics treated every support as an equal observation, so scenes with many selected
supports dominated mean, scale, and clipping bounds. v2 assigns each support in scene `i`
weight `1/K_i`, giving every scene equal total mass.

- Shape is fixed to `[8, 3]`.
- XY uses per-step weighted mean/std or weighted median/MAD.
- Heading center is exactly zero; its scale is RMS about zero or weighted median absolute
  heading delta in robust mode.
- Clip bounds are scene-balanced weighted quantiles in normalized delta space.
- Only explicit selected supports are read; hard negatives and generic GT/IL fallbacks are
  excluded.
- v2 metadata records representation, `p0`, balancing/heading flags, counts, archive path,
  and fingerprint.
- v1 files remain loadable. The PTA main config requires `fs_norm_min_version: 2`.

Target encoding always uses `apply_clip=False`. Inference output bounds use the single
planner helper; requesting `stats_bounds` without `clip_lower/clip_upper` fails instead of
silently falling back to scalar clipping.

## TrainingTarget Contract

Every Stage2 diffusion path now carries:

```text
TrainingTarget(
  raw_trajectory,
  selected_repr,
  diffusion_target_repr,
  residual_anchor_repr,
  residual_alpha,
  diagnostics,
)
```

GT, Last-VLA teacher selection, DPSI/ASMI supports, and residual diffusion all use this
object. Epsilon construction uses `diffusion_target_repr`; trajectory and geometry
auxiliaries use `raw_trajectory`. For residual diffusion:

```text
full_x0_repr = predicted_residual_x0 + residual_alpha * residual_anchor_repr
```

The full representation is decoded before trajectory or reference-relative geometry loss.
Last-VLA coarse supervision receives the same selected representation. Under FS-Norm its
coarse head is linear plus the planner output bound, and legacy heading/progress losses are
forbidden.

## Losses

The epsilon prediction and original DPSI/ASMI weighting are unchanged.

```text
xy_i = mean SmoothL1(pred_raw_xy, target_raw_xy; beta=trajectory_huber_beta)
heading_i = mean(1 - cos(wrap(pred_heading - target_heading)))
trajectory_i = xy_i + trajectory_heading_weight * heading_i

alpha_weight_i = alpha_bar[t_i] ** aux_alpha_power
ramp = min((epoch + 1) / max(aux_warmup_epochs, 1), 1)

total = existing_loss
      + ramp * trajectory_aux_weight * mean(alpha_weight_i * trajectory_i)
      + ramp * feasibility_aux_weight * mean(alpha_weight_i * feasibility_i)
```

`feasibility_i` contains only reference-relative tangent-heading excess and curvature
excess. Both compare detached target geometry, apply configured margins, mask short
segments, and use SmoothL1 on positive excess.

Primary implementation locations:

- Adapter: `navsim/agents/recogdrive/planning_token_adapter.py`
- Planning branch: `navsim/agents/recogdrive/recogdrive_dit.py`
- Target/loss/sampler contract: `navsim/agents/recogdrive/recogdrive_diffusion_planner.py`
- FS transforms/statistics: `navsim/agents/recogdrive/fs_norm.py` and
  `navsim/agents/recogdrive/pareto_support/fs_norm.py`
- Training barrier: `navsim/agents/recogdrive/reference_relative_geometry.py`

## Main Configuration

`navsim/planning/script/config/experiment/pta_fs_dit_stage2_103k.yaml`:

- `use_fs_norm: true`
- `fs_norm_min_version: 2`
- `use_planning_token_adapter: true`
- `planning_token_source: adapter`
- `planning_num_tokens: 16`
- `planning_num_heads: 8`
- `planning_condition_layers: cross_attention`
- `planning_gate_init: 0.05`
- `planning_context_gate_init: 0.05`
- `planning_condition_dropout: 0.10`
- `trajectory_aux_weight: 0.05`
- `feasibility_aux_weight: 0.01`
- `aux_alpha_power: 1.0`
- `aux_warmup_epochs: 10`
- `tangent_margin_rad: 0.08`
- `curvature_margin: 0.05`
- `min_segment_length: 0.20`
- `x0_aux_weight: 0.0`
- `delta_aux_weight: 0.0`
- `geo_aux_weight: 0.0`

All new common agent defaults are off. The launcher is
`scripts/training/sg_fps/run_train_pta_fs_dit.sh`.

## A0-A5 Ablations

All rows preserve epsilon prediction, SG-FPS/DPSI/ASMI target weighting, and the same DiT.
Legacy `x0/delta/geo` weights are zero in every row.

| ID | FS v2 | PTA | Trajectory aux | Feasibility aux | Purpose |
|---|---:|---:|---:|---:|---|
| A0 | off | off | 0 | 0 | Flag-off SG-FPS/DPSI baseline |
| A1 | on | off | 0 | 0 | Scene-balanced representation only |
| A2 | off | on | 0 | 0 | Planning adapter with legacy representation |
| A3 | on | on | 0 | 0 | Conditioning plus representation |
| A4 | on | on | 0.05 | 0 | Add selected-target trajectory reconstruction |
| A5 | on | on | 0.05 | 0.01 | Full PTA-FS-DiT main proposal |

## Bounded Validation

- Unit/integration tests cover adapter shape/dropout/gradient, layer selection/gates, static
  DDIM caching, explicit FS target clipping, scene-balanced v2 stats, target consistency,
  residual reconstruction, barrier behavior, auxiliary alpha/gradient, coarse FS semantics,
  flag-off output, and old strict-false checkpoint loading.
- A real-support smoke built v2 stats from 16 scenes/179 selected supports, ran 10 optimizer
  steps with the production 16-layer/384-dim DiT, and ran deterministic 5-step DDIM.
- No 200-epoch PTA training or new PTA NAVSIM evaluation was launched.
