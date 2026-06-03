# Stage 3 Planner Integration

RISK-VLA is a risk-conditioned multi-strategy VLA planning framework for low-score risk scenarios / critical-risk subsets:

```text
VLM state + ego history + command + optional BiT path intent
  -> RiskStateEncoder
  -> RiskConditionedStrategyRouter
  -> RiskConditionedStrategyBank
  -> diffusion context tokens + horizon residual
  -> ReCogDrive diffusion planner
```

BiT is not the core algorithm. In this integration it is only the path/terminal intent formulation used by the `PathIntentStrategy` inside the strategy bank.

## Config Fields

- `use_risk_vla`: master gate, default `false`.
- `risk_vla_num_classes`: default `6`, using `[low_score, path_dac, interaction_nc, ttc, progress, comfort]`.
- `risk_vla_router_mode`: `independent` by default; `learned_softmax` is an ablation.
- `risk_vla_use_bit_summary`: allows optional BiT path/terminal summaries as path-intent strategy inputs.
- `risk_vla_use_oracle_router`: analysis-only upper bound using provided labels for routing.
- `risk_vla_strategy_token_scale`: scales appended strategy tokens.
- `risk_vla_horizon_residual_scale`: scales added horizon residuals.
- `risk_vla_risk_loss_weight`: BCE risk-head diagnostic loss weight.
- `risk_vla_focal_loss_weight`: optional focal risk loss weight.
- `risk_vla_strategy_entropy_weight`: optional router entropy regularizer.
- `risk_vla_detach_risk_for_strategy`: detaches risk-state tensors before routing by default.
- `risk_vla_log_diagnostics`: emits risk/router/strategy summaries.

Default ReCogDrive, BiT, and LaST-RD behavior remains unchanged when `use_risk_vla=false`.

## Modes

`configs/risk_vla/risk_vla_diagnostic.yaml` enables risk-head supervision with both strategy scales set to zero. It is for checking label quality and prediction signal before planner modulation.

`configs/risk_vla/risk_vla_predicted_router_pilot.yaml` enables predicted-risk routing with small token/residual scales. This is the first actual RISK-VLA conditioning pilot.

`configs/risk_vla/risk_vla_oracle_router_pilot.yaml` routes from labels and is analysis-only. It must not be used as a standard training setting, and it must not use test labels for training.

## Planner Injection

RISK-VLA appends strategy tokens to `context_tokens` and adds a horizon residual to `expert_step_condition`. It does not overwrite, clamp, or post-process the final predicted trajectory.

Supported labels:

- Extended: `risk_labels` with shape `[B, 6]` or `[B, H, 6]`.
- MVP: `generic_risk_labels`, `drivable_risk_labels`, `ttc_risk_labels`, `comfort_risk_labels`.

Loss/log keys when enabled:

- `risk_vla_risk_loss`
- `risk_vla_focal_loss`
- `risk_vla_strategy_entropy_loss`
- `risk_vla_total_aux_loss`
- `risk_vla_prob_low_score`
- `risk_vla_prob_path_dac`
- `risk_vla_prob_interaction_nc`
- `risk_vla_prob_ttc`
- `risk_vla_prob_progress`
- `risk_vla_prob_comfort`
- `risk_vla_weight_base`
- `risk_vla_weight_path_intent`
- `risk_vla_weight_interaction`
- `risk_vla_weight_progress`
- `risk_vla_weight_comfort`
- `risk_vla_strategy_entropy`

## Smoke Commands

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/risk_vla/smoke_risk_vla_planner_forward.py --device cpu
PYTHONDONTWRITEBYTECODE=1 python scripts/risk_vla/smoke_risk_vla_agent_forward.py --device cpu
```

These use synthetic tensors only. They do not require NAVSIM data, metric caches, checkpoints, GPU, or PDM evaluation.

## Leakage Rules

Train and validation labels can supervise diagnostics and pilots. Test labels are for analysis only and must not be injected into training configs or chunk caches used for training.
