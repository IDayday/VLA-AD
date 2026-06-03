# RISK-VLA Research Plan

## Goal

RISK-VLA is a Risk-Conditioned Multi-Strategy Vision-Language-Action Driving research line. The core method is:

```text
risk state -> strategy routing/modulation -> diffusion planning
```

The objective is to improve planning behavior on PDM low-score risk scenarios / critical-risk subsets by diagnosing heterogeneous risk states and routing the planner toward appropriate strategy families. The method should not treat a single path formulation as the algorithm. It should learn when a scene needs path repair, interaction caution, TTC-aware timing, progress recovery, comfort stabilization, uncertainty handling, or a mixture of these.

## Risk Taxonomy

PDM low-score risk scenarios are heterogeneous. Stage 0-1 infrastructure tracks at least these risk classes:

- `low_score`: generic critical-risk subset indicator from low or zero PDM score.
- `path_dac`: path/drivable-area-compliance risk.
- `interaction_nc`: interaction and no-at-fault-collision risk.
- `ttc`: time-to-collision risk.
- `progress`: low ego-progress risk.
- `comfort`: comfort and smoothness risk.
- `uncertainty`: model disagreement, stochastic instability, or low-confidence risk.

The first implementation focuses on labels derivable from PDM CSV metrics. Later stages can add learned uncertainty and interaction-risk teacher signals.

## BiT As A Strategy

BiT is not the core RISK-VLA method. In this branch it should be treated as a path/terminal intent output formulation: one path-risk strategy inside a larger risk-conditioned multi-strategy VLA framework.

Current BiT evidence suggests useful path/DAC repair signal, but incomplete interaction safety. Direct path/terminal conditioning alone can repair some DAC failures while regressing NC/TTC in critical-risk subsets. RISK-VLA therefore uses BiT-style terminal/path outputs as one strategy candidate, not as the central algorithm.

## Stage 0: Risk Taxonomy Analysis

Build matched PDM tables across Base, BiT-style methods, and later risk-conditioned methods. Analyze which low-score risk classes dominate each subset:

- path/DAC failures,
- interaction/NC failures,
- TTC failures,
- progress failures,
- comfort failures,
- mixed-risk cases.

Deliverables:

- matched per-token PDM tables,
- risk transition tables,
- candidate critical-risk subset definitions,
- evidence about which strategies help or harm each risk class.

## Stage 1: Risk Label Construction

Construct train/analysis labels from non-test PDM metrics:

- MVP class-specific tensors compatible with current loaders:
  - `generic_risk_labels`: low-score / generic risk, shape `[H]`,
  - `drivable_risk_labels`: path/DAC risk, shape `[H]`,
  - `ttc_risk_labels`: interaction + TTC risk, shape `[H]`,
  - `comfort_risk_labels`: comfort + progress risk, shape `[H]`.
- Extended tensors:
  - `risk_labels`: shape `[H, 6]`,
  - order `[low_score, path_dac, interaction_nc, ttc, progress, comfort]`.

Labels from navtest/test splits are analysis-only and must not be used for training.

## Stage 2: Risk Head Diagnostic

Train or evaluate lightweight risk heads without changing planner behavior. The diagnostic question is whether scene/context features can predict risk classes well enough to support strategy routing.

Required checks:

- per-class AUROC / AUPRC when available,
- positive rate calibration,
- error analysis on critical-risk subsets,
- train/test leakage audit.

## Stage 3: Risk-Conditioned Strategy Routing

Add risk-conditioned routing/modulation around existing diffusion planning. Candidate strategy families include:

- path/terminal intent strategy, including BiT-style path anchors,
- interaction caution strategy,
- TTC timing strategy,
- progress recovery strategy,
- comfort stabilization strategy,
- uncertainty fallback strategy.

The router should modulate strategy activation from predicted risk state, while preserving the diffusion planner as the action generation backbone.

## Stage 4: Hard Negatives And Optional Risk-Specific GRPO

After a safe supervised initialization is selected, add hard negatives and optional risk-specific online optimization. GRPO/RL is not a Stage 0-1 task and should not be run until supervised diagnostics show a defensible initialization.

Risk-specific optimization should target clearly scoped classes, for example TTC or interaction risk, and should be evaluated against path/DAC regressions and progress/comfort tradeoffs.

## Required Evaluation Tables

### Main PDM Table

Report mean/P10/zero counts and key PDM submetrics for Base, path/terminal strategy variants, and later RISK-VLA variants.

### Risk Transition Table

Report Base-vs-method transitions:

- Path Repair Rate,
- Path Regression Rate,
- Interaction Regression Rate,
- Safe Path Repair Rate,
- Progress Recovery Rate,
- Comfort Recovery Rate,
- raw matched-token counts.

### Risk Prediction Table

Report risk-label prediction diagnostics:

- class name,
- positive rate,
- AUROC/AUPRC when labels and scores are available,
- calibration summary,
- train/analysis split.

### Strategy Activation Table

Report strategy-routing behavior:

- strategy activation rate by risk class,
- activation on critical-risk subset,
- activation on neutral subset,
- downstream PDM delta by activated strategy,
- failure cases where an activated strategy repaired one risk while regressing another.
