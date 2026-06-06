# RISK-VLA v2 SOTA Roadmap

Method name: **RISK-VLA v2: Risk-Conditioned Multi-Candidate Strategy Planning for Vision-Language-Action Driving**.

## Core Shift

RISK-VLA v1 uses a risk head, strategy routing, strategy tokens, and horizon residuals to modulate one diffusion sample. RISK-VLA v2 keeps the ReCogDrive diffusion backbone but moves the decision point to:

```text
fine-grained risk/world state
  -> candidate generation
  -> trajectory-conditioned risk/utility critic
  -> constrained utility routing / candidate scoring
  -> optional supervised safe alignment
```

BiT/path intent remains one `PathIntentStrategy`, not the core algorithm.

## Stage Order

1. Build strategy utility labels and safe alignment pairs from train/val matched candidates.
2. Train/evaluate `TrajectoryRiskCritic` with risk, submetric-delta, pairwise preference, constrained-safety, and CVaR losses.
3. Export/evaluate a multi-candidate bank with K=4 pilots and K=8/16 SOTA mode where compute allows.
4. Train learned utility router and `scorer_select`.
5. Run Round2 small pilots at max 256/1024/4096 samples.
6. Scale only after gates show no material NC/TTC regression.
7. Add world/risk expert tokens.
8. Add optional VLM LoRA only if risk prediction is the bottleneck.
9. Add anchor-GRPO only after supervised gates pass.

## Leakage Rules

- Train/val PDM labels can build training labels.
- Navtest/test labels are analysis-only.
- No navtest/test threshold tuning, router supervision, hard-negative mining, or GRPO reward tuning.

## Small Pilot Gates

- Fine-grained risk prediction must beat enrichment gates.
- The learned critic must rank positive anchors above negative anchors on held-out val.
- Candidate scorer must not increase NC/TTC on matched eval.
- The method must improve zero/DAC without material NC/TTC regression before full-scale training.

## Reporting

Every stage writes Markdown plus JSON/CSV outputs. Do not claim SOTA unless the benchmark, split, and protocol match exactly.

## Round3 Scale Policy Update

Round3 upgrades this roadmap to the v3 full-scale execution policy:

- Real experiment deliverables must use at least 10000 samples; smaller runs are debug-only and not evidence.
- Full train/val/navtest is preferred whenever explicit data and checkpoints exist.
- Navtest/test labels remain analysis-only and never feed training labels, threshold tuning, router supervision, hard-negative mining, VLM instruction data, or GRPO rewards.
- If full inputs are missing, discovery and blocker reports are the deliverable; do not replace the run with a toy subset.

Current Round3 discovery found enough train chunk tokens and navtest analysis tokens, but not enough held-out validation tokens and not enough matched train PDM tables for utility-label construction. Formal critic/router/safealign training is blocked until those inputs are materialized.
