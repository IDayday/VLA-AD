# BiT-Select Report

Updated: `2026-06-02 03:00:32 UTC`

Branch: `research/bit-drive-left-tail`
Commit: `593f05d`

No Target-Constrained GRPO/RL was run. Navtest labels were not used for selector training or non-test threshold selection. Oracle rows below are analysis-only.

## Status

Direct BiT conditioning is still promising but unsafe. B5/C1 improves left-tail metrics but raises NC0 from `6` to `9`; v3 C1-C4 did not fix that. BiT-Select moved the problem to selector/reranker design, but the current FeatureMLP over trajectory-delta features does not yet meet the navtest-1024 acceptance criteria.

Acceptance target on matched navtest-1024:

- Mean `>= 0.8710`
- P10 `>= 0.8003`
- Zero `<= 60`
- DAC0 `<= 51`
- TTC0 `<= 43`
- NC0 `<= 7`

## Navtest Baselines

| Method | Mean | P10 | Zero | DAC0 | NC0 | TTC0 | Ego | Use Bit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 Base | 0.8690 | 0.8003 | 62 | 56 | 6 | 43 | 0.8007 | 0 |
| B5/C1 BiT | 0.8738 | 0.8157 | 58 | 49 | 9 | 40 | 0.8007 | 1024 |
| Rule fallback | 0.8712 | 0.8060 | 59 | 52 | 7 | 40 | 0.7979 | 712 |
| Safety oracle | 0.8776 | 0.8184 | 53 | 49 | 4 | 37 | 0.8052 | 1017 |
| Oracle best | 0.8846 | 0.8277 | 48 | 44 | 4 | 37 | 0.8151 | 239 |

Safety oracle shows there is headroom, but it uses per-sample metrics and is not deployable.

## Selector Experiments

| Experiment | Train Data | Threshold Source | Mean | P10 | Zero | DAC0 | NC0 | TTC0 | Ego | Use Bit | Result |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| P4 FeatureMLP | train 2048, margin 0.01 | val 1024 | 0.8738 | 0.8157 | 58 | 49 | 9 | 40 | 0.8008 | 975 | Fails NC |
| P5 FeatureMLP | train 2048, margin 0.005 | val 1024 | 0.8740 | 0.8168 | 58 | 49 | 9 | 41 | 0.8022 | 930 | Fails NC |
| P6 FeatureMLP | train 2048, margin 0.0 | val 1024 | 0.8740 | 0.8157 | 58 | 49 | 9 | 41 | 0.8024 | 967 | Fails NC |
| P6 + override | P6 selector | P4 val + P7 chunk013 | 0.8704 | 0.8052 | 60 | 53 | 7 | 41 | 0.7984 | 612 | Fails Mean/DAC |
| P9 FeatureMLP | P6 + P7 safety mining, 3336 rows | val 1024 | 0.8755 | 0.8155 | 57 | 48 | 9 | 42 | 0.8055 | 730 | Fails NC |
| P9 + val override | P9 selector | P6 val only | 0.8744 | 0.8128 | 58 | 49 | 9 | 42 | 0.8043 | 716 | Fails NC |
| P9 + P10 override | P9 selector | P6 val + P10 holdout | 0.8741 | 0.8116 | 58 | 50 | 8 | 41 | 0.8035 | 540 | Fails NC by 1 |
| P9 + P10 thr030 | P9 selector | P6 val + P10, fixed selector threshold 0.30 | 0.8705 | 0.8009 | 61 | 55 | 6 | 41 | 0.8008 | 257 | Safe but loses left-tail gains |
| P9 + P10 thr035 | P9 selector | P6 val + P10, fixed selector threshold 0.35 | 0.8698 | 0.8007 | 62 | 56 | 6 | 41 | 0.8006 | 197 | Safe but near Base |
| P9 + P10 thr040 | P9 selector | P6 val + P10, fixed selector threshold 0.40 | 0.8732 | 0.8112 | 59 | 51 | 8 | 42 | 0.8038 | 463 | Fails NC by 1 |

Best pre-P11 tradeoff was P9 + P10 override: it preserved most BiT left-tail gains but had `NC0=8`, one above acceptance. P11 improves utility further but keeps the same core tradeoff: strong configurations remain at `NC0=8-9`, while safe configurations lose too much DAC.

## P10 Safety Mining

New non-test holdout chunks were built from NAVSIM train chunks and were not used for navtest training/tuning.

| Dataset | Rows | Valid | Label 1 | DAC Fix | DAC Reg | NC Reg | NC Fix | TTC Reg | TTC Fix | Base NC0 | BiT NC0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P10 chunk018 | 256 | 253 | 53 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 1 |
| P10 chunk019 | 256 | 256 | 65 | 0 | 0 | 0 | 1 | 1 | 2 | 3 | 2 |
| P10 merged | 512 | 509 | 118 | 0 | 0 | 0 | 1 | 2 | 2 | 4 | 3 |
| Navtest 1024 | 1024 | 1024 | 49 | 12 | 5 | 5 | 2 | 3 | 6 | 6 | 9 |

Key diagnosis: P10 still contains zero NC regression examples, while navtest has five. This confirms the current non-test train/val mining distribution does not cover the navtest NC failure mode.

The attempted chunk000014 1024 metric cache was stopped because the first 1024 samples pulled a 452-scenario log and was too slow for rapid iteration. It was not used for selector training or threshold selection.

## Code Changes

- `scripts/tune_bit_selector_safety_override.py` was optimized with precomputed numpy arrays so selector/override threshold sweeps are fast enough for iterative use.
- P9 augmented selector training was run on P6 train plus P7 safety-mining samples relabeled with `margin_pdm=0.0`.
- P10 metric caches and counterfactuals were generated for train chunks 000018 and 000019.
- P11 enriched trajectory features were added in `navsim/agents/recogdrive/bit_safety_selector.py`.
- `scripts/augment_bit_counterfactual_features.py` was added to recompute selector features without rerunning Base/BiT inference.

## P11 Enriched Features

P11 expanded selector feature count from `15` to `37`, adding per-point early x/y deltas, heading deltas, absolute terminal deltas, max forward/lateral deltas, endpoint-distance deltas, and step-length summaries.

| Experiment | Threshold Source | Mean | P10 | Zero | DAC0 | NC0 | TTC0 | Ego | Use Bit | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| P11 enriched selector | P6 val | 0.8762 | 0.8183 | 56 | 47 | 9 | 41 | 0.8048 | 869 | Strong gains, fails NC |
| P11 + override | P6 val + P10 | 0.8716 | 0.8056 | 60 | 54 | 6 | 40 | 0.8013 | 305 | Safe, fails DAC |
| P11 + override thr015 | P6 val + P10, fixed selector threshold 0.15 | 0.8749 | 0.8161 | 57 | 49 | 8 | 40 | 0.8031 | 653 | Strong gains, fails NC by 1 |
| P11 + override thr020 | P6 val + P10, fixed selector threshold 0.20 | 0.8751 | 0.8161 | 57 | 49 | 8 | 40 | 0.8035 | 614 | Strong gains, fails NC by 1 |

P11 improves selector utility but still cannot identify enough NC-risk scenes without over-filtering useful DAC fixes.

P11 thr015 false-positive NC diagnosis:

- `3` selected samples turn Base NC nonzero into BiT NC zero.
- All three have Base DAC=1 and BiT DAC=1, so they are pure NC regressions rather than DAC/NC tradeoffs.
- Their trajectory deltas are not consistently large: one is forward-aggressive, one moves backward, and one is nearly neutral.
- This implies the missing signal is likely scene context/object interaction, not just trajectory shape.

## Decision

Do not claim BiT-Select success. The current learned selector can either:

- keep BiT's Mean/P10/Zero/DAC/TTC gains but leave NC0 at `8-9`, or
- suppress NC0 to `6-7` but lose too much DAC improvement.

Selector/reranker experiments are now diagnostic only, not the main method. A learned gate is too engineering-heavy as a core contribution and would mask the method issue rather than improving BiT itself.

Do not proceed to Target-Constrained GRPO in this branch yet. Do not tune deployable thresholds on navtest. Do not broaden failure sampling using navtest labels.

## P12 Taxonomy Result

A new taxonomy pass was run at:

`/mnt/project/VLA-AD/experiments/bit_drive/select/P12_dac_nc_focused/case_taxonomy/case_taxonomy_report.md`

Key result on matched navtest-1024:

- `DAC0` and `NC0` remain the important zero-score failure modes.
- BiT has useful path/DAC signal, but the strict metric-defined DAC/zero-fix domain is small: `14/1024` samples.
- BiT's `NC` regressions are mostly pure interaction-safety regressions, not DAC/NC tradeoffs: `5` new NC regressions and `0` path-safety overlap in this taxonomy.
- This means a selector that only learns to identify a tiny DAC-fix subset is not a sufficiently general method.

## BiT-v4 Direction

The next method-centric direction is BiT-v4 risk-aware intent conditioning:

- Keep BiT as a single planner, not Base/BiT gated selection.
- Preserve terminal/path intent conditioning as the IL-stage prior.
- Add a BiT risk head predicting `zero`, `DAC0`, `NC0`, and `TTC0` risks from context + terminal/path intent.
- Optionally inject a learned risk token into diffusion context so the planner receives risk-aware intent information.
- Train risk supervision only from non-test counterfactual/PDM labels; navtest labels remain analysis-only.
- Treat this as IL-stage initialization improvement before later RL, not as a replacement for RL.

Implemented and smoke-tested:

- `BitRiskHead` and `RiskTokenEncoder` in `navsim/agents/recogdrive/bit_drive.py`.
- BiT risk config/loss/token plumbing in `navsim/agents/recogdrive/recogdrive_diffusion_planner.py`.
- Optional `--risk-label-jsonl` and `--require-risk-labels` in `scripts/train_bit_drive_chunked.py`.
- `configs/bit_drive/v4/bit_v4_D1_risk_aware_terminal.yaml`.
- `scripts/run_bit_v4_risk_aware_plan.py`.
- CPU smoke test passed: `python scripts/smoke_test_bit_drive_dummy_flow.py --device cpu`.

## BiT-v4 D1-D4 Result

BiT-v4 D1-D4 were run on matched navtest-1024. No RL/GRPO was run.

| Method | Mean | P10 | Zero | DAC0 | NC0 | TTC0 | Ego | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A0 Base | 0.8690 | 0.8003 | 62 | 56 | 6 | 43 | 0.8007 | baseline |
| B5/C1 BiT | 0.8738 | 0.8158 | 58 | 49 | 9 | 40 | 0.8008 | useful but unsafe NC |
| D1 risk token | 0.8732 | 0.8030 | 59 | 52 | 7 | 46 | 0.8075 | controls NC, fails DAC/TTC |
| D2 aux no token | 0.8714 | 0.8029 | 60 | 53 | 7 | 44 | 0.8037 | safer, weak DAC/TTC |
| D3 low risk loss | 0.8702 | 0.8024 | 62 | 55 | 7 | 44 | 0.8025 | too weak |
| D4 weak risk token | 0.8732 | 0.8031 | 59 | 52 | 7 | 46 | 0.8076 | same tradeoff as D1 |

Conclusion: risk-aware BiT improves NC containment compared with B5/C1, but current risk-token/risk-BCE design does not meet acceptance because DAC0 remains too high and TTC0 regresses. The result is useful as IL-stage risk prior work, not a standalone success.

## Next Plan: D5 Conservative Risk-Regularized BiT

Do not keep tuning selectors or risk-token scalar strengths. Next, use non-test risk labels to constrain unsafe trajectories directly:

1. Extend training data loading to attach `base_pred_traj` from non-test counterfactual JSONL.
2. For samples where B5/BiT counterfactual has `NC0` or `TTC0`, add early-longitudinal SmoothL1/KD toward Base trajectory.
3. Keep terminal/path conditioning so DAC/path correction remains available.
4. Evaluate whether D5 keeps `NC0 <= 7` while recovering DAC/TTC gains.

This remains supervised IL. Later RL can use the improved initialization, but this branch should still not run GRPO yet.
