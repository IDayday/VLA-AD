# BiT v3 Safety Report

Branch: `research/bit-drive-left-tail`
Commit: `593f05d`

No Target-Constrained GRPO was run. No broad failure-focused sampling was run. All v3 training/eval artifacts are under `/mnt/project/VLA-AD/experiments/bit_drive/v3`.

## v2 Summary

Matched 1024 NAVSIM navtest eval, seed `20260601`:

| Run | Mean | P10 | Zero | DAC0 | NC0 | TTC0 | Ego |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | 0.8690 | 0.8003 | 62 | 56 | 6 | 43 | 0.8007 |
| B5 | 0.8738 | 0.8158 | 58 | 49 | 9 | 40 | 0.8008 |
| B6 | 0.8737 | 0.8125 | 60 | 51 | 9 | 40 | 0.8023 |

B5 improves Mean, P10, Zero, DAC0, and TTC0, but increases NC0 from 6 to 9. B6 does not fix the NC regression.

## NC Diagnosis

Per-sample B5/B6 diagnosis output: `/mnt/project/VLA-AD/experiments/bit_drive/v3/nc_regression_diagnosis`.

| Run | NC Newly Broken | NC Fixed | DAC Fixed | DAC Regressed | TTC Newly Broken | DAC Fixed and NC Broken |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B5 | 5 | 2 | 12 | 5 | 3 | 0 |
| B6 | 4 | 1 | 9 | 4 | 3 | 0 |

Direct diagnosis:

- B5 NC regression is not concentrated in samples where DAC was fixed: `0/5` overlap.
- B5 new NC cases are not clearly explained by more aggressive longitudinal motion: only `2/5` cases have larger early longitudinal displacement, mean early x delta is `-0.0003`.
- Mean B5 endpoint dx in new NC cases is `-0.1390`; terminal conditioning is not simply pushing endpoints farther forward.

## Seed Stability

Seed stability output: `/mnt/project/VLA-AD/experiments/bit_drive/v3/seed_stability`.

Same 1024 navtest sample order, seeds `0,1,2`, deterministic and stochastic denoising:

| Run | Mode | Mean | P10 | Zero | DAC0 | NC0 | TTC0 | Ego | Delta NC0 vs A0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | deterministic | 0.8689 | 0.7856 | 62.33 | 53.67 | 8.67 | 46.33 | 0.8001 | 0.00 |
| B5 | deterministic | 0.8723 | 0.8134 | 60.67 | 53.00 | 7.67 | 35.67 | 0.7973 | -1.00 |
| B6 | deterministic | 0.8755 | 0.8143 | 57.00 | 47.33 | 9.67 | 40.67 | 0.8036 | +1.00 |
| A0 | stochastic | 0.8624 | 0.7390 | 68.67 | 61.00 | 7.67 | 44.67 | 0.7932 | 0.00 |
| B5 | stochastic | 0.8708 | 0.8118 | 62.33 | 55.00 | 7.33 | 35.00 | 0.7957 | -0.33 |
| B6 | stochastic | 0.8741 | 0.8116 | 58.00 | 47.67 | 10.33 | 42.00 | 0.8023 | +2.67 |

Interpretation: B5's NC regression is not stable across seeds/modes; it appears on the matched seed `20260601` but not in the seed `0,1,2` stability average. B6 has a more persistent NC increase and should not be preferred.

## v3 C1-C4 Results

Matched 1024 NAVSIM navtest eval, seed `20260601`:

| Run | Intent | Mean | P10 | Zero | DAC0 | NC0 | TTC0 | Ego | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A0 | Base | 0.8690 | 0.8003 | 62 | 56 | 6 | 43 | 0.8007 | Baseline |
| C1 | lateral terminal | 0.8738 | 0.8157 | 58 | 49 | 9 | 40 | 0.8007 | Fail: NC0 |
| C2 | lateral path | 0.8729 | 0.8141 | 59 | 50 | 9 | 40 | 0.8000 | Fail: NC0 |
| C3 | lateral terminal+path low strength | 0.8738 | 0.8157 | 58 | 49 | 9 | 40 | 0.8007 | Fail: NC0 |
| C4 | no-x condition | 0.8728 | 0.8140 | 59 | 50 | 9 | 40 | 0.7999 | Fail: NC0 |

Acceptance thresholds from A0 are: Mean `>=0.8710`, P10 `>=0.8003`, Zero `<=60`, DAC0 `<=51`, TTC0 `<=43`, NC0 `<=7`. C1-C4 meet all listed thresholds except NC0.

## Fallback Analysis

Oracle fallback output: `/mnt/project/VLA-AD/experiments/bit_drive/v3/oracle_fallback`.

For B5/C1-like behavior, SafetyConstrainedOracle gives:

| Method | Mean | P10 | Zero | DAC0 | NC0 | TTC0 | Ego |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | 0.8690 | 0.8003 | 62 | 56 | 6 | 43 | 0.8007 |
| B5/C1 | 0.8738 | 0.8157 | 58 | 49 | 9 | 40 | 0.8007 |
| Safety oracle | 0.8776 | 0.8184 | 53 | 49 | 4 | 37 | 0.8052 |

This shows a selector could theoretically preserve DAC/TTC gains while suppressing NC, but oracle fallback uses per-sample ground-truth metrics and is not deployable.

Rule fallback sweep output: `/mnt/project/VLA-AD/experiments/bit_drive/v3/rule_fallback`.

Best NC-containing rule variant observed:

| Rule | Mean | P10 | Zero | DAC0 | NC0 | TTC0 | Ego |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| early 0.02, terminal 0.2 | 0.8712 | 0.8061 | 59 | 52 | 7 | 40 | 0.7980 |

This controls NC0 but fails the DAC0 threshold (`52 > 51`) and sacrifices part of the B5 gain. C5 rule fallback therefore does not satisfy acceptance. C6 learned router was not run because it needs train/val counterfactual labels; navtest labels were not used for training.

## Decision

No v3 direct-conditioning config satisfies the matched 1024 acceptance criteria because NC0 remains `9` for C1-C4. Do not claim BiT-v3 success.

Recommendation:

- Do not proceed to Target-Constrained GRPO yet.
- Keep B5/C1 as a promising but unsafe left-tail signal, not a deployable planner update.
- Prioritize a learned safety router or reranker trained on train/val counterfactual labels, because oracle fallback has a strong upper bound while simple longitudinal rules are insufficient.
- If no train/val router can control NC without losing DAC gains, use BiT terminal/path heads as diagnostics or reward-stage features rather than direct planner conditioning.
