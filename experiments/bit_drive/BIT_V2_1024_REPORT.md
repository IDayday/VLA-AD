# BiT v2 1024-Sample Report

Branch: `research/bit-drive-left-tail`
Commit: `593f05d`

Eval split: `navtest`
Subset size: 1024 samples
Seed: `20260601`
Chunk cache: `/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1`
Metric cache: `/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1`

## Results

| Run | Mean | Median | P5 | P10 | Zero | DAC0 | NC0 | TTC0 | Ego |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | 0.8690 | 0.9290 | 0.0000 | 0.8003 | 62 | 56 | 6 | 43 | 0.8007 |
| B5 terminal-only context | 0.8738 | 0.9272 | 0.0000 | 0.8158 | 58 | 49 | 9 | 40 | 0.8008 |
| B6 reverse low weight | 0.8737 | 0.9293 | 0.0000 | 0.8125 | 60 | 51 | 9 | 40 | 0.8023 |

## Delta vs A0

| Run | Mean | P10 | Zero | DAC0 | NC0 | TTC0 | Ego |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B5 | +0.0048 | +0.0154 | -4 | -7 | +3 | -3 | +0.0001 |
| B6 | +0.0046 | +0.0122 | -2 | -5 | +3 | -3 | +0.0016 |

## Interpretation

B5 is the best v2 candidate by zero-score, DAC0, mean, and P10. B6 is slightly safer on TTC in the 256 run, but on 1024 it does not remove the NC regression. Both B5 and B6 reduce zero-score, DAC0, and TTC0 counts without ego-progress collapse, but both increase NC0 from 6 to 9.

## Recommendation

Do not claim BiT v2 success yet and do not proceed to Target-Constrained GRPO. The correct next step is NC-regression diagnosis and a revised safety-aware conditioning/loss design. A reasonable working hypothesis is that terminal-only weak conditioning helps drivable-area and progress intent, while collision responsibility cases still need explicit interaction/safety modeling rather than stronger target conditioning or broad failure oversampling.
