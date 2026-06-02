# BiT v2 Report

Branch: `research/bit-drive-left-tail`
Commit: `593f05d`

## V1 Diagnosis

| Run | Mean | P10 | Zero | DAC0 | NC0 | TTC0 | Ego |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | 0.8557 | 0.5833 | 21 | 17 | 4 | 15 | 0.8076 |
| Step1 | 0.8608 | 0.5833 | 19 | 10 | 9 | 19 | 0.8050 |
| Step2 | 0.8591 | 0.5833 | 20 | 15 | 5 | 16 | 0.8102 |
| Step3 | 0.8488 | 0.5751 | 22 | 18 | 4 | 15 | 0.7965 |

Step1 fixed DAC/zero cases but introduced NC/TTC regressions. The v1 per-sample transition matrix is in `experiments/bit_drive/diagnostics_v1/transition_matrix.md`: Step1 fixed 8 DAC0 cases and 8 zero-score cases, but newly broke 5 NC and 6 TTC cases. Step3 broad failure sampling was harmful.

## V2 256-Sample Results

| Run | Mean | P10 | Zero | DAC0 | NC0 | TTC0 | Ego | Safe vs A0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B1 auxiliary only | 0.8657 | 0.5833 | 19 | 16 | 3 | 12 | 0.8129 | yes |
| B2 context low strength | 0.8659 | 0.5833 | 17 | 13 | 4 | 17 | 0.8179 | no, TTC +2 |
| B3 predicted-only context | 0.8670 | 0.5833 | 18 | 15 | 3 | 13 | 0.8143 | yes |
| B4 path-only context | 0.8658 | 0.5833 | 18 | 14 | 4 | 15 | 0.8154 | yes |
| B5 terminal-only context | 0.8731 | 0.6735 | 16 | 12 | 4 | 15 | 0.8220 | yes |
| B6 reverse low weight | 0.8698 | 0.6685 | 18 | 14 | 4 | 13 | 0.8173 | yes |

B5 was best on the 256 subset by mean, P10, zero-score, DAC0, and ego progress. B6 improved TTC more than B5 but did not beat B5 on zero/DAC.

## 1024-Sample Check

| Run | Mean | P10 | Zero | DAC0 | NC0 | TTC0 | Ego |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 1024 | 0.8690 | 0.8003 | 62 | 56 | 6 | 43 | 0.8007 |
| B5 1024 | 0.8738 | 0.8158 | 58 | 49 | 9 | 40 | 0.8008 |
| B6 1024 | 0.8737 | 0.8125 | 60 | 51 | 9 | 40 | 0.8023 |

B5/B6 preserve the DAC/zero/TTC/P10 improvement signal on 1024 samples, but both regress NC0 from 6 to 9. This blocks a strong success claim under the stated rules.

## Recommendation

Do not proceed to Target-Constrained GRPO. BiT v2 is promising for DAC/zero/TTC left-tail reduction, but the NC regression remains unresolved on 1024 samples. Next step should be per-sample NC regression diagnosis and a safety-aware conditioning/loss change, not broader failure sampling and not GRPO.
