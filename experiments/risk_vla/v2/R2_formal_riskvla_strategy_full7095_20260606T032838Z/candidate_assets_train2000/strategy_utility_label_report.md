# Counterfactual Candidate Asset Report

Input: `/mnt/project/bit_drive_left_tail/experiments/bit_drive/d5_mining_v2_final/counterfactual_candidates.jsonl`
Split: `navtrain`
Purpose: `training`
Tokens: `2000`
Utility label rows: `10000`
Safe alignment pairs: `1353`
Candidate tensor shape: `[2000, 5, 8, 3]`

| Candidate | Rows | Mean PDMS | Zero | DAC0 | NC0 | TTC0 | Path Repair | Path Regression | NC Regression | TTC Regression | Tail Risk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base_det | 2000 | 0.916047 | 60 | 54 | 6 | 35 | 0 | 0 | 0 | 0 | 94 |
| bit_det | 2000 | 0.916370 | 58 | 54 | 4 | 29 | 7 | 7 | 0 | 2 | 88 |
| bit_stochastic_seed0 | 2000 | 0.918172 | 53 | 51 | 2 | 28 | 9 | 6 | 0 | 2 | 84 |
| bit_stochastic_seed1 | 2000 | 0.917755 | 54 | 52 | 2 | 28 | 8 | 6 | 0 | 2 | 85 |
| bit_stochastic_seed2 | 2000 | 0.918813 | 53 | 50 | 3 | 26 | 9 | 5 | 0 | 1 | 81 |

Positive anchor counts:
{"base_det": 1490, "bit_det": 201, "bit_stochastic_seed0": 103, "bit_stochastic_seed1": 108, "bit_stochastic_seed2": 98}

Negative anchor counts:
{"base_det": 837, "bit_det": 396, "bit_stochastic_seed0": 272, "bit_stochastic_seed1": 245, "bit_stochastic_seed2": 250}

Leakage guard: training mode rejects rows whose split name contains `test` or `navtest`.
