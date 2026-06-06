# RISK-VLA v3 Full Matrix Plan

## Launchability

- `train_labels_10k`: `False`
- `critic_10k`: `False`
- `router_10k`: `False`
- `navtest_10k_analysis`: `True`
- `full_training`: `False`

## Matrix

- A0 Base
- B3/BiT direct
- D5 conservative BiT
- RISK-VLA v1
- RISK-VLA v3 heuristic router
- RISK-VLA v3 learned router
- RISK-VLA v3 scorer_select K=8
- RISK-VLA v3 scorer_select K=16
- RISK-VLA v3 safealign SFT
- RISK-VLA v3 safealign SFT + CVaR
- RISK-VLA v3 world tokens
- RISK-VLA v3 VLM risk LoRA

## Blockers

- held-out val chunk tokens below 10000: 1024; do not substitute navtest for tuning
- Need at least two train split PDM tables with >=10k rows to build strategy utility labels.
