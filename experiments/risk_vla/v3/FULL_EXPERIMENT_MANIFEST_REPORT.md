# RISK-VLA v3 Full Experiment Manifest Summary

Git commit: `2a00819a3e3bd4993869ab65c4656da889e9eede`
Minimum real samples: `10000`

## Launchability

- `train_labels_10k`: `False`
- `critic_10k`: `False`
- `router_10k`: `False`
- `navtest_10k_analysis`: `True`
- `full_training`: `False`

## Blockers

- held-out val chunk tokens below 10000: 1024; do not substitute navtest for tuning
- Need at least two train split PDM tables with >=10k rows to build strategy utility labels.

## Leakage Guard

Manifest keeps navtest assets under `*_analysis_only` keys and never under training keys.
