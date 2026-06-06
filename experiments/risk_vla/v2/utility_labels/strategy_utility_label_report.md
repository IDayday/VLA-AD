# Strategy Utility Label Report

Status: not built.

The strategy utility label stage requires explicit matched train/val PDM tables for A0/Base and one or more candidate strategies. No such matched input bundle was provided in this workspace, and this stage must not guess private paths or use navtest/test labels for training labels, router supervision, threshold tuning, or hard-negative mining.

Expected outputs after valid inputs are supplied:
- `strategy_utility_labels.jsonl`
- `strategy_utility_labels.csv`
- `summary.json`
- constrained best strategy labels
- positive and negative anchor pairs for safe alignment
