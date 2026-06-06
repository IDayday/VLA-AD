# RISK-VLA v3 Round3 Full Results Report

Date: 2026-06-06

## Full-Scale Status

Full-scale RISK-VLA v3 training/evaluation did not run to completion because the full experiment manifest is not launchable for training.

Launchability from manifest:

- train_labels_10k: false
- critic_10k: false
- router_10k: false
- navtest_10k_analysis: true
- full_training: false

## Available Full-Scale Assets

- Full/large train chunk caches exist.
- Navtest chunk cache exists at >=10000 tokens.
- Navtest PDM analysis tables exist at >=10000 rows for multiple prior methods.
- A0/Base and D5 checkpoints were discovered.
- Expert-token caches for Last/VGGT/JEPA-style paths were discovered.

## Missing for Full Training

- Explicit multi-method train PDM tables on matched >=10000 train tokens.
- Explicit train candidate bank PDM table at K=8/K=16.
- Held-out validation assets at >=10000 tokens for tuning and model selection.

## Next Launch Commands After Inputs Are Registered

```bash
scripts/risk_vla/round3/run_00_discover_full_inputs.sh
BASELINE_PDM_SPEC='name=A0,path=/abs/a0_train_pdm.csv,strategy=base,source_method=A0' \
CANDIDATE_PDM_SPECS='name=BiT,path=/abs/bit_train_pdm.csv,strategy=path_intent;name=D5,path=/abs/d5_train_pdm.csv,strategy=d5_conservative' \
scripts/risk_vla/round3/run_01_build_utility_labels_full.sh
BASE_PREDICTIONS_JSONL=/abs/a0_train_predictions.jsonl \
BIT_PREDICTIONS_JSONL=/abs/bit_train_predictions.jsonl \
scripts/risk_vla/round3/run_02_export_candidate_bank_full.sh
```

No SOTA or PDMS improvement claim is made without same-split, same-protocol full benchmark results.
