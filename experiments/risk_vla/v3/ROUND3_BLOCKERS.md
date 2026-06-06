# RISK-VLA v3 Round3 Blockers

Date: 2026-06-06

## Current Status

Full input discovery completed. Formal downstream training/evaluation was not launched because the manifest marks the required train/validation inputs as not launchable under the >=10k and no-leakage rules.

## Blocking Inputs

- Held-out validation chunk cache is below formal scale: 1024 tokens, minimum 10000.
- There are not at least two train-split matched PDM tables with >=10000 rows. Strategy utility labels require A0/Base plus at least one candidate method on matched train tokens.
- Existing >=10000 PDM tables are mostly navtest/test analysis assets. They cannot be used for training labels, router supervision, threshold tuning, hard-negative mining, or VLM instruction data.
- Existing large train chunk caches are present, but corresponding multi-method train predictions/PDM evaluations are not present at >=10000 matched tokens.

## Remedies

- Materialize/evaluate A0/Base train predictions on >=10000 train tokens.
- Materialize/evaluate at least one candidate method on the same >=10000 train tokens: BiT/B3, D5, RISK-VLA v1/v2, or candidate-bank K=8.
- Build a held-out val split with >=10000 tokens, or register an existing val asset with explicit split provenance. Do not substitute navtest.
- Re-run `scripts/risk_vla/round3/run_00_discover_full_inputs.sh` after adding assets, then launch `run_01` to `run_05`.

## Leakage Audit

Navtest/test PDM labels discovered in this run are analysis-only. They were not used for training labels, router/critic supervision, threshold tuning, safe-alignment pair mining, or VLM instruction construction.
