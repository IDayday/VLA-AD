# Canonicalization Report

**Status:** `complete`

Canonical scene-level records: **37738**.

No experimental value, seed, scene identifier, or feasibility label is imputed by this script.

## Per-source ingestion

| Source | Input rows | Output rows | Status | Notes |
| --- | --- | --- | --- | --- |
| supplementary/derived/source_adapters/hard658.csv | 1316 | 1316 | ingested |  |
| supplementary/derived/source_adapters/v1_anchor.csv | 12138 | 12138 | ingested |  |
| supplementary/derived/source_adapters/v1_final.csv | 12138 | 12138 | ingested |  |
| supplementary/derived/source_adapters/v2_final.csv | 12146 | 12146 | ingested |  |

## Output schema

| Column | Pandas dtype |
| --- | --- |
| scene_id | string |
| split | string |
| benchmark | string |
| method | string |
| stage | string |
| variant | string |
| seed | Int64 |
| round | Int64 |
| sample_index | Int64 |
| aggregate_score | Float64 |
| NC | Float64 |
| DAC | Float64 |
| DDC | Float64 |
| TLC | Float64 |
| TTC | Float64 |
| EP | Float64 |
| C | Float64 |
| LK | Float64 |
| HC | Float64 |
| EC | Float64 |
| feasible | boolean |
| zero_score | boolean |
| initial_failure | boolean |
| recovered | boolean |
| new_failure | boolean |
| advantage | Float64 |
| positive_credit | boolean |
| pareto_front | boolean |
| reference_guard_pass | boolean |
| teacher_source | string |
| teacher_status | string |
| source_file | string |
| source_record | string |
