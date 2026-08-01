# Data Validation Report

**Status:** `valid`

- Canonical rows: 37738
- Unique scenes: 12146
- Errors: 0
- Warnings: 1
- Input SHA-256: `010814a9b9f63575b517b9c541c1e54ec81b74ccba681af4d09d791c637cf244`


## Checks

| Severity | Check | Count | Message |
| --- | --- | --- | --- |
| pass | schema |  | All required canonical columns are present. |
| pass | scene_id |  | Every row has an explicit scene identifier. |
| pass | duplicates |  | No duplicate canonical identity keys. |
| pass | metric_ranges |  | All observed metrics lie in [0.0, 100.0]. |
| pass | state_logic |  | Observed state and credit flags are internally consistent. |
| pass | aggregate_formula:navsim_v1_pdms |  | Formula matches 25592 comparable rows. |
| pass | aggregate_formula:navsim_v2_epdms |  | Formula matches 10040 comparable rows. |
| pass | scene_count:v1_final_scored |  | Observed expected 12138 scenes. |
| pass | scene_count:v1_paired_anchor |  | Observed expected 12138 scenes. |
| pass | scene_count:v2_final |  | Observed expected 12146 scenes. |
| pass | scene_count:hard658_scalar |  | Observed expected 658 scenes. |
| pass | scene_count:hard658_paper_ampt |  | Observed expected 658 scenes. |
| pass | scene_set:final_vs_anchor_full_navtest |  | All selectors share 12138 scenes. |
| pass | scene_set:paper_hard658_pair |  | All selectors share 658 scenes. |
| warning | candidate_matching |  | No candidate-count/acceptance-rate matching checks are configured. |
| pass | paper_claim_means |  | All 17 matched claims agree within 0.06. |

## Seed and round coverage

| benchmark | split | method | stage | variant | rows | unique_scenes | seeds | rounds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NAVSIM v1 | navtest | AMPT | APR | paired_anchor | 12138 | 12138 | 1 | 0 |
| NAVSIM v1 | navtest | AMPT | APR | paper_final | 12138 | 12138 | 1 | 1 |
| NAVSIM v1 | navtest | AMPT | FF-PGRPO | paper_recovery_checkpoint | 658 | 658 | 0 | 0 |
| NAVSIM v1 | navtest | scalar GRPO | FF-PGRPO | scalar_grpo | 658 | 658 | 0 | 0 |
| NAVSIM v2 | navtest | AMPT | APR | paper_final | 12146 | 12146 | 0 | 1 |

## Per-run scene counts

| benchmark | split | method | stage | variant | seed | round | unique_scenes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NAVSIM v1 | navtest | AMPT | APR | paired_anchor | 20260726 |  | 12138 |
| NAVSIM v1 | navtest | AMPT | APR | paper_final | 20260726 | 3 | 12138 |
| NAVSIM v1 | navtest | AMPT | FF-PGRPO | paper_recovery_checkpoint |  |  | 658 |
| NAVSIM v1 | navtest | scalar GRPO | FF-PGRPO | scalar_grpo |  |  | 658 |
| NAVSIM v2 | navtest | AMPT | APR | paper_final |  | 3 | 12146 |
