# Result Discrepancy Audit

This report keeps manuscript claims separate from later or differently configured artifacts. It is not an instruction to overwrite the paper.

## 1. Fixed 658-scene recovery set

The main paper's `367 -> 440` comparison is exactly reproduced by the archived pair labelled `original_stage3` and `old_pareto_v2` in `outputs/navtest_hard_stable_zero658_recogdrive_stage3_vs_pdms914505_20260727/summary.json`. The scene transition is 93 zero-to-positive repairs and 20 positive-to-zero regressions, hence a net repair of 73.

The later 91.45 checkpoint is also present in that audit and yields 415 positive-score scenes, 80 repairs, 32 regressions, and net repair 48. It is a different checkpoint and must not replace the historical pair in the paper's 658-scene argument. Following the manuscript protocol, all appendix recovery tables and figures use 367/440/73; the later result remains audit-only.

## 2. NAVSIM v1 aggregate row

The final aggregated CSV contains 12,139 rows, but one token is literally `average`; the remaining 12,138 are real scene tokens. Its adjacent `aggregate_summary.json` and `paired_analysis.json` therefore correctly record 12,138 paired scenes. The canonical adapter explicitly removes the summary row, and formula validation is performed on the 12,138 scene rows.

The scene mean is 91.450465 points. Main Table 1 prints 91.4 at one decimal,
whereas Table 5 prints 91.45 at two decimals. The supplement preserves both
paper displays and reports the reproduced mean at higher precision; it does not
rewrite either main-paper cell.

## 3. FF-PGRPO epoch count

The paper states 10 GRPO epochs. The archived `core_pareto_launch_config.txt` for one historical run records `max_epochs=20`. These may be different ablation/final runs; no manifest was found that proves the paper result used the 20-epoch launch. The supplement therefore uses 10 as the reported paper protocol and labels 20 as an archived non-canonical run.

## 4. GT-only PDMS

Main Table 2 reports GT-only imitation learning as 86.4, while Main Table 4 reports the no-stage baseline as 86.5. No scene export or documented protocol distinction was found. Both values are preserved with their table identity; the supplement does not average or silently reconcile them.

## 5. APR round lineage

Main Table 5 reports 91.10, 91.21, 91.37, and 91.45. Archived APR runbooks contain multiple accepted/rejected branches and a finer sequence (including 91.0834, 91.2766, 91.3331, 91.3707, 91.4153, and later candidates). Thus Table 5 is treated as the paper's compressed round-level report, not as a claim that every number is a direct parent-child checkpoint in a single filesystem run.

## 6. Abstract wording

`440/658 - 367/658 = 11.09` percentage points. The abstract phrase “11.1% higher” should read “11.1 percentage points higher” to match the body and avoid confusing absolute and relative improvement.

## 7. Camera protocol

The paper describes a multi-view camera input, while one audited APR resolved
command records `cam_type=single`. The exact final-checkpoint training manifest
is incomplete, so the supplement preserves the paper-level architecture and
discloses the resolved-command conflict rather than asserting that every result
row used either camera setting.
