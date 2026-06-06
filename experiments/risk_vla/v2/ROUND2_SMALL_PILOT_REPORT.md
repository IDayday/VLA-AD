# RISK-VLA v2 Round2 Small Pilot Report

Status: not run.

Reason: no explicit matched train/val candidate inputs were provided or discoverable for a valid small pilot. The workspace contains several standalone `pdm_results.csv` files, but not a matched-token A0/Base table plus B3/BiT and RISK-VLA candidate tables with a declared train/val split. I did not infer private paths or reuse navtest/test labels.

Required inputs to run the first real pilot:
- `A0_PDM_TABLE`: train/val matched PDM CSV or JSONL for the baseline.
- `CANDIDATE_PDM_TABLES`: semicolon-separated candidate specs such as `name=B3,path=/abs/path/pdm.csv,strategy=path_intent`.
- Candidate trajectory cache or explicit trajectory source paths for candidate-bank export/eval.
- Checkpoint paths for any learned critic/router training stage.

Current generated artifacts:
- `ROUND2_DRYRUN_REPORT.md`
- `round2_plan.json`
- `risk_vla_v2_results_summary.json`
- `risk_vla_v2_results_summary.md`

No PDMS, EPDMS, NC/TTC/DAC transition, or scorer accuracy numbers are claimed in this report.
