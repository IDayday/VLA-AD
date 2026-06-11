# Last-VLA v2 Decoupled HighCap No-Risk PDMS Summary

Generated: `2026-06-08T02:03:21Z`

This report records the lightweight NAVTEST PDMS artifacts for the current
decoupled high-cap no-risk experiment round. Full prediction files, checkpoints,
and caches remain outside git under `/mnt/project/VLA-AD/outputs` and
`/mnt/project/VLA-AD/cache`.

## Full NAVTEST Results

All full NAVTEST rows below use `12146` samples, with `12138` valid PDM rows and
`8` missing metric-cache rows.

| Run | Checkpoint / stage | Trajectory key | PDMS | Trajectory L1 | Metrics artifact |
| --- | --- | --- | ---: | ---: | --- |
| Base 2B IL | `ReCogDrive-2B-IL` | `pred_traj` | `0.858575` | `0.260132` | `/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_navtest_cot_lora_eval_20260607T142428Z/eval/base_2b_il_full/metrics.json` |
| B LoRA direct-online | `epoch_003.ckpt` | `pred_coarse_traj` | `0.766702` | `0.660935` | `/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_navtest_cot_lora_eval_20260607T142428Z/eval/B_epoch003_lora_direct_online_full_shards/metrics.json` |
| A COT only | `cot_alignment/latest.ckpt` | `pred_coarse_traj` | `0.561656` | `2.082963` | `/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_navtest_cot_lora_eval_20260607T142428Z/eval/A_cot_latest_full_coarse/metrics.json` |
| A progressive SFT | `step_00080000.ckpt` | `pred_traj` | `0.374729` | `2.313641` | `/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_a_progressive_ckpt_eval_20260607T2237Z/eval/step_00080000_pred_traj/metrics.json` |
| A progressive SFT | `epoch=135-step=90304.ckpt` | `pred_traj` | `0.302877` | `2.613959` | `/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_a_progressive_ckpt_eval_20260607T2237Z/eval/epoch=135-step=90304_pred_traj/metrics.json` |

## Smoke / Diagnostic Results

| Run | Samples | Trajectory key | PDMS | Trajectory L1 | Notes |
| --- | ---: | --- | ---: | ---: | --- |
| Base 2B IL smoke | `16` | `pred_traj` | `0.906848` | `0.147440` | Small-sample diagnostic only. |
| B LoRA epoch003 smoke | `16` | `pred_coarse_traj` | `0.951351` | `0.613159` | Over-optimistic versus full NAVTEST. |
| A COT latest smoke coarse | `16` | `pred_coarse_traj` | `0.546875` | `1.667113` | Consistent with poor full result. |
| A progressive epoch135 smoke | `32` | `pred_traj` | `0.407825` | `1.896875` | Used to choose `pred_traj` over coarse for full sweep. |
| A progressive epoch135 smoke coarse | `32` | `pred_coarse_traj` | `0.295247` | `2.424109` | Worse than `pred_traj` smoke. |

## Current Interpretation

- The current best full NAVTEST result is still the original 2B IL baseline:
  `PDMS=0.858575`.
- B LoRA epoch003 direct-online is the best current experimental variant:
  `PDMS=0.766702`, below the baseline by about `0.0919`.
- A COT-only is substantially worse than baseline: `PDMS=0.561656`.
- A progressive SFT checkpoints evaluated so far are worse than A COT-only:
  `step_00080000` is `0.374729`, and `epoch=135-step=90304` is `0.302877`.
- The A progressive checkpoint sweep was launched with
  `configs/last_vla_v2/decoupled_highcap_no_risk/progressive_sft_eval_flat.yaml`
  to avoid the old `12`-token fallback in the raw YAML evaluator.
- At generation time, the A progressive sweep was still running on
  `step_00060000.ckpt`; additional rows should be appended after completion.
