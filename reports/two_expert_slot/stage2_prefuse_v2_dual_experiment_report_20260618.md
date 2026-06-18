# Stage2 Prefuse-v2 Dual Experiment Report

- Generated at: 2026-06-18T09:11:04+00:00
- Eval root: `/mnt/project/VLA-AD/outputs/two_expert_slot_stage2_prefuse_v2_dual_val6000_navtest_watch_20260617T020404Z`
- Common cache: `/mnt/project/VLA-AD/cache/two_expert_slot/stage1_v2_20260615_174154/hidden_navtrain_stage1_v2_clean_bf16_20260616T164153Z`
- Common Stage1 checkpoint: `/mnt/project/VLA-AD/experiments/two_expert_stage1_v2_20260615_174154/stage1_clean/clean_replay_8gpu_local_warmup_status_20260616T013758Z/stage1.ckpt`
- DiT initialization: random initialization, not A0 checkpoint initialization.
- Stage2 design: prefuse cross-attention fusion before the original ReCogDrive DiT, plus adaLN-Zero/dropout; no Stage3, no residual diffusion.
- Eval policy: per experiment val6000 checkpoint screening, then per experiment top checkpoints on full navtest. The two experiments are not merged for top-k selection.

## Executive Summary

| Experiment | Train samples | Target labels | Best val6000 | Best full navtest | Delta vs A0 0.864891 | Delta vs random-HMEF-v2 0.866397 | Verdict |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| GT-prefuse | 84,918 | NAVSIM GT | 0.923052 (`step_00120000.ckpt`) | 0.863598 (`step_00120000.ckpt`) | -0.001293 | -0.002799 | Not better than baselines |
| elite-full103k | 103,036 | elite-best-valid-above-GT-or-GT | 0.952659 (`step_00160000.ckpt`) | 0.872863 (`epoch=142-step=115115.ckpt`) | +0.007972 | +0.006466 | Strong positive |

Main conclusion: the architectural change alone is not enough. The GT-prefuse run underperforms A0/random-HMEF on full navtest, while the elite-target full103k run gives the first clear jump: best full navtest PDMS 0.872863, +0.007972 over A0 and +0.006466 over random-HMEF-v2. This strongly suggests the target trajectory quality / data mixture is the dominant improvement source in this round.

## Experiment Setup

### GT-prefuse
- Run root: `/mnt/project/VLA-AD/outputs/two_expert_slot_stage2_prefuse_v2_random_8gpu_minlr5e6_20260616T164153Z/stage2_train`
- Training command: `+experiment=two_expert_slot_stage2_dit_sft_prefuse_cross_attention`, `max_epochs=200`, `scheduler_min_lr=5e-6`, `seed=0`, 8 GPUs.
- Dataset from training log: 84,918 training samples, 18,118 validation samples.
- Final training status: `max_epochs=200 reached`. Final displayed loss: `train/loss_epoch=0.002`, `val/loss_epoch=0.013`.
- Checkpoint range: first saved `epoch=34-step=23240.ckpt`; latest extra step checkpoint `step_00120000.ckpt`.

### elite-full103k
- Run root: `/mnt/project/VLA-AD/outputs/two_expert_slot_stage2_prefuse_v2_elite_full103k_random_8gpu_minlr5e6_20260616T230916Z_jsonfix/stage2_train`
- Training command: `+experiment=two_expert_slot_stage2_dit_sft_prefuse_cross_attention_elite_fullcache`, `max_epochs=200`, `scheduler_min_lr=5e-6`, `cache_train_all_records=true`, `stage2_target_source=awac_elite_best_valid_above_gt_or_gt`, 8 GPUs.
- Elite target index: `/mnt/project/VLA-AD/outputs/two_expert_slot_stage2_prefuse_v2_elite_full103k_random_8gpu_minlr5e6_20260616T222308Z/artifacts/stage2_elite_best_valid_above_gt_targets.pt`
- Dataset from training log: 103,036 training samples, 18,118 validation samples.
- Final training status: `max_epochs=200 reached`. Final displayed loss: `train/loss_epoch=0.009`, `val/loss_epoch=0.002`.
- Checkpoint range: first step checkpoint `step_00050000.ckpt`; final `last.ckpt`.

## Result Coverage

| Result table | Count | Repo CSV |
| --- | ---: | --- |
| GT val6000 | 17 | `reports/two_expert_slot/stage2_prefuse_v2_dual_results_20260618/gt_prefuse_val6000_summary.csv` |
| GT navtest | 10 | `reports/two_expert_slot/stage2_prefuse_v2_dual_results_20260618/gt_prefuse_navtest_summary.csv` |
| Elite val6000 | 61 | `reports/two_expert_slot/stage2_prefuse_v2_dual_results_20260618/elite_full103k_val6000_summary.csv` |
| Elite navtest | 27 | `reports/two_expert_slot/stage2_prefuse_v2_dual_results_20260618/elite_full103k_navtest_summary.csv` |

The original live evaluator outputs remain under `/mnt/project/VLA-AD/outputs/two_expert_slot_stage2_prefuse_v2_dual_val6000_navtest_watch_20260617T020404Z`.

## GT-prefuse Results

### GT val6000 Top 12

| Rank | Checkpoint | PDMS | traj_l1 | Completed |
| --- | --- | --- | --- | --- |
| 1 | step_00120000.ckpt | 0.923052 | 0.124221 | 2026-06-18T01:47:03+00:00 |
| 2 | step_00100000.ckpt | 0.922154 | 0.132802 | 2026-06-17T19:56:26+00:00 |
| 3 | epoch=82-step=55112.ckpt | 0.917251 | 0.173023 | 2026-06-17T12:06:19+00:00 |
| 4 | last.ckpt | 0.915691 | 0.173491 | 2026-06-17T17:16:59+00:00 |
| 5 | epoch=102-step=68392.ckpt | 0.915160 | 0.173134 | 2026-06-17T16:31:56+00:00 |
| 6 | step_00080000.ckpt | 0.913802 | 0.162948 | 2026-06-17T17:16:59+00:00 |
| 7 | epoch=84-step=56440.ckpt | 0.912659 | 0.195458 | 2026-06-17T12:55:31+00:00 |
| 8 | step_00060000.ckpt | 0.911730 | 0.173731 | 2026-06-17T17:16:59+00:00 |
| 9 | epoch=63-step=42496.ckpt | 0.910896 | 0.235455 | 2026-06-17T10:22:24+00:00 |
| 10 | epoch=62-step=41832.ckpt | 0.906450 | 0.204237 | 2026-06-17T09:33:41+00:00 |
| 11 | epoch=64-step=43160.ckpt | 0.904662 | 0.227629 | 2026-06-17T11:10:06+00:00 |
| 12 | epoch=51-step=34528.ckpt | 0.897865 | 0.231909 | 2026-06-17T07:43:15+00:00 |

### GT full navtest Top 10

| Rank | Checkpoint | PDMS | source_val | traj_l1 | Completed |
| --- | --- | --- | --- | --- | --- |
| 1 | step_00120000.ckpt | 0.863598 | 0.923052 | 0.261812 | 2026-06-18T03:15:51+00:00 |
| 2 | step_00100000.ckpt | 0.862615 | 0.922154 | 0.262203 | 2026-06-17T20:13:40+00:00 |
| 3 | epoch=102-step=68392.ckpt | 0.859000 | 0.915160 | 0.272667 | 2026-06-17T18:02:49+00:00 |
| 4 | last.ckpt | 0.858144 | 0.915691 | 0.272826 | 2026-06-17T18:02:49+00:00 |
| 5 | epoch=82-step=55112.ckpt | 0.856601 | 0.917251 | 0.263112 | 2026-06-17T13:32:57+00:00 |
| 6 | epoch=63-step=42496.ckpt | 0.852901 | 0.910896 | 0.302301 | 2026-06-17T12:16:58+00:00 |
| 7 | epoch=84-step=56440.ckpt | 0.850357 | 0.912659 | 0.283638 | 2026-06-17T13:10:46+00:00 |
| 8 | epoch=62-step=41832.ckpt | 0.849150 | 0.906450 | 0.287298 | 2026-06-17T10:53:43+00:00 |
| 9 | epoch=51-step=34528.ckpt | 0.845078 | 0.897865 | 0.301719 | 2026-06-17T09:33:14+00:00 |
| 10 | epoch=21-step=14608.ckpt | 0.834807 | 0.893225 | 0.329394 | 2026-06-17T08:04:57+00:00 |

Analysis: GT-prefuse improves steadily through the late explicit step checkpoints; `step_00120000` is best on both val6000 and navtest. However, the absolute navtest score 0.863598 is still below A0 official 0.864891 and random-HMEF-v2 0.866397. The val6000 improvement is real but does not cross the full-navtest baseline threshold.

## elite-full103k Results

### Elite val6000 Top 15

| Rank | Checkpoint | PDMS | traj_l1 | Completed |
| --- | --- | --- | --- | --- |
| 1 | step_00160000.ckpt | 0.952659 | 0.236277 | 2026-06-18T08:09:45+00:00 |
| 2 | epoch=196-step=158585.ckpt | 0.951970 | 0.241483 | 2026-06-18T08:09:44+00:00 |
| 3 | epoch=199-step=161000.ckpt | 0.951937 | 0.234220 | 2026-06-18T08:09:45+00:00 |
| 4 | epoch=194-step=156975.ckpt | 0.951499 | 0.236930 | 2026-06-18T08:09:44+00:00 |
| 5 | epoch=164-step=132825.ckpt | 0.950588 | 0.244358 | 2026-06-18T03:50:48+00:00 |
| 6 | epoch=142-step=115115.ckpt | 0.950543 | 0.261287 | 2026-06-18T01:47:04+00:00 |
| 7 | epoch=178-step=144095.ckpt | 0.950139 | 0.249610 | 2026-06-18T06:51:26+00:00 |
| 8 | epoch=155-step=125580.ckpt | 0.949541 | 0.234791 | 2026-06-18T03:50:48+00:00 |
| 9 | epoch=167-step=135240.ckpt | 0.948969 | 0.230471 | 2026-06-18T04:45:47+00:00 |
| 10 | epoch=161-step=130410.ckpt | 0.948895 | 0.237825 | 2026-06-18T03:50:48+00:00 |
| 11 | step_00140000.ckpt | 0.948663 | 0.232252 | 2026-06-18T08:09:45+00:00 |
| 12 | epoch=176-step=142485.ckpt | 0.948537 | 0.243236 | 2026-06-18T06:13:31+00:00 |
| 13 | epoch=158-step=127995.ckpt | 0.947967 | 0.251416 | 2026-06-18T03:38:53+00:00 |
| 14 | epoch=173-step=140070.ckpt | 0.947825 | 0.245946 | 2026-06-18T05:37:46+00:00 |
| 15 | epoch=149-step=120750.ckpt | 0.947301 | 0.238421 | 2026-06-18T02:13:09+00:00 |

### Elite full navtest Top 15

| Rank | Checkpoint | PDMS | source_val | traj_l1 | Completed |
| --- | --- | --- | --- | --- | --- |
| 1 | epoch=142-step=115115.ckpt | 0.872863 | 0.950543 | 0.350992 | 2026-06-18T03:15:51+00:00 |
| 2 | epoch=199-step=161000.ckpt | 0.868685 | 0.951937 | 0.335159 | 2026-06-18T08:35:27+00:00 |
| 3 | epoch=149-step=120750.ckpt | 0.868268 | 0.947301 | 0.333365 | 2026-06-18T03:15:51+00:00 |
| 4 | epoch=132-step=107065.ckpt | 0.867640 | 0.947022 | 0.341883 | 2026-06-18T00:33:09+00:00 |
| 5 | epoch=117-step=94990.ckpt | 0.867591 | 0.941017 | 0.322202 | 2026-06-17T20:53:40+00:00 |
| 6 | epoch=196-step=158585.ckpt | 0.867317 | 0.951970 | 0.342228 | 2026-06-18T08:24:25+00:00 |
| 7 | epoch=164-step=132825.ckpt | 0.866925 | 0.950588 | 0.344391 | 2026-06-18T05:05:50+00:00 |
| 8 | epoch=155-step=125580.ckpt | 0.866721 | 0.949541 | 0.331784 | 2026-06-18T04:16:12+00:00 |
| 9 | epoch=52-step=42665.ckpt | 0.866470 | 0.932261 | 0.332311 | 2026-06-17T12:52:40+00:00 |
| 10 | step_00160000.ckpt | 0.866282 | 0.952659 | 0.337243 | 2026-06-18T08:24:25+00:00 |
| 11 | epoch=128-step=103845.ckpt | 0.865893 | 0.944563 | 0.337959 | 2026-06-17T22:57:14+00:00 |
| 12 | epoch=104-step=84525.ckpt | 0.865333 | 0.939054 | 0.344216 | 2026-06-17T19:07:11+00:00 |
| 13 | epoch=129-step=104650.ckpt | 0.865295 | 0.943525 | 0.340000 | 2026-06-17T22:57:14+00:00 |
| 14 | step_00100000.ckpt | 0.865124 | 0.943537 | 0.332986 | 2026-06-17T21:57:49+00:00 |
| 15 | epoch=134-step=108675.ckpt | 0.864336 | 0.945474 | 0.355750 | 2026-06-18T00:33:09+00:00 |

Analysis: elite-full103k is the successful run. The best full navtest checkpoint is not the best val6000 checkpoint: `epoch=142-step=115115.ckpt` gives full navtest 0.872863, while later checkpoints such as `step_00160000.ckpt` and `epoch=199-step=161000.ckpt` have higher val6000 but lower navtest. This means val6000 remains useful for screening, but late-stage val gains are not monotonic evidence of navtest gains.

## Val6000 vs Navtest Reliability

- GT-prefuse: val6000 and navtest agree on the best checkpoint (`step_00120000`), but the final absolute navtest remains weak.
- elite-full103k: val6000 top checkpoints after roughly `epoch=164` do not dominate navtest. The best navtest checkpoint appears earlier (`epoch=142-step=115115`).
- Practical implication: keep val6000 as a cheap filter, but do not select final ckpt solely by val6000 after training passes roughly 110k-130k steps. For this run, navtest is the only reliable final selector.

## Interpretation

1. The Stage1-v2 cache and prefuse architecture are usable, but not sufficient by themselves: GT-only Stage2 is below prior baselines.
2. The elite target replacement is the largest factor. Training on labels that are at least GT or better from the buffer shifts full navtest from 0.863598 to 0.872863 under the same general Stage2 architecture/cache family.
3. Late training appears to over-optimize the validation-screening distribution or the elite target distribution. The best val6000 score 0.952659 (`step_00160000`) only gives navtest 0.866282, much worse than `epoch=142` despite higher val6000.
4. The best elite checkpoint has higher trajectory L1 than some lower-PDMS checkpoints, confirming PDMS is not reducible to trajectory L1; route/rule/safety components matter.
5. The current best result, 0.872863, is a meaningful improvement over both A0 official and random-HMEF-v2, large enough to justify preserving this code/config/cache/checkpoint lineage.

## Recommended Checkpoints

| Purpose | Checkpoint | Reason |
| --- | --- | --- |
| Primary report / next-stage candidate | `epoch=142-step=115115.ckpt` from elite-full103k | Best full navtest PDMS 0.872863 |
| Backup candidate | `epoch=149-step=120750.ckpt` from elite-full103k | Second/third strong navtest 0.868268, still above baselines |
| Do not select by val alone | `step_00160000.ckpt` from elite-full103k | Best val6000 0.952659 but navtest only 0.866282 |
| GT-only reference | `step_00120000.ckpt` from GT-prefuse | Best GT-prefuse navtest 0.863598; useful negative control |

## Operational Notes

- Evaluation watcher used per-experiment independent top selection. Results were not merged across GT and elite when choosing navtest jobs.
- Local and zt3 cooperative eval used shared shard claim files to avoid duplicate shard execution.
- At the latest snapshot there were no partial shard results pending; local/zt3 GPUs were idle.
- One local nav wave had SIGTERM failures for `epoch=161-step=130410`; the scheduler was patched to continue after wave failures and restarted. Completed aggregate tables are intact.

## Next Actions

1. Freeze and back up elite `epoch=142-step=115115.ckpt` as the current best Stage2 candidate.
2. Run expert-corruption/full ablations on the best elite checkpoint: normal, raw_vlm_only, zero_all, zero_h_dyn, zero_h_geo, dyn_only, geo_only, random_slots.
3. Treat `epoch=142` as the selection point for this recipe; do not continue selecting later checkpoints based on val6000 alone.
4. For the next run, keep the simple prefuse architecture, but focus on target construction/regularization and navtest-correlated selection rather than adding more DiT complexity.
