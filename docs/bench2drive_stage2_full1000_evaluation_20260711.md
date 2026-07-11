# Bench2Drive Stage2 Full-1000 Evaluation — 2026-07-11

## Decision

Stage2 training completed normally, but the resulting policy **does not pass
the closed-loop reproduction gate**. The complete 220-route run has no missing
routes, yet Driving Score, strict success, and mean multi-ability are well below
the final ReCogDrive numbers reported by the authors.

Stage3 remains deferred. These results should first be diagnosed against the
released IL checkpoint and the strict serving configuration rather than
assuming that GRPO alone will close the gap.

## Artifacts and timing

- Stage1 VLM:
  `outputs/bench2drive_recogdrive_stage1_sft_20260710T065201Z`
- Stage2 checkpoint:
  `outputs/bench2drive_recogdrive_stage2_scratch_2b_full1000_20260710T093210Z/latest.ckpt`
- Evaluation root:
  `outputs/bench2drive_recogdrive_closed_loop/stage2_scratch_full1000_epoch200_220route_20260711T000033Z`
- Merged route result: `route_json/merged.json`
- Multi-ability result: `route_json/merged_ability.json`
- Efficiency/smoothness result: `efficiency_smoothness.log`

The launch began at 00:00:59 UTC. The final route shard finished at about
04:17:17 UTC, so the 220-route rollout took approximately 4 hours 16 minutes,
including server warm-up, staggered worker start, and automatic CARLA retries.
Metric post-processing finished at 04:40:32 UTC; the end-to-end wall time was
approximately 4 hours 40 minutes. Seven CARLA worker crashes were automatically
retried, and all eight shards reached their expected 27 or 28 routes.

The run used eight inference servers and eight CARLA workers, the accelerated
`bench2drive_recogdrive_closed_loop.remote.yaml` serving profile, the epoch-200
Stage2 checkpoint, and the paired Bench2Drive Stage1 VLM. All 353 Stage2 state
keys loaded with no missing, unexpected, or shape-mismatched keys.

## Closed-loop metrics

| Metric | Result |
|---|---:|
| Evaluated routes | 220 / 220 |
| Driving Score | 45.11 |
| Strict success | 21.36% (47 / 220) |
| Mean route completion | 72.55% |
| Mean route penalty | 0.6289 |
| Efficiency | 137.51 |
| Comfort / smoothness | 37.71% (raw 0.37714) |

The official merge script defines success as a `Completed`/`Perfect` route with
no non-min-speed infraction. Raw route status is therefore less strict than the
reported success rate:

| Route status | Count | Share |
|---|---:|---:|
| Completed | 107 | 48.64% |
| Failed — TickRuntime | 81 | 36.82% |
| Failed — agent blocked | 32 | 14.55% |

In this Bench2Drive evaluator, `TickRuntime` means the route exceeded 4,000
simulation ticks. It is a scored route failure rather than a missing result or
an inference-server crash.

## Multi-ability metrics

| Ability | Successes / denominator | Result |
|---|---:|---:|
| Merging | 15 / 80 | 18.75% |
| Overtaking | 2 / 45 | 4.44% |
| Emergency brake | 13 / 60 | 21.67% |
| Give way | 5 / 10 | 50.00% |
| Traffic signs | official junction-aware calculation | 48.95% |
| Mean | — | 28.76% |

## Infraction diagnostics

| Infraction | Routes affected | Events |
|---|---:|---:|
| Vehicle collision | 89 (40.45%) | 127 |
| Layout collision | 58 (26.36%) | 112 |
| Pedestrian collision | 12 (5.45%) | 16 |
| Outside route lanes | 75 (34.09%) | 75 |
| Stop-sign infraction | 13 (5.91%) | 13 |
| Red-light infraction | 1 (0.45%) | 1 |
| Failed to yield to emergency vehicle | 5 (2.27%) | 5 |
| Vehicle blocked | 32 (14.55%) | 32 |
| Min-speed infraction | 214 (97.27%) | 2,936 |

Route completion is strongly bimodal: 107 routes reached 100%, while 66 routes
ended below 50% completion. The median completion was 84.62%.

## Comparison with the reported ReCogDrive result

The authors report their final ReCogDrive result at
<https://github.com/xiaomi-research/recogdrive#checkpoint>. Values below use
the same official Bench2Drive aggregation scripts and table scaling.

| Metric | This Stage2 | Reported ReCogDrive | Absolute difference |
|---|---:|---:|---:|
| Efficiency | 137.51 | 138.18 | -0.67 |
| Comfort | 37.71 | 17.45 | +20.26 |
| Success | 21.36 | 45.45 | -24.09 pp |
| Driving Score | 45.11 | 71.36 | -26.25 |
| Merging | 18.75 | 29.73 | -10.98 pp |
| Overtaking | 4.44 | 20.00 | -15.56 pp |
| Emergency brake | 21.67 | 69.09 | -47.42 pp |
| Give way | 50.00 | 20.00 | +30.00 pp |
| Traffic signs | 48.95 | 71.34 | -22.39 pp |
| Multi-ability mean | 28.76 | 42.03 | -13.27 pp |

Driving Score is 36.8% below the reported value, strict success is 53.0% below,
and multi-ability mean is 31.6% below. The strongest deficit is emergency
braking, followed by overtaking. Give-way and comfort are higher, but comfort
must be interpreted cautiously: slow, blocked, or prematurely terminated
motion can satisfy acceleration/jerk thresholds while still failing the route.
Likewise, near-official efficiency does not offset the fact that 214 routes
recorded at least one min-speed infraction.

## Comparability limits and next diagnostics

This is a useful gap measurement, not an exact stage-equivalent reproduction:

1. This checkpoint is Stage1 + scratch Stage2 imitation learning. The reported
   row is the paper's final ReCogDrive system, whose framework includes
   DiffGRPO safety/comfort reinforcement.
2. The official repository describes mixed-data/real-world adaptation followed
   by Bench2Drive-Traj and Bench2Drive-QA training. This run adapted the released
   2B VLM to the local trajectory contract, then fit Stage2 on all 1,000 local
   clips; it did not reproduce the official mixed QA recipe.
3. The run used the accelerated 2 Hz visual-refresh serving profile. The
   authors have not released an identical Bench2Drive evaluation wrapper, so
   simulator/agent details are not guaranteed to match their table.
4. All 1,000 clips were used for Stage2 fitting, so the former 50-clip split
   cannot be used for unbiased checkpoint selection. Epoch 200 was evaluated;
   the training `best.ckpt` is merely the minimum single-batch training loss.

Before any Stage3 work, run three bounded diagnostics: compare our epoch-200
and epoch-147 checkpoints on Dev10, evaluate the released ReCogDrive IL
checkpoint through this exact wrapper, and compare accelerated versus strict
serving on the same Dev10 routes. Then inspect command-conditioned trajectory
scale and collision-heavy scenarios, especially emergency braking and
overtaking.
