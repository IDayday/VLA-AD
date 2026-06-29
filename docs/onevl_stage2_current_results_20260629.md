# OneVL AR Answer Stage2 Current Results - 2026-06-29

This note records the current repository-tracked OneVL AR Answer -> ReCogDrive
DiT stage2 workflow and the latest corrected evaluation results. Runtime
caches, checkpoints, logs, and metric files are intentionally kept outside git.

## Active Code Paths

Stage1 AR Answer prompt4hist:

- `scripts/onevl/prepare_ar_answer_prompt4hist_dataset.py`
- `scripts/onevl/run_answer_training_full.sh`
- `scripts/onevl/launch_answer_training_full.sh`
- `scripts/onevl/run_ar_answer_infer_eval_full.sh`
- `scripts/onevl/navsim_predictions_to_submission.py`

Stage2 cache and training:

- `scripts/onevl/bridge_ar_answer_to_recogdrive_dit.py`
- `scripts/onevl/run_ar_answer_prompt4hist_stage2_cache_then_train.sh`
- `scripts/onevl/watch_prompt4hist_stage2_cache_then_train.sh`
- `scripts/train_recogdrive_expert_chunked.py`
- `configs/onevl_ar_answer_stage2_small.yaml`

Stage2 evaluation:

- `scripts/onevl/build_prompt4hist_stage2_eval_caches.sh`
- `scripts/onevl/watch_onevl_stage2_eval_top5_remote.py`
- `scripts/onevl/run_navtest_epoch_range_parallel.py`
- `scripts/eval_recogdrive_expert_pdm.py`

## Cache Alignment Findings

Training cache:

- Prompt command vs cached `high_command_one_hot` / `status_feature` command
  was checked on the first 1024 train samples with zero mismatches.
- Train prompt command distribution:
  - `MOVE FORWARD`: 65494
  - `TURN LEFT`: 25966
  - `TURN RIGHT`: 11828
- `history_trajectory` shape is `[4, 3]`.
- The last history row is the current ego-local state `[0, 0, 0]`, matching the
  ReCogDrive-style history contract.

Val6000 cache:

- Old val6000 prompt command vs cache command was checked on the first 1024
  samples with zero mismatches.
- A corrected scene-prompt val6000 cache was generated for consistency checks.

Navtest cache:

- The old navtest JSON row prompts were wrong: all 12146 prompts used
  `MOVE FORWARD`.
- Cached planner tensors/status still contained left/right commands, so prompt
  hidden and planner command tensors were inconsistent.
- First 1024 old navtest samples had 485 prompt-vs-cache command mismatches.
- A corrected navtest scene-prompt cache was generated with
  `prompt_source=scene` and `prompt_scene_alignment_pass=true`.

Important interpretation:

- The training cache is not currently proven wrong.
- The old navtest evaluation cache was wrong and should not be used for current
  conclusions.
- The corrected navtest scene cache is the current cache for stage2 navtest
  reporting.

## Current Artifact Paths

Stage1 prompt4hist checkpoint:

```text
/mnt/project/onevl_navsim_exp/answer_bs64_prompt4hist_20260627_160112/swift_output/v0-20260627-160133/checkpoint-3228
```

Stage2 train root:

```text
/mnt/project/onevl_navsim_exp/ar_answer_prompt4hist_stage2_apsd_cache_20260628T052126Z/train_full200
```

Corrected val6000 scene cache:

```text
/mnt/project/onevl_navsim_exp/ar_answer_prompt4hist_stage2_val6000_scene_cache_20260629T120638Z/cache_val6000
```

Corrected navtest scene cache:

```text
/mnt/project/onevl_navsim_exp/ar_answer_prompt4hist_stage2_navtest_scene_cache_20260629T124549Z/cache_navtest
```

Full epoch 120-200 navtest evaluation output:

```text
/mnt/project/onevl_navsim_exp/ar_answer_prompt4hist_stage2_navtest_scene_epoch120_200_8shard4q_20260629T145209Z
```

## Stage1 Baseline

Stage1 AR Answer prompt4hist navtest after abnormal-output repair:

```text
PDMS = 0.8641601403126207
```

## Corrected Val6000 Scene Evaluations

Only selected checkpoints were evaluated on the corrected val6000 scene cache.
The currently recorded best selected checkpoint is `epoch_200`.

| checkpoint | PDMS |
| --- | ---: |
| epoch_200 | 0.896627809020 |
| epoch_119 | 0.893325548592 |
| epoch_107 | 0.889908137975 |
| epoch_115 | 0.886434678555 |
| epoch_109 | 0.880186901498 |
| epoch_102 | 0.877690792556 |

## Corrected Navtest Scene Evaluation

All checkpoints from `epoch_120` through `epoch_200` were evaluated with the
corrected navtest scene cache. The latest full-range run used two servers:

- local: `epoch_120` through `epoch_160`
- `training-rl-zt3`: `epoch_161` through `epoch_200`

Each server ran 4 checkpoints at a time. Each checkpoint was split into 8 PDM
shards.

Summary:

```text
checkpoint count: 81
epoch range: 120-200
failed checkpoints: 0
num_samples per checkpoint: 12146
num_pdm_valid per checkpoint: 12138
num_pdm_missing_metric_cache per checkpoint: 8
PDMS min: 0.761708292954
PDMS mean: 0.838581128660
PDMS max: 0.857763505657
```

Top corrected navtest checkpoints:

| rank | checkpoint | PDMS | NC | DAC | TTC | EP | DDC |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | epoch_147 | 0.857763505657 | 0.962432 | 0.953534 | 0.897594 | 0.835372 | 0.976891 |
| 2 | epoch_178 | 0.856391511668 | 0.962844 | 0.951145 | 0.898089 | 0.835676 | 0.976644 |
| 3 | epoch_139 | 0.856087774087 | 0.963915 | 0.951228 | 0.900560 | 0.830707 | 0.976602 |
| 4 | epoch_138 | 0.854892415167 | 0.963956 | 0.948756 | 0.902949 | 0.829473 | 0.975490 |
| 5 | epoch_169 | 0.854807376584 | 0.963627 | 0.950816 | 0.891827 | 0.836761 | 0.977715 |
| 6 | epoch_192 | 0.854059847283 | 0.962473 | 0.949333 | 0.898006 | 0.832272 | 0.978827 |
| 7 | epoch_184 | 0.853909037557 | 0.962803 | 0.949086 | 0.893805 | 0.835793 | 0.977591 |
| 8 | epoch_183 | 0.853401127571 | 0.961361 | 0.949745 | 0.894381 | 0.834087 | 0.976396 |
| 9 | epoch_187 | 0.852761696717 | 0.961897 | 0.948262 | 0.896688 | 0.831431 | 0.977756 |
| 10 | epoch_182 | 0.852741605768 | 0.962885 | 0.947932 | 0.897759 | 0.830675 | 0.976973 |

Ten-epoch trend:

| epoch range | mean PDMS | best checkpoint | best PDMS |
| --- | ---: | --- | ---: |
| 120-129 | 0.817445 | epoch_127 | 0.846609 |
| 130-139 | 0.823292 | epoch_139 | 0.856088 |
| 140-149 | 0.838853 | epoch_147 | 0.857764 |
| 150-159 | 0.838322 | epoch_159 | 0.849419 |
| 160-169 | 0.844571 | epoch_169 | 0.854807 |
| 170-179 | 0.845266 | epoch_178 | 0.856392 |
| 180-189 | 0.849759 | epoch_184 | 0.853909 |
| 190-199 | 0.849730 | epoch_192 | 0.854060 |
| 200 | 0.852687 | epoch_200 | 0.852687 |

## Current Conclusion

The corrected navtest evaluation is much better than the earlier broken
navtest-cache results, but the best stage2 checkpoint is still slightly below
the repaired stage1 AR Answer navtest baseline:

```text
stage1 AR Answer prompt4hist navtest: 0.864160140313
best corrected stage2 navtest:      0.857763505657
gap:                                -0.006396634656
```

The strongest current stage2 checkpoint for navtest is `epoch_147`. Late
checkpoints from `epoch_178` through `epoch_200` are more stable near 0.85 but
do not exceed `epoch_147`.

