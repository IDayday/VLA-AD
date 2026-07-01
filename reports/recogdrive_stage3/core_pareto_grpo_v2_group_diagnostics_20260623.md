# Core-Pareto GRPO v2 Group Diagnostics

- Run: `stage3_grpo_core_pareto_s16_lr1e4_b2acc4_20e_local8_20260616T020136Z`
- TensorBoard event: `/mnt/project/VLA-AD/outputs/stage3_grpo_core_pareto_s16_lr1e4_b2acc4_20e_local8_20260616T020136Z/train/hydra/training_recogdrive_agent/2026.06.16.02.04.42/lightning_logs/version_0/events.out.tfevents.1781575642.training-vla-zt-worker-0.4148042.0`
- Navtest table: `/mnt/project/VLA-AD/outputs/stage3_grpo_core_pareto_s16_lr1e4_b2acc4_20e_local8_20260616T020136Z/navtest_pdms_analysis.tsv`
- Evaluated checkpoints aligned: 91
- Best checkpoint: step 21600, PDMS 0.910274164, Core 0.923720536

## Key Findings

- Valid candidate ratio stayed high: navtest-aligned mean 0.9083; final epoch 0.9212. With sample_time=16, this is about 14.74/16 valid samples per scene in the final epoch.
- Valid+EP-floor progress ratio was lower than pure validity: navtest-aligned mean 0.6984; final epoch 0.7338. This shows the EP floor actively filtered conservative samples rather than being a no-op.
- All-invalid groups were almost absent: max aligned 0.0625, final epoch 0.0010. All-slow groups also stayed low: max aligned 0.1875, final epoch 0.0167.
- Pareto front was consistently non-trivial: aligned mean 0.3552; final epoch 0.3722. Pareto-front candidates received positive advantage frequently: aligned mean 0.9481.
- Slow-fail positive advantage was exactly suppressed in the logged run: aligned mean 0.0000; final epoch 0.0000. This supports that the EP-floor cap worked as intended.
- Important caveat: most groups became all-valid in the late stage, so the method increasingly acts as ranking among valid trajectories by Core/Pareto/EP tradeoff, not as a safety-rescue mechanism. The remaining hard part is making those within-valid rankings translate to sampled navtest trajectories.

## Phase Means From Step Scalars

| phase | base_reward | pdms_core | safe_count/16 | valid | valid+EP | mixed | all_valid | all_invalid | all_slow | pareto | pareto+adv | slowfail+adv | group_w | score_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0-3k | 0.7933 | 0.8172 | 14.03 | 0.8770 | 0.6271 | 0.4865 | 0.5052 | 0.0083 | 0.0208 | 0.3089 | 0.9484 | 0.0000 | 0.9776 | 0.4832 |
| 3k-5k | 0.8070 | 0.8296 | 14.12 | 0.8827 | 0.6511 | 0.5047 | 0.4922 | 0.0031 | 0.0281 | 0.3299 | 0.9660 | 0.0000 | 0.9750 | 0.4697 |
| 5k-13.3k | 0.8277 | 0.8419 | 14.43 | 0.9022 | 0.7020 | 0.4605 | 0.5377 | 0.0019 | 0.0241 | 0.3539 | 0.9525 | 0.0000 | 0.9786 | 0.4487 |
| 13.3k-21.6k | 0.8387 | 0.8505 | 14.60 | 0.9126 | 0.7207 | 0.4130 | 0.5855 | 0.0015 | 0.0203 | 0.3657 | 0.9551 | 0.0000 | 0.9804 | 0.4185 |
| 21.6k-26.6k | 0.8484 | 0.8589 | 14.73 | 0.9207 | 0.7365 | 0.4075 | 0.5919 | 0.0006 | 0.0112 | 0.3777 | 0.9557 | 0.0000 | 0.9861 | 0.4033 |

## Epoch Trend

| epoch | step | base_reward | core | safe_count/16 | valid | valid+EP | mixed | all_valid | all_invalid | all_slow | pareto | pareto+adv | slowfail+adv | score_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1329 | 0.7842 | 0.8136 | 13.96 | 0.8724 | 0.6223 | 0.5237 | 0.4740 | 0.0022 | 0.0255 | 0.3041 | 0.9440 | 0.0000 | 0.5023 |
| 1 | 2659 | 0.7998 | 0.8232 | 14.15 | 0.8846 | 0.6451 | 0.4927 | 0.5053 | 0.0021 | 0.0256 | 0.3182 | 0.9456 | 0.0000 | 0.4791 |
| 2 | 3989 | 0.8053 | 0.8267 | 14.23 | 0.8892 | 0.6570 | 0.4811 | 0.5167 | 0.0021 | 0.0257 | 0.3231 | 0.9472 | 0.0000 | 0.4703 |
| 3 | 5319 | 0.8105 | 0.8300 | 14.29 | 0.8932 | 0.6647 | 0.4700 | 0.5280 | 0.0020 | 0.0256 | 0.3294 | 0.9481 | 0.0000 | 0.4625 |
| 4 | 6649 | 0.8154 | 0.8336 | 14.33 | 0.8957 | 0.6761 | 0.4611 | 0.5367 | 0.0022 | 0.0245 | 0.3367 | 0.9486 | 0.0000 | 0.4567 |
| 8 | 11969 | 0.8272 | 0.8418 | 14.49 | 0.9057 | 0.7001 | 0.4384 | 0.5601 | 0.0015 | 0.0221 | 0.3487 | 0.9505 | 0.0000 | 0.4377 |
| 9 | 13299 | 0.8306 | 0.8445 | 14.52 | 0.9078 | 0.7050 | 0.4334 | 0.5649 | 0.0017 | 0.0209 | 0.3529 | 0.9519 | 0.0000 | 0.4325 |
| 13 | 18619 | 0.8387 | 0.8509 | 14.63 | 0.9145 | 0.7176 | 0.4224 | 0.5765 | 0.0011 | 0.0185 | 0.3629 | 0.9531 | 0.0000 | 0.4202 |
| 15 | 21279 | 0.8423 | 0.8534 | 14.68 | 0.9176 | 0.7250 | 0.4138 | 0.5851 | 0.0012 | 0.0173 | 0.3665 | 0.9528 | 0.0000 | 0.4136 |
| 16 | 22609 | 0.8433 | 0.8544 | 14.70 | 0.9189 | 0.7267 | 0.4107 | 0.5882 | 0.0010 | 0.0184 | 0.3683 | 0.9549 | 0.0000 | 0.4126 |
| 17 | 23939 | 0.8450 | 0.8554 | 14.72 | 0.9200 | 0.7303 | 0.4098 | 0.5894 | 0.0008 | 0.0164 | 0.3700 | 0.9542 | 0.0000 | 0.4095 |
| 18 | 25269 | 0.8451 | 0.8555 | 14.72 | 0.9198 | 0.7318 | 0.4098 | 0.5893 | 0.0009 | 0.0163 | 0.3701 | 0.9537 | 0.0000 | 0.4092 |
| 19 | 26599 | 0.8465 | 0.8568 | 14.74 | 0.9212 | 0.7338 | 0.4029 | 0.5961 | 0.0010 | 0.0167 | 0.3722 | 0.9541 | 0.0000 | 0.4065 |

## Navtest Checkpoints With Aligned Training Diagnostics

| step | nav_PDMS | nav_Core | NC | DAC | TTC | EP | safe_count/16 | valid | valid+EP | mixed | all_valid | all_invalid | all_slow | pareto | score_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 300 | 0.887645 | 0.911337 | 0.9857 | 0.9610 | 0.9567 | 0.8305 | 14.31 | 0.8945 | 0.5742 | 0.4375 | 0.5625 | 0.0000 | 0.0000 | 0.2734 | 0.4781 |
| 600 | 0.888936 | 0.910401 | 0.9851 | 0.9654 | 0.9542 | 0.8308 | 14.12 | 0.8828 | 0.5938 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0.2852 | 0.5317 |
| 900 | 0.885056 | 0.910578 | 0.9835 | 0.9569 | 0.9468 | 0.8386 | 13.12 | 0.8203 | 0.6719 | 0.5625 | 0.3750 | 0.0625 | 0.0000 | 0.3320 | 0.5589 |
| 3000 | 0.890736 | 0.913310 | 0.9780 | 0.9650 | 0.9427 | 0.8493 | 13.94 | 0.8711 | 0.7383 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0.3789 | 0.4836 |
| 5100 | 0.878800 | 0.906702 | 0.9743 | 0.9554 | 0.9251 | 0.8510 | 13.94 | 0.8711 | 0.7227 | 0.5625 | 0.4375 | 0.0000 | 0.0000 | 0.3555 | 0.4825 |
| 5400 | 0.890187 | 0.912326 | 0.9815 | 0.9650 | 0.9566 | 0.8330 | 12.81 | 0.8008 | 0.6250 | 0.6875 | 0.3125 | 0.0000 | 0.0000 | 0.3086 | 0.7310 |
| 9300 | 0.893266 | 0.912327 | 0.9816 | 0.9717 | 0.9538 | 0.8358 | 13.56 | 0.8477 | 0.5430 | 0.6250 | 0.3750 | 0.0000 | 0.1250 | 0.2539 | 0.5153 |
| 13200 | 0.900218 | 0.918305 | 0.9781 | 0.9740 | 0.9464 | 0.8575 | 15.12 | 0.9453 | 0.6172 | 0.6250 | 0.3750 | 0.0000 | 0.0000 | 0.3594 | 0.4477 |
| 18600 | 0.906764 | 0.923797 | 0.9782 | 0.9756 | 0.9460 | 0.8711 | 14.69 | 0.9180 | 0.5312 | 0.3750 | 0.6250 | 0.0000 | 0.1875 | 0.2812 | 0.3805 |
| 19800 | 0.901633 | 0.918166 | 0.9793 | 0.9764 | 0.9464 | 0.8573 | 15.12 | 0.9453 | 0.8594 | 0.2500 | 0.7500 | 0.0000 | 0.0000 | 0.3906 | 0.3739 |
| 20400 | 0.907306 | 0.922248 | 0.9823 | 0.9781 | 0.9525 | 0.8608 | 13.88 | 0.8672 | 0.7578 | 0.4375 | 0.5625 | 0.0000 | 0.0000 | 0.3398 | 0.4159 |
| 21600 | 0.910274 | 0.923721 | 0.9844 | 0.9805 | 0.9581 | 0.8588 | 14.50 | 0.9062 | 0.6758 | 0.4375 | 0.5625 | 0.0000 | 0.0000 | 0.3398 | 0.4824 |
| 22500 | 0.903310 | 0.921580 | 0.9784 | 0.9733 | 0.9504 | 0.8614 | 14.12 | 0.8828 | 0.7461 | 0.6250 | 0.3750 | 0.0000 | 0.0000 | 0.3594 | 0.5249 |
| 23100 | 0.907023 | 0.922344 | 0.9805 | 0.9782 | 0.9515 | 0.8622 | 15.25 | 0.9531 | 0.6797 | 0.4375 | 0.5625 | 0.0000 | 0.0000 | 0.2852 | 0.4101 |
| 24000 | 0.907301 | 0.922169 | 0.9797 | 0.9792 | 0.9469 | 0.8663 | 14.81 | 0.9258 | 0.7148 | 0.4375 | 0.5625 | 0.0000 | 0.0625 | 0.4883 | 0.3895 |
| 24300 | 0.908567 | 0.923048 | 0.9817 | 0.9788 | 0.9539 | 0.8614 | 14.62 | 0.9141 | 0.6758 | 0.4375 | 0.5625 | 0.0000 | 0.0000 | 0.2969 | 0.4891 |
| 25500 | 0.908960 | 0.923480 | 0.9805 | 0.9798 | 0.9524 | 0.8640 | 15.19 | 0.9492 | 0.6758 | 0.3125 | 0.6875 | 0.0000 | 0.0000 | 0.2930 | 0.3898 |
| 25800 | 0.906940 | 0.922400 | 0.9790 | 0.9783 | 0.9512 | 0.8625 | 14.12 | 0.8828 | 0.6719 | 0.7500 | 0.2500 | 0.0000 | 0.0000 | 0.2773 | 0.4922 |
| 26400 | 0.907600 | 0.923201 | 0.9801 | 0.9781 | 0.9504 | 0.8653 | 14.69 | 0.9180 | 0.7422 | 0.4375 | 0.5625 | 0.0000 | 0.0000 | 0.3750 | 0.3825 |

## Correlation: Aligned Training Diagnostics vs Navtest

Pearson/Spearman are computed across evaluated navtest checkpoints using the latest logged train scalar at or before each checkpoint step. This is diagnostic only, not a causal proof.
| train_metric | Pearson_PDMS | Spearman_PDMS | Pearson_EP | Pearson_TTC | Pearson_NC*DAC |
| --- | --- | --- | --- | --- | --- |
| safe_count_mean | 0.297 | 0.310 | 0.279 | -0.023 | 0.220 |
| valid_ratio | 0.297 | 0.310 | 0.279 | -0.023 | 0.220 |
| all_valid_group | 0.280 | 0.278 | 0.281 | -0.029 | 0.199 |
| tradeoff_bad | -0.271 | -0.266 | -0.289 | 0.115 | -0.199 |
| mixed_group | -0.263 | -0.265 | -0.267 | 0.033 | -0.186 |
| score_std | -0.261 | -0.339 | -0.363 | 0.203 | -0.121 |
| pdms_core | 0.240 | 0.238 | 0.315 | -0.168 | 0.127 |
| base_reward | 0.228 | 0.217 | 0.364 | -0.249 | 0.071 |
| slow_violation | -0.202 | -0.209 | -0.332 | 0.230 | -0.057 |
| all_invalid_group | -0.179 | -0.190 | -0.133 | -0.052 | -0.129 |
| ddc_guard_pass | 0.158 | 0.143 | -0.030 | 0.294 | 0.216 |
| valid_progress_ratio | 0.144 | 0.142 | 0.303 | -0.197 | -0.043 |
| ep_floor_pass | 0.125 | 0.132 | 0.323 | -0.259 | -0.081 |
| pareto_front | 0.122 | 0.078 | 0.268 | -0.184 | -0.047 |

## Interpretation

- The grouping mechanics did operate: every group had 16 samples, most groups had many valid samples, all-invalid rescue was rarely needed, and Pareto/non-dominated samples were repeatedly assigned positive advantage.
- The EP-floor mechanism was useful: `valid_ratio` was high but `valid_progress_ratio` was lower, and `positive_advantage_slow_fail_ratio` stayed 0. This prevents the old failure mode where slow/conservative trajectories receive positive update just because TTC is high.
- The best navtest checkpoint was not explained by maximum training valid ratio alone. Late checkpoints have similar or higher valid ratios but lower PDMS, so the next improvement should focus on advantage quality/ranking sharpness and checkpoint selection, not only more hard safety filtering.
- Because `all_valid_group_ratio` rises late, further gains likely need richer within-valid comparisons: phenotype bucket GRPO, better per-scene normalization, or a small train-only buffer-neighborhood bonus, instead of stricter NC/DAC/DDC gates.
