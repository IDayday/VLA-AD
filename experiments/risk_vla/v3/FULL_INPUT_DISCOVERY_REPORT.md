# RISK-VLA v3 Full Input Discovery Report

Git commit: `2a00819a3e3bd4993869ab65c4656da889e9eede`
Minimum real scale: `10000` tokens

## Chunk Caches

| Split | Chunks | Tokens | >=10k |
| --- | ---: | ---: | --- |
| navtest | 4 | 12146 | True |
| train | 54 | 206072 | True |
| unknown | 1 | 720 | False |
| val | 1 | 1024 | False |

## Metric Caches

- `/mnt/project/VLA-AD/cache/metric_cache_navtest_first1024` split=`navtest` metadata_rows=`1024`
- `/mnt/project/VLA-AD/cache/metric_cache_navtest_smoke16` split=`navtest` metadata_rows=`16`
- `/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1` split=`navtest` metadata_rows=`0`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/pdm_flow/metric_cache_navtest_first1024` split=`navtest` metadata_rows=`0`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/pdm_flow/metric_cache_navtest_full_v1` split=`navtest` metadata_rows=`0`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/pdm_flow/metric_cache_navtest_full_v1_shard_backfill_20260528_0000` split=`navtest` metadata_rows=`0`
- `/mnt/project/VLA-AD/experiments/bit_drive/select/P7_safety_mining/metric_cache_train_chunk000013_1024` split=`train` metadata_rows=`1021`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/active_mining/metric_cache_backfill_chunks000000_000005_balanced_rr_next1024_interaction_ttc` split=`train` metadata_rows=`1021`
- `/mnt/project/VLA-AD/experiments/bit_drive/d5_mining_background/metric_cache_train_chunk000015_512` split=`train` metadata_rows=`512`
- `/mnt/project/VLA-AD/experiments/bit_drive/d5_mining_background/metric_cache_train_chunk000016_512` split=`train` metadata_rows=`511`
- `/mnt/project/VLA-AD/experiments/bit_drive/d5_mining_background/metric_cache_train_chunk000017_512` split=`train` metadata_rows=`511`
- `/mnt/project/bit_drive_left_tail/experiments/bit_drive/d5_mining_background_gpu2/metric_cache_train_chunk000019_512` split=`train` metadata_rows=`511`
- `/mnt/project/bit_drive_left_tail/experiments/bit_drive/d5_mining_background_gpu2/metric_cache_train_chunk000018_512` split=`train` metadata_rows=`509`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/active_mining/metric_cache_train_backfill_chunk000000_top512_interaction_ttc` split=`train` metadata_rows=`508`
- `/mnt/project/VLA-AD/experiments/bit_drive/select/P7_safety_mining/metric_cache_train_chunk000007_264` split=`train` metadata_rows=`263`
- `/mnt/project/VLA-AD/experiments/bit_drive/select/P10_safety_holdout_mining/metric_cache_train_chunk000019_256` split=`train` metadata_rows=`256`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/active_mining/metric_cache_backfill_chunks000000_000005_balanced_rr_next256_interaction_ttc` split=`train` metadata_rows=`256`
- `/mnt/project/VLA-AD/experiments/bit_drive/select/P10_safety_holdout_mining/metric_cache_train_chunk000018_256` split=`train` metadata_rows=`253`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/active_mining/metric_cache_backfill_chunks000000_000005_balanced_rr_next64_interaction_ttc_smoke` split=`train` metadata_rows=`64`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/active_mining/metric_cache_train_backfill_chunk000000_balanced_next32_interaction_ttc_smoke` split=`train` metadata_rows=`32`

## PDM CSV Candidates

- `/mnt/project/VLA-AD/experiments/recogdrive_expert/pdm_flow/full_navtest_suite_wait_metric12138_20260527_1830/a0_no_expert/pdm_results.csv` split=`navtest` method=`a0_base` rows=`10002`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/pdm_flow/full_navtest_suite_wait_metric12138_20260527_1830/a1_jepa_only/pdm_results.csv` split=`navtest` method=`jepa` rows=`10002`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/pdm_flow/full_navtest_suite_wait_metric12138_20260527_1830/a2_vggt_only/pdm_results.csv` split=`navtest` method=`vggt` rows=`10002`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/pdm_flow/full_navtest_suite_wait_metric12138_20260527_1830/a3_context_only/pdm_results.csv` split=`navtest` method=`unknown` rows=`10002`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/pdm_flow/full_navtest_suite_wait_metric12138_20260527_1830/a4_jepa_vggt/pdm_results.csv` split=`navtest` method=`jepa` rows=`10002`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/pdm_flow/full_navtest_suite_wait_metric12138_20260527_1830/base_il/pdm_results.csv` split=`navtest` method=`a0_base` rows=`10002`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R10_riskvla_only_riskprob_semantic_navtestfull_pdm_parallel32/merged_full/per_sample_metrics.csv` split=`navtest` method=`risk_vla` rows=`10002`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R11_riskvla_only_riskprob_semantic_navtestfull_pdm_parallel32/merged_full/per_sample_metrics.csv` split=`navtest` method=`risk_vla` rows=`10002`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R14_vlm_full5506_provided_mask09_navtestfull_pdm_parallel32/merged_full/per_sample_metrics.csv` split=`navtest` method=`risk_vla` rows=`10002`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R15_vs_R12_vs_R4_navtest10k_delta.csv` split=`navtest` method=`risk_vla` rows=`10002`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R16_best_latest_vs_R15_R12_R4_navtest10k_delta.csv` split=`navtest` method=`risk_vla` rows=`10002`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R16_vs_R15_vs_R12_vs_R4_navtest10k_delta.csv` split=`navtest` method=`risk_vla` rows=`10002`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R17_vs_R12_R4_R15_R16_navtest10k_delta.csv` split=`navtest` method=`risk_vla` rows=`10002`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R6_vs_R5_vs_R4_vs_R3_navtest10k_compare/deltas.csv` split=`navtest` method=`risk_vla` rows=`10002`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R7_vs_R6_vs_R5_vs_R4_vs_R3_navtest10k_compare/deltas.csv` split=`navtest` method=`risk_vla` rows=`10002`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R8_riskprob_semantic_vs_prior_navtest10k_compare/deltas.csv` split=`navtest` method=`risk_vla` rows=`10002`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R9_vs_R8_vs_prior_navtest10k_compare/deltas.csv` split=`navtest` method=`risk_vla` rows=`10002`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R10_riskvla_only_riskprob_semantic_navtest10k_pdm_parallel32/merged_10k/per_sample_metrics.csv` split=`navtest` method=`risk_vla` rows=`10000`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R11_R10_riskprob_semantic_mask09_navtest10k_pdm_parallel32/merged_10k/per_sample_metrics.csv` split=`navtest` method=`risk_vla` rows=`10000`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R11_riskvla_only_riskprob_semantic_navtest10k_pdm_parallel32/merged_10k/per_sample_metrics.csv` split=`navtest` method=`risk_vla` rows=`10000`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R12_riskvla_only_riskprob_mask09_navtest10k_pdm_parallel32/merged_10k/per_sample_metrics.csv` split=`navtest` method=`risk_vla` rows=`10000`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R13_R12_perclass_dense08_progress095_other0925_navtest10k_pdm_parallel32/merged_10k/per_sample_metrics.csv` split=`navtest` method=`risk_vla` rows=`10000`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R13_vlm_pilot256_provided_mask09_navtest10k_pdm_parallel32/merged_10k/per_sample_metrics.csv` split=`navtest` method=`risk_vla` rows=`10000`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R14_R12_mask09_scale020_navtest10k_pdm_parallel32/merged_10k/per_sample_metrics.csv` split=`navtest` method=`risk_vla` rows=`10000`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R14_vlm_full5506_provided_mask09_navtest10k_pdm_parallel32/merged_10k/per_sample_metrics.csv` split=`navtest` method=`risk_vla` rows=`10000`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R15_riskvla_ttc_calibrated_navtest10k_pdm_parallel32/merged_10k/per_sample_metrics.csv` split=`navtest` method=`risk_vla` rows=`10000`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R16_latest_riskvla_ttc_trajloss_navtest10k_pdm_parallel32/merged_10k/per_sample_metrics.csv` split=`navtest` method=`risk_vla` rows=`10000`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R16_riskvla_ttc_trajloss_navtest10k_pdm_parallel32/merged_10k/per_sample_metrics.csv` split=`navtest` method=`risk_vla` rows=`10000`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R17_R12_eval_ttc_slowdown09_scale03_navtest10k_pdm_parallel32/merged_10k/per_sample_metrics.csv` split=`navtest` method=`risk_vla` rows=`10000`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R3_predicted_router_navtest10k_pdm_parallel32/merged_10k/per_sample_metrics.csv` split=`navtest` method=`risk_vla` rows=`10000`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R4_predicted_router_navtest10k_pdm_parallel8/merged_10k/per_sample_metrics.csv` split=`navtest` method=`risk_vla` rows=`10000`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R5_predicted_router_navtest10k_pdm_parallel32/merged_10k/per_sample_metrics.csv` split=`navtest` method=`risk_vla` rows=`10000`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R6_predicted_router_navtest10k_pdm_parallel32/merged_10k/per_sample_metrics.csv` split=`navtest` method=`risk_vla` rows=`10000`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R7_predicted_router_navtest10k_pdm_parallel32/merged_10k/per_sample_metrics.csv` split=`navtest` method=`risk_vla` rows=`10000`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R8_R4_riskprob_semantic_navtest10k_pdm_parallel32/merged_10k/per_sample_metrics.csv` split=`navtest` method=`risk_vla` rows=`10000`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R9_riskprob_semantic_navtest10k_pdm_parallel32/merged_10k/per_sample_metrics.csv` split=`navtest` method=`risk_vla` rows=`10000`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R4_predicted_router_navtest10k_pdm_parallel8/shard_00/per_sample_metrics.csv` split=`navtest` method=`risk_vla` rows=`1250`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R4_predicted_router_navtest10k_pdm_parallel8/shard_01/per_sample_metrics.csv` split=`navtest` method=`risk_vla` rows=`1250`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R4_predicted_router_navtest10k_pdm_parallel8/shard_02/per_sample_metrics.csv` split=`navtest` method=`risk_vla` rows=`1250`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R4_predicted_router_navtest10k_pdm_parallel8/shard_03/per_sample_metrics.csv` split=`navtest` method=`risk_vla` rows=`1250`

## Candidate / Counterfactual Assets

- `/mnt/project/VLA-AD/experiments/bit_drive/select/P11_enriched_features/counterfactual_navtest_1024_enriched/base_predictions.jsonl` split=`navtest` method=`bit` count=`1024`
- `/mnt/project/VLA-AD/experiments/bit_drive/select/P11_enriched_features/counterfactual_navtest_1024_enriched/bit_predictions.jsonl` split=`navtest` method=`bit` count=`1024`
- `/mnt/project/VLA-AD/experiments/bit_drive/select/P11_enriched_features/counterfactual_navtest_1024_enriched/counterfactual_samples.jsonl` split=`navtest` method=`bit` count=`1024`
- `/mnt/project/VLA-AD/experiments/bit_drive/select/counterfactual_navtest_1024/base_predictions.jsonl` split=`navtest` method=`bit` count=`1024`
- `/mnt/project/VLA-AD/experiments/bit_drive/select/counterfactual_navtest_1024/bit_predictions.jsonl` split=`navtest` method=`bit` count=`1024`
- `/mnt/project/VLA-AD/experiments/bit_drive/select/counterfactual_navtest_1024/counterfactual_samples.jsonl` split=`navtest` method=`bit` count=`1024`
- `/mnt/project/VLA-AD/experiments/bit_drive/v4/eval_D1_latest_navtest_1024/predictions.jsonl` split=`navtest` method=`bit` count=`None`
- `/mnt/project/VLA-AD/experiments/bit_drive/v4/eval_D1_navtest_1024/predictions.jsonl` split=`navtest` method=`bit` count=`None`
- `/mnt/project/VLA-AD/experiments/bit_drive/v4/eval_D2_navtest_1024/predictions.jsonl` split=`navtest` method=`bit` count=`None`
- `/mnt/project/VLA-AD/experiments/bit_drive/v4/eval_D3_navtest_1024/predictions.jsonl` split=`navtest` method=`bit` count=`None`
- `/mnt/project/VLA-AD/experiments/bit_drive/v4/eval_D4_navtest_1024/predictions.jsonl` split=`navtest` method=`bit` count=`None`
- `/mnt/project/VLA-AD/experiments/bit_drive/v5_d5/eval_D5A_navtest_1024/predictions.jsonl` split=`navtest` method=`d5` count=`None`
- `/mnt/project/VLA-AD/experiments/bit_drive/v5_d5/eval_D5B_navtest_1024/predictions.jsonl` split=`navtest` method=`d5` count=`None`
- `/mnt/project/VLA-AD/experiments/bit_drive/v5_d5/eval_D5C_navtest_1024/predictions.jsonl` split=`navtest` method=`d5` count=`None`
- `/mnt/project/VLA-AD/experiments/bit_drive/v5_d5/eval_D5D_navtest_1024/predictions.jsonl` split=`navtest` method=`d5` count=`None`
- `/mnt/project/VLA-AD/experiments/bit_drive/v5_d5/eval_D5E_navtest_1024/predictions.jsonl` split=`navtest` method=`d5` count=`None`
- `/mnt/project/VLA-AD/experiments/risk_vla/v2/R2_formal_riskvla_strategy_full7095_20260606T032838Z/eval_navtest10k_step2000_debug_shard00/predictions.jsonl` split=`navtest` method=`risk_vla` count=`None`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R10_riskvla_diag_navtest10k_parallel32/merged_10k/predictions.jsonl` split=`navtest` method=`risk_vla` count=`None`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R10_riskvla_diag_navtest10k_parallel32/shard_00/predictions.jsonl` split=`navtest` method=`risk_vla` count=`None`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R10_riskvla_diag_navtest10k_parallel32/shard_01/predictions.jsonl` split=`navtest` method=`risk_vla` count=`None`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R10_riskvla_diag_navtest10k_parallel32/shard_02/predictions.jsonl` split=`navtest` method=`risk_vla` count=`None`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R10_riskvla_diag_navtest10k_parallel32/shard_03/predictions.jsonl` split=`navtest` method=`risk_vla` count=`None`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R10_riskvla_diag_navtest10k_parallel32/shard_04/predictions.jsonl` split=`navtest` method=`risk_vla` count=`None`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R10_riskvla_diag_navtest10k_parallel32/shard_05/predictions.jsonl` split=`navtest` method=`risk_vla` count=`None`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R10_riskvla_diag_navtest10k_parallel32/shard_06/predictions.jsonl` split=`navtest` method=`risk_vla` count=`None`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R10_riskvla_diag_navtest10k_parallel32/shard_07/predictions.jsonl` split=`navtest` method=`risk_vla` count=`None`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R10_riskvla_diag_navtest10k_parallel32/shard_08/predictions.jsonl` split=`navtest` method=`risk_vla` count=`None`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R10_riskvla_diag_navtest10k_parallel32/shard_09/predictions.jsonl` split=`navtest` method=`risk_vla` count=`None`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R10_riskvla_diag_navtest10k_parallel32/shard_10/predictions.jsonl` split=`navtest` method=`risk_vla` count=`None`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R10_riskvla_diag_navtest10k_parallel32/shard_11/predictions.jsonl` split=`navtest` method=`risk_vla` count=`None`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R10_riskvla_diag_navtest10k_parallel32/shard_12/predictions.jsonl` split=`navtest` method=`risk_vla` count=`None`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R10_riskvla_diag_navtest10k_parallel32/shard_13/predictions.jsonl` split=`navtest` method=`risk_vla` count=`None`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R10_riskvla_diag_navtest10k_parallel32/shard_14/predictions.jsonl` split=`navtest` method=`risk_vla` count=`None`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R10_riskvla_diag_navtest10k_parallel32/shard_15/predictions.jsonl` split=`navtest` method=`risk_vla` count=`None`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R10_riskvla_diag_navtest10k_parallel32/shard_16/predictions.jsonl` split=`navtest` method=`risk_vla` count=`None`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R10_riskvla_diag_navtest10k_parallel32/shard_17/predictions.jsonl` split=`navtest` method=`risk_vla` count=`None`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R10_riskvla_diag_navtest10k_parallel32/shard_18/predictions.jsonl` split=`navtest` method=`risk_vla` count=`None`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R10_riskvla_diag_navtest10k_parallel32/shard_19/predictions.jsonl` split=`navtest` method=`risk_vla` count=`None`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R10_riskvla_diag_navtest10k_parallel32/shard_20/predictions.jsonl` split=`navtest` method=`risk_vla` count=`None`
- `/mnt/project/bit_drive_left_tail/experiments/risk_vla/round1/R10_riskvla_diag_navtest10k_parallel32/shard_21/predictions.jsonl` split=`navtest` method=`risk_vla` count=`None`

## Checkpoints

- `/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/ablations/A0_no_expert_finetune_1024/best.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/ablations/A0_no_expert_finetune_1024/latest.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/ablations/A0_no_expert_finetune_1024/step_00000250.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/ablations/A0_no_expert_finetune_1024/step_00000500.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/ablations/A0_no_expert_finetune_1024/step_00000750.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/ablations/A0_no_expert_finetune_1024/step_00001000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/main_il_full_v1_A0_no_expert_20260527_1500/best.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/main_il_full_v1_A0_no_expert_20260527_1500/latest.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/main_il_full_v1_A0_no_expert_20260527_1500/step_00002000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/main_il_full_v1_A0_no_expert_20260527_1500/step_00004000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/main_il_full_v1_A0_no_expert_20260527_1500/step_00006000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/main_il_full_v1_A0_no_expert_20260527_1500/step_00008000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/main_il_full_v1_A0_no_expert_20260527_1500/step_00010000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/main_il_full_v1_A0_no_expert_20260527_1500/step_00012000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/main_il_full_v1_A0_no_expert_20260527_1500/step_00014000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/main_il_full_v1_A0_no_expert_20260527_1500/step_00016000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/main_il_full_v1_A0_no_expert_20260527_1500/step_00018000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/main_il_full_v1_A0_no_expert_20260527_1500/step_00020000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/main_il_full_v1_A0_no_expert_20260527_1500/step_00022000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/main_il_full_v1_A0_no_expert_20260527_1500/step_00024000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a0_no_expert/best.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a0_no_expert/latest.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a0_no_expert/step_00010000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a0_no_expert/step_00020000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a0_no_expert/step_00030000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a0_no_expert/step_00040000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a0_no_expert/step_00050000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a0_no_expert/step_00060000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a0_no_expert/step_00070000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a0_no_expert/step_00080000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a0_no_expert/step_00090000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a0_no_expert/step_00100000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a0_no_expert/step_00110000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a0_no_expert/step_00120000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a0_no_expert/step_00130000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a0_no_expert/step_00140000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a0_no_expert/step_00150000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a0_no_expert/step_00160000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a4_v2/best.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a4_v2/latest.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a4_v2/step_00010000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a4_v2/step_00020000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a4_v2/step_00030000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a4_v2/step_00040000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a4_v2/step_00050000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a4_v2/step_00060000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a4_v2/step_00070000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a4_v2/step_00080000.ckpt` method=`a0_base` debug=`False`
- `/mnt/project/VLA-AD/experiments/recogdrive_expert/stage2_a4v2_stage1base_full200_20260528_1501/a4_v2/step_00090000.ckpt` method=`a0_base` debug=`False`

## Expert / World Token Caches

- `/mnt/project/VLA-AD/cache/last_vla_v2` kind=`last_rd` index_files=`0`
- `/mnt/project/VLA-AD/cache/last_vla_v2/highcap_no_risk/train_full_highcap_chunks` kind=`last_rd` index_files=`27`
- `/mnt/project/VLA-AD/cache/last_vla_v2/highcap_no_risk/train_geometry192_overlay` kind=`last_rd` index_files=`16`
- `/mnt/project/VLA-AD/cache/last_vla_v2/highcap_no_risk/train_jepa128_overlay` kind=`jepa` index_files=`43`
- `/mnt/project/VLA-AD/cache/recogdrive_expert_chunks` kind=`unknown` index_files=`0`
- `/mnt/project/VLA-AD/checkpoints/teachers` kind=`unknown` index_files=`0`

## System

- CPU count: `128`
- GPU: `{'index': 0, 'name': 'NVIDIA A800-SXM4-80GB', 'total_memory_gb': 85.17}`
- GPU: `{'index': 1, 'name': 'NVIDIA A800-SXM4-80GB', 'total_memory_gb': 85.17}`
- GPU: `{'index': 2, 'name': 'NVIDIA A800-SXM4-80GB', 'total_memory_gb': 85.17}`
- GPU: `{'index': 3, 'name': 'NVIDIA A800-SXM4-80GB', 'total_memory_gb': 85.17}`
- GPU: `{'index': 4, 'name': 'NVIDIA A800-SXM4-80GB', 'total_memory_gb': 85.17}`
- GPU: `{'index': 5, 'name': 'NVIDIA A800-SXM4-80GB', 'total_memory_gb': 85.17}`
- GPU: `{'index': 6, 'name': 'NVIDIA A800-SXM4-80GB', 'total_memory_gb': 85.17}`
- GPU: `{'index': 7, 'name': 'NVIDIA A800-SXM4-80GB', 'total_memory_gb': 85.17}`
- Disk: `{'path': '/mnt/project/VLA-AD', 'total_gb': 340966.72, 'used_gb': 337287.7, 'free_gb': 3679.01}`
- Disk: `{'path': '/mnt/project/bit_drive_left_tail', 'total_gb': 340966.72, 'used_gb': 337287.7, 'free_gb': 3679.01}`
- Disk: `{'path': '/mnt/project/bit_drive_left_tail/experiments/risk_vla_v3', 'total_gb': 340966.72, 'used_gb': 337287.7, 'free_gb': 3679.01}`
- Disk: `{'path': '/mnt/project/bit_drive_left_tail/cache/risk_vla_v3', 'total_gb': 340966.72, 'used_gb': 337287.7, 'free_gb': 3679.01}`

## Readiness

- `min_real_samples`: `10000`
- `train_10k_possible`: `True`
- `val_10k_possible`: `False`
- `navtest_10k_possible`: `True`
- `large_train_utility_labels_possible`: `False`
- `large_val_utility_labels_possible`: `True`
- `large_candidate_assets_found`: `True`
- `base_checkpoint_found`: `True`
- `bit_checkpoint_found`: `True`
- `risk_vla_checkpoint_found`: `True`
- `navtest_analysis_pdm_found`: `True`

## Blockers

- held-out val chunk tokens below 10000: 1024; do not substitute navtest for tuning

## Leakage Guard

Navtest/test PDM labels are analysis-only. Discovery reports them, but the manifest builder must not mark them as training labels, router supervision, threshold tuning, hard-negative mining, or VLM instruction data.
