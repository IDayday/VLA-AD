import pytorch_lightning as pl

from torch import Tensor
from typing import Dict, Tuple,Any
import torch

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.recogdrive.stage3_optimization import stable_gradient_norm
from navsim.planning.training.checkpoint_utils import (
    _filter_recogdrive_checkpoint_state_dict,
    _load_recogdrive_checkpoint_state_dict,
)


def _prediction_get(prediction: Any, key: str) -> Any:
    if isinstance(prediction, dict) and key in prediction:
        return prediction[key]
    return getattr(prediction, key, None)


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
        self._manual_grpo_replay = getattr(agent, "stage3_objective", "none") == "grpo_replay"
        if self._manual_grpo_replay:
            self.automatic_optimization = False

    def _log_optional_recogdrive_metrics(self, prediction: Any, logging_prefix: str) -> None:
        for key in (
            "diffusion_loss",
            "jepa_alignment_loss",
            "vggt_alignment_loss",
            "future_jepa_loss",
            "vggt_geometry_loss",
            "coarse_traj_loss",
            "coarse_heading_loss",
            "risk_loss",
            "policy_kd_loss",
            "jepa_gate_value",
            "vggt_gate_value",
            "branch_weight_vlm",
            "branch_weight_jepa",
            "branch_weight_vggt",
            "last_rd_group_weight_vlm",
            "last_rd_group_weight_dynamic",
            "last_rd_group_weight_geometry",
            "last_rd_group_weight_ego",
            "last_rd_group_weight_risk",
            "last_rd_token_norms",
            "coarse_traj_l1",
            "future_jepa_loss_raw",
            "last_rd_geometry_mode_code",
            "last_vla_geometry_loss",
            "last_vla_dynamic_loss",
            "last_vla_coarse_loss",
            "last_vla_heading_loss",
            "last_vla_progress_loss",
            "last_vla_risk_loss",
            "last_vla_cot_consistency_loss",
            "last_vla_policy_kd_loss",
            "x0_aux_loss",
            "delta_aux_loss",
            "geo_aux_loss",
            "planning_token_norm",
            "planning_token_pairwise_cosine",
            "planning_condition_keep_ratio",
            "planning_context_gate",
            "planning_layer_gate_mean",
            "planning_delta_norm",
            "planning_adapter_forward_count",
            "trajectory_aux_loss",
            "feasibility_aux_loss",
            "tangent_excess_loss",
            "curvature_excess_loss",
            "aux_alpha_weight_mean",
            "aux_warmup_ramp",
            "selected_target_is_gt_ratio",
            "selected_target_source_code",
            "residual_alpha",
            "full_x0_reconstruction_l1",
            "fs_target_abs_gt3_ratio",
            "fs_target_abs_gt5_ratio",
            "fs_output_bound_hit_ratio",
            "early_kink_rate",
            "tail_reverse_rate",
            "curvature_violation_rate",
            "last_vla_coarse_traj_l1",
            "last_vla_dynamic_loss_raw",
            "last_vla_geometry_loss_raw",
            "last_vla_geometry_weight_effective",
            "last_vla_dynamic_weight_effective",
            "last_vla_coarse_weight_effective",
            "last_vla_progress_weight_effective",
            "hidden_anchor_loss",
            "hidden_anchor_loss_weighted",
            "hidden_drift_cosine",
            "hidden_drift_l2",
            "hidden_anchor_computed",
            "hidden_anchor_every_n_steps_tensor",
            "hidden_anchor_step_index",
            "lora_trainable_param_count",
            "lora_matched_module_count",
            "base_reward",
            "lfp_scalar_mean",
            "lfp_ref_scalar_mean",
            "lfp_delta_scalar_mean",
            "lfp_ep_mean",
            "lfp_ttc_mean",
            "lfp_quality_mean",
            "lfp_nc_mean",
            "lfp_dac_mean",
            "lfp_ddc_mean",
            "lfp_tlc_mean",
            "lfp_feasible_ratio",
            "lfp_progress_ok_ratio",
            "lfp_ttc_ok_ratio",
            "lfp_quality_ok_ratio",
            "lfp_reference_pareto_ok_ratio",
            "lfp_reference_dominated_ratio",
            "lfp_quality_reference_regression_ratio",
            "lfp_pareto_front_ratio",
            "lfp_dominated_ratio",
            "lfp_positive_advantage_ratio",
            "lfp_negative_advantage_ratio",
            "lfp_zero_advantage_ratio",
            "lfp_advantage_mean",
            "lfp_advantage_std",
            "lfp_positive_eligible_advantage_mean",
            "lfp_ttc_fail_ratio",
            "lfp_ttc_fail_advantage_mean",
            "lfp_ttc_fail_zero_credit_ratio",
            "lfp_quality_fail_ratio",
            "lfp_quality_fail_advantage_mean",
            "lfp_quality_fail_zero_credit_ratio",
            "lfp_reference_dominated_advantage_mean",
            "lfp_reference_dominated_zero_credit_ratio",
            "lfp_positive_advantage_below_ref_scalar_ratio",
            "lfp_positive_advantage_below_ref_quality_ratio",
            "lfp_positive_advantage_reference_dominated_ratio",
            "lfp_positive_advantage_scalar_delta_mean",
            "lfp_positive_advantage_quality_delta_mean",
            "lfp_progress_fail_ratio",
            "lfp_progress_fail_advantage_mean",
            "lfp_progress_fail_zero_credit_ratio",
            "lfp_unsafe_advantage_mean",
            "lfp_all_infeasible_group_ratio",
            "lfp_all_infeasible_rescue_ratio",
            "lfp_advantage_clip_ratio",
            "lfp_group_pairwise_ade_m_mean",
            "lfp_group_pairwise_ade_m_p50",
            "lfp_group_endpoint_std_x_m",
            "lfp_group_endpoint_std_y_m",
            "lfp_group_endpoint_std_heading_rad",
            "lfp_group_scalar_span_mean",
            "lfp_low_spread_group_ratio",
            "lfp_low_scalar_span_group_ratio",
            "lfp_credit_active_group_ratio",
            "lfp_advantage_deadband_zero_ratio",
            "lfp_fs_transition_std_enabled",
            "lfp_transition_floor_normalized_mean",
            "lfp_transition_floor_normalized_max",
            "lfp_transition_floor_endpoint_std_x_m",
            "lfp_transition_floor_endpoint_std_y_m",
            "lfp_transition_floor_endpoint_std_heading_rad",
            "lfp_global_centered_mean",
            "lfp_global_centered_std",
            "lfp_global_normalization_count",
            "lfp_scene_group_std_mean",
            "lfp_advantage_normalization_scene_group",
            "lfp_frontier_energy_mean",
            "lfp_frontier_base_energy_mean",
            "lfp_frontier_pareto_energy_mean",
            "lfp_frontier_safety_energy_mean",
            "lfp_frontier_safety_first_enabled",
            "lfp_all_feasible_group_ratio",
            "lfp_mixed_feasibility_group_ratio",
            "lfp_unsafe_fraction_mean",
            "lfp_pareto_tradeoff_intensity_mean",
            "lfp_pareto_tradeoff_intensity_p50",
            "lfp_frontier_tradeoff_multiplier_mean",
            "lfp_diversity_mode_capacity_mean",
            "lfp_diversity_coverage_ratio_mean",
            "lfp_diversity_coverage_gap_mean",
            "lfp_diversity_frontier_bonus_mean",
            "lfp_diversity_capacity_active_ratio",
            "lfp_diversity_priority_uplift_ratio",
            "lfp_diversity_priority_mode_code",
            "lfp_diversity_frontier_energy_mean",
            "lfp_v1_ep_ttc_ddc_tradeoff_intensity_mean",
            "lfp_v1_ep_ttc_ddc_pareto_front_ratio",
            "lfp_v1_ep_scalar_ddc_tradeoff_intensity_mean",
            "lfp_v1_ep_scalar_ddc_pareto_front_ratio",
            "lfp_frontier_energy_p50",
            "lfp_frontier_energy_p90",
            "lfp_positive_fraction_mean",
            "lfp_negative_fraction_mean",
            "lfp_advantage_magnitude_mean",
            "lfp_frontier_fast_ema_mean",
            "lfp_frontier_slow_ema_mean",
            "lfp_learning_progress_mean",
            "lfp_sampler_entropy",
            "lfp_sampler_priority_max_to_median",
            "lfp_sampler_uniform_ratio_actual",
            "lfp_sampler_unseen_scene_ratio",
            "lfp_policy_loss",
            "lfp_exact_kl",
            "lfp_reference_kl_coeff",
            "lfp_trajectory_logprob_mean",
            "lfp_policy_gradient_norm",
            "lfp_planning_adapter_gradient_norm",
            "lfp_reference_fallback_ratio",
            "lfp_reference_source_code",
            "lfp_metric_ddc_fallback_ratio",
            "lfp_metric_raw_ddc_available_ratio",
            "scene_stage_type",
            "shaped_reward",
            "safe_ratio",
            "hard_safe_ratio",
            "mean_ep",
            "mean_nc",
            "mean_dac",
            "mean_ttc",
            "mean_comfort",
            "mean_ddc",
            "mean_tlc",
            "soft_safety_penalty_mean",
            "soft_safety_penalty_max",
            "soft_safety_penalty_weight",
            "soft_safety_mode_enabled",
            "diversity_bonus",
            "safe_diversity",
            "group_reward_std",
            "safe_count_mean",
            "group_weight_mean",
            "group_weight_min",
            "group_weight_max",
            "mixed_group_ratio",
            "all_safe_group_ratio",
            "all_unsafe_group_ratio",
            "mean_advantage",
            "mean_abs_advantage",
            "grpo_advantage_mean_before_transform",
            "grpo_advantage_std_before_transform",
            "grpo_advantage_mean_after_transform",
            "grpo_advantage_std_after_transform",
            "grpo_advantage_min",
            "grpo_advantage_max",
            "grpo_advantage_positive_ratio",
            "grpo_advantage_zero_ratio",
            "grpo_advantage_clip_frac",
            "grpo_advantage_batch_normalized",
            "grpo_advantage_clip_abs",
            "core_pareto_enabled",
            "pdms_core",
            "pdms_core_ref",
            "delta_core_vs_ref",
            "delta_pdms_vs_ref",
            "delta_ep_vs_ref",
            "delta_ttc_vs_ref",
            "delta_ddc_vs_ref",
            "ep_floor_pass_ratio",
            "ddc_guard_pass_ratio",
            "core_pareto_valid_ratio",
            "core_pareto_valid_progress_ratio",
            "pareto_front_ratio",
            "dominated_ratio",
            "positive_advantage_slow_fail_ratio",
            "pareto_positive_advantage_ratio",
            "core_pareto_mixed_group_ratio",
            "core_pareto_all_valid_group_ratio",
            "core_pareto_all_invalid_group_ratio",
            "core_pareto_all_slow_group_ratio",
            "core_pareto_effective_group_weight_mean",
            "core_pareto_score_mean",
            "core_pareto_score_std",
            "core_pareto_slow_violation_mean",
            "core_pareto_tradeoff_bad_mean",
            "core_pareto_ttc_violation_mean",
            "core_pareto_lambda_slow",
            "core_pareto_lambda_safety",
            "core_pareto_slow_rate_ema",
            "core_pareto_unsafe_rate_ema",
            "core_pareto_ddc_drop_rate_ema",
            "core_pareto_ref_gt_pdms",
            "core_pareto_ref_il_pdms",
            "core_pareto_ref_gt_core",
            "core_pareto_ref_il_core",
            "phenotype_bucket_count_mean",
            "buffer_bonus_mean",
            "buffer_bonus_max",
            "buffer_bonus_target_ratio",
            "buffer_bonus_distance_mean",
            "core_pareto_mode_enabled",
            "core_pareto_core_mean",
            "core_pareto_adjusted_core_mean",
            "core_pareto_pdms_formula_mean",
            "core_pareto_ep_floor_gap_mean",
            "core_pareto_ep_floor_penalty_mean",
            "core_pareto_ddc_penalty_mean",
            "core_pareto_dual_penalty_mean",
            "core_pareto_nc_dac_feasible_ratio",
            "core_pareto_ddc_guard_pass_ratio",
            "core_pareto_front_ratio",
            "core_pareto_valid_front_ratio",
            "core_pareto_ep_low_ratio",
            "core_pareto_progress_bucket_ratio",
            "core_pareto_ttc_bucket_ratio",
            "core_pareto_balanced_bucket_ratio",
            "core_pareto_core_std",
            "core_pareto_ep_floor",
            "core_pareto_ddc_guard_threshold",
            "core_pareto_pareto_bonus",
            "feasible_pareto_enabled",
            "fp_utility_mean",
            "fp_valid_ratio",
            "fp_pareto_front_ratio",
            "fp_dominated_ratio",
            "fp_ddc_guard_pass_ratio",
            "fp_ddc_regression_ratio",
            "fp_feas_cost_mean",
            "fp_geometry_bad_ratio",
            "fp_regression_0_from_positive_ratio",
            "fp_positive_adv_cap_ratio",
            "fp_bucket_count_mean",
            "fp_intra_adv_mean",
            "fp_inter_adv_mean",
            "pdas_success_rate",
            "pdas_learnability",
            "pdas_reward_std",
            "pdas_bon_gap",
            "pdas_bucket_diversity",
            "pdas_regression_risk",
            "pdas_weight_mean",
            "pdas_weight_min",
            "pdas_weight_max",
            "support_tag_ratios",
            "dpsi_enabled",
            "dpsi_support_weight_mean",
            "dpsi_loss",
            "dpsi_pairwise_rank_loss",
            "dpsi_invalid_repulsion_loss",
            "dpsi_preference_dpo_loss",
            "dpsi_loss_weight_effective",
            "dpsi_bc_loss",
            "dpsi_per_sample_loss_mean",
            "dpsi_target_norm_mean",
            "dpsi_gt_target_loss",
            "dpsi_non_gt_target_loss",
            "dpsi_non_gt_to_gt_loss_ratio",
            "dpsi_non_gt_residual_mass_ratio",
            "dpsi_non_gt_target_weight_ratio",
            "dpsi_paired_target_randomness",
            "dpsi_residual_budget_enabled",
            "dpsi_residual_budget_cap",
            "dpsi_residual_budget_active_ratio",
            "dpsi_residual_budget_scale_mean",
            "dpsi_residual_budget_unanchored_ratio",
            "dpsi_gt_anchor_injected_ratio",
            "dpsi_non_gt_exposure_ratio",
            "dpsi_non_gt_residual_mass_ratio_pre_budget",
            "dpsi_non_gt_residual_mass_ratio_post_budget",
            "dpsi_non_gt_target_weight_ratio_pre_budget",
            "dpsi_non_gt_target_weight_ratio_post_budget",
            "dpsi_effective_weight_sum",
            "dpsi_zero_weight_ratio",
            "dpsi_sampled_weight_sum_mean",
            "dpsi_sampled_real_ratio",
            "dpsi_support_count_mean",
            "dpsi_pairwise_distance_mean",
            "dpsi_source_entropy_mean",
            "dpsi_beta_mean",
            "dpsi_beta_max",
            "dpsi_low_support_ratio",
            "dpsi_high_gt_saturated_ratio",
            "dpsi_selected_has_improver_ratio",
            "dpsi_strong_improver_ratio",
            "dpsi_scene_weight_mean",
            "dpsi_scene_uniform_weighting",
            "dpsi_policy_reachability_mean",
            "dpsi_policy_reachability_max",
            "dpsi_pareto_objective_novelty_mean",
            "dpsi_mode_balance_enabled",
            "dpsi_frontier_v4_enabled",
            "dpsi_learning_frontier_v5_enabled",
            "dpsi_mode_effective_count_mean",
            "dpsi_mode_capacity_mean",
            "dpsi_mode_density_ess_mean",
            "dpsi_mode_nearest_neighbor_distance_mean",
            "dpsi_mode_beta_mean",
            "dpsi_anchor_weight_mean",
            "dpsi_pareto_weight_mean",
            "dpsi_effective_target_count_mean",
            "dpsi_row_weight_sum_mean",
            "dpsi_zero_row_ratio",
            "dpsi_external_weight_ratio",
            "dpsi_gt_weight_ratio",
            "dpsi_il_weight_ratio",
            "dpsi_sampled_target_count_mean",
            "dpsi_sampled_anchor_ratio",
            "dpsi_sampled_best_ratio",
            "dpsi_sampled_unique_target_ratio",
            "dpsi_sampled_gt_weight_ratio",
            "dpsi_residual_budget_forced_gt_ratio",
            "dpsi_unbiased_systematic_sampling",
            "scorer_selection_regret",
            "oracle_topk_gap",
            "trajectory_logp",
            "gspo_ratio_mean",
            "gspo_ratio_min",
            "gspo_ratio_max",
            "gspo_ratio_clip_frac",
            "gspo_log_ratio_mean",
            "gspo_log_ratio_std",
            "gspo_log_ratio_min",
            "gspo_log_ratio_max",
            "gspo_abs_log_ratio_mean",
            "gspo_approx_kl",
            "gspo_reverse_approx_kl",
            "use_gspo_ratio",
            "sampled_from_behavior_policy",
            "behavior_policy_synced",
            "behavior_policy_sync_interval",
            "gspo_clip_low",
            "gspo_clip_high",
            "ppo_replay_valid_ratio",
            "ppo_replay_valid_count",
            "ppo_replay_inner_epochs",
            "ppo_replay_minibatch_size",
            "ppo_replay_minibatch_valid_ratio",
            "ppo_replay_minibatch_valid_count",
            "ppo_replay_loss_active",
            "ppo_replay_optimizer_steps",
            "ppo_replay_step_logprob_mode",
            "ppo_replay_step_minibatch_mode",
            "ppo_replay_transition_mode",
            "ppo_replay_selected_transition_count",
            "ppo_replay_selected_denoising_step_mean",
            "ppo_replay_selected_denoising_step_min",
            "ppo_replay_selected_denoising_step_max",
            "ppo_replay_step_clip_mean",
            "ppo_replay_step_clip_min",
            "ppo_replay_step_clip_max",
            "ppo_replay_logprob_clamped_frac",
            "ppo_replay_new_logprob_clamped_frac",
            "ppo_replay_old_logprob_clamped_frac",
            "bc_coeff",
            "reference_kl_loss",
            "reference_kl_coeff",
            "reference_kl_chunk_size",
            "grpo_buffer_guidance_enabled",
            "grpo_buffer_reward_bonus",
            "grpo_buffer_reward_bonus_weight",
            "grpo_buffer_reward_bonus_max",
            "grpo_buffer_target_distance_mean",
            "grpo_buffer_guidance_target_ratio",
            "grpo_buffer_selected_valid_ratio",
            "grpo_buffer_best_valid_minus_gt",
            "grpo_buffer_best_valid_minus_il",
            "grpo_buffer_distill_loss",
            "grpo_buffer_distill_weight",
            "grpo_buffer_distill_weight_sum",
            "grpo_buffer_distill_zero_weight_batch",
            "grpo_buffer_preference_dpo_loss",
            "grpo_buffer_preference_dpo_weight",
            "grpo_buffer_preference_dpo_target_ratio",
            "grpo_buffer_preference_dpo_il_loser_ratio",
            "grpo_buffer_preference_dpo_pair_count",
            "grpo_buffer_preference_dpo_active_row_ratio",
            "grpo_buffer_preference_dpo_reward_gap_mean",
            "grpo_buffer_preference_dpo_gap_weight_mean",
            "grpo_buffer_preference_dpo_logit_mean",
            "grpo_buffer_preference_dpo_logit_abs_mean",
            "grpo_buffer_preference_dpo_implicit_accuracy",
            "grpo_buffer_preference_dpo_current_margin_mean",
            "grpo_buffer_preference_dpo_reference_margin_mean",
            "grpo_buffer_preference_dpo_timestep_mean",
            "grpo_buffer_preference_dpo_timestep_min",
            "grpo_buffer_preference_dpo_timestep_max",
            "grpo_self_imitation_enabled",
            "grpo_self_imitation_loss",
            "grpo_self_imitation_weight",
            "grpo_self_imitation_candidate_ratio",
            "grpo_self_imitation_safety_candidate_ratio",
            "grpo_self_imitation_nc_pass_ratio",
            "grpo_self_imitation_dac_pass_ratio",
            "grpo_self_imitation_ttc_pass_ratio",
            "grpo_self_imitation_ddc_pass_ratio",
            "grpo_self_imitation_pre_cap_target_ratio",
            "grpo_self_imitation_target_ratio",
            "grpo_self_imitation_target_scene_cap_ratio",
            "grpo_self_imitation_target_scene_cap_active",
            "grpo_self_imitation_target_reward_mean",
            "grpo_self_imitation_target_reward_max",
            "grpo_self_imitation_target_margin_mean",
            "grpo_self_imitation_baseline_mean",
            "grpo_self_imitation_baseline_from_buffer",
            "grpo_self_imitation_target_nc_mean",
            "grpo_self_imitation_target_dac_mean",
            "grpo_self_imitation_target_ttc_mean",
            "grpo_self_imitation_target_ep_mean",
            "grpo_self_imitation_target_comfort_mean",
            "grpo_self_imitation_target_ddc_mean",
            "grpo_self_imitation_target_tlc_mean",
            "grpo_self_imitation_weight_sum",
            "grpo_self_imitation_zero_weight_batch",
            "grpo_self_imitation_timestep_mean",
            "awac_loss",
            "bc_loss_weight_effective",
            "grpo_loss",
            "grpo_loss_weight_effective",
            "reward_mean",
            "reward_max",
            "gt_reward_mean",
            "il_reward_mean",
            "best_raw_reward_mean",
            "best_valid_reward_mean",
            "best_selected_reward_mean",
            "best_raw_minus_gt_mean",
            "best_valid_minus_gt_mean",
            "best_selected_minus_gt_mean",
            "best_raw_minus_il_mean",
            "best_valid_minus_il_mean",
            "best_selected_minus_il_mean",
            "best_reward_mean",
            "best_minus_gt_mean",
            "best_minus_il_mean",
            "pct_best_raw_above_gt",
            "pct_best_valid_above_gt",
            "pct_best_selected_above_gt",
            "pct_best_valid_above_il",
            "pct_best_above_gt",
            "pct_best_above_il",
            "pct_candidates_above_gt",
            "pct_candidates_above_il",
            "selected_reward_mean",
            "selected_reward_max",
            "awac_weight_mean",
            "awac_weight_max",
            "awac_advantage_mean",
            "awac_advantage_max",
            "awac_pairwise_rank_loss",
            "awac_invalid_repulsion_loss",
            "awac_dpo_loss",
            "awac_loss_weight_effective",
            "shaped_reward_mean",
            "shaped_reward_delta_mean",
            "shaped_reward_delta_min",
            "shaped_reward_delta_max",
            "component_progress_bonus_mean",
            "component_safety_penalty_mean",
            "component_valid_reward_delta_mean",
            "source_balance_factor_mean",
            "source_balance_factor_min",
            "source_balance_factor_max",
            "source_balance_active_sources",
            "target_blend_alpha_effective",
            "target_blend_l2_to_original",
            "target_blend_anchor_gt_ratio",
            "target_blend_anchor_il_ratio",
            "target_blend_anchor_fallback_gt_ratio",
            "pairwise_rank_pair_count",
            "invalid_repulsion_pair_count",
            "pairwise_rank_active_row_ratio",
            "invalid_repulsion_active_row_ratio",
            "preference_dpo_pair_count",
            "preference_dpo_active_row_ratio",
            "preference_dpo_reward_gap_mean",
            "preference_dpo_gap_weight_mean",
            "preference_dpo_loss_weight_effective",
            "preference_dpo_current_logratio_mean",
            "preference_dpo_reference_logratio_mean",
            "preference_dpo_logit_mean",
            "preference_dpo_logit_abs_mean",
            "preference_dpo_implicit_accuracy",
            "preference_dpo_current_winner_loss_mean",
            "preference_dpo_current_loser_loss_mean",
            "preference_dpo_reference_winner_loss_mean",
            "preference_dpo_reference_loser_loss_mean",
            "preference_dpo_current_margin_mean",
            "preference_dpo_reference_margin_mean",
            "preference_dpo_timestep_mean",
            "preference_dpo_timestep_min",
            "preference_dpo_timestep_max",
            "empty_awac_row_ratio",
            "positive_weight_row_ratio",
            "positive_weight_candidate_ratio",
            "target_filter_row_ratio",
            "target_filter_candidate_ratio",
            "has_valid_candidate_ratio",
            "valid_candidate_ratio",
            "selected_valid_ratio",
            "fallback_candidate_ratio",
            "selected_nc_mean",
            "selected_dac_mean",
            "selected_ttc_mean",
            "selected_ep_mean",
            "selected_comfort_mean",
            "selected_ddc_mean",
            "selected_tlc_mean",
            "source_gt_ratio",
            "source_il_ratio",
            "source_policy_ratio",
            "source_progress_ratio",
            "source_lateral_ratio",
            "source_timing_ratio",
            "awac_per_sample_loss_mean",
            "awac_target_norm_mean",
            "awac_timestep_mean",
            "awac_timestep_min",
            "awac_timestep_max",
            "awac_effective_weight_sum",
            "awac_zero_weight_ratio",
            "awac_zero_weight_batch",
        ):
            value = _prediction_get(prediction, key)
            if value is not None:
                self.log(
                    f"{logging_prefix}/{key}",
                    value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )

        action_head = getattr(self.agent, "action_head", None)
        if action_head is None:
            return
        if getattr(action_head.config, "use_expert_features", False) and getattr(action_head.config, "use_expert_gates", False):
            for stream in ("jepa", "vggt"):
                gate = getattr(action_head, f"{stream}_gate", None)
                if gate is not None:
                    self.log(
                        f"{logging_prefix}/{stream}_gate",
                        torch.sigmoid(gate.detach()),
                        on_step=True,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                    )

    def _propagate_training_progress(self, logging_prefix: str) -> None:
        if logging_prefix != "train":
            return
        if not hasattr(self.agent, "set_training_progress"):
            return
        trainer = getattr(self, "trainer", None)
        current_epoch = int(getattr(trainer, "current_epoch", 0)) if trainer is not None else int(self.current_epoch)
        max_epochs = int(getattr(trainer, "max_epochs", 1)) if trainer is not None else 1
        global_step = int(getattr(trainer, "global_step", self.global_step)) if trainer is not None else int(self.global_step)
        self.agent.set_training_progress(current_epoch, max_epochs, global_step)

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets, tokens_list = batch
        self._propagate_training_progress(logging_prefix)
        prediction = self.agent.forward(features,targets,tokens_list)
        #prediction = self.agent.forward(features,targets)
        loss = self.agent.compute_loss(features, targets, prediction)
        self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{logging_prefix}/total_loss", loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self._log_optional_recogdrive_metrics(prediction, logging_prefix)
        
        return loss
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        每次保存 checkpoint 时，只保留 state_dict 中不以 'agent.model' 开头的条目。
        """
        checkpoint["state_dict"] = _filter_recogdrive_checkpoint_state_dict(checkpoint["state_dict"])
        action_head = getattr(self.agent, "action_head", None)
        if action_head is not None and hasattr(action_head, "lfp_runtime_state_dict"):
            lfp_state = action_head.lfp_runtime_state_dict()
            if lfp_state:
                checkpoint["lfp_runtime"] = lfp_state

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        action_head = getattr(self.agent, "action_head", None)
        if action_head is not None and hasattr(action_head, "load_lfp_runtime_state_dict"):
            action_head.load_lfp_runtime_state_dict(checkpoint.get("lfp_runtime", {}))

    def on_fit_end(self) -> None:
        action_head = getattr(self.agent, "action_head", None)
        evaluator = getattr(action_head, "lfp_v2_rollout_evaluator", None)
        if evaluator is not None:
            evaluator.close()

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Tensor]:
        return _filter_recogdrive_checkpoint_state_dict(super().state_dict(*args, **kwargs))

    def load_state_dict(
        self,
        state_dict: Dict[str, Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        return _load_recogdrive_checkpoint_state_dict(
            self,
            state_dict,
            strict=strict,
            assign=assign,
        )

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()


class AgentLightningDiT(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
        self._manual_grpo_replay = getattr(agent, "stage3_objective", "none") == "grpo_replay"
        if self._manual_grpo_replay:
            self.automatic_optimization = False

    def _log_optional_recogdrive_metrics(self, prediction: Any, logging_prefix: str) -> None:
        AgentLightningModule._log_optional_recogdrive_metrics(self, prediction, logging_prefix)

    def _propagate_training_progress(self, logging_prefix: str) -> None:
        AgentLightningModule._propagate_training_progress(self, logging_prefix)

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets, tokens_list = batch
        self._propagate_training_progress(logging_prefix)
        prediction = self.agent.forward(features,targets,tokens_list)
        if logging_prefix == 'train':
            predictions = self.agent.compute_loss(features, targets, prediction)
            if isinstance(predictions, torch.Tensor):
                loss = predictions
                metric_source = prediction
            else:
                loss = _prediction_get(predictions, "loss")
                if loss is None:
                    raise KeyError("Training prediction object is missing a 'loss' field.")
                metric_source = predictions
            self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/total_loss", loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            for key in ("reward", "policy_loss", "bc_loss"):
                value = _prediction_get(metric_source, key)
                if value is not None:
                    self.log(
                        f"{logging_prefix}/{key}",
                        value,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=True,
                        sync_dist=True,
                    )
            self._log_optional_recogdrive_metrics(metric_source, logging_prefix)
        else:
            prediction = self.agent.forward(features,targets)
            loss = self.agent.compute_loss(features, targets, prediction)
            self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/total_loss", loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            self._log_optional_recogdrive_metrics(prediction, logging_prefix)
        return loss
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        每次保存 checkpoint 时，只保留 state_dict 中不以 'agent.model' 开头的条目。
        """
        checkpoint["state_dict"] = _filter_recogdrive_checkpoint_state_dict(checkpoint["state_dict"])
        action_head = getattr(self.agent, "action_head", None)
        if action_head is not None and hasattr(action_head, "lfp_runtime_state_dict"):
            lfp_state = action_head.lfp_runtime_state_dict()
            if lfp_state:
                checkpoint["lfp_runtime"] = lfp_state

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        action_head = getattr(self.agent, "action_head", None)
        if action_head is not None and hasattr(action_head, "load_lfp_runtime_state_dict"):
            action_head.load_lfp_runtime_state_dict(checkpoint.get("lfp_runtime", {}))

    def on_fit_end(self) -> None:
        action_head = getattr(self.agent, "action_head", None)
        evaluator = getattr(action_head, "lfp_v2_rollout_evaluator", None)
        if evaluator is not None:
            evaluator.close()

    @staticmethod
    def _gradient_norm(parameters) -> torch.Tensor:
        return stable_gradient_norm(parameters)

    def _validate_lfp_reference_gradients(self) -> bool:
        if getattr(self.agent, "stage3_algorithm", "legacy") != "lfp_grpo":
            return False
        action_head = getattr(self.agent, "action_head", None)
        if action_head is None:
            return False
        reference = getattr(action_head, "old_policy", None)
        if reference is None or any(parameter.requires_grad for parameter in reference.parameters()):
            raise RuntimeError("LFP Stage2 reference policy must remain frozen.")
        if any(parameter.grad is not None for parameter in reference.parameters()):
            raise RuntimeError("LFP Stage2 reference policy received gradients.")
        return True

    def on_after_backward(self) -> None:
        self._validate_lfp_reference_gradients()

    def on_before_optimizer_step(self, optimizer) -> None:
        # Lightning invokes this hook after AMP unscaling and before clipping.
        # Recording in on_after_backward reports GradScaler-scaled values.
        if not self._validate_lfp_reference_gradients():
            return
        action_head = getattr(self.agent, "action_head")
        policy_norm = self._gradient_norm(
            parameter for parameter in action_head.parameters() if parameter.requires_grad
        ).to(self.device)
        planning_parameters = []
        for name, parameter in self.agent.named_parameters():
            if self.agent._is_planning_adapter_parameter_key(name) and parameter.requires_grad:
                planning_parameters.append(parameter)
        planning_norm = self._gradient_norm(planning_parameters).to(self.device)
        self.log(
            "train/lfp_policy_gradient_norm",
            policy_norm,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/lfp_planning_adapter_gradient_norm",
            planning_norm,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/lfp_policy_gradient_finite_ratio",
            torch.isfinite(policy_norm).float(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/lfp_planning_adapter_gradient_finite_ratio",
            torch.isfinite(planning_norm).float(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Tensor]:
        return _filter_recogdrive_checkpoint_state_dict(super().state_dict(*args, **kwargs))

    def load_state_dict(
        self,
        state_dict: Dict[str, Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        return _load_recogdrive_checkpoint_state_dict(
            self,
            state_dict,
            strict=strict,
            assign=assign,
        )

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        #print(batch_idx)
        if self._manual_grpo_replay:
            return self._training_step_grpo_replay(batch, batch_idx)
        return self._step(batch, "train")

    def _training_step_grpo_replay(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        features, targets, tokens_list = batch
        self._propagate_training_progress("train")
        rollout = self.agent.forward(features, targets, tokens_list)
        action_head = getattr(self.agent, "action_head")
        opt = self.optimizers()
        if isinstance(opt, (list, tuple)):
            opt = opt[0]

        num_samples = int(rollout["num_samples"])
        minibatch_size = int(getattr(action_head, "ppo_replay_minibatch_size", 0))
        inner_epochs = max(1, int(getattr(action_head, "ppo_replay_inner_epochs", 1)))
        max_grad_norm = float(getattr(action_head, "ppo_replay_max_grad_norm", 0.0))
        transition_replay = (
            str(getattr(action_head, "ppo_replay_logprob_mode", "trajectory")) == "step"
            and str(getattr(action_head, "ppo_replay_step_minibatch_mode", "trajectory_all_steps")) == "transition"
        )
        num_denoising_steps = int(rollout["num_denoising_steps"])
        replay_items = num_samples * num_denoising_steps if transition_replay else num_samples
        if minibatch_size <= 0 or minibatch_size > replay_items:
            minibatch_size = replay_items

        last_metrics = None
        loss_accum = rollout["loss"].new_zeros(())
        update_count = 0
        for _ in range(inner_epochs):
            perm = torch.randperm(replay_items, device=rollout["chains"].device)
            for start in range(0, replay_items, minibatch_size):
                replay_indices = perm[start:start + minibatch_size]
                denoising_step_indices = None
                if transition_replay:
                    indices = torch.div(replay_indices, num_denoising_steps, rounding_mode="floor")
                    denoising_step_indices = replay_indices.remainder(num_denoising_steps)
                else:
                    indices = replay_indices
                opt.zero_grad(set_to_none=True)
                metrics = action_head.compute_grpo_replay_loss(
                    rollout,
                    indices,
                    denoising_step_indices=denoising_step_indices,
                )
                loss = metrics["loss"]
                self.manual_backward(loss)
                if max_grad_norm > 0.0:
                    self.clip_gradients(opt, gradient_clip_val=max_grad_norm, gradient_clip_algorithm="norm")
                opt.step()
                last_metrics = metrics
                loss_accum = loss_accum + loss.detach()
                update_count += 1

        bc_metrics = None
        if bool(getattr(action_head, "ppo_replay_bc_update", True)) and float(action_head._current_bc_coeff()) > 0.0:
            opt.zero_grad(set_to_none=True)
            bc_metrics = action_head.compute_grpo_replay_bc_loss(rollout)
            bc_loss = bc_metrics["loss"]
            self.manual_backward(bc_loss)
            if max_grad_norm > 0.0:
                self.clip_gradients(opt, gradient_clip_val=max_grad_norm, gradient_clip_algorithm="norm")
            opt.step()
            loss_accum = loss_accum + bc_loss.detach()
            update_count += 1

        if last_metrics is None:
            raise RuntimeError("GRPO replay produced no optimizer updates.")

        mean_loss = loss_accum / max(1, update_count)
        log_source = dict(rollout)
        log_source.update(dict(last_metrics))
        if bc_metrics is not None:
            log_source.update({"bc_loss": bc_metrics["bc_loss"], "bc_coeff": bc_metrics["bc_coeff"]})
        log_source["loss"] = mean_loss
        log_source["total_loss"] = mean_loss
        log_source["ppo_replay_optimizer_steps"] = mean_loss.new_tensor(float(update_count))

        self.log("train/loss", mean_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/total_loss", mean_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        for key in ("reward", "policy_loss", "bc_loss"):
            value = _prediction_get(log_source, key)
            if value is not None:
                self.log(
                    f"train/{key}",
                    value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )
        self.log(
            "train/ppo_replay_optimizer_steps",
            log_source["ppo_replay_optimizer_steps"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self._log_optional_recogdrive_metrics(log_source, "train")
        return mean_loss

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()

    def on_train_epoch_end(self) -> None:
        if not self._manual_grpo_replay:
            return
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            if isinstance(scheduler, (list, tuple)):
                for item in scheduler:
                    if item is not None:
                        item.step()
            else:
                scheduler.step()
