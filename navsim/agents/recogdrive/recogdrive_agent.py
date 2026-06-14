from typing import Any, List, Dict, Literal, Optional, Union
import os
from pathlib import Path
import warnings
import torch
from torch.optim import Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from omegaconf import DictConfig, OmegaConf
from transformers.feature_extraction_utils import BatchFeature
import math

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from .utils.internvl_preprocess import load_image
from .utils.lr_scheduler import WarmupCosLR
from .utils.utils import format_number, build_from_configs
from .expert_backends import (
    DUMMY_EXPERT_WARNING,
    DummyExpertBackend,
    build_dummy_expert_backend,
    normalize_expert_feature_source,
)
from .recogdrive_features import (
    EXPERT_FEATURE_KEYS,
    EXPERT_TARGET_FEATURE_KEYS,
    ReCogDriveFeatureBuilder,
    TrajectoryTargetBuilder,
)
from .recogdrive_backbone import RecogDriveBackbone
from .recogdrive_diffusion_planner import (
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)
from .vlm_lora_utils import (
    audit_actual_trainable_lora_modules,
    audit_lora_target_modules,
    build_lora_config_payload,
    infer_lora_module_category,
    peft_lora_config_kwargs_supported,
    resolve_lora_target_modules,
    validate_lora_scope_audit,
)

LAST_VLA_FEATURE_KEYS = (
    "jepa_context_tokens",
    "jepa_target_tokens",
    "vggt_context_tokens",
    "vggt_target_tokens",
    "vggt_geometry_tokens",
    "vggt_geometry_target_tokens",
    "vggt_depth_tokens",
    "vggt_pointmap_tokens",
    "vggt_camera_tokens",
    "teacher_trajectory",
    "teacher_trajectory_norm",
    "teacher_score",
    "gt_score",
    "oracle_best_of_k_score",
    "candidate_count",
    "vlm_text_trajectory",
    "vlm_text_trajectory_norm",
    "vlm_text_parse_ok",
    "risk_labels",
    "generic_risk_labels",
    "drivable_risk_labels",
    "ttc_risk_labels",
    "comfort_risk_labels",
)
LAST_VLA_TARGET_KEYS = (
    "jepa_target_tokens",
    "vggt_target_tokens",
    "vggt_geometry_target_tokens",
    "vggt_depth_target_tokens",
    "vggt_pointmap_target_tokens",
    "teacher_trajectory",
    "teacher_trajectory_norm",
    "teacher_score",
    "gt_score",
    "oracle_best_of_k_score",
)
TWO_EXPERT_FEATURE_KEYS = (
    "two_expert_h_dyn",
    "two_expert_h_geo",
    "two_expert_corruption_mode",
    "jepa_dynamic_teacher_tokens",
    "vggt_feature23_tokens",
)
TWO_EXPERT_TARGET_KEYS = (
    "jepa_dynamic_teacher_tokens",
    "vggt_feature23_tokens",
)


class ReCogDriveAgent(AbstractAgent):
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        vlm_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        cam_type: Optional[str] = 'single', 
        vlm_type: Optional[str] = 'internvl', 
        dit_type: Optional[str] = 'small', 
        sampling_method: Optional[str] = 'ddim', 
        cache_mode: bool = False, 
        cache_hidden_state: bool = True, 
        lr: float = 1e-4,
        grpo: bool = False,
        stage3_objective: Literal["none", "grpo", "grpo_replay", "awac_iql", "hybrid"] = "none",
        grpo_sample_time: int = 8,
        bc_anneal: bool = False,
        bc_coeff_start: float = 0.1,
        bc_coeff_end: float = 0.1,
        bc_anneal_epochs: int = 1,
        reference_kl_coeff: float = 0.0,
        reference_kl_chunk_size: int = 0,
        grpo_use_gspo_ratio: bool = False,
        grpo_gspo_clip_low: float = 0.05,
        grpo_gspo_clip_high: float = 0.05,
        grpo_behavior_policy_sync_interval: int = 4,
        grpo_behavior_policy_sample: bool = True,
        grpo_normalize_advantage_batch: bool = False,
        grpo_advantage_clip_abs: float = 0.0,
        grpo_ppo_replay_inner_epochs: int = 1,
        grpo_ppo_replay_minibatch_size: int = 0,
        grpo_ppo_replay_max_grad_norm: float = 1.0,
        grpo_ppo_replay_min_abs_advantage: float = 1e-6,
        grpo_ppo_replay_filter_zero_advantage: bool = True,
        grpo_ppo_replay_sync_behavior_each_batch: bool = True,
        grpo_ppo_replay_bc_update: bool = True,
        grpo_ppo_replay_logprob_mode: Literal["trajectory", "step"] = "trajectory",
        grpo_ppo_replay_step_minibatch_mode: Literal["trajectory_all_steps", "transition"] = "trajectory_all_steps",
        grpo_ppo_replay_logprob_clamp_min: float = -5.0,
        grpo_ppo_replay_logprob_clamp_max: float = 2.0,
        grpo_ppo_replay_step_clip_schedule: Literal["constant", "dppo_exp"] = "constant",
        grpo_ppo_replay_step_clip_base: float = 0.001,
        grpo_ppo_replay_step_clip_rate: float = 3.0,
        metric_cache_path: Optional[str] = '', 
        reference_policy_checkpoint: Optional[str] = '', 
        offline_rl_enabled: bool = False,
        offline_rl_elite_buffer_path: str = "",
        offline_rl_missing_buffer_policy: str = "error",
        offline_rl_elite_top_m: int = 8,
        offline_rl_elite_min_candidates: int = 2,
        offline_rl_keep_gt_candidate: bool = True,
        offline_rl_keep_il_candidate: bool = True,
        offline_rl_elite_buffer_version: int = 2,
        offline_rl_require_buffer_valid_mask: bool = True,
        offline_rl_allow_v1_buffer_recompute_valid_mask: bool = True,
        offline_rl_recompute_buffer_valid_mask_on_load: bool = False,
        offline_rl_cache_elite_records_in_memory: bool = False,
        offline_rl_preload_elite_buffer: bool = False,
        offline_rl_elite_buffer_preload_max_records: int = 0,
        offline_rl_build_candidates_online: bool = False,
        offline_rl_online_policy_samples: int = 8,
        offline_rl_online_use_current_policy: bool = True,
        offline_rl_online_use_old_policy: bool = True,
        offline_rl_online_use_gt: bool = True,
        offline_rl_perturb_gt: bool = True,
        offline_rl_perturb_il: bool = True,
        offline_rl_progress_endpoint_deltas_m: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0),
        offline_rl_progress_speed_scales: tuple[float, ...] = (0.95, 1.02, 1.05, 1.08, 1.12),
        offline_rl_progress_time_gammas: tuple[float, ...] = (0.75, 0.85, 0.95, 1.05),
        offline_rl_lateral_offsets_m: tuple[float, ...] = (-0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8),
        offline_rl_endpoint_lateral_offsets_m: tuple[float, ...] = (-0.8, -0.4, 0.4, 0.8),
        offline_rl_timing_slow_first_scales: tuple[float, ...] = (0.7, 0.8, 0.9),
        offline_rl_timing_delay_strengths: tuple[float, ...] = (0.15, 0.25, 0.35),
        offline_rl_strict_reward_submetrics: bool = True,
        offline_rl_required_reward_submetrics: tuple[str, ...] = (
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "time_to_collision_within_bound",
            "driving_direction_compliance",
        ),
        offline_rl_missing_submetric_policy: str = "error",
        offline_rl_use_batched_pdm_scoring: bool = True,
        offline_rl_use_exact_array_pdm_state_conversion: bool = True,
        offline_rl_use_fast_pdm_scorer: bool = True,
        offline_rl_pdm_batch_chunk_size: int = 0,
        offline_rl_pdm_shadow_check: bool = False,
        offline_rl_pdm_shadow_max_samples: int = 4,
        offline_rl_pdm_shadow_max_abs_diff: float = 0.0,
        offline_rl_clip_candidates_to_norm_range: bool = True,
        offline_rl_enforce_forward_monotonic_x: bool = True,
        offline_rl_max_heading_step_rad: float = 0.25,
        offline_rl_max_final_heading_delta_rad: float = 0.4,
        offline_rl_use_final_heading_guard: bool = True,
        offline_rl_require_nc: bool = True,
        offline_rl_require_dac: bool = True,
        offline_rl_require_ddc_guard: bool = True,
        offline_rl_ddc_guard_mode: str = "relative_or_absolute",
        offline_rl_ddc_min_absolute: float = 0.99,
        offline_rl_ddc_max_relative_drop: float = 0.01,
        offline_rl_require_ttc_guard: bool = False,
        offline_rl_ttc_guard_mode: str = "relative_or_absolute",
        offline_rl_ttc_min_absolute: float = 0.95,
        offline_rl_ttc_max_relative_drop: float = 0.02,
        offline_rl_select_valid_topk_only: bool = True,
        offline_rl_train_invalid_fallback_candidates: bool = False,
        offline_rl_fallback_invalid_candidate_weight: float = 0.0,
        offline_rl_prior_distance_weight: float = 0.02,
        offline_rl_jerk_penalty_weight: float = 0.005,
        offline_rl_select_by: str = "pdms_minus_prior",
        offline_rl_baseline_mode: str = "max_gt_il",
        offline_rl_expectile_tau: float = 0.8,
        offline_rl_expectile_iters: int = 20,
        offline_rl_top_mean_frac: float = 0.2,
        offline_rl_advantage_temperature: float = 0.03,
        offline_rl_advantage_clip_min: float = -0.2,
        offline_rl_advantage_clip_max: float = 0.2,
        offline_rl_weight_min: float = 0.05,
        offline_rl_weight_max: float = 20.0,
        offline_rl_normalize_weights_per_scene: bool = True,
        offline_rl_train_only_valid_candidates: bool = True,
        offline_rl_min_reward_margin_to_gt_for_extra_weight: float = 0.0,
        offline_rl_allow_zero_weight_rows: bool = True,
        offline_rl_target_filter_mode: str = "none",
        offline_rl_target_top_k: int = 1,
        offline_rl_target_min_advantage: float = 0.0,
        offline_rl_awac_timestep_sampling: str = "uniform",
        offline_rl_preference_dpo_timestep_sampling: str = "uniform",
        offline_rl_low_noise_timestep_frac: float = 0.35,
        offline_rl_mid_noise_timestep_low_frac: float = 0.15,
        offline_rl_mid_noise_timestep_high_frac: float = 0.65,
        offline_rl_target_blend_mode: str = "none",
        offline_rl_target_blend_alpha: float = 1.0,
        offline_rl_target_blend_schedule: str = "constant",
        offline_rl_target_blend_alpha_start: float = 0.0,
        offline_rl_target_blend_warmup_start_epoch: int = 0,
        offline_rl_target_blend_warmup_epochs: int = 1,
        offline_rl_component_advantage_enabled: bool = False,
        offline_rl_component_progress_weight: float = 0.05,
        offline_rl_component_safety_penalty_weight: float = 0.20,
        offline_rl_component_advantage_clip_min: float = -0.10,
        offline_rl_component_advantage_clip_max: float = 0.05,
        offline_rl_source_balance_enabled: bool = False,
        offline_rl_source_balance_min_factor: float = 0.50,
        offline_rl_source_balance_max_factor: float = 2.00,
        offline_rl_pairwise_rank_loss_weight: float = 0.0,
        offline_rl_pairwise_rank_margin: float = 0.02,
        offline_rl_pairwise_rank_min_reward_gap: float = 0.02,
        offline_rl_pairwise_rank_max_pairs_per_scene: int = 2,
        offline_rl_invalid_repulsion_loss_weight: float = 0.0,
        offline_rl_invalid_repulsion_margin: float = 0.05,
        offline_rl_invalid_repulsion_max_pairs_per_scene: int = 2,
        offline_rl_preference_dpo_loss_weight: float = 0.0,
        offline_rl_preference_dpo_beta: float = 8.0,
        offline_rl_preference_dpo_label_smoothing: float = 0.0,
        offline_rl_preference_dpo_reference_free: bool = False,
        offline_rl_preference_dpo_pair_mode: str = "best_vs_gt_il",
        offline_rl_preference_dpo_min_reward_gap: float = 0.02,
        offline_rl_preference_dpo_max_pairs_per_scene: int = 2,
        offline_rl_preference_dpo_gap_weight_mode: str = "none",
        offline_rl_preference_dpo_gap_weight_scale: float = 0.05,
        offline_rl_preference_dpo_gap_weight_min: float = 0.0,
        offline_rl_preference_dpo_gap_weight_max: float = 3.0,
        offline_rl_awac_loss_weight: float = 1.0,
        offline_rl_awac_loss_schedule: str = "constant",
        offline_rl_awac_loss_weight_start: float = 0.0,
        offline_rl_awac_loss_warmup_start_epoch: int = 0,
        offline_rl_awac_loss_warmup_epochs: int = 1,
        offline_rl_preference_dpo_loss_schedule: str = "constant",
        offline_rl_preference_dpo_loss_weight_start: float = 0.0,
        offline_rl_preference_dpo_loss_warmup_start_epoch: int = 0,
        offline_rl_preference_dpo_loss_warmup_epochs: int = 1,
        offline_rl_bc_loss_weight: float = 0.05,
        offline_rl_bc_loss_schedule: str = "constant",
        offline_rl_bc_loss_weight_start: float = 0.05,
        offline_rl_bc_loss_weight_end: float = 0.05,
        offline_rl_bc_loss_schedule_epochs: int = 1,
        offline_rl_grpo_loss_weight: float = 0.0,
        offline_rl_grpo_loss_schedule: str = "constant",
        offline_rl_grpo_loss_weight_start: float = 0.0,
        offline_rl_grpo_loss_warmup_start_epoch: int = 0,
        offline_rl_grpo_loss_warmup_epochs: int = 1,
        offline_rl_grpo_buffer_guidance_enabled: bool = False,
        offline_rl_grpo_buffer_reward_bonus_weight: float = 0.0,
        offline_rl_grpo_buffer_reward_bonus_scale_m: float = 4.0,
        offline_rl_grpo_buffer_reward_bonus_use_margin: bool = True,
        offline_rl_grpo_buffer_distill_loss_weight: float = 0.0,
        offline_rl_grpo_buffer_distill_loss_schedule: str = "linear_warmup",
        offline_rl_grpo_buffer_distill_loss_weight_start: float = 0.0,
        offline_rl_grpo_buffer_distill_warmup_start_epoch: int = 0,
        offline_rl_grpo_buffer_distill_warmup_epochs: int = 3,
        offline_rl_grpo_buffer_distill_top_k: int = 1,
        offline_rl_grpo_buffer_distill_min_reward_margin: float = 0.0,
        offline_rl_grpo_buffer_distill_timestep_sampling: str = "low_noise",
        offline_rl_grpo_self_imitation_loss_weight: float = 0.0,
        offline_rl_grpo_self_imitation_loss_schedule: str = "linear_warmup",
        offline_rl_grpo_self_imitation_loss_weight_start: float = 0.0,
        offline_rl_grpo_self_imitation_warmup_start_epoch: int = 0,
        offline_rl_grpo_self_imitation_warmup_epochs: int = 3,
        offline_rl_grpo_self_imitation_top_k: int = 1,
        offline_rl_grpo_self_imitation_min_reward: float = 0.85,
        offline_rl_grpo_self_imitation_min_reward_margin: float = 0.01,
        offline_rl_grpo_self_imitation_max_target_scene_ratio: float = 1.0,
        offline_rl_grpo_self_imitation_batch_cap_score: str = "reward",
        offline_rl_grpo_self_imitation_baseline_mode: str = "buffer_or_group_mean",
        offline_rl_grpo_self_imitation_timestep_sampling: str = "low_noise",
        offline_rl_grpo_self_imitation_require_nc: bool = True,
        offline_rl_grpo_self_imitation_require_dac: bool = True,
        offline_rl_grpo_self_imitation_require_ttc: bool = True,
        offline_rl_grpo_self_imitation_require_ddc: bool = True,
        offline_rl_grpo_self_imitation_nc_min_absolute: float = 1.0,
        offline_rl_grpo_self_imitation_dac_min_absolute: float = 1.0,
        offline_rl_grpo_self_imitation_ttc_min_absolute: float = 0.95,
        offline_rl_grpo_self_imitation_ddc_min_absolute: float = 0.99,
        offline_rl_log_candidate_sources: bool = True,
        offline_rl_log_submetrics: bool = True,
        offline_rl_log_oracle_stats: bool = True,
        offline_rl_require_reference_policy_checkpoint: bool = True,
        offline_rl_report_raw_and_valid_best: bool = True,
        vlm_size: Optional[str] = 'small', 
        train_backbone: bool = False,
        use_expert_features: bool = False,
        expert_feature_source: str = "none",
        expert_cache_dir: Optional[str] = None,
        chunk_cache_dir: Optional[str] = None,
        allow_dummy_cache: bool = False,
        expert_adapter_dim: int = 768,
        num_jepa_tokens: int = 12,
        num_vggt_tokens: int = 12,
        allow_expert_target_features: bool = False,
        allow_random_init: bool = True,
        allow_dummy_expert_cache: bool = False,
        use_jepa: bool = True,
        use_vggt: bool = True,
        jepa_dim: int = 1024,
        vggt_dim: int = 2048,
        expert_dropout: float = 0.10,
        expert_fusion_mode: str = "concat_context",
        use_teacher_context_tokens: bool = True,
        use_student_latent_adapters: bool = True,
        use_branch_weighted_mean: bool = True,
        use_expert_type_embedding: bool = True,
        use_expert_gates: bool = True,
        expert_alignment_weight: float = 0.0,
        expert_stream_dropout: float = 0.0,
        expert_context_scale: float = 1.0,
        use_horizon_expert_residual: bool = False,
        expert_horizon_residual_scale: float = 0.0,
        diffusion_loss_weight: float = 1.0,
        use_alignment_loss: bool = True,
        jepa_alignment_weight: float = 0.03,
        vggt_alignment_weight: float = 0.05,
        alignment_loss_type: str = "normalized_mse",
        jepa_gate_init: float = 0.05,
        vggt_gate_init: float = 0.05,
        branch_init_vlm: float = 0.90,
        branch_init_jepa: float = 0.05,
        branch_init_vggt: float = 0.05,
        allow_future_targets_in_inference: bool = False,
        use_last_rd: bool = False,
        last_rd_stage: str = "disabled",
        use_future_jepa_prediction: bool = True,
        use_vggt_geometry_tokens: bool = True,
        use_ego_trajectory_tokens: bool = True,
        use_risk_tokens: bool = True,
        use_scene_aware_expert_gate: bool = True,
        use_timestep_aware_expert_gate: bool = True,
        last_rd_latent_dim: int = 384,
        num_dynamic_tokens: int = 12,
        num_geometry_tokens: int = 12,
        num_ego_tokens: int = 8,
        num_risk_tokens: int = 8,
        require_vggt_geometry: bool = False,
        allow_patch_geometry_fallback: bool = True,
        future_jepa_loss_weight: float = 0.0,
        vggt_geometry_loss_weight: float = 0.0,
        coarse_traj_loss_weight: float = 0.0,
        coarse_heading_loss_weight: float = 0.0,
        risk_loss_weight: float = 0.0,
        policy_kd_loss_weight: float = 0.0,
        future_jepa_loss_floor: float = 0.0,
        vggt_geometry_loss_floor: float = 0.0,
        coarse_traj_loss_floor: float = 0.0,
        risk_loss_floor: float = 0.0,
        last_rd_context_scale: float = 1.0,
        last_rd_horizon_condition_scale: float = 1.0,
        last_rd_token_dropout: float = 0.0,
        last_rd_group_dropout: float = 0.0,
        reference_a0_checkpoint: Optional[str] = None,
        last_rd_adapter_checkpoint: Optional[str] = None,
        policy_kd_mode: str = "none",
        current_train_epoch: int = 0,
        total_train_epochs: int = 200,
        use_last_vla: bool = False,
        last_vla_stage: str = "disabled",
        last_vla_cot_num_tokens: int = 32,
        last_vla_cot_num_steps: int = 4,
        last_vla_condition_mode: str = "decoupled_cot_residual",
        last_vla_raw_vlm_context_to_dit: bool = True,
        last_vla_cot_bottleneck_mode: bool = False,
        last_vla_vlm_context_dropout_start: float = 0.0,
        last_vla_vlm_context_dropout_end: float = 0.7,
        last_vla_use_scene_step: bool = True,
        last_vla_use_parallel_geometry_dynamic: bool = True,
        last_vla_fusion_tokens: int = 192,
        last_vla_dynamic_uses_geometry_memory: bool = True,
        last_vla_use_geometry_step: bool = True,
        last_vla_use_dynamic_step: bool = True,
        last_vla_use_ego_step: bool = True,
        last_vla_use_action_refine_step: bool = True,
        last_vla_use_action_conditioned_dynamics: bool = True,
        last_vla_use_risk_head: bool = True,
        last_vla_require_full_geometry: bool = False,
        last_vla_allow_patch_geometry_fallback: bool = False,
        last_vla_geometry_teacher_dim: int = 512,
        last_vla_geometry_grid_rows: int = 3,
        last_vla_geometry_grid_cols: int = 4,
        last_vla_use_residual_diffusion: bool = False,
        last_vla_residual_detach_coarse: bool = True,
        last_vla_residual_anchor_source: str = "vlm_text_traj",
        last_vla_require_residual_anchor: bool = True,
        last_vla_residual_anchor_cache_dir: Optional[str] = None,
        last_vla_coarse_prior_clip: float = 1.0,
        last_vla_residual_alpha_start: float = 0.0,
        last_vla_residual_alpha_end: float = 1.0,
        last_vla_residual_alpha_warmup_epochs: int = 80,
        last_vla_cot_condition_zero_init: bool = True,
        last_vla_cot_condition_dropout: float = 0.0,
        last_vla_cot_condition_layers: str = "all",
        last_vla_cot_condition_scale_init: float = 1.0,
        last_vla_cot_condition_trainable_scale: bool = True,
        last_vla_context_mean_mode: str = "raw_vlm_plus_zero_init_cot_residual",
        last_vla_horizon_condition_mode: str = "raw_vlm_plus_zero_init_cot_residual",
        last_vla_aux_decay_epochs: int = 160,
        last_vla_teacher_traj_mode: str = "none",
        last_vla_teacher_traj_mix_start: float = 0.0,
        last_vla_teacher_traj_mix_end: float = 1.0,
        last_vla_teacher_score_margin: float = 0.0,
        last_vla_geometry_loss_weight: float = 0.0,
        last_vla_dynamic_loss_weight: float = 0.0,
        last_vla_coarse_loss_weight: float = 0.0,
        last_vla_heading_loss_weight: float = 0.0,
        last_vla_progress_loss_weight: float = 0.0,
        last_vla_risk_loss_weight: float = 0.0,
        last_vla_cot_consistency_loss_weight: float = 0.0,
        last_vla_geometry_loss_floor: float = 0.0,
        last_vla_dynamic_loss_floor: float = 0.0,
        last_vla_coarse_loss_floor: float = 0.0,
        last_vla_progress_loss_floor: float = 0.0,
        last_vla_adapter_checkpoint: Optional[str] = None,
        last_vla_train_vlm_lora: bool = False,
        last_vla_vlm_lora_preset: str = "attention_mlp",
        last_vla_vlm_lora_scope: str = "llm",
        last_vla_vlm_lora_r: int = 32,
        last_vla_vlm_lora_alpha: int = 64,
        last_vla_vlm_lora_dropout: float = 0.05,
        last_vla_vlm_lora_bias: str = "none",
        last_vla_vlm_lora_target_modules: str = "",
        last_vla_vlm_lora_use_rslora: bool = True,
        last_vla_vlm_lora_use_dora: bool = False,
        last_vla_vlm_lora_init: str = "default",
        last_vla_vlm_lora_vision_last_n: int = 0,
        last_vla_lora_allow_mixed_scope: bool = False,
        last_vla_lora_allow_all_linear_global: bool = False,
        use_two_expert_slots: bool = False,
        two_expert_cache_mode: bool = True,
        two_expert_slot_mode: str = "vlm_soft_slots",
        two_expert_condition_mode: str = "horizon_hmef_lite",
        two_expert_dit_condition_mode: str = "horizon_hmef_lite",
        two_expert_planner_dim: int = 384,
        two_expert_use_raw_vlm_base: bool = True,
        two_expert_zero_init_deltas: bool = True,
        two_expert_dyn_loss_floor: float = 0.0,
        two_expert_geo_loss_floor: float = 0.0,
        two_expert_num_dyn_groups: int = 3,
        two_expert_dyn_tokens_per_group: int = 12,
        two_expert_num_geo_tokens: int = 12,
        two_expert_memory_tokens_to_dit: bool = False,
        two_expert_denoise_gate_hidden_dim: int = 384,
        two_expert_denoise_gate_temperature: float = 1.0,
        two_expert_denoise_condition_scale_init: float = 1.0,
        two_expert_memory_scale_init: float = 1.0,
        lr_vlm_lora: Optional[float] = 1e-5,
        weight_decay_vlm_lora: float = 0.0,
        lr_last_vla_cot: Optional[float] = 1e-4,
        weight_decay_last_vla_cot: float = 1e-4,
        last_vla_hidden_anchor_weight: float = 0.01,
        last_vla_hidden_anchor_mode: str = "summary_cosine",
        last_vla_hidden_anchor_every_n_steps: int = 4,
        last_vla_log_lora_diagnostics: bool = True,
        lr_action_head: Optional[float] = None,
        lr_expert: Optional[float] = None,
        lr_expert_gate: Optional[float] = None,
        train_expert_only: bool = False,
        freeze_base_action_head: bool = False,
        freeze_expert: bool = False,
        scheduler_epochs: int = 200,
        scheduler_warmup_epochs: int = 3,
        scheduler_min_lr: float = 1e-6,
        grpo_scheduler_epochs: int = 10,
        grpo_scheduler_warmup_epochs: int = 0,
        grpo_scheduler_min_lr: float = 0.0,
    ):
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        self.vlm_path = vlm_path
        self.checkpoint_path = checkpoint_path
        self.vlm_type = vlm_type
        self.dit_type = dit_type
        self.cache_mode = cache_mode
        self.cache_hidden_state = cache_hidden_state
        self._lr = lr
        if stage3_objective not in {"none", "grpo", "grpo_replay", "awac_iql", "hybrid"}:
            raise ValueError(
                "stage3_objective must be one of 'none', 'grpo', 'grpo_replay', 'awac_iql', or 'hybrid'."
            )
        resolved_stage3_objective = str(stage3_objective)
        if resolved_stage3_objective == "none" and bool(grpo):
            resolved_stage3_objective = "grpo"
        if resolved_stage3_objective in {"grpo", "grpo_replay"}:
            grpo = True
        if resolved_stage3_objective in {"awac_iql", "hybrid"}:
            offline_rl_enabled = True
        if resolved_stage3_objective == "hybrid" and float(offline_rl_grpo_loss_weight) > 0.0:
            grpo = True
        self.stage3_objective = resolved_stage3_objective
        self.offline_rl_enabled = bool(offline_rl_enabled)
        self.grpo = bool(grpo)
        self.grpo_sample_time = int(grpo_sample_time)
        if self.grpo_sample_time <= 0:
            raise ValueError("grpo_sample_time must be positive.")
        self.bc_anneal = bool(bc_anneal)
        self.bc_coeff_start = float(bc_coeff_start)
        self.bc_coeff_end = float(bc_coeff_end)
        self.bc_anneal_epochs = int(bc_anneal_epochs)
        self.reference_kl_coeff = float(reference_kl_coeff)
        self.reference_kl_chunk_size = int(reference_kl_chunk_size)
        self.grpo_use_gspo_ratio = bool(grpo_use_gspo_ratio)
        self.grpo_gspo_clip_low = float(grpo_gspo_clip_low)
        self.grpo_gspo_clip_high = float(grpo_gspo_clip_high)
        self.grpo_behavior_policy_sync_interval = int(grpo_behavior_policy_sync_interval)
        self.grpo_behavior_policy_sample = bool(grpo_behavior_policy_sample)
        self.grpo_normalize_advantage_batch = bool(grpo_normalize_advantage_batch)
        self.grpo_advantage_clip_abs = float(grpo_advantage_clip_abs)
        self.grpo_ppo_replay_inner_epochs = int(grpo_ppo_replay_inner_epochs)
        self.grpo_ppo_replay_minibatch_size = int(grpo_ppo_replay_minibatch_size)
        self.grpo_ppo_replay_max_grad_norm = float(grpo_ppo_replay_max_grad_norm)
        self.grpo_ppo_replay_min_abs_advantage = float(grpo_ppo_replay_min_abs_advantage)
        self.grpo_ppo_replay_filter_zero_advantage = bool(grpo_ppo_replay_filter_zero_advantage)
        self.grpo_ppo_replay_sync_behavior_each_batch = bool(grpo_ppo_replay_sync_behavior_each_batch)
        self.grpo_ppo_replay_bc_update = bool(grpo_ppo_replay_bc_update)
        self.grpo_ppo_replay_logprob_mode = str(grpo_ppo_replay_logprob_mode)
        self.grpo_ppo_replay_step_minibatch_mode = str(grpo_ppo_replay_step_minibatch_mode)
        self.grpo_ppo_replay_logprob_clamp_min = float(grpo_ppo_replay_logprob_clamp_min)
        self.grpo_ppo_replay_logprob_clamp_max = float(grpo_ppo_replay_logprob_clamp_max)
        self.grpo_ppo_replay_step_clip_schedule = str(grpo_ppo_replay_step_clip_schedule)
        self.grpo_ppo_replay_step_clip_base = float(grpo_ppo_replay_step_clip_base)
        self.grpo_ppo_replay_step_clip_rate = float(grpo_ppo_replay_step_clip_rate)
        if self.bc_coeff_start < 0.0 or self.bc_coeff_end < 0.0:
            raise ValueError("BC coefficients must be non-negative.")
        if self.bc_anneal_epochs <= 0:
            raise ValueError("bc_anneal_epochs must be positive.")
        if self.reference_kl_coeff < 0.0:
            raise ValueError("reference_kl_coeff must be non-negative.")
        if self.reference_kl_chunk_size < 0:
            raise ValueError("reference_kl_chunk_size must be non-negative.")
        if not (0.0 <= self.grpo_gspo_clip_low < 1.0):
            raise ValueError("grpo_gspo_clip_low must be in [0, 1).")
        if self.grpo_gspo_clip_high < 0.0:
            raise ValueError("grpo_gspo_clip_high must be non-negative.")
        if self.grpo_behavior_policy_sync_interval <= 0:
            raise ValueError("grpo_behavior_policy_sync_interval must be positive.")
        if self.grpo_advantage_clip_abs < 0.0:
            raise ValueError("grpo_advantage_clip_abs must be non-negative.")
        if self.grpo_ppo_replay_inner_epochs <= 0:
            raise ValueError("grpo_ppo_replay_inner_epochs must be positive.")
        if self.grpo_ppo_replay_minibatch_size < 0:
            raise ValueError("grpo_ppo_replay_minibatch_size must be non-negative.")
        if self.grpo_ppo_replay_max_grad_norm < 0.0:
            raise ValueError("grpo_ppo_replay_max_grad_norm must be non-negative.")
        if self.grpo_ppo_replay_min_abs_advantage < 0.0:
            raise ValueError("grpo_ppo_replay_min_abs_advantage must be non-negative.")
        if self.grpo_ppo_replay_logprob_mode not in {"trajectory", "step"}:
            raise ValueError("grpo_ppo_replay_logprob_mode must be either 'trajectory' or 'step'.")
        if self.grpo_ppo_replay_step_minibatch_mode not in {"trajectory_all_steps", "transition"}:
            raise ValueError(
                "grpo_ppo_replay_step_minibatch_mode must be 'trajectory_all_steps' or 'transition'."
            )
        if self.grpo_ppo_replay_logprob_clamp_max < self.grpo_ppo_replay_logprob_clamp_min:
            raise ValueError("grpo_ppo_replay_logprob_clamp_max must be >= grpo_ppo_replay_logprob_clamp_min.")
        if self.grpo_ppo_replay_step_clip_schedule not in {"constant", "dppo_exp"}:
            raise ValueError("grpo_ppo_replay_step_clip_schedule must be 'constant' or 'dppo_exp'.")
        if self.grpo_ppo_replay_step_clip_base < 0.0:
            raise ValueError("grpo_ppo_replay_step_clip_base must be non-negative.")
        if self.grpo_ppo_replay_step_clip_rate < 0.0:
            raise ValueError("grpo_ppo_replay_step_clip_rate must be non-negative.")
        self.backbone = None
        self.metric_cache_path = metric_cache_path
        self.reference_policy_checkpoint = reference_policy_checkpoint
        self.offline_rl_elite_buffer_path = offline_rl_elite_buffer_path
        self.offline_rl_missing_buffer_policy = offline_rl_missing_buffer_policy
        self.offline_rl_elite_top_m = int(offline_rl_elite_top_m)
        self.offline_rl_elite_min_candidates = int(offline_rl_elite_min_candidates)
        self.offline_rl_keep_gt_candidate = bool(offline_rl_keep_gt_candidate)
        self.offline_rl_keep_il_candidate = bool(offline_rl_keep_il_candidate)
        self.offline_rl_elite_buffer_version = int(offline_rl_elite_buffer_version)
        self.offline_rl_require_buffer_valid_mask = bool(offline_rl_require_buffer_valid_mask)
        self.offline_rl_allow_v1_buffer_recompute_valid_mask = bool(offline_rl_allow_v1_buffer_recompute_valid_mask)
        self.offline_rl_recompute_buffer_valid_mask_on_load = bool(offline_rl_recompute_buffer_valid_mask_on_load)
        self.offline_rl_cache_elite_records_in_memory = bool(offline_rl_cache_elite_records_in_memory)
        self.offline_rl_preload_elite_buffer = bool(offline_rl_preload_elite_buffer)
        self.offline_rl_elite_buffer_preload_max_records = int(offline_rl_elite_buffer_preload_max_records)
        self.offline_rl_build_candidates_online = bool(offline_rl_build_candidates_online)
        self.offline_rl_online_policy_samples = int(offline_rl_online_policy_samples)
        self.offline_rl_online_use_current_policy = bool(offline_rl_online_use_current_policy)
        self.offline_rl_online_use_old_policy = bool(offline_rl_online_use_old_policy)
        self.offline_rl_online_use_gt = bool(offline_rl_online_use_gt)
        self.offline_rl_perturb_gt = bool(offline_rl_perturb_gt)
        self.offline_rl_perturb_il = bool(offline_rl_perturb_il)
        self.offline_rl_progress_endpoint_deltas_m = tuple(float(x) for x in offline_rl_progress_endpoint_deltas_m)
        self.offline_rl_progress_speed_scales = tuple(float(x) for x in offline_rl_progress_speed_scales)
        self.offline_rl_progress_time_gammas = tuple(float(x) for x in offline_rl_progress_time_gammas)
        self.offline_rl_lateral_offsets_m = tuple(float(x) for x in offline_rl_lateral_offsets_m)
        self.offline_rl_endpoint_lateral_offsets_m = tuple(float(x) for x in offline_rl_endpoint_lateral_offsets_m)
        self.offline_rl_timing_slow_first_scales = tuple(float(x) for x in offline_rl_timing_slow_first_scales)
        self.offline_rl_timing_delay_strengths = tuple(float(x) for x in offline_rl_timing_delay_strengths)
        self.offline_rl_strict_reward_submetrics = bool(offline_rl_strict_reward_submetrics)
        self.offline_rl_required_reward_submetrics = tuple(str(x) for x in offline_rl_required_reward_submetrics)
        self.offline_rl_missing_submetric_policy = offline_rl_missing_submetric_policy
        self.offline_rl_use_batched_pdm_scoring = bool(offline_rl_use_batched_pdm_scoring)
        self.offline_rl_use_exact_array_pdm_state_conversion = bool(offline_rl_use_exact_array_pdm_state_conversion)
        self.offline_rl_use_fast_pdm_scorer = bool(offline_rl_use_fast_pdm_scorer)
        self.offline_rl_pdm_batch_chunk_size = int(offline_rl_pdm_batch_chunk_size)
        self.offline_rl_pdm_shadow_check = bool(offline_rl_pdm_shadow_check)
        self.offline_rl_pdm_shadow_max_samples = int(offline_rl_pdm_shadow_max_samples)
        self.offline_rl_pdm_shadow_max_abs_diff = float(offline_rl_pdm_shadow_max_abs_diff)
        self.offline_rl_clip_candidates_to_norm_range = bool(offline_rl_clip_candidates_to_norm_range)
        self.offline_rl_enforce_forward_monotonic_x = bool(offline_rl_enforce_forward_monotonic_x)
        self.offline_rl_max_heading_step_rad = float(offline_rl_max_heading_step_rad)
        self.offline_rl_max_final_heading_delta_rad = float(offline_rl_max_final_heading_delta_rad)
        self.offline_rl_use_final_heading_guard = bool(offline_rl_use_final_heading_guard)
        self.offline_rl_require_nc = bool(offline_rl_require_nc)
        self.offline_rl_require_dac = bool(offline_rl_require_dac)
        self.offline_rl_require_ddc_guard = bool(offline_rl_require_ddc_guard)
        self.offline_rl_ddc_guard_mode = offline_rl_ddc_guard_mode
        self.offline_rl_ddc_min_absolute = float(offline_rl_ddc_min_absolute)
        self.offline_rl_ddc_max_relative_drop = float(offline_rl_ddc_max_relative_drop)
        self.offline_rl_require_ttc_guard = bool(offline_rl_require_ttc_guard)
        self.offline_rl_ttc_guard_mode = offline_rl_ttc_guard_mode
        self.offline_rl_ttc_min_absolute = float(offline_rl_ttc_min_absolute)
        self.offline_rl_ttc_max_relative_drop = float(offline_rl_ttc_max_relative_drop)
        self.offline_rl_select_valid_topk_only = bool(offline_rl_select_valid_topk_only)
        self.offline_rl_train_invalid_fallback_candidates = bool(offline_rl_train_invalid_fallback_candidates)
        self.offline_rl_fallback_invalid_candidate_weight = float(offline_rl_fallback_invalid_candidate_weight)
        self.offline_rl_prior_distance_weight = float(offline_rl_prior_distance_weight)
        self.offline_rl_jerk_penalty_weight = float(offline_rl_jerk_penalty_weight)
        self.offline_rl_select_by = offline_rl_select_by
        self.offline_rl_baseline_mode = offline_rl_baseline_mode
        self.offline_rl_expectile_tau = float(offline_rl_expectile_tau)
        self.offline_rl_expectile_iters = int(offline_rl_expectile_iters)
        self.offline_rl_top_mean_frac = float(offline_rl_top_mean_frac)
        self.offline_rl_advantage_temperature = float(offline_rl_advantage_temperature)
        self.offline_rl_advantage_clip_min = float(offline_rl_advantage_clip_min)
        self.offline_rl_advantage_clip_max = float(offline_rl_advantage_clip_max)
        self.offline_rl_weight_min = float(offline_rl_weight_min)
        self.offline_rl_weight_max = float(offline_rl_weight_max)
        self.offline_rl_normalize_weights_per_scene = bool(offline_rl_normalize_weights_per_scene)
        self.offline_rl_train_only_valid_candidates = bool(offline_rl_train_only_valid_candidates)
        self.offline_rl_min_reward_margin_to_gt_for_extra_weight = float(
            offline_rl_min_reward_margin_to_gt_for_extra_weight
        )
        self.offline_rl_allow_zero_weight_rows = bool(offline_rl_allow_zero_weight_rows)
        self.offline_rl_target_filter_mode = offline_rl_target_filter_mode
        self.offline_rl_target_top_k = int(offline_rl_target_top_k)
        self.offline_rl_target_min_advantage = float(offline_rl_target_min_advantage)
        self.offline_rl_awac_timestep_sampling = offline_rl_awac_timestep_sampling
        self.offline_rl_preference_dpo_timestep_sampling = offline_rl_preference_dpo_timestep_sampling
        self.offline_rl_low_noise_timestep_frac = float(offline_rl_low_noise_timestep_frac)
        self.offline_rl_mid_noise_timestep_low_frac = float(offline_rl_mid_noise_timestep_low_frac)
        self.offline_rl_mid_noise_timestep_high_frac = float(offline_rl_mid_noise_timestep_high_frac)
        self.offline_rl_target_blend_mode = offline_rl_target_blend_mode
        self.offline_rl_target_blend_alpha = float(offline_rl_target_blend_alpha)
        self.offline_rl_target_blend_schedule = offline_rl_target_blend_schedule
        self.offline_rl_target_blend_alpha_start = float(offline_rl_target_blend_alpha_start)
        self.offline_rl_target_blend_warmup_start_epoch = int(offline_rl_target_blend_warmup_start_epoch)
        self.offline_rl_target_blend_warmup_epochs = int(offline_rl_target_blend_warmup_epochs)
        self.offline_rl_component_advantage_enabled = bool(offline_rl_component_advantage_enabled)
        self.offline_rl_component_progress_weight = float(offline_rl_component_progress_weight)
        self.offline_rl_component_safety_penalty_weight = float(offline_rl_component_safety_penalty_weight)
        self.offline_rl_component_advantage_clip_min = float(offline_rl_component_advantage_clip_min)
        self.offline_rl_component_advantage_clip_max = float(offline_rl_component_advantage_clip_max)
        self.offline_rl_source_balance_enabled = bool(offline_rl_source_balance_enabled)
        self.offline_rl_source_balance_min_factor = float(offline_rl_source_balance_min_factor)
        self.offline_rl_source_balance_max_factor = float(offline_rl_source_balance_max_factor)
        self.offline_rl_pairwise_rank_loss_weight = float(offline_rl_pairwise_rank_loss_weight)
        self.offline_rl_pairwise_rank_margin = float(offline_rl_pairwise_rank_margin)
        self.offline_rl_pairwise_rank_min_reward_gap = float(offline_rl_pairwise_rank_min_reward_gap)
        self.offline_rl_pairwise_rank_max_pairs_per_scene = int(offline_rl_pairwise_rank_max_pairs_per_scene)
        self.offline_rl_invalid_repulsion_loss_weight = float(offline_rl_invalid_repulsion_loss_weight)
        self.offline_rl_invalid_repulsion_margin = float(offline_rl_invalid_repulsion_margin)
        self.offline_rl_invalid_repulsion_max_pairs_per_scene = int(offline_rl_invalid_repulsion_max_pairs_per_scene)
        self.offline_rl_preference_dpo_loss_weight = float(offline_rl_preference_dpo_loss_weight)
        self.offline_rl_preference_dpo_beta = float(offline_rl_preference_dpo_beta)
        self.offline_rl_preference_dpo_label_smoothing = float(offline_rl_preference_dpo_label_smoothing)
        self.offline_rl_preference_dpo_reference_free = bool(offline_rl_preference_dpo_reference_free)
        self.offline_rl_preference_dpo_pair_mode = offline_rl_preference_dpo_pair_mode
        self.offline_rl_preference_dpo_min_reward_gap = float(offline_rl_preference_dpo_min_reward_gap)
        self.offline_rl_preference_dpo_max_pairs_per_scene = int(offline_rl_preference_dpo_max_pairs_per_scene)
        self.offline_rl_preference_dpo_gap_weight_mode = offline_rl_preference_dpo_gap_weight_mode
        self.offline_rl_preference_dpo_gap_weight_scale = float(offline_rl_preference_dpo_gap_weight_scale)
        self.offline_rl_preference_dpo_gap_weight_min = float(offline_rl_preference_dpo_gap_weight_min)
        self.offline_rl_preference_dpo_gap_weight_max = float(offline_rl_preference_dpo_gap_weight_max)
        self.offline_rl_awac_loss_weight = float(offline_rl_awac_loss_weight)
        self.offline_rl_awac_loss_schedule = offline_rl_awac_loss_schedule
        self.offline_rl_awac_loss_weight_start = float(offline_rl_awac_loss_weight_start)
        self.offline_rl_awac_loss_warmup_start_epoch = int(offline_rl_awac_loss_warmup_start_epoch)
        self.offline_rl_awac_loss_warmup_epochs = int(offline_rl_awac_loss_warmup_epochs)
        self.offline_rl_preference_dpo_loss_schedule = offline_rl_preference_dpo_loss_schedule
        self.offline_rl_preference_dpo_loss_weight_start = float(offline_rl_preference_dpo_loss_weight_start)
        self.offline_rl_preference_dpo_loss_warmup_start_epoch = int(
            offline_rl_preference_dpo_loss_warmup_start_epoch
        )
        self.offline_rl_preference_dpo_loss_warmup_epochs = int(offline_rl_preference_dpo_loss_warmup_epochs)
        self.offline_rl_bc_loss_weight = float(offline_rl_bc_loss_weight)
        self.offline_rl_bc_loss_schedule = offline_rl_bc_loss_schedule
        self.offline_rl_bc_loss_weight_start = float(offline_rl_bc_loss_weight_start)
        self.offline_rl_bc_loss_weight_end = float(offline_rl_bc_loss_weight_end)
        self.offline_rl_bc_loss_schedule_epochs = int(offline_rl_bc_loss_schedule_epochs)
        self.offline_rl_grpo_loss_weight = float(offline_rl_grpo_loss_weight)
        self.offline_rl_grpo_loss_schedule = offline_rl_grpo_loss_schedule
        self.offline_rl_grpo_loss_weight_start = float(offline_rl_grpo_loss_weight_start)
        self.offline_rl_grpo_loss_warmup_start_epoch = int(offline_rl_grpo_loss_warmup_start_epoch)
        self.offline_rl_grpo_loss_warmup_epochs = int(offline_rl_grpo_loss_warmup_epochs)
        self.offline_rl_grpo_buffer_guidance_enabled = bool(offline_rl_grpo_buffer_guidance_enabled)
        self.offline_rl_grpo_buffer_reward_bonus_weight = float(offline_rl_grpo_buffer_reward_bonus_weight)
        self.offline_rl_grpo_buffer_reward_bonus_scale_m = float(offline_rl_grpo_buffer_reward_bonus_scale_m)
        self.offline_rl_grpo_buffer_reward_bonus_use_margin = bool(offline_rl_grpo_buffer_reward_bonus_use_margin)
        self.offline_rl_grpo_buffer_distill_loss_weight = float(offline_rl_grpo_buffer_distill_loss_weight)
        self.offline_rl_grpo_buffer_distill_loss_schedule = offline_rl_grpo_buffer_distill_loss_schedule
        self.offline_rl_grpo_buffer_distill_loss_weight_start = float(
            offline_rl_grpo_buffer_distill_loss_weight_start
        )
        self.offline_rl_grpo_buffer_distill_warmup_start_epoch = int(
            offline_rl_grpo_buffer_distill_warmup_start_epoch
        )
        self.offline_rl_grpo_buffer_distill_warmup_epochs = int(offline_rl_grpo_buffer_distill_warmup_epochs)
        self.offline_rl_grpo_buffer_distill_top_k = int(offline_rl_grpo_buffer_distill_top_k)
        self.offline_rl_grpo_buffer_distill_min_reward_margin = float(
            offline_rl_grpo_buffer_distill_min_reward_margin
        )
        self.offline_rl_grpo_buffer_distill_timestep_sampling = offline_rl_grpo_buffer_distill_timestep_sampling
        self.offline_rl_grpo_self_imitation_loss_weight = float(offline_rl_grpo_self_imitation_loss_weight)
        self.offline_rl_grpo_self_imitation_loss_schedule = offline_rl_grpo_self_imitation_loss_schedule
        self.offline_rl_grpo_self_imitation_loss_weight_start = float(
            offline_rl_grpo_self_imitation_loss_weight_start
        )
        self.offline_rl_grpo_self_imitation_warmup_start_epoch = int(
            offline_rl_grpo_self_imitation_warmup_start_epoch
        )
        self.offline_rl_grpo_self_imitation_warmup_epochs = int(offline_rl_grpo_self_imitation_warmup_epochs)
        self.offline_rl_grpo_self_imitation_top_k = int(offline_rl_grpo_self_imitation_top_k)
        self.offline_rl_grpo_self_imitation_min_reward = float(offline_rl_grpo_self_imitation_min_reward)
        self.offline_rl_grpo_self_imitation_min_reward_margin = float(
            offline_rl_grpo_self_imitation_min_reward_margin
        )
        self.offline_rl_grpo_self_imitation_max_target_scene_ratio = float(
            offline_rl_grpo_self_imitation_max_target_scene_ratio
        )
        self.offline_rl_grpo_self_imitation_batch_cap_score = offline_rl_grpo_self_imitation_batch_cap_score
        self.offline_rl_grpo_self_imitation_baseline_mode = offline_rl_grpo_self_imitation_baseline_mode
        self.offline_rl_grpo_self_imitation_timestep_sampling = offline_rl_grpo_self_imitation_timestep_sampling
        self.offline_rl_grpo_self_imitation_require_nc = bool(offline_rl_grpo_self_imitation_require_nc)
        self.offline_rl_grpo_self_imitation_require_dac = bool(offline_rl_grpo_self_imitation_require_dac)
        self.offline_rl_grpo_self_imitation_require_ttc = bool(offline_rl_grpo_self_imitation_require_ttc)
        self.offline_rl_grpo_self_imitation_require_ddc = bool(offline_rl_grpo_self_imitation_require_ddc)
        self.offline_rl_grpo_self_imitation_nc_min_absolute = float(
            offline_rl_grpo_self_imitation_nc_min_absolute
        )
        self.offline_rl_grpo_self_imitation_dac_min_absolute = float(
            offline_rl_grpo_self_imitation_dac_min_absolute
        )
        self.offline_rl_grpo_self_imitation_ttc_min_absolute = float(
            offline_rl_grpo_self_imitation_ttc_min_absolute
        )
        self.offline_rl_grpo_self_imitation_ddc_min_absolute = float(
            offline_rl_grpo_self_imitation_ddc_min_absolute
        )
        self.offline_rl_log_candidate_sources = bool(offline_rl_log_candidate_sources)
        self.offline_rl_log_submetrics = bool(offline_rl_log_submetrics)
        self.offline_rl_log_oracle_stats = bool(offline_rl_log_oracle_stats)
        self.offline_rl_require_reference_policy_checkpoint = bool(offline_rl_require_reference_policy_checkpoint)
        self.offline_rl_report_raw_and_valid_best = bool(offline_rl_report_raw_and_valid_best)
        self.vlm_size = vlm_size
        self.train_backbone = train_backbone
        self.use_expert_features = use_expert_features
        if expert_cache_dir is None and chunk_cache_dir is not None:
            expert_cache_dir = chunk_cache_dir
        if allow_dummy_cache:
            allow_dummy_expert_cache = True
        self.expert_feature_source = normalize_expert_feature_source(
            expert_feature_source,
            use_expert_features=use_expert_features,
            expert_cache_dir=expert_cache_dir,
        )
        self.expert_cache_dir = expert_cache_dir
        self.chunk_cache_dir = chunk_cache_dir
        self.expert_adapter_dim = expert_adapter_dim
        self.num_jepa_tokens = num_jepa_tokens
        self.num_vggt_tokens = num_vggt_tokens
        self.allow_expert_target_features = allow_expert_target_features
        self.allow_random_init = allow_random_init
        self.allow_dummy_expert_cache = allow_dummy_expert_cache
        self.use_jepa = use_jepa
        self.use_vggt = use_vggt
        self.jepa_dim = jepa_dim
        self.vggt_dim = vggt_dim
        self.expert_dropout = expert_dropout
        self.expert_fusion_mode = expert_fusion_mode
        self.use_teacher_context_tokens = use_teacher_context_tokens
        self.use_student_latent_adapters = use_student_latent_adapters
        self.use_branch_weighted_mean = use_branch_weighted_mean
        self.use_expert_type_embedding = use_expert_type_embedding
        self.use_expert_gates = use_expert_gates
        self.expert_alignment_weight = expert_alignment_weight
        self.expert_stream_dropout = expert_stream_dropout
        self.expert_context_scale = expert_context_scale
        self.use_horizon_expert_residual = use_horizon_expert_residual
        self.expert_horizon_residual_scale = expert_horizon_residual_scale
        self.diffusion_loss_weight = diffusion_loss_weight
        if self.diffusion_loss_weight < 0.0:
            raise ValueError("diffusion_loss_weight must be non-negative.")
        self.use_alignment_loss = use_alignment_loss
        self.jepa_alignment_weight = jepa_alignment_weight
        self.vggt_alignment_weight = vggt_alignment_weight
        if not self.use_alignment_loss:
            self.expert_alignment_weight = 0.0
            self.jepa_alignment_weight = 0.0
            self.vggt_alignment_weight = 0.0
        self.alignment_loss_type = alignment_loss_type
        self.jepa_gate_init = jepa_gate_init
        self.vggt_gate_init = vggt_gate_init
        self.branch_init_vlm = branch_init_vlm
        self.branch_init_jepa = branch_init_jepa
        self.branch_init_vggt = branch_init_vggt
        self.allow_future_targets_in_inference = allow_future_targets_in_inference
        self.use_last_rd = use_last_rd
        self.last_rd_stage = last_rd_stage
        self.use_last_vla = use_last_vla
        self.last_vla_stage = last_vla_stage
        if self.use_last_vla and self.use_last_rd:
            raise ValueError("use_last_vla and use_last_rd are mutually exclusive.")
        self.use_two_expert_slots = bool(use_two_expert_slots)
        if self.use_two_expert_slots and (self.use_last_vla or self.use_last_rd or self.use_expert_features):
            raise ValueError("use_two_expert_slots is mutually exclusive with Last-VLA, Last-RD, and A4 direct experts.")
        if self.use_last_vla and self.last_vla_stage == "disabled":
            raise ValueError("use_last_vla=True requires last_vla_stage to be non-disabled.")
        if not self.use_last_vla and self.last_vla_stage != "disabled":
            raise ValueError("last_vla_stage must be 'disabled' when use_last_vla=False.")
        self.two_expert_cache_mode = bool(two_expert_cache_mode)
        self.two_expert_slot_mode = str(two_expert_slot_mode)
        self.two_expert_condition_mode = str(two_expert_condition_mode)
        self.two_expert_dit_condition_mode = str(two_expert_dit_condition_mode)
        self.two_expert_planner_dim = int(two_expert_planner_dim)
        self.two_expert_use_raw_vlm_base = bool(two_expert_use_raw_vlm_base)
        self.two_expert_zero_init_deltas = bool(two_expert_zero_init_deltas)
        self.two_expert_dyn_loss_floor = float(two_expert_dyn_loss_floor)
        self.two_expert_geo_loss_floor = float(two_expert_geo_loss_floor)
        self.two_expert_num_dyn_groups = int(two_expert_num_dyn_groups)
        self.two_expert_dyn_tokens_per_group = int(two_expert_dyn_tokens_per_group)
        self.two_expert_num_geo_tokens = int(two_expert_num_geo_tokens)
        self.two_expert_memory_tokens_to_dit = bool(two_expert_memory_tokens_to_dit)
        self.two_expert_denoise_gate_hidden_dim = int(two_expert_denoise_gate_hidden_dim)
        self.two_expert_denoise_gate_temperature = float(two_expert_denoise_gate_temperature)
        self.two_expert_denoise_condition_scale_init = float(two_expert_denoise_condition_scale_init)
        self.two_expert_memory_scale_init = float(two_expert_memory_scale_init)
        self.use_future_jepa_prediction = use_future_jepa_prediction
        self.use_vggt_geometry_tokens = use_vggt_geometry_tokens
        self.use_ego_trajectory_tokens = use_ego_trajectory_tokens
        self.use_risk_tokens = use_risk_tokens
        self.use_scene_aware_expert_gate = use_scene_aware_expert_gate
        self.use_timestep_aware_expert_gate = use_timestep_aware_expert_gate
        self.last_rd_latent_dim = last_rd_latent_dim
        self.num_dynamic_tokens = num_dynamic_tokens
        self.num_geometry_tokens = num_geometry_tokens
        self.num_ego_tokens = num_ego_tokens
        self.num_risk_tokens = num_risk_tokens
        self.require_vggt_geometry = require_vggt_geometry
        self.allow_patch_geometry_fallback = allow_patch_geometry_fallback
        self.future_jepa_loss_weight = future_jepa_loss_weight
        self.vggt_geometry_loss_weight = vggt_geometry_loss_weight
        self.coarse_traj_loss_weight = coarse_traj_loss_weight
        self.coarse_heading_loss_weight = coarse_heading_loss_weight
        self.risk_loss_weight = risk_loss_weight
        self.policy_kd_loss_weight = policy_kd_loss_weight
        self.future_jepa_loss_floor = future_jepa_loss_floor
        self.vggt_geometry_loss_floor = vggt_geometry_loss_floor
        self.coarse_traj_loss_floor = coarse_traj_loss_floor
        self.risk_loss_floor = risk_loss_floor
        self.last_rd_context_scale = last_rd_context_scale
        self.last_rd_horizon_condition_scale = last_rd_horizon_condition_scale
        self.last_rd_token_dropout = last_rd_token_dropout
        self.last_rd_group_dropout = last_rd_group_dropout
        self.reference_a0_checkpoint = reference_a0_checkpoint
        self.last_rd_adapter_checkpoint = last_rd_adapter_checkpoint
        self.policy_kd_mode = policy_kd_mode
        self.current_train_epoch = current_train_epoch
        self.total_train_epochs = total_train_epochs
        self.last_vla_cot_num_tokens = last_vla_cot_num_tokens
        self.last_vla_cot_num_steps = last_vla_cot_num_steps
        self.last_vla_condition_mode = str(last_vla_condition_mode)
        self.last_vla_raw_vlm_context_to_dit = last_vla_raw_vlm_context_to_dit
        self.last_vla_cot_bottleneck_mode = last_vla_cot_bottleneck_mode
        self.last_vla_vlm_context_dropout_start = last_vla_vlm_context_dropout_start
        self.last_vla_vlm_context_dropout_end = last_vla_vlm_context_dropout_end
        self.last_vla_use_scene_step = bool(last_vla_use_scene_step)
        self.last_vla_use_parallel_geometry_dynamic = bool(last_vla_use_parallel_geometry_dynamic)
        self.last_vla_fusion_tokens = int(last_vla_fusion_tokens)
        self.last_vla_dynamic_uses_geometry_memory = bool(last_vla_dynamic_uses_geometry_memory)
        self.last_vla_use_geometry_step = last_vla_use_geometry_step
        self.last_vla_use_dynamic_step = last_vla_use_dynamic_step
        self.last_vla_use_ego_step = last_vla_use_ego_step
        self.last_vla_use_action_refine_step = last_vla_use_action_refine_step
        self.last_vla_use_action_conditioned_dynamics = last_vla_use_action_conditioned_dynamics
        self.last_vla_use_risk_head = last_vla_use_risk_head
        self.last_vla_require_full_geometry = last_vla_require_full_geometry
        self.last_vla_allow_patch_geometry_fallback = last_vla_allow_patch_geometry_fallback
        self.last_vla_geometry_teacher_dim = last_vla_geometry_teacher_dim
        self.last_vla_geometry_grid_rows = last_vla_geometry_grid_rows
        self.last_vla_geometry_grid_cols = last_vla_geometry_grid_cols
        self.last_vla_use_residual_diffusion = last_vla_use_residual_diffusion
        self.last_vla_residual_detach_coarse = last_vla_residual_detach_coarse
        self.last_vla_residual_anchor_source = str(last_vla_residual_anchor_source)
        self.last_vla_require_residual_anchor = bool(last_vla_require_residual_anchor)
        self.last_vla_residual_anchor_cache_dir = last_vla_residual_anchor_cache_dir
        self.last_vla_coarse_prior_clip = last_vla_coarse_prior_clip
        self.last_vla_residual_alpha_start = last_vla_residual_alpha_start
        self.last_vla_residual_alpha_end = last_vla_residual_alpha_end
        self.last_vla_residual_alpha_warmup_epochs = last_vla_residual_alpha_warmup_epochs
        self.last_vla_cot_condition_zero_init = bool(last_vla_cot_condition_zero_init)
        self.last_vla_cot_condition_dropout = float(last_vla_cot_condition_dropout)
        self.last_vla_cot_condition_layers = str(last_vla_cot_condition_layers)
        self.last_vla_cot_condition_scale_init = float(last_vla_cot_condition_scale_init)
        self.last_vla_cot_condition_trainable_scale = bool(last_vla_cot_condition_trainable_scale)
        self.last_vla_context_mean_mode = str(last_vla_context_mean_mode)
        self.last_vla_horizon_condition_mode = str(last_vla_horizon_condition_mode)
        self.last_vla_aux_decay_epochs = last_vla_aux_decay_epochs
        self.last_vla_teacher_traj_mode = last_vla_teacher_traj_mode
        self.last_vla_teacher_traj_mix_start = last_vla_teacher_traj_mix_start
        self.last_vla_teacher_traj_mix_end = last_vla_teacher_traj_mix_end
        self.last_vla_teacher_score_margin = last_vla_teacher_score_margin
        self.last_vla_geometry_loss_weight = last_vla_geometry_loss_weight
        self.last_vla_dynamic_loss_weight = last_vla_dynamic_loss_weight
        self.last_vla_coarse_loss_weight = last_vla_coarse_loss_weight
        self.last_vla_heading_loss_weight = last_vla_heading_loss_weight
        self.last_vla_progress_loss_weight = last_vla_progress_loss_weight
        self.last_vla_risk_loss_weight = last_vla_risk_loss_weight
        self.last_vla_cot_consistency_loss_weight = last_vla_cot_consistency_loss_weight
        self.last_vla_geometry_loss_floor = last_vla_geometry_loss_floor
        self.last_vla_dynamic_loss_floor = last_vla_dynamic_loss_floor
        self.last_vla_coarse_loss_floor = last_vla_coarse_loss_floor
        self.last_vla_progress_loss_floor = last_vla_progress_loss_floor
        self.last_vla_adapter_checkpoint = last_vla_adapter_checkpoint
        self.last_vla_train_vlm_lora = last_vla_train_vlm_lora
        self.last_vla_vlm_lora_preset = str(last_vla_vlm_lora_preset)
        self.last_vla_vlm_lora_scope = str(last_vla_vlm_lora_scope)
        self.last_vla_vlm_lora_r = int(last_vla_vlm_lora_r)
        self.last_vla_vlm_lora_alpha = int(last_vla_vlm_lora_alpha)
        self.last_vla_vlm_lora_dropout = float(last_vla_vlm_lora_dropout)
        self.last_vla_vlm_lora_bias = str(last_vla_vlm_lora_bias)
        self.last_vla_vlm_lora_target_modules = last_vla_vlm_lora_target_modules
        self.last_vla_vlm_lora_use_rslora = bool(last_vla_vlm_lora_use_rslora)
        self.last_vla_vlm_lora_use_dora = bool(last_vla_vlm_lora_use_dora)
        self.last_vla_vlm_lora_init = str(last_vla_vlm_lora_init)
        self.last_vla_vlm_lora_vision_last_n = int(last_vla_vlm_lora_vision_last_n)
        self.last_vla_lora_allow_mixed_scope = bool(last_vla_lora_allow_mixed_scope)
        self.last_vla_lora_allow_all_linear_global = bool(last_vla_lora_allow_all_linear_global)
        self.lr_vlm_lora = lr_vlm_lora
        self.weight_decay_vlm_lora = float(weight_decay_vlm_lora)
        self.lr_last_vla_cot = lr_last_vla_cot
        self.weight_decay_last_vla_cot = float(weight_decay_last_vla_cot)
        self.last_vla_hidden_anchor_weight = float(last_vla_hidden_anchor_weight)
        self.last_vla_hidden_anchor_mode = str(last_vla_hidden_anchor_mode)
        self.last_vla_hidden_anchor_every_n_steps = int(last_vla_hidden_anchor_every_n_steps)
        self.last_vla_log_lora_diagnostics = bool(last_vla_log_lora_diagnostics)
        self._last_vla_lora_audit: Optional[Dict[str, Any]] = None
        self._last_vla_lora_config_payload: Optional[Dict[str, Any]] = None
        self._optimizer_group_report: List[Dict[str, Any]] = []
        self._hidden_anchor_forward_index = 0
        self.lr_action_head = lr_action_head
        self.lr_expert = lr_expert
        self.lr_expert_gate = lr_expert_gate
        self.train_expert_only = train_expert_only
        self.freeze_base_action_head = freeze_base_action_head
        self.freeze_expert = freeze_expert
        self.scheduler_epochs = scheduler_epochs
        self.scheduler_warmup_epochs = scheduler_warmup_epochs
        self.scheduler_min_lr = scheduler_min_lr
        self.grpo_scheduler_epochs = int(grpo_scheduler_epochs)
        self.grpo_scheduler_warmup_epochs = int(grpo_scheduler_warmup_epochs)
        self.grpo_scheduler_min_lr = float(grpo_scheduler_min_lr)
        if self.grpo_scheduler_epochs <= 0:
            raise ValueError("grpo_scheduler_epochs must be positive.")
        if self.grpo_scheduler_warmup_epochs < 0:
            raise ValueError("grpo_scheduler_warmup_epochs must be non-negative.")
        if self.grpo_scheduler_warmup_epochs >= self.grpo_scheduler_epochs:
            raise ValueError("grpo_scheduler_warmup_epochs must be smaller than grpo_scheduler_epochs.")
        if self.grpo_scheduler_min_lr < 0.0:
            raise ValueError("grpo_scheduler_min_lr must be non-negative.")
        self._warned_random_init = False
        self._warned_dummy_features = False
        self._dummy_expert_backend: Optional[DummyExpertBackend] = None

        if self.use_expert_features and self.expert_feature_source == "online":
            raise NotImplementedError("expert_feature_source='online' is reserved for future JEPA/VGGT teacher integration.")
        if self.use_expert_features and self.expert_feature_source == "dummy":
            self._dummy_expert_backend = build_dummy_expert_backend(
                num_jepa_tokens=self.num_jepa_tokens,
                num_vggt_tokens=self.num_vggt_tokens,
                jepa_dim=self.jepa_dim,
                vggt_dim=self.vggt_dim,
                use_jepa=self.use_jepa,
                use_vggt=self.use_vggt,
            )
        if self.last_vla_train_vlm_lora and self.cache_hidden_state:
            raise ValueError("VLM LoRA training requires no-cache/online VLM forward or regenerated hidden cache.")
        self._validate_last_vla_lora_config()

        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        self.device = device
        if not self.cache_hidden_state and not self.cache_mode:
            print("Agent running in 'no-cache' mode. Initializing internal backbone.")
            if not self.vlm_path or not self.vlm_type:
                raise ValueError("In 'no-cache' mode, vlm_path and vlm_type are required.")
            self.backbone = RecogDriveBackbone(
                model_type=self.vlm_type,
                checkpoint_path=self.vlm_path,
                device=str(device)
            )

            if not self.train_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad = False
            else:
                for p in self.backbone.parameters():
                    p.requires_grad = True
            if self.last_vla_train_vlm_lora:
                self._enable_last_vla_vlm_lora()

        if self.dit_type == "large":
            cfg = make_recogdrive_config(self.dit_type, action_dim=3, action_horizon=8, grpo=self.grpo, input_embedding_dim=1536,sampling_method=sampling_method)
        elif self.dit_type == "small":
            cfg = make_recogdrive_config(self.dit_type, action_dim=3, action_horizon=8, grpo=self.grpo, input_embedding_dim=384,sampling_method=sampling_method)

        cfg.vlm_size = self.vlm_size
        cfg.planner_dim = cfg.input_embedding_dim
        cfg.use_expert_features = self.use_expert_features
        cfg.expert_feature_source = self.expert_feature_source
        cfg.expert_adapter_dim = self.expert_adapter_dim
        cfg.allow_random_init = self.allow_random_init
        cfg.allow_dummy_expert_cache = self.allow_dummy_expert_cache
        cfg.use_jepa = self.use_jepa
        cfg.use_vggt = self.use_vggt
        cfg.jepa_dim = self.jepa_dim
        cfg.vggt_dim = self.vggt_dim
        cfg.num_jepa_tokens = self.num_jepa_tokens
        cfg.num_vggt_tokens = self.num_vggt_tokens
        cfg.use_teacher_context_tokens = self.use_teacher_context_tokens
        cfg.use_student_latent_adapters = self.use_student_latent_adapters
        cfg.use_branch_weighted_mean = self.use_branch_weighted_mean
        cfg.expert_dropout = self.expert_dropout
        cfg.expert_fusion_mode = self.expert_fusion_mode
        cfg.use_expert_type_embedding = self.use_expert_type_embedding
        cfg.use_expert_gates = self.use_expert_gates
        cfg.expert_alignment_weight = self.expert_alignment_weight
        cfg.expert_stream_dropout = self.expert_stream_dropout
        cfg.expert_context_scale = self.expert_context_scale
        cfg.use_horizon_expert_residual = self.use_horizon_expert_residual
        cfg.expert_horizon_residual_scale = self.expert_horizon_residual_scale
        cfg.diffusion_loss_weight = self.diffusion_loss_weight
        cfg.jepa_alignment_weight = self.jepa_alignment_weight
        cfg.vggt_alignment_weight = self.vggt_alignment_weight
        cfg.alignment_loss_type = self.alignment_loss_type
        cfg.jepa_gate_init = self.jepa_gate_init
        cfg.vggt_gate_init = self.vggt_gate_init
        cfg.branch_init_vlm = self.branch_init_vlm
        cfg.branch_init_jepa = self.branch_init_jepa
        cfg.branch_init_vggt = self.branch_init_vggt
        cfg.allow_future_targets_in_inference = self.allow_future_targets_in_inference
        cfg.use_last_rd = self.use_last_rd
        cfg.last_rd_stage = self.last_rd_stage
        cfg.use_future_jepa_prediction = self.use_future_jepa_prediction
        cfg.use_vggt_geometry_tokens = self.use_vggt_geometry_tokens
        cfg.use_ego_trajectory_tokens = self.use_ego_trajectory_tokens
        cfg.use_risk_tokens = self.use_risk_tokens
        cfg.use_scene_aware_expert_gate = self.use_scene_aware_expert_gate
        cfg.use_timestep_aware_expert_gate = self.use_timestep_aware_expert_gate
        cfg.last_rd_latent_dim = self.last_rd_latent_dim
        cfg.num_dynamic_tokens = self.num_dynamic_tokens
        cfg.num_geometry_tokens = self.num_geometry_tokens
        cfg.num_ego_tokens = self.num_ego_tokens
        cfg.num_risk_tokens = self.num_risk_tokens
        cfg.require_vggt_geometry = self.require_vggt_geometry
        cfg.allow_patch_geometry_fallback = self.allow_patch_geometry_fallback
        cfg.future_jepa_loss_weight = self.future_jepa_loss_weight
        cfg.vggt_geometry_loss_weight = self.vggt_geometry_loss_weight
        cfg.coarse_traj_loss_weight = self.coarse_traj_loss_weight
        cfg.coarse_heading_loss_weight = self.coarse_heading_loss_weight
        cfg.risk_loss_weight = self.risk_loss_weight
        cfg.policy_kd_loss_weight = self.policy_kd_loss_weight
        cfg.future_jepa_loss_floor = self.future_jepa_loss_floor
        cfg.vggt_geometry_loss_floor = self.vggt_geometry_loss_floor
        cfg.coarse_traj_loss_floor = self.coarse_traj_loss_floor
        cfg.risk_loss_floor = self.risk_loss_floor
        cfg.last_rd_context_scale = self.last_rd_context_scale
        cfg.last_rd_horizon_condition_scale = self.last_rd_horizon_condition_scale
        cfg.last_rd_token_dropout = self.last_rd_token_dropout
        cfg.last_rd_group_dropout = self.last_rd_group_dropout
        cfg.reference_a0_checkpoint = self.reference_a0_checkpoint
        cfg.policy_kd_mode = self.policy_kd_mode
        cfg.current_train_epoch = self.current_train_epoch
        cfg.total_train_epochs = self.total_train_epochs
        cfg.use_last_vla = self.use_last_vla
        cfg.last_vla_stage = self.last_vla_stage
        cfg.last_vla_cot_num_tokens = self.last_vla_cot_num_tokens
        cfg.last_vla_cot_num_steps = self.last_vla_cot_num_steps
        cfg.last_vla_condition_mode = self.last_vla_condition_mode
        cfg.last_vla_raw_vlm_context_to_dit = self.last_vla_raw_vlm_context_to_dit
        cfg.last_vla_cot_bottleneck_mode = self.last_vla_cot_bottleneck_mode
        cfg.last_vla_vlm_context_dropout_start = self.last_vla_vlm_context_dropout_start
        cfg.last_vla_vlm_context_dropout_end = self.last_vla_vlm_context_dropout_end
        cfg.last_vla_use_scene_step = self.last_vla_use_scene_step
        cfg.last_vla_use_parallel_geometry_dynamic = self.last_vla_use_parallel_geometry_dynamic
        cfg.last_vla_fusion_tokens = self.last_vla_fusion_tokens
        cfg.last_vla_dynamic_uses_geometry_memory = self.last_vla_dynamic_uses_geometry_memory
        cfg.last_vla_use_geometry_step = self.last_vla_use_geometry_step
        cfg.last_vla_use_dynamic_step = self.last_vla_use_dynamic_step
        cfg.last_vla_use_ego_step = self.last_vla_use_ego_step
        cfg.last_vla_use_action_refine_step = self.last_vla_use_action_refine_step
        cfg.last_vla_use_action_conditioned_dynamics = self.last_vla_use_action_conditioned_dynamics
        cfg.last_vla_use_risk_head = self.last_vla_use_risk_head
        cfg.last_vla_require_full_geometry = self.last_vla_require_full_geometry
        cfg.last_vla_allow_patch_geometry_fallback = self.last_vla_allow_patch_geometry_fallback
        cfg.last_vla_geometry_teacher_dim = self.last_vla_geometry_teacher_dim
        cfg.last_vla_geometry_grid_rows = self.last_vla_geometry_grid_rows
        cfg.last_vla_geometry_grid_cols = self.last_vla_geometry_grid_cols
        cfg.last_vla_use_residual_diffusion = self.last_vla_use_residual_diffusion
        cfg.last_vla_residual_detach_coarse = self.last_vla_residual_detach_coarse
        cfg.last_vla_residual_anchor_source = self.last_vla_residual_anchor_source
        cfg.last_vla_require_residual_anchor = self.last_vla_require_residual_anchor
        cfg.last_vla_coarse_prior_clip = self.last_vla_coarse_prior_clip
        cfg.last_vla_residual_alpha_start = self.last_vla_residual_alpha_start
        cfg.last_vla_residual_alpha_end = self.last_vla_residual_alpha_end
        cfg.last_vla_residual_alpha_warmup_epochs = self.last_vla_residual_alpha_warmup_epochs
        cfg.last_vla_cot_condition_zero_init = self.last_vla_cot_condition_zero_init
        cfg.last_vla_cot_condition_dropout = self.last_vla_cot_condition_dropout
        cfg.last_vla_cot_condition_layers = self.last_vla_cot_condition_layers
        cfg.last_vla_cot_condition_scale_init = self.last_vla_cot_condition_scale_init
        cfg.last_vla_cot_condition_trainable_scale = self.last_vla_cot_condition_trainable_scale
        cfg.last_vla_context_mean_mode = self.last_vla_context_mean_mode
        cfg.last_vla_horizon_condition_mode = self.last_vla_horizon_condition_mode
        cfg.last_vla_aux_decay_epochs = self.last_vla_aux_decay_epochs
        cfg.last_vla_teacher_traj_mode = self.last_vla_teacher_traj_mode
        cfg.last_vla_teacher_traj_mix_start = self.last_vla_teacher_traj_mix_start
        cfg.last_vla_teacher_traj_mix_end = self.last_vla_teacher_traj_mix_end
        cfg.last_vla_teacher_score_margin = self.last_vla_teacher_score_margin
        cfg.last_vla_geometry_loss_weight = self.last_vla_geometry_loss_weight
        cfg.last_vla_dynamic_loss_weight = self.last_vla_dynamic_loss_weight
        cfg.last_vla_coarse_loss_weight = self.last_vla_coarse_loss_weight
        cfg.last_vla_heading_loss_weight = self.last_vla_heading_loss_weight
        cfg.last_vla_progress_loss_weight = self.last_vla_progress_loss_weight
        cfg.last_vla_risk_loss_weight = self.last_vla_risk_loss_weight
        cfg.last_vla_cot_consistency_loss_weight = self.last_vla_cot_consistency_loss_weight
        cfg.last_vla_geometry_loss_floor = self.last_vla_geometry_loss_floor
        cfg.last_vla_dynamic_loss_floor = self.last_vla_dynamic_loss_floor
        cfg.last_vla_coarse_loss_floor = self.last_vla_coarse_loss_floor
        cfg.last_vla_progress_loss_floor = self.last_vla_progress_loss_floor
        cfg.last_vla_train_vlm_lora = self.last_vla_train_vlm_lora
        cfg.last_vla_vlm_lora_r = self.last_vla_vlm_lora_r
        cfg.last_vla_vlm_lora_alpha = self.last_vla_vlm_lora_alpha
        cfg.last_vla_vlm_lora_target_modules = self.last_vla_vlm_lora_target_modules
        cfg.last_vla_vlm_lora_preset = self.last_vla_vlm_lora_preset
        cfg.last_vla_vlm_lora_scope = self.last_vla_vlm_lora_scope
        cfg.last_vla_vlm_lora_dropout = self.last_vla_vlm_lora_dropout
        cfg.last_vla_vlm_lora_bias = self.last_vla_vlm_lora_bias
        cfg.last_vla_vlm_lora_use_rslora = self.last_vla_vlm_lora_use_rslora
        cfg.last_vla_vlm_lora_use_dora = self.last_vla_vlm_lora_use_dora
        cfg.last_vla_vlm_lora_init = self.last_vla_vlm_lora_init
        cfg.last_vla_vlm_lora_vision_last_n = self.last_vla_vlm_lora_vision_last_n
        cfg.use_two_expert_slots = self.use_two_expert_slots
        cfg.two_expert_cache_mode = self.two_expert_cache_mode
        cfg.two_expert_slot_mode = self.two_expert_slot_mode
        cfg.two_expert_condition_mode = self.two_expert_condition_mode
        cfg.two_expert_dit_condition_mode = self.two_expert_dit_condition_mode
        cfg.two_expert_planner_dim = self.two_expert_planner_dim
        cfg.two_expert_use_raw_vlm_base = self.two_expert_use_raw_vlm_base
        cfg.two_expert_zero_init_deltas = self.two_expert_zero_init_deltas
        cfg.two_expert_dyn_loss_floor = self.two_expert_dyn_loss_floor
        cfg.two_expert_geo_loss_floor = self.two_expert_geo_loss_floor
        cfg.two_expert_num_dyn_groups = self.two_expert_num_dyn_groups
        cfg.two_expert_dyn_tokens_per_group = self.two_expert_dyn_tokens_per_group
        cfg.two_expert_num_geo_tokens = self.two_expert_num_geo_tokens
        cfg.two_expert_memory_tokens_to_dit = self.two_expert_memory_tokens_to_dit
        cfg.two_expert_denoise_gate_hidden_dim = self.two_expert_denoise_gate_hidden_dim
        cfg.two_expert_denoise_gate_temperature = self.two_expert_denoise_gate_temperature
        cfg.two_expert_denoise_condition_scale_init = self.two_expert_denoise_condition_scale_init
        cfg.two_expert_memory_scale_init = self.two_expert_memory_scale_init

        offline_cfg = cfg.offline_rl_cfg
        offline_cfg.enabled = self.offline_rl_enabled
        offline_cfg.elite_buffer_path = self.offline_rl_elite_buffer_path
        offline_cfg.missing_buffer_policy = self.offline_rl_missing_buffer_policy
        offline_cfg.elite_top_m = self.offline_rl_elite_top_m
        offline_cfg.elite_min_candidates = self.offline_rl_elite_min_candidates
        offline_cfg.keep_gt_candidate = self.offline_rl_keep_gt_candidate
        offline_cfg.keep_il_candidate = self.offline_rl_keep_il_candidate
        offline_cfg.elite_buffer_version = self.offline_rl_elite_buffer_version
        offline_cfg.require_buffer_valid_mask = self.offline_rl_require_buffer_valid_mask
        offline_cfg.allow_v1_buffer_recompute_valid_mask = self.offline_rl_allow_v1_buffer_recompute_valid_mask
        offline_cfg.recompute_buffer_valid_mask_on_load = self.offline_rl_recompute_buffer_valid_mask_on_load
        offline_cfg.cache_elite_records_in_memory = self.offline_rl_cache_elite_records_in_memory
        offline_cfg.preload_elite_buffer = self.offline_rl_preload_elite_buffer
        offline_cfg.elite_buffer_preload_max_records = self.offline_rl_elite_buffer_preload_max_records
        offline_cfg.build_candidates_online = self.offline_rl_build_candidates_online
        offline_cfg.online_policy_samples = self.offline_rl_online_policy_samples
        offline_cfg.online_use_current_policy = self.offline_rl_online_use_current_policy
        offline_cfg.online_use_old_policy = self.offline_rl_online_use_old_policy
        offline_cfg.online_use_gt = self.offline_rl_online_use_gt
        offline_cfg.perturb_gt = self.offline_rl_perturb_gt
        offline_cfg.perturb_il = self.offline_rl_perturb_il
        offline_cfg.progress_endpoint_deltas_m = self.offline_rl_progress_endpoint_deltas_m
        offline_cfg.progress_speed_scales = self.offline_rl_progress_speed_scales
        offline_cfg.progress_time_gammas = self.offline_rl_progress_time_gammas
        offline_cfg.lateral_offsets_m = self.offline_rl_lateral_offsets_m
        offline_cfg.endpoint_lateral_offsets_m = self.offline_rl_endpoint_lateral_offsets_m
        offline_cfg.timing_slow_first_scales = self.offline_rl_timing_slow_first_scales
        offline_cfg.timing_delay_strengths = self.offline_rl_timing_delay_strengths
        offline_cfg.strict_reward_submetrics = self.offline_rl_strict_reward_submetrics
        offline_cfg.required_reward_submetrics = self.offline_rl_required_reward_submetrics
        offline_cfg.missing_submetric_policy = self.offline_rl_missing_submetric_policy
        offline_cfg.use_batched_pdm_scoring = self.offline_rl_use_batched_pdm_scoring
        offline_cfg.use_exact_array_pdm_state_conversion = self.offline_rl_use_exact_array_pdm_state_conversion
        offline_cfg.use_fast_pdm_scorer = self.offline_rl_use_fast_pdm_scorer
        offline_cfg.pdm_batch_chunk_size = self.offline_rl_pdm_batch_chunk_size
        offline_cfg.pdm_shadow_check = self.offline_rl_pdm_shadow_check
        offline_cfg.pdm_shadow_max_samples = self.offline_rl_pdm_shadow_max_samples
        offline_cfg.pdm_shadow_max_abs_diff = self.offline_rl_pdm_shadow_max_abs_diff
        offline_cfg.clip_candidates_to_norm_range = self.offline_rl_clip_candidates_to_norm_range
        offline_cfg.enforce_forward_monotonic_x = self.offline_rl_enforce_forward_monotonic_x
        offline_cfg.max_heading_step_rad = self.offline_rl_max_heading_step_rad
        offline_cfg.max_final_heading_delta_rad = self.offline_rl_max_final_heading_delta_rad
        offline_cfg.use_final_heading_guard = self.offline_rl_use_final_heading_guard
        offline_cfg.require_nc = self.offline_rl_require_nc
        offline_cfg.require_dac = self.offline_rl_require_dac
        offline_cfg.require_ddc_guard = self.offline_rl_require_ddc_guard
        offline_cfg.ddc_guard_mode = self.offline_rl_ddc_guard_mode
        offline_cfg.ddc_min_absolute = self.offline_rl_ddc_min_absolute
        offline_cfg.ddc_max_relative_drop = self.offline_rl_ddc_max_relative_drop
        offline_cfg.require_ttc_guard = self.offline_rl_require_ttc_guard
        offline_cfg.ttc_guard_mode = self.offline_rl_ttc_guard_mode
        offline_cfg.ttc_min_absolute = self.offline_rl_ttc_min_absolute
        offline_cfg.ttc_max_relative_drop = self.offline_rl_ttc_max_relative_drop
        offline_cfg.select_valid_topk_only = self.offline_rl_select_valid_topk_only
        offline_cfg.train_invalid_fallback_candidates = self.offline_rl_train_invalid_fallback_candidates
        offline_cfg.fallback_invalid_candidate_weight = self.offline_rl_fallback_invalid_candidate_weight
        offline_cfg.prior_distance_weight = self.offline_rl_prior_distance_weight
        offline_cfg.jerk_penalty_weight = self.offline_rl_jerk_penalty_weight
        offline_cfg.select_by = self.offline_rl_select_by
        offline_cfg.baseline_mode = self.offline_rl_baseline_mode
        offline_cfg.expectile_tau = self.offline_rl_expectile_tau
        offline_cfg.expectile_iters = self.offline_rl_expectile_iters
        offline_cfg.top_mean_frac = self.offline_rl_top_mean_frac
        offline_cfg.advantage_temperature = self.offline_rl_advantage_temperature
        offline_cfg.advantage_clip_min = self.offline_rl_advantage_clip_min
        offline_cfg.advantage_clip_max = self.offline_rl_advantage_clip_max
        offline_cfg.weight_min = self.offline_rl_weight_min
        offline_cfg.weight_max = self.offline_rl_weight_max
        offline_cfg.normalize_weights_per_scene = self.offline_rl_normalize_weights_per_scene
        offline_cfg.train_only_valid_candidates = self.offline_rl_train_only_valid_candidates
        offline_cfg.min_reward_margin_to_gt_for_extra_weight = self.offline_rl_min_reward_margin_to_gt_for_extra_weight
        offline_cfg.allow_zero_weight_rows = self.offline_rl_allow_zero_weight_rows
        offline_cfg.target_filter_mode = self.offline_rl_target_filter_mode
        offline_cfg.target_top_k = self.offline_rl_target_top_k
        offline_cfg.target_min_advantage = self.offline_rl_target_min_advantage
        offline_cfg.awac_timestep_sampling = self.offline_rl_awac_timestep_sampling
        offline_cfg.preference_dpo_timestep_sampling = self.offline_rl_preference_dpo_timestep_sampling
        offline_cfg.low_noise_timestep_frac = self.offline_rl_low_noise_timestep_frac
        offline_cfg.mid_noise_timestep_low_frac = self.offline_rl_mid_noise_timestep_low_frac
        offline_cfg.mid_noise_timestep_high_frac = self.offline_rl_mid_noise_timestep_high_frac
        offline_cfg.target_blend_mode = self.offline_rl_target_blend_mode
        offline_cfg.target_blend_alpha = self.offline_rl_target_blend_alpha
        offline_cfg.target_blend_schedule = self.offline_rl_target_blend_schedule
        offline_cfg.target_blend_alpha_start = self.offline_rl_target_blend_alpha_start
        offline_cfg.target_blend_warmup_start_epoch = self.offline_rl_target_blend_warmup_start_epoch
        offline_cfg.target_blend_warmup_epochs = self.offline_rl_target_blend_warmup_epochs
        offline_cfg.component_advantage_enabled = self.offline_rl_component_advantage_enabled
        offline_cfg.component_progress_weight = self.offline_rl_component_progress_weight
        offline_cfg.component_safety_penalty_weight = self.offline_rl_component_safety_penalty_weight
        offline_cfg.component_advantage_clip_min = self.offline_rl_component_advantage_clip_min
        offline_cfg.component_advantage_clip_max = self.offline_rl_component_advantage_clip_max
        offline_cfg.source_balance_enabled = self.offline_rl_source_balance_enabled
        offline_cfg.source_balance_min_factor = self.offline_rl_source_balance_min_factor
        offline_cfg.source_balance_max_factor = self.offline_rl_source_balance_max_factor
        offline_cfg.pairwise_rank_loss_weight = self.offline_rl_pairwise_rank_loss_weight
        offline_cfg.pairwise_rank_margin = self.offline_rl_pairwise_rank_margin
        offline_cfg.pairwise_rank_min_reward_gap = self.offline_rl_pairwise_rank_min_reward_gap
        offline_cfg.pairwise_rank_max_pairs_per_scene = self.offline_rl_pairwise_rank_max_pairs_per_scene
        offline_cfg.invalid_repulsion_loss_weight = self.offline_rl_invalid_repulsion_loss_weight
        offline_cfg.invalid_repulsion_margin = self.offline_rl_invalid_repulsion_margin
        offline_cfg.invalid_repulsion_max_pairs_per_scene = self.offline_rl_invalid_repulsion_max_pairs_per_scene
        offline_cfg.preference_dpo_loss_weight = self.offline_rl_preference_dpo_loss_weight
        offline_cfg.preference_dpo_beta = self.offline_rl_preference_dpo_beta
        offline_cfg.preference_dpo_label_smoothing = self.offline_rl_preference_dpo_label_smoothing
        offline_cfg.preference_dpo_reference_free = self.offline_rl_preference_dpo_reference_free
        offline_cfg.preference_dpo_pair_mode = self.offline_rl_preference_dpo_pair_mode
        offline_cfg.preference_dpo_min_reward_gap = self.offline_rl_preference_dpo_min_reward_gap
        offline_cfg.preference_dpo_max_pairs_per_scene = self.offline_rl_preference_dpo_max_pairs_per_scene
        offline_cfg.preference_dpo_gap_weight_mode = self.offline_rl_preference_dpo_gap_weight_mode
        offline_cfg.preference_dpo_gap_weight_scale = self.offline_rl_preference_dpo_gap_weight_scale
        offline_cfg.preference_dpo_gap_weight_min = self.offline_rl_preference_dpo_gap_weight_min
        offline_cfg.preference_dpo_gap_weight_max = self.offline_rl_preference_dpo_gap_weight_max
        offline_cfg.awac_loss_weight = self.offline_rl_awac_loss_weight
        offline_cfg.awac_loss_schedule = self.offline_rl_awac_loss_schedule
        offline_cfg.awac_loss_weight_start = self.offline_rl_awac_loss_weight_start
        offline_cfg.awac_loss_warmup_start_epoch = self.offline_rl_awac_loss_warmup_start_epoch
        offline_cfg.awac_loss_warmup_epochs = self.offline_rl_awac_loss_warmup_epochs
        offline_cfg.preference_dpo_loss_schedule = self.offline_rl_preference_dpo_loss_schedule
        offline_cfg.preference_dpo_loss_weight_start = self.offline_rl_preference_dpo_loss_weight_start
        offline_cfg.preference_dpo_loss_warmup_start_epoch = self.offline_rl_preference_dpo_loss_warmup_start_epoch
        offline_cfg.preference_dpo_loss_warmup_epochs = self.offline_rl_preference_dpo_loss_warmup_epochs
        offline_cfg.bc_loss_weight = self.offline_rl_bc_loss_weight
        offline_cfg.bc_loss_schedule = self.offline_rl_bc_loss_schedule
        offline_cfg.bc_loss_weight_start = self.offline_rl_bc_loss_weight_start
        offline_cfg.bc_loss_weight_end = self.offline_rl_bc_loss_weight_end
        offline_cfg.bc_loss_schedule_epochs = self.offline_rl_bc_loss_schedule_epochs
        offline_cfg.grpo_loss_weight = self.offline_rl_grpo_loss_weight
        offline_cfg.grpo_loss_schedule = self.offline_rl_grpo_loss_schedule
        offline_cfg.grpo_loss_weight_start = self.offline_rl_grpo_loss_weight_start
        offline_cfg.grpo_loss_warmup_start_epoch = self.offline_rl_grpo_loss_warmup_start_epoch
        offline_cfg.grpo_loss_warmup_epochs = self.offline_rl_grpo_loss_warmup_epochs
        offline_cfg.grpo_buffer_guidance_enabled = self.offline_rl_grpo_buffer_guidance_enabled
        offline_cfg.grpo_buffer_reward_bonus_weight = self.offline_rl_grpo_buffer_reward_bonus_weight
        offline_cfg.grpo_buffer_reward_bonus_scale_m = self.offline_rl_grpo_buffer_reward_bonus_scale_m
        offline_cfg.grpo_buffer_reward_bonus_use_margin = self.offline_rl_grpo_buffer_reward_bonus_use_margin
        offline_cfg.grpo_buffer_distill_loss_weight = self.offline_rl_grpo_buffer_distill_loss_weight
        offline_cfg.grpo_buffer_distill_loss_schedule = self.offline_rl_grpo_buffer_distill_loss_schedule
        offline_cfg.grpo_buffer_distill_loss_weight_start = self.offline_rl_grpo_buffer_distill_loss_weight_start
        offline_cfg.grpo_buffer_distill_warmup_start_epoch = self.offline_rl_grpo_buffer_distill_warmup_start_epoch
        offline_cfg.grpo_buffer_distill_warmup_epochs = self.offline_rl_grpo_buffer_distill_warmup_epochs
        offline_cfg.grpo_buffer_distill_top_k = self.offline_rl_grpo_buffer_distill_top_k
        offline_cfg.grpo_buffer_distill_min_reward_margin = self.offline_rl_grpo_buffer_distill_min_reward_margin
        offline_cfg.grpo_buffer_distill_timestep_sampling = self.offline_rl_grpo_buffer_distill_timestep_sampling
        offline_cfg.grpo_self_imitation_loss_weight = self.offline_rl_grpo_self_imitation_loss_weight
        offline_cfg.grpo_self_imitation_loss_schedule = self.offline_rl_grpo_self_imitation_loss_schedule
        offline_cfg.grpo_self_imitation_loss_weight_start = self.offline_rl_grpo_self_imitation_loss_weight_start
        offline_cfg.grpo_self_imitation_warmup_start_epoch = (
            self.offline_rl_grpo_self_imitation_warmup_start_epoch
        )
        offline_cfg.grpo_self_imitation_warmup_epochs = self.offline_rl_grpo_self_imitation_warmup_epochs
        offline_cfg.grpo_self_imitation_top_k = self.offline_rl_grpo_self_imitation_top_k
        offline_cfg.grpo_self_imitation_min_reward = self.offline_rl_grpo_self_imitation_min_reward
        offline_cfg.grpo_self_imitation_min_reward_margin = (
            self.offline_rl_grpo_self_imitation_min_reward_margin
        )
        offline_cfg.grpo_self_imitation_max_target_scene_ratio = (
            self.offline_rl_grpo_self_imitation_max_target_scene_ratio
        )
        offline_cfg.grpo_self_imitation_batch_cap_score = self.offline_rl_grpo_self_imitation_batch_cap_score
        offline_cfg.grpo_self_imitation_baseline_mode = self.offline_rl_grpo_self_imitation_baseline_mode
        offline_cfg.grpo_self_imitation_timestep_sampling = self.offline_rl_grpo_self_imitation_timestep_sampling
        offline_cfg.grpo_self_imitation_require_nc = self.offline_rl_grpo_self_imitation_require_nc
        offline_cfg.grpo_self_imitation_require_dac = self.offline_rl_grpo_self_imitation_require_dac
        offline_cfg.grpo_self_imitation_require_ttc = self.offline_rl_grpo_self_imitation_require_ttc
        offline_cfg.grpo_self_imitation_require_ddc = self.offline_rl_grpo_self_imitation_require_ddc
        offline_cfg.grpo_self_imitation_nc_min_absolute = (
            self.offline_rl_grpo_self_imitation_nc_min_absolute
        )
        offline_cfg.grpo_self_imitation_dac_min_absolute = (
            self.offline_rl_grpo_self_imitation_dac_min_absolute
        )
        offline_cfg.grpo_self_imitation_ttc_min_absolute = (
            self.offline_rl_grpo_self_imitation_ttc_min_absolute
        )
        offline_cfg.grpo_self_imitation_ddc_min_absolute = (
            self.offline_rl_grpo_self_imitation_ddc_min_absolute
        )
        offline_cfg.log_candidate_sources = self.offline_rl_log_candidate_sources
        offline_cfg.log_submetrics = self.offline_rl_log_submetrics
        offline_cfg.log_oracle_stats = self.offline_rl_log_oracle_stats
        offline_cfg.require_reference_policy_checkpoint = self.offline_rl_require_reference_policy_checkpoint
        offline_cfg.report_raw_and_valid_best = self.offline_rl_report_raw_and_valid_best

        if self.grpo or self.offline_rl_enabled:
            cfg.grpo_cfg.metric_cache_path = self.metric_cache_path
            cfg.grpo_cfg.reference_policy_checkpoint = self.reference_policy_checkpoint
            cfg.grpo_cfg.sample_time = self.grpo_sample_time
            cfg.grpo_cfg.bc_anneal = self.bc_anneal
            cfg.grpo_cfg.bc_coeff_start = self.bc_coeff_start
            cfg.grpo_cfg.bc_coeff_end = self.bc_coeff_end
            cfg.grpo_cfg.bc_anneal_epochs = self.bc_anneal_epochs
            cfg.grpo_cfg.reference_kl_coeff = self.reference_kl_coeff
            cfg.grpo_cfg.reference_kl_chunk_size = self.reference_kl_chunk_size
            cfg.grpo_cfg.use_gspo_ratio = self.grpo_use_gspo_ratio
            cfg.grpo_cfg.gspo_clip_low = self.grpo_gspo_clip_low
            cfg.grpo_cfg.gspo_clip_high = self.grpo_gspo_clip_high
            cfg.grpo_cfg.behavior_policy_sync_interval = self.grpo_behavior_policy_sync_interval
            cfg.grpo_cfg.behavior_policy_sample = self.grpo_behavior_policy_sample
            cfg.grpo_cfg.normalize_advantage_batch = self.grpo_normalize_advantage_batch
            cfg.grpo_cfg.advantage_clip_abs = self.grpo_advantage_clip_abs
            cfg.grpo_cfg.ppo_replay_inner_epochs = self.grpo_ppo_replay_inner_epochs
            cfg.grpo_cfg.ppo_replay_minibatch_size = self.grpo_ppo_replay_minibatch_size
            cfg.grpo_cfg.ppo_replay_max_grad_norm = self.grpo_ppo_replay_max_grad_norm
            cfg.grpo_cfg.ppo_replay_min_abs_advantage = self.grpo_ppo_replay_min_abs_advantage
            cfg.grpo_cfg.ppo_replay_filter_zero_advantage = self.grpo_ppo_replay_filter_zero_advantage
            cfg.grpo_cfg.ppo_replay_sync_behavior_each_batch = self.grpo_ppo_replay_sync_behavior_each_batch
            cfg.grpo_cfg.ppo_replay_bc_update = self.grpo_ppo_replay_bc_update
            cfg.grpo_cfg.ppo_replay_logprob_mode = self.grpo_ppo_replay_logprob_mode
            cfg.grpo_cfg.ppo_replay_step_minibatch_mode = self.grpo_ppo_replay_step_minibatch_mode
            cfg.grpo_cfg.ppo_replay_logprob_clamp_min = self.grpo_ppo_replay_logprob_clamp_min
            cfg.grpo_cfg.ppo_replay_logprob_clamp_max = self.grpo_ppo_replay_logprob_clamp_max
            cfg.grpo_cfg.ppo_replay_step_clip_schedule = self.grpo_ppo_replay_step_clip_schedule
            cfg.grpo_cfg.ppo_replay_step_clip_base = self.grpo_ppo_replay_step_clip_base
            cfg.grpo_cfg.ppo_replay_step_clip_rate = self.grpo_ppo_replay_step_clip_rate
            
        self.action_head = ReCogDriveDiffusionPlanner(cfg).to(device)
        if self.last_rd_adapter_checkpoint:
            self._safe_load_last_rd_adapter(self.last_rd_adapter_checkpoint)
        if self.last_vla_adapter_checkpoint:
            self._safe_load_last_vla_adapter(self.last_vla_adapter_checkpoint)
        self._set_trainable_parameters()
        self.num_inference_samples = 1
        self.inference_selection_mode = "median"

    def name(self) -> str:
        return self.__class__.__name__

    def set_training_progress(self, epoch: int, total_epochs: int) -> None:
        if hasattr(self.action_head, "set_training_progress"):
            self.action_head.set_training_progress(epoch, total_epochs)

    def _validate_last_vla_lora_config(self) -> None:
        if self.last_vla_vlm_lora_preset not in {"attention_only", "attention_mlp", "all_linear", "vision_last_n", "custom"}:
            raise ValueError(f"Unknown last_vla_vlm_lora_preset={self.last_vla_vlm_lora_preset!r}.")
        if self.last_vla_vlm_lora_scope not in {"llm", "vision", "llm_vision", "projector", "all"}:
            raise ValueError(f"Unknown last_vla_vlm_lora_scope={self.last_vla_vlm_lora_scope!r}.")
        if self.last_vla_vlm_lora_r <= 0:
            raise ValueError("last_vla_vlm_lora_r must be positive.")
        if self.last_vla_vlm_lora_alpha <= 0:
            raise ValueError("last_vla_vlm_lora_alpha must be positive.")
        if not (0.0 <= self.last_vla_vlm_lora_dropout < 1.0):
            raise ValueError("last_vla_vlm_lora_dropout must be in [0, 1).")
        if self.last_vla_hidden_anchor_mode not in {"none", "summary_cosine", "token_mean_cosine", "mse_mean"}:
            raise ValueError(f"Unknown last_vla_hidden_anchor_mode={self.last_vla_hidden_anchor_mode!r}.")
        if self.last_vla_residual_anchor_source not in {"vlm_text_traj", "none"}:
            raise ValueError(f"Unknown last_vla_residual_anchor_source={self.last_vla_residual_anchor_source!r}.")
        if self.lr_vlm_lora is not None and float(self.lr_vlm_lora) <= 0.0:
            raise ValueError("lr_vlm_lora must be positive when set.")
        if self.lr_last_vla_cot is not None and float(self.lr_last_vla_cot) <= 0.0:
            raise ValueError("lr_last_vla_cot must be positive when set.")
        if self.weight_decay_vlm_lora < 0.0 or self.weight_decay_last_vla_cot < 0.0:
            raise ValueError("LoRA/Last-VLA CoT weight decay must be non-negative.")
        if self.last_vla_hidden_anchor_every_n_steps <= 0:
            raise ValueError("last_vla_hidden_anchor_every_n_steps must be positive.")

    def _enable_last_vla_vlm_lora(self) -> None:
        if self.backbone is None:
            raise ValueError("VLM LoRA training requires an initialized online backbone.")
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "last_vla_train_vlm_lora=True requires peft. Install it with `pip install peft` "
                "or disable Last-VLA VLM LoRA."
            ) from exc
        base_vlm = getattr(self.backbone, "model", None)
        if base_vlm is None:
            raise ValueError("VLM LoRA training requires RecogDriveBackbone.model to be initialized.")
        target_modules = resolve_lora_target_modules(
            base_vlm,
            preset=self.last_vla_vlm_lora_preset,
            scope=self.last_vla_vlm_lora_scope,
            custom_target_modules=self.last_vla_vlm_lora_target_modules,
            vision_last_n=self.last_vla_vlm_lora_vision_last_n,
            allow_all_linear_global=self.last_vla_lora_allow_all_linear_global,
        )
        pre_audit = audit_lora_target_modules(
            base_vlm,
            target_modules=target_modules,
            scope=self.last_vla_vlm_lora_scope,
            preset=self.last_vla_vlm_lora_preset,
        )
        validate_lora_scope_audit(pre_audit, allow_mixed_scope=self.last_vla_lora_allow_mixed_scope)
        supported_kwargs = peft_lora_config_kwargs_supported()
        if self.last_vla_vlm_lora_use_rslora and "use_rslora" not in supported_kwargs:
            raise RuntimeError("last_vla_vlm_lora_use_rslora=True requires a PEFT version with LoraConfig(use_rslora=...).")
        if self.last_vla_vlm_lora_use_dora and "use_dora" not in supported_kwargs:
            raise RuntimeError("last_vla_vlm_lora_use_dora=True requires a PEFT version with LoraConfig(use_dora=...).")
        lora_kwargs: Dict[str, Any] = {
            "r": int(self.last_vla_vlm_lora_r),
            "lora_alpha": int(self.last_vla_vlm_lora_alpha),
            "lora_dropout": float(self.last_vla_vlm_lora_dropout),
            "target_modules": target_modules,
            "bias": self.last_vla_vlm_lora_bias,
        }
        if "use_rslora" in supported_kwargs:
            lora_kwargs["use_rslora"] = bool(self.last_vla_vlm_lora_use_rslora)
        if "use_dora" in supported_kwargs:
            lora_kwargs["use_dora"] = bool(self.last_vla_vlm_lora_use_dora)
        if self.last_vla_vlm_lora_init != "default" and "init_lora_weights" in supported_kwargs:
            lora_kwargs["init_lora_weights"] = self.last_vla_vlm_lora_init
        lora_cfg = LoraConfig(**lora_kwargs)
        peft_vlm = get_peft_model(base_vlm, lora_cfg)
        # Keep the RecogDriveBackbone wrapper intact so image/token preprocessing and
        # hidden-state extraction continue to use the existing forward path.
        for attr in ("img_context_token_id", "system_message"):
            if hasattr(base_vlm, attr) and not hasattr(peft_vlm, attr):
                setattr(peft_vlm, attr, getattr(base_vlm, attr))
        if hasattr(self.backbone, "_patch_internvl_visual_feature_dtype"):
            self.backbone._patch_internvl_visual_feature_dtype(base_vlm)
        self.backbone.model = peft_vlm
        post_audit = audit_lora_target_modules(
            self.backbone.model,
            target_modules=target_modules,
            scope=self.last_vla_vlm_lora_scope,
            preset=self.last_vla_vlm_lora_preset,
        )
        if post_audit["matched_total"] <= 0:
            post_audit["matched_total"] = pre_audit["matched_total"]
            post_audit["matched_module_names"] = pre_audit["matched_module_names"]
            post_audit["matched_by_category"] = pre_audit["matched_by_category"]
        actual_audit = audit_actual_trainable_lora_modules(
            self.backbone.model,
            scope=self.last_vla_vlm_lora_scope,
        )
        validate_lora_scope_audit(actual_audit, allow_mixed_scope=self.last_vla_lora_allow_mixed_scope)
        if int(actual_audit.get("actual_trainable_lora_param_count", 0)) <= 0:
            raise RuntimeError("PEFT injection produced zero trainable LoRA parameters.")
        self._last_vla_lora_audit = {
            "preset": self.last_vla_vlm_lora_preset,
            "scope": self.last_vla_vlm_lora_scope,
            "resolved_target_modules": target_modules,
            "matched_total": post_audit.get("matched_total", 0),
            "matched_by_category": post_audit.get("matched_by_category", {}),
            "trainable_lora_param_count": actual_audit.get("actual_trainable_lora_param_count", 0),
            "base_model_param_count": post_audit.get("base_model_param_count", 0),
            "trainable_ratio": (
                float(actual_audit.get("actual_trainable_lora_param_count", 0))
                / float(post_audit.get("base_model_param_count", 1) or 1)
            ),
            "intended_audit": pre_audit,
            "post_injection_intended_audit": post_audit,
            "actual_trainable_audit": actual_audit,
            "high_risk_warnings": list(post_audit.get("high_risk_warnings", [])),
        }
        self._last_vla_lora_config_payload = build_lora_config_payload(
            preset=self.last_vla_vlm_lora_preset,
            scope=self.last_vla_vlm_lora_scope,
            target_modules=target_modules,
            r=self.last_vla_vlm_lora_r,
            alpha=self.last_vla_vlm_lora_alpha,
            dropout=self.last_vla_vlm_lora_dropout,
            bias=self.last_vla_vlm_lora_bias,
            use_rslora=self.last_vla_vlm_lora_use_rslora,
            use_dora=self.last_vla_vlm_lora_use_dora,
            init=self.last_vla_vlm_lora_init,
            vision_last_n=self.last_vla_vlm_lora_vision_last_n,
            allow_all_linear_global=self.last_vla_lora_allow_all_linear_global,
            hidden_anchor_every_n_steps=self.last_vla_hidden_anchor_every_n_steps,
        )
        if int(os.getenv("LOCAL_RANK", os.getenv("RANK", "0"))) == 0:
            print(
                "Last-VLA VLM-LoRA enabled: "
                f"preset={self.last_vla_vlm_lora_preset}, scope={self.last_vla_vlm_lora_scope}, "
                f"r={self.last_vla_vlm_lora_r}, alpha={self.last_vla_vlm_lora_alpha}, "
                f"dropout={self.last_vla_vlm_lora_dropout}, use_rslora={self.last_vla_vlm_lora_use_rslora}, "
                f"use_dora={self.last_vla_vlm_lora_use_dora}, matched={post_audit['matched_total']}, "
                f"actual_lora_modules={actual_audit['actual_trainable_lora_module_count']}, "
                f"trainable_lora_params={actual_audit['actual_trainable_lora_param_count']}"
            )

    def count_trainable_parameters_by_group(self) -> Dict[str, Dict[str, int]]:
        groups = {
            "last_vla_cot": {"trainable": 0, "total": 0},
            "cot_condition_branch": {"trainable": 0, "total": 0},
            "last_rd": {"trainable": 0, "total": 0},
            "legacy_a4_expert": {"trainable": 0, "total": 0},
            "action_base": {"trainable": 0, "total": 0},
            "backbone": {"trainable": 0, "total": 0},
            "backbone_non_lora": {"trainable": 0, "total": 0},
            "vlm_lora": {"trainable": 0, "total": 0},
            "other": {"trainable": 0, "total": 0},
        }
        lora_by_category: Dict[str, Dict[str, int]] = {}
        legacy_markers = (
            "jepa_projector",
            "vggt_projector",
            "jepa_adapter",
            "vggt_adapter",
            "jepa_alignment_head",
            "vggt_alignment_head",
            "jepa_type_embedding",
            "vggt_type_embedding",
            "z_jepa_type_embedding",
            "z_vggt_type_embedding",
            "jepa_gate",
            "vggt_gate",
            "jepa_horizon_conditioner",
            "vggt_horizon_conditioner",
            "branch_logits",
        )
        for name, parameter in self.named_parameters():
            count = int(parameter.numel())
            if self._is_last_vla_condition_parameter_key(name):
                group = "cot_condition_branch"
            elif "action_head.last_vla_cot." in name:
                group = "last_vla_cot"
            elif "action_head.last_rd." in name:
                group = "last_rd"
            elif name.startswith("action_head.") and any(marker in name for marker in legacy_markers):
                group = "legacy_a4_expert"
            elif name.startswith("action_head."):
                group = "action_base"
            elif "lora_" in name:
                group = "vlm_lora"
            elif name.startswith("backbone."):
                group = "backbone_non_lora"
            else:
                group = "other"
            groups[group]["total"] += count
            if parameter.requires_grad:
                groups[group]["trainable"] += count
            if group in {"backbone_non_lora", "vlm_lora"}:
                groups["backbone"]["total"] += count
                if parameter.requires_grad:
                    groups["backbone"]["trainable"] += count
            if group == "vlm_lora":
                category = infer_lora_module_category(name.split(".lora_", 1)[0])
                lora_by_category.setdefault(category, {"trainable": 0, "total": 0})
                lora_by_category[category]["total"] += count
                if parameter.requires_grad:
                    lora_by_category[category]["trainable"] += count
        groups["all"] = {
            "total": sum(item["total"] for key, item in groups.items() if key != "backbone"),
            "trainable": sum(item["trainable"] for key, item in groups.items() if key != "backbone"),
        }
        groups["vlm_lora_by_category"] = lora_by_category
        return groups

    def get_lora_target_report(self) -> Optional[Dict[str, Any]]:
        return self._last_vla_lora_audit

    def get_lora_training_config(self) -> Optional[Dict[str, Any]]:
        return self._last_vla_lora_config_payload

    def get_optimizer_group_report(self) -> List[Dict[str, Any]]:
        return list(self._optimizer_group_report)

    def initialize(self) -> None:
        if self.checkpoint_path:
            self._safe_load_checkpoint(self.checkpoint_path)
            return

        if self.allow_random_init and not self._warned_random_init:
            warnings.warn(
                "ReCogDriveAgent is using random initialization because checkpoint_path is empty. "
                "Random initialization is for computation-flow validation only and must not be "
                "interpreted as driving performance.",
                RuntimeWarning,
            )
            self._warned_random_init = True

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_all_sensors(include=[0, 1, 2, 3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [TrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [ReCogDriveFeatureBuilder(
            cache_hidden_state=self.cache_hidden_state,
            model_type=self.vlm_type,
            checkpoint_path=self.vlm_path,
            device=str(self.device),
            cache_mode=self.cache_mode,
            use_expert_features=self.use_expert_features,
            expert_feature_source=self.expert_feature_source,
            expert_cache_dir=self.expert_cache_dir,
            num_jepa_tokens=self.num_jepa_tokens,
            num_vggt_tokens=self.num_vggt_tokens,
            num_geometry_tokens=self.num_geometry_tokens,
            allow_expert_target_features=self.allow_expert_target_features and self.training,
            use_jepa=self.use_jepa,
            use_vggt=self.use_vggt,
            jepa_dim=self.jepa_dim,
            vggt_dim=self.vggt_dim,
            geometry_teacher_dim=self.last_vla_geometry_teacher_dim if self.use_last_vla else self.vggt_dim,
        )]

    def forward(self, features: Dict[str, torch.Tensor], targets=None, tokens_list=None) -> Dict[str, torch.Tensor]:
        action_device = next(self.action_head.parameters()).device
        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                features[key] = tensor.to(action_device)

        model_dtype = next(self.action_head.parameters()).dtype
        for key in EXPERT_FEATURE_KEYS:
            if key in features and isinstance(features[key], torch.Tensor):
                features[key] = features[key].to(model_dtype)
        self._add_dummy_expert_features_if_needed(features, action_device, model_dtype)

        history_trajectory = features["history_trajectory"].to(action_device)
        high_command_one_hot = features["high_command_one_hot"].to(action_device)
        
        if history_trajectory.ndim == 2:
            history_trajectory = history_trajectory.unsqueeze(0)
        if high_command_one_hot.ndim == 1:
            high_command_one_hot = high_command_one_hot.unsqueeze(0)

        hidden_anchor_loss: Optional[torch.Tensor] = None
        hidden_drift_cosine: Optional[torch.Tensor] = None
        hidden_drift_l2: Optional[torch.Tensor] = None
        hidden_anchor_computed = False
        hidden_anchor_step_index: Optional[int] = None
        if self.cache_hidden_state:
            last_hidden_state = features["last_hidden_state"].to(action_device)
        else:
            if self.backbone is None:
                raise RuntimeError("Agent is in 'no-cache' mode, but backbone is not initialized.")
            image_path_tensor = features["image_path_tensor"]
            if image_path_tensor.ndim == 1:
                image_path_tensor = image_path_tensor.unsqueeze(0)
            image_paths = self._decode_paths_from_tensor(image_path_tensor)
            
            pixel_values_list = [load_image(path) for path in image_paths]
            
            num_patches_list = [p.shape[0] for p in pixel_values_list]
            pixel_values_cat = torch.cat(pixel_values_list, dim=0).to(action_device)
            

            navigation_commands = ['turn left', 'go straight', 'turn right']
            command_indices = torch.argmax(high_command_one_hot, dim=-1)
            command_str_list = [navigation_commands[idx.item()] for idx in command_indices]

            questions = []
            batch_size = high_command_one_hot.shape[0]
            for i in range(batch_size):
                history_trajectory_sample = history_trajectory[i]
                command_str_sample = command_str_list[i]

                history_str = ' '.join([
                    f'   - t-{3-j}: ({format_number(history_trajectory_sample[j, 0].item())}, '
                    f'{format_number(history_trajectory_sample[j, 1].item())}, '
                    f'{format_number(history_trajectory_sample[j, 2].item())})'
                    for j in range(history_trajectory_sample.shape[0])
                ])
                
                prompt = (
                    "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
                    "1. Visual perception from front camera view\n"
                    f"2. Historical motion context (last 4 timesteps):{history_str}\n"
                    f"3. Active navigation command: [{command_str_sample.upper()}]"
                )
                output_requirements = (
                    "\nOutput requirements:\n- Predict 8 future trajectory points\n"
                    "- Each point format: (x:float, y:float, heading:float)\n"
                    "- Use [PT, ...] to encapsulate the trajectory\n"
                    "- Maintain numerical precision to 2 decimal places"
                )
                questions.append(f"{prompt}{output_requirements}")

            frozen_hidden_state = None
            should_compute_anchor, hidden_anchor_step_index = self._next_hidden_anchor_decision()
            if should_compute_anchor:
                frozen_hidden_state = self._compute_frozen_vlm_hidden(pixel_values_cat, questions, num_patches_list)
                hidden_anchor_computed = frozen_hidden_state is not None
            outputs = self.backbone(pixel_values_cat, questions, num_patches_list=num_patches_list)
            last_hidden_state = outputs.hidden_states[-1]
            if frozen_hidden_state is not None:
                hidden_anchor_loss, hidden_drift_cosine, hidden_drift_l2 = self._hidden_anchor_metrics(
                    last_hidden_state,
                    frozen_hidden_state.to(device=last_hidden_state.device, dtype=last_hidden_state.dtype),
                )

        status_feature = features["status_feature"].to(action_device)
        if status_feature.ndim == 1:
            status_feature = status_feature.unsqueeze(0)
        if last_hidden_state.ndim == 2:
            last_hidden_state = last_hidden_state.unsqueeze(0)

        last_hidden_state = last_hidden_state.to(model_dtype)
        history_trajectory_reshaped = history_trajectory.view(history_trajectory.size(0), -1)
        input_state = torch.cat([status_feature, history_trajectory_reshaped], dim=1)
        action_input_data = {
            "state": input_state.to(model_dtype),
            "his_traj": history_trajectory_reshaped.to(model_dtype),
            "history_trajectory": history_trajectory.to(model_dtype),
            "status_feature": status_feature.to(model_dtype),
            "high_command_one_hot": high_command_one_hot.to(model_dtype),
        }
        target_loss_mode = self.training or targets is not None
        optional_feature_keys = tuple(
            dict.fromkeys(
                (
                    *EXPERT_FEATURE_KEYS,
                    *(LAST_VLA_FEATURE_KEYS if self.use_last_vla else ()),
                    *(TWO_EXPERT_FEATURE_KEYS if self.use_two_expert_slots else ()),
                )
            )
        )
        target_feature_keys = set(EXPERT_TARGET_FEATURE_KEYS)
        if self.use_last_vla:
            target_feature_keys.update(LAST_VLA_TARGET_KEYS)
        if self.use_two_expert_slots:
            target_feature_keys.update(TWO_EXPERT_TARGET_KEYS)
        for key in optional_feature_keys:
            if key in features and isinstance(features[key], torch.Tensor):
                if key in target_feature_keys and not target_loss_mode:
                    continue
                action_input_data[key] = features[key].to(model_dtype)

        stage3_objective = getattr(self, "stage3_objective", "grpo" if self.grpo else "none")
        if targets is not None and stage3_objective == "none":
            action_inputs = BatchFeature(
                data={
                    **action_input_data,
                    "action": targets["trajectory"].to(device=action_device, dtype=model_dtype),
                    "_allow_target_tokens_for_loss": True,
                }
            )
            predictions = self.action_head(last_hidden_state, action_inputs)
            self._attach_hidden_anchor_outputs(
                predictions,
                hidden_anchor_loss,
                hidden_drift_cosine,
                hidden_drift_l2,
                hidden_anchor_computed=hidden_anchor_computed,
                hidden_anchor_step_index=hidden_anchor_step_index,
            )
            return predictions
        elif self.training and stage3_objective == "grpo":
            action_inputs = BatchFeature(
                data={**action_input_data, "action": targets["trajectory"].to(device=action_device, dtype=model_dtype)}
            )
            return self.action_head.forward_grpo(
                last_hidden_state,
                action_inputs,
                tokens_list,
                sample_time=self.grpo_sample_time,
            )
        elif self.training and stage3_objective == "grpo_replay":
            action_inputs = BatchFeature(
                data={**action_input_data, "action": targets["trajectory"].to(device=action_device, dtype=model_dtype)}
            )
            return self.action_head.collect_grpo_replay_rollout(
                last_hidden_state,
                action_inputs,
                tokens_list,
                sample_time=self.grpo_sample_time,
            )
        elif self.training and stage3_objective in {"awac_iql", "hybrid"}:
            action_inputs = BatchFeature(
                data={**action_input_data, "action": targets["trajectory"].to(device=action_device, dtype=model_dtype)}
            )
            return self.action_head.forward_awac_iql(
                last_hidden_state,
                action_inputs,
                tokens_list,
            )
        else: 
            action_inputs = BatchFeature(action_input_data)
            return self.action_head.get_action(last_hidden_state.to(model_dtype), action_inputs)

    @staticmethod
    def _is_expert_parameter_key(key: str) -> bool:
        expert_markers = (
            "jepa_projector",
            "vggt_projector",
            "jepa_adapter",
            "vggt_adapter",
            "jepa_alignment_head",
            "vggt_alignment_head",
            "jepa_type_embedding",
            "vggt_type_embedding",
            "z_jepa_type_embedding",
            "z_vggt_type_embedding",
            "jepa_gate",
            "vggt_gate",
            "jepa_horizon_conditioner",
            "vggt_horizon_conditioner",
            "branch_logits",
            "last_rd",
            "two_expert_",
            "two_expert_slots",
            "two_expert_h_dyn",
            "two_expert_h_geo",
        )
        return key.startswith("action_head.") and any(marker in key for marker in expert_markers)

    @staticmethod
    def _is_last_vla_parameter_key(key: str) -> bool:
        return key.startswith("action_head.last_vla_cot.")

    @staticmethod
    def _is_last_vla_condition_parameter_key(key: str) -> bool:
        if not key.startswith("action_head."):
            return False
        condition_markers = (
            "last_vla_context_mean_cot_proj",
            "last_vla_horizon_cot_queries",
            "last_vla_horizon_cot_attn",
            "last_vla_horizon_cot_proj",
            "last_vla_cot_condition_scale",
            "cot_cross_attn",
            "cot_out_proj",
            "cot_condition_scale",
        )
        return any(marker in key for marker in condition_markers)

    @staticmethod
    def _is_gate_parameter_key(key: str) -> bool:
        gate_markers = ("jepa_gate", "vggt_gate", "branch_logits", "scene_gate", "timestep_gate", "two_expert_denoise_gate")
        return key.startswith("action_head.") and any(marker in key for marker in gate_markers)

    @staticmethod
    def _is_action_head_parameter_key(key: str) -> bool:
        return key.startswith("action_head.")

    def _optimizer_grouping_requested(self) -> bool:
        return (
            self.last_vla_train_vlm_lora
            or self.lr_action_head is not None
            or self.lr_expert is not None
            or self.lr_expert_gate is not None
            or self.train_expert_only
            or self.freeze_base_action_head
            or self.freeze_expert
        )

    def _set_trainable_parameters(self) -> None:
        if self.last_vla_train_vlm_lora:
            for name, parameter in self.named_parameters():
                is_last_vla_cot = self._is_last_vla_parameter_key(name)
                is_last_vla_condition = self._is_last_vla_condition_parameter_key(name)
                is_lora = "lora_" in name
                parameter.requires_grad = bool(is_last_vla_cot or is_last_vla_condition or is_lora)
            counts = self.count_trainable_parameters_by_group()
            if counts["vlm_lora"]["trainable"] <= 0:
                raise RuntimeError("last_vla_train_vlm_lora=True but no trainable LoRA parameters were found.")
            if counts["last_vla_cot"]["trainable"] <= 0:
                raise RuntimeError("last_vla_train_vlm_lora=True but no trainable Last-VLA CoT parameters were found.")
            if counts["cot_condition_branch"]["trainable"] <= 0:
                raise RuntimeError("last_vla_train_vlm_lora=True but no trainable CoT condition branch parameters were found.")
            if counts["action_base"]["trainable"] > 0:
                raise RuntimeError("Last-VLA VLM-LoRA training must not train action_base parameters.")
            if counts["backbone_non_lora"]["trainable"] > 0:
                raise RuntimeError("Last-VLA VLM-LoRA training must not train non-LoRA backbone parameters.")
            return
        if not self.use_last_vla:
            for name, parameter in self.named_parameters():
                if self._is_last_vla_condition_parameter_key(name):
                    parameter.requires_grad = False
        if self.freeze_expert and (self.train_expert_only or self.freeze_base_action_head):
            raise ValueError(
                "freeze_expert leaves no trainable A4 parameters when combined with "
                "train_expert_only/freeze_base_action_head."
            )
        if not (self.train_expert_only or self.freeze_base_action_head or self.freeze_expert):
            return

        for name, parameter in self.named_parameters():
            if not self._is_action_head_parameter_key(name):
                if self.train_expert_only or self.freeze_base_action_head:
                    parameter.requires_grad = False
                continue
            is_expert = self._is_expert_parameter_key(name)
            is_last_vla = self._is_last_vla_parameter_key(name)
            is_last_vla_condition = self._is_last_vla_condition_parameter_key(name)
            if self.train_expert_only or self.freeze_base_action_head:
                if self.use_last_vla:
                    parameter.requires_grad = (is_last_vla or is_last_vla_condition) and not self.freeze_expert
                else:
                    parameter.requires_grad = is_expert and not self.freeze_expert
            elif self.freeze_expert:
                parameter.requires_grad = not (is_expert or is_last_vla)

    @staticmethod
    def _append_optimizer_group(
        groups: List[Dict[str, Any]],
        *,
        params: List[torch.nn.Parameter],
        lr: float,
        base_lr: float,
        weight_decay: float,
        name: str,
    ) -> None:
        if not params or lr <= 0.0:
            return
        group: Dict[str, Any] = {
            "params": params,
            "lr": lr,
            "weight_decay": weight_decay,
            "name": name,
        }
        if base_lr > 0.0:
            group["lr_scale"] = lr / base_lr
        groups.append(group)

    def _build_grouped_optimizer(self) -> Optimizer:
        base_lr = float(self._lr)
        if self.last_vla_train_vlm_lora:
            return self._build_last_vla_lora_optimizer(base_lr)
        action_lr = base_lr if self.lr_action_head is None else float(self.lr_action_head)
        expert_lr = base_lr if self.lr_expert is None else float(self.lr_expert)
        gate_lr = None if self.lr_expert_gate is None else float(self.lr_expert_gate)

        action_params: List[torch.nn.Parameter] = []
        expert_params: List[torch.nn.Parameter] = []
        gate_params: List[torch.nn.Parameter] = []
        backbone_params: List[torch.nn.Parameter] = []

        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if self._is_gate_parameter_key(name) and gate_lr is not None:
                gate_params.append(parameter)
            elif self._is_expert_parameter_key(name):
                expert_params.append(parameter)
            elif self._is_action_head_parameter_key(name):
                action_params.append(parameter)
            elif self.backbone is not None and name.startswith("backbone."):
                backbone_params.append(parameter)

        groups: List[Dict[str, Any]] = []
        self._append_optimizer_group(
            groups,
            params=expert_params,
            lr=expert_lr,
            base_lr=base_lr,
            weight_decay=1e-4,
            name="expert",
        )
        self._append_optimizer_group(
            groups,
            params=gate_params,
            lr=gate_lr if gate_lr is not None else 0.0,
            base_lr=base_lr,
            weight_decay=0.0,
            name="expert_gate",
        )
        self._append_optimizer_group(
            groups,
            params=action_params,
            lr=action_lr,
            base_lr=base_lr,
            weight_decay=1e-4,
            name="action_head",
        )
        self._append_optimizer_group(
            groups,
            params=backbone_params,
            lr=action_lr,
            base_lr=base_lr,
            weight_decay=1e-4,
            name="backbone",
        )
        if not groups:
            raise RuntimeError("No trainable parameters for ReCogDrive optimizer. Check freeze flags and learning rates.")
        self._optimizer_group_report = self._optimizer_group_report_from_groups(groups)
        return optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))

    def _build_last_vla_lora_optimizer(self, base_lr: float) -> Optimizer:
        self._set_trainable_parameters()
        last_vla_cot_params: List[torch.nn.Parameter] = []
        vlm_lora_params: List[torch.nn.Parameter] = []
        action_base_params: List[torch.nn.Parameter] = []
        backbone_non_lora_params: List[torch.nn.Parameter] = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if self._is_last_vla_parameter_key(name) or self._is_last_vla_condition_parameter_key(name):
                last_vla_cot_params.append(parameter)
            elif "lora_" in name:
                vlm_lora_params.append(parameter)
            elif name.startswith("action_head."):
                action_base_params.append(parameter)
            elif name.startswith("backbone."):
                backbone_non_lora_params.append(parameter)
        if action_base_params:
            raise RuntimeError("LoRA alignment optimizer found trainable action_base parameters.")
        if backbone_non_lora_params:
            raise RuntimeError("LoRA alignment optimizer found trainable non-LoRA backbone parameters.")
        if not last_vla_cot_params:
            raise RuntimeError("LoRA alignment optimizer requires a non-empty Last-VLA CoT parameter group.")
        if not vlm_lora_params:
            raise RuntimeError("LoRA alignment optimizer requires a non-empty VLM LoRA parameter group.")
        cot_lr = float(self.lr_last_vla_cot) if self.lr_last_vla_cot is not None else float(self.lr_action_head or base_lr)
        lora_lr = float(self.lr_vlm_lora) if self.lr_vlm_lora is not None else 1e-5
        groups: List[Dict[str, Any]] = [
            {
                "params": last_vla_cot_params,
                "lr": cot_lr,
                "weight_decay": float(self.weight_decay_last_vla_cot),
                "name": "last_vla_cot",
                "lr_scale": cot_lr / base_lr if base_lr > 0 else 1.0,
            },
            {
                "params": vlm_lora_params,
                "lr": lora_lr,
                "weight_decay": float(self.weight_decay_vlm_lora),
                "name": "vlm_lora",
                "lr_scale": lora_lr / base_lr if base_lr > 0 else 1.0,
            },
        ]
        self._optimizer_group_report = self._optimizer_group_report_from_groups(groups)
        if int(os.getenv("LOCAL_RANK", os.getenv("RANK", "0"))) == 0:
            for item in self._optimizer_group_report:
                print(
                    "Optimizer group "
                    f"{item['name']}: lr={item['lr']}, weight_decay={item['weight_decay']}, "
                    f"tensors={item['num_tensors']}, parameters={item['parameter_count']}"
                )
        return optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))

    @staticmethod
    def _optimizer_group_report_from_groups(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        report = []
        for group in groups:
            params = [param for param in group.get("params", []) if isinstance(param, torch.nn.Parameter)]
            report.append(
                {
                    "name": group.get("name", "unnamed"),
                    "lr": float(group.get("lr", 0.0)),
                    "weight_decay": float(group.get("weight_decay", 0.0)),
                    "num_tensors": int(len(params)),
                    "parameter_count": int(sum(param.numel() for param in params)),
                }
            )
        return report

    def _hidden_anchor_active(self) -> bool:
        if not (self.training and self.last_vla_train_vlm_lora):
            return False
        if self.cache_hidden_state:
            return False
        if self.last_vla_stage != "cot_alignment":
            return False
        if self.last_vla_hidden_anchor_mode == "none" or self.last_vla_hidden_anchor_weight <= 0.0:
            return False
        return True

    def _next_hidden_anchor_decision(self) -> tuple[bool, Optional[int]]:
        if not self._hidden_anchor_active():
            return False, None
        step_index = self._hidden_anchor_forward_index
        self._hidden_anchor_forward_index += 1
        return step_index % self.last_vla_hidden_anchor_every_n_steps == 0, step_index

    def _compute_frozen_vlm_hidden(self, pixel_values: torch.Tensor, questions: List[str], num_patches_list: List[int]) -> Optional[torch.Tensor]:
        if self.backbone is None or getattr(self.backbone, "model", None) is None:
            return None
        model = self.backbone.model
        disable_adapter = getattr(model, "disable_adapter", None)
        if disable_adapter is None:
            warnings.warn(
                "PEFT model does not expose disable_adapter(); hidden-anchor loss is skipped for this step.",
                RuntimeWarning,
            )
            return None
        with torch.no_grad():
            with disable_adapter():
                outputs = self.backbone(pixel_values, questions, num_patches_list=num_patches_list)
        return outputs.hidden_states[-1].detach()

    def _hidden_anchor_metrics(
        self,
        lora_hidden: torch.Tensor,
        frozen_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if lora_hidden.ndim == 2:
            lora_hidden = lora_hidden.unsqueeze(0)
        if frozen_hidden.ndim == 2:
            frozen_hidden = frozen_hidden.unsqueeze(0)
        token_count = min(lora_hidden.shape[1], frozen_hidden.shape[1])
        lora = lora_hidden[:, :token_count].float()
        frozen = frozen_hidden[:, :token_count].float().detach()
        lora_mean = lora.mean(dim=1)
        frozen_mean = frozen.mean(dim=1)
        cosine = torch.nn.functional.cosine_similarity(lora_mean, frozen_mean, dim=-1).mean()
        l2 = torch.nn.functional.mse_loss(lora_mean, frozen_mean)
        if self.last_vla_hidden_anchor_mode in {"summary_cosine", "token_mean_cosine"}:
            loss = 1.0 - cosine
        elif self.last_vla_hidden_anchor_mode == "mse_mean":
            loss = l2
        else:
            loss = lora.new_zeros(())
        return loss.to(lora_hidden), cosine.to(lora_hidden), l2.to(lora_hidden)

    def _attach_hidden_anchor_outputs(
        self,
        predictions: Dict[str, torch.Tensor],
        hidden_anchor_loss: Optional[torch.Tensor],
        hidden_drift_cosine: Optional[torch.Tensor],
        hidden_drift_l2: Optional[torch.Tensor],
        *,
        hidden_anchor_computed: bool = False,
        hidden_anchor_step_index: Optional[int] = None,
    ) -> None:
        ref_tensor = predictions.get("loss")
        if ref_tensor is None and hidden_anchor_loss is not None:
            ref_tensor = hidden_anchor_loss
        if ref_tensor is not None and hidden_anchor_step_index is not None:
            ref = ref_tensor.detach()
            predictions["hidden_anchor_computed"] = ref.new_tensor(1.0 if hidden_anchor_computed else 0.0)
            predictions["hidden_anchor_every_n_steps_tensor"] = ref.new_tensor(float(self.last_vla_hidden_anchor_every_n_steps))
            predictions["hidden_anchor_step_index"] = ref.new_tensor(float(hidden_anchor_step_index))
        if hidden_anchor_loss is None:
            if ref_tensor is not None and self._last_vla_lora_audit is not None:
                ref = ref_tensor.detach()
                predictions["lora_trainable_param_count"] = ref.new_tensor(
                    float(self._last_vla_lora_audit.get("trainable_lora_param_count", 0))
                )
                predictions["lora_matched_module_count"] = ref.new_tensor(
                    float(self._last_vla_lora_audit.get("matched_total", 0))
                )
            return
        weighted = hidden_anchor_loss * float(self.last_vla_hidden_anchor_weight)
        if "loss" in predictions:
            predictions["loss"] = predictions["loss"] + weighted.to(predictions["loss"])
        predictions["hidden_anchor_loss"] = hidden_anchor_loss.detach()
        predictions["hidden_anchor_loss_weighted"] = weighted.detach()
        if hidden_drift_cosine is not None:
            predictions["hidden_drift_cosine"] = hidden_drift_cosine.detach()
        if hidden_drift_l2 is not None:
            predictions["hidden_drift_l2"] = hidden_drift_l2.detach()
        if self._last_vla_lora_audit is not None:
            ref = hidden_anchor_loss.detach()
            predictions["lora_trainable_param_count"] = ref.new_tensor(
                float(self._last_vla_lora_audit.get("trainable_lora_param_count", 0))
            )
            predictions["lora_matched_module_count"] = ref.new_tensor(
                float(self._last_vla_lora_audit.get("matched_total", 0))
            )

    @staticmethod
    def _checkpoint_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
        if isinstance(checkpoint, dict):
            return checkpoint
        raise TypeError(f"Checkpoint must be a dict or contain a 'state_dict' dict, got {type(checkpoint).__name__}.")

    @staticmethod
    def _checkpoint_file_candidates(path: Path) -> List[Path]:
        if path.is_file():
            return [path]
        if not path.exists():
            return []
        suffixes = (".ckpt", ".pth", ".pt", ".safetensors")
        candidates: List[Path] = []
        for suffix in suffixes:
            candidates.extend(p for p in path.rglob(f"*{suffix}") if p.is_file())
        return sorted(set(candidates))

    @staticmethod
    def _checkpoint_candidate_score(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        size = path.stat().st_size if path.is_file() else 0
        score = 0
        if "il" in name:
            score += 1000
        if "model" in name:
            score += 500
        if path.suffix in {".ckpt", ".pth", ".pt"}:
            score += 100
        return score, size, str(path)

    def _resolve_checkpoint_path(self, checkpoint_path: str) -> Optional[Path]:
        path = Path(checkpoint_path)
        candidates = self._checkpoint_file_candidates(path)
        if not candidates:
            return None
        selected = max(candidates, key=self._checkpoint_candidate_score)
        if path.is_dir():
            print(f"Resolved checkpoint directory {path} to {selected}")
            print("  checkpoint candidates:")
            for candidate in candidates:
                marker = " <= selected" if candidate == selected else ""
                print(f"    - {candidate} ({candidate.stat().st_size} bytes){marker}")
        return selected

    def _load_checkpoint_file(self, path: Path) -> Dict[str, torch.Tensor]:
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            return dict(load_file(str(path), device="cpu"))
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        return self._checkpoint_state_dict(checkpoint)

    def _safe_load_checkpoint(self, checkpoint_path: str) -> None:
        path = self._resolve_checkpoint_path(checkpoint_path)
        if path is None:
            if self.allow_random_init:
                warnings.warn(
                    f"Checkpoint not found at {checkpoint_path}. Continuing with random initialization because "
                    "allow_random_init=True. This must not be interpreted as driving performance.",
                    RuntimeWarning,
                )
                return
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state_dict = self._load_checkpoint_file(path)
        model_dict = self.state_dict()

        filtered_state: Dict[str, torch.Tensor] = {}
        unexpected_keys: List[str] = []
        skipped_expert_shape: List[str] = []
        shape_mismatches: List[str] = []

        for key, value in state_dict.items():
            mapped_key = key[len("agent."):] if key.startswith("agent.") else key
            if mapped_key not in model_dict:
                unexpected_keys.append(mapped_key)
                continue
            if not isinstance(value, torch.Tensor):
                unexpected_keys.append(mapped_key)
                continue
            expected = model_dict[mapped_key]
            if expected.shape != value.shape:
                message = f"{mapped_key}: checkpoint {tuple(value.shape)} vs model {tuple(expected.shape)}"
                if self._is_expert_parameter_key(mapped_key) or self._is_last_vla_parameter_key(mapped_key):
                    skipped_expert_shape.append(message)
                    continue
                shape_mismatches.append(message)
                continue
            filtered_state[mapped_key] = value

        if shape_mismatches:
            raise RuntimeError(
                "Checkpoint shape mismatch for existing baseline parameters:\n"
                + "\n".join(f"  - {item}" for item in shape_mismatches)
            )

        incompatible = self.load_state_dict(filtered_state, strict=False)
        missing_keys = list(incompatible.missing_keys)
        missing_expert = [key for key in missing_keys if self._is_expert_parameter_key(key) or self._is_last_vla_parameter_key(key)]
        missing_other = [key for key in missing_keys if key not in missing_expert]
        verbose_key_log = os.environ.get("RECOGDRIVE_CHECKPOINT_LOAD_VERBOSE", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        try:
            key_print_limit = int(os.environ.get("RECOGDRIVE_CHECKPOINT_LOAD_KEY_PRINT_LIMIT", "40"))
        except ValueError:
            key_print_limit = 40
        key_print_limit = max(0, key_print_limit)

        def _print_key_summary(title: str, keys: List[str]) -> None:
            if not keys:
                return
            print(f"  {title}: {len(keys)}")
            shown_keys = keys if verbose_key_log else keys[:key_print_limit]
            for key in shown_keys:
                print(f"    - {key}")
            omitted = len(keys) - len(shown_keys)
            if omitted > 0:
                print(
                    f"    ... omitted {omitted} keys; set RECOGDRIVE_CHECKPOINT_LOAD_VERBOSE=1 "
                    "to print the full list"
                )

        print(f"Loaded checkpoint from {path} with strict=False.")
        print(f"  loaded keys: {len(filtered_state)}")
        print(f"  missing keys: {len(missing_keys)}")
        _print_key_summary("expected missing expert keys", missing_expert)
        _print_key_summary("missing non-expert keys", missing_other)
        _print_key_summary("unexpected keys", [*unexpected_keys, *incompatible.unexpected_keys])
        _print_key_summary("skipped expert shape mismatches", skipped_expert_shape)

    def _safe_load_last_rd_adapter(self, checkpoint_path: str) -> None:
        path = self._resolve_checkpoint_path(checkpoint_path)
        if path is None:
            raise FileNotFoundError(f"LaST-RD adapter checkpoint not found: {checkpoint_path}")
        state_dict = self._load_checkpoint_file(path)
        model_dict = self.state_dict()
        filtered: Dict[str, torch.Tensor] = {}
        skipped: List[str] = []
        for key, value in state_dict.items():
            mapped_key = key[len("agent."):] if key.startswith("agent.") else key
            if mapped_key.startswith("action_head.last_rd."):
                candidate = mapped_key
            elif mapped_key.startswith("last_rd."):
                candidate = f"action_head.{mapped_key}"
            elif mapped_key.startswith("action_head.") and ".last_rd." in mapped_key:
                candidate = mapped_key
            else:
                continue
            if candidate in model_dict and isinstance(value, torch.Tensor) and tuple(model_dict[candidate].shape) == tuple(value.shape):
                filtered[candidate] = value
            else:
                skipped.append(candidate)
        if not filtered:
            raise RuntimeError(f"No compatible LaST-RD adapter weights found in {path}.")
        incompatible = self.load_state_dict(filtered, strict=False)
        print(f"Loaded LaST-RD adapter weights from {path}: {len(filtered)} tensors.")
        if skipped:
            print("  skipped incompatible LaST-RD keys:")
            for key in skipped[:20]:
                print(f"    - {key}")
        missing_last_rd = [key for key in incompatible.missing_keys if "last_rd" in key]
        if missing_last_rd:
            print(f"  remaining missing LaST-RD keys: {len(missing_last_rd)}")

    def _safe_load_last_vla_adapter(self, checkpoint_path: str) -> None:
        path = self._resolve_checkpoint_path(checkpoint_path)
        if path is None:
            raise FileNotFoundError(f"Last-VLA adapter checkpoint not found: {checkpoint_path}")
        state_dict = self._load_checkpoint_file(path)
        model_dict = self.state_dict()
        filtered: Dict[str, torch.Tensor] = {}
        skipped: List[str] = []
        for key, value in state_dict.items():
            mapped_key = key[len("agent."):] if key.startswith("agent.") else key
            if mapped_key.startswith("action_head.last_vla_cot."):
                candidate = mapped_key
            elif mapped_key.startswith("last_vla_cot."):
                candidate = f"action_head.{mapped_key}"
            elif mapped_key.startswith("action_head.") and ".last_vla_cot." in mapped_key:
                candidate = mapped_key
            else:
                continue
            if candidate in model_dict and isinstance(value, torch.Tensor) and tuple(model_dict[candidate].shape) == tuple(value.shape):
                filtered[candidate] = value
            else:
                skipped.append(candidate)
        if not filtered:
            raise RuntimeError(f"No compatible Last-VLA adapter weights found in {path}.")
        incompatible = self.load_state_dict(filtered, strict=False)
        print(f"Loaded Last-VLA adapter weights from {path}: {len(filtered)} tensors.")
        if skipped:
            print("  skipped incompatible Last-VLA keys:")
            for key in skipped[:20]:
                print(f"    - {key}")
        missing_last_vla = [key for key in incompatible.missing_keys if "last_vla_cot" in key]
        if missing_last_vla:
            print(f"  remaining missing Last-VLA keys: {len(missing_last_vla)}")

    def _add_dummy_expert_features_if_needed(
        self,
        features: Dict[str, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if (
            not self.use_expert_features
            or self.expert_feature_source != "dummy"
            or self._dummy_expert_backend is None
        ):
            return

        required_keys = []
        if self.use_jepa:
            required_keys.append("jepa_context_tokens")
        if self.use_vggt:
            required_keys.append("vggt_context_tokens")
        if all(key in features for key in required_keys):
            return

        batch_size = features["history_trajectory"].shape[0] if features["history_trajectory"].ndim >= 3 else 1
        include_targets = self.training and self.allow_expert_target_features
        dummy_features = self._dummy_expert_backend.generate(
            batch_size,
            device=device,
            dtype=dtype,
            include_targets=include_targets,
        )
        for key, value in dummy_features.items():
            features.setdefault(key, value)

        if not self._warned_dummy_features:
            warnings.warn(
                f"{DUMMY_EXPERT_WARNING} No benchmark metrics should be reported in dummy mode.",
                RuntimeWarning,
            )
            self._warned_dummy_features = True

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        self.eval()

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["pred_traj"].float().cpu().squeeze(0)

        return Trajectory(poses)

    def compute_trajectory_vis(self, agent_input: AgentInput) -> Trajectory:
        self.eval()

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["pred_traj"].float().cpu().squeeze(0)
        return Trajectory(poses)


    def compute_loss(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.training and getattr(self, "stage3_objective", "none") in {"grpo", "grpo_replay", "awac_iql", "hybrid"}:
            return predictions
        elif isinstance(predictions, dict) and "loss" in predictions:
            return predictions["loss"]
        elif hasattr(predictions, "loss"):
            return predictions.loss
        else:
            return torch.nn.functional.l1_loss(predictions["pred_traj"], targets["trajectory"])

    def get_optimizers(self) -> Union[Optimizer, Dict[str, LRScheduler]]:
        self._set_trainable_parameters()

        if self._optimizer_grouping_requested():
            optimizer = self._build_grouped_optimizer()
        else:
            optimizer_cfg = DictConfig(dict(type="AdamW", lr=self._lr, weight_decay=1e-4, betas=(0.9, 0.95)))
            params = [parameter for parameter in self.action_head.parameters() if parameter.requires_grad]
            if self.backbone is not None and self.train_backbone:
                params += [parameter for parameter in self.backbone.parameters() if parameter.requires_grad]
            if not params:
                raise RuntimeError("No trainable parameters for ReCogDrive optimizer.")
            optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        
        if self.grpo:
            scheduler = WarmupCosLR(
                optimizer=optimizer,
                lr=self._lr,
                min_lr=self.grpo_scheduler_min_lr,
                epochs=self.grpo_scheduler_epochs,
                warmup_epochs=self.grpo_scheduler_warmup_epochs,
            )
        else:
            scheduler = WarmupCosLR(
                optimizer=optimizer,
                lr=self._lr,
                min_lr=self.scheduler_min_lr,
                epochs=self.scheduler_epochs,
                warmup_epochs=self.scheduler_warmup_epochs,
            )
            
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    @staticmethod
    def _decode_paths_from_tensor(path_tensor: torch.Tensor) -> List[str]:
        """
        Decodes a batch of path tensors back into a list of file path strings.
        
        Args:
            path_tensor (torch.Tensor): A 2D tensor of shape 
                (batch_size, max_path_length) from the collate_fn.
        
        Returns:
            List[str]: A list of decoded file path strings.
        """
        decoded_paths = []
        for single_path_tensor in path_tensor:
            chars = []
            for code in single_path_tensor:
                code_item = code.item()
                if code_item == 0: 
                    break
                chars.append(chr(code_item))
            decoded_paths.append("".join(chars))
        return decoded_paths

def make_recogdrive_config(
    size: str,
    *,
    action_dim: int,
    action_horizon: int,
    input_embedding_dim: int,
    sampling_method: str = 'ddim',
    num_inference_steps: int = 5,
    grpo: bool = False,
    model_dtype: str = "float16",
) -> ReCogDriveDiffusionPlannerConfig:
    """
    A factory function to create a ReCogDriveDiffusionPlannerConfig object.

    This function simplifies configuration by using a size preset ("small",
    "large", "large_new") to define the core DiT architecture, while allowing
    other important planner settings to be specified.

    Args:
        size (str): The size preset for the DiT backbone.
        action_dim (int): The dimension of the action space.
        action_horizon (int): The number of future action steps to predict.
        input_embedding_dim (int): Dimension of the input embeddings to the DiT.
        sampling_method (str): The core training and sampling methodology.
        num_inference_steps (int): Number of steps for inference sampling.
        grpo (bool): If True, enables GRPO-specific logic.
        model_dtype (str): The data type for model computations.

    Returns:
        ReCogDriveDiffusionPlannerConfig: An instantiated and configured planner config object.
    """
    size = size.lower()
    if size == "small":
        diffusion_model_cfg = {"num_heads": 8, "head_dim": 48, "num_layers": 16,"output_dim":512}
    elif size == "large":
        diffusion_model_cfg = {"num_heads": 32, "head_dim": 48, "num_layers": 16,"output_dim":1536}
    else:
        raise ValueError(f"Unknown model size: {size!r}")

    common_params: Dict[str, any] = {
        "dropout": 0.0,
        "attention_bias": True,
        "norm_eps": 1e-5,
        "interleave_attention": True,
    }
    diffusion_model_cfg.update(common_params)

    config = ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg=diffusion_model_cfg,
        action_dim=action_dim,
        action_horizon=action_horizon,
        input_embedding_dim=input_embedding_dim,
        sampling_method=sampling_method,
        num_inference_steps=num_inference_steps,
        grpo=grpo,
        model_dtype=model_dtype,
    )
    
    return config
