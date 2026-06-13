# Copyright 2025 The Xiaomi Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import copy
import hashlib
import lzma
import pickle
import warnings
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import torch
from timm.models.layers import Mlp
from torch import nn
from torch.distributions import Beta, Normal, kl_divergence
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.evaluate.pdm_score_batch import pdm_score_batch_same_cache
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
    PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.scoring.fast_pdm_scorer import FastPDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)

from .blocks.encoder import (
    ActionEncoder,
    SinusoidalPositionalEncoding,
    StateAttentionEncoder,
    SwiGLUFFN,
)
from .recogdrive_dit import LightningDiT
from .offline_action_explorer import build_structured_perturbations
from .offline_rl_buffer import REQUIRED_COMPONENT_KEYS, load_elite_record, load_elite_record_path
from .expert_fusion import (
    AlignmentHead,
    ExpertAdapter768,
    HorizonAwareExpertConditioner,
    TeacherTokenProjector,
    branch_logits_from_probs,
    init_logit_from_prob,
    normalized_mse_loss,
)

@dataclass
class FlowConfig:
    """Configuration specific to Flow Matching."""
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    mean_variance_net: bool = False

@dataclass
class DDPMConfig:
    """Configuration specific to DDPM."""
    num_train_timesteps: int = 100

@dataclass
class DDIMConfig:
    """Configuration specific to DDIM."""
    num_train_timesteps: int = 100
    ddim_eta: float = 0.0

@dataclass
class GRPOConfig:
    """Configuration specific to GRPO training."""
    denoised_clip_value: float = 1.0
    eval_randn_clip_value: float = 1.0
    randn_clip_value: float = 5.0
    final_action_clip_value: float = 1.0
    eps_clip_value: Optional[float] = None
    eval_min_sampling_denoising_std: float = 0.0001
    min_sampling_denoising_std: float = 0.04
    min_logprob_denoising_std: float = 0.1
    clip_advantage_lower_quantile: float = 0.0
    clip_advantage_upper_quantile: float = 1.0
    gamma_denoising: float = 0.6
    sample_time: int = 8
    bc_anneal: bool = False
    bc_coeff_start: float = 0.1
    bc_coeff_end: float = 0.1
    bc_anneal_epochs: int = 1
    
    metric_cache_path: str = "/path/to/metric_cache_train"
    reference_policy_checkpoint: str = "/path/to/IL_Model.ckpt"
    scorer_config: PDMScorerConfig = field(default_factory=lambda: PDMScorerConfig(
        progress_weight=10.0, ttc_weight=5.0, comfortable_weight=2.0
    ))

    # reward extraction and shaping
    use_safety_shaped_reward: bool = True
    hard_gate_nc: bool = True
    hard_gate_dac: bool = True
    hard_gate_ttc: bool = False
    hard_gate_ddc: bool = False
    hard_gate_tlc: bool = False

    nc_safe_threshold: float = 1.0
    dac_safe_threshold: float = 1.0
    ttc_safe_threshold: float = 1.0
    ddc_safe_threshold: float = 1.0
    tlc_safe_threshold: float = 1.0

    progress_bonus_weight: float = 0.05
    unsafe_reward_floor: float = -0.5
    unsafe_pdms_scale: float = 0.05

    # asymmetric safe advantage
    use_asymmetric_safe_advantage: bool = True
    advantage_std_floor: float = 0.05
    safe_negative_adv_scale: float = 0.2
    unsafe_advantage_offset: float = 1.0

    # trajectory-level objective
    use_trajectory_level_objective: bool = True
    trajectory_logprob_reduce: Literal["mean", "discounted_mean"] = "discounted_mean"

    # optional strict GSPO ratio; default OFF
    use_gspo_ratio: bool = False
    gspo_clip_low: float = 0.05
    gspo_clip_high: float = 0.05
    behavior_policy_sync_interval: int = 4
    behavior_policy_sample: bool = True

    # dynamic group weighting
    use_dynamic_group_weight: bool = True
    min_group_reward_std: float = 0.02
    all_safe_low_std_group_weight: float = 0.2
    all_unsafe_group_weight: float = 0.25

    # diversity logging / optional reward
    log_safe_diversity: bool = True
    use_diversity_reward: bool = False
    diversity_reward_weight: float = 0.005
    diversity_pdms_threshold: float = 0.7
    diversity_distance_scale: float = 5.0
    diversity_metric: Literal["endpoint", "trajectory"] = "endpoint"


@dataclass
class OfflineRLConfig:
    """Configuration for Stage3 offline AWAC/IQL policy improvement."""

    enabled: bool = False

    # elite buffer
    elite_buffer_path: str = ""
    missing_buffer_policy: Literal["error", "fallback_gt"] = "error"
    elite_top_m: int = 8
    elite_min_candidates: int = 2
    keep_gt_candidate: bool = True
    keep_il_candidate: bool = True
    elite_buffer_version: int = 2
    require_buffer_valid_mask: bool = True
    allow_v1_buffer_recompute_valid_mask: bool = True
    recompute_buffer_valid_mask_on_load: bool = False
    cache_elite_records_in_memory: bool = False
    preload_elite_buffer: bool = False
    elite_buffer_preload_max_records: int = 0

    # online candidate generation fallback / optional mode
    build_candidates_online: bool = False
    online_policy_samples: int = 8
    online_use_current_policy: bool = True
    online_use_old_policy: bool = True
    online_use_gt: bool = True

    # structured perturbation
    perturb_gt: bool = True
    perturb_il: bool = True
    progress_endpoint_deltas_m: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)
    progress_speed_scales: tuple[float, ...] = (0.95, 1.02, 1.05, 1.08, 1.12)
    progress_time_gammas: tuple[float, ...] = (0.75, 0.85, 0.95, 1.05)
    lateral_offsets_m: tuple[float, ...] = (-0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8)
    endpoint_lateral_offsets_m: tuple[float, ...] = (-0.8, -0.4, 0.4, 0.8)
    timing_slow_first_scales: tuple[float, ...] = (0.7, 0.8, 0.9)
    timing_delay_strengths: tuple[float, ...] = (0.15, 0.25, 0.35)

    # strict reward component extraction for AWAC/IQL
    strict_reward_submetrics: bool = True
    required_reward_submetrics: tuple[str, ...] = (
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "time_to_collision_within_bound",
        "driving_direction_compliance",
    )
    missing_submetric_policy: Literal["error", "unsafe_zero", "warn_default"] = "error"
    use_batched_pdm_scoring: bool = True
    use_exact_array_pdm_state_conversion: bool = True
    use_fast_pdm_scorer: bool = True
    pdm_batch_chunk_size: int = 0
    pdm_shadow_check: bool = False
    pdm_shadow_max_samples: int = 4
    pdm_shadow_max_abs_diff: float = 0.0

    # candidate repair / guards
    clip_candidates_to_norm_range: bool = True
    enforce_forward_monotonic_x: bool = True
    max_heading_step_rad: float = 0.25
    max_final_heading_delta_rad: float = 0.4
    use_final_heading_guard: bool = True

    # candidate selection guards
    require_nc: bool = True
    require_dac: bool = True
    require_ddc_guard: bool = True
    ddc_guard_mode: Literal["relative_or_absolute", "absolute", "relative"] = "relative_or_absolute"
    ddc_min_absolute: float = 0.99
    ddc_max_relative_drop: float = 0.01
    require_ttc_guard: bool = False
    ttc_guard_mode: Literal["relative_or_absolute", "absolute", "relative"] = "relative_or_absolute"
    ttc_min_absolute: float = 0.95
    ttc_max_relative_drop: float = 0.02
    select_valid_topk_only: bool = True
    train_invalid_fallback_candidates: bool = False
    fallback_invalid_candidate_weight: float = 0.0

    # selection score
    prior_distance_weight: float = 0.02
    jerk_penalty_weight: float = 0.005
    select_by: Literal["pdms", "pdms_minus_prior"] = "pdms_minus_prior"

    # AWAC/IQL weighting
    baseline_mode: Literal["max_gt_il", "expectile", "top_mean", "mean"] = "max_gt_il"
    expectile_tau: float = 0.8
    expectile_iters: int = 20
    top_mean_frac: float = 0.2
    advantage_temperature: float = 0.03
    advantage_clip_min: float = -0.2
    advantage_clip_max: float = 0.2
    weight_min: float = 0.05
    weight_max: float = 20.0
    normalize_weights_per_scene: bool = True
    train_only_valid_candidates: bool = True
    min_reward_margin_to_gt_for_extra_weight: float = 0.0
    allow_zero_weight_rows: bool = True
    target_filter_mode: Literal["none", "topk", "positive_advantage", "topk_positive"] = "none"
    target_top_k: int = 1
    target_min_advantage: float = 0.0
    awac_timestep_sampling: Literal["uniform", "ddim", "low_noise", "mid_noise"] = "uniform"
    preference_dpo_timestep_sampling: Literal["uniform", "ddim", "low_noise", "mid_noise"] = "uniform"
    low_noise_timestep_frac: float = 0.35
    mid_noise_timestep_low_frac: float = 0.15
    mid_noise_timestep_high_frac: float = 0.65
    target_blend_mode: Literal["none", "towards_behavior_anchor"] = "none"
    target_blend_alpha: float = 1.0
    target_blend_schedule: Literal["constant", "linear_warmup"] = "constant"
    target_blend_alpha_start: float = 0.0
    target_blend_warmup_start_epoch: int = 0
    target_blend_warmup_epochs: int = 1

    # component-aware advantage shaping
    component_advantage_enabled: bool = False
    component_progress_weight: float = 0.05
    component_safety_penalty_weight: float = 0.20
    component_advantage_clip_min: float = -0.10
    component_advantage_clip_max: float = 0.05

    # source balancing across GT/IL/policy/structured perturbation targets
    source_balance_enabled: bool = False
    source_balance_min_factor: float = 0.50
    source_balance_max_factor: float = 2.00

    # same-scene preference losses on diffusion target likelihood
    pairwise_rank_loss_weight: float = 0.0
    pairwise_rank_margin: float = 0.02
    pairwise_rank_min_reward_gap: float = 0.02
    pairwise_rank_max_pairs_per_scene: int = 2
    invalid_repulsion_loss_weight: float = 0.0
    invalid_repulsion_margin: float = 0.05
    invalid_repulsion_max_pairs_per_scene: int = 2
    preference_dpo_loss_weight: float = 0.0
    preference_dpo_beta: float = 8.0
    preference_dpo_label_smoothing: float = 0.0
    preference_dpo_reference_free: bool = False
    preference_dpo_pair_mode: Literal["best_vs_gt_il", "best_vs_low_valid", "best_vs_all"] = "best_vs_gt_il"
    preference_dpo_min_reward_gap: float = 0.02
    preference_dpo_max_pairs_per_scene: int = 2
    preference_dpo_gap_weight_mode: Literal["none", "pair_gap", "winner_advantage"] = "none"
    preference_dpo_gap_weight_scale: float = 0.05
    preference_dpo_gap_weight_min: float = 0.0
    preference_dpo_gap_weight_max: float = 3.0
    preference_dpo_loss_schedule: Literal["constant", "linear_warmup"] = "constant"
    preference_dpo_loss_weight_start: float = 0.0
    preference_dpo_loss_warmup_start_epoch: int = 0
    preference_dpo_loss_warmup_epochs: int = 1

    # loss weights
    awac_loss_weight: float = 1.0
    awac_loss_schedule: Literal["constant", "linear_warmup"] = "constant"
    awac_loss_weight_start: float = 0.0
    awac_loss_warmup_start_epoch: int = 0
    awac_loss_warmup_epochs: int = 1
    bc_loss_weight: float = 0.05
    bc_loss_schedule: Literal["constant", "linear"] = "constant"
    bc_loss_weight_start: float = 0.05
    bc_loss_weight_end: float = 0.05
    bc_loss_schedule_epochs: int = 1
    grpo_loss_weight: float = 0.0
    grpo_loss_schedule: Literal["constant", "linear_warmup"] = "constant"
    grpo_loss_weight_start: float = 0.0
    grpo_loss_warmup_start_epoch: int = 0
    grpo_loss_warmup_epochs: int = 1

    # debugging / logging
    log_candidate_sources: bool = True
    log_submetrics: bool = True
    log_oracle_stats: bool = True
    require_reference_policy_checkpoint: bool = True
    report_raw_and_valid_best: bool = True


@dataclass
class ReCogDriveDiffusionPlannerConfig(PretrainedConfig):
    """A refined configuration for the ReCogDriveDiffusionPlanner."""
    # --- Core Architecture ---
    diffusion_model_cfg: dict = field(default_factory=dict)
    input_embedding_dim: int = 1536
    hidden_size: int = 1024
    action_dim: int = 3
    action_horizon: int = 8
    add_pos_embed: bool = True
    max_seq_len: int = 8
    ego_status_encoder_type: Literal['mlp', 'attention'] = 'mlp'

    sampling_method: Literal['flow', 'ddpm', 'ddim'] = 'ddim'
    num_inference_steps: int = 5
    model_dtype: str = "float16"
    grpo: bool = False
    vlm_size: str = 'large'
    planner_dim: int = 384
    use_expert_features: bool = False
    expert_feature_source: Literal['none', 'dummy', 'chunk', 'disk', 'online', 'cache', 'real'] = 'none'
    expert_adapter_dim: int = 768
    allow_random_init: bool = True
    allow_dummy_expert_cache: bool = False
    use_jepa: bool = True
    use_vggt: bool = True
    jepa_dim: int = 1024
    vggt_dim: int = 2048
    num_jepa_tokens: int = 12
    num_vggt_tokens: int = 12
    use_teacher_context_tokens: bool = True
    use_student_latent_adapters: bool = True
    use_branch_weighted_mean: bool = True
    use_expert_type_embedding: bool = True
    use_expert_gates: bool = True
    expert_dropout: float = 0.10
    expert_stream_dropout: float = 0.0
    expert_context_scale: float = 1.0
    use_horizon_expert_residual: bool = False
    expert_horizon_residual_scale: float = 0.0
    expert_fusion_mode: str = "concat_context"
    diffusion_loss_weight: float = 1.0
    expert_alignment_weight: float = 0.0
    jepa_alignment_weight: float = 0.03
    vggt_alignment_weight: float = 0.05
    alignment_loss_type: str = "normalized_mse"
    jepa_gate_init: float = 0.05
    vggt_gate_init: float = 0.05
    branch_init_vlm: float = 0.90
    branch_init_jepa: float = 0.05
    branch_init_vggt: float = 0.05
    allow_future_targets_in_inference: bool = False

    use_last_rd: bool = False
    last_rd_stage: Literal["disabled", "stage1_5", "progressive_sft"] = "disabled"
    use_future_jepa_prediction: bool = True
    use_vggt_geometry_tokens: bool = True
    use_ego_trajectory_tokens: bool = True
    use_risk_tokens: bool = True
    use_scene_aware_expert_gate: bool = True
    use_timestep_aware_expert_gate: bool = True
    last_rd_latent_dim: int = 384
    num_dynamic_tokens: int = 12
    num_geometry_tokens: int = 12
    num_ego_tokens: int = 8
    num_risk_tokens: int = 8
    require_vggt_geometry: bool = False
    allow_patch_geometry_fallback: bool = True
    future_jepa_loss_weight: float = 0.0
    vggt_geometry_loss_weight: float = 0.0
    coarse_traj_loss_weight: float = 0.0
    coarse_heading_loss_weight: float = 0.0
    risk_loss_weight: float = 0.0
    policy_kd_loss_weight: float = 0.0
    future_jepa_loss_floor: float = 0.0
    vggt_geometry_loss_floor: float = 0.0
    coarse_traj_loss_floor: float = 0.0
    risk_loss_floor: float = 0.0
    last_rd_context_scale: float = 1.0
    last_rd_horizon_condition_scale: float = 1.0
    last_rd_token_dropout: float = 0.0
    last_rd_group_dropout: float = 0.0
    reference_a0_checkpoint: Optional[str] = None
    policy_kd_mode: Literal["none", "noise", "x0"] = "none"
    current_train_epoch: int = 0
    total_train_epochs: int = 200

    use_last_vla: bool = False
    last_vla_stage: Literal[
        "disabled",
        "cot_alignment",
        "progressive_sft_bottleneck",
        "progressive_sft_decoupled",
        "teacher_traj_sft",
    ] = "disabled"
    last_vla_cot_num_tokens: int = 32
    last_vla_cot_num_steps: int = 4
    last_vla_condition_mode: str = "decoupled_cot_residual"
    last_vla_raw_vlm_context_to_dit: bool = True
    last_vla_cot_bottleneck_mode: bool = False
    last_vla_vlm_context_dropout_start: float = 0.0
    last_vla_vlm_context_dropout_end: float = 0.7
    last_vla_use_scene_step: bool = True
    last_vla_use_parallel_geometry_dynamic: bool = True
    last_vla_fusion_tokens: int = 192
    last_vla_dynamic_uses_geometry_memory: bool = True
    last_vla_use_geometry_step: bool = True
    last_vla_use_dynamic_step: bool = True
    last_vla_use_ego_step: bool = True
    last_vla_use_action_refine_step: bool = True
    last_vla_use_action_conditioned_dynamics: bool = True
    last_vla_use_risk_head: bool = True
    last_vla_require_full_geometry: bool = False
    last_vla_allow_patch_geometry_fallback: bool = False
    last_vla_geometry_teacher_dim: int = 512
    last_vla_geometry_grid_rows: int = 3
    last_vla_geometry_grid_cols: int = 4
    last_vla_use_residual_diffusion: bool = False
    last_vla_residual_detach_coarse: bool = True
    last_vla_residual_anchor_source: Literal["vlm_text_traj", "none"] = "vlm_text_traj"
    last_vla_require_residual_anchor: bool = True
    last_vla_coarse_prior_clip: float = 1.0
    last_vla_residual_alpha_start: float = 0.0
    last_vla_residual_alpha_end: float = 1.0
    last_vla_residual_alpha_warmup_epochs: int = 80
    last_vla_cot_condition_zero_init: bool = True
    last_vla_cot_condition_dropout: float = 0.0
    last_vla_cot_condition_layers: str = "all"
    last_vla_cot_condition_scale_init: float = 1.0
    last_vla_cot_condition_trainable_scale: bool = True
    last_vla_context_mean_mode: str = "raw_vlm_plus_zero_init_cot_residual"
    last_vla_horizon_condition_mode: str = "raw_vlm_plus_zero_init_cot_residual"
    last_vla_aux_decay_epochs: int = 160
    last_vla_teacher_traj_mode: Literal["none", "gt", "teacher_if_better", "mix"] = "none"
    last_vla_teacher_traj_mix_start: float = 0.0
    last_vla_teacher_traj_mix_end: float = 1.0
    last_vla_teacher_score_margin: float = 0.0
    last_vla_geometry_loss_weight: float = 0.0
    last_vla_dynamic_loss_weight: float = 0.0
    last_vla_coarse_loss_weight: float = 0.0
    last_vla_heading_loss_weight: float = 0.0
    last_vla_progress_loss_weight: float = 0.0
    last_vla_risk_loss_weight: float = 0.0
    last_vla_cot_consistency_loss_weight: float = 0.0
    last_vla_geometry_loss_floor: float = 0.0
    last_vla_dynamic_loss_floor: float = 0.0
    last_vla_coarse_loss_floor: float = 0.0
    last_vla_progress_loss_floor: float = 0.0
    last_vla_train_vlm_lora: bool = False
    last_vla_vlm_lora_preset: str = "attention_mlp"
    last_vla_vlm_lora_scope: str = "llm"
    last_vla_vlm_lora_r: int = 32
    last_vla_vlm_lora_alpha: int = 64
    last_vla_vlm_lora_dropout: float = 0.05
    last_vla_vlm_lora_bias: str = "none"
    last_vla_vlm_lora_target_modules: str = ""
    last_vla_vlm_lora_use_rslora: bool = True
    last_vla_vlm_lora_use_dora: bool = False
    last_vla_vlm_lora_init: str = "default"
    last_vla_vlm_lora_vision_last_n: int = 0
    last_vla_lora_allow_all_linear_global: bool = False

    use_two_expert_slots: bool = False
    two_expert_cache_mode: bool = True
    two_expert_slot_mode: Literal["vlm_soft_slots"] = "vlm_soft_slots"
    two_expert_condition_mode: Literal["horizon_hmef_lite"] = "horizon_hmef_lite"
    two_expert_dit_condition_mode: Literal["horizon_hmef_lite"] = "horizon_hmef_lite"
    two_expert_planner_dim: int = 384
    two_expert_use_raw_vlm_base: bool = True
    two_expert_zero_init_deltas: bool = True
    two_expert_dyn_loss_floor: float = 0.0
    two_expert_geo_loss_floor: float = 0.0
    two_expert_num_dyn_groups: int = 3
    two_expert_dyn_tokens_per_group: int = 12
    two_expert_num_geo_tokens: int = 12

    tune_projector: bool = True
    tune_diffusion_model: bool = True

    flow_cfg: FlowConfig = field(default_factory=FlowConfig)
    ddpm_cfg: DDPMConfig = field(default_factory=DDPMConfig)
    ddim_cfg: DDIMConfig = field(default_factory=DDIMConfig)
    grpo_cfg: GRPOConfig = field(default_factory=GRPOConfig)
    offline_rl_cfg: OfflineRLConfig = field(default_factory=OfflineRLConfig)


class ReCogDriveDiffusionPlanner(nn.Module):
    config_class = ReCogDriveDiffusionPlannerConfig
    expert_target_keys = (
        "jepa_target_tokens",
        "vggt_target_tokens",
        "vggt_geometry_target_tokens",
        "vggt_depth_target_tokens",
        "vggt_pointmap_target_tokens",
        "jepa_dynamic_teacher_tokens",
        "vggt_feature23_tokens",
    )

    def __init__(self, config: ReCogDriveDiffusionPlannerConfig):
        super().__init__()
        self.config = config
        
        self.model = LightningDiT(**config.diffusion_model_cfg)

        self.his_traj_encoder = Mlp(
            in_features=12,
            hidden_features=config.hidden_size,
            out_features=config.input_embedding_dim,
            norm_layer=nn.LayerNorm
        )

        if config.ego_status_encoder_type == 'attention':
            self.ego_status_encoder = StateAttentionEncoder(
                state_dim=8,
                embed_dim=config.input_embedding_dim,
                num_kinematic_states=4 
            )
        else: 
            self.ego_status_encoder = Mlp(
                in_features=8,
                hidden_features=config.hidden_size,
                out_features=config.input_embedding_dim,
                norm_layer=nn.LayerNorm
            )

        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_size=config.input_embedding_dim,
        )
        if config.vlm_size == "large":
            self.feature_encoder = nn.Linear(3584, config.input_embedding_dim)
        else:
            self.feature_encoder = nn.Linear(1536, config.input_embedding_dim)

        if config.alignment_loss_type not in {"normalized_mse", "mse", "cosine"}:
            raise ValueError("alignment_loss_type must be one of 'normalized_mse', 'mse', or 'cosine'.")
        if config.diffusion_loss_weight < 0.0:
            raise ValueError("diffusion_loss_weight must be non-negative.")
        if config.use_last_rd and config.last_rd_stage == "disabled":
            raise ValueError("use_last_rd=True requires last_rd_stage='stage1_5' or 'progressive_sft'.")
        if not config.use_last_rd and config.last_rd_stage != "disabled":
            raise ValueError("last_rd_stage must be 'disabled' when use_last_rd=False.")
        if config.use_last_vla and config.use_last_rd:
            raise ValueError("use_last_vla and use_last_rd are mutually exclusive.")
        if config.use_two_expert_slots:
            if config.use_last_vla or config.use_last_rd or config.use_expert_features:
                raise ValueError(
                    "two_expert_slot is mutually exclusive with old Last-VLA, Last-RD, and A4 direct expert paths."
                )
            if config.two_expert_slot_mode != "vlm_soft_slots":
                raise ValueError("two_expert_slot_mode must be 'vlm_soft_slots'.")
            if config.two_expert_condition_mode != "horizon_hmef_lite":
                raise ValueError("two_expert_condition_mode must be 'horizon_hmef_lite'.")
            if config.two_expert_dit_condition_mode != "horizon_hmef_lite":
                raise ValueError("two_expert_dit_condition_mode must be 'horizon_hmef_lite'.")
            if config.last_vla_use_residual_diffusion:
                raise ValueError("two_expert_slot forbids residual diffusion.")
            if config.last_vla_teacher_traj_mode != "none":
                raise ValueError("two_expert_slot Stage2 target must remain GT normalized trajectory.")
            if not config.two_expert_use_raw_vlm_base:
                raise ValueError("two_expert_slot must preserve raw VLM context for DiT.")
        if config.use_last_vla and config.last_vla_stage == "disabled":
            raise ValueError("use_last_vla=True requires last_vla_stage to be non-disabled.")
        if not config.use_last_vla and config.last_vla_stage != "disabled":
            raise ValueError("last_vla_stage must be 'disabled' when use_last_vla=False.")
        if (
            config.use_last_vla
            and (
                config.last_vla_condition_mode != "decoupled_cot_residual"
                or config.last_vla_cot_bottleneck_mode
                or not config.last_vla_raw_vlm_context_to_dit
            )
        ):
            raise ValueError(
                "Last-VLA v2 only supports decoupled_cot_residual with "
                "last_vla_cot_bottleneck_mode=False and last_vla_raw_vlm_context_to_dit=True. "
                "Hard bottleneck and summary replacement were removed."
            )
        if config.last_vla_require_full_geometry and config.last_vla_allow_patch_geometry_fallback:
            raise ValueError("last_vla_require_full_geometry=True is incompatible with patch fallback.")
        if config.use_last_vla and config.last_vla_geometry_teacher_dim <= 0:
            raise ValueError("last_vla_geometry_teacher_dim must be positive when use_last_vla=True.")
        if config.use_last_vla and config.num_geometry_tokens != int(config.last_vla_geometry_grid_rows) * int(config.last_vla_geometry_grid_cols):
            if (int(config.last_vla_geometry_grid_rows), int(config.last_vla_geometry_grid_cols)) == (3, 4):
                pass
            else:
                raise ValueError(
                    "num_geometry_tokens must equal last_vla_geometry_grid_rows * last_vla_geometry_grid_cols "
                    f"({config.num_geometry_tokens} != {config.last_vla_geometry_grid_rows} * {config.last_vla_geometry_grid_cols})."
                )
        if config.policy_kd_mode not in {"none", "noise", "x0"}:
            raise ValueError("policy_kd_mode must be one of 'none', 'noise', or 'x0'.")
        for weight_name in (
            "expert_alignment_weight",
            "jepa_alignment_weight",
            "vggt_alignment_weight",
            "future_jepa_loss_weight",
            "vggt_geometry_loss_weight",
            "coarse_traj_loss_weight",
            "coarse_heading_loss_weight",
            "risk_loss_weight",
            "policy_kd_loss_weight",
            "last_vla_geometry_loss_weight",
            "last_vla_dynamic_loss_weight",
            "last_vla_coarse_loss_weight",
            "last_vla_heading_loss_weight",
            "last_vla_progress_loss_weight",
            "last_vla_risk_loss_weight",
            "last_vla_cot_consistency_loss_weight",
            "two_expert_dyn_loss_floor",
            "two_expert_geo_loss_floor",
        ):
            if getattr(config, weight_name) < 0.0:
                raise ValueError(f"{weight_name} must be non-negative.")

        if config.use_two_expert_slots:
            vlm_dim = 3584 if config.vlm_size == "large" else 1536
            planner_dim = config.input_embedding_dim
            self.two_expert_dyn_proj = nn.Sequential(
                nn.LayerNorm(vlm_dim),
                nn.Linear(vlm_dim, planner_dim),
                nn.GELU(),
                nn.Linear(planner_dim, planner_dim),
                nn.LayerNorm(planner_dim),
            )
            self.two_expert_geo_proj = nn.Sequential(
                nn.LayerNorm(vlm_dim),
                nn.Linear(vlm_dim, planner_dim),
                nn.GELU(),
                nn.Linear(planner_dim, planner_dim),
                nn.LayerNorm(planner_dim),
            )
            self.two_expert_horizon_queries = nn.Parameter(torch.randn(config.action_horizon, planner_dim) * 0.02)
            heads = 8 if planner_dim % 8 == 0 else 1
            self.two_expert_dyn_horizon_attn = nn.MultiheadAttention(planner_dim, heads, batch_first=True)
            self.two_expert_geo_horizon_attn = nn.MultiheadAttention(planner_dim, heads, batch_first=True)
            self.two_expert_dyn_delta_proj = nn.Linear(planner_dim, planner_dim)
            self.two_expert_geo_delta_proj = nn.Linear(planner_dim, planner_dim)
            if config.two_expert_zero_init_deltas:
                nn.init.constant_(self.two_expert_dyn_delta_proj.weight, 0.0)
                nn.init.constant_(self.two_expert_dyn_delta_proj.bias, 0.0)
                nn.init.constant_(self.two_expert_geo_delta_proj.weight, 0.0)
                nn.init.constant_(self.two_expert_geo_delta_proj.bias, 0.0)

        elif config.use_expert_features:
            if not config.use_jepa and not config.use_vggt:
                raise ValueError("use_expert_features=True requires use_jepa=True and/or use_vggt=True.")
            if config.expert_fusion_mode != "concat_context":
                raise ValueError(
                    f"Unsupported expert_fusion_mode={config.expert_fusion_mode!r}. "
                    "Only 'concat_context' is implemented."
                )
            if config.use_jepa and config.jepa_dim <= 0:
                raise ValueError("use_expert_features=True with use_jepa=True requires positive jepa_dim.")
            if config.use_vggt and config.vggt_dim <= 0:
                raise ValueError("use_expert_features=True with use_vggt=True requires positive vggt_dim.")
            if not config.use_teacher_context_tokens and not config.use_student_latent_adapters:
                raise ValueError("Expert mode requires teacher context tokens and/or student latent adapters.")
            if not 0.0 <= config.expert_dropout < 1.0:
                raise ValueError("expert_dropout must be in [0.0, 1.0).")
            if not 0.0 <= config.expert_stream_dropout < 1.0:
                raise ValueError("expert_stream_dropout must be in [0.0, 1.0).")
            if config.expert_context_scale < 0.0:
                raise ValueError("expert_context_scale must be non-negative.")
            if config.expert_horizon_residual_scale < 0.0:
                raise ValueError("expert_horizon_residual_scale must be non-negative.")

            planner_dim = config.input_embedding_dim
            if config.use_jepa:
                self.jepa_projector = TeacherTokenProjector(
                    config.jepa_dim,
                    expert_adapter_dim=config.expert_adapter_dim,
                    planner_dim=planner_dim,
                )
                self.jepa_adapter = ExpertAdapter768(
                    planner_dim=planner_dim,
                    expert_adapter_dim=config.expert_adapter_dim,
                    num_tokens=config.num_jepa_tokens,
                )
                self.jepa_alignment_head = AlignmentHead(config.expert_adapter_dim, config.jepa_dim)
                if config.use_horizon_expert_residual:
                    self.jepa_horizon_conditioner = HorizonAwareExpertConditioner(
                        planner_dim=planner_dim,
                        action_horizon=config.action_horizon,
                    )
            if config.use_vggt:
                self.vggt_projector = TeacherTokenProjector(
                    config.vggt_dim,
                    expert_adapter_dim=config.expert_adapter_dim,
                    planner_dim=planner_dim,
                )
                self.vggt_adapter = ExpertAdapter768(
                    planner_dim=planner_dim,
                    expert_adapter_dim=config.expert_adapter_dim,
                    num_tokens=config.num_vggt_tokens,
                )
                self.vggt_alignment_head = AlignmentHead(config.expert_adapter_dim, config.vggt_dim)
                if config.use_horizon_expert_residual:
                    self.vggt_horizon_conditioner = HorizonAwareExpertConditioner(
                        planner_dim=planner_dim,
                        action_horizon=config.action_horizon,
                    )

            if config.use_expert_type_embedding:
                self.jepa_type_embedding = nn.Parameter(torch.empty(1, 1, planner_dim))
                self.vggt_type_embedding = nn.Parameter(torch.empty(1, 1, planner_dim))
                self.z_jepa_type_embedding = nn.Parameter(torch.empty(1, 1, planner_dim))
                self.z_vggt_type_embedding = nn.Parameter(torch.empty(1, 1, planner_dim))
                nn.init.normal_(self.jepa_type_embedding, mean=0.0, std=0.02)
                nn.init.normal_(self.vggt_type_embedding, mean=0.0, std=0.02)
                nn.init.normal_(self.z_jepa_type_embedding, mean=0.0, std=0.02)
                nn.init.normal_(self.z_vggt_type_embedding, mean=0.0, std=0.02)

            self.jepa_gate = nn.Parameter(torch.tensor(init_logit_from_prob(config.jepa_gate_init), dtype=torch.float32))
            self.vggt_gate = nn.Parameter(torch.tensor(init_logit_from_prob(config.vggt_gate_init), dtype=torch.float32))
            self.branch_logits = nn.Parameter(branch_logits_from_probs([
                config.branch_init_vlm,
                config.branch_init_jepa,
                config.branch_init_vggt,
            ]))

        if config.use_last_vla:
            from .last_vla_cot_planning import LastVLACoTConfig, LastVLACoTTransformer

            self.last_vla_cot = LastVLACoTTransformer(
                LastVLACoTConfig(
                    planner_dim=config.input_embedding_dim,
                    vlm_dim=config.input_embedding_dim,
                    jepa_dim=config.jepa_dim,
                    vggt_dim=config.vggt_dim,
                    hidden_dim=config.hidden_size,
                    action_dim=config.action_dim,
                    action_horizon=config.action_horizon,
                    cot_num_tokens=config.last_vla_cot_num_tokens,
                    cot_num_steps=config.last_vla_cot_num_steps,
                    condition_mode=config.last_vla_condition_mode,
                    use_scene_step=config.last_vla_use_scene_step,
                    use_parallel_geometry_dynamic=config.last_vla_use_parallel_geometry_dynamic,
                    fusion_tokens=config.last_vla_fusion_tokens,
                    dynamic_uses_geometry_memory=config.last_vla_dynamic_uses_geometry_memory,
                    geometry_tokens=config.num_geometry_tokens,
                    dynamic_tokens=config.num_dynamic_tokens,
                    ego_tokens=config.num_ego_tokens,
                    risk_tokens=config.num_risk_tokens,
                    use_geometry_step=config.last_vla_use_geometry_step,
                    use_dynamic_step=config.last_vla_use_dynamic_step,
                    use_ego_step=config.last_vla_use_ego_step,
                    use_action_refine_step=config.last_vla_use_action_refine_step,
                    use_action_conditioned_dynamics=config.last_vla_use_action_conditioned_dynamics,
                    use_cot_risk_head=config.last_vla_use_risk_head,
                    raw_vlm_context_to_dit=config.last_vla_raw_vlm_context_to_dit,
                    cot_bottleneck_mode=config.last_vla_cot_bottleneck_mode,
                    vlm_context_dropout_start=config.last_vla_vlm_context_dropout_start,
                    vlm_context_dropout_end=config.last_vla_vlm_context_dropout_end,
                    use_residual_diffusion=False,
                    residual_detach_coarse_for_diffusion=config.last_vla_residual_detach_coarse,
                    coarse_prior_clip=config.last_vla_coarse_prior_clip,
                    require_full_geometry=config.last_vla_require_full_geometry,
                    allow_patch_geometry_fallback=config.last_vla_allow_patch_geometry_fallback,
                    geometry_teacher_dim=config.last_vla_geometry_teacher_dim,
                    geometry_grid_rows=config.last_vla_geometry_grid_rows,
                    geometry_grid_cols=config.last_vla_geometry_grid_cols,
                    teacher_traj_mode=config.last_vla_teacher_traj_mode,
                    teacher_traj_mix_start=config.last_vla_teacher_traj_mix_start,
                    teacher_traj_mix_end=config.last_vla_teacher_traj_mix_end,
                    cot_condition_zero_init=config.last_vla_cot_condition_zero_init,
                    cot_condition_dropout=config.last_vla_cot_condition_dropout,
                    cot_condition_scale_init=config.last_vla_cot_condition_scale_init,
                    cot_condition_trainable_scale=config.last_vla_cot_condition_trainable_scale,
                )
            )
            self.last_vla_context_mean_cot_proj = nn.Linear(config.input_embedding_dim, config.input_embedding_dim)
            self.last_vla_horizon_cot_queries = nn.Parameter(
                torch.randn(config.action_horizon, config.input_embedding_dim) * 0.02
            )
            cot_heads = 8 if config.input_embedding_dim % 8 == 0 else 1
            self.last_vla_horizon_cot_attn = nn.MultiheadAttention(
                config.input_embedding_dim,
                cot_heads,
                batch_first=True,
            )
            self.last_vla_horizon_cot_proj = nn.Linear(config.input_embedding_dim, config.input_embedding_dim)
            if config.last_vla_cot_condition_zero_init:
                nn.init.constant_(self.last_vla_context_mean_cot_proj.weight, 0.0)
                nn.init.constant_(self.last_vla_context_mean_cot_proj.bias, 0.0)
                nn.init.constant_(self.last_vla_horizon_cot_proj.weight, 0.0)
                nn.init.constant_(self.last_vla_horizon_cot_proj.bias, 0.0)
            scale = torch.tensor(float(config.last_vla_cot_condition_scale_init), dtype=torch.float32)
            if config.last_vla_cot_condition_trainable_scale:
                self.last_vla_cot_condition_scale = nn.Parameter(scale)
            else:
                self.register_buffer("last_vla_cot_condition_scale", scale)

        if config.use_last_rd:
            from .latent_spatiotemporal_planning import LastRDConfig, LatentSpatioTemporalReasoner

            self.last_rd = LatentSpatioTemporalReasoner(
                LastRDConfig(
                    planner_dim=config.input_embedding_dim,
                    jepa_dim=config.jepa_dim,
                    vggt_dim=config.vggt_dim,
                    latent_dim=config.last_rd_latent_dim,
                    hidden_dim=config.hidden_size,
                    action_dim=config.action_dim,
                    action_horizon=config.action_horizon,
                    num_dynamic_tokens=config.num_dynamic_tokens,
                    num_geometry_tokens=config.num_geometry_tokens,
                    num_ego_tokens=config.num_ego_tokens,
                    num_risk_tokens=config.num_risk_tokens,
                    use_future_jepa_prediction=config.use_future_jepa_prediction,
                    use_vggt_geometry_tokens=config.use_vggt_geometry_tokens,
                    use_ego_trajectory_tokens=config.use_ego_trajectory_tokens,
                    use_risk_tokens=config.use_risk_tokens,
                    use_scene_aware_gate=config.use_scene_aware_expert_gate,
                    use_timestep_aware_gate=config.use_timestep_aware_expert_gate,
                    require_vggt_geometry=config.require_vggt_geometry,
                    allow_patch_geometry_fallback=config.allow_patch_geometry_fallback,
                )
            )

        self.reference_a0_policy: Optional[ReCogDriveDiffusionPlanner] = None
        if config.policy_kd_loss_weight > 0.0 and config.policy_kd_mode != "none":
            if not config.reference_a0_checkpoint:
                raise ValueError(
                    "policy_kd_loss_weight > 0 and policy_kd_mode != 'none' requires "
                    "reference_a0_checkpoint. Set policy_kd_loss_weight=0.0 or provide "
                    "the A0-official-aligned reference checkpoint."
                )
            reference_cfg = copy.deepcopy(config)
            reference_cfg.use_last_rd = False
            reference_cfg.last_rd_stage = "disabled"
            reference_cfg.use_last_vla = False
            reference_cfg.last_vla_stage = "disabled"
            reference_cfg.use_expert_features = False
            reference_cfg.policy_kd_loss_weight = 0.0
            reference_cfg.reference_a0_checkpoint = None
            reference_cfg.policy_kd_mode = "none"
            self.reference_a0_policy = ReCogDriveDiffusionPlanner(reference_cfg)
            self.reference_a0_policy._safe_load_reference_policy(config.reference_a0_checkpoint)
            self.reference_a0_policy.eval()
            for parameter in self.reference_a0_policy.parameters():
                parameter.requires_grad = False
            
        self.fusion_projector = nn.Linear(config.input_embedding_dim * 3, config.input_embedding_dim)

        output_dim = 2 * config.action_dim if (
            config.sampling_method == 'flow' and config.flow_cfg.mean_variance_net
        ) else config.action_dim
        
        self.action_decoder = Mlp(
            in_features=self.model.output_dim,
            hidden_features=config.hidden_size,
            out_features=output_dim,
            norm_layer=nn.LayerNorm
        )
        
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, config.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        
        if self.config.sampling_method == 'flow':
            self._init_flow_sampler(config.flow_cfg)
        elif self.config.sampling_method == 'ddpm':
            self._init_ddpm_sampler(config.ddpm_cfg)
        elif self.config.sampling_method == 'ddim':
            self._init_ddim_sampler(config.ddim_cfg)

        self.offline_rl_cfg = config.offline_rl_cfg
        self._validate_offline_rl_config(self.offline_rl_cfg)
        if config.grpo:
            self._init_grpo(config.grpo_cfg)
        elif self.offline_rl_cfg.enabled:
            self._init_offline_rl(self.offline_rl_cfg, config.grpo_cfg)

    def _init_flow_sampler(self, cfg: FlowConfig):
        """Initializes components required for Flow Matching."""
        self.beta_dist = Beta(cfg.noise_beta_alpha, cfg.noise_beta_beta)
        self.num_timestep_buckets = cfg.num_timestep_buckets

    def _init_ddpm_sampler(self, cfg: DDPMConfig):
        """Initializes buffers required for DDPM, using original naming."""
        ddpm_betas = self.cosine_beta_schedule(cfg.num_train_timesteps)
        self.register_buffer('ddpm_betas', ddpm_betas)

        ddpm_alphas = 1.0 - ddpm_betas
        self.register_buffer('ddpm_alphas', ddpm_alphas)

        ddpm_alphas_cumprod = torch.cumprod(ddpm_alphas, dim=0)
        self.register_buffer('ddpm_alphas_cumprod', ddpm_alphas_cumprod)

        ddpm_alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), ddpm_alphas_cumprod[:-1]])
        self.register_buffer('ddpm_alphas_cumprod_prev', ddpm_alphas_cumprod_prev)

        self.register_buffer('ddpm_sqrt_alphas_cumprod', torch.sqrt(ddpm_alphas_cumprod))
        self.register_buffer('ddpm_sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - ddpm_alphas_cumprod))
        self.register_buffer('ddpm_sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / ddpm_alphas_cumprod))
        self.register_buffer('ddpm_sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / ddpm_alphas_cumprod - 1.0))

        ddpm_var = ddpm_betas * (1.0 - ddpm_alphas_cumprod_prev) / (1.0 - ddpm_alphas_cumprod)
        self.register_buffer('ddpm_var', ddpm_var)
        self.register_buffer('ddpm_logvar_clipped', torch.log(ddpm_var.clamp(min=1e-20)))
        
        self.register_buffer('ddpm_mu_coef1', ddpm_betas * torch.sqrt(ddpm_alphas_cumprod_prev) / (1.0 - ddpm_alphas_cumprod))
        self.register_buffer('ddpm_mu_coef2', (1.0 - ddpm_alphas_cumprod_prev) * torch.sqrt(ddpm_alphas) / (1.0 - ddpm_alphas_cumprod))

    def _init_ddim_sampler(self, cfg: DDIMConfig):
        """Initializes buffers required for DDIM sampling, using original naming."""
        self._init_ddpm_sampler(DDPMConfig(num_train_timesteps=cfg.num_train_timesteps))

        self.eta = EtaFixed(base_eta=1.0).to(self.device)
        for param in self.eta.parameters():
            param.requires_grad = False
        ddim_steps = self.config.num_inference_steps
        self.ddim_steps = ddim_steps
        self.ft_denoising_steps = ddim_steps
        ddim_eta = cfg.ddim_eta 

        self.ddpm_num_train_timesteps = cfg.num_train_timesteps
        step_ratio = self.ddpm_num_train_timesteps // ddim_steps
        ddim_t = torch.arange(0, ddim_steps) * step_ratio
        self.register_buffer('ddim_t_schedule', ddim_t.long()) # Die originalen Zeitpunkte

        ddim_alphas = self.ddpm_alphas_cumprod[self.ddim_t_schedule].clone().to(torch.float32)
        ddim_alphas_prev = torch.cat([
            torch.tensor([1.0], dtype=torch.float32),
            self.ddpm_alphas_cumprod[self.ddim_t_schedule[:-1]]
        ])
        ddim_sqrt_one_minus_alphas = (1.0 - ddim_alphas) ** 0.5

        ddim_sigmas = ddim_eta * (
            (1 - ddim_alphas_prev) / (1 - ddim_alphas) *
            (1 - ddim_alphas / ddim_alphas_prev)
        )**0.5

        # Flip all for sampling order (T -> 0)
        def flip_buffer(name, tensor):
            self.register_buffer(name, torch.flip(tensor, [0]))

        flip_buffer('ddim_t', self.ddim_t_schedule)
        flip_buffer('ddim_alphas', ddim_alphas)
        flip_buffer('ddim_alphas_sqrt', torch.sqrt(ddim_alphas))
        flip_buffer('ddim_alphas_prev', ddim_alphas_prev)
        flip_buffer('ddim_sqrt_one_minus_alphas', ddim_sqrt_one_minus_alphas)
        flip_buffer('ddim_sigmas', ddim_sigmas)

    @staticmethod
    def _validate_offline_rl_config(cfg: OfflineRLConfig) -> None:
        if int(cfg.elite_top_m) <= 0:
            raise ValueError("offline_rl_cfg.elite_top_m must be positive.")
        if int(cfg.elite_min_candidates) <= 0:
            raise ValueError("offline_rl_cfg.elite_min_candidates must be positive.")
        if int(cfg.online_policy_samples) < 0:
            raise ValueError("offline_rl_cfg.online_policy_samples must be non-negative.")
        if str(cfg.missing_submetric_policy) not in {"error", "unsafe_zero", "warn_default"}:
            raise ValueError("offline_rl_cfg.missing_submetric_policy must be error, unsafe_zero, or warn_default.")
        if int(cfg.pdm_batch_chunk_size) < 0:
            raise ValueError("offline_rl_cfg.pdm_batch_chunk_size must be non-negative.")
        if int(cfg.pdm_shadow_max_samples) < 0:
            raise ValueError("offline_rl_cfg.pdm_shadow_max_samples must be non-negative.")
        if float(cfg.pdm_shadow_max_abs_diff) < 0.0:
            raise ValueError("offline_rl_cfg.pdm_shadow_max_abs_diff must be non-negative.")
        if int(cfg.elite_buffer_preload_max_records) < 0:
            raise ValueError("offline_rl_cfg.elite_buffer_preload_max_records must be non-negative.")
        if int(cfg.elite_buffer_version) < 2:
            raise ValueError("offline_rl_cfg.elite_buffer_version must be >= 2.")
        if float(cfg.advantage_temperature) <= 0.0:
            raise ValueError("offline_rl_cfg.advantage_temperature must be positive.")
        if not (0.0 <= float(cfg.weight_min) <= float(cfg.weight_max)):
            raise ValueError("offline_rl_cfg.weight_max must be >= weight_min >= 0.")
        if float(cfg.fallback_invalid_candidate_weight) < 0.0:
            raise ValueError("offline_rl_cfg.fallback_invalid_candidate_weight must be non-negative.")
        unknown_submetrics = sorted(set(cfg.required_reward_submetrics).difference(REQUIRED_COMPONENT_KEYS))
        if unknown_submetrics:
            raise ValueError(
                "offline_rl_cfg.required_reward_submetrics must be a subset of REQUIRED_COMPONENT_KEYS; "
                f"unknown={unknown_submetrics}"
            )
        if not (0.0 < float(cfg.expectile_tau) < 1.0):
            raise ValueError("offline_rl_cfg.expectile_tau must be in (0, 1).")
        if int(cfg.expectile_iters) <= 0:
            raise ValueError("offline_rl_cfg.expectile_iters must be positive.")
        if not (0.0 < float(cfg.top_mean_frac) <= 1.0):
            raise ValueError("offline_rl_cfg.top_mean_frac must be in (0, 1].")
        if float(cfg.advantage_clip_min) > float(cfg.advantage_clip_max):
            raise ValueError("offline_rl_cfg.advantage_clip_min must be <= advantage_clip_max.")
        if str(cfg.target_filter_mode) not in {"none", "topk", "positive_advantage", "topk_positive"}:
            raise ValueError("offline_rl_cfg.target_filter_mode must be none, topk, positive_advantage, or topk_positive.")
        if int(cfg.target_top_k) <= 0:
            raise ValueError("offline_rl_cfg.target_top_k must be positive.")
        if float(cfg.target_min_advantage) < 0.0:
            raise ValueError("offline_rl_cfg.target_min_advantage must be non-negative.")
        for name in ("ddc_guard_mode", "ttc_guard_mode"):
            if str(getattr(cfg, name)) not in {"relative_or_absolute", "absolute", "relative"}:
                raise ValueError(f"offline_rl_cfg.{name} must be relative_or_absolute, absolute, or relative.")
        for name in ("awac_timestep_sampling", "preference_dpo_timestep_sampling"):
            if str(getattr(cfg, name)) not in {"uniform", "ddim", "low_noise", "mid_noise"}:
                raise ValueError(f"offline_rl_cfg.{name} must be uniform, ddim, low_noise, or mid_noise.")
        if not (0.0 < float(cfg.low_noise_timestep_frac) <= 1.0):
            raise ValueError("offline_rl_cfg.low_noise_timestep_frac must be in (0, 1].")
        if not (0.0 <= float(cfg.mid_noise_timestep_low_frac) < float(cfg.mid_noise_timestep_high_frac) <= 1.0):
            raise ValueError(
                "offline_rl_cfg.mid_noise_timestep_low_frac/high_frac must satisfy 0 <= low < high <= 1."
            )
        if str(cfg.target_blend_mode) not in {"none", "towards_behavior_anchor"}:
            raise ValueError("offline_rl_cfg.target_blend_mode must be none or towards_behavior_anchor.")
        if not (0.0 <= float(cfg.target_blend_alpha) <= 1.0):
            raise ValueError("offline_rl_cfg.target_blend_alpha must be in [0, 1].")
        if str(cfg.target_blend_schedule) not in {"constant", "linear_warmup"}:
            raise ValueError("offline_rl_cfg.target_blend_schedule must be constant or linear_warmup.")
        if not (0.0 <= float(cfg.target_blend_alpha_start) <= 1.0):
            raise ValueError("offline_rl_cfg.target_blend_alpha_start must be in [0, 1].")
        if float(cfg.component_advantage_clip_min) > float(cfg.component_advantage_clip_max):
            raise ValueError("offline_rl_cfg.component_advantage_clip_min must be <= component_advantage_clip_max.")
        if not (0.0 <= float(cfg.source_balance_min_factor) <= float(cfg.source_balance_max_factor)):
            raise ValueError("offline_rl_cfg.source_balance_max_factor must be >= source_balance_min_factor >= 0.")
        for name in ("awac_loss_schedule", "preference_dpo_loss_schedule", "grpo_loss_schedule"):
            if str(getattr(cfg, name)) not in {"constant", "linear_warmup"}:
                raise ValueError(f"offline_rl_cfg.{name} must be constant or linear_warmup.")
        if str(cfg.bc_loss_schedule) not in {"constant", "linear"}:
            raise ValueError("offline_rl_cfg.bc_loss_schedule must be constant or linear.")
        for name in (
            "awac_loss_weight",
            "awac_loss_weight_start",
            "bc_loss_weight",
            "bc_loss_weight_start",
            "bc_loss_weight_end",
            "grpo_loss_weight",
            "grpo_loss_weight_start",
            "component_progress_weight",
            "component_safety_penalty_weight",
            "pairwise_rank_loss_weight",
            "pairwise_rank_margin",
            "pairwise_rank_min_reward_gap",
            "invalid_repulsion_loss_weight",
            "invalid_repulsion_margin",
            "preference_dpo_loss_weight",
            "preference_dpo_loss_weight_start",
            "preference_dpo_beta",
            "preference_dpo_min_reward_gap",
            "preference_dpo_gap_weight_scale",
            "preference_dpo_gap_weight_min",
            "preference_dpo_gap_weight_max",
        ):
            if float(getattr(cfg, name)) < 0.0:
                raise ValueError(f"offline_rl_cfg.{name} must be non-negative.")
        if not (0.0 <= float(cfg.preference_dpo_label_smoothing) < 0.5):
            raise ValueError("offline_rl_cfg.preference_dpo_label_smoothing must be in [0, 0.5).")
        if str(cfg.preference_dpo_pair_mode) not in {"best_vs_gt_il", "best_vs_low_valid", "best_vs_all"}:
            raise ValueError("offline_rl_cfg.preference_dpo_pair_mode must be best_vs_gt_il, best_vs_low_valid, or best_vs_all.")
        if str(cfg.preference_dpo_gap_weight_mode) not in {"none", "pair_gap", "winner_advantage"}:
            raise ValueError(
                "offline_rl_cfg.preference_dpo_gap_weight_mode must be none, pair_gap, or winner_advantage."
            )
        if float(cfg.preference_dpo_gap_weight_scale) <= 0.0:
            raise ValueError("offline_rl_cfg.preference_dpo_gap_weight_scale must be positive.")
        if float(cfg.preference_dpo_gap_weight_min) > float(cfg.preference_dpo_gap_weight_max):
            raise ValueError("offline_rl_cfg.preference_dpo_gap_weight_min must be <= gap_weight_max.")
        if int(cfg.pairwise_rank_max_pairs_per_scene) < 0:
            raise ValueError("offline_rl_cfg.pairwise_rank_max_pairs_per_scene must be non-negative.")
        if int(cfg.invalid_repulsion_max_pairs_per_scene) < 0:
            raise ValueError("offline_rl_cfg.invalid_repulsion_max_pairs_per_scene must be non-negative.")
        if int(cfg.preference_dpo_max_pairs_per_scene) < 0:
            raise ValueError("offline_rl_cfg.preference_dpo_max_pairs_per_scene must be non-negative.")
        if int(cfg.bc_loss_schedule_epochs) <= 0:
            raise ValueError("offline_rl_cfg.bc_loss_schedule_epochs must be positive.")
        if int(cfg.awac_loss_warmup_start_epoch) < 0:
            raise ValueError("offline_rl_cfg.awac_loss_warmup_start_epoch must be non-negative.")
        if int(cfg.awac_loss_warmup_epochs) <= 0:
            raise ValueError("offline_rl_cfg.awac_loss_warmup_epochs must be positive.")
        if int(cfg.preference_dpo_loss_warmup_start_epoch) < 0:
            raise ValueError("offline_rl_cfg.preference_dpo_loss_warmup_start_epoch must be non-negative.")
        if int(cfg.preference_dpo_loss_warmup_epochs) <= 0:
            raise ValueError("offline_rl_cfg.preference_dpo_loss_warmup_epochs must be positive.")
        if int(cfg.grpo_loss_warmup_start_epoch) < 0:
            raise ValueError("offline_rl_cfg.grpo_loss_warmup_start_epoch must be non-negative.")
        if int(cfg.grpo_loss_warmup_epochs) <= 0:
            raise ValueError("offline_rl_cfg.grpo_loss_warmup_epochs must be positive.")
        for name in (
            "prior_distance_weight",
            "jerk_penalty_weight",
            "ddc_min_absolute",
            "ddc_max_relative_drop",
            "ttc_min_absolute",
            "ttc_max_relative_drop",
            "max_heading_step_rad",
            "max_final_heading_delta_rad",
        ):
            if float(getattr(cfg, name)) < 0.0:
                raise ValueError(f"offline_rl_cfg.{name} must be non-negative.")

    def _init_stage3_oracle(self, cfg: GRPOConfig) -> None:
        if not hasattr(self, "metric_cache_loader"):
            self.metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
        if not hasattr(self, "simulator"):
            proposal_sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)
            self.simulator = PDMSimulator(proposal_sampling)
        if not hasattr(self, "train_scorer"):
            self.train_scorer = PDMScorer(self.simulator.proposal_sampling, cfg.scorer_config)

    @staticmethod
    def _freeze_policy(policy: "ReCogDriveDiffusionPlanner") -> None:
        policy.eval()
        for param in policy.parameters():
            param.requires_grad = False

    @staticmethod
    def _resolve_reference_policy_checkpoint(checkpoint_path: str, *, required: bool) -> Optional[Path]:
        path = Path(checkpoint_path) if checkpoint_path else None
        if path is not None and path.is_file():
            return path
        if path is not None and path.is_dir():
            candidates = []
            for suffix in ("*.ckpt", "*.pth", "*.pt", "*.safetensors"):
                candidates.extend(path.glob(suffix))
            if candidates:
                return max(candidates, key=lambda item: item.stat().st_mtime)
        if required:
            raise FileNotFoundError(
                "AWAC/IQL requires a valid reference_policy_checkpoint when IL/reference candidates "
                f"or max_gt_il baseline are enabled; got {checkpoint_path!r}."
            )
        return None

    def _init_offline_rl(self, cfg: OfflineRLConfig, stage3_cfg: GRPOConfig) -> None:
        self.offline_rl_cfg = cfg
        self._init_stage3_runtime(stage3_cfg)
        self._init_stage3_oracle(stage3_cfg)
        if not hasattr(self, "old_policy"):
            reference_required = (
                bool(cfg.require_reference_policy_checkpoint)
                and (
                    bool(cfg.online_use_old_policy)
                    or bool(cfg.keep_il_candidate)
                    or str(cfg.baseline_mode) == "max_gt_il"
                    or (
                        float(cfg.preference_dpo_loss_weight) > 0.0
                        and not bool(cfg.preference_dpo_reference_free)
                    )
                )
            )
            checkpoint_path = self._resolve_reference_policy_checkpoint(
                stage3_cfg.reference_policy_checkpoint,
                required=reference_required,
            )
            self._safe_load_reference_policy(str(checkpoint_path) if checkpoint_path is not None else stage3_cfg.reference_policy_checkpoint)
            self.old_policy = copy.deepcopy(self)
            self._freeze_policy(self.old_policy)
        self._awac_elite_record_cache: Dict[str, Dict[str, Any]] = {}
        self._awac_elite_record_cache_root = ""
        if (
            bool(cfg.preload_elite_buffer)
            and bool(str(cfg.elite_buffer_path))
            and not bool(cfg.build_candidates_online)
        ):
            self._preload_awac_elite_buffer(cfg)

    def _reset_awac_elite_record_cache_if_needed(self, buffer_root: Path) -> None:
        root_key = str(Path(buffer_root).resolve())
        if getattr(self, "_awac_elite_record_cache_root", "") != root_key:
            self._awac_elite_record_cache = {}
            self._awac_elite_record_cache_root = root_key

    def _preload_awac_elite_buffer(self, cfg: OfflineRLConfig) -> None:
        buffer_root = Path(cfg.elite_buffer_path)
        if not buffer_root.is_dir():
            raise FileNotFoundError(f"AWAC elite buffer path does not exist: {buffer_root}")
        self._reset_awac_elite_record_cache_if_needed(buffer_root)
        paths = sorted(buffer_root.glob("*.pkl.xz"))
        max_records = int(cfg.elite_buffer_preload_max_records)
        if max_records > 0:
            paths = paths[:max_records]
        if not paths:
            raise FileNotFoundError(f"AWAC elite buffer path has no *.pkl.xz records: {buffer_root}")
        cache = self._awac_elite_record_cache
        for path in paths:
            record = load_elite_record_path(path)
            cache[str(record["token"])] = record

    def _load_elite_record_cached(
        self,
        buffer_root: Path,
        token: str,
        cfg: OfflineRLConfig,
    ) -> Dict[str, Any]:
        if not (bool(cfg.cache_elite_records_in_memory) or bool(cfg.preload_elite_buffer)):
            return load_elite_record(buffer_root, token)
        self._reset_awac_elite_record_cache_if_needed(buffer_root)
        cache = self._awac_elite_record_cache
        token = str(token)
        record = cache.get(token)
        if record is None:
            record = load_elite_record(buffer_root, token)
            cache[token] = record
        elif str(record.get("token", "")) != token:
            raise ValueError(
                f"AWAC elite buffer cache token mismatch: requested={token!r}, record={record.get('token')!r}."
            )
        return record

    def _init_stage3_runtime(self, cfg: GRPOConfig) -> None:
        """Initializes Stage3 sampling/runtime hyperparameters shared by GRPO and AWAC/IQL."""
        self.denoised_clip_value = cfg.denoised_clip_value
        self.eval_randn_clip_value = cfg.eval_randn_clip_value
        self.randn_clip_value = cfg.randn_clip_value
        self.final_action_clip_value = cfg.final_action_clip_value
        self.eps_clip_value = cfg.eps_clip_value
        self.eval_min_sampling_denoising_std = cfg.eval_min_sampling_denoising_std
        self.min_sampling_denoising_std = cfg.min_sampling_denoising_std
        self.min_logprob_denoising_std = cfg.min_logprob_denoising_std
        self.clip_advantage_lower_quantile = cfg.clip_advantage_lower_quantile
        self.clip_advantage_upper_quantile = cfg.clip_advantage_upper_quantile
        self.gamma_denoising = cfg.gamma_denoising
        self.grpo_sample_time = int(cfg.sample_time)
        if self.grpo_sample_time <= 0:
            raise ValueError("GRPO sample_time must be positive.")
        self.bc_anneal = bool(cfg.bc_anneal)
        self.bc_coeff_start = float(cfg.bc_coeff_start)
        self.bc_coeff_end = float(cfg.bc_coeff_end)
        self.bc_anneal_epochs = int(cfg.bc_anneal_epochs)
        if self.bc_coeff_start < 0.0 or self.bc_coeff_end < 0.0:
            raise ValueError("BC coefficients must be non-negative.")
        if self.bc_anneal_epochs <= 0:
            raise ValueError("bc_anneal_epochs must be positive.")
        for name in (
            "use_safety_shaped_reward",
            "hard_gate_nc",
            "hard_gate_dac",
            "hard_gate_ttc",
            "hard_gate_ddc",
            "hard_gate_tlc",
            "nc_safe_threshold",
            "dac_safe_threshold",
            "ttc_safe_threshold",
            "ddc_safe_threshold",
            "tlc_safe_threshold",
            "progress_bonus_weight",
            "unsafe_reward_floor",
            "unsafe_pdms_scale",
            "use_asymmetric_safe_advantage",
            "advantage_std_floor",
            "safe_negative_adv_scale",
            "unsafe_advantage_offset",
            "use_trajectory_level_objective",
            "trajectory_logprob_reduce",
            "use_gspo_ratio",
            "gspo_clip_low",
            "gspo_clip_high",
            "behavior_policy_sync_interval",
            "behavior_policy_sample",
            "use_dynamic_group_weight",
            "min_group_reward_std",
            "all_safe_low_std_group_weight",
            "all_unsafe_group_weight",
            "log_safe_diversity",
            "use_diversity_reward",
            "diversity_reward_weight",
            "diversity_pdms_threshold",
            "diversity_distance_scale",
            "diversity_metric",
        ):
            setattr(self, name, getattr(cfg, name))
        self.grpo_update_counter = 0

    def _init_grpo(self, cfg: GRPOConfig):
        """Initializes components and hyperparameters for GRPO training."""
        self._init_stage3_runtime(cfg)

        self._init_stage3_oracle(cfg)

        self._safe_load_reference_policy(cfg.reference_policy_checkpoint)

        behavior_policy = None
        if self.use_gspo_ratio:
            behavior_policy = copy.deepcopy(self)
            self._freeze_policy(behavior_policy)

        self.old_policy = copy.deepcopy(self)
        self._freeze_policy(self.old_policy)

        if behavior_policy is not None:
            self.behavior_policy = behavior_policy

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
        )
        return any(marker in key for marker in expert_markers)

    def _safe_load_reference_policy(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path) if checkpoint_path else None
        if path is None or not path.is_file():
            if self.config.allow_random_init:
                warnings.warn(
                    f"GRPO reference policy checkpoint is missing: {checkpoint_path!r}. "
                    "Continuing with random initialization because allow_random_init=True.",
                    RuntimeWarning,
                )
                return
            raise FileNotFoundError(f"GRPO reference policy checkpoint not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        if not isinstance(state_dict, dict):
            raise TypeError(f"Reference policy checkpoint must contain a state_dict dict, got {type(state_dict).__name__}.")

        model_dict = self.state_dict()
        filtered_ckpt: Dict[str, torch.Tensor] = {}
        unexpected_keys = []
        shape_mismatches = []
        skipped_expert_shape = []
        for key, value in state_dict.items():
            mapped_key = key[len("agent.action_head."):] if key.startswith("agent.action_head.") else key
            if mapped_key not in model_dict or not isinstance(value, torch.Tensor):
                unexpected_keys.append(mapped_key)
                continue
            if value.shape != model_dict[mapped_key].shape:
                message = f"{mapped_key}: checkpoint {tuple(value.shape)} vs model {tuple(model_dict[mapped_key].shape)}"
                if self._is_expert_parameter_key(mapped_key):
                    skipped_expert_shape.append(message)
                    continue
                shape_mismatches.append(message)
                continue
            filtered_ckpt[mapped_key] = value

        if shape_mismatches:
            raise RuntimeError(
                "Reference policy checkpoint shape mismatch for baseline parameters:\n"
                + "\n".join(f"  - {item}" for item in shape_mismatches)
            )

        incompatible = self.load_state_dict(filtered_ckpt, strict=False)
        print(f"Loaded GRPO reference policy from {path} with strict=False.")
        print(f"  loaded keys: {len(filtered_ckpt)}")
        if incompatible.missing_keys:
            print("  missing keys:")
            for key in incompatible.missing_keys:
                prefix = "expected expert " if self._is_expert_parameter_key(key) else ""
                print(f"    - {prefix}{key}")
        if unexpected_keys or incompatible.unexpected_keys:
            print("  unexpected keys:")
            for key in [*unexpected_keys, *incompatible.unexpected_keys]:
                print(f"    - {key}")
        if skipped_expert_shape:
            print("  skipped expert shape mismatches:")
            for item in skipped_expert_shape:
                print(f"    - {item}")

    @staticmethod
    def cosine_beta_schedule(timesteps: int, s: float = 0.008, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Calculates a cosine noise schedule as proposed in the iDDPM paper.
        
        This method is static as it does not depend on the instance's state.
        """
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
        return torch.tensor(betas_clipped, dtype=dtype)

    @staticmethod
    def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """
        Extracts values from tensor `a` at indices `t` and reshapes them
        to be broadcastable with a tensor of shape `x_shape`.
        
        This method is static as it does not depend on the instance's state.
        """
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    @staticmethod
    def make_timesteps(batch_size: int, i: int, device: torch.device) -> torch.Tensor:
        """
        Creates a tensor of a constant value `i` for a given batch size and device.
        
        This method is static as it does not depend on the instance's state.
        """
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        return t

    def set_training_progress(self, epoch: int, total_epochs: int):
        self.config.current_train_epoch = int(epoch)
        self.config.total_train_epochs = max(1, int(total_epochs))
        if hasattr(self, "last_rd"):
            self.last_rd.set_training_progress(epoch, total_epochs)
        if hasattr(self, "last_vla_cot"):
            self.last_vla_cot.set_training_progress(epoch, total_epochs)

    def set_frozen_modules_to_eval_mode(self):
        """
        Sets frozen parts of the model to evaluation mode during training.
        This is necessary to disable behaviors like dropout in the frozen layers.
        """
        if self.training:

            if not self.config.tune_projector:
                self.his_traj_encoder.eval()
                self.ego_status_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                self.feature_encoder.eval()
                self.fusion_projector.eval()
                if self.config.use_expert_features:
                    for module_name in (
                        "jepa_projector",
                        "vggt_projector",
                        "jepa_adapter",
                        "vggt_adapter",
                        "jepa_alignment_head",
                        "vggt_alignment_head",
                        "jepa_horizon_conditioner",
                        "vggt_horizon_conditioner",
                    ):
                        module = getattr(self, module_name, None)
                        if module is not None:
                            module.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            
            if not self.config.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        """Samples time for training based on the sampling method."""
        if self.config.sampling_method == 'flow':
            sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
            return (self.config.flow_cfg.noise_s - sample) / self.config.flow_cfg.noise_s
        elif self.config.sampling_method in ['ddpm', 'ddim']:
            return torch.randint(0, self.ddpm_num_train_timesteps, (batch_size,), device=device).long()
        else:
            raise ValueError(f"Unsupported sampling method: {self.config.sampling_method}")

    def _sample_offline_rl_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        mode: str = "uniform",
    ) -> torch.Tensor:
        """Sample DDPM/DDIM train timesteps for offline RL losses without changing default behavior."""
        mode = str(mode)
        if mode == "uniform" or self.config.sampling_method not in ["ddpm", "ddim"]:
            return self.sample_time(batch_size, device=device, dtype=dtype)
        if mode == "ddim" and hasattr(self, "ddim_t_schedule"):
            idx = torch.randint(0, int(self.ddim_t_schedule.numel()), (batch_size,), device=device)
            return self.ddim_t_schedule.to(device=device)[idx].long()

        num_steps = int(self.ddpm_num_train_timesteps)
        if mode == "low_noise":
            high = max(1, int(round(num_steps * float(self.offline_rl_cfg.low_noise_timestep_frac))))
            return torch.randint(0, high, (batch_size,), device=device).long()
        if mode == "mid_noise":
            low = int(round(num_steps * float(self.offline_rl_cfg.mid_noise_timestep_low_frac)))
            high = max(low + 1, int(round(num_steps * float(self.offline_rl_cfg.mid_noise_timestep_high_frac))))
            high = min(high, num_steps)
            return torch.randint(low, high, (batch_size,), device=device).long()
        if mode == "ddim":
            return self.sample_time(batch_size, device=device, dtype=dtype)
        raise ValueError(f"Unsupported offline RL timestep sampling mode: {mode!r}.")

    def _resolve_expert_tokens(
        self,
        action_input: Optional[BatchFeature],
        stream: str,
        kind: str,
        *,
        required: bool,
    ) -> Optional[torch.Tensor]:
        if action_input is None:
            if required:
                raise KeyError(f"use_expert_features=True requires action_input for {stream} {kind} tokens.")
            return None

        if kind == "context":
            keys = [f"{stream}_context_tokens"]
            legacy_key = f"{stream}_tokens"
            if legacy_key not in keys:
                keys.append(legacy_key)
        elif kind == "target":
            keys = [f"{stream}_target_tokens"]
        else:
            raise ValueError(f"Unknown expert token kind: {kind!r}")

        for key in keys:
            if key in action_input:
                return action_input[key]
        if required:
            raise KeyError(
                f"use_expert_features=True requires one of {keys} in action_input for {stream} {kind} tokens."
            )
        return None

    def _warn_if_expert_targets_present(
        self,
        action_input: Optional[BatchFeature],
        caller: str,
    ) -> None:
        if action_input is None:
            return
        present = [key for key in self.expert_target_keys if key in action_input]
        if present:
            message = (
                f"{caller} received train-only expert target keys {present}. "
                "They are ignored and are never used for inference conditioning."
            )
            if not self.training and not self.config.allow_future_targets_in_inference:
                warnings.warn(message, RuntimeWarning)
            elif not self.training:
                warnings.warn(message, RuntimeWarning)

    def _validate_teacher_tokens(
        self,
        tokens: torch.Tensor,
        *,
        expected_dim: int,
        expected_tokens: Optional[int],
        name: str,
        vl_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(tokens, torch.Tensor):
            raise TypeError(f"action_input.{name} must be a torch.Tensor, got {type(tokens).__name__}.")
        if tokens.ndim != 3:
            raise ValueError(f"action_input.{name} must have shape [B, K, D], got {tuple(tokens.shape)}.")
        if tokens.shape[0] != vl_embeds.shape[0]:
            raise ValueError(
                f"action_input.{name} batch size {tokens.shape[0]} does not match vl_features batch size {vl_embeds.shape[0]}."
            )
        if expected_tokens is not None and tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"action_input.{name} token count {tokens.shape[1]} does not match expected {expected_tokens}."
            )
        if tokens.shape[-1] != expected_dim:
            raise ValueError(
                f"action_input.{name} dim {tokens.shape[-1]} does not match expected_dim={expected_dim}."
            )
        return tokens.to(device=vl_embeds.device, dtype=vl_embeds.dtype)

    def _encode_vlm(self, vl_features: torch.Tensor) -> torch.Tensor:
        return self.feature_encoder(vl_features)

    def _apply_stream_gate_and_dropout(
        self,
        embeds: torch.Tensor,
        *,
        stream: str,
        training: bool,
    ) -> torch.Tensor:
        if self.config.use_expert_gates:
            gate = self.jepa_gate if stream == "jepa" else self.vggt_gate
            embeds = embeds * torch.sigmoid(gate).to(device=embeds.device, dtype=embeds.dtype)
        if self.config.expert_context_scale != 1.0:
            embeds = embeds * embeds.new_tensor(float(self.config.expert_context_scale))
        if training and self.config.expert_stream_dropout > 0.0:
            keep = torch.rand(embeds.shape[0], 1, 1, device=embeds.device) >= self.config.expert_stream_dropout
            embeds = embeds * keep.to(dtype=embeds.dtype)
        if training and self.config.expert_dropout > 0.0:
            embeds = F.dropout(embeds, p=self.config.expert_dropout, training=True)
        return embeds

    def _build_stream_context(
        self,
        stream: str,
        vl_embeds: torch.Tensor,
        action_input: Optional[BatchFeature],
        training: bool,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if stream == "jepa":
            enabled = self.config.use_jepa
            projector = getattr(self, "jepa_projector", None)
            adapter = getattr(self, "jepa_adapter", None)
            expected_dim = self.config.jepa_dim
            num_tokens = self.config.num_jepa_tokens
            teacher_type = getattr(self, "jepa_type_embedding", None)
            latent_type = getattr(self, "z_jepa_type_embedding", None)
        elif stream == "vggt":
            enabled = self.config.use_vggt
            projector = getattr(self, "vggt_projector", None)
            adapter = getattr(self, "vggt_adapter", None)
            expected_dim = self.config.vggt_dim
            num_tokens = self.config.num_vggt_tokens
            teacher_type = getattr(self, "vggt_type_embedding", None)
            latent_type = getattr(self, "z_vggt_type_embedding", None)
        else:
            raise ValueError(f"Unknown expert stream: {stream!r}")

        if not enabled:
            return None, None

        parts: list[torch.Tensor] = []
        latent_768: Optional[torch.Tensor] = None

        if self.config.use_teacher_context_tokens:
            tokens = self._resolve_expert_tokens(action_input, stream, "context", required=True)
            assert tokens is not None
            tokens = self._validate_teacher_tokens(
                tokens,
                expected_dim=expected_dim,
                expected_tokens=num_tokens,
                name=f"{stream}_context_tokens",
                vl_embeds=vl_embeds,
            )
            teacher_384 = projector(tokens)
            if self.config.use_expert_type_embedding and teacher_type is not None:
                teacher_384 = teacher_384 + teacher_type.to(device=teacher_384.device, dtype=teacher_384.dtype)
            parts.append(teacher_384)

        if self.config.use_student_latent_adapters:
            latent_768, latent_384 = adapter(vl_embeds)
            if self.config.use_expert_type_embedding and latent_type is not None:
                latent_384 = latent_384 + latent_type.to(device=latent_384.device, dtype=latent_384.dtype)
            parts.append(latent_384)

        if not parts:
            return None, latent_768
        stream_context = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        stream_context = self._apply_stream_gate_and_dropout(stream_context, stream=stream, training=training)
        return stream_context, latent_768

    def _compute_horizon_expert_residual(
        self,
        jepa_all: Optional[torch.Tensor],
        vggt_all: Optional[torch.Tensor],
        *,
        reference: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if (
            not self.config.use_expert_features
            or not self.config.use_horizon_expert_residual
        ):
            return None

        conditions: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        if jepa_all is not None and hasattr(self, "jepa_horizon_conditioner"):
            conditions.append(self.jepa_horizon_conditioner(jepa_all))
            logits.append(self.branch_logits[1])
        if vggt_all is not None and hasattr(self, "vggt_horizon_conditioner"):
            conditions.append(self.vggt_horizon_conditioner(vggt_all))
            logits.append(self.branch_logits[2])
        if not conditions:
            return None

        weights = torch.softmax(torch.stack(logits).float(), dim=0).to(device=reference.device, dtype=reference.dtype)
        stacked = torch.stack(conditions, dim=1)
        residual = (stacked * weights.view(1, -1, 1, 1)).sum(dim=1)
        return residual * residual.new_tensor(float(self.config.expert_horizon_residual_scale))

    def _build_expert_context(
        self,
        vl_embeds: torch.Tensor,
        action_input: Optional[BatchFeature],
        training: bool,
    ) -> Dict[str, Any]:
        jepa_all, z_jepa_768 = self._build_stream_context("jepa", vl_embeds, action_input, training)
        vggt_all, z_vggt_768 = self._build_stream_context("vggt", vl_embeds, action_input, training)
        horizon_residual = self._compute_horizon_expert_residual(jepa_all, vggt_all, reference=vl_embeds)
        return {
            "jepa_all": jepa_all,
            "vggt_all": vggt_all,
            "z_jepa_768": z_jepa_768,
            "z_vggt_768": z_vggt_768,
            "horizon_residual": horizon_residual,
            "diagnostics": {
                "jepa_gate": torch.sigmoid(self.jepa_gate.detach()).float(),
                "vggt_gate": torch.sigmoid(self.vggt_gate.detach()).float(),
                "branch_weights": torch.softmax(self.branch_logits.detach().float(), dim=0),
                "expert_context_scale": torch.tensor(float(self.config.expert_context_scale)),
                "expert_horizon_residual_scale": torch.tensor(float(self.config.expert_horizon_residual_scale)),
            },
        }

    def _compute_branch_context_mean(
        self,
        vl_embeds: torch.Tensor,
        jepa_all: Optional[torch.Tensor],
        vggt_all: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.config.use_expert_features or not self.config.use_branch_weighted_mean:
            parts = [vl_embeds]
            if jepa_all is not None:
                parts.append(jepa_all)
            if vggt_all is not None:
                parts.append(vggt_all)
            return torch.cat(parts, dim=1).mean(1)

        means = [vl_embeds.mean(1)]
        logits = [self.branch_logits[0]]
        if jepa_all is not None:
            means.append(jepa_all.mean(1))
            logits.append(self.branch_logits[1])
        if vggt_all is not None:
            means.append(vggt_all.mean(1))
            logits.append(self.branch_logits[2])
        weights = torch.softmax(torch.stack(logits).float(), dim=0).to(device=vl_embeds.device, dtype=vl_embeds.dtype)
        stacked_means = torch.stack(means, dim=1)
        return (stacked_means * weights.view(1, -1, 1)).sum(dim=1)

    def _stream_alignment_weight(self, stream: str) -> float:
        stream_weight = self.config.jepa_alignment_weight if stream == "jepa" else self.config.vggt_alignment_weight
        return self.config.expert_alignment_weight + stream_weight

    def _alignment_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.alignment_loss_type == "normalized_mse":
            return normalized_mse_loss(pred, target)
        if self.config.alignment_loss_type == "mse":
            return F.mse_loss(pred.float(), target.detach().float())
        if self.config.alignment_loss_type == "cosine":
            return (1.0 - F.cosine_similarity(pred.float(), target.detach().float(), dim=-1)).mean()
        raise ValueError(f"Unsupported alignment_loss_type: {self.config.alignment_loss_type}")

    def _compute_alignment_losses(
        self,
        z_jepa_768: Optional[torch.Tensor],
        z_vggt_768: Optional[torch.Tensor],
        action_input: Optional[BatchFeature],
        *,
        training: bool,
        reference: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        losses = {
            "jepa_alignment_loss": reference.new_zeros(()),
            "vggt_alignment_loss": reference.new_zeros(()),
        }
        if not self.config.use_expert_features or not training:
            return losses

        for stream, latent, head, expected_dim, expected_tokens in (
            ("jepa", z_jepa_768, getattr(self, "jepa_alignment_head", None), self.config.jepa_dim, self.config.num_jepa_tokens),
            ("vggt", z_vggt_768, getattr(self, "vggt_alignment_head", None), self.config.vggt_dim, self.config.num_vggt_tokens),
        ):
            if latent is None or head is None:
                continue
            target = self._resolve_expert_tokens(action_input, stream, "target", required=False)
            if target is None:
                continue
            target = self._validate_teacher_tokens(
                target,
                expected_dim=expected_dim,
                expected_tokens=expected_tokens,
                name=f"{stream}_target_tokens",
                vl_embeds=latent,
            )
            pred = head(latent)
            if pred.shape != target.shape:
                raise ValueError(
                    f"{stream} alignment prediction shape {tuple(pred.shape)} does not match target {tuple(target.shape)}."
                )
            losses[f"{stream}_alignment_loss"] = self._alignment_loss(pred, target).to(dtype=reference.dtype)
        return losses

    @staticmethod
    def _two_expert_corruption_mode(action_input: Optional[BatchFeature]) -> str:
        if action_input is None:
            return "normal"
        value = action_input.get("two_expert_corruption_mode", "normal")
        if isinstance(value, torch.Tensor):
            code = int(value.detach().reshape(-1)[0].cpu().item()) if value.numel() else 0
            return {
                0: "normal",
                1: "zero_h_dyn",
                2: "zero_h_geo",
                3: "zero_all_experts",
                4: "raw_vlm_only",
                5: "dyn_only",
                6: "geo_only",
            }.get(code, "normal")
        return str(value)

    def _build_two_expert_context(
        self,
        vl_embeds: torch.Tensor,
        action_input: Optional[BatchFeature],
    ) -> Dict[str, Any]:
        mode = self._two_expert_corruption_mode(action_input)
        if action_input is None:
            raise KeyError("use_two_expert_slots=True requires action_input with two_expert_h_dyn and two_expert_h_geo.")
        h_dyn = action_input.get("two_expert_h_dyn", None)
        h_geo = action_input.get("two_expert_h_geo", None)
        if not isinstance(h_dyn, torch.Tensor) or not isinstance(h_geo, torch.Tensor):
            raise KeyError("two_expert_slot cache requires two_expert_h_dyn and two_expert_h_geo.")
        if h_dyn.ndim != 4:
            raise ValueError(f"two_expert_h_dyn must have shape [B,3,12,D], got {tuple(h_dyn.shape)}.")
        if h_geo.ndim != 3:
            raise ValueError(f"two_expert_h_geo must have shape [B,12,D], got {tuple(h_geo.shape)}.")
        if h_dyn.shape[0] != vl_embeds.shape[0] or h_geo.shape[0] != vl_embeds.shape[0]:
            raise ValueError("two_expert hidden batch size must match VLM batch size.")
        if h_dyn.shape[1] != self.config.two_expert_num_dyn_groups:
            raise ValueError(f"two_expert_h_dyn group count {h_dyn.shape[1]} does not match config.")
        if h_dyn.shape[2] != self.config.two_expert_dyn_tokens_per_group:
            raise ValueError(f"two_expert_h_dyn tokens/group {h_dyn.shape[2]} does not match config.")
        if h_geo.shape[1] != self.config.two_expert_num_geo_tokens:
            raise ValueError(f"two_expert_h_geo token count {h_geo.shape[1]} does not match config.")

        h_dyn = h_dyn.to(device=vl_embeds.device, dtype=vl_embeds.dtype)
        h_geo = h_geo.to(device=vl_embeds.device, dtype=vl_embeds.dtype)
        if mode in {"zero_h_dyn", "zero_all_experts", "raw_vlm_only", "geo_only"}:
            h_dyn = torch.zeros_like(h_dyn)
        if mode in {"zero_h_geo", "zero_all_experts", "raw_vlm_only", "dyn_only"}:
            h_geo = torch.zeros_like(h_geo)
        if mode not in {"normal", "zero_h_dyn", "zero_h_geo", "zero_all_experts", "raw_vlm_only", "dyn_only", "geo_only"}:
            raise ValueError(f"Unknown two_expert corruption mode: {mode!r}.")

        dyn_proj = self.two_expert_dyn_proj(h_dyn.reshape(h_dyn.shape[0], -1, h_dyn.shape[-1]))
        geo_proj = self.two_expert_geo_proj(h_geo)
        queries = self.two_expert_horizon_queries.unsqueeze(0).expand(vl_embeds.shape[0], -1, -1).to(vl_embeds)
        f_dyn, _ = self.two_expert_dyn_horizon_attn(queries, dyn_proj, dyn_proj, need_weights=False)
        f_geo, _ = self.two_expert_geo_horizon_attn(queries, geo_proj, geo_proj, need_weights=False)
        dyn_delta = self.two_expert_dyn_delta_proj(f_dyn)
        geo_delta = self.two_expert_geo_delta_proj(f_geo)
        if mode == "raw_vlm_only":
            dyn_delta = torch.zeros_like(dyn_delta)
            geo_delta = torch.zeros_like(geo_delta)
        expert_step_condition = dyn_delta + geo_delta
        diagnostics = {
            "two_expert_condition_enabled": vl_embeds.new_tensor(float(mode != "raw_vlm_only")),
            "two_expert_corruption_mode_code": vl_embeds.new_tensor(
                {
                    "normal": 0,
                    "zero_h_dyn": 1,
                    "zero_h_geo": 2,
                    "zero_all_experts": 3,
                    "raw_vlm_only": 4,
                    "dyn_only": 5,
                    "geo_only": 6,
                }[mode]
            ),
            "two_expert_raw_vlm_context_used": vl_embeds.new_tensor(1.0),
            "two_expert_h_dyn_norm": h_dyn.detach().float().norm(dim=-1).mean().to(dtype=vl_embeds.dtype),
            "two_expert_h_geo_norm": h_geo.detach().float().norm(dim=-1).mean().to(dtype=vl_embeds.dtype),
            "two_expert_f_dyn_norm": f_dyn.detach().float().norm(dim=-1).mean().to(dtype=vl_embeds.dtype),
            "two_expert_f_geo_norm": f_geo.detach().float().norm(dim=-1).mean().to(dtype=vl_embeds.dtype),
            "two_expert_dyn_expert_delta_norm": dyn_delta.detach().float().norm(dim=-1).mean().to(dtype=vl_embeds.dtype),
            "two_expert_geo_expert_delta_norm": geo_delta.detach().float().norm(dim=-1).mean().to(dtype=vl_embeds.dtype),
            "two_expert_zero_init_dyn": vl_embeds.new_tensor(
                float(torch.count_nonzero(self.two_expert_dyn_delta_proj.weight.detach()).item() == 0)
            ),
            "two_expert_zero_init_geo": vl_embeds.new_tensor(
                float(torch.count_nonzero(self.two_expert_geo_delta_proj.weight.detach()).item() == 0)
            ),
        }
        return {
            "context_tokens": vl_embeds,
            "context_mean": vl_embeds.mean(1),
            "expert_step_condition": expert_step_condition,
            "f_dyn": f_dyn,
            "f_geo": f_geo,
            "diagnostics": diagnostics,
        }

    def _prepare_dit_context(
        self,
        vl_features: torch.Tensor,
        action_input: Optional[BatchFeature],
        training: bool,
        noisy_actions: Optional[torch.Tensor] = None,
        diffusion_timestep: Optional[torch.Tensor] = None,
        target_action_norm: Optional[torch.Tensor] = None,
        allow_target_tokens: Optional[bool] = None,
    ) -> Dict[str, Any]:
        vl_embeds = self._encode_vlm(vl_features)
        zero = vl_embeds.new_zeros(())
        base_losses = {
            "jepa_alignment_loss": zero,
            "vggt_alignment_loss": zero,
            "future_jepa_loss": zero,
            "vggt_geometry_loss": zero,
            "coarse_traj_loss": zero,
            "coarse_heading_loss": zero,
            "risk_loss": zero,
            "policy_kd_loss": zero,
            "last_vla_geometry_loss": zero,
            "last_vla_dynamic_loss": zero,
            "last_vla_coarse_loss": zero,
            "last_vla_heading_loss": zero,
            "last_vla_progress_loss": zero,
            "last_vla_risk_loss": zero,
            "last_vla_cot_consistency_loss": zero,
        }
        if (
            not self.config.use_expert_features
            and not self.config.use_last_rd
            and not self.config.use_last_vla
            and not self.config.use_two_expert_slots
        ):
            return {
                "vl_embeds": vl_embeds,
                "context_tokens": vl_embeds,
                "context_mean": vl_embeds.mean(1),
                "expert_step_condition": None,
                **base_losses,
                "diagnostics": {},
            }

        if self.config.use_two_expert_slots:
            two_expert_context = self._build_two_expert_context(vl_embeds, action_input)
            return {
                "vl_embeds": vl_embeds,
                "context_tokens": two_expert_context["context_tokens"],
                "context_mean": two_expert_context["context_mean"],
                "expert_step_condition": two_expert_context["expert_step_condition"],
                "cot_condition_tokens": None,
                **base_losses,
                "diagnostics": two_expert_context["diagnostics"],
                "two_expert_f_dyn": two_expert_context["f_dyn"],
                "two_expert_f_geo": two_expert_context["f_geo"],
                "selected_target_norm": target_action_norm,
            }

        if self.config.use_last_vla:
            allow_targets = training if allow_target_tokens is None else bool(allow_target_tokens)
            last_vla_output = self.last_vla_cot(
                vl_embeds,
                action_input,
                training=training,
                target_action_norm=target_action_norm,
                noisy_action_norm=noisy_actions,
                diffusion_timestep=diffusion_timestep,
                current_epoch=self.config.current_train_epoch,
                total_epochs=self.config.total_train_epochs,
                allow_target_tokens=allow_targets,
            )
            if (
                training
                and allow_targets
                and self.config.last_vla_dynamic_loss_weight > 0.0
                and float(last_vla_output.diagnostics.get("dynamic_teacher_missing", zero).detach().float().item()) > 0.0
            ):
                raise KeyError(
                    "Last-VLA dynamic teacher loss has positive weight but action_input is missing "
                    "jepa_target_tokens. Disable the loss or provide train-only future JEPA targets."
                )
            if (
                training
                and allow_targets
                and self.config.last_vla_geometry_loss_weight > 0.0
                and float(last_vla_output.diagnostics.get("geometry_teacher_missing", zero).detach().float().item()) > 0.0
            ):
                raise KeyError(
                    "Last-VLA geometry teacher loss has positive weight but no geometry target is available. "
                    "Provide explicit full_geometry cache, or enable patch fallback with vggt_context_tokens."
                )
            loss_map = {
                "geometry_loss": "last_vla_geometry_loss",
                "dynamic_loss": "last_vla_dynamic_loss",
                "coarse_loss": "last_vla_coarse_loss",
                "heading_loss": "last_vla_heading_loss",
                "progress_loss": "last_vla_progress_loss",
                "risk_loss": "last_vla_risk_loss",
                "cot_consistency_loss": "last_vla_cot_consistency_loss",
            }
            for source_key, output_key in loss_map.items():
                if source_key in last_vla_output.losses:
                    base_losses[output_key] = last_vla_output.losses[source_key]
            decoupled_mode = (
                self.config.last_vla_condition_mode == "decoupled_cot_residual"
                and not self.config.last_vla_cot_bottleneck_mode
            )
            context_tokens = last_vla_output.planner_context_tokens
            context_mean = last_vla_output.context_mean
            expert_step_condition = last_vla_output.horizon_condition
            cot_condition_tokens = None
            diagnostics = dict(last_vla_output.diagnostics)
            if decoupled_mode:
                context_tokens = last_vla_output.base_context_tokens
                cot_condition_tokens = last_vla_output.cot_condition_tokens
                if training and self.config.last_vla_cot_condition_dropout > 0.0:
                    cot_condition_tokens = F.dropout(
                        cot_condition_tokens,
                        p=float(self.config.last_vla_cot_condition_dropout),
                        training=True,
                    )
                context_mean_base = context_tokens.mean(1)
                cot_context_mean_residual = self.last_vla_context_mean_cot_proj(cot_condition_tokens.mean(1))
                cot_scale = self.last_vla_cot_condition_scale.to(
                    device=cot_context_mean_residual.device,
                    dtype=cot_context_mean_residual.dtype,
                )
                cot_context_mean_residual = cot_context_mean_residual * cot_scale
                context_mean = context_mean_base + cot_context_mean_residual
                horizon_queries = self.last_vla_horizon_cot_queries.unsqueeze(0).expand(
                    context_tokens.shape[0],
                    -1,
                    -1,
                ).to(cot_condition_tokens)
                cot_horizon_raw, _ = self.last_vla_horizon_cot_attn(
                    horizon_queries,
                    cot_condition_tokens,
                    cot_condition_tokens,
                    need_weights=False,
                )
                cot_horizon_residual = self.last_vla_horizon_cot_proj(cot_horizon_raw) * cot_scale
                expert_step_condition = last_vla_output.horizon_condition + cot_horizon_residual
                diagnostics.update(
                    {
                        "last_vla_cot_condition_norm": cot_condition_tokens.detach().float().norm(dim=-1).mean(),
                        "last_vla_cot_condition_delta_norm": cot_horizon_residual.detach().float().norm(dim=-1).mean(),
                        "last_vla_cot_branch_zero_init": cot_horizon_residual.new_tensor(
                            float(
                                torch.count_nonzero(self.last_vla_context_mean_cot_proj.weight.detach()).item() == 0
                                and torch.count_nonzero(self.last_vla_horizon_cot_proj.weight.detach()).item() == 0
                            )
                        ),
                        "last_vla_raw_vlm_base_context_norm": context_tokens.detach().float().norm(dim=-1).mean(),
                        "last_vla_context_mean_cot_residual_norm": cot_context_mean_residual.detach().float().norm(dim=-1).mean(),
                        "last_vla_horizon_cot_residual_norm": cot_horizon_residual.detach().float().norm(dim=-1).mean(),
                    }
                )
            return {
                "vl_embeds": vl_embeds,
                "context_tokens": context_tokens,
                "context_mean": context_mean,
                "expert_step_condition": expert_step_condition,
                "cot_condition_tokens": cot_condition_tokens,
                **base_losses,
                "diagnostics": diagnostics,
                "last_vla_output": last_vla_output,
                "selected_target_norm": target_action_norm,
            }

        expert: Optional[Dict[str, Any]] = None
        if self.config.use_expert_features:
            expert = self._build_expert_context(vl_embeds, action_input, training)
            alignment_losses = self._compute_alignment_losses(
                expert["z_jepa_768"],
                expert["z_vggt_768"],
                action_input,
                training=training if allow_target_tokens is None else bool(allow_target_tokens),
                reference=zero,
            )
            base_losses.update(alignment_losses)

        if not self.config.use_last_rd:
            assert expert is not None
            context_parts = [vl_embeds]
            if expert["jepa_all"] is not None:
                context_parts.append(expert["jepa_all"])
            if expert["vggt_all"] is not None:
                context_parts.append(expert["vggt_all"])
            context_tokens = torch.cat(context_parts, dim=1)
            context_mean = self._compute_branch_context_mean(vl_embeds, expert["jepa_all"], expert["vggt_all"])
            return {
                "vl_embeds": vl_embeds,
                "context_tokens": context_tokens,
                "context_mean": context_mean,
                "expert_step_condition": expert["horizon_residual"],
                **base_losses,
                "diagnostics": expert["diagnostics"],
            }

        last_rd_input = action_input
        if last_rd_input is not None and (
            self.config.last_rd_token_dropout > 0.0
            or self.config.last_rd_group_dropout > 0.0
        ):
            last_rd_data = dict(last_rd_input)
            last_rd_data.setdefault("last_rd_token_dropout", float(self.config.last_rd_token_dropout))
            last_rd_data.setdefault("last_rd_group_dropout", float(self.config.last_rd_group_dropout))
            last_rd_input = BatchFeature(data=last_rd_data)
        last_rd_output = self.last_rd(
            vl_embeds,
            last_rd_input,
            training=training,
            allow_target_tokens=allow_target_tokens,
            noisy_actions=noisy_actions,
            diffusion_timestep=diffusion_timestep,
            norm_odo=self.norm_odo,
        )

        context_parts = [vl_embeds]
        if expert is not None:
            if expert["jepa_all"] is not None:
                context_parts.append(expert["jepa_all"])
            if expert["vggt_all"] is not None:
                context_parts.append(expert["vggt_all"])
        last_rd_tokens = last_rd_output.planning_tokens
        if self.config.last_rd_context_scale != 1.0:
            last_rd_tokens = last_rd_tokens * last_rd_tokens.new_tensor(float(self.config.last_rd_context_scale))
        context_parts.append(last_rd_tokens)
        context_tokens = torch.cat(context_parts, dim=1)

        if last_rd_output.group_weights is not None:
            summaries = [vl_embeds.mean(1)]
            for tokens in (
                last_rd_output.dynamic_tokens,
                last_rd_output.geometry_tokens,
                last_rd_output.ego_tokens,
                last_rd_output.risk_tokens,
            ):
                summaries.append(tokens.mean(1) if tokens is not None else vl_embeds.new_zeros(vl_embeds.shape[0], vl_embeds.shape[-1]))
            stacked = torch.stack(summaries, dim=1)
            context_mean = (stacked * last_rd_output.group_weights.to(stacked).unsqueeze(-1)).sum(dim=1)
        else:
            context_mean = context_tokens.mean(1)

        expert_step_condition = expert["horizon_residual"] if expert is not None else None
        if last_rd_output.horizon_condition is not None:
            last_rd_condition = last_rd_output.horizon_condition * last_rd_output.horizon_condition.new_tensor(
                float(self.config.last_rd_horizon_condition_scale)
            )
            expert_step_condition = (
                last_rd_condition
                if expert_step_condition is None
                else expert_step_condition.to(last_rd_condition) + last_rd_condition
            )

        diagnostics = dict(expert["diagnostics"]) if expert is not None else {}
        diagnostics.update(last_rd_output.diagnostics)
        for key, value in last_rd_output.losses.items():
            if key in base_losses:
                base_losses[key] = value
        return {
            "vl_embeds": vl_embeds,
            "context_tokens": context_tokens,
            "context_mean": context_mean,
            "expert_step_condition": expert_step_condition,
            **base_losses,
            "diagnostics": diagnostics,
            "last_rd_output": last_rd_output,
        }

    def _repeat_expert_action_input(
        self,
        action_input: BatchFeature,
        repeat: int,
    ) -> Optional[BatchFeature]:
        if (
            not self.config.use_expert_features
            and not self.config.use_last_rd
            and not self.config.use_last_vla
            and not self.config.use_two_expert_slots
            and not self.config.last_vla_use_residual_diffusion
        ):
            return None

        data: Dict[str, torch.Tensor] = {}
        for key in (
            "his_traj",
            "history_trajectory",
            "status_feature",
            "high_command_one_hot",
            "jepa_context_tokens",
            "jepa_tokens",
            "vggt_context_tokens",
            "vggt_tokens",
            "vggt_geometry_tokens",
            "vggt_geometry_mode_code",
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
            "last_vla_corrupt_zero_all_cot",
            "last_vla_corrupt_zero_geometry_cot",
            "last_vla_corrupt_zero_dynamic_cot",
            "last_vla_corrupt_zero_ego_cot",
            "last_vla_corrupt_zero_action_refine_cot",
            "last_vla_corrupt_zero_fusion_cot",
            "last_vla_corrupt_zero_cot_condition_branch",
            "last_vla_raw_vlm_only",
            "last_vla_cot_only_for_debug_only",
            "last_vla_corrupt_zero_coarse_prior",
            "last_vla_corrupt_geometry_cot",
            "last_vla_corrupt_dynamic_cot",
            "last_vla_corrupt_ego_cot",
            "last_vla_corrupt_action_refine_cot",
            "last_vla_zero_coarse_prior",
        ):
            if key in action_input and isinstance(action_input[key], torch.Tensor):
                data[key] = action_input[key].repeat_interleave(repeat, 0)
            elif key in action_input and isinstance(action_input[key], bool):
                data[key] = action_input[key]
        if self.config.use_expert_features:
            for stream, enabled in (("jepa", self.config.use_jepa), ("vggt", self.config.use_vggt)):
                if not enabled or f"{stream}_context_tokens" in data:
                    continue
                tokens = self._resolve_expert_tokens(action_input, stream, "context", required=True)
                assert tokens is not None
                data[f"{stream}_context_tokens"] = tokens.repeat_interleave(repeat, 0)
        return BatchFeature(data=data)

    def _last_vla_progress(self) -> float:
        return min(
            max(float(self.config.current_train_epoch) / max(1.0, float(self.config.total_train_epochs)), 0.0),
            1.0,
        )

    def _last_vla_residual_alpha(self, *, training: bool) -> float:
        if not self.config.last_vla_use_residual_diffusion:
            return 0.0
        if str(self.config.last_vla_residual_anchor_source) == "none":
            return 0.0
        start = float(self.config.last_vla_residual_alpha_start)
        end = float(self.config.last_vla_residual_alpha_end)
        if not training:
            return end
        warmup_epochs = int(self.config.last_vla_residual_alpha_warmup_epochs)
        if warmup_epochs <= 0:
            return end
        progress = min(max(float(self.config.current_train_epoch) / float(warmup_epochs), 0.0), 1.0)
        return start + (end - start) * progress

    def _last_vla_residual_anchor_norm(
        self,
        action_input: Optional[BatchFeature],
        reference_norm: torch.Tensor,
        *,
        required: Optional[bool] = None,
    ) -> tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        diagnostics: Dict[str, torch.Tensor] = {
            "residual_anchor_source_code": reference_norm.new_tensor(0.0),
            "residual_anchor_missing": reference_norm.new_tensor(0.0),
        }
        if not self.config.last_vla_use_residual_diffusion:
            return None, diagnostics

        source = str(self.config.last_vla_residual_anchor_source)
        if source == "none":
            return None, diagnostics
        if source != "vlm_text_traj":
            raise ValueError(f"Unsupported Last-VLA residual anchor source: {source!r}.")

        require_anchor = self.config.last_vla_require_residual_anchor if required is None else bool(required)
        if action_input is None:
            diagnostics["residual_anchor_missing"] = reference_norm.new_tensor(1.0)
            if require_anchor:
                raise KeyError("Last-VLA residual diffusion requires action_input with a VLM text trajectory anchor.")
            return None, diagnostics

        anchor_value = action_input.get("vlm_text_trajectory_norm", None)
        if isinstance(anchor_value, torch.Tensor):
            anchor_norm = anchor_value.to(device=reference_norm.device, dtype=reference_norm.dtype)
        else:
            anchor_value = action_input.get("vlm_text_trajectory", None)
            if isinstance(anchor_value, torch.Tensor):
                anchor_norm = self.norm_odo(anchor_value.to(device=reference_norm.device, dtype=reference_norm.dtype))
            else:
                diagnostics["residual_anchor_missing"] = reference_norm.new_tensor(1.0)
                if require_anchor:
                    raise KeyError(
                        "Last-VLA residual diffusion anchor_source='vlm_text_traj' requires "
                        "'vlm_text_trajectory_norm' or 'vlm_text_trajectory' in action_input."
                    )
                return None, diagnostics

        if anchor_norm.ndim == 2 and reference_norm.ndim == 3 and reference_norm.shape[0] == 1:
            anchor_norm = anchor_norm.unsqueeze(0)
        if tuple(anchor_norm.shape) != tuple(reference_norm.shape):
            raise ValueError(
                f"VLM text trajectory anchor shape {tuple(anchor_norm.shape)} does not match "
                f"reference trajectory shape {tuple(reference_norm.shape)}."
            )
        if not torch.isfinite(anchor_norm).all():
            raise ValueError("VLM text trajectory anchor contains non-finite values.")

        parse_ok = action_input.get("vlm_text_parse_ok", None)
        if isinstance(parse_ok, torch.Tensor):
            parse_ok_value = parse_ok.to(device=reference_norm.device).float()
            if parse_ok_value.numel() > 0 and not bool((parse_ok_value > 0.5).all().item()):
                raise ValueError("VLM text trajectory anchor has parse failures in this batch.")

        anchor_norm = anchor_norm.detach()
        diagnostics.update(
            {
                "residual_anchor_source_code": reference_norm.new_tensor(1.0),
                "residual_anchor_norm": anchor_norm.float().norm(dim=-1).mean().to(reference_norm.dtype),
                "residual_anchor_l1_to_reference": F.l1_loss(anchor_norm, reference_norm, reduction="mean").detach(),
            }
        )
        return anchor_norm, diagnostics

    def _last_vla_diffusion_target_info(
        self,
        selected_target_norm: torch.Tensor,
        *,
        training: bool,
        action_input: Optional[BatchFeature] = None,
    ) -> tuple[torch.Tensor, float, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        alpha = self._last_vla_residual_alpha(training=training)
        if alpha == 0.0:
            diagnostics = {
                "residual_anchor_source_code": selected_target_norm.new_tensor(0.0),
                "residual_anchor_missing": selected_target_norm.new_tensor(0.0),
                "residual_target_norm": selected_target_norm.float().norm(dim=-1).mean().to(selected_target_norm.dtype),
            }
            return selected_target_norm, 0.0, None, diagnostics

        anchor_norm, diagnostics = self._last_vla_residual_anchor_norm(
            action_input,
            selected_target_norm,
            required=True,
        )
        assert anchor_norm is not None
        diffusion_target = selected_target_norm - float(alpha) * anchor_norm
        diagnostics["residual_target_norm"] = diffusion_target.float().norm(dim=-1).mean().to(selected_target_norm.dtype)
        return diffusion_target, float(alpha), anchor_norm, diagnostics

    def _last_vla_diffusion_target(
        self,
        selected_target_norm: torch.Tensor,
        *,
        training: bool,
        action_input: Optional[BatchFeature] = None,
    ) -> tuple[torch.Tensor, float]:
        diffusion_target, alpha, _, _ = self._last_vla_diffusion_target_info(
            selected_target_norm,
            training=training,
            action_input=action_input,
        )
        return diffusion_target, alpha

    def _select_last_vla_training_target(
        self,
        action_input: BatchFeature,
        gt_actions_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mode = self.config.last_vla_teacher_traj_mode
        diagnostics = {
            "teacher_traj_used_ratio": gt_actions_norm.new_zeros(()),
            "teacher_traj_mix": gt_actions_norm.new_zeros(()),
            "teacher_score_mean": gt_actions_norm.new_zeros(()),
            "gt_score_mean": gt_actions_norm.new_zeros(()),
        }
        if mode in {"none", "gt"}:
            return gt_actions_norm, diagnostics

        teacher = action_input.get("teacher_trajectory_norm", None)
        if isinstance(teacher, torch.Tensor):
            teacher_norm = teacher.to(device=gt_actions_norm.device, dtype=gt_actions_norm.dtype)
        elif isinstance(action_input.get("teacher_trajectory", None), torch.Tensor):
            teacher_norm = self.norm_odo(action_input["teacher_trajectory"].to(device=gt_actions_norm.device, dtype=gt_actions_norm.dtype))
        else:
            if self.config.last_vla_stage == "teacher_traj_sft":
                raise KeyError(
                    "last_vla_stage='teacher_traj_sft' requires teacher_trajectory_norm or teacher_trajectory in action_input."
                )
            warnings.warn(
                "Last-VLA teacher trajectory target requested but missing; falling back to GT target.",
                RuntimeWarning,
            )
            return gt_actions_norm, diagnostics
        if teacher_norm.shape != gt_actions_norm.shape:
            raise ValueError(
                f"teacher trajectory shape {tuple(teacher_norm.shape)} does not match GT {tuple(gt_actions_norm.shape)}."
            )

        teacher_score = action_input.get("teacher_score", None)
        gt_score = action_input.get("gt_score", None)
        if isinstance(teacher_score, torch.Tensor):
            teacher_score = teacher_score.to(device=gt_actions_norm.device, dtype=gt_actions_norm.dtype).view(-1)
            if teacher_score.numel() == 1 and gt_actions_norm.shape[0] > 1:
                teacher_score = teacher_score.expand(gt_actions_norm.shape[0])
            if teacher_score.numel() != gt_actions_norm.shape[0]:
                raise ValueError(f"teacher_score must be scalar or [B], got {tuple(action_input['teacher_score'].shape)}.")
            diagnostics["teacher_score_mean"] = teacher_score.detach().float().mean()
        if isinstance(gt_score, torch.Tensor):
            gt_score = gt_score.to(device=gt_actions_norm.device, dtype=gt_actions_norm.dtype).view(-1)
            if gt_score.numel() == 1 and gt_actions_norm.shape[0] > 1:
                gt_score = gt_score.expand(gt_actions_norm.shape[0])
            if gt_score.numel() != gt_actions_norm.shape[0]:
                raise ValueError(f"gt_score must be scalar or [B], got {tuple(action_input['gt_score'].shape)}.")
            diagnostics["gt_score_mean"] = gt_score.detach().float().mean()

        if mode == "teacher_if_better":
            if isinstance(teacher_score, torch.Tensor) and isinstance(gt_score, torch.Tensor):
                margin = gt_actions_norm.new_tensor(float(self.config.last_vla_teacher_score_margin))
                use_teacher = teacher_score >= gt_score + margin
                mask = use_teacher.view(-1, 1, 1).to(dtype=gt_actions_norm.dtype)
                selected = gt_actions_norm * (1.0 - mask) + teacher_norm * mask
                diagnostics["teacher_traj_used_ratio"] = use_teacher.detach().float().mean()
                return selected, diagnostics
            warnings.warn(
                "teacher_if_better requested without teacher_score/gt_score; using teacher trajectory for all samples.",
                RuntimeWarning,
            )
            diagnostics["teacher_traj_used_ratio"] = gt_actions_norm.new_ones(())
            return teacher_norm, diagnostics

        if mode == "mix":
            mix = self.config.last_vla_teacher_traj_mix_start + (
                self.config.last_vla_teacher_traj_mix_end - self.config.last_vla_teacher_traj_mix_start
            ) * self._last_vla_progress()
            mix = min(max(float(mix), 0.0), 1.0)
            diagnostics["teacher_traj_mix"] = gt_actions_norm.new_tensor(mix)
            diagnostics["teacher_traj_used_ratio"] = gt_actions_norm.new_tensor(float(mix > 0.0))
            return gt_actions_norm * (1.0 - mix) + teacher_norm * mix, diagnostics

        raise ValueError(f"Unsupported last_vla_teacher_traj_mode={mode!r}.")

    def _last_vla_aux_weight(self, loss_name: str) -> float:
        base = {
            "last_vla_geometry_loss": self.config.last_vla_geometry_loss_weight,
            "last_vla_dynamic_loss": self.config.last_vla_dynamic_loss_weight,
            "last_vla_coarse_loss": self.config.last_vla_coarse_loss_weight,
            "last_vla_heading_loss": self.config.last_vla_heading_loss_weight,
            "last_vla_progress_loss": self.config.last_vla_progress_loss_weight,
            "last_vla_risk_loss": self.config.last_vla_risk_loss_weight,
            "last_vla_cot_consistency_loss": self.config.last_vla_cot_consistency_loss_weight,
        }[loss_name]
        if self.config.last_vla_stage not in {"progressive_sft_bottleneck", "progressive_sft_decoupled"}:
            return float(base)
        floors = {
            "last_vla_geometry_loss": self.config.last_vla_geometry_loss_floor,
            "last_vla_dynamic_loss": self.config.last_vla_dynamic_loss_floor,
            "last_vla_coarse_loss": self.config.last_vla_coarse_loss_floor,
            "last_vla_heading_loss": 0.0,
            "last_vla_progress_loss": self.config.last_vla_progress_loss_floor,
            "last_vla_risk_loss": 0.0,
            "last_vla_cot_consistency_loss": 0.0,
        }
        floor = float(floors[loss_name])
        if self.config.last_vla_aux_decay_epochs > 0:
            progress = min(
                max(float(self.config.current_train_epoch) / float(max(1, self.config.last_vla_aux_decay_epochs)), 0.0),
                1.0,
            )
        else:
            progress = self._last_vla_progress()
        return floor + max(float(base) - floor, 0.0) * (1.0 - progress)

    def _last_vla_effective_aux_weights(self, reference: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "geometry_weight_effective": reference.new_tensor(self._last_vla_aux_weight("last_vla_geometry_loss")),
            "dynamic_weight_effective": reference.new_tensor(self._last_vla_aux_weight("last_vla_dynamic_loss")),
            "coarse_weight_effective": reference.new_tensor(self._last_vla_aux_weight("last_vla_coarse_loss")),
            "progress_weight_effective": reference.new_tensor(self._last_vla_aux_weight("last_vla_progress_loss")),
        }

    def _last_vla_aux_loss(self, dit_context: Dict[str, Any], dtype: torch.dtype) -> torch.Tensor:
        loss = dit_context["last_vla_geometry_loss"].new_zeros(()).to(dtype=dtype)
        for key in (
            "last_vla_geometry_loss",
            "last_vla_dynamic_loss",
            "last_vla_coarse_loss",
            "last_vla_heading_loss",
            "last_vla_progress_loss",
            "last_vla_risk_loss",
            "last_vla_cot_consistency_loss",
        ):
            loss = loss + self._last_vla_aux_weight(key) * dit_context[key].to(dtype=dtype)
        return loss

    def _last_rd_aux_weight(self, loss_name: str) -> float:
        base = {
            "future_jepa_loss": self.config.future_jepa_loss_weight,
            "vggt_geometry_loss": self.config.vggt_geometry_loss_weight,
            "coarse_traj_loss": self.config.coarse_traj_loss_weight,
            "coarse_heading_loss": self.config.coarse_heading_loss_weight,
            "risk_loss": self.config.risk_loss_weight,
            "policy_kd_loss": self.config.policy_kd_loss_weight,
        }[loss_name]
        if self.config.last_rd_stage != "progressive_sft":
            return float(base)
        floors = {
            "future_jepa_loss": self.config.future_jepa_loss_floor,
            "vggt_geometry_loss": self.config.vggt_geometry_loss_floor,
            "coarse_traj_loss": self.config.coarse_traj_loss_floor,
            "coarse_heading_loss": 0.0,
            "risk_loss": self.config.risk_loss_floor,
            "policy_kd_loss": self.config.policy_kd_loss_weight,
        }
        floor = float(floors[loss_name])
        progress = min(max(float(self.config.current_train_epoch) / max(1.0, float(self.config.total_train_epochs)), 0.0), 1.0)
        return floor + max(float(base) - floor, 0.0) * (1.0 - progress)

    def _last_rd_aux_loss(self, dit_context: Dict[str, Any], dtype: torch.dtype) -> torch.Tensor:
        loss = dit_context["future_jepa_loss"].new_zeros(()).to(dtype=dtype)
        for key in (
            "future_jepa_loss",
            "vggt_geometry_loss",
            "coarse_traj_loss",
            "coarse_heading_loss",
            "risk_loss",
            "policy_kd_loss",
        ):
            loss = loss + self._last_rd_aux_weight(key) * dit_context[key].to(dtype=dtype)
        return loss

    @staticmethod
    def _policy_kd_action_input(action_input: BatchFeature) -> BatchFeature:
        safe_keys = ("his_traj", "history_trajectory", "status_feature", "high_command_one_hot", "action", "state")
        return BatchFeature(data={key: action_input[key] for key in safe_keys if key in action_input})

    def _denoise_model_output(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        dit_context: Dict[str, Any],
        action_input: BatchFeature,
    ) -> torch.Tensor:
        context_embeds = dit_context["context_tokens"]
        context_mean = dit_context["context_mean"]
        expert_step_condition = dit_context["expert_step_condition"]
        cot_condition_tokens = dit_context.get("cot_condition_tokens")
        his_traj_features = self.his_traj_encoder(
            action_input.his_traj.unsqueeze(1)
        ).repeat(1, self.config.action_horizon, 1)
        ego_status_features = self.ego_status_encoder(action_input.status_feature)

        action_features = self.action_encoder(noisy_actions, timesteps)
        if hasattr(self, 'position_embedding'):
            pos_ids = torch.arange(action_features.shape[1], device=noisy_actions.device)
            action_features = action_features + self.position_embedding(pos_ids)
        context_mean_features = context_mean.unsqueeze(1).repeat(1, self.config.action_horizon, 1)
        fused_input = self.fusion_projector(
            torch.cat((his_traj_features, context_mean_features, action_features), dim=2)
        )
        if expert_step_condition is not None:
            fused_input = fused_input + expert_step_condition.to(device=fused_input.device, dtype=fused_input.dtype)
        model_output = self.model(
            hidden_states=fused_input,
            encoder_hidden_states=context_embeds,
            conditioning_features=ego_status_features,
            timesteps=timesteps,
            cot_condition_tokens=cot_condition_tokens,
        )
        if self.config.use_last_vla:
            diagnostics = dit_context.setdefault("diagnostics", {})
            diagnostics["last_vla_cot_condition_delta_norm"] = getattr(
                self.model,
                "last_cot_condition_delta_norm",
                context_embeds.new_zeros(()),
            )
            diagnostics["last_vla_cot_branch_zero_init"] = getattr(
                self.model,
                "last_cot_branch_zero_init",
                context_embeds.new_zeros(()),
            )
        return self.action_decoder(model_output)

    def _x0_from_noise(self, noisy_actions: torch.Tensor, timesteps: torch.Tensor, pred_noise: torch.Tensor) -> torch.Tensor:
        return (
            self.extract(self.ddpm_sqrt_recip_alphas_cumprod, timesteps, noisy_actions.shape) * noisy_actions
            - self.extract(self.ddpm_sqrt_recipm1_alphas_cumprod, timesteps, noisy_actions.shape) * pred_noise
        )

    def _compute_policy_kd_loss(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        student_prediction: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.reference_a0_policy is None
            or self.config.policy_kd_mode == "none"
            or self.config.policy_kd_loss_weight <= 0.0
            or self.config.sampling_method == "flow"
        ):
            return student_prediction.new_zeros(())
        reference = self.reference_a0_policy.to(device=student_prediction.device)
        reference.eval()
        safe_action_input = self._policy_kd_action_input(action_input)
        with torch.no_grad():
            teacher_context = reference._prepare_dit_context(
                vl_features,
                safe_action_input,
                training=False,
            )
            teacher_prediction = reference._denoise_model_output(
                noisy_actions,
                timesteps,
                teacher_context,
                safe_action_input,
            )
        if self.config.policy_kd_mode == "noise":
            return F.mse_loss(student_prediction.float(), teacher_prediction.detach().float())
        if self.config.policy_kd_mode == "x0":
            student_x0 = self._x0_from_noise(noisy_actions, timesteps, student_prediction)
            teacher_x0 = reference._x0_from_noise(noisy_actions, timesteps, teacher_prediction)
            return F.mse_loss(student_x0.float(), teacher_x0.detach().float())
        raise ValueError(f"Unsupported policy_kd_mode: {self.config.policy_kd_mode}")


    def p_mean_variance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        index: torch.Tensor,
        context_embeds: torch.Tensor,
        his_traj_features: torch.Tensor,
        ego_status_features: torch.Tensor,
        deterministic: bool = True,
        context_mean: Optional[torch.Tensor] = None,
        expert_step_condition: Optional[torch.Tensor] = None,
        cot_condition_tokens: Optional[torch.Tensor] = None,
        vl_features: Optional[torch.Tensor] = None,
        action_input: Optional[BatchFeature] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the mean and log variance of the reverse process p(x_{t-1} | x_t).
        Also returns the predicted x0.
        """
        if (
            self.config.use_last_rd
            or self.config.use_last_vla
            or self.config.use_two_expert_slots
        ) and vl_features is not None and action_input is not None:
            latent_context = self._prepare_dit_context(
                vl_features,
                action_input,
                training=False,
                noisy_actions=x,
                diffusion_timestep=t,
                allow_target_tokens=False,
            )
            context_embeds = latent_context["context_tokens"]
            context_mean = latent_context["context_mean"]
            expert_step_condition = latent_context["expert_step_condition"]
            cot_condition_tokens = latent_context.get("cot_condition_tokens")
        model_dtype = next(self.model.parameters()).dtype
        x = x.to(model_dtype)
        action_features = self.action_encoder(x, t)
        if hasattr(self, 'position_embedding'):
            pos_ids = torch.arange(action_features.shape[1], device=x.device)
            action_features = action_features + self.position_embedding(pos_ids)

        context_mean_features = context_embeds.mean(1) if context_mean is None else context_mean
        context_mean_features = context_mean_features.unsqueeze(1).repeat(1, self.config.action_horizon, 1)
        fused_input = self.fusion_projector(
            torch.cat((his_traj_features, context_mean_features, action_features), dim=2)
        )
        if expert_step_condition is not None:
            fused_input = fused_input + expert_step_condition.to(device=fused_input.device, dtype=fused_input.dtype)

        model_output = self.model(
            hidden_states=fused_input,
            encoder_hidden_states=context_embeds,
            conditioning_features=ego_status_features,
            timesteps=t,
            cot_condition_tokens=cot_condition_tokens,
        )
        pred_noise = self.action_decoder(model_output)

        if self.config.sampling_method == 'ddpm':
            x_recon = self.extract(self.ddpm_sqrt_recip_alphas_cumprod, t, x.shape) * x - \
                      self.extract(self.ddpm_sqrt_recipm1_alphas_cumprod, t, x.shape) * pred_noise
        elif self.config.sampling_method == 'ddim':
            alpha_t = self.extract(self.ddim_alphas, index, x.shape)
            sqrt_one_minus_alpha_t = self.extract(self.ddim_sqrt_one_minus_alphas, index, x.shape)
            x_recon = (x - sqrt_one_minus_alpha_t * pred_noise) / (alpha_t**0.5)
        else:
             raise ValueError(f"p_mean_variance not supported for method: {self.config.sampling_method}")

        denoised_clip_value = getattr(self, 'denoised_clip_value', 1.0)
        x_recon.clamp_(-denoised_clip_value, denoised_clip_value)

        if self.config.sampling_method == 'ddpm':
            model_mean = self.extract(self.ddpm_mu_coef1, t, x.shape) * x_recon + \
                         self.extract(self.ddpm_mu_coef2, t, x.shape) * x
            model_log_variance = self.extract(self.ddpm_logvar_clipped, t, x.shape)
        elif self.config.sampling_method == 'ddim':
            alpha_prev = self.extract(self.ddim_alphas_prev, index, x.shape)
            
            ddim_denom = sqrt_one_minus_alpha_t.clamp_min(1e-6)

            pred_noise = (x - (alpha_t**0.5) * x_recon) / ddim_denom
            pred_noise = torch.where(
                sqrt_one_minus_alpha_t > 0,
                pred_noise,
                torch.zeros_like(pred_noise),
            )

            eps_clip_value = getattr(self, 'eps_clip_value', None)
            if eps_clip_value is not None:
                pred_noise.clamp_(-eps_clip_value, eps_clip_value)

            if deterministic:
                sigma = torch.zeros_like(alpha_t)
            else:
                etas = self.eta(x).unsqueeze(1)
                sigma_variance = (
                    (1 - alpha_prev)
                    / (1 - alpha_t).clamp_min(1e-6)
                    * (1 - alpha_t / alpha_prev)
                )
                sigma_variance = torch.where(
                    (1 - alpha_t) > 0,
                    sigma_variance,
                    torch.zeros_like(sigma_variance),
                )
                sigma = (etas * sigma_variance.clamp_min(0).sqrt()).clamp_(min=1e-10)

            pred_dir_xt = (1.0 - alpha_prev - sigma**2).clamp(min=0).sqrt() * pred_noise
            model_mean = (alpha_prev**0.5) * x_recon + pred_dir_xt
            model_log_variance = torch.log(sigma**2 + 1e-20)

        return model_mean, model_log_variance, x_recon

    def forward(self, vl_features: torch.Tensor, action_input: BatchFeature) -> BatchFeature:
        """
        Computes the training loss for a given batch.

        Args:
            vl_features (torch.Tensor): The vision-language features from the backbone.
            action_input (BatchFeature): A batch containing ground truth actions and other conditioning.

        Returns:
            BatchFeature: A batch containing the computed loss.
        """
        gt_actions = self.norm_odo(action_input.action)
        allow_target_tokens = self.training or bool(action_input.get("_allow_target_tokens_for_loss", False))
        if not allow_target_tokens:
            self._warn_if_expert_targets_present(action_input, "forward")

        if self.config.use_last_vla:
            selected_target_norm, target_diagnostics = self._select_last_vla_training_target(action_input, gt_actions)
            zero = gt_actions.new_zeros(())

            if self.config.last_vla_stage == "cot_alignment" and float(self.config.diffusion_loss_weight) == 0.0:
                dit_context = self._prepare_dit_context(
                    vl_features,
                    action_input,
                    training=self.training,
                    target_action_norm=selected_target_norm,
                    allow_target_tokens=allow_target_tokens,
                )
                dit_context["diagnostics"].update(target_diagnostics)
                dit_context["diagnostics"]["residual_alpha"] = selected_target_norm.new_tensor(0.0)
                dit_context["diagnostics"].update(self._last_vla_effective_aux_weights(zero))
                dit_context["policy_kd_loss"] = zero
                loss = self._last_vla_aux_loss(dit_context, gt_actions.dtype)
                return self._format_training_output(loss, zero, zero, zero, dit_context)

            diffusion_target, residual_alpha, _, residual_diagnostics = self._last_vla_diffusion_target_info(
                selected_target_norm,
                training=self.training,
                action_input=action_input,
            )

            if self.config.sampling_method == "flow":
                noise = torch.randn_like(diffusion_target)
                t_cont = self.sample_time(diffusion_target.shape[0], device=diffusion_target.device, dtype=diffusion_target.dtype)
                t_cont_reshaped = t_cont[:, None, None]
                noisy_actions = (1 - t_cont_reshaped) * noise + t_cont_reshaped * diffusion_target
                velocity_target = diffusion_target - noise
                t_discrete = (t_cont * self.num_timestep_buckets).long()
                dit_context = self._prepare_dit_context(
                    vl_features,
                    action_input,
                    training=self.training,
                    noisy_actions=noisy_actions,
                    diffusion_timestep=t_discrete,
                    target_action_norm=selected_target_norm,
                    allow_target_tokens=allow_target_tokens,
                )
                dit_context["diagnostics"]["residual_alpha"] = diffusion_target.new_tensor(float(residual_alpha))
                dit_context["diagnostics"].update(residual_diagnostics)
                pred_velocity = self._denoise_model_output(noisy_actions, t_discrete, dit_context, action_input)
                diffusion_loss = F.mse_loss(pred_velocity, velocity_target, reduction="mean")
                policy_kd_loss = diffusion_loss.new_zeros(())
            else:
                noise = torch.randn_like(diffusion_target)
                t_discrete = self.sample_time(diffusion_target.shape[0], device=diffusion_target.device, dtype=diffusion_target.dtype)
                noisy_actions = (
                    self.extract(self.ddpm_sqrt_alphas_cumprod, t_discrete, diffusion_target.shape) * diffusion_target
                    + self.extract(self.ddpm_sqrt_one_minus_alphas_cumprod, t_discrete, diffusion_target.shape) * noise
                )
                dit_context = self._prepare_dit_context(
                    vl_features,
                    action_input,
                    training=self.training,
                    noisy_actions=noisy_actions,
                    diffusion_timestep=t_discrete,
                    target_action_norm=selected_target_norm,
                    allow_target_tokens=allow_target_tokens,
                )
                dit_context["diagnostics"]["residual_alpha"] = diffusion_target.new_tensor(float(residual_alpha))
                dit_context["diagnostics"].update(residual_diagnostics)
                pred_noise = self._denoise_model_output(noisy_actions, t_discrete, dit_context, action_input)
                diffusion_loss = F.mse_loss(pred_noise, noise, reduction="mean")
                policy_kd_loss = self._compute_policy_kd_loss(vl_features, action_input, noisy_actions, t_discrete, pred_noise)

            dit_context["policy_kd_loss"] = policy_kd_loss.to(dtype=diffusion_loss.dtype)
            dit_context["diagnostics"].update(target_diagnostics)
            dit_context["diagnostics"].update(self._last_vla_effective_aux_weights(diffusion_loss))
            loss = (
                float(self.config.diffusion_loss_weight) * diffusion_loss
                + self._last_vla_aux_loss(dit_context, diffusion_loss.dtype)
                + float(self.config.policy_kd_loss_weight) * policy_kd_loss.to(dtype=diffusion_loss.dtype)
            )
            return self._format_training_output(loss, diffusion_loss, zero, zero, dit_context)

        if (
            self.config.use_last_rd
            and self.config.last_rd_stage == "stage1_5"
            and float(self.config.diffusion_loss_weight) == 0.0
        ):
            dit_context = self._prepare_dit_context(
                vl_features,
                action_input,
                training=self.training,
                allow_target_tokens=allow_target_tokens,
            )
            diffusion_loss = gt_actions.new_zeros(())
            jepa_alignment_loss = dit_context["jepa_alignment_loss"].to(dtype=diffusion_loss.dtype)
            vggt_alignment_loss = dit_context["vggt_alignment_loss"].to(dtype=diffusion_loss.dtype)
            policy_kd_loss = diffusion_loss
            dit_context["policy_kd_loss"] = policy_kd_loss
            loss = (
                self._stream_alignment_weight("jepa") * jepa_alignment_loss
                + self._stream_alignment_weight("vggt") * vggt_alignment_loss
                + self._last_rd_aux_loss(dit_context, diffusion_loss.dtype)
            )
            return self._format_training_output(loss, diffusion_loss, jepa_alignment_loss, vggt_alignment_loss, dit_context)

        diffusion_target, residual_alpha, _, residual_diagnostics = self._last_vla_diffusion_target_info(
            gt_actions,
            training=self.training,
            action_input=action_input,
        )

        if self.config.sampling_method == 'flow':
            noise = torch.randn_like(diffusion_target)
            t_cont = self.sample_time(diffusion_target.shape[0], device=diffusion_target.device, dtype=diffusion_target.dtype)
            t_cont_reshaped = t_cont[:, None, None]
            
            noisy_actions = (1 - t_cont_reshaped) * noise + t_cont_reshaped * diffusion_target
            velocity_target = diffusion_target - noise
            t_discrete = (t_cont * self.num_timestep_buckets).long()
            dit_context = self._prepare_dit_context(
                vl_features,
                action_input,
                training=self.training,
                noisy_actions=noisy_actions,
                diffusion_timestep=t_discrete,
                allow_target_tokens=allow_target_tokens,
            )
            dit_context["diagnostics"]["residual_alpha"] = diffusion_target.new_tensor(float(residual_alpha))
            dit_context["diagnostics"].update(residual_diagnostics)
            pred_velocity = self._denoise_model_output(noisy_actions, t_discrete, dit_context, action_input)
            diffusion_loss = F.mse_loss(pred_velocity, velocity_target, reduction='mean')
            policy_kd_loss = diffusion_loss.new_zeros(())
        else: 
            noise = torch.randn_like(diffusion_target)
            t_discrete = self.sample_time(diffusion_target.shape[0], device=diffusion_target.device, dtype=diffusion_target.dtype)
            
            noisy_actions = (
                self.extract(self.ddpm_sqrt_alphas_cumprod, t_discrete, diffusion_target.shape) * diffusion_target +
                self.extract(self.ddpm_sqrt_one_minus_alphas_cumprod, t_discrete, diffusion_target.shape) * noise
            )
            dit_context = self._prepare_dit_context(
                vl_features,
                action_input,
                training=self.training,
                noisy_actions=noisy_actions,
                diffusion_timestep=t_discrete,
                allow_target_tokens=allow_target_tokens,
            )
            dit_context["diagnostics"]["residual_alpha"] = diffusion_target.new_tensor(float(residual_alpha))
            dit_context["diagnostics"].update(residual_diagnostics)
            pred_noise = self._denoise_model_output(noisy_actions, t_discrete, dit_context, action_input)
            diffusion_loss = F.mse_loss(pred_noise, noise, reduction='mean')
            policy_kd_loss = self._compute_policy_kd_loss(vl_features, action_input, noisy_actions, t_discrete, pred_noise)
        dit_context["policy_kd_loss"] = policy_kd_loss.to(dtype=diffusion_loss.dtype)

        jepa_alignment_loss = dit_context["jepa_alignment_loss"].to(dtype=diffusion_loss.dtype)
        vggt_alignment_loss = dit_context["vggt_alignment_loss"].to(dtype=diffusion_loss.dtype)
        loss = (
            float(self.config.diffusion_loss_weight) * diffusion_loss
            + self._stream_alignment_weight("jepa") * jepa_alignment_loss
            + self._stream_alignment_weight("vggt") * vggt_alignment_loss
            + self._last_rd_aux_loss(dit_context, diffusion_loss.dtype)
        )

        return self._format_training_output(loss, diffusion_loss, jepa_alignment_loss, vggt_alignment_loss, dit_context)

    def _format_training_output(
        self,
        loss: torch.Tensor,
        diffusion_loss: torch.Tensor,
        jepa_alignment_loss: torch.Tensor,
        vggt_alignment_loss: torch.Tensor,
        dit_context: Dict[str, Any],
    ) -> BatchFeature:
        output = {
            "loss": loss,
            "diffusion_loss": diffusion_loss,
            "jepa_alignment_loss": jepa_alignment_loss,
            "vggt_alignment_loss": vggt_alignment_loss,
            "future_jepa_loss": dit_context["future_jepa_loss"].to(dtype=loss.dtype),
            "vggt_geometry_loss": dit_context["vggt_geometry_loss"].to(dtype=loss.dtype),
            "coarse_traj_loss": dit_context["coarse_traj_loss"].to(dtype=loss.dtype),
            "coarse_heading_loss": dit_context["coarse_heading_loss"].to(dtype=loss.dtype),
            "risk_loss": dit_context["risk_loss"].to(dtype=loss.dtype),
            "policy_kd_loss": dit_context["policy_kd_loss"].to(dtype=loss.dtype),
            "last_vla_geometry_loss": dit_context.get("last_vla_geometry_loss", loss.detach().new_tensor(0.0)).to(dtype=loss.dtype),
            "last_vla_dynamic_loss": dit_context.get("last_vla_dynamic_loss", loss.detach().new_tensor(0.0)).to(dtype=loss.dtype),
            "last_vla_coarse_loss": dit_context.get("last_vla_coarse_loss", loss.detach().new_tensor(0.0)).to(dtype=loss.dtype),
            "last_vla_heading_loss": dit_context.get("last_vla_heading_loss", loss.detach().new_tensor(0.0)).to(dtype=loss.dtype),
            "last_vla_progress_loss": dit_context.get("last_vla_progress_loss", loss.detach().new_tensor(0.0)).to(dtype=loss.dtype),
            "last_vla_risk_loss": dit_context.get("last_vla_risk_loss", loss.detach().new_tensor(0.0)).to(dtype=loss.dtype),
            "last_vla_cot_consistency_loss": dit_context.get("last_vla_cot_consistency_loss", loss.detach().new_tensor(0.0)).to(dtype=loss.dtype),
            "last_vla_policy_kd_loss": dit_context["policy_kd_loss"].to(dtype=loss.dtype) if self.config.use_last_vla else loss.detach().new_tensor(0.0),
        }
        diagnostics = dit_context.get("diagnostics", {})
        if diagnostics:
            branch_weights = diagnostics.get("branch_weights")
            output["jepa_gate_value"] = diagnostics.get("jepa_gate", loss.detach().new_tensor(0.0)).to(device=loss.device)
            output["vggt_gate_value"] = diagnostics.get("vggt_gate", loss.detach().new_tensor(0.0)).to(device=loss.device)
            output["expert_context_scale"] = diagnostics.get("expert_context_scale", loss.detach().new_tensor(0.0)).to(device=loss.device)
            output["expert_horizon_residual_scale"] = diagnostics.get("expert_horizon_residual_scale", loss.detach().new_tensor(0.0)).to(device=loss.device)
            if branch_weights is not None:
                branch_weights = branch_weights.to(device=loss.device, dtype=loss.dtype)
                output["branch_weight_vlm"] = branch_weights[0]
                output["branch_weight_jepa"] = branch_weights[1]
                output["branch_weight_vggt"] = branch_weights[2]
            for name in ("vlm", "dynamic", "geometry", "ego", "risk"):
                key = f"last_rd_group_weight_{name}"
                if key in diagnostics:
                    output[key] = diagnostics[key].to(device=loss.device, dtype=loss.dtype)
            if self.config.use_last_rd:
                for key in ("last_rd_token_norms", "coarse_traj_l1", "future_jepa_loss_raw", "geometry_mode_code"):
                    if key in diagnostics and isinstance(diagnostics[key], torch.Tensor):
                        output[f"last_rd_{key}" if key == "geometry_mode_code" else key] = diagnostics[key].to(
                            device=loss.device,
                            dtype=loss.dtype,
                        )
            for key in (
                "two_expert_condition_enabled",
                "two_expert_corruption_mode_code",
                "two_expert_raw_vlm_context_used",
                "two_expert_h_dyn_norm",
                "two_expert_h_geo_norm",
                "two_expert_f_dyn_norm",
                "two_expert_f_geo_norm",
                "two_expert_dyn_expert_delta_norm",
                "two_expert_geo_expert_delta_norm",
                "two_expert_zero_init_dyn",
                "two_expert_zero_init_geo",
                "teacher_traj_used_ratio",
                "teacher_traj_mix",
                "teacher_score_mean",
                "gt_score_mean",
                "geometry_mode_code",
                "cot_token_norm",
                "cot_bottleneck_active",
                "raw_vlm_context_used",
                "coarse_traj_l1",
                "dynamic_loss_raw",
                "geometry_loss_raw",
                "residual_alpha",
                "residual_anchor_source_code",
                "residual_anchor_missing",
                "residual_anchor_norm",
                "residual_anchor_l1_to_reference",
                "residual_target_norm",
                "geometry_weight_effective",
                "dynamic_weight_effective",
                "coarse_weight_effective",
                "progress_weight_effective",
                "last_vla_condition_mode_code",
                "last_vla_no_global_gate",
                "raw_vlm_token_count",
                "cot_condition_token_count",
                "cot_scene_norm",
                "cot_geometry_norm",
                "cot_dynamic_norm",
                "cot_fusion_norm",
                "cot_action_norm",
                "geometry_dynamic_parallel",
                "last_vla_cot_condition_norm",
                "last_vla_cot_condition_delta_norm",
                "last_vla_cot_branch_zero_init",
                "last_vla_raw_vlm_base_context_norm",
                "last_vla_context_mean_cot_residual_norm",
                "last_vla_horizon_cot_residual_norm",
            ):
                if key in diagnostics and isinstance(diagnostics[key], torch.Tensor):
                    output_key = key if key.startswith(("last_vla_", "two_expert_")) else f"last_vla_{key}"
                    output[output_key] = diagnostics[key].to(device=loss.device, dtype=loss.dtype)
        return BatchFeature(data=output)

    def get_action(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        init_actions: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> BatchFeature:
        """
        Generates action trajectories via the configured sampling method.

        This method strictly preserves the original logic for each sampler,
        including specific clipping and noise handling for DDPM and DDIM.

        Args:
            vl_features (torch.Tensor): Vision-language features from the backbone.
            action_input (BatchFeature): Input containing conditioning features like
                historical trajectory and ego status.
            init_actions (Optional[torch.Tensor]): An initial trajectory to start
                the denoising from. If None, starts from pure noise.
            deterministic (bool): If True, DDIM sampling will be deterministic (eta=0).

        Returns:
            BatchFeature: A batch containing the final predicted trajectory.
        """
        self._warn_if_expert_targets_present(action_input, "get_action")
        dit_context = self._prepare_dit_context(vl_features, action_input, training=False, allow_target_tokens=False)
        context_embeds = dit_context["context_tokens"]
        context_mean = dit_context["context_mean"]
        expert_step_condition = dit_context["expert_step_condition"]
        cot_condition_tokens = dit_context.get("cot_condition_tokens")
        coarse_prior_norm = None
        if self.config.use_last_vla:
            coarse_prior_norm = dit_context["last_vla_output"].coarse_traj_norm.detach()
        
        history_embeds = self.his_traj_encoder(
            action_input.his_traj.unsqueeze(1)
        ).repeat(1, self.config.action_horizon, 1)

        ego_embeds = self.ego_status_encoder(
            action_input.status_feature
        )

        B, D = context_embeds.shape[0], self.config.action_dim
        device, dtype = context_embeds.device, context_embeds.dtype
        
        current_actions = init_actions if init_actions is not None else torch.randn(
            (B, self.config.action_horizon, D), device=device, dtype=dtype
        )

        if self.config.sampling_method == 'flow':
            dt = 1.0 / self.config.num_inference_steps
            for step in range(self.config.num_inference_steps):
                idx = int(step / self.config.num_inference_steps * self.config.flow_cfg.num_timestep_buckets)
                t = torch.full((B,), idx, device=device, dtype=torch.long)
                if self.config.use_last_rd or self.config.use_last_vla or self.config.use_two_expert_slots:
                    dit_context = self._prepare_dit_context(
                        vl_features,
                        action_input,
                        training=False,
                        noisy_actions=current_actions,
                        diffusion_timestep=t,
                        allow_target_tokens=False,
                    )
                    context_embeds = dit_context["context_tokens"]
                    context_mean = dit_context["context_mean"]
                    expert_step_condition = dit_context["expert_step_condition"]
                    cot_condition_tokens = dit_context.get("cot_condition_tokens")

                action_features = self.action_encoder(current_actions, t)
                if hasattr(self, 'position_embedding'):
                    action_features += self.position_embedding(torch.arange(self.config.action_horizon, device=device))
                
                context_mean_features = context_mean.unsqueeze(1).repeat(1, self.config.action_horizon, 1)
                fused_input = self.fusion_projector(
                    torch.cat((history_embeds, context_mean_features, action_features), dim=2)
                )
                if expert_step_condition is not None:
                    fused_input = fused_input + expert_step_condition.to(device=fused_input.device, dtype=fused_input.dtype)
                
                model_output = self.model(
                    fused_input,
                    context_embeds,
                    ego_embeds,
                    t,
                    cot_condition_tokens=cot_condition_tokens,
                )
                pred = self.action_decoder(model_output)
                
                pred_flow = pred.chunk(2, dim=-1)[0] if self.config.flow_cfg.mean_variance_net else pred
                current_actions = current_actions + dt * pred_flow

        elif self.config.sampling_method == 'ddpm':
            step_size = self.config.ddpm_cfg.num_train_timesteps // self.config.num_inference_steps
            timesteps_to_iterate = list(reversed(range(0, self.config.ddpm_cfg.num_train_timesteps, step_size)))
            
            for i, t_int in enumerate(timesteps_to_iterate):
                t_batch = self.make_timesteps(B, t_int, device)
                index_batch = self.make_timesteps(B, i, device)

                mean, logvar, _ = self.p_mean_variance(
                    current_actions, t_batch, index_batch, context_embeds, history_embeds, ego_embeds, deterministic,
                    context_mean=context_mean,
                    expert_step_condition=expert_step_condition,
                    cot_condition_tokens=cot_condition_tokens,
                    vl_features=vl_features,
                    action_input=action_input,
                )

                noise_sample = torch.randn_like(current_actions)
                std = torch.exp(0.5 * logvar)

                std = std.to(dtype)

                if t_int == 0:
                    std.zero_()
                else:
                    std = torch.clamp(std, min=1e-3)

                if hasattr(self, 'eval_randn_clip_value') and self.eval_randn_clip_value is not None:
                    noise_sample.clamp_(-self.eval_randn_clip_value, self.eval_randn_clip_value)

                current_actions = mean + std * noise_sample
                
                if hasattr(self, 'final_action_clip_value') and self.final_action_clip_value is not None and i == len(timesteps_to_iterate) - 1:
                    current_actions.clamp_(-self.final_action_clip_value, self.final_action_clip_value)

        elif self.config.sampling_method == 'ddim':
            eval_min_sampling_denoising_std = getattr(self, 'eval_min_sampling_denoising_std', 0.0001)
            eval_randn_clip_value = getattr(self, 'eval_randn_clip_value', 1.0)
            for i in range(self.ddim_steps):
                t_batch = self.make_timesteps(B, self.ddim_t[i], device)
                index_batch = self.make_timesteps(B, i, device)

                mean, logvar, _ = self.p_mean_variance(
                    current_actions, t_batch, index_batch, context_embeds, history_embeds, ego_embeds, deterministic,
                    context_mean=context_mean,
                    expert_step_condition=expert_step_condition,
                    cot_condition_tokens=cot_condition_tokens,
                    vl_features=vl_features,
                    action_input=action_input,
                )

                std = torch.exp(0.5 * logvar)

                std = std.to(dtype)

                noise_sample = torch.randn_like(current_actions)
                
                if deterministic:
                    std.zero_()
                else:
                    std = std.clamp(min=eval_min_sampling_denoising_std)
                
                noise_sample.clamp_(-eval_randn_clip_value, eval_randn_clip_value)
                
                current_actions = mean + std * noise_sample
        else:
            raise ValueError(f"Unsupported sampling method: {self.config.sampling_method}")

        final_action_clip_value = getattr(self, 'final_action_clip_value', 1.0)
        if final_action_clip_value is not None:
            current_actions.clamp_(-final_action_clip_value, final_action_clip_value)

        residual_alpha = self._last_vla_residual_alpha(training=False)
        residual_anchor_norm = None
        final_norm = current_actions
        if residual_alpha != 0.0:
            residual_anchor_norm, _ = self._last_vla_residual_anchor_norm(
                action_input,
                current_actions,
                required=True,
            )
            assert residual_anchor_norm is not None
            final_norm = current_actions + float(residual_alpha) * residual_anchor_norm.to(current_actions)

        output_data: Dict[str, torch.Tensor] = {}
        if coarse_prior_norm is not None:
            output_data["pred_coarse_traj"] = self.denorm_odo(coarse_prior_norm.to(current_actions))
        if residual_anchor_norm is not None:
            output_data["pred_residual_norm"] = current_actions
            output_data["pred_vlm_text_anchor_traj"] = self.denorm_odo(residual_anchor_norm.to(current_actions))
            output_data["pred_residual_anchor_alpha"] = current_actions.new_tensor(float(residual_alpha))

        final_action_clip_value = getattr(self, 'final_action_clip_value', 1.0)
        if final_action_clip_value is not None:
            final_norm = final_norm.clamp(-final_action_clip_value, final_action_clip_value)

        final_actions = self.denorm_odo(final_norm)
        output_data["pred_traj"] = final_actions

        return BatchFeature(data=output_data)

    def sample_chain(
        self,
        vl_features: torch.Tensor,
        his_traj_features: torch.Tensor,
        ego_status_features: torch.Tensor,
        init_actions: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        action_input: Optional[BatchFeature] = None,
        allow_target_tokens: Optional[bool] = None,
    ):
        """
        Generates the full denoising chain and the final trajectory.
        This method reuses the logic from get_action but stores intermediate steps.

        Args:
            vl_features (torch.Tensor): Vision-language features from the backbone.
            his_traj_features (torch.Tensor): Encoded historical trajectory features.
            ego_status_features (torch.Tensor): Encoded ego status features.
            init_actions (Optional[torch.Tensor]): An initial trajectory to start from.
                If None, starts from pure noise.
            deterministic (bool): If True, DDIM sampling will be deterministic.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The full denoising chain as a tensor of shape (B, K+1, H, D).
                - The final, denormalized trajectory of shape (B, H, D).
        """
        context_allow_target_tokens = self.training if allow_target_tokens is None else bool(allow_target_tokens)
        if not context_allow_target_tokens:
            self._warn_if_expert_targets_present(action_input, "sample_chain")
        dit_context = self._prepare_dit_context(
            vl_features,
            action_input,
            training=self.training,
            allow_target_tokens=context_allow_target_tokens,
        )
        context_embeds = dit_context["context_tokens"]
        context_mean = dit_context["context_mean"]
        expert_step_condition = dit_context["expert_step_condition"]
        cot_condition_tokens = dit_context.get("cot_condition_tokens")
        B, D = context_embeds.shape[0], self.config.action_dim
        device, dtype = context_embeds.device, context_embeds.dtype
        
        his_traj_features = self.his_traj_encoder(
            his_traj_features.unsqueeze(1)               
        ).repeat(1, self.config.action_horizon, 1) 

        ego_status_features = self.ego_status_encoder(
            ego_status_features       
        )

        current_actions = init_actions if init_actions is not None else torch.randn(
            (B, self.config.action_horizon, D), device=device, dtype=dtype
        )
        denoising_chain = [current_actions.clone()]

        if self.config.sampling_method == 'flow':
            dt = 1.0 / self.config.num_inference_steps
            for step in range(self.config.num_inference_steps):
                idx = int(step / self.config.num_inference_steps * self.config.flow_cfg.num_timestep_buckets)
                t_batch = torch.full((B,), idx, device=device, dtype=torch.long)
                if (
                    self.config.use_last_rd
                    or self.config.use_last_vla
                    or self.config.use_two_expert_slots
                ) and action_input is not None:
                    dit_context = self._prepare_dit_context(
                        vl_features,
                        action_input,
                        training=self.training,
                        noisy_actions=current_actions,
                        diffusion_timestep=t_batch,
                        allow_target_tokens=context_allow_target_tokens,
                    )
                    context_embeds = dit_context["context_tokens"]
                    context_mean = dit_context["context_mean"]
                    expert_step_condition = dit_context["expert_step_condition"]
                    cot_condition_tokens = dit_context.get("cot_condition_tokens")
                
                action_features = self.action_encoder(current_actions, t_batch)
                if hasattr(self, 'position_embedding'):
                    action_features += self.position_embedding(torch.arange(self.config.action_horizon, device=device))
                
                context_mean_features = context_mean.unsqueeze(1).repeat(1, self.config.action_horizon, 1)
                fused_input = self.fusion_projector(
                    torch.cat((his_traj_features, context_mean_features, action_features), dim=2)
                )
                if expert_step_condition is not None:
                    fused_input = fused_input + expert_step_condition.to(device=fused_input.device, dtype=fused_input.dtype)
                
                model_output = self.model(
                    fused_input,
                    context_embeds,
                    ego_status_features,
                    t_batch,
                    cot_condition_tokens=cot_condition_tokens,
                )
                pred = self.action_decoder(model_output)
                
                pred_flow = pred.chunk(2, dim=-1)[0] if self.config.flow_cfg.mean_variance_net else pred
                current_actions = current_actions + dt * pred_flow
                denoising_chain.append(current_actions.clone())

        elif self.config.sampling_method in ['ddpm', 'ddim']:
            if self.config.sampling_method == 'ddpm':
                step_size = self.config.ddpm_cfg.num_train_timesteps // self.config.num_inference_steps
                timesteps = list(reversed(range(0, self.config.ddpm_cfg.num_train_timesteps, step_size)))
            else: 
                timesteps = self.ddim_t
            
            for i, t_int in enumerate(timesteps):
                t_batch = self.make_timesteps(B, t_int, device)
                index_batch = self.make_timesteps(B, i, device) if self.config.sampling_method == 'ddim' else t_batch

                mean, logvar, _ = self.p_mean_variance(
                    current_actions, t_batch, index_batch, context_embeds, his_traj_features, ego_status_features, deterministic,
                    context_mean=context_mean,
                    expert_step_condition=expert_step_condition,
                    cot_condition_tokens=cot_condition_tokens,
                    vl_features=vl_features if action_input is not None else None,
                    action_input=action_input,
                )

                std = torch.exp(0.5 * logvar).to(dtype)
                noise_sample = torch.randn_like(current_actions)

                if self.config.sampling_method == 'ddim':
                    if deterministic:
                        std.zero_()
                    else:
                        std = std.clamp(min=self.min_sampling_denoising_std)
                else: # ddpm
                    if deterministic and t_int == 0:
                        std = torch.zeros_like(std)
                    elif deterministic:
                        std = std.clamp(min=1e-3)
                    else:
                        std = std.clamp(min=self.min_sampling_denoising_std)
                
                if hasattr(self, 'randn_clip_value') and self.randn_clip_value is not None:
                    noise_sample = noise_sample.clamp_(-self.randn_clip_value, self.randn_clip_value)

                current_actions = mean + std * noise_sample
                
                if i == len(timesteps) - 1 and hasattr(self, 'final_action_clip_value') and self.final_action_clip_value is not None:
                    current_actions = current_actions.clamp_(-self.final_action_clip_value, self.final_action_clip_value)
                
                denoising_chain.append(current_actions.clone())
        else:
            raise ValueError(f"Unsupported sampling method: {self.config.sampling_method}")

        residual_alpha = self._last_vla_residual_alpha(training=self.training)
        final_norm = current_actions
        if residual_alpha != 0.0:
            residual_anchor_norm, _ = self._last_vla_residual_anchor_norm(
                action_input,
                current_actions,
                required=True,
            )
            assert residual_anchor_norm is not None
            final_norm = current_actions + float(residual_alpha) * residual_anchor_norm.to(current_actions)
            final_action_clip_value = getattr(self, 'final_action_clip_value', 1.0)
            if final_action_clip_value is not None:
                final_norm = final_norm.clamp(-final_action_clip_value, final_action_clip_value)
        final_actions = self.denorm_odo(final_norm)
        chain_tensor = torch.stack(denoising_chain, dim=1)
        
        return chain_tensor.detach(), final_actions.detach()

    def get_logprobs(
        self,
        vl_features: torch.Tensor,
        his_traj_features: torch.Tensor,
        ego_status_features: torch.Tensor,
        chains: torch.Tensor,
        deterministic: bool = False,
        action_input: Optional[BatchFeature] = None,
    ) -> torch.Tensor:
        """Calculates the log probability of a full denoising chain."""
        if not self.training:
            self._warn_if_expert_targets_present(action_input, "get_logprobs")
        B, K1, H, D = chains.shape
        num_denoising_steps = K1 - 1
        
        dit_context = self._prepare_dit_context(vl_features, action_input, training=self.training)
        context_embeds = dit_context["context_tokens"]
        context_mean = dit_context["context_mean"]
        expert_step_condition = dit_context["expert_step_condition"]
        cot_condition_tokens = dit_context.get("cot_condition_tokens")

        his_traj_features = self.his_traj_encoder(
            his_traj_features.unsqueeze(1)          
        ).repeat(1, self.config.action_horizon, 1) 

        ego_status_features = self.ego_status_encoder(
            ego_status_features      
        )

        conditioning_embeds = {
            'context_embeds': context_embeds,
            'his_traj_features': his_traj_features,
            'ego_status_features': ego_status_features,
            'context_mean': context_mean,
        }
        if expert_step_condition is not None:
            conditioning_embeds['expert_step_condition'] = expert_step_condition
        if cot_condition_tokens is not None:
            conditioning_embeds['cot_condition_tokens'] = cot_condition_tokens
        
        batched_conditioning = {}
        for key, value in conditioning_embeds.items():
            batched_conditioning[key] = value.unsqueeze(1).repeat(
                1, num_denoising_steps, *(1,) * (value.ndim - 1)
            ).flatten(0, 1)

        if self.config.sampling_method == 'ddim':
            t_single = self.ddim_t[-num_denoising_steps:]
            indices_single = torch.arange(
                start=self.ddim_steps - num_denoising_steps,
                end=self.ddim_steps,
                device=chains.device
            )
            indices_batch = indices_single.repeat(B)
        else:
            t_single = torch.arange(start=K1 - 2, end=-1, step=-1, device=chains.device)
            indices_batch = None 
            
        t_batch = t_single.repeat(B)

        x_t = chains[:, :-1].reshape(-1, H, D)
        x_t_minus_1 = chains[:, 1:].reshape(-1, H, D)

        mean, logvar, _ = self.p_mean_variance(
            x_t, t_batch, indices_batch,
            batched_conditioning['context_embeds'],
            batched_conditioning['his_traj_features'],
            batched_conditioning['ego_status_features'],
            deterministic=deterministic,
            context_mean=batched_conditioning['context_mean'],
            expert_step_condition=batched_conditioning.get('expert_step_condition'),
            cot_condition_tokens=batched_conditioning.get('cot_condition_tokens'),
        )

        std = torch.exp(0.5 * logvar).clamp(min=self.min_logprob_denoising_std)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(x_t_minus_1)
        
        return log_prob

    def _sync_behavior_policy(self):
        """Synchronizes the frozen GRPO behavior policy with the current policy."""
        if not hasattr(self, "behavior_policy"):
            return

        behavior_state = self.behavior_policy.state_dict()
        current_state = self.state_dict()
        filtered_state = {
            key: current_state[key].detach()
            for key, value in behavior_state.items()
            if key in current_state and current_state[key].shape == value.shape
        }
        self.behavior_policy.load_state_dict(filtered_state, strict=False)
        self.behavior_policy.eval()
        for param in self.behavior_policy.parameters():
            param.requires_grad = False

    @staticmethod
    def _to_python_float(value: Any, default: float) -> float:
        if value is None:
            return default
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return default
            value = value.detach().reshape(-1)[0].cpu().item()
        elif isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            value = value.reshape(-1)[0].item()
        elif isinstance(value, np.generic):
            value = value.item()
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _pdm_component_aliases() -> Dict[str, tuple[str, ...]]:
        return {
            "pdms": ("score", "pdm_score", "pdms"),
            "no_at_fault_collisions": ("no_at_fault_collisions", "nc"),
            "drivable_area_compliance": ("drivable_area_compliance", "dac"),
            "time_to_collision_within_bound": ("time_to_collision_within_bound", "ttc"),
            "ego_progress": ("ego_progress", "ep", "progress"),
            "history_comfort": ("history_comfort", "comfort", "comfortable"),
            "lane_keeping": ("lane_keeping", "lane_keeping_compliance"),
            "driving_direction_compliance": ("driving_direction_compliance", "ddc"),
            "traffic_light_compliance": ("traffic_light_compliance", "tlc"),
        }

    def _pdm_result_to_raw_dict(self, pdm_result) -> Dict[str, Any]:
        while isinstance(pdm_result, tuple) and len(pdm_result) > 0:
            pdm_result = pdm_result[0]

        if is_dataclass(pdm_result) and not isinstance(pdm_result, type):
            return asdict(pdm_result)
        elif isinstance(pdm_result, dict):
            return dict(pdm_result)
        elif hasattr(pdm_result, "iloc") and hasattr(pdm_result, "columns"):
            return pdm_result.iloc[0].to_dict() if len(pdm_result) > 0 else {}
        elif hasattr(pdm_result, "_asdict"):
            return pdm_result._asdict()
        elif hasattr(pdm_result, "__dict__"):
            return {
                key: value
                for key, value in vars(pdm_result).items()
                if not key.startswith("_")
            }
        return {}

    def _extract_pdm_components(
        self,
        pdm_result,
        *,
        strict_submetrics: bool = False,
        required_submetrics: Optional[tuple[str, ...]] = None,
        missing_policy: str = "warn_default",
    ) -> Dict[str, float]:
        """Normalizes PDM result containers into Stage-3 reward components."""
        raw = self._pdm_result_to_raw_dict(pdm_result)
        raw_by_key = {str(key).lower(): value for key, value in raw.items()}
        aliases = self._pdm_component_aliases()
        defaults = {
            "pdms": 0.0,
            "no_at_fault_collisions": 1.0,
            "drivable_area_compliance": 1.0,
            "time_to_collision_within_bound": 1.0,
            "ego_progress": 0.0,
            "history_comfort": 0.0,
            "lane_keeping": 0.0,
            "driving_direction_compliance": 1.0,
            "traffic_light_compliance": 1.0,
        }

        if strict_submetrics:
            if missing_policy not in {"error", "unsafe_zero", "warn_default"}:
                raise ValueError(f"Unsupported missing_submetric_policy={missing_policy!r}.")
            required = tuple(required_submetrics or ())
            missing = []
            for key in required:
                if key not in aliases:
                    raise KeyError(f"Unknown required PDM submetric {key!r}.")
                if not any(alias.lower() in raw_by_key for alias in aliases[key]):
                    missing.append(key)
            if missing:
                if missing_policy == "error":
                    raise KeyError(
                        "PDM result is missing required AWAC/IQL safety submetrics "
                        f"{missing}; available keys={sorted(raw_by_key)}."
                    )
                if missing_policy == "unsafe_zero":
                    for key in missing:
                        defaults[key] = 0.0
                else:
                    warned = getattr(self, "_warned_missing_awac_submetrics", set())
                    signature = tuple(sorted(missing))
                    if signature not in warned:
                        warnings.warn(
                            "PDM result is missing required AWAC/IQL safety submetrics "
                            f"{missing}; using backward-compatible defaults.",
                            RuntimeWarning,
                        )
                        warned = set(warned)
                        warned.add(signature)
                        self._warned_missing_awac_submetrics = warned

        def pick(component: str) -> float:
            keys = aliases[component]
            for key in keys:
                normalized_key = key.lower()
                if normalized_key in raw_by_key:
                    return self._to_python_float(raw_by_key[normalized_key], defaults[component])
            return defaults[component]

        return {
            key: pick(key)
            for key in REQUIRED_COMPONENT_KEYS
        }

    def _compute_hard_safety_mask(self, components: Dict[str, torch.Tensor]) -> torch.Tensor:
        pdms = components["pdms"]
        hard_safe_mask = torch.ones(pdms.shape, device=pdms.device, dtype=torch.bool)
        if self.hard_gate_nc:
            hard_safe_mask &= components["no_at_fault_collisions"] >= float(self.nc_safe_threshold)
        if self.hard_gate_dac:
            hard_safe_mask &= components["drivable_area_compliance"] >= float(self.dac_safe_threshold)
        if self.hard_gate_ttc:
            hard_safe_mask &= components["time_to_collision_within_bound"] >= float(self.ttc_safe_threshold)
        if self.hard_gate_ddc:
            hard_safe_mask &= components["driving_direction_compliance"] >= float(self.ddc_safe_threshold)
        if self.hard_gate_tlc:
            hard_safe_mask &= components["traffic_light_compliance"] >= float(self.tlc_safe_threshold)
        return hard_safe_mask

    def _compute_safe_diversity_bonus(
        self,
        trajs: torch.Tensor,
        hard_safe_mask: torch.Tensor,
        pdms: torch.Tensor,
        B: int,
        G: int,
    ) -> torch.Tensor:
        # trajs: [B * G, H, D], hard_safe_mask/pdms: [B * G]
        assert trajs.shape[0] == B * G
        assert hard_safe_mask.shape == (B * G,)
        assert pdms.shape == (B * G,)

        xy = trajs.reshape(B, G, trajs.shape[1], trajs.shape[2])[..., :2]
        valid = (hard_safe_mask.reshape(B, G) & (pdms.reshape(B, G) >= float(self.diversity_pdms_threshold)))
        scale = max(float(self.diversity_distance_scale), 1e-8)

        if self.diversity_metric == "endpoint":
            points = xy[:, :, -1, :]
            pairwise = torch.linalg.norm(points[:, :, None, :] - points[:, None, :, :], dim=-1)
        elif self.diversity_metric == "trajectory":
            delta = xy[:, :, None, :, :] - xy[:, None, :, :, :]
            pairwise = torch.linalg.norm(delta, dim=-1).mean(dim=-1)
        else:
            raise ValueError(f"Unsupported diversity_metric: {self.diversity_metric!r}")

        eye = torch.eye(G, device=trajs.device, dtype=torch.bool).unsqueeze(0)
        pair_mask = valid[:, :, None] & valid[:, None, :] & ~eye
        pair_count = pair_mask.sum(dim=2)
        pair_sum = (pairwise * pair_mask.to(dtype=pairwise.dtype)).sum(dim=2)
        bonus = pair_sum / pair_count.clamp(min=1).to(dtype=pairwise.dtype)
        bonus = (bonus / scale).clamp(min=0.0, max=1.0)
        bonus = torch.where(valid & (pair_count > 0), bonus, torch.zeros_like(bonus))
        return bonus.reshape(B * G)

    def _compose_stage3_reward(
        self,
        base_rewards: torch.Tensor,
        components: Dict[str, torch.Tensor],
        trajs: torch.Tensor,
        B: int,
        G: int,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # base_rewards/reward/hard_safe_mask: [B * G]
        assert base_rewards.shape == (B * G,)
        pdms = base_rewards
        hard_safe_mask = self._compute_hard_safety_mask(components)
        assert hard_safe_mask.shape == (B * G,)

        ego_progress = components["ego_progress"]
        if self.use_safety_shaped_reward:
            safe_reward = pdms + float(self.progress_bonus_weight) * ego_progress
            unsafe_reward = float(self.unsafe_reward_floor) + float(self.unsafe_pdms_scale) * pdms
            reward = torch.where(hard_safe_mask, safe_reward, unsafe_reward)
        else:
            reward = pdms

        if self.log_safe_diversity or self.use_diversity_reward:
            diversity_bonus = self._compute_safe_diversity_bonus(trajs, hard_safe_mask, pdms, B, G)
        else:
            diversity_bonus = torch.zeros_like(reward)

        if self.use_diversity_reward:
            diversity_valid = hard_safe_mask & (pdms >= float(self.diversity_pdms_threshold))
            reward = reward + torch.where(
                diversity_valid,
                float(self.diversity_reward_weight) * diversity_bonus,
                torch.zeros_like(diversity_bonus),
            )

        aux = {
            "pdms": pdms,
            "ego_progress": ego_progress,
            "diversity_bonus": diversity_bonus,
        }
        for key in (
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "time_to_collision_within_bound",
            "history_comfort",
            "lane_keeping",
            "driving_direction_compliance",
            "traffic_light_compliance",
        ):
            if key in components:
                aux[key] = components[key]
        return reward, hard_safe_mask, aux

    def _compute_stage3_advantages(
        self,
        rewards_matrix: torch.Tensor,
        hard_safe_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # rewards_matrix/hard_safe_matrix: [B, G]
        assert rewards_matrix.shape == hard_safe_matrix.shape
        B, G = rewards_matrix.shape

        mean_r = rewards_matrix.mean(dim=1, keepdim=True)
        if G > 1:
            std_r = rewards_matrix.std(dim=1, keepdim=True).clamp(min=float(self.advantage_std_floor))
            reward_std = rewards_matrix.std(dim=1)
        else:
            std_r = rewards_matrix.new_full((B, 1), float(self.advantage_std_floor))
            reward_std = rewards_matrix.new_zeros(B)
        z = (rewards_matrix - mean_r) / std_r

        if self.use_asymmetric_safe_advantage:
            safe_adv = torch.clamp(z, min=0.0) + float(self.safe_negative_adv_scale) * torch.clamp(z, max=0.0)
            unsafe_adv = -float(self.unsafe_advantage_offset) + torch.clamp(z, max=0.0)
            advantages = torch.where(hard_safe_matrix, safe_adv, unsafe_adv)
        else:
            advantages = z

        flat_advantages = advantages.flatten()
        adv_min = torch.quantile(flat_advantages.float(), float(self.clip_advantage_lower_quantile)).to(advantages)
        adv_max = torch.quantile(flat_advantages.float(), float(self.clip_advantage_upper_quantile)).to(advantages)
        advantages = advantages.clamp(min=adv_min, max=adv_max)

        safe_count = hard_safe_matrix.sum(dim=1)
        mixed = (safe_count > 0) & (safe_count < G)
        all_safe = safe_count == G
        all_unsafe = safe_count == 0

        if self.use_dynamic_group_weight:
            group_weight = torch.ones(B, device=rewards_matrix.device, dtype=rewards_matrix.dtype)
            group_weight[all_safe & (reward_std <= float(self.min_group_reward_std))] = float(self.all_safe_low_std_group_weight)
            group_weight[all_unsafe] = float(self.all_unsafe_group_weight)
        else:
            group_weight = torch.ones(B, device=rewards_matrix.device, dtype=rewards_matrix.dtype)

        aux = {
            "reward_std": reward_std.detach(),
            "safe_count": safe_count.detach().to(dtype=rewards_matrix.dtype),
            "mixed_group_ratio": mixed.float().mean().to(dtype=rewards_matrix.dtype),
            "all_safe_group_ratio": all_safe.float().mean().to(dtype=rewards_matrix.dtype),
            "all_unsafe_group_ratio": all_unsafe.float().mean().to(dtype=rewards_matrix.dtype),
        }
        return advantages.reshape(B * G).detach(), group_weight.detach(), aux

    def _reduce_chain_logprobs(
        self,
        log_probs: torch.Tensor,
        B: int,
        G: int,
        K: int,
        discount: torch.Tensor,
    ) -> torch.Tensor:
        # raw log_probs: [B * G * K, H, D]
        assert log_probs.shape[0] == B * G * K
        step_logp = log_probs.clamp(min=-5, max=2).mean(dim=[1, 2])
        step_logp = step_logp.view(B * G, K)

        if self.trajectory_logprob_reduce == "discounted_mean":
            discount = discount.to(device=step_logp.device, dtype=step_logp.dtype)
            discount_norm = discount / discount.sum().clamp(min=1e-8)
            traj_logp = (step_logp * discount_norm.view(1, K)).sum(dim=1)
        elif self.trajectory_logprob_reduce == "mean":
            traj_logp = step_logp.mean(dim=1)
        else:
            raise ValueError(f"Unsupported trajectory_logprob_reduce: {self.trajectory_logprob_reduce!r}")

        # traj_logp: [B * G]
        assert traj_logp.shape == (B * G,)
        return traj_logp

    def _current_bc_coeff(self, override: Optional[float] = None) -> float:
        if override is not None:
            return float(override)
        if not bool(getattr(self, "bc_anneal", False)):
            return float(getattr(self, "bc_coeff_start", 0.1))

        current_epoch = max(0, int(getattr(self.config, "current_train_epoch", 0)))
        anneal_epochs = max(1, int(getattr(self, "bc_anneal_epochs", 1)))
        progress = min(float(current_epoch) / float(anneal_epochs), 1.0)
        start = float(getattr(self, "bc_coeff_start", 0.1))
        end = float(getattr(self, "bc_coeff_end", start))
        return start + (end - start) * progress

    def _current_awac_bc_loss_weight(self, cfg: OfflineRLConfig) -> float:
        if str(cfg.bc_loss_schedule) == "constant":
            return float(cfg.bc_loss_weight)

        current_epoch = max(0, int(getattr(self.config, "current_train_epoch", 0)))
        schedule_epochs = max(1, int(cfg.bc_loss_schedule_epochs))
        progress = min(float(current_epoch) / float(schedule_epochs), 1.0)
        start = float(cfg.bc_loss_weight_start)
        end = float(cfg.bc_loss_weight_end)
        return start + (end - start) * progress

    def _current_linear_warmup_loss_weight(
        self,
        *,
        target: float,
        schedule: str,
        start: float,
        start_epoch: int,
        warmup_epochs: int,
    ) -> float:
        if target <= 0.0:
            return 0.0
        if str(schedule) == "constant":
            return target

        current_epoch = max(0, int(getattr(self.config, "current_train_epoch", 0)))
        if current_epoch < start_epoch:
            return float(start)
        warmup_epochs = max(1, int(warmup_epochs))
        progress = min(float(current_epoch - start_epoch + 1) / float(warmup_epochs), 1.0)
        return start + (target - start) * progress

    def _current_awac_loss_weight(self, cfg: OfflineRLConfig) -> float:
        return self._current_linear_warmup_loss_weight(
            target=float(cfg.awac_loss_weight),
            schedule=str(cfg.awac_loss_schedule),
            start=float(cfg.awac_loss_weight_start),
            start_epoch=int(cfg.awac_loss_warmup_start_epoch),
            warmup_epochs=int(cfg.awac_loss_warmup_epochs),
        )

    def _current_awac_preference_dpo_loss_weight(self, cfg: OfflineRLConfig) -> float:
        return self._current_linear_warmup_loss_weight(
            target=float(cfg.preference_dpo_loss_weight),
            schedule=str(cfg.preference_dpo_loss_schedule),
            start=float(cfg.preference_dpo_loss_weight_start),
            start_epoch=int(cfg.preference_dpo_loss_warmup_start_epoch),
            warmup_epochs=int(cfg.preference_dpo_loss_warmup_epochs),
        )

    def _current_awac_grpo_loss_weight(self, cfg: OfflineRLConfig) -> float:
        return self._current_linear_warmup_loss_weight(
            target=float(cfg.grpo_loss_weight),
            schedule=str(cfg.grpo_loss_schedule),
            start=float(cfg.grpo_loss_weight_start),
            start_epoch=int(cfg.grpo_loss_warmup_start_epoch),
            warmup_epochs=int(cfg.grpo_loss_warmup_epochs),
        )

    def _current_awac_target_blend_alpha(self, cfg: OfflineRLConfig) -> float:
        alpha = self._current_linear_warmup_loss_weight(
            target=float(cfg.target_blend_alpha),
            schedule=str(cfg.target_blend_schedule),
            start=float(cfg.target_blend_alpha_start),
            start_epoch=int(cfg.target_blend_warmup_start_epoch),
            warmup_epochs=int(cfg.target_blend_warmup_epochs),
        )
        return min(max(float(alpha), 0.0), 1.0)

    def _load_metric_cache_for_tokens(self, tokens_list) -> Dict[str, Any]:
        if not hasattr(self, "metric_cache_loader"):
            self._init_stage3_oracle(self.config.grpo_cfg)
        metric_cache = {}
        for token in set(str(token) for token in tokens_list):
            if token not in self.metric_cache_loader.metric_cache_paths:
                raise KeyError(f"Metric cache missing token={token!r}; Stage3 training must use train metric cache.")
            path = self.metric_cache_loader.metric_cache_paths[token]
            with lzma.open(path, "rb") as f:
                metric_cache[token] = pickle.load(f)
        return metric_cache

    def _score_candidate_trajectories(
        self,
        candidates: torch.Tensor,
        tokens_list: list[str],
        metric_cache: Dict[str, Any],
        cfg: Optional[OfflineRLConfig] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if candidates.ndim != 4 or candidates.shape[-1] != 3:
            raise ValueError(f"candidates must have shape [B, K, H, 3], got {tuple(candidates.shape)}.")
        if not torch.isfinite(candidates).all():
            raise ValueError("AWAC candidates contain non-finite values before scoring.")
        B, K, H, D = candidates.shape
        if len(tokens_list) != B:
            raise ValueError(f"tokens_list length {len(tokens_list)} does not match B={B}.")
        flat = candidates.detach().float().reshape(B * K, H, D)
        tokens_rep = [str(token) for token in tokens_list for _ in range(K)]
        unique_indices: list[int] = []
        unique_tokens: list[str] = []
        inverse_indices: list[int] = []
        seen: Dict[bytes, int] = {}
        flat_np = flat.detach().cpu().numpy()
        for idx, (token, trajectory) in enumerate(zip(tokens_rep, flat_np)):
            digest = hashlib.sha1()
            digest.update(token.encode("utf-8"))
            digest.update(np.ascontiguousarray(trajectory).tobytes())
            key = digest.digest()
            unique_idx = seen.get(key)
            if unique_idx is None:
                unique_idx = len(unique_indices)
                seen[key] = unique_idx
                unique_indices.append(idx)
                unique_tokens.append(token)
            inverse_indices.append(unique_idx)

        score_flat = flat if len(unique_indices) == flat.shape[0] else flat[unique_indices]
        score_tokens = tokens_rep if len(unique_indices) == flat.shape[0] else unique_tokens
        rewards, components = self.reward_fn(
            score_flat,
            score_tokens,
            metric_cache,
            return_components=True,
            strict_submetrics=bool(cfg.strict_reward_submetrics) if cfg is not None else False,
            required_submetrics=cfg.required_reward_submetrics if cfg is not None else None,
            missing_submetric_policy=str(cfg.missing_submetric_policy) if cfg is not None else "warn_default",
            use_batched_pdm_scoring=bool(cfg.use_batched_pdm_scoring) if cfg is not None else False,
            use_exact_array_pdm_state_conversion=(
                bool(cfg.use_exact_array_pdm_state_conversion) if cfg is not None else False
            ),
            use_fast_pdm_scorer=bool(cfg.use_fast_pdm_scorer) if cfg is not None else False,
            pdm_batch_chunk_size=int(cfg.pdm_batch_chunk_size) if cfg is not None else 0,
            pdm_shadow_check=bool(cfg.pdm_shadow_check) if cfg is not None else False,
            pdm_shadow_max_samples=int(cfg.pdm_shadow_max_samples) if cfg is not None else 0,
            pdm_shadow_max_abs_diff=float(cfg.pdm_shadow_max_abs_diff) if cfg is not None else 0.0,
        )
        if len(unique_indices) != flat.shape[0]:
            inverse = torch.tensor(inverse_indices, device=rewards.device, dtype=torch.long)
            rewards = rewards.index_select(0, inverse)
            components = {key: value.index_select(0, inverse) for key, value in components.items()}
        rewards = rewards.reshape(B, K).float()
        components = {key: value.reshape(B, K).float() for key, value in components.items()}
        if not torch.isfinite(rewards).all():
            raise ValueError("PDMS rewards contain non-finite values.")
        return rewards, components

    @staticmethod
    def _trajectory_jerk_penalty(trajs: torch.Tensor) -> torch.Tensor:
        if trajs.shape[-2] < 4:
            return trajs.new_zeros(trajs.shape[:2])
        xy = trajs[..., :2].float()
        velocity = xy[..., 1:, :] - xy[..., :-1, :]
        acceleration = velocity[..., 1:, :] - velocity[..., :-1, :]
        jerk = acceleration[..., 1:, :] - acceleration[..., :-1, :]
        return torch.linalg.norm(jerk, dim=-1).mean(dim=-1).to(dtype=trajs.dtype)

    @staticmethod
    def _source_bucket(source: str) -> str:
        source = str(source)
        if source == "gt":
            return "gt"
        if source == "il":
            return "il"
        if source.startswith("policy"):
            return "policy"
        if source.startswith("progress"):
            return "progress"
        if "lateral" in source:
            return "lateral"
        if source.startswith("timing"):
            return "timing"
        return "other"

    @classmethod
    def _source_code(cls, source: str) -> int:
        return {
            "gt": 1,
            "il": 2,
            "policy": 3,
            "progress": 4,
            "lateral": 5,
            "timing": 6,
        }.get(cls._source_bucket(source), 0)

    def _compute_awac_candidate_valid_mask(
        self,
        components: Dict[str, torch.Tensor],
        sources: list[str],
        cfg: OfflineRLConfig,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pdms = components["pdms"]
        valid = torch.ones_like(pdms, dtype=torch.bool)
        pass_masks: Dict[str, torch.Tensor] = {}

        nc_pass = components["no_at_fault_collisions"] >= 1.0
        dac_pass = components["drivable_area_compliance"] >= 1.0
        if bool(cfg.require_nc):
            valid &= nc_pass
        if bool(cfg.require_dac):
            valid &= dac_pass

        source_to_index: Dict[str, int] = {}
        for idx, source in enumerate(sources):
            source_to_index.setdefault(source, idx)
        gt_idx = source_to_index.get("gt")
        il_idx = source_to_index.get("il")

        ddc = components["driving_direction_compliance"]
        if bool(cfg.require_ddc_guard):
            ddc_anchor = None
            if gt_idx is not None:
                ddc_anchor = ddc[:, gt_idx]
            if il_idx is not None:
                ddc_anchor = ddc[:, il_idx] if ddc_anchor is None else torch.maximum(ddc_anchor, ddc[:, il_idx])
            if ddc_anchor is None:
                ddc_pass = ddc >= float(cfg.ddc_min_absolute)
            else:
                ddc_absolute_pass = ddc >= float(cfg.ddc_min_absolute)
                ddc_relative_pass = ddc >= ddc_anchor[:, None] - float(cfg.ddc_max_relative_drop)
                if str(cfg.ddc_guard_mode) == "absolute":
                    ddc_pass = ddc_absolute_pass
                elif str(cfg.ddc_guard_mode) == "relative":
                    ddc_pass = ddc_relative_pass
                else:
                    ddc_pass = ddc_absolute_pass | ddc_relative_pass
            valid &= ddc_pass
        else:
            ddc_pass = torch.ones_like(valid)

        ttc = components["time_to_collision_within_bound"]
        if bool(cfg.require_ttc_guard):
            ttc_anchor = None
            if gt_idx is not None:
                ttc_anchor = ttc[:, gt_idx]
            if il_idx is not None:
                ttc_anchor = ttc[:, il_idx] if ttc_anchor is None else torch.maximum(ttc_anchor, ttc[:, il_idx])
            if ttc_anchor is None:
                ttc_pass = ttc >= float(cfg.ttc_min_absolute)
            else:
                ttc_absolute_pass = ttc >= float(cfg.ttc_min_absolute)
                ttc_relative_pass = ttc >= ttc_anchor[:, None] - float(cfg.ttc_max_relative_drop)
                if str(cfg.ttc_guard_mode) == "absolute":
                    ttc_pass = ttc_absolute_pass
                elif str(cfg.ttc_guard_mode) == "relative":
                    ttc_pass = ttc_relative_pass
                else:
                    ttc_pass = ttc_absolute_pass | ttc_relative_pass
            valid &= ttc_pass
        else:
            ttc_pass = torch.ones_like(valid)

        pass_masks["nc_pass"] = nc_pass
        pass_masks["dac_pass"] = dac_pass
        pass_masks["ddc_pass"] = ddc_pass
        pass_masks["ttc_pass"] = ttc_pass
        diagnostics = {
            "nc_pass_ratio": nc_pass.float().mean(),
            "dac_pass_ratio": dac_pass.float().mean(),
            "ddc_pass_ratio": ddc_pass.float().mean(),
            "ttc_pass_ratio": ttc_pass.float().mean(),
            "valid_candidate_ratio": valid.float().mean(),
            "has_valid_candidate_ratio": valid.any(dim=1).float().mean(),
            **pass_masks,
        }
        return valid, diagnostics

    def _select_elite_candidates(
        self,
        candidates: torch.Tensor,
        rewards: torch.Tensor,
        components: Dict[str, torch.Tensor],
        sources: list[str],
        anchor_distance: torch.Tensor,
        gt_reward: torch.Tensor,
        il_reward: torch.Tensor,
        cfg: OfflineRLConfig,
    ) -> Dict[str, torch.Tensor]:
        if candidates.ndim != 4 or candidates.shape[-1] != 3:
            raise ValueError(f"candidates must have shape [B, K, H, 3], got {tuple(candidates.shape)}.")
        B, K, H, D = candidates.shape
        if rewards.shape != (B, K):
            raise ValueError(f"rewards must have shape [B, K], got {tuple(rewards.shape)}.")
        if anchor_distance.shape != (B, K):
            raise ValueError(f"anchor_distance must have shape [B, K], got {tuple(anchor_distance.shape)}.")
        if len(sources) != K:
            raise ValueError(f"sources length {len(sources)} does not match K={K}.")

        source_to_index: Dict[str, int] = {}
        for idx, source in enumerate(sources):
            source_to_index.setdefault(source, idx)
        gt_idx = source_to_index.get("gt")
        il_idx = source_to_index.get("il")
        device = candidates.device
        valid, valid_diag = self._compute_awac_candidate_valid_mask(components, sources, cfg)

        if str(cfg.select_by) == "pdms":
            selection_score = rewards.float()
        else:
            jerk = self._trajectory_jerk_penalty(candidates).float()
            selection_score = (
                rewards.float()
                - float(cfg.prior_distance_weight) * anchor_distance.float()
                - float(cfg.jerk_penalty_weight) * jerk
            )

        selected_indices: list[torch.Tensor] = []
        fallback_flags = []
        has_valid_candidate = valid.any(dim=1)
        min_candidates = max(1, int(cfg.elite_min_candidates))
        top_m = max(1, int(cfg.elite_top_m))
        for b in range(B):
            row_valid = valid[b]
            valid_idx = torch.nonzero(row_valid, as_tuple=False).flatten()
            if int(valid_idx.numel()) > 0 and bool(cfg.select_valid_topk_only):
                k = min(top_m, int(valid_idx.numel()))
                valid_scores = selection_score[b, valid_idx]
                idx = valid_idx[torch.topk(valid_scores, k=k, largest=True).indices].tolist()
                fallback = False
            elif int(valid_idx.numel()) > 0:
                k = min(top_m, K)
                idx = torch.topk(selection_score[b], k=k, largest=True).indices.tolist()
                fallback = False
            else:
                k = min(top_m, K)
                idx = torch.topk(rewards[b].float(), k=k, largest=True).indices.tolist()
                fallback = True
            if bool(cfg.keep_gt_candidate) and gt_idx is not None:
                idx.append(gt_idx)
            if bool(cfg.keep_il_candidate) and il_idx is not None:
                idx.append(il_idx)
            if len(set(idx)) < min(min_candidates, K):
                remaining_score = rewards[b].float().clone()
                for used_idx in set(int(item) for item in idx):
                    remaining_score[used_idx] = -torch.inf
                for extra_idx in torch.topk(remaining_score, k=min(K, min_candidates), largest=True).indices.tolist():
                    if torch.isfinite(remaining_score[int(extra_idx)]):
                        idx.append(extra_idx)
            unique_idx = []
            seen = set()
            for item in idx:
                if int(item) not in seen:
                    unique_idx.append(int(item))
                    seen.add(int(item))
            selected_indices.append(torch.tensor(unique_idx, device=device, dtype=torch.long))
            fallback_flags.append(fallback)

        max_m = max(int(idx.numel()) for idx in selected_indices)
        selected_trajs = candidates.new_zeros((B, max_m, H, D))
        selected_rewards = rewards.new_zeros((B, max_m))
        selected_anchor_distance = anchor_distance.new_zeros((B, max_m))
        selected_real_mask = torch.zeros((B, max_m), device=device, dtype=torch.bool)
        selected_valid_mask = torch.zeros((B, max_m), device=device, dtype=torch.bool)
        selected_source_code = torch.zeros((B, max_m), device=device, dtype=torch.long)
        selected_source_index = torch.full((B, max_m), -1, device=device, dtype=torch.long)
        selected_selection_score = selection_score.new_zeros((B, max_m))
        selected_components = {
            key: value.new_zeros((B, max_m))
            for key, value in components.items()
        }
        for b, idx in enumerate(selected_indices):
            m = int(idx.numel())
            selected_trajs[b, :m] = candidates[b, idx]
            selected_rewards[b, :m] = rewards[b, idx]
            selected_anchor_distance[b, :m] = anchor_distance[b, idx]
            selected_real_mask[b, :m] = True
            selected_valid_mask[b, :m] = valid[b, idx]
            selected_source_index[b, :m] = idx
            selected_selection_score[b, :m] = selection_score[b, idx]
            selected_source_code[b, :m] = torch.tensor(
                [self._source_code(sources[int(i)]) for i in idx.tolist()],
                device=device,
                dtype=torch.long,
            )
            for key, value in components.items():
                selected_components[key][b, :m] = value[b, idx]

        fallback_mask = torch.tensor(fallback_flags, device=device, dtype=torch.bool)
        best_raw_idx = rewards.argmax(dim=1)
        best_raw_reward = rewards.gather(1, best_raw_idx[:, None]).squeeze(1)
        valid_rewards = rewards.masked_fill(~valid, -torch.inf)
        best_valid_idx = valid_rewards.argmax(dim=1)
        best_valid_reward_candidate = valid_rewards.gather(1, best_valid_idx[:, None]).squeeze(1)
        best_valid_reward = torch.where(has_valid_candidate, best_valid_reward_candidate, best_raw_reward)
        best_valid_idx = torch.where(has_valid_candidate, best_valid_idx, best_raw_idx)
        selected_masked_rewards = selected_rewards.masked_fill(~selected_real_mask, -torch.inf)
        best_selected_pos = selected_masked_rewards.argmax(dim=1)
        best_selected_reward = selected_masked_rewards.gather(1, best_selected_pos[:, None]).squeeze(1)
        best_selected_source_code = selected_source_code.gather(1, best_selected_pos[:, None]).squeeze(1)
        best_selected_index = selected_source_index.gather(1, best_selected_pos[:, None]).squeeze(1)
        best_raw_source_code = torch.tensor(
            [self._source_code(sources[int(idx.item())]) for idx in best_raw_idx],
            device=device,
            dtype=torch.long,
        )
        best_valid_source_code = torch.tensor(
            [self._source_code(sources[int(idx.item())]) for idx in best_valid_idx],
            device=device,
            dtype=torch.long,
        )
        return {
            "selected_trajs": selected_trajs,
            "selected_rewards": selected_rewards,
            "selected_components": selected_components,
            "selected_anchor_distance": selected_anchor_distance,
            "selected_selection_score": selected_selection_score,
            "selected_valid_mask": selected_valid_mask,
            "selected_real_mask": selected_real_mask,
            "selected_source_code": selected_source_code,
            "selected_source_index": selected_source_index,
            "candidate_valid_mask": valid,
            "valid_candidate_ratio": valid_diag["valid_candidate_ratio"],
            "has_valid_candidate_ratio": valid_diag["has_valid_candidate_ratio"],
            "nc_pass_ratio": valid_diag["nc_pass_ratio"],
            "dac_pass_ratio": valid_diag["dac_pass_ratio"],
            "ddc_pass_ratio": valid_diag["ddc_pass_ratio"],
            "ttc_pass_ratio": valid_diag["ttc_pass_ratio"],
            "fallback_candidate_ratio": fallback_mask.float().mean(),
            "fallback_candidate": fallback_mask,
            "has_valid_candidate": has_valid_candidate,
            "gt_reward": gt_reward.float(),
            "il_reward": il_reward.float(),
            "best_raw_reward": best_raw_reward.float(),
            "best_valid_reward": best_valid_reward.float(),
            "best_selected_reward": best_selected_reward.float(),
            "best_raw_source_code": best_raw_source_code,
            "best_valid_source_code": best_valid_source_code,
            "best_selected_source_code": best_selected_source_code,
            "best_raw_index": best_raw_idx,
            "best_valid_index": best_valid_idx,
            "best_selected_index": best_selected_index,
            "best_reward": best_valid_reward.float(),
            "best_source_code": best_valid_source_code,
        }

    def _compute_iql_baseline(
        self,
        rewards: torch.Tensor,
        gt_reward: torch.Tensor,
        il_reward: torch.Tensor,
        cfg: OfflineRLConfig,
        real_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if rewards.ndim != 2:
            raise ValueError(f"rewards must have shape [B, M], got {tuple(rewards.shape)}.")
        if real_mask is None:
            real_mask = torch.ones_like(rewards, dtype=torch.bool)
        real_f = real_mask.to(dtype=rewards.dtype)
        mode = str(cfg.baseline_mode)
        if mode == "max_gt_il":
            baseline = torch.maximum(gt_reward.to(rewards), il_reward.to(rewards))
        elif mode == "mean":
            denom = real_f.sum(dim=1).clamp(min=1.0)
            baseline = (rewards * real_f).sum(dim=1) / denom
        elif mode == "top_mean":
            masked_rewards = rewards.masked_fill(~real_mask, -torch.inf)
            counts = real_mask.sum(dim=1).clamp(min=1)
            top_k = max(1, int(rewards.shape[1] * float(cfg.top_mean_frac)))
            top_k = min(top_k, rewards.shape[1])
            values = torch.topk(masked_rewards, k=top_k, dim=1).values
            finite = torch.isfinite(values)
            baseline = (values.masked_fill(~finite, 0.0).sum(dim=1) / finite.sum(dim=1).clamp(min=1))
            baseline = torch.where(counts > 0, baseline, rewards.mean(dim=1))
        elif mode == "expectile":
            denom = real_f.sum(dim=1, keepdim=True).clamp(min=1.0)
            v = (rewards * real_f).sum(dim=1, keepdim=True) / denom
            for _ in range(int(cfg.expectile_iters)):
                diff = rewards - v
                weights = torch.where(diff > 0, float(cfg.expectile_tau), 1.0 - float(cfg.expectile_tau))
                weights = weights.to(rewards) * real_f
                v = (weights * rewards).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
            baseline = v.squeeze(1)
        else:
            raise ValueError(f"Unsupported offline_rl baseline_mode: {cfg.baseline_mode!r}")
        return baseline.float()

    @staticmethod
    def _component_anchor_value(
        values: torch.Tensor,
        source_code: torch.Tensor,
        real_mask: torch.Tensor,
    ) -> torch.Tensor:
        anchor_mask = ((source_code == 1) | (source_code == 2)) & real_mask
        anchor_values = values.float().masked_fill(~anchor_mask, -torch.inf)
        anchor = anchor_values.max(dim=1).values
        fallback_values = values.float().masked_fill(~real_mask, -torch.inf)
        fallback = fallback_values.max(dim=1).values
        fallback = torch.where(torch.isfinite(fallback), fallback, torch.zeros_like(fallback))
        return torch.where(torch.isfinite(anchor), anchor, fallback)

    def _compute_component_aware_rewards(
        self,
        rewards: torch.Tensor,
        components: Dict[str, torch.Tensor],
        source_code: torch.Tensor,
        real_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        cfg: OfflineRLConfig,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = rewards.new_zeros(())
        if not bool(cfg.component_advantage_enabled):
            return rewards.float(), {
                "component_reward_delta_mean": zero,
                "component_reward_delta_min": zero,
                "component_reward_delta_max": zero,
                "component_progress_bonus_mean": zero,
                "component_safety_penalty_mean": zero,
            }

        ep = components["ego_progress"].float()
        ep_anchor = self._component_anchor_value(ep, source_code, real_mask)
        progress_bonus = (ep - ep_anchor[:, None]).clamp(min=0.0) * float(cfg.component_progress_weight)

        safety_penalty = torch.zeros_like(rewards, dtype=torch.float32)
        for key in (
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "time_to_collision_within_bound",
            "driving_direction_compliance",
        ):
            value = components[key].float()
            anchor = self._component_anchor_value(value, source_code, real_mask)
            safety_penalty = safety_penalty + (anchor[:, None] - value).clamp(min=0.0)
        safety_penalty = safety_penalty * float(cfg.component_safety_penalty_weight)

        delta = (progress_bonus - safety_penalty).clamp(
            float(cfg.component_advantage_clip_min),
            float(cfg.component_advantage_clip_max),
        )
        delta = delta * real_mask.to(delta)
        shaped_rewards = rewards.float() + delta
        active = real_mask
        if bool(active.any().item()):
            delta_active = delta[active]
            progress_active = progress_bonus[active]
            penalty_active = safety_penalty[active]
        else:
            delta_active = delta.reshape(-1)
            progress_active = progress_bonus.reshape(-1)
            penalty_active = safety_penalty.reshape(-1)
        valid_active = valid_mask & real_mask
        return shaped_rewards, {
            "component_reward_delta_mean": delta_active.mean().to(rewards),
            "component_reward_delta_min": delta_active.min().to(rewards),
            "component_reward_delta_max": delta_active.max().to(rewards),
            "component_progress_bonus_mean": progress_active.mean().to(rewards),
            "component_safety_penalty_mean": penalty_active.mean().to(rewards),
            "component_valid_reward_delta_mean": (
                delta[valid_active].mean().to(rewards) if bool(valid_active.any().item()) else zero
            ),
        }

    def _compute_awac_weights(
        self,
        rewards: torch.Tensor,
        baseline: torch.Tensor,
        cfg: OfflineRLConfig,
        valid_mask: Optional[torch.Tensor] = None,
        real_mask: Optional[torch.Tensor] = None,
        gt_reward: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if rewards.ndim != 2 or baseline.shape != (rewards.shape[0],):
            raise ValueError(
                f"rewards must be [B, M] and baseline [B], got rewards={tuple(rewards.shape)} baseline={tuple(baseline.shape)}."
            )
        if real_mask is None:
            real_mask = torch.ones_like(rewards, dtype=torch.bool)
        if valid_mask is None:
            valid_mask = real_mask
        adv = rewards.float() - baseline.float()[:, None]
        if gt_reward is not None and float(cfg.min_reward_margin_to_gt_for_extra_weight) > 0.0:
            improved = rewards.float() > gt_reward.float()[:, None] + float(cfg.min_reward_margin_to_gt_for_extra_weight)
            adv = torch.where(improved, adv, adv.clamp(max=0.0))
        adv = adv.clamp(float(cfg.advantage_clip_min), float(cfg.advantage_clip_max))
        weights = torch.exp((adv / float(cfg.advantage_temperature)).clamp(min=-60.0, max=60.0))
        weights = weights.clamp(float(cfg.weight_min), float(cfg.weight_max))
        weights = weights * real_mask.to(dtype=weights.dtype)
        if bool(cfg.train_only_valid_candidates):
            weights = weights * valid_mask.to(dtype=weights.dtype)
        filter_mode = str(cfg.target_filter_mode)
        filter_mask = real_mask.clone()
        if bool(cfg.train_only_valid_candidates):
            filter_mask = filter_mask & valid_mask
        if filter_mode != "none":
            eligible_mask = filter_mask.clone()
            if filter_mode in {"positive_advantage", "topk_positive"}:
                filter_mask = filter_mask & (adv > float(cfg.target_min_advantage))
            if filter_mode in {"topk", "topk_positive"}:
                topk_mask = torch.zeros_like(filter_mask)
                ranking_mask = filter_mask if filter_mode == "topk_positive" else eligible_mask
                top_k = max(1, int(cfg.target_top_k))
                for row in range(rewards.shape[0]):
                    row_idx = torch.nonzero(ranking_mask[row], as_tuple=False).flatten()
                    if row_idx.numel() == 0:
                        continue
                    k = min(top_k, int(row_idx.numel()))
                    row_score = rewards[row, row_idx].float()
                    keep = row_idx[torch.topk(row_score, k=k, largest=True).indices]
                    topk_mask[row, keep] = True
                filter_mask = topk_mask
            weights = weights * filter_mask.to(dtype=weights.dtype)
        empty_rows = weights.sum(dim=1) <= 0.0
        if bool(empty_rows.any().item()) and bool(cfg.train_invalid_fallback_candidates):
            safe_rewards = rewards.masked_fill(~real_mask, -torch.inf)
            fallback_idx = safe_rewards.argmax(dim=1)
            row_idx = torch.arange(rewards.shape[0], device=rewards.device)
            weights[row_idx[empty_rows], fallback_idx[empty_rows]] = float(cfg.fallback_invalid_candidate_weight)
        if bool(cfg.normalize_weights_per_scene):
            positive_rows = weights.sum(dim=1, keepdim=True) > 0.0
            norm_mask = (weights > 0).to(dtype=weights.dtype)
            mean = weights.sum(dim=1, keepdim=True) / norm_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            weights = torch.where(positive_rows, weights / mean.clamp(min=1e-6), weights)
        if not torch.isfinite(weights).all():
            raise ValueError("AWAC weights contain non-finite values after clipping.")
        positive_rows = weights.sum(dim=1) > 0.0
        real_count = real_mask.float().sum().clamp(min=1.0)
        diagnostics = {
            "empty_awac_row_ratio": (~positive_rows).float().mean(),
            "positive_weight_row_ratio": positive_rows.float().mean(),
            "positive_weight_candidate_ratio": ((weights > 0) & real_mask).float().sum() / real_count,
            "target_filter_row_ratio": (filter_mask & real_mask).any(dim=1).float().mean(),
            "target_filter_candidate_ratio": (filter_mask & real_mask).float().sum() / real_count,
        }
        if (not bool(cfg.allow_zero_weight_rows)) and bool((~positive_rows).any().item()):
            raise RuntimeError("AWAC/IQL produced zero positive-weight rows and allow_zero_weight_rows=False.")
        return weights.float(), adv.float(), diagnostics

    def _apply_awac_source_balance(
        self,
        weights: torch.Tensor,
        source_code: torch.Tensor,
        real_mask: torch.Tensor,
        cfg: OfflineRLConfig,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = weights.new_zeros(())
        if not bool(cfg.source_balance_enabled):
            return weights, {
                "source_balance_factor_mean": weights.new_tensor(1.0),
                "source_balance_factor_min": weights.new_tensor(1.0),
                "source_balance_factor_max": weights.new_tensor(1.0),
                "source_balance_active_sources": zero,
            }

        balanced = weights.float().clone()
        active_mask = (balanced > 0.0) & real_mask
        if not bool(active_mask.any().item()):
            return balanced, {
                "source_balance_factor_mean": weights.new_tensor(1.0),
                "source_balance_factor_min": weights.new_tensor(1.0),
                "source_balance_factor_max": weights.new_tensor(1.0),
                "source_balance_active_sources": zero,
            }

        factors = []
        source_sums = []
        active_codes = []
        for code in range(1, 7):
            mask = active_mask & (source_code == code)
            source_sum = balanced[mask].sum()
            if float(source_sum.detach().cpu().item()) > 0.0:
                active_codes.append(code)
                source_sums.append(source_sum)
        if not source_sums:
            return balanced, {
                "source_balance_factor_mean": weights.new_tensor(1.0),
                "source_balance_factor_min": weights.new_tensor(1.0),
                "source_balance_factor_max": weights.new_tensor(1.0),
                "source_balance_active_sources": zero,
            }

        total = torch.stack(source_sums).sum()
        target_per_source = total / max(1, len(source_sums))
        for code, source_sum in zip(active_codes, source_sums):
            factor = (target_per_source / source_sum.clamp(min=1e-6)).clamp(
                float(cfg.source_balance_min_factor),
                float(cfg.source_balance_max_factor),
            )
            balanced = torch.where((source_code == code) & active_mask, balanced * factor, balanced)
            factors.append(factor)

        factor_tensor = torch.stack(factors) if factors else weights.new_ones((1,))
        return balanced.to(weights), {
            "source_balance_factor_mean": factor_tensor.mean().to(weights),
            "source_balance_factor_min": factor_tensor.min().to(weights),
            "source_balance_factor_max": factor_tensor.max().to(weights),
            "source_balance_active_sources": weights.new_tensor(float(len(active_codes))),
        }

    def _apply_awac_target_blend(
        self,
        target_trajs: torch.Tensor,
        rewards: torch.Tensor,
        source_code: torch.Tensor,
        real_mask: torch.Tensor,
        action_input: BatchFeature,
        cfg: OfflineRLConfig,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = target_trajs.new_zeros(())
        alpha = self._current_awac_target_blend_alpha(cfg)
        if str(cfg.target_blend_mode) == "none" or alpha >= 1.0:
            return target_trajs, {
                "target_blend_alpha_effective": target_trajs.new_tensor(1.0),
                "target_blend_l2_to_original": zero,
                "target_blend_anchor_gt_ratio": zero,
                "target_blend_anchor_il_ratio": zero,
                "target_blend_anchor_fallback_gt_ratio": zero,
            }
        if target_trajs.ndim != 4 or rewards.shape != target_trajs.shape[:2]:
            raise ValueError("target_trajs must be [B, M, H, 3] and rewards/source masks must be [B, M].")
        if not hasattr(action_input, "action"):
            raise KeyError("AWAC target blending requires action_input.action as the GT fallback anchor.")

        B, _, H, D = target_trajs.shape
        gt_action = action_input.action.detach().to(device=target_trajs.device, dtype=target_trajs.dtype)
        if gt_action.shape != (B, H, D):
            raise ValueError(f"action_input.action shape {tuple(gt_action.shape)} does not match targets {(B, H, D)}.")

        anchors = []
        anchor_codes = []
        for b in range(B):
            anchor_mask = real_mask[b] & ((source_code[b] == 1) | (source_code[b] == 2))
            anchor_idx = torch.nonzero(anchor_mask, as_tuple=False).flatten()
            if anchor_idx.numel() == 0:
                anchors.append(gt_action[b])
                anchor_codes.append(0)
                continue
            row_rewards = rewards[b, anchor_idx].float()
            best_pos = int(torch.argmax(row_rewards).item())
            idx = anchor_idx[best_pos]
            anchors.append(target_trajs[b, idx].detach())
            anchor_codes.append(int(source_code[b, idx].detach().cpu().item()))

        anchor = torch.stack(anchors, dim=0).to(target_trajs)
        blended = anchor[:, None] + alpha * (target_trajs - anchor[:, None])
        blended = torch.where(real_mask[:, :, None, None], blended, target_trajs)
        if not torch.isfinite(blended).all():
            raise ValueError("AWAC blended target trajectories contain non-finite values.")

        anchor_code_tensor = torch.tensor(anchor_codes, device=target_trajs.device)
        blend_delta = (blended - target_trajs).detach().float().norm(dim=-1).mean()
        return blended, {
            "target_blend_alpha_effective": target_trajs.new_tensor(alpha),
            "target_blend_l2_to_original": blend_delta.to(target_trajs),
            "target_blend_anchor_gt_ratio": (anchor_code_tensor == 1).float().mean().to(target_trajs),
            "target_blend_anchor_il_ratio": (anchor_code_tensor == 2).float().mean().to(target_trajs),
            "target_blend_anchor_fallback_gt_ratio": (anchor_code_tensor == 0).float().mean().to(target_trajs),
        }

    def _repeat_action_input_for_loss(self, action_input: BatchFeature, repeat: int) -> BatchFeature:
        repeated = self._repeat_expert_action_input(action_input, repeat)
        data: Dict[str, Any] = dict(repeated) if repeated is not None else {}
        for key in ("his_traj", "history_trajectory", "status_feature", "high_command_one_hot", "state"):
            if key in action_input and isinstance(action_input[key], torch.Tensor) and key not in data:
                data[key] = action_input[key].repeat_interleave(repeat, 0)
        return BatchFeature(data=data)

    def _diffusion_per_target_loss_on_targets(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        target_trajs: torch.Tensor,
        *,
        policy: Optional["ReCogDriveDiffusionPlanner"] = None,
        noise: Optional[torch.Tensor] = None,
        t_discrete: Optional[torch.Tensor] = None,
        timestep_sampling: str = "uniform",
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        if target_trajs.ndim != 4 or target_trajs.shape[-1] != 3:
            raise ValueError(f"target_trajs must have shape [B, M, H, 3], got {tuple(target_trajs.shape)}.")
        if self.config.sampling_method == "flow":
            raise NotImplementedError("Offline preference diffusion losses are implemented for DDPM/DDIM, not flow.")
        policy = self if policy is None else policy
        B, M, H, D = target_trajs.shape
        targets = target_trajs.reshape(B * M, H, D).detach()
        if not torch.isfinite(targets).all():
            raise ValueError("AWAC target trajectories contain non-finite values.")
        target_norm = policy.norm_odo(targets).clamp(-1.0, 1.0)
        if noise is None:
            noise = torch.randn_like(target_norm)
        else:
            noise = noise.to(device=target_norm.device, dtype=target_norm.dtype)
        if t_discrete is None:
            t_discrete = policy._sample_offline_rl_timesteps(
                B * M,
                device=target_norm.device,
                dtype=target_norm.dtype,
                mode=timestep_sampling,
            )
        else:
            t_discrete = t_discrete.to(device=target_norm.device)
        noisy_actions = (
            policy.extract(policy.ddpm_sqrt_alphas_cumprod, t_discrete, target_norm.shape) * target_norm
            + policy.extract(policy.ddpm_sqrt_one_minus_alphas_cumprod, t_discrete, target_norm.shape) * noise
        )
        vl_features_rep = vl_features.repeat_interleave(M, 0)
        action_input_rep = policy._repeat_action_input_for_loss(action_input, M)
        dit_context = policy._prepare_dit_context(
            vl_features_rep,
            action_input_rep,
            training=policy.training,
            noisy_actions=noisy_actions,
            diffusion_timestep=t_discrete,
            allow_target_tokens=False,
        )
        pred_noise = policy._denoise_model_output(noisy_actions, t_discrete, dit_context, action_input_rep)
        per_sample_loss = ((pred_noise.float() - noise.float()) ** 2).mean(dim=(1, 2))
        if per_sample_loss.shape != (B * M,):
            raise ValueError(f"per_sample_loss must have shape [B*M], got {tuple(per_sample_loss.shape)}.")
        diagnostics = {
            "per_sample_loss_mean": per_sample_loss.detach().mean(),
            "target_norm_mean": target_norm.detach().float().abs().mean(),
            "diffusion_timestep_mean": t_discrete.detach().float().mean(),
            "diffusion_timestep_min": t_discrete.detach().float().min(),
            "diffusion_timestep_max": t_discrete.detach().float().max(),
        }
        return per_sample_loss.reshape(B, M), diagnostics, noise.detach(), t_discrete.detach()

    def _weighted_diffusion_loss_on_targets(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        target_trajs: torch.Tensor,
        weights: torch.Tensor,
        return_per_sample_loss: bool = False,
        timestep_sampling: str = "uniform",
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if target_trajs.ndim != 4 or target_trajs.shape[-1] != 3:
            raise ValueError(f"target_trajs must have shape [B, M, H, 3], got {tuple(target_trajs.shape)}.")
        B, M, H, D = target_trajs.shape
        if weights.shape != (B, M):
            raise ValueError(f"weights must have shape [B, M], got {tuple(weights.shape)}.")
        flat_weights = weights.reshape(B * M).to(device=target_trajs.device, dtype=torch.float32)
        raw_weight_sum = flat_weights.sum()
        zero_weight_batch = raw_weight_sum.detach() <= 0.0
        if bool(zero_weight_batch.cpu().item()) and not return_per_sample_loss:
            zero_loss = next(self.parameters()).float().sum() * 0.0
            return zero_loss, {
                "per_sample_loss_mean": zero_loss.detach(),
                "target_norm_mean": zero_loss.detach(),
                "diffusion_timestep_mean": zero_loss.detach(),
                "diffusion_timestep_min": zero_loss.detach(),
                "diffusion_timestep_max": zero_loss.detach(),
                "effective_weight_sum": raw_weight_sum.detach().to(device=zero_loss.device, dtype=zero_loss.dtype),
                "zero_weight_ratio": flat_weights.new_tensor(1.0).to(device=zero_loss.device, dtype=zero_loss.dtype),
                "zero_weight_batch": flat_weights.new_tensor(1.0).to(device=zero_loss.device, dtype=zero_loss.dtype),
                **(
                    {"per_sample_loss_matrix": target_trajs.new_zeros((B, M), dtype=torch.float32)}
                    if return_per_sample_loss
                    else {}
                ),
            }
        per_target_loss, loss_diag, _, _ = self._diffusion_per_target_loss_on_targets(
            vl_features,
            action_input,
            target_trajs,
            timestep_sampling=timestep_sampling,
        )
        per_sample_loss = per_target_loss.reshape(B * M)
        valid_weight_sum = raw_weight_sum.clamp(min=1e-6)
        awac_loss = (flat_weights.to(per_sample_loss) * per_sample_loss).sum() / valid_weight_sum.to(per_sample_loss)
        if not torch.isfinite(awac_loss):
            raise ValueError("AWAC weighted diffusion loss is non-finite.")
        diagnostics = {
            "per_sample_loss_mean": loss_diag["per_sample_loss_mean"].to(device=awac_loss.device, dtype=awac_loss.dtype),
            "target_norm_mean": loss_diag["target_norm_mean"].to(device=awac_loss.device, dtype=awac_loss.dtype),
            "diffusion_timestep_mean": loss_diag["diffusion_timestep_mean"].to(device=awac_loss.device, dtype=awac_loss.dtype),
            "diffusion_timestep_min": loss_diag["diffusion_timestep_min"].to(device=awac_loss.device, dtype=awac_loss.dtype),
            "diffusion_timestep_max": loss_diag["diffusion_timestep_max"].to(device=awac_loss.device, dtype=awac_loss.dtype),
            "effective_weight_sum": valid_weight_sum.detach().to(dtype=awac_loss.dtype),
            "zero_weight_ratio": (flat_weights <= 0).float().mean().to(device=awac_loss.device, dtype=awac_loss.dtype),
            "zero_weight_batch": zero_weight_batch.to(device=awac_loss.device, dtype=awac_loss.dtype),
        }
        if return_per_sample_loss:
            diagnostics["per_sample_loss_matrix"] = per_target_loss
        return awac_loss, diagnostics

    def _compute_awac_preference_losses(
        self,
        per_target_loss: torch.Tensor,
        ordering_rewards: torch.Tensor,
        valid_mask: torch.Tensor,
        real_mask: torch.Tensor,
        cfg: OfflineRLConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if per_target_loss.shape != ordering_rewards.shape:
            raise ValueError(
                "per_target_loss and ordering_rewards must have the same shape, got "
                f"{tuple(per_target_loss.shape)} and {tuple(ordering_rewards.shape)}."
            )
        zero = per_target_loss.sum() * 0.0
        rank_terms = []
        invalid_terms = []
        rank_pair_count = 0
        invalid_pair_count = 0
        max_rank_pairs = int(cfg.pairwise_rank_max_pairs_per_scene)
        max_invalid_pairs = int(cfg.invalid_repulsion_max_pairs_per_scene)
        rank_enabled = float(cfg.pairwise_rank_loss_weight) > 0.0 and max_rank_pairs > 0
        invalid_enabled = float(cfg.invalid_repulsion_loss_weight) > 0.0 and max_invalid_pairs > 0
        if not rank_enabled and not invalid_enabled:
            return zero, zero, {
                "pairwise_rank_pair_count": per_target_loss.new_zeros(()),
                "invalid_repulsion_pair_count": per_target_loss.new_zeros(()),
                "pairwise_rank_active_row_ratio": per_target_loss.new_zeros(()),
                "invalid_repulsion_active_row_ratio": per_target_loss.new_zeros(()),
            }

        rank_rows = 0
        invalid_rows = 0
        B = int(per_target_loss.shape[0])
        for b in range(B):
            row_real = real_mask[b]
            row_valid = valid_mask[b] & row_real
            valid_idx = torch.nonzero(row_valid, as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                continue
            valid_rewards = ordering_rewards[b, valid_idx].float()
            high_pos = int(torch.argmax(valid_rewards).item())
            high_idx = valid_idx[high_pos]
            high_reward = ordering_rewards[b, high_idx].float()
            high_loss = per_target_loss[b, high_idx]

            if rank_enabled and valid_idx.numel() > 1:
                reward_gap = high_reward - ordering_rewards[b, valid_idx].float()
                low_mask = reward_gap >= float(cfg.pairwise_rank_min_reward_gap)
                low_mask[high_pos] = False
                low_idx = valid_idx[low_mask]
                if low_idx.numel() > 0:
                    low_rewards = ordering_rewards[b, low_idx].float()
                    order = torch.argsort(low_rewards, descending=False)
                    low_idx = low_idx[order[:max_rank_pairs]]
                    terms = F.relu(float(cfg.pairwise_rank_margin) + high_loss - per_target_loss[b, low_idx])
                    rank_terms.append(terms.mean())
                    rank_pair_count += int(low_idx.numel())
                    rank_rows += 1

            if invalid_enabled:
                invalid_idx = torch.nonzero((~valid_mask[b]) & row_real, as_tuple=False).flatten()
                if invalid_idx.numel() > 0:
                    invalid_rewards = ordering_rewards[b, invalid_idx].float()
                    order = torch.argsort(invalid_rewards, descending=True)
                    invalid_idx = invalid_idx[order[:max_invalid_pairs]]
                    terms = F.relu(float(cfg.invalid_repulsion_margin) + high_loss - per_target_loss[b, invalid_idx])
                    invalid_terms.append(terms.mean())
                    invalid_pair_count += int(invalid_idx.numel())
                    invalid_rows += 1

        rank_loss = torch.stack(rank_terms).mean() if rank_terms else zero
        invalid_loss = torch.stack(invalid_terms).mean() if invalid_terms else zero
        return rank_loss, invalid_loss, {
            "pairwise_rank_pair_count": per_target_loss.new_tensor(float(rank_pair_count)),
            "invalid_repulsion_pair_count": per_target_loss.new_tensor(float(invalid_pair_count)),
            "pairwise_rank_active_row_ratio": per_target_loss.new_tensor(float(rank_rows) / max(1, B)),
            "invalid_repulsion_active_row_ratio": per_target_loss.new_tensor(float(invalid_rows) / max(1, B)),
        }

    def _compute_diffusion_dpo_preference_loss(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        target_trajs: torch.Tensor,
        ordering_rewards: torch.Tensor,
        valid_mask: torch.Tensor,
        real_mask: torch.Tensor,
        source_code: torch.Tensor,
        cfg: OfflineRLConfig,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = target_trajs.sum() * 0.0
        max_pairs = int(cfg.preference_dpo_max_pairs_per_scene)
        enabled = float(cfg.preference_dpo_loss_weight) > 0.0 and max_pairs > 0
        if not enabled:
            return zero, {
                "preference_dpo_pair_count": zero.detach(),
                "preference_dpo_active_row_ratio": zero.detach(),
                "preference_dpo_reward_gap_mean": zero.detach(),
                "preference_dpo_gap_weight_mean": zero.detach(),
                "preference_dpo_current_logratio_mean": zero.detach(),
                "preference_dpo_reference_logratio_mean": zero.detach(),
                "preference_dpo_timestep_mean": zero.detach(),
                "preference_dpo_timestep_min": zero.detach(),
                "preference_dpo_timestep_max": zero.detach(),
            }
        if ordering_rewards.shape != valid_mask.shape or ordering_rewards.shape != real_mask.shape:
            raise ValueError("DPO ordering_rewards, valid_mask, and real_mask must have matching [B, M] shapes.")
        if not bool(cfg.preference_dpo_reference_free) and not hasattr(self, "old_policy"):
            raise RuntimeError("Diffusion-DPO preference loss requires old_policy unless preference_dpo_reference_free=True.")

        current_loss, current_diag, noise, t_discrete = self._diffusion_per_target_loss_on_targets(
            vl_features,
            action_input,
            target_trajs,
            timestep_sampling=cfg.preference_dpo_timestep_sampling,
        )
        if bool(cfg.preference_dpo_reference_free):
            reference_loss = torch.zeros_like(current_loss)
        else:
            self.old_policy.eval()
            with torch.no_grad():
                reference_loss, _, _, _ = self._diffusion_per_target_loss_on_targets(
                    vl_features,
                    action_input,
                    target_trajs,
                    policy=self.old_policy,
                    noise=noise,
                    t_discrete=t_discrete,
                )
            reference_loss = reference_loss.detach().to(current_loss)

        pair_terms = []
        reward_gaps = []
        gap_weights = []
        current_logratios = []
        reference_logratios = []
        active_rows = 0
        pair_count = 0
        min_gap = float(cfg.preference_dpo_min_reward_gap)
        label_smoothing = float(cfg.preference_dpo_label_smoothing)
        beta = float(cfg.preference_dpo_beta)
        pair_mode = str(cfg.preference_dpo_pair_mode)
        gap_weight_mode = str(cfg.preference_dpo_gap_weight_mode)
        gap_weight_scale = float(cfg.preference_dpo_gap_weight_scale)
        gap_weight_min = float(cfg.preference_dpo_gap_weight_min)
        gap_weight_max = float(cfg.preference_dpo_gap_weight_max)
        B = int(ordering_rewards.shape[0])
        for b in range(B):
            row_valid = valid_mask[b] & real_mask[b]
            valid_idx = torch.nonzero(row_valid, as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                continue
            valid_rewards = ordering_rewards[b, valid_idx].float()
            winner_pos = int(torch.argmax(valid_rewards).item())
            winner_idx = valid_idx[winner_pos]
            winner_reward = ordering_rewards[b, winner_idx].float()

            if pair_mode == "best_vs_gt_il":
                loser_mask = real_mask[b] & ((source_code[b] == 1) | (source_code[b] == 2))
            elif pair_mode == "best_vs_low_valid":
                loser_mask = row_valid.clone()
            else:
                loser_mask = real_mask[b].clone()
            loser_mask[winner_idx] = False
            loser_idx = torch.nonzero(loser_mask, as_tuple=False).flatten()
            if loser_idx.numel() == 0:
                continue
            reward_gap = winner_reward - ordering_rewards[b, loser_idx].float()
            loser_idx = loser_idx[reward_gap >= min_gap]
            reward_gap = winner_reward - ordering_rewards[b, loser_idx].float()
            if loser_idx.numel() == 0:
                continue
            if pair_mode == "best_vs_gt_il":
                order = torch.argsort(reward_gap, descending=True)
            else:
                loser_rewards = ordering_rewards[b, loser_idx].float()
                order = torch.argsort(loser_rewards, descending=False)
            loser_idx = loser_idx[order[:max_pairs]]
            reward_gap = winner_reward - ordering_rewards[b, loser_idx].float()

            current_logratio = current_loss[b, loser_idx] - current_loss[b, winner_idx]
            reference_logratio = reference_loss[b, loser_idx] - reference_loss[b, winner_idx]
            logits = beta * (current_logratio - reference_logratio)
            loss = (
                -(1.0 - label_smoothing) * F.logsigmoid(logits)
                - label_smoothing * F.logsigmoid(-logits)
            )
            if gap_weight_mode == "pair_gap":
                pair_weight = (reward_gap.to(current_loss) / gap_weight_scale).clamp(
                    min=gap_weight_min,
                    max=gap_weight_max,
                )
            elif gap_weight_mode == "winner_advantage":
                anchor_mask = real_mask[b] & ((source_code[b] == 1) | (source_code[b] == 2))
                if bool(anchor_mask.any().item()):
                    anchor_reward = ordering_rewards[b, anchor_mask].float().max()
                else:
                    anchor_reward = ordering_rewards[b, valid_idx].float().mean()
                winner_advantage = (winner_reward - anchor_reward).clamp(min=0.0)
                pair_weight = (winner_advantage.to(current_loss) / gap_weight_scale).clamp(
                    min=gap_weight_min,
                    max=gap_weight_max,
                ).expand_as(loss)
            else:
                pair_weight = torch.ones_like(loss)
            loss = loss * pair_weight
            pair_terms.append(loss.mean())
            reward_gaps.append(reward_gap.detach().float().mean())
            gap_weights.append(pair_weight.detach().float().mean())
            current_logratios.append(current_logratio.detach().float().mean())
            reference_logratios.append(reference_logratio.detach().float().mean())
            pair_count += int(loser_idx.numel())
            active_rows += 1

        if not pair_terms:
            return zero, {
                "preference_dpo_pair_count": zero.detach(),
                "preference_dpo_active_row_ratio": zero.detach(),
                "preference_dpo_reward_gap_mean": zero.detach(),
                "preference_dpo_gap_weight_mean": zero.detach(),
                "preference_dpo_current_logratio_mean": zero.detach(),
                "preference_dpo_reference_logratio_mean": zero.detach(),
                "preference_dpo_timestep_mean": current_diag["diffusion_timestep_mean"].to(zero).detach(),
                "preference_dpo_timestep_min": current_diag["diffusion_timestep_min"].to(zero).detach(),
                "preference_dpo_timestep_max": current_diag["diffusion_timestep_max"].to(zero).detach(),
            }

        dpo_loss = torch.stack(pair_terms).mean()
        return dpo_loss, {
            "preference_dpo_pair_count": current_loss.new_tensor(float(pair_count)),
            "preference_dpo_active_row_ratio": current_loss.new_tensor(float(active_rows) / max(1, B)),
            "preference_dpo_reward_gap_mean": torch.stack(reward_gaps).mean().to(current_loss),
            "preference_dpo_gap_weight_mean": torch.stack(gap_weights).mean().to(current_loss),
            "preference_dpo_current_logratio_mean": torch.stack(current_logratios).mean().to(current_loss),
            "preference_dpo_reference_logratio_mean": torch.stack(reference_logratios).mean().to(current_loss),
            "preference_dpo_timestep_mean": current_diag["diffusion_timestep_mean"].to(current_loss),
            "preference_dpo_timestep_min": current_diag["diffusion_timestep_min"].to(current_loss),
            "preference_dpo_timestep_max": current_diag["diffusion_timestep_max"].to(current_loss),
        }

    def _load_awac_buffer_candidates(
        self,
        action_input: BatchFeature,
        tokens_list: list[str],
        metric_cache: Dict[str, Any],
        cfg: OfflineRLConfig,
    ) -> Dict[str, Any]:
        buffer_root = Path(cfg.elite_buffer_path)
        if not str(buffer_root):
            raise ValueError("offline_rl_cfg.elite_buffer_path is empty and build_candidates_online=False.")
        device = action_input.action.device
        dtype = action_input.action.dtype
        records = []
        for batch_idx, token in enumerate(tokens_list):
            try:
                records.append(self._load_elite_record_cached(buffer_root, str(token), cfg))
            except FileNotFoundError:
                if str(cfg.missing_buffer_policy) != "fallback_gt":
                    raise
                gt = action_input.action[batch_idx : batch_idx + 1].detach().float()
                rewards, components = self._score_candidate_trajectories(gt[:, None], [str(token)], metric_cache, cfg)
                valid_mask, _ = self._compute_awac_candidate_valid_mask(components, ["gt"], cfg)
                selection_score = rewards[0].detach().cpu().numpy().astype(np.float32)
                record = {
                    "token": str(token),
                    "candidates": gt[0].cpu().numpy()[None],
                    "rewards": rewards[0].detach().cpu().numpy(),
                    "components": {key: value[0].detach().cpu().numpy() for key, value in components.items()},
                    "sources": ["gt"],
                    "anchor_distance": np.zeros((1,), dtype=np.float32),
                    "valid_mask": valid_mask[0].detach().cpu().numpy(),
                    "selection_score": selection_score,
                    "gt_reward": float(rewards[0, 0].detach().cpu().item()),
                    "il_reward": float(rewards[0, 0].detach().cpu().item()),
                    "best_reward": float(rewards[0, 0].detach().cpu().item()),
                    "best_source": "gt",
                    "best_raw_reward": float(rewards[0, 0].detach().cpu().item()),
                    "best_valid_reward": float(rewards[0, 0].detach().cpu().item()),
                    "best_selected_reward": float(rewards[0, 0].detach().cpu().item()),
                    "best_raw_source": "gt",
                    "best_valid_source": "gt",
                    "best_selected_source": "gt",
                    "has_valid_candidate": bool(valid_mask[0, 0].detach().cpu().item()),
                    "version": 2,
                }
                records.append(record)

        B = len(records)
        max_m = max(int(np.asarray(record["candidates"]).shape[0]) for record in records)
        H, D = action_input.action.shape[-2], action_input.action.shape[-1]
        selected_trajs = torch.zeros((B, max_m, H, D), device=device, dtype=dtype)
        selected_rewards = torch.zeros((B, max_m), device=device, dtype=torch.float32)
        selected_anchor_distance = torch.zeros((B, max_m), device=device, dtype=torch.float32)
        selected_real_mask = torch.zeros((B, max_m), device=device, dtype=torch.bool)
        selected_valid_mask = torch.zeros((B, max_m), device=device, dtype=torch.bool)
        selected_source_code = torch.zeros((B, max_m), device=device, dtype=torch.long)
        selected_source_index = torch.full((B, max_m), -1, device=device, dtype=torch.long)
        selected_selection_score = torch.zeros((B, max_m), device=device, dtype=torch.float32)
        selected_components = {
            key: torch.zeros((B, max_m), device=device, dtype=torch.float32)
            for key in REQUIRED_COMPONENT_KEYS
        }
        gt_reward = torch.zeros((B,), device=device, dtype=torch.float32)
        il_reward = torch.zeros((B,), device=device, dtype=torch.float32)
        best_raw_reward = torch.zeros((B,), device=device, dtype=torch.float32)
        best_valid_reward = torch.zeros((B,), device=device, dtype=torch.float32)
        best_selected_reward = torch.zeros((B,), device=device, dtype=torch.float32)
        best_raw_source_code = torch.zeros((B,), device=device, dtype=torch.long)
        best_valid_source_code = torch.zeros((B,), device=device, dtype=torch.long)
        best_selected_source_code = torch.zeros((B,), device=device, dtype=torch.long)
        has_valid_candidate = torch.zeros((B,), device=device, dtype=torch.bool)
        fallback_candidate = torch.zeros((B,), device=device, dtype=torch.bool)
        for b, record in enumerate(records):
            candidates_np = np.asarray(record["candidates"], dtype=np.float32)
            if candidates_np.ndim != 3 or candidates_np.shape[-1] != 3:
                raise ValueError(f"Elite buffer token={record['token']!r} has invalid candidate shape {candidates_np.shape}.")
            m = candidates_np.shape[0]
            sources = [str(source) for source in record["sources"]]
            selected_trajs[b, :m] = torch.from_numpy(candidates_np).to(device=device, dtype=dtype)
            selected_rewards[b, :m] = torch.as_tensor(record["rewards"], device=device, dtype=torch.float32)
            selected_anchor_distance[b, :m] = torch.as_tensor(record["anchor_distance"], device=device, dtype=torch.float32)
            selected_real_mask[b, :m] = True
            for key in REQUIRED_COMPONENT_KEYS:
                selected_components[key][b, :m] = torch.as_tensor(
                    record["components"][key],
                    device=device,
                    dtype=torch.float32,
                )
            valid_mask_recomputed = False
            if bool(cfg.recompute_buffer_valid_mask_on_load):
                row_components = {
                    key: selected_components[key][b : b + 1, :m]
                    for key in REQUIRED_COMPONENT_KEYS
                }
                valid_mask = self._compute_awac_candidate_valid_mask(row_components, sources, cfg)[0][0]
                valid_mask_recomputed = True
            elif "valid_mask" in record:
                valid_mask = torch.as_tensor(record["valid_mask"], device=device, dtype=torch.bool)
            elif bool(cfg.allow_v1_buffer_recompute_valid_mask):
                row_components = {
                    key: selected_components[key][b : b + 1, :m]
                    for key in REQUIRED_COMPONENT_KEYS
                }
                valid_mask = self._compute_awac_candidate_valid_mask(row_components, sources, cfg)[0][0]
                valid_mask_recomputed = True
            elif bool(cfg.require_buffer_valid_mask):
                raise KeyError(
                    f"Elite buffer token={record['token']!r} is missing valid_mask. "
                    "Rebuild v2 buffer or set allow_v1_buffer_recompute_valid_mask=True."
                )
            else:
                valid_mask = torch.zeros((m,), device=device, dtype=torch.bool)
            selected_valid_mask[b, :m] = valid_mask

            if "selection_score" in record:
                selection_score = torch.as_tensor(record["selection_score"], device=device, dtype=torch.float32)
            else:
                trajs = selected_trajs[b : b + 1, :m].float()
                jerk = self._trajectory_jerk_penalty(trajs)[0].to(device=device, dtype=torch.float32)
                selection_score = (
                    selected_rewards[b, :m]
                    - float(cfg.prior_distance_weight) * selected_anchor_distance[b, :m]
                    - float(cfg.jerk_penalty_weight) * jerk
                )
            selected_selection_score[b, :m] = selection_score
            selected_source_index[b, :m] = torch.arange(m, device=device, dtype=torch.long)
            source_codes = [self._source_code(source) for source in sources]
            selected_source_code[b, :m] = torch.tensor(source_codes, device=device, dtype=torch.long)
            gt_reward[b] = float(record["gt_reward"])
            il_reward[b] = float(record["il_reward"])
            has_valid = bool(valid_mask.any().detach().cpu().item()) if valid_mask_recomputed else bool(
                record.get("has_valid_candidate", bool(valid_mask.any().detach().cpu().item()))
            )
            has_valid_candidate[b] = has_valid
            fallback_candidate[b] = not has_valid
            raw_idx = int(selected_rewards[b, :m].argmax().item())
            if bool(valid_mask.any().item()):
                valid_rewards = selected_rewards[b, :m].masked_fill(~valid_mask, -torch.inf)
                valid_idx = int(valid_rewards.argmax().item())
            else:
                valid_idx = raw_idx
            selected_idx = raw_idx
            best_raw_reward[b] = float(record.get("best_raw_reward", float(selected_rewards[b, raw_idx].item())))
            best_valid_reward[b] = float(selected_rewards[b, valid_idx].item()) if valid_mask_recomputed else float(
                record.get("best_valid_reward", float(selected_rewards[b, valid_idx].item()))
            )
            best_selected_reward[b] = float(record.get("best_selected_reward", float(selected_rewards[b, selected_idx].item())))
            best_raw_source_code[b] = self._source_code(record.get("best_raw_source", sources[raw_idx]))
            best_valid_source_code[b] = self._source_code(sources[valid_idx]) if valid_mask_recomputed else self._source_code(
                record.get("best_valid_source", sources[valid_idx])
            )
            best_selected_source_code[b] = self._source_code(record.get("best_selected_source", sources[selected_idx]))
        valid_real = selected_valid_mask & selected_real_mask
        real_count = selected_real_mask.float().sum().clamp(min=1.0)
        return {
            "selected_trajs": selected_trajs,
            "selected_rewards": selected_rewards,
            "selected_components": selected_components,
            "selected_anchor_distance": selected_anchor_distance,
            "selected_selection_score": selected_selection_score,
            "selected_valid_mask": selected_valid_mask,
            "selected_real_mask": selected_real_mask,
            "selected_source_code": selected_source_code,
            "selected_source_index": selected_source_index,
            "valid_candidate_ratio": valid_real.float().sum() / real_count,
            "has_valid_candidate_ratio": has_valid_candidate.float().mean(),
            "fallback_candidate_ratio": fallback_candidate.float().mean(),
            "fallback_candidate": fallback_candidate,
            "has_valid_candidate": has_valid_candidate,
            "gt_reward": gt_reward,
            "il_reward": il_reward,
            "best_raw_reward": best_raw_reward,
            "best_valid_reward": best_valid_reward,
            "best_selected_reward": best_selected_reward,
            "best_raw_source_code": best_raw_source_code,
            "best_valid_source_code": best_valid_source_code,
            "best_selected_source_code": best_selected_source_code,
            "best_reward": best_valid_reward,
            "best_source_code": best_valid_source_code,
        }

    def _build_online_awac_candidates(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        tokens_list,
        metric_cache: Dict[str, Any],
        cfg: OfflineRLConfig,
    ) -> Dict[str, Any]:
        B, H, D = action_input.action.shape
        groups = []
        sources: list[str] = []
        distances = []
        gt = action_input.action.detach()
        if bool(cfg.online_use_gt):
            groups.append(gt[:, None])
            sources.append("gt")
            distances.append(gt.new_zeros((B, 1)))

        il_traj = None
        if bool(cfg.online_use_old_policy):
            if not hasattr(self, "old_policy"):
                raise RuntimeError("AWAC online candidate generation requires old_policy; check reference_policy_checkpoint.")
            self.old_policy.eval()
            with torch.no_grad():
                _, il_traj = self.old_policy.sample_chain(
                    vl_features,
                    action_input.his_traj,
                    action_input.status_feature,
                    deterministic=False,
                    action_input=action_input,
                    allow_target_tokens=False,
                )
            il_traj = il_traj.to(device=gt.device, dtype=gt.dtype).detach()
            groups.append(il_traj[:, None])
            sources.append("il")
            distances.append(torch.linalg.norm(il_traj[..., :2] - gt[..., :2], dim=-1).mean(dim=-1, keepdim=True))

        policy_samples = int(cfg.online_policy_samples) if bool(cfg.online_use_current_policy) else 0
        if policy_samples > 0:
            with torch.no_grad():
                vl_features_rep = vl_features.repeat_interleave(policy_samples, 0)
                his_traj_rep = action_input.his_traj.repeat_interleave(policy_samples, 0)
                status_feature_rep = action_input.status_feature.repeat_interleave(policy_samples, 0)
                action_input_rep = self._repeat_expert_action_input(action_input, policy_samples)
                _, policy_trajs = self.sample_chain(
                    vl_features_rep,
                    his_traj_rep,
                    status_feature_rep,
                    deterministic=False,
                    action_input=action_input_rep,
                    allow_target_tokens=False,
                )
            policy_trajs = policy_trajs.to(device=gt.device, dtype=gt.dtype).reshape(B, policy_samples, H, D).detach()
            groups.append(policy_trajs)
            sources.extend(["policy"] * policy_samples)
            distances.append(torch.linalg.norm(policy_trajs[..., :2] - gt[:, None, :, :2], dim=-1).mean(dim=-1))

        anchors = []
        anchor_sources = []
        if bool(cfg.perturb_gt):
            anchors.append(gt)
            anchor_sources.append("gt")
        if bool(cfg.perturb_il) and il_traj is not None:
            anchors.append(il_traj)
            anchor_sources.append("il")
        if anchors:
            anchor_tensor = torch.stack(anchors, dim=1)
            perturbations, perturb_sources, perturb_distance = build_structured_perturbations(
                anchor_tensor.detach(),
                anchor_sources,
                cfg,
            )
            if perturbations.shape[1] > 0:
                groups.append(perturbations.to(device=gt.device, dtype=gt.dtype))
                sources.extend(perturb_sources)
                distances.append(perturb_distance.to(device=gt.device, dtype=gt.dtype))

        if not groups:
            raise RuntimeError("AWAC online candidate builder produced no candidates.")
        candidates = torch.cat(groups, dim=1).detach()
        anchor_distance = torch.cat(distances, dim=1).detach().to(candidates)
        if candidates.shape[:3] != (B, len(sources), H):
            raise ValueError(
                f"AWAC candidate shape/source mismatch: candidates={tuple(candidates.shape)} sources={len(sources)}."
            )
        rewards, components = self._score_candidate_trajectories(
            candidates,
            [str(token) for token in tokens_list],
            metric_cache,
            cfg,
        )
        gt_idx = sources.index("gt") if "gt" in sources else None
        il_idx = sources.index("il") if "il" in sources else None
        gt_reward = rewards[:, gt_idx] if gt_idx is not None else rewards.max(dim=1).values
        il_reward = rewards[:, il_idx] if il_idx is not None else gt_reward
        selected = self._select_elite_candidates(
            candidates,
            rewards,
            components,
            sources,
            anchor_distance,
            gt_reward,
            il_reward,
            cfg,
        )
        selected["candidate_rewards"] = rewards
        selected["candidate_components"] = components
        selected["candidate_sources"] = sources
        selected["candidate_anchor_distance"] = anchor_distance
        return selected

    def forward_awac_iql(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        tokens_list,
        use_bc_loss: bool = True,
    ) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        cfg = self.offline_rl_cfg
        if not bool(cfg.enabled):
            raise RuntimeError("forward_awac_iql requires offline_rl_cfg.enabled=True.")
        if tokens_list is None:
            raise ValueError("forward_awac_iql requires tokens_list for train metric-cache reward lookup.")
        token_strs = [str(token) for token in tokens_list]
        use_offline_buffer = bool(str(cfg.elite_buffer_path)) and not bool(cfg.build_candidates_online)
        metric_cache: Dict[str, Any] = {}
        if not use_offline_buffer or str(cfg.missing_buffer_policy) == "fallback_gt":
            metric_cache = self._load_metric_cache_for_tokens(token_strs)
        if use_offline_buffer:
            awac_batch = self._load_awac_buffer_candidates(action_input, token_strs, metric_cache, cfg)
        else:
            awac_batch = self._build_online_awac_candidates(
                vl_features,
                action_input,
                token_strs,
                metric_cache,
                cfg,
            )

        selected_rewards = awac_batch["selected_rewards"]
        selected_real_mask = awac_batch["selected_real_mask"]
        selected_valid_mask = awac_batch["selected_valid_mask"]
        gt_reward = awac_batch["gt_reward"]
        il_reward = awac_batch["il_reward"]
        selected_components = awac_batch["selected_components"]
        source_code = awac_batch["selected_source_code"]
        shaped_rewards, component_diag = self._compute_component_aware_rewards(
            selected_rewards,
            selected_components,
            source_code,
            selected_real_mask,
            selected_valid_mask,
            cfg,
        )
        shaped_gt_reward = gt_reward
        shaped_il_reward = il_reward
        baseline = self._compute_iql_baseline(shaped_rewards, shaped_gt_reward, shaped_il_reward, cfg, selected_real_mask)
        weights, advantages, weight_diag = self._compute_awac_weights(
            shaped_rewards,
            baseline,
            cfg,
            valid_mask=selected_valid_mask,
            real_mask=selected_real_mask,
            gt_reward=shaped_gt_reward,
        )
        weights, source_balance_diag = self._apply_awac_source_balance(
            weights,
            source_code,
            selected_real_mask,
            cfg,
        )
        loss_target_trajs, target_blend_diag = self._apply_awac_target_blend(
            awac_batch["selected_trajs"],
            selected_rewards,
            source_code,
            selected_real_mask,
            action_input,
            cfg,
        )
        awac_loss, awac_diag = self._weighted_diffusion_loss_on_targets(
            vl_features,
            action_input,
            loss_target_trajs,
            weights,
            timestep_sampling=cfg.awac_timestep_sampling,
            return_per_sample_loss=(
                float(cfg.pairwise_rank_loss_weight) > 0.0 or float(cfg.invalid_repulsion_loss_weight) > 0.0
            ),
        )
        per_target_loss = awac_diag.pop("per_sample_loss_matrix", None)
        if per_target_loss is None:
            rank_loss = awac_loss.new_zeros(())
            invalid_repulsion_loss = awac_loss.new_zeros(())
            preference_diag = {
                "pairwise_rank_pair_count": awac_loss.new_zeros(()),
                "invalid_repulsion_pair_count": awac_loss.new_zeros(()),
                "pairwise_rank_active_row_ratio": awac_loss.new_zeros(()),
                "invalid_repulsion_active_row_ratio": awac_loss.new_zeros(()),
            }
        else:
            rank_loss, invalid_repulsion_loss, preference_diag = self._compute_awac_preference_losses(
                per_target_loss,
                shaped_rewards,
                selected_valid_mask,
                selected_real_mask,
                cfg,
            )
        preference_dpo_loss_weight_effective = self._current_awac_preference_dpo_loss_weight(cfg)
        dpo_cfg = cfg
        if float(preference_dpo_loss_weight_effective) != float(cfg.preference_dpo_loss_weight):
            dpo_cfg = copy.copy(cfg)
            dpo_cfg.preference_dpo_loss_weight = float(preference_dpo_loss_weight_effective)
        dpo_loss, dpo_diag = self._compute_diffusion_dpo_preference_loss(
            vl_features,
            action_input,
            loss_target_trajs,
            shaped_rewards,
            selected_valid_mask,
            selected_real_mask,
            source_code,
            dpo_cfg,
        )

        awac_loss_weight_effective = self._current_awac_loss_weight(cfg)
        bc_loss_weight_effective = self._current_awac_bc_loss_weight(cfg)
        bc_loss = awac_loss.new_zeros(())
        if use_bc_loss and bc_loss_weight_effective > 0.0:
            gt_targets = action_input.action.detach()[:, None]
            gt_weights = torch.ones((gt_targets.shape[0], 1), device=gt_targets.device, dtype=torch.float32)
            bc_loss, _ = self._weighted_diffusion_loss_on_targets(vl_features, action_input, gt_targets, gt_weights)

        grpo_loss = awac_loss.new_zeros(())
        grpo_loss_weight_effective = self._current_awac_grpo_loss_weight(cfg)
        if grpo_loss_weight_effective > 0.0:
            if not hasattr(self, "gamma_denoising"):
                raise RuntimeError("offline_rl_grpo_loss_weight > 0 requires GRPO initialization; set agent.grpo=True.")
            grpo_out = self.forward_grpo(
                vl_features,
                action_input,
                tokens_list,
                sample_time=min(int(getattr(self, "grpo_sample_time", 8)), 8),
                use_bc_loss=False,
            )
            grpo_loss = grpo_out.loss.to(dtype=awac_loss.dtype)

        total_loss = (
            float(awac_loss_weight_effective) * awac_loss
            + float(cfg.pairwise_rank_loss_weight) * rank_loss
            + float(cfg.invalid_repulsion_loss_weight) * invalid_repulsion_loss
            + float(preference_dpo_loss_weight_effective) * dpo_loss
            + float(bc_loss_weight_effective) * bc_loss
            + float(grpo_loss_weight_effective) * grpo_loss
        )
        if not torch.isfinite(total_loss):
            raise ValueError("AWAC/IQL total loss is non-finite.")

        real_f = selected_real_mask.to(dtype=total_loss.dtype)
        valid_f = selected_valid_mask.to(dtype=total_loss.dtype)
        denom = real_f.sum().clamp(min=1.0)
        selected_reward_mean = (selected_rewards.to(total_loss) * real_f).sum() / denom
        selected_reward_max = selected_rewards.masked_fill(~selected_real_mask, -torch.inf).max().to(total_loss)
        best_raw_reward = awac_batch["best_raw_reward"].to(total_loss)
        best_valid_reward = awac_batch["best_valid_reward"].to(total_loss)
        best_selected_reward = awac_batch["best_selected_reward"].to(total_loss)
        best_reward = best_valid_reward
        weight_real = weights[selected_real_mask]
        adv_real = advantages[selected_real_mask]
        if weight_real.numel() == 0:
            weight_real = weights.reshape(-1)
            adv_real = advantages.reshape(-1)

        def comp_mean(key: str) -> torch.Tensor:
            value = selected_components[key].to(total_loss)
            return (value * real_f).sum() / denom

        def source_ratio(code: int) -> torch.Tensor:
            return (
                ((source_code == code) & selected_real_mask).float().sum()
                / selected_real_mask.float().sum().clamp(min=1.0)
            ).to(dtype=total_loss.dtype)

        pct_candidates_above_gt = ((selected_rewards > gt_reward[:, None]) & selected_real_mask).float().sum()
        pct_candidates_above_gt = pct_candidates_above_gt / selected_real_mask.float().sum().clamp(min=1.0)
        pct_candidates_above_il = ((selected_rewards > il_reward[:, None]) & selected_real_mask).float().sum()
        pct_candidates_above_il = pct_candidates_above_il / selected_real_mask.float().sum().clamp(min=1.0)
        pct_best_raw_above_gt = (best_raw_reward > gt_reward.to(best_raw_reward)).float().mean()
        pct_best_valid_above_gt = (best_valid_reward > gt_reward.to(best_valid_reward)).float().mean()
        pct_best_selected_above_gt = (best_selected_reward > gt_reward.to(best_selected_reward)).float().mean()
        pct_best_valid_above_il = (best_valid_reward > il_reward.to(best_valid_reward)).float().mean()
        pct_best_above_gt = pct_best_valid_above_gt
        pct_best_above_il = pct_best_valid_above_il
        selected_valid_ratio = (
            ((selected_valid_mask & selected_real_mask).float().sum())
            / selected_real_mask.float().sum().clamp(min=1.0)
        ).to(total_loss)

        zero = total_loss.new_zeros(())
        return BatchFeature(data={
            "loss": total_loss,
            "diffusion_loss": awac_loss.detach(),
            "jepa_alignment_loss": zero,
            "vggt_alignment_loss": zero,
            "reward": selected_reward_mean.detach(),
            "policy_loss": awac_loss,
            "awac_loss": awac_loss,
            "awac_pairwise_rank_loss": rank_loss,
            "awac_invalid_repulsion_loss": invalid_repulsion_loss,
            "awac_dpo_loss": dpo_loss,
            "bc_loss": bc_loss,
            "awac_loss_weight_effective": total_loss.new_tensor(float(awac_loss_weight_effective)).detach(),
            "preference_dpo_loss_weight_effective": total_loss.new_tensor(
                float(preference_dpo_loss_weight_effective)
            ).detach(),
            "bc_loss_weight_effective": total_loss.new_tensor(float(bc_loss_weight_effective)).detach(),
            "grpo_loss": grpo_loss,
            "grpo_loss_weight_effective": total_loss.new_tensor(float(grpo_loss_weight_effective)).detach(),
            "reward_mean": selected_reward_mean.detach(),
            "reward_max": selected_reward_max.detach(),
            "shaped_reward_mean": ((shaped_rewards.to(total_loss) * real_f).sum() / denom).detach(),
            "shaped_reward_delta_mean": component_diag["component_reward_delta_mean"].to(total_loss).detach(),
            "shaped_reward_delta_min": component_diag["component_reward_delta_min"].to(total_loss).detach(),
            "shaped_reward_delta_max": component_diag["component_reward_delta_max"].to(total_loss).detach(),
            "component_progress_bonus_mean": component_diag["component_progress_bonus_mean"].to(total_loss).detach(),
            "component_safety_penalty_mean": component_diag["component_safety_penalty_mean"].to(total_loss).detach(),
            "component_valid_reward_delta_mean": component_diag.get(
                "component_valid_reward_delta_mean",
                total_loss.new_zeros(()),
            ).to(total_loss).detach(),
            "gt_reward_mean": gt_reward.mean().to(total_loss).detach(),
            "il_reward_mean": il_reward.mean().to(total_loss).detach(),
            "best_raw_reward_mean": best_raw_reward.mean().detach(),
            "best_valid_reward_mean": best_valid_reward.mean().detach(),
            "best_selected_reward_mean": best_selected_reward.mean().detach(),
            "best_raw_minus_gt_mean": (best_raw_reward - gt_reward.to(best_raw_reward)).mean().detach(),
            "best_valid_minus_gt_mean": (best_valid_reward - gt_reward.to(best_valid_reward)).mean().detach(),
            "best_selected_minus_gt_mean": (best_selected_reward - gt_reward.to(best_selected_reward)).mean().detach(),
            "best_raw_minus_il_mean": (best_raw_reward - il_reward.to(best_raw_reward)).mean().detach(),
            "best_valid_minus_il_mean": (best_valid_reward - il_reward.to(best_valid_reward)).mean().detach(),
            "best_selected_minus_il_mean": (best_selected_reward - il_reward.to(best_selected_reward)).mean().detach(),
            "best_reward_mean": best_reward.mean().detach(),
            "best_minus_gt_mean": (best_reward - gt_reward.to(best_reward)).mean().detach(),
            "best_minus_il_mean": (best_reward - il_reward.to(best_reward)).mean().detach(),
            "pct_best_raw_above_gt": pct_best_raw_above_gt.to(total_loss).detach(),
            "pct_best_valid_above_gt": pct_best_valid_above_gt.to(total_loss).detach(),
            "pct_best_selected_above_gt": pct_best_selected_above_gt.to(total_loss).detach(),
            "pct_best_valid_above_il": pct_best_valid_above_il.to(total_loss).detach(),
            "pct_best_above_gt": pct_best_above_gt.to(total_loss).detach(),
            "pct_best_above_il": pct_best_above_il.to(total_loss).detach(),
            "pct_candidates_above_gt": pct_candidates_above_gt.to(total_loss).detach(),
            "pct_candidates_above_il": pct_candidates_above_il.to(total_loss).detach(),
            "selected_reward_mean": selected_reward_mean.detach(),
            "selected_reward_max": selected_reward_max.detach(),
            "awac_weight_mean": weight_real.mean().to(total_loss).detach(),
            "awac_weight_max": weight_real.max().to(total_loss).detach(),
            "awac_advantage_mean": adv_real.mean().to(total_loss).detach(),
            "awac_advantage_max": adv_real.max().to(total_loss).detach(),
            "source_balance_factor_mean": source_balance_diag["source_balance_factor_mean"].to(total_loss).detach(),
            "source_balance_factor_min": source_balance_diag["source_balance_factor_min"].to(total_loss).detach(),
            "source_balance_factor_max": source_balance_diag["source_balance_factor_max"].to(total_loss).detach(),
            "source_balance_active_sources": source_balance_diag["source_balance_active_sources"].to(total_loss).detach(),
            "target_blend_alpha_effective": target_blend_diag["target_blend_alpha_effective"].to(total_loss).detach(),
            "target_blend_l2_to_original": target_blend_diag["target_blend_l2_to_original"].to(total_loss).detach(),
            "target_blend_anchor_gt_ratio": target_blend_diag["target_blend_anchor_gt_ratio"].to(total_loss).detach(),
            "target_blend_anchor_il_ratio": target_blend_diag["target_blend_anchor_il_ratio"].to(total_loss).detach(),
            "target_blend_anchor_fallback_gt_ratio": target_blend_diag["target_blend_anchor_fallback_gt_ratio"].to(total_loss).detach(),
            "pairwise_rank_pair_count": preference_diag["pairwise_rank_pair_count"].to(total_loss).detach(),
            "invalid_repulsion_pair_count": preference_diag["invalid_repulsion_pair_count"].to(total_loss).detach(),
            "pairwise_rank_active_row_ratio": preference_diag["pairwise_rank_active_row_ratio"].to(total_loss).detach(),
            "invalid_repulsion_active_row_ratio": preference_diag["invalid_repulsion_active_row_ratio"].to(total_loss).detach(),
            "preference_dpo_pair_count": dpo_diag["preference_dpo_pair_count"].to(total_loss).detach(),
            "preference_dpo_active_row_ratio": dpo_diag["preference_dpo_active_row_ratio"].to(total_loss).detach(),
            "preference_dpo_reward_gap_mean": dpo_diag["preference_dpo_reward_gap_mean"].to(total_loss).detach(),
            "preference_dpo_gap_weight_mean": dpo_diag["preference_dpo_gap_weight_mean"].to(total_loss).detach(),
            "preference_dpo_current_logratio_mean": dpo_diag["preference_dpo_current_logratio_mean"].to(total_loss).detach(),
            "preference_dpo_reference_logratio_mean": dpo_diag["preference_dpo_reference_logratio_mean"].to(total_loss).detach(),
            "preference_dpo_timestep_mean": dpo_diag["preference_dpo_timestep_mean"].to(total_loss).detach(),
            "preference_dpo_timestep_min": dpo_diag["preference_dpo_timestep_min"].to(total_loss).detach(),
            "preference_dpo_timestep_max": dpo_diag["preference_dpo_timestep_max"].to(total_loss).detach(),
            "empty_awac_row_ratio": weight_diag["empty_awac_row_ratio"].to(total_loss).detach(),
            "positive_weight_row_ratio": weight_diag["positive_weight_row_ratio"].to(total_loss).detach(),
            "positive_weight_candidate_ratio": weight_diag["positive_weight_candidate_ratio"].to(total_loss).detach(),
            "target_filter_row_ratio": weight_diag["target_filter_row_ratio"].to(total_loss).detach(),
            "target_filter_candidate_ratio": weight_diag["target_filter_candidate_ratio"].to(total_loss).detach(),
            "has_valid_candidate_ratio": awac_batch["has_valid_candidate_ratio"].to(total_loss).detach(),
            "valid_candidate_ratio": awac_batch["valid_candidate_ratio"].to(total_loss).detach(),
            "selected_valid_ratio": selected_valid_ratio.detach(),
            "fallback_candidate_ratio": awac_batch["fallback_candidate_ratio"].to(total_loss).detach(),
            "selected_nc_mean": comp_mean("no_at_fault_collisions").detach(),
            "selected_dac_mean": comp_mean("drivable_area_compliance").detach(),
            "selected_ttc_mean": comp_mean("time_to_collision_within_bound").detach(),
            "selected_ep_mean": comp_mean("ego_progress").detach(),
            "selected_comfort_mean": comp_mean("history_comfort").detach(),
            "selected_ddc_mean": comp_mean("driving_direction_compliance").detach(),
            "selected_tlc_mean": comp_mean("traffic_light_compliance").detach(),
            "source_gt_ratio": source_ratio(1).detach(),
            "source_il_ratio": source_ratio(2).detach(),
            "source_policy_ratio": source_ratio(3).detach(),
            "source_progress_ratio": source_ratio(4).detach(),
            "source_lateral_ratio": source_ratio(5).detach(),
            "source_timing_ratio": source_ratio(6).detach(),
            "awac_per_sample_loss_mean": awac_diag["per_sample_loss_mean"].detach(),
            "awac_target_norm_mean": awac_diag["target_norm_mean"].detach(),
            "awac_timestep_mean": awac_diag["diffusion_timestep_mean"].detach(),
            "awac_timestep_min": awac_diag["diffusion_timestep_min"].detach(),
            "awac_timestep_max": awac_diag["diffusion_timestep_max"].detach(),
            "awac_effective_weight_sum": awac_diag["effective_weight_sum"].detach(),
            "awac_zero_weight_ratio": awac_diag["zero_weight_ratio"].detach(),
            "awac_zero_weight_batch": awac_diag["zero_weight_batch"].detach(),
        })

    def forward_grpo(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        tokens_list,
        sample_time: Optional[int] = None,
        deterministic=False,
        bc_coeff: Optional[float] = None,
        use_bc_loss: bool = True
    ) -> BatchFeature:
        """Computes the Diffusion-GRPO loss."""
        self.set_frozen_modules_to_eval_mode()
        B = vl_features.shape[0]
        G = int(sample_time if sample_time is not None else getattr(self, "grpo_sample_time", 8))
        if G <= 0:
            raise ValueError("GRPO sample_time must be positive.")

        vl_features_rep = vl_features.repeat_interleave(G, 0)
        his_traj_rep = action_input.his_traj.repeat_interleave(G, 0)
        status_feature_rep = action_input.status_feature.repeat_interleave(G, 0)
        expert_action_input_rep = self._repeat_expert_action_input(action_input, G)

        sampled_from_behavior_policy = False
        if self.use_trajectory_level_objective and self.use_gspo_ratio and self.behavior_policy_sample:
            if not hasattr(self, "behavior_policy"):
                raise RuntimeError("use_gspo_ratio=True requires a frozen behavior_policy initialized in _init_grpo.")
            sync_interval = max(1, int(self.behavior_policy_sync_interval))
            if int(getattr(self, "grpo_update_counter", 0)) % sync_interval == 0:
                self._sync_behavior_policy()
            sampled_from_behavior_policy = True
            self.behavior_policy.eval()
            with torch.no_grad():
                chains, trajs = self.behavior_policy.sample_chain(
                    vl_features_rep,
                    his_traj_rep,
                    status_feature_rep,
                    deterministic=False,
                    action_input=expert_action_input_rep,
                )
        else:
            with torch.no_grad():
                chains, trajs = self.sample_chain(
                    vl_features_rep,
                    his_traj_rep,
                    status_feature_rep,
                    deterministic=False,
                    action_input=expert_action_input_rep,
                )

        tokens_rep = [tok for tok in tokens_list for _ in range(G)]
        unique_tokens = set(tokens_list)
        metric_cache = {}
        for token in unique_tokens:
            path = self.metric_cache_loader.metric_cache_paths[token]
            with lzma.open(path, 'rb') as f:
                metric_cache[token] = pickle.load(f)
        base_rewards, components = self.reward_fn(trajs, tokens_rep, metric_cache, return_components=True)
        # rewards: [B * G]
        assert base_rewards.shape == (B * G,)
        rewards, hard_safe_mask, reward_aux = self._compose_stage3_reward(
            base_rewards,
            components,
            trajs,
            B,
            G,
        )
        assert rewards.shape == (B * G,)
        assert hard_safe_mask.shape == (B * G,)

        # rewards_matrix: [B, G], hard_safe_matrix: [B, G]
        rewards_matrix = rewards.view(B, G)
        hard_safe_matrix = hard_safe_mask.view(B, G)
        advantages, group_weight, advantage_aux = self._compute_stage3_advantages(rewards_matrix, hard_safe_matrix)
        # advantages: [B * G], group_weight: [B]
        assert advantages.shape == (B * G,)
        assert group_weight.shape == (B,)

        num_denoising_steps = chains.shape[1] - 1
        denoising_indices = torch.arange(num_denoising_steps, device=advantages.device)
        discount = (float(self.gamma_denoising) ** (num_denoising_steps - denoising_indices - 1)).to(
            device=advantages.device,
            dtype=advantages.dtype,
        )

        adv = advantages * group_weight.repeat_interleave(G)
        new_log_probs = self.get_logprobs(
            vl_features_rep,
            his_traj_rep,
            status_feature_rep,
            chains,
            deterministic=False,
            action_input=expert_action_input_rep,
        )
        # raw log_probs: [B * G * K, H, D]
        assert new_log_probs.shape[0] == B * G * num_denoising_steps

        gspo_ratio_mean = new_log_probs.new_tensor(1.0)
        gspo_ratio_clip_frac = new_log_probs.new_tensor(0.0)

        use_strict_gspo = (
            self.use_trajectory_level_objective
            and self.use_gspo_ratio
            and sampled_from_behavior_policy
        )
        if self.use_trajectory_level_objective:
            new_traj_logp = self._reduce_chain_logprobs(new_log_probs, B, G, num_denoising_steps, discount)
            # traj_logp: [B * G]
            assert new_traj_logp.shape == (B * G,)

            if use_strict_gspo:
                with torch.no_grad():
                    old_log_probs = self.behavior_policy.get_logprobs(
                        vl_features_rep,
                        his_traj_rep,
                        status_feature_rep,
                        chains,
                        deterministic=False,
                        action_input=expert_action_input_rep,
                    )
                    old_traj_logp = self.behavior_policy._reduce_chain_logprobs(
                        old_log_probs,
                        B,
                        G,
                        num_denoising_steps,
                        discount,
                    )

                log_ratio = new_traj_logp - old_traj_logp
                ratio = torch.exp(log_ratio.clamp(min=-20.0, max=20.0))
                ratio_clip_low = 1.0 - float(self.gspo_clip_low)
                ratio_clip_high = 1.0 + float(self.gspo_clip_high)
                clipped_ratio = ratio.clamp(ratio_clip_low, ratio_clip_high)
                surrogate = torch.minimum(ratio * adv, clipped_ratio * adv)
                policy_loss = -torch.mean(surrogate)
                gspo_ratio_mean = ratio.detach().mean()
                gspo_ratio_clip_frac = (
                    ((ratio < ratio_clip_low) | (ratio > ratio_clip_high)).detach().float().mean().to(ratio)
                )
            else:
                policy_loss = -torch.mean(new_traj_logp * adv)
            trajectory_logp = new_traj_logp.detach()
        else:
            step_logp = new_log_probs.clamp(min=-5, max=2).mean(dim=[1, 2])
            adv_steps = adv.view(B, G, 1).expand(-1, -1, num_denoising_steps)
            discount_steps = discount.view(1, 1, num_denoising_steps).expand(B, G, num_denoising_steps)
            adv_weighted_flat = (adv_steps * discount_steps).reshape(-1)
            policy_loss = -torch.mean(step_logp * adv_weighted_flat)
            trajectory_logp = self._reduce_chain_logprobs(
                new_log_probs,
                B,
                G,
                num_denoising_steps,
                discount,
            ).detach()

        total_loss = policy_loss

        effective_bc_coeff = self._current_bc_coeff(bc_coeff)
        bc_loss = policy_loss.new_zeros(())
        if use_bc_loss:
            self.old_policy.eval()
            with torch.no_grad():
                teacher_chains, _ = self.old_policy.sample_chain(
                    vl_features,
                    action_input.his_traj,
                    action_input.status_feature,
                    deterministic=False,
                    action_input=action_input,
                )
            bc_logp = self.get_logprobs(
                vl_features,
                action_input.his_traj,
                action_input.status_feature,
                teacher_chains,
                deterministic=False,
                action_input=action_input,
            )
            bc_logp = bc_logp.clamp(min=-5, max=2)
            K_steps = chains.shape[1] - 1
            bc_logp = bc_logp.view(-1, K_steps, chains.shape[2], chains.shape[3]).mean(dim=[1,2,3])
            bc_loss = -bc_logp.mean()
            total_loss = total_loss + effective_bc_coeff * bc_loss

        self.grpo_update_counter = int(getattr(self, "grpo_update_counter", 0)) + 1

        zero_loss = total_loss.new_zeros(())
        hard_safe_ratio = hard_safe_mask.detach().float().mean().to(dtype=total_loss.dtype)
        diversity_bonus = reward_aux["diversity_bonus"].detach()
        if bool(hard_safe_mask.detach().any().item()):
            safe_diversity = diversity_bonus[hard_safe_mask.detach()].mean().to(dtype=total_loss.dtype)
        else:
            safe_diversity = zero_loss

        return BatchFeature(data={
            "loss": total_loss,
            "diffusion_loss": zero_loss,
            "jepa_alignment_loss": zero_loss,
            "vggt_alignment_loss": zero_loss,
            "reward": rewards.mean(),
            "base_reward": base_rewards.mean(),
            "shaped_reward": rewards.mean(),
            "policy_loss": policy_loss,
            "bc_loss": bc_loss,
            "bc_coeff": total_loss.new_tensor(effective_bc_coeff),
            "safe_ratio": hard_safe_ratio,
            "hard_safe_ratio": hard_safe_ratio,
            "mean_ep": reward_aux["ego_progress"].mean(),
            "mean_ttc": reward_aux["time_to_collision_within_bound"].mean(),
            "mean_comfort": reward_aux["history_comfort"].mean(),
            "diversity_bonus": diversity_bonus.mean(),
            "safe_diversity": safe_diversity,
            "group_reward_std": advantage_aux["reward_std"].mean(),
            "mixed_group_ratio": advantage_aux["mixed_group_ratio"],
            "all_safe_group_ratio": advantage_aux["all_safe_group_ratio"],
            "all_unsafe_group_ratio": advantage_aux["all_unsafe_group_ratio"],
            "mean_advantage": advantages.mean(),
            "mean_abs_advantage": advantages.abs().mean(),
            "trajectory_logp": trajectory_logp.mean(),
            "gspo_ratio_mean": gspo_ratio_mean.to(dtype=total_loss.dtype),
            "gspo_ratio_clip_frac": gspo_ratio_clip_frac.to(dtype=total_loss.dtype),
        })

    def norm_odo(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Normalizes trajectory coordinates and heading to the range [-1, 1]."""
        x = 2 * (trajectory[..., 0:1] + 1.57) / 66.74 - 1
        y = 2 * (trajectory[..., 1:2] + 19.68) / 42 - 1
        heading = 2 * (trajectory[..., 2:3] + 1.67) / 3.53 - 1
        return torch.cat([x, y, heading], dim=-1)
    
    def denorm_odo(self, normalized_trajectory: torch.Tensor) -> torch.Tensor:
        """Denormalizes trajectory from [-1, 1] back to original coordinate space."""
        x = (normalized_trajectory[..., 0:1] + 1) / 2 * 66.74 - 1.57
        y = (normalized_trajectory[..., 1:2] + 1) / 2 * 42 - 19.68
        heading = (normalized_trajectory[..., 2:3] + 1) / 2 * 3.53 - 1.67
        return torch.cat([x, y, heading], dim=-1)

    def _get_pdm_scorer_for_awac(self, *, use_fast_pdm_scorer: bool):
        if not use_fast_pdm_scorer:
            return self.train_scorer
        if not hasattr(self, "fast_train_scorer"):
            self.fast_train_scorer = FastPDMScorer(
                self.simulator.proposal_sampling,
                self.config.grpo_cfg.scorer_config,
            )
        return self.fast_train_scorer

    def _shadow_check_pdm_component_row(
        self,
        *,
        trajectory_np: np.ndarray,
        token: str,
        metric_cache,
        batch_row: Dict[str, float],
        component_keys: tuple[str, ...],
        strict_submetrics: bool,
        required_submetrics: Optional[tuple[str, ...]],
        missing_submetric_policy: str,
        max_abs_diff: float,
    ) -> None:
        scalar_result = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=Trajectory(trajectory_np),
            future_sampling=self.simulator.proposal_sampling,
            simulator=PDMSimulator(self.simulator.proposal_sampling),
            scorer=PDMScorer(self.simulator.proposal_sampling, self.config.grpo_cfg.scorer_config),
        )
        scalar_row = self._extract_pdm_components(
            scalar_result,
            strict_submetrics=strict_submetrics,
            required_submetrics=required_submetrics,
            missing_policy=missing_submetric_policy,
        )
        diffs = {
            key: abs(float(scalar_row[key]) - float(batch_row[key]))
            for key in component_keys
        }
        max_key, observed = max(diffs.items(), key=lambda item: item[1])
        if observed > max_abs_diff:
            raise ValueError(
                "Batched/Fast PDM scoring changed a PDM component compared with scalar pdm_score: "
                f"token={token!r}, key={max_key}, scalar={scalar_row[max_key]}, "
                f"batched={batch_row[max_key]}, diff={observed}, allowed={max_abs_diff}."
            )

    def reward_fn(
        self,
        pred_traj: torch.Tensor,
        tokens_list,
        cache_dict,
        return_components: bool = False,
        strict_submetrics: bool = False,
        required_submetrics: Optional[tuple[str, ...]] = None,
        missing_submetric_policy: str = "warn_default",
        use_batched_pdm_scoring: bool = False,
        use_exact_array_pdm_state_conversion: bool = False,
        use_fast_pdm_scorer: bool = False,
        pdm_batch_chunk_size: int = 0,
        pdm_shadow_check: bool = False,
        pdm_shadow_max_samples: int = 0,
        pdm_shadow_max_abs_diff: float = 0.0,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Calculates PDM scores for a batch of predicted trajectories."""
        pred_np = pred_traj.detach().cpu().numpy()
        component_keys = (
            "pdms",
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "time_to_collision_within_bound",
            "ego_progress",
            "history_comfort",
            "lane_keeping",
            "driving_direction_compliance",
            "traffic_light_compliance",
        )
        component_rows = [None] * len(tokens_list)
        if use_batched_pdm_scoring:
            token_to_indices: Dict[str, list[int]] = {}
            for idx, token in enumerate(tokens_list):
                token_to_indices.setdefault(str(token), []).append(idx)
            shadow_checked = 0
            scorer = self._get_pdm_scorer_for_awac(use_fast_pdm_scorer=use_fast_pdm_scorer)
            for token, indices in token_to_indices.items():
                metric_cache = cache_dict[token]
                chunk_size = int(pdm_batch_chunk_size)
                if chunk_size <= 0:
                    chunk_size = len(indices)
                for chunk_start in range(0, len(indices), chunk_size):
                    chunk_indices = indices[chunk_start : chunk_start + chunk_size]
                    pdm_results = pdm_score_batch_same_cache(
                        metric_cache=metric_cache,
                        model_trajectories=pred_np[chunk_indices],
                        future_sampling=self.simulator.proposal_sampling,
                        simulator=self.simulator,
                        scorer=scorer,
                        use_exact_array_conversion=bool(use_exact_array_pdm_state_conversion),
                    )
                    if len(pdm_results) != len(chunk_indices):
                        raise RuntimeError(
                            f"Batched PDM scorer returned {len(pdm_results)} rows for {len(chunk_indices)} trajectories."
                        )
                    for local_idx, pdm_result in enumerate(pdm_results):
                        row = self._extract_pdm_components(
                            pdm_result,
                            strict_submetrics=strict_submetrics,
                            required_submetrics=required_submetrics,
                            missing_policy=missing_submetric_policy,
                        )
                        global_idx = chunk_indices[local_idx]
                        component_rows[global_idx] = row
                        if pdm_shadow_check and shadow_checked < int(pdm_shadow_max_samples):
                            self._shadow_check_pdm_component_row(
                                trajectory_np=pred_np[global_idx],
                                token=token,
                                metric_cache=metric_cache,
                                batch_row=row,
                                component_keys=component_keys,
                                strict_submetrics=strict_submetrics,
                                required_submetrics=required_submetrics,
                                missing_submetric_policy=missing_submetric_policy,
                                max_abs_diff=float(pdm_shadow_max_abs_diff),
                            )
                            shadow_checked += 1
        else:
            for i, token in enumerate(tokens_list):
                trajectory = Trajectory(pred_np[i])
                metric_cache = cache_dict[token]
                pdm_result = pdm_score(
                    metric_cache=metric_cache,
                    model_trajectory=trajectory,
                    future_sampling=self.simulator.proposal_sampling,
                    simulator=self.simulator,
                    scorer=self.train_scorer,
                )
                component_rows[i] = self._extract_pdm_components(
                    pdm_result,
                    strict_submetrics=strict_submetrics,
                    required_submetrics=required_submetrics,
                    missing_policy=missing_submetric_policy,
                )
        if any(row is None for row in component_rows):
            raise RuntimeError("PDM scoring did not produce a component row for every trajectory.")
        components = {
            key: torch.tensor(
                [row[key] for row in component_rows],
                device=pred_traj.device,
                dtype=pred_traj.dtype,
            ).detach()
            for key in component_keys
        }
        rewards = components["pdms"]
        if return_components:
            return rewards, components
        return rewards

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

#https://github.com/irom-princeton/dppo/blob/cc7234ad7ff39a8f32de3af903606723a16f0648/model/diffusion/eta.py#L12
class EtaFixed(nn.Module):

    def __init__(
        self,
        base_eta=0.5,
        min_eta=0.1,
        max_eta=1.0,
        **kwargs,
    ):
        super().__init__()
        self.eta_logit = nn.Parameter(torch.ones(1))
        self.min = min_eta
        self.max = max_eta

        self.eta_logit.data = torch.atanh(
            torch.tensor([2 * (base_eta - min_eta) / (max_eta - min_eta) - 1])
        )

    def __call__(self, x):
        """Match input batch size, but do not depend on input"""
        B = len(x)
        device = x.device
        eta_normalized = torch.tanh(self.eta_logit)

        eta = 0.5 * (eta_normalized + 1) * (self.max - self.min) + self.min
        return torch.full((B, 1), eta.item()).to(device)
