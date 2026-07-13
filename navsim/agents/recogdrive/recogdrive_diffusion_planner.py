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
import math
import os
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
from .fs_norm import FSNormTransform, load_fs_norm_stats
from .planning_token_adapter import PlanningTokenAdapter, PlanningTokenAdapterConfig
from .reference_relative_geometry import compute_reference_relative_geometry_components
from .stage3_lfp_grpo import (
    LFPGRPOConfig,
    coerce_lfp_grpo_config,
    compute_lfp_advantages,
    trajectory_reinforce_loss,
    validate_lfp_config_exclusivity,
)
from .stage3_metric_adapter import Stage3MetricAdapter
from .stage3_policy_geometry import (
    apply_transition_std_floor,
    compute_group_trajectory_spread,
    endpoint_std_from_normalized_transition_floor,
    normalized_transition_floor_from_endpoint_std,
)
from .stage3_reference_cache import Stage3ReferenceCache
from .stage3_diversity_curriculum import (
    Stage3DiversityCapacityCache,
    compute_capacity_normalized_frontier_energy,
)
from .stage3_v2_official_evaluator import OfficialNAVSIMV2MetricEvaluator
from .pdas import compute_pdas_metrics
from .trajectory_feasibility import compute_feasibility_metrics
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
    reference_kl_coeff: float = 0.0
    reference_kl_chunk_size: int = 0
    
    metric_cache_path: str = "/path/to/metric_cache_train"
    reference_policy_checkpoint: str = "/path/to/IL_Model.ckpt"
    scorer_config: PDMScorerConfig = field(default_factory=lambda: PDMScorerConfig(
        progress_weight=10.0, ttc_weight=5.0, comfortable_weight=2.0
    ))

    # reward extraction and shaping
    reward_mode: Literal["safe_diffgrpo", "core_pareto", "feasible_pareto"] = "safe_diffgrpo"
    use_safety_shaped_reward: bool = True
    hard_gate_nc: bool = True
    hard_gate_dac: bool = True
    hard_gate_ttc: bool = False
    hard_gate_ddc: bool = False
    hard_gate_tlc: bool = False
    safety_advantage_mode: Literal["hard", "soft_penalty"] = "hard"

    nc_safe_threshold: float = 1.0
    dac_safe_threshold: float = 1.0
    ttc_safe_threshold: float = 1.0
    ddc_safe_threshold: float = 1.0
    tlc_safe_threshold: float = 1.0

    progress_bonus_weight: float = 0.05
    unsafe_reward_floor: float = -0.5
    unsafe_pdms_scale: float = 0.05
    soft_safety_penalty_weight: float = 0.50
    soft_safety_penalty_clip: float = 0.50
    soft_safety_min_reward: float = 0.0

    # Core-Pareto GRPO v2. This aligns the policy-gradient reward with the
    # navtest PDMS formula inside the NC/DAC-feasible set:
    # core = (5*EP + 5*TTC + 2*comfort) / 12. DDC is a guard only.
    use_core_pareto_grpo: bool = False
    core_ep_weight: float = 5.0
    core_ttc_weight: float = 5.0
    core_comfort_weight: float = 2.0
    core_normalizer: float = 12.0
    core_pareto_reference_mode: Literal["gt", "il", "max_gt_il"] = "max_gt_il"
    core_pareto_reference_sample_deterministic: bool = False
    core_pareto_use_reference_margin: bool = True
    core_pareto_reference_margin_weight: float = 0.3
    core_pareto_reference_margin_clip: float = 1.0
    core_pareto_reference_margin_scale: float = 0.05
    core_pareto_require_nc: bool = True
    core_pareto_require_dac: bool = True
    core_pareto_ddc_reference_mode: Literal["gt", "il", "max_gt_il"] = "max_gt_il"
    core_pareto_ddc_drop_tolerance: float = 0.01
    core_pareto_ddc_min_absolute: float = 0.95
    core_pareto_use_ep_floor: bool = True
    core_pareto_ep_reference_mode: Literal["gt", "il", "max_gt_il"] = "max_gt_il"
    core_pareto_ep_floor_tolerance: float = 0.02
    core_pareto_slow_penalty_weight: float = 0.5
    core_pareto_slow_invalid_advantage: float = -0.3
    core_pareto_use_ttc_tradeoff_penalty: bool = True
    core_pareto_ttc_reference_mode: Literal["gt", "il", "max_gt_il"] = "max_gt_il"
    core_pareto_tradeoff_tolerance: float = 0.01
    core_pareto_tradeoff_penalty_weight: float = 0.2
    core_pareto_ttc_floor_tolerance: float = 0.03
    core_pareto_use_ttc_floor_penalty: bool = False
    core_pareto_ttc_floor_penalty_weight: float = 0.1
    core_pareto_score_mode: Literal[
        "pdms_minus_slow",
        "pdms_plus_core_margin_minus_slow",
        "core_minus_slow",
    ] = "pdms_plus_core_margin_minus_slow"
    core_pareto_core_margin_weight: float = 0.3
    core_pareto_normalize_score_per_group: bool = True
    core_pareto_use_pareto_front: bool = True
    core_pareto_pareto_front_bonus: float = 0.2
    core_pareto_dominated_positive_adv_cap: float = 0.0
    core_pareto_pareto_objectives: tuple[str, ...] = (
        "ego_progress",
        "time_to_collision_within_bound",
        "history_comfort",
    )
    core_pareto_all_valid_objective: Literal["core", "score", "pdms"] = "core"
    core_pareto_all_safe_low_std_group_weight: float = 0.5
    core_pareto_min_group_reward_std: float = 0.005
    core_pareto_all_slow_group_weight: float = 0.25
    core_pareto_use_all_unsafe_rescue_advantage: bool = True
    core_pareto_all_unsafe_base_offset: float = 0.7
    core_pareto_all_unsafe_rescue_weight: float = 0.4
    core_pareto_all_unsafe_adv_min: float = -1.5
    core_pareto_all_unsafe_adv_max: float = 0.0
    core_pareto_unsafe_advantage_offset: float = 1.0
    core_pareto_advantage_clip_abs: float = 5.0
    core_pareto_positive_slow_fail_cap: float = 0.0
    core_pareto_use_phenotype_bucket_grpo: bool = False
    core_pareto_bucket_by_progress: bool = True
    core_pareto_bucket_by_lateral_endpoint: bool = True
    core_pareto_progress_fast_margin: float = 0.02
    core_pareto_progress_slow_margin: float = 0.02
    core_pareto_lateral_bucket_threshold_m: float = 0.5
    core_pareto_intra_bucket_weight: float = 1.0
    core_pareto_inter_bucket_weight: float = 0.3
    core_pareto_inter_bucket_clip: float = 0.5
    core_pareto_use_adaptive_dual: bool = False
    core_pareto_dual_ema: float = 0.95
    core_pareto_dual_lr: float = 0.02
    core_pareto_target_slow_rate: float = 0.10
    core_pareto_target_unsafe_rate: float = 0.05
    core_pareto_target_ddc_drop_rate: float = 0.05
    core_pareto_lambda_slow_init: float = 0.5
    core_pareto_lambda_slow_min: float = 0.1
    core_pareto_lambda_slow_max: float = 1.5
    core_pareto_lambda_safety_init: float = 1.0
    core_pareto_lambda_safety_min: float = 0.5
    core_pareto_lambda_safety_max: float = 2.0
    core_pareto_buffer_bonus_enabled: bool = False
    core_pareto_buffer_bonus_weight: float = 0.0
    core_pareto_buffer_bonus_scale_m: float = 3.0
    core_pareto_buffer_min_reward_margin: float = 0.02
    core_pareto_buffer_require_ep_floor: bool = True
    core_pareto_buffer_require_ddc_guard: bool = True
    core_pareto_buffer_max_targets_per_scene: int = 1

    # Legacy prototype fields kept for backward-compatible dry runs.
    core_pareto_ep_floor: float = 0.75
    core_pareto_ep_floor_penalty_weight: float = 0.40
    core_pareto_use_ddc_guard: bool = True
    core_pareto_ddc_guard_threshold: float = 0.95
    core_pareto_ddc_penalty_weight: float = 0.25
    core_pareto_ddc_penalty_clip: float = 1.0
    core_pareto_pareto_bonus: float = 0.15
    core_pareto_low_ep_adv_scale: float = 0.25
    core_pareto_infeasible_advantage_offset: float = 1.0
    core_pareto_min_core_std: float = 0.03
    core_pareto_advantage_mode: Literal["group_zscore", "loo_zscore"] = "group_zscore"
    core_pareto_use_phenotype_buckets: bool = True
    core_pareto_ep_ttc_balance_margin: float = 0.05
    core_pareto_target_nc: float = 1.0
    core_pareto_target_dac: float = 1.0
    core_pareto_target_ddc: float = 0.95

    # SG-FPS feasible Pareto-GRPO. Defaults preserve existing GRPO behavior.
    use_feasible_pareto_grpo: bool = False
    fp_ep_weight: float = 0.60
    fp_ttc_weight: float = 0.25
    fp_ddc_weight: float = 0.10
    fp_feas_weight: float = 0.05
    fp_ddc_min_absolute: float = 0.95
    fp_ddc_ref_tolerance: float = 0.01
    fp_feas_max: float = 0.0
    fp_comfort_min: float = 0.95
    fp_pareto_front_bonus: float = 0.15
    fp_tradeoff_penalty_weight: float = 0.2
    fp_tradeoff_ttc_rho: float = 1.0
    fp_tradeoff_tolerance: float = 0.01
    fp_regression_negative_advantage: float = -1.0
    fp_invalid_negative_advantage: float = -1.0
    fp_dominated_positive_cap: float = 0.0
    fp_geometry_positive_cap: float = 0.0
    fp_ddc_regression_positive_cap: float = 0.0
    fp_offsupport_positive_cap: float = 0.0
    fp_use_bucketed_advantage: bool = True
    fp_progress_fast_margin: float = 0.02
    fp_progress_slow_margin: float = 0.02
    fp_lateral_bucket_threshold_m: float = 0.5
    fp_feas_bucket_threshold: float = 0.1
    fp_inter_bucket_weight: float = 0.25
    fp_inter_bucket_clip: float = 0.5
    fp_use_pdas: bool = True
    pdas_eps: float = 0.05
    pdas_alpha: float = 1.0
    pdas_beta: float = 1.0
    pdas_gamma: float = 0.5
    pdas_lambda_bucket: float = 0.2
    pdas_lambda_regression: float = 0.5
    pdas_lambda_coverage: float = 0.2
    pdas_min_support_count_for_coverage: int = 4

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
    advantage_mode: Literal["safe_zscore", "safe_rpp"] = "safe_zscore"
    normalize_advantage_batch: bool = False
    advantage_clip_abs: float = 0.0

    # PPO-style replay over a fixed rollout. This is disabled unless the
    # Lightning wrapper selects the grpo_replay objective.
    ppo_replay_inner_epochs: int = 1
    ppo_replay_minibatch_size: int = 0
    ppo_replay_max_grad_norm: float = 1.0
    ppo_replay_min_abs_advantage: float = 1e-6
    ppo_replay_filter_zero_advantage: bool = True
    ppo_replay_sync_behavior_each_batch: bool = True
    ppo_replay_bc_update: bool = True
    ppo_replay_logprob_mode: Literal["trajectory", "step"] = "trajectory"
    ppo_replay_step_minibatch_mode: Literal["trajectory_all_steps", "transition"] = "trajectory_all_steps"
    ppo_replay_logprob_clamp_min: float = -5.0
    ppo_replay_logprob_clamp_max: float = 2.0
    ppo_replay_step_clip_schedule: Literal["constant", "dppo_exp"] = "constant"
    ppo_replay_step_clip_base: float = 0.001
    ppo_replay_step_clip_rate: float = 3.0

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
    init_stage3_oracle: bool = True
    init_reference_policy: bool = True

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
    grpo_buffer_guidance_enabled: bool = False
    grpo_buffer_reward_bonus_weight: float = 0.0
    grpo_buffer_reward_bonus_scale_m: float = 4.0
    grpo_buffer_reward_bonus_use_margin: bool = True
    grpo_buffer_distill_loss_weight: float = 0.0
    grpo_buffer_distill_loss_schedule: Literal[
        "constant",
        "linear_warmup",
        "linear_warmup_linear_decay",
        "linear_warmup_cosine_decay",
    ] = "linear_warmup"
    grpo_buffer_distill_loss_weight_start: float = 0.0
    grpo_buffer_distill_loss_weight_end: float = 0.0
    grpo_buffer_distill_warmup_start_epoch: int = 0
    grpo_buffer_distill_warmup_epochs: int = 3
    grpo_buffer_distill_decay_start_step: int = -1
    grpo_buffer_distill_decay_end_step: int = -1
    grpo_buffer_distill_top_k: int = 1
    grpo_buffer_distill_min_reward_margin: float = 0.0
    grpo_buffer_distill_timestep_sampling: Literal["uniform", "ddim", "low_noise", "mid_noise"] = "low_noise"
    grpo_buffer_preference_dpo_loss_weight: float = 0.0
    grpo_buffer_preference_dpo_loss_schedule: Literal["constant", "linear_warmup"] = "linear_warmup"
    grpo_buffer_preference_dpo_loss_weight_start: float = 0.0
    grpo_buffer_preference_dpo_warmup_start_epoch: int = 0
    grpo_buffer_preference_dpo_warmup_epochs: int = 2
    grpo_buffer_preference_dpo_timestep_sampling: Literal["uniform", "ddim", "low_noise", "mid_noise"] = "uniform"
    grpo_buffer_preference_dpo_beta: float = 8.0
    grpo_buffer_preference_dpo_label_smoothing: float = 0.0
    grpo_buffer_preference_dpo_reference_free: bool = False
    grpo_buffer_preference_dpo_pair_mode: Literal["best_vs_gt_il", "best_vs_low_valid", "best_vs_all"] = "best_vs_gt_il"
    grpo_buffer_preference_dpo_min_reward_gap: float = 0.02
    grpo_buffer_preference_dpo_max_pairs_per_scene: int = 2
    grpo_buffer_preference_dpo_gap_weight_mode: Literal["none", "pair_gap", "winner_advantage"] = "none"
    grpo_buffer_preference_dpo_gap_weight_scale: float = 0.05
    grpo_buffer_preference_dpo_gap_weight_min: float = 0.0
    grpo_buffer_preference_dpo_gap_weight_max: float = 3.0
    grpo_buffer_preference_dpo_include_il: bool = True
    grpo_self_imitation_loss_weight: float = 0.0
    grpo_self_imitation_loss_schedule: Literal[
        "constant",
        "linear_warmup",
        "linear_warmup_linear_decay",
        "linear_warmup_cosine_decay",
    ] = "linear_warmup"
    grpo_self_imitation_loss_weight_start: float = 0.0
    grpo_self_imitation_loss_weight_end: float = 0.0
    grpo_self_imitation_warmup_start_epoch: int = 0
    grpo_self_imitation_warmup_epochs: int = 3
    grpo_self_imitation_decay_start_step: int = -1
    grpo_self_imitation_decay_end_step: int = -1
    grpo_self_imitation_top_k: int = 1
    grpo_self_imitation_min_reward: float = 0.85
    grpo_self_imitation_min_reward_margin: float = 0.01
    grpo_self_imitation_max_target_scene_ratio: float = 1.0
    grpo_self_imitation_batch_cap_score: Literal["reward", "margin"] = "reward"
    grpo_self_imitation_baseline_mode: Literal[
        "buffer_gt_il",
        "group_mean",
        "group_leave_one_out",
        "buffer_or_group_mean",
    ] = "buffer_or_group_mean"
    grpo_self_imitation_timestep_sampling: Literal["uniform", "ddim", "low_noise", "mid_noise"] = "low_noise"
    grpo_self_imitation_require_nc: bool = True
    grpo_self_imitation_require_dac: bool = True
    grpo_self_imitation_require_ttc: bool = True
    grpo_self_imitation_require_ddc: bool = True
    grpo_self_imitation_nc_min_absolute: float = 1.0
    grpo_self_imitation_dac_min_absolute: float = 1.0
    grpo_self_imitation_ttc_min_absolute: float = 0.95
    grpo_self_imitation_ddc_min_absolute: float = 0.99

    # SG-FPS DPSI. These fields are inert unless use_dpsi=True.
    support_archive_path: str = ""
    use_dpsi: bool = False
    dpsi_top_m: int = 12
    dpsi_filter_to_support_indices: bool = True
    dpsi_empty_tag_zero: bool = True
    dpsi_scene_normalize_weights: bool = True
    dpsi_use_adaptive_beta: bool = True
    dpsi_target_sample_m: int = 4
    dpsi_target_sample_m_after_warmup: int = 6
    dpsi_force_anchor_target: bool = True
    dpsi_force_best_target: bool = True
    dpsi_beta_warmup_epochs: int = 40
    dpsi_beta_max: float = 0.75
    dpsi_support_count_mid: int = 4
    dpsi_distance_scale_m: float = 0.8
    dpsi_entropy_scale: float = 1.0
    dpsi_improver_margin: float = 0.03
    dpsi_strong_improver_margin: float = 0.05
    dpsi_low_support_count: int = 2
    dpsi_high_support_count: int = 6
    dpsi_high_gt_reward: float = 0.95
    dpsi_scene_weight_min: float = 0.75
    dpsi_scene_weight_max: float = 1.5
    dpsi_improver_scene_weight_gain: float = 2.0
    dpsi_weight_safe_ep: float = 1.0
    dpsi_weight_vector_pareto: float = 0.9
    dpsi_weight_safety_repair: float = 0.8
    dpsi_weight_ddc_repair: float = 0.8
    dpsi_weight_best_pdms: float = 0.75
    dpsi_weight_diversity: float = 0.60
    dpsi_weight_fallback: float = 0.45
    dpsi_weight_smooth: float = 0.6
    dpsi_weight_il: float = 0.5
    dpsi_weight_gt: float = 0.45
    dpsi_weight_unknown: float = 0.0
    dpsi_use_reward_margin_weight: bool = True
    dpsi_margin_scale: float = 2.0
    dpsi_margin_weight_min: float = 0.5
    dpsi_margin_weight_max: float = 1.5
    dpsi_bad_margin_threshold: float = -0.05
    dpsi_bad_margin_weight: float = 0.3
    dpsi_use_source_weight: bool = True
    dpsi_source_weight_gt: float = 1.0
    dpsi_source_weight_il: float = 0.9
    dpsi_source_weight_external: float = 0.85
    dpsi_source_weight_failure_expand: float = 0.75
    dpsi_source_weight_trust_region: float = 0.85
    dpsi_source_weight_structured: float = 0.75
    dpsi_source_weight_other: float = 0.7
    dpsi_pairwise_rank_weight: float = 0.0
    dpsi_pairwise_beta: float = 8.0

    # debugging / logging
    log_candidate_sources: bool = True
    log_submetrics: bool = True
    log_oracle_stats: bool = True
    require_reference_policy_checkpoint: bool = True
    report_raw_and_valid_best: bool = True


def _dpsi_source_matches(source: str, names: tuple[str, ...]) -> bool:
    lower = str(source or "").lower()
    for raw_name in names:
        name = str(raw_name).lower()
        if lower == name or lower.startswith(f"{name}:"):
            return True
    return False


def _compute_dpsi_support_profile(
    selected_trajs: torch.Tensor,
    selected_rewards: torch.Tensor,
    selected_real_mask: torch.Tensor,
    selected_source_code: torch.Tensor,
    gt_reward: torch.Tensor,
    il_reward: torch.Tensor,
    selected_valid_mask: torch.Tensor,
    cfg: OfflineRLConfig,
    current_epoch: int = 0,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    B = selected_rewards.shape[0]
    device = selected_rewards.device
    dtype = torch.float32
    real = selected_real_mask.bool()
    valid_real = real & selected_valid_mask.bool()
    support_count = real.float().sum(dim=1)
    pairwise_distance = selected_rewards.new_zeros((B,), dtype=dtype)
    source_entropy = selected_rewards.new_zeros((B,), dtype=dtype)
    best_selected = selected_rewards.new_zeros((B,), dtype=dtype)

    for b in range(B):
        idx = torch.nonzero(real[b], as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        best_mask = valid_real[b] if bool(valid_real[b].any().item()) else real[b]
        best_selected[b] = selected_rewards[b].masked_fill(~best_mask, -torch.inf).max()
        if idx.numel() >= 2:
            xy = selected_trajs[b, idx, :, :2].float()
            mean_xy = torch.cdist(xy.reshape(idx.numel(), -1), xy.reshape(idx.numel(), -1), p=2)
            mean_xy = mean_xy / math.sqrt(max(int(xy.shape[1]) * 2, 1))
            endpoint = torch.cdist(xy[:, -1], xy[:, -1], p=2)
            tri = torch.triu(torch.ones_like(mean_xy, dtype=torch.bool), diagonal=1)
            pairwise_distance[b] = 0.5 * (mean_xy[tri].mean() + endpoint[tri].mean())
            codes, counts = torch.unique(selected_source_code[b, idx], return_counts=True)
            probs = counts.float() / counts.float().sum().clamp(min=1.0)
            source_entropy[b] = -(probs * probs.clamp(min=1e-6).log()).sum()

    best_selected_minus_gt = best_selected - gt_reward.to(device=device, dtype=dtype)
    selected_has_improver = best_selected_minus_gt >= float(cfg.dpsi_improver_margin)
    high_gt_saturated = (
        gt_reward.to(device=device, dtype=dtype) >= float(cfg.dpsi_high_gt_reward)
    ) & (best_selected_minus_gt <= float(cfg.dpsi_improver_margin))
    low_support = support_count <= float(cfg.dpsi_low_support_count)

    beta_max = float(cfg.dpsi_beta_max)
    count_score = torch.sigmoid(support_count - float(cfg.dpsi_support_count_mid))
    dist_score = pairwise_distance / (pairwise_distance + float(cfg.dpsi_distance_scale_m))
    entropy_score = source_entropy / (source_entropy + float(cfg.dpsi_entropy_scale))
    if bool(cfg.dpsi_use_adaptive_beta):
        beta = beta_max * count_score * dist_score * entropy_score
    else:
        beta = torch.full((B,), beta_max, device=device, dtype=dtype)
    beta = torch.where(low_support, beta * 0.25, beta)
    beta = torch.where(high_gt_saturated, torch.zeros_like(beta), beta)
    strong = best_selected_minus_gt >= float(cfg.dpsi_strong_improver_margin)
    beta = torch.where(strong, torch.maximum(beta, beta.new_full(beta.shape, 0.4)), beta)
    beta = beta.clamp(min=0.0, max=beta_max)
    warmup = int(cfg.dpsi_beta_warmup_epochs)
    if warmup > 0:
        ramp = max(0.0, min(float(current_epoch) / float(warmup), 1.0))
        beta = beta * ramp

    scene_weight = 1.0 + float(cfg.dpsi_improver_scene_weight_gain) * best_selected_minus_gt.clamp(0.0, 0.25)
    scene_weight = scene_weight.clamp(
        min=float(cfg.dpsi_scene_weight_min),
        max=float(cfg.dpsi_scene_weight_max),
    )
    scene_weight = torch.where(high_gt_saturated, torch.minimum(scene_weight, scene_weight.new_ones(())), scene_weight)
    scene_weight = torch.where(
        low_support & (best_selected_minus_gt <= 0.0),
        torch.minimum(scene_weight, scene_weight.new_ones(())),
        scene_weight,
    )

    profile = {
        "support_count": support_count,
        "pairwise_distance": pairwise_distance,
        "source_entropy": source_entropy,
        "best_selected_minus_gt": best_selected_minus_gt,
        "selected_has_improver": selected_has_improver,
        "high_gt_saturated": high_gt_saturated,
        "low_support": low_support,
        "multimodal_profile_score": count_score * dist_score * entropy_score,
        "beta_profile": beta,
        "scene_weight": scene_weight,
    }
    diag = {
        "dpsi_support_count_mean": support_count.mean(),
        "dpsi_pairwise_distance_mean": pairwise_distance.mean(),
        "dpsi_source_entropy_mean": source_entropy.mean(),
        "dpsi_beta_mean": beta.mean(),
        "dpsi_beta_max": beta.max() if beta.numel() else selected_rewards.new_zeros(()),
        "dpsi_low_support_ratio": low_support.float().mean(),
        "dpsi_high_gt_saturated_ratio": high_gt_saturated.float().mean(),
        "dpsi_selected_has_improver_ratio": selected_has_improver.float().mean(),
        "dpsi_scene_weight_mean": scene_weight.mean(),
    }
    return profile, diag


def _build_asmi_weights(
    selected_rewards: torch.Tensor,
    selected_real_mask: torch.Tensor,
    selected_valid_mask: torch.Tensor,
    selected_source_code: torch.Tensor,
    selected_support_weight: torch.Tensor,
    gt_reward: torch.Tensor,
    il_reward: torch.Tensor,
    support_profile: Dict[str, torch.Tensor],
    cfg: OfflineRLConfig,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = selected_rewards.device
    real = selected_real_mask.bool()
    valid = selected_valid_mask.bool()
    source_code = selected_source_code.to(device=device)
    anchor_mask = real & ((source_code == 1) | (source_code == 2))
    for b in range(selected_rewards.shape[0]):
        if not bool(anchor_mask[b].any().item()):
            row = real[b] & valid[b]
            if not bool(row.any().item()):
                row = real[b]
            if bool(row.any().item()):
                idx = int(selected_rewards[b].masked_fill(~row, -torch.inf).argmax().item())
                anchor_mask[b, idx] = True

    base_weight = selected_support_weight.to(device=device, dtype=torch.float32).clamp(min=0.0)
    base_weight = torch.where(real, base_weight, torch.zeros_like(base_weight))
    base_weight = torch.where(valid | anchor_mask, base_weight, torch.zeros_like(base_weight))

    margin_weight = torch.ones_like(base_weight)
    if bool(cfg.dpsi_use_reward_margin_weight):
        ref_reward = torch.maximum(gt_reward.to(device=device), il_reward.to(device=device)).float()
        margin = selected_rewards.float() - ref_reward[:, None]
        margin_weight = (1.0 + float(cfg.dpsi_margin_scale) * margin).clamp(
            min=float(cfg.dpsi_margin_weight_min),
            max=float(cfg.dpsi_margin_weight_max),
        )
        bad = (margin < float(cfg.dpsi_bad_margin_threshold)) & (~anchor_mask)
        margin_weight = torch.where(bad, margin_weight * float(cfg.dpsi_bad_margin_weight), margin_weight)

    source_weight = torch.ones_like(base_weight)
    if bool(cfg.dpsi_use_source_weight):
        source_weight = torch.full_like(base_weight, float(cfg.dpsi_source_weight_other))
        source_weight = torch.where(source_code == 1, source_weight.new_full((), float(cfg.dpsi_source_weight_gt)), source_weight)
        source_weight = torch.where(source_code == 2, source_weight.new_full((), float(cfg.dpsi_source_weight_il)), source_weight)
        source_weight = torch.where((source_code == 3) | (source_code == 7), source_weight.new_full((), float(cfg.dpsi_source_weight_external)), source_weight)
        source_weight = torch.where(source_code == 8, source_weight.new_full((), float(cfg.dpsi_source_weight_failure_expand)), source_weight)
        source_weight = torch.where(source_code == 9, source_weight.new_full((), float(cfg.dpsi_source_weight_trust_region)), source_weight)
        structured = (source_code == 4) | (source_code == 5) | (source_code == 6) | (source_code == 10)
        source_weight = torch.where(structured, source_weight.new_full((), float(cfg.dpsi_source_weight_structured)), source_weight)

    pareto_raw = base_weight * margin_weight * source_weight
    pareto_raw = torch.where(real & valid, pareto_raw, torch.zeros_like(pareto_raw))
    anchor_raw = torch.where(
        anchor_mask,
        base_weight.clamp(min=1e-6) * selected_rewards.float().clamp(min=0.0).add(1e-3),
        torch.zeros_like(base_weight),
    )

    zero_row = torch.zeros((selected_rewards.shape[0],), device=device, dtype=torch.bool)
    for raw, mask in ((anchor_raw, anchor_mask), (pareto_raw, real & valid)):
        row_sum = raw.sum(dim=1)
        for b in torch.nonzero(row_sum <= 0.0, as_tuple=False).flatten().tolist():
            row = mask[b]
            if not bool(row.any().item()):
                row = real[b]
            if bool(row.any().item()):
                idx = int(selected_rewards[b].masked_fill(~row, -torch.inf).argmax().item())
                raw[b, idx] = 1.0
            else:
                zero_row[b] = True

    q_anchor = anchor_raw / anchor_raw.sum(dim=1, keepdim=True).clamp(min=1e-6)
    q_pareto = pareto_raw / pareto_raw.sum(dim=1, keepdim=True).clamp(min=1e-6)
    beta = support_profile["beta_profile"].to(device=device, dtype=torch.float32)[:, None]
    q = (1.0 - beta) * q_anchor + beta * q_pareto
    pre_row_sum = q.sum(dim=1)
    for b in torch.nonzero(pre_row_sum <= 0.0, as_tuple=False).flatten().tolist():
        row = anchor_mask[b] if bool(anchor_mask[b].any().item()) else real[b]
        if bool(row.any().item()):
            idx = int(selected_rewards[b].masked_fill(~row, -torch.inf).argmax().item())
            q[b, idx] = 1.0
        zero_row[b] = True
    q = q / q.sum(dim=1, keepdim=True).clamp(min=1e-6)
    weights = q * support_profile["scene_weight"].to(device=device, dtype=torch.float32)[:, None]
    weights = torch.where(real, weights, torch.zeros_like(weights))

    total = weights.sum().clamp(min=1e-6)
    normalized_q = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
    effective_count = 1.0 / normalized_q.square().sum(dim=1).clamp(min=1e-6)
    diag = {
        "dpsi_anchor_weight_mean": weights.masked_fill(~anchor_mask, 0.0).sum(dim=1).mean(),
        "dpsi_pareto_weight_mean": weights.masked_fill(anchor_mask | (~real), 0.0).sum(dim=1).mean(),
        "dpsi_effective_target_count_mean": effective_count.mean(),
        "dpsi_row_weight_sum_mean": weights.sum(dim=1).mean(),
        "dpsi_zero_row_ratio": zero_row.float().mean(),
        "dpsi_external_weight_ratio": weights[((source_code == 3) | (source_code == 7)) & real].sum() / total,
        "dpsi_gt_weight_ratio": weights[(source_code == 1) & real].sum() / total,
        "dpsi_il_weight_ratio": weights[(source_code == 2) & real].sum() / total,
    }
    return weights, diag


def _sample_asmi_targets(
    selected_trajs: torch.Tensor,
    weights: torch.Tensor,
    selected_real_mask: torch.Tensor,
    selected_valid_mask: torch.Tensor,
    selected_source_code: torch.Tensor,
    selected_rewards: torch.Tensor,
    cfg: OfflineRLConfig,
    current_epoch: int,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    B, M, H, D = selected_trajs.shape
    warm = int(cfg.dpsi_beta_warmup_epochs)
    target_m = int(cfg.dpsi_target_sample_m_after_warmup) if current_epoch >= warm else int(cfg.dpsi_target_sample_m)
    target_m = max(1, min(target_m, M))
    out_trajs = selected_trajs.new_zeros((B, target_m, H, D))
    out_weights = weights.new_zeros((B, target_m))
    out_mask = torch.zeros((B, target_m), device=selected_trajs.device, dtype=torch.bool)
    out_source_code = torch.zeros((B, target_m), device=selected_trajs.device, dtype=selected_source_code.dtype)
    anchor_hits = weights.new_zeros((B,))
    best_hits = weights.new_zeros((B,))
    counts = weights.new_zeros((B,))

    for b in range(B):
        available = torch.nonzero(selected_real_mask[b] & (weights[b] > 0.0), as_tuple=False).flatten()
        if available.numel() == 0:
            available = torch.nonzero(selected_real_mask[b], as_tuple=False).flatten()
        if available.numel() == 0:
            continue
        if available.numel() <= target_m:
            keep = available
        else:
            chosen: list[int] = []
            if bool(cfg.dpsi_force_anchor_target):
                anchor = available[(selected_source_code[b, available] == 1) | (selected_source_code[b, available] == 2)]
                if anchor.numel() == 0:
                    anchor = available
                idx = anchor[torch.argmax(selected_rewards[b, anchor])].item()
                chosen.append(int(idx))
            if bool(cfg.dpsi_force_best_target):
                valid_avail = available[selected_valid_mask[b, available]]
                if valid_avail.numel() == 0:
                    valid_avail = available
                idx = valid_avail[torch.argmax(selected_rewards[b, valid_avail])].item()
                if int(idx) not in chosen:
                    chosen.append(int(idx))
            remaining = [int(i) for i in available.tolist() if int(i) not in chosen]
            slots = target_m - len(chosen)
            if slots > 0 and remaining:
                rem = torch.tensor(remaining, device=selected_trajs.device, dtype=torch.long)
                probs = weights[b, rem].float().clamp(min=0.0)
                if bool(training) and bool((probs.sum() > 0).item()):
                    pick_rel = torch.multinomial(probs / probs.sum().clamp(min=1e-6), min(slots, rem.numel()), replacement=False)
                else:
                    pick_rel = torch.topk(probs, k=min(slots, rem.numel()), largest=True).indices
                chosen.extend(int(i) for i in rem[pick_rel].tolist())
            keep = torch.tensor(chosen[:target_m], device=selected_trajs.device, dtype=torch.long)
        n = int(keep.numel())
        out_trajs[b, :n] = selected_trajs[b, keep]
        raw = weights[b, keep].float()
        row_sum = weights[b].sum().clamp(min=1e-6)
        out_weights[b, :n] = raw / raw.sum().clamp(min=1e-6) * row_sum
        out_mask[b, :n] = True
        out_source_code[b, :n] = selected_source_code[b, keep]
        counts[b] = float(n)
        anchor_hits[b] = ((selected_source_code[b, keep] == 1) | (selected_source_code[b, keep] == 2)).any().float()
        best_idx = int(selected_rewards[b].masked_fill(~selected_real_mask[b], -torch.inf).argmax().item())
        best_hits[b] = (keep == best_idx).any().float()

    diag = {
        "dpsi_sampled_target_count_mean": counts.mean(),
        "dpsi_sampled_anchor_ratio": anchor_hits.mean(),
        "dpsi_sampled_best_ratio": best_hits.mean(),
        # Consumed by forward_dpsi before diagnostics are exposed to the logger.
        "_sampled_target_source_code": out_source_code,
    }
    return out_trajs, out_weights, out_mask, diag


@dataclass
class TrainingTarget:
    raw_trajectory: torch.Tensor
    selected_repr: torch.Tensor
    diffusion_target_repr: torch.Tensor
    residual_anchor_repr: Optional[torch.Tensor]
    residual_alpha: float
    diagnostics: Dict[str, torch.Tensor] = field(default_factory=dict)

    def reconstruct_full_x0(self, predicted_diffusion_x0: torch.Tensor) -> torch.Tensor:
        if self.residual_anchor_repr is None or float(self.residual_alpha) == 0.0:
            return predicted_diffusion_x0
        return predicted_diffusion_x0 + float(self.residual_alpha) * self.residual_anchor_repr.to(
            device=predicted_diffusion_x0.device,
            dtype=predicted_diffusion_x0.dtype,
        )


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
    stage3_algorithm: Literal["legacy", "lfp_grpo"] = "legacy"
    vlm_size: str = 'large'
    planner_dim: int = 384
    use_planning_token_adapter: bool = False
    planning_num_tokens: int = 16
    planning_num_heads: int = 8
    planning_condition_layers: Literal["cross_attention", "all"] = "cross_attention"
    planning_gate_init: float = 0.05
    planning_context_gate_init: float = 0.05
    planning_condition_dropout: float = 0.10
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
    current_train_step: int = 0

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
    two_expert_condition_mode: Literal[
        "horizon_hmef_lite",
        "denoise_hmef_v2",
        "flat_context",
        "prefuse_cross_attention",
    ] = "horizon_hmef_lite"
    two_expert_dit_condition_mode: Literal[
        "horizon_hmef_lite",
        "denoise_hmef_v2",
        "flat_context",
        "prefuse_cross_attention",
    ] = "horizon_hmef_lite"
    two_expert_planner_dim: int = 384
    two_expert_use_raw_vlm_base: bool = True
    two_expert_zero_init_deltas: bool = True
    two_expert_dyn_loss_floor: float = 0.0
    two_expert_geo_loss_floor: float = 0.0
    two_expert_num_dyn_groups: int = 3
    two_expert_dyn_tokens_per_group: int = 12
    two_expert_num_geo_tokens: int = 12
    two_expert_memory_tokens_to_dit: bool = False
    two_expert_denoise_gate_hidden_dim: int = 384
    two_expert_denoise_gate_temperature: float = 1.0
    two_expert_denoise_condition_scale_init: float = 1.0
    two_expert_memory_scale_init: float = 1.0
    two_expert_prefusion_heads: int = 8
    two_expert_prefusion_dropout: float = 0.10
    two_expert_prefusion_token_dropout: float = 0.05
    two_expert_prefusion_condition_dropout: float = 0.10
    two_expert_prefusion_scale_init: float = 1.0
    two_expert_prefusion_zero_init: bool = True

    tune_projector: bool = True
    tune_diffusion_model: bool = True

    # SG-FPS FS-Norm and trajectory-centric auxiliary losses.
    use_fs_norm: bool = False
    fs_norm_stats_path: str = ""
    fs_norm_use_robust: bool = False
    fs_norm_clip: float = 5.0
    fs_norm_target_clip: float = -1.0
    fs_norm_output_clip: float = -1.0
    fs_norm_output_clip_mode: str = "scalar"
    fs_norm_min_version: int = 1
    x0_aux_weight: float = 0.0
    delta_aux_weight: float = 0.0
    geo_aux_weight: float = 0.0
    x0_aux_low_noise_frac: float = 0.5
    geo_curvature_weight: float = 1.0
    geo_reverse_weight: float = 1.0
    geo_tail_reverse_weight: float = 2.0
    geo_early_kink_weight: float = 2.0
    geo_jerk_weight: float = 0.2
    trajectory_aux_weight: float = 0.0
    trajectory_heading_weight: float = 1.0
    trajectory_huber_beta: float = 0.5
    feasibility_aux_weight: float = 0.0
    aux_alpha_power: float = 1.0
    aux_warmup_epochs: int = 10
    tangent_margin_rad: float = 0.08
    curvature_margin: float = 0.05
    min_segment_length: float = 0.20

    flow_cfg: FlowConfig = field(default_factory=FlowConfig)
    ddpm_cfg: DDPMConfig = field(default_factory=DDPMConfig)
    ddim_cfg: DDIMConfig = field(default_factory=DDIMConfig)
    grpo_cfg: GRPOConfig = field(default_factory=GRPOConfig)
    offline_rl_cfg: OfflineRLConfig = field(default_factory=OfflineRLConfig)
    lfp_grpo_cfg: LFPGRPOConfig = field(default_factory=LFPGRPOConfig)


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
        config.lfp_grpo_cfg = coerce_lfp_grpo_config(config.lfp_grpo_cfg)

        if str(config.stage3_algorithm) not in {"legacy", "lfp_grpo"}:
            raise ValueError("stage3_algorithm must be 'legacy' or 'lfp_grpo'.")
        self.stage3_algorithm = str(config.stage3_algorithm)
        if self.stage3_algorithm == "lfp_grpo":
            if not bool(config.grpo):
                raise ValueError("stage3_algorithm='lfp_grpo' requires grpo=True.")
            if not bool(config.lfp_grpo_cfg.enabled):
                raise ValueError("stage3_algorithm='lfp_grpo' requires lfp_grpo_cfg.enabled=True.")
            config.lfp_grpo_cfg.validate()
            if bool(config.lfp_grpo_cfg.fs_transition_std_enabled):
                if not bool(config.use_fs_norm):
                    raise ValueError(
                        "lfp_grpo_cfg.fs_transition_std_enabled requires use_fs_norm=True."
                    )
                if int(config.action_dim) != 3:
                    raise ValueError(
                        "LFP FS-aware transition covariance requires action_dim=3."
                    )
                if str(config.sampling_method) not in {"ddpm", "ddim"}:
                    raise ValueError(
                        "LFP FS-aware transition covariance only supports DDPM/DDIM."
                    )
            validate_lfp_config_exclusivity(
                self.stage3_algorithm,
                config.grpo_cfg,
                config.offline_rl_cfg,
                stage3_objective="grpo",
            )

        if config.planning_condition_layers not in {"cross_attention", "all"}:
            raise ValueError("planning_condition_layers must be 'cross_attention' or 'all'.")
        if int(config.planning_num_tokens) <= 0:
            raise ValueError("planning_num_tokens must be positive.")
        if int(config.planning_num_heads) <= 0 or int(config.planner_dim) % int(config.planning_num_heads) != 0:
            raise ValueError("planning_num_heads must be positive and divide planner_dim.")
        if not 0.0 <= float(config.planning_condition_dropout) < 1.0:
            raise ValueError("planning_condition_dropout must be in [0.0, 1.0).")
        for gate_name in ("planning_gate_init", "planning_context_gate_init"):
            if not 0.0 < float(getattr(config, gate_name)) < 1.0:
                raise ValueError(f"{gate_name} must be in (0.0, 1.0).")
        if bool(config.use_planning_token_adapter) and int(config.planner_dim) != int(config.input_embedding_dim):
            raise ValueError("Standalone planning adapter requires planner_dim == input_embedding_dim.")
        if bool(config.use_planning_token_adapter) and bool(config.use_last_vla):
            raise ValueError("PlanningTokenAdapter and Last-VLA are separate alternatives and cannot be enabled together.")

        self.model = LightningDiT(**config.diffusion_model_cfg)
        self.model.set_gradient_checkpointing(
            self.stage3_algorithm == "lfp_grpo" and bool(config.lfp_grpo_cfg.gradient_checkpointing)
        )
        self.planning_adapter: Optional[PlanningTokenAdapter] = None
        if config.use_planning_token_adapter:
            self.planning_adapter = PlanningTokenAdapter(
                PlanningTokenAdapterConfig(
                    planner_dim=int(config.planner_dim),
                    hidden_dim=int(config.hidden_size),
                    num_tokens=int(config.planning_num_tokens),
                    num_heads=int(config.planning_num_heads),
                    condition_dropout=float(config.planning_condition_dropout),
                    context_gate_init=float(config.planning_context_gate_init),
                )
            )
            self.model.set_planning_gate_init(float(config.planning_gate_init))
            self.model.set_planning_trainable_layers(str(config.planning_condition_layers))

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
        self.fs_norm_transform: Optional[FSNormTransform] = None
        if bool(getattr(config, "use_fs_norm", False)):
            if not str(getattr(config, "fs_norm_stats_path", "")):
                raise ValueError("use_fs_norm=True requires fs_norm_stats_path.")
            stats = load_fs_norm_stats(str(config.fs_norm_stats_path), map_location="cpu")
            if int(getattr(stats, "version", 1)) < int(getattr(config, "fs_norm_min_version", 1)):
                raise ValueError(
                    f"FS-Norm stats version {getattr(stats, 'version', 1)} is below required "
                    f"version {config.fs_norm_min_version}."
                )
            requested_robust = bool(getattr(config, "fs_norm_use_robust", stats.use_robust))
            if int(getattr(stats, "version", 1)) >= 2 and requested_robust != bool(stats.use_robust):
                raise ValueError(
                    "FS-Norm v2 use_robust must match the stats file because clip bounds are "
                    "representation-specific. Rebuild the stats with the requested mode."
                )
            stats.use_robust = requested_robust
            support_archive_path = str(getattr(config.offline_rl_cfg, "support_archive_path", "") or "")
            if int(getattr(stats, "version", 1)) >= 2 and stats.archive_path and support_archive_path:
                stats_archive = Path(stats.archive_path).expanduser().resolve()
                configured_archive = Path(support_archive_path).expanduser().resolve()
                if stats_archive != configured_archive:
                    raise ValueError(
                        "FS-Norm stats archive_path does not match the configured support archive: "
                        f"{stats_archive} != {configured_archive}."
                    )
            stats.clip = float(getattr(config, "fs_norm_clip", stats.clip))
            self.fs_norm_transform = FSNormTransform(stats)

        if str(config.fs_norm_output_clip_mode) not in {"scalar", "stats_bounds"}:
            raise ValueError("fs_norm_output_clip_mode must be 'scalar' or 'stats_bounds'.")
        if bool(config.use_fs_norm) and float(config.fs_norm_target_clip) not in {-1.0, 0.0}:
            warnings.warn(
                "fs_norm_target_clip is deprecated and ignored: FS diffusion targets are never clipped.",
                DeprecationWarning,
                stacklevel=2,
            )

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
        if bool(config.use_last_vla) and bool(config.use_fs_norm):
            if float(config.last_vla_heading_loss_weight) != 0.0 or float(config.last_vla_progress_loss_weight) != 0.0:
                raise ValueError(
                    "Last-VLA with FS-Norm requires last_vla_heading_loss_weight=0 and "
                    "last_vla_progress_loss_weight=0 because those losses use legacy normalized semantics."
                )
        if config.sampling_method == "flow" and (
            float(config.trajectory_aux_weight) > 0.0 or float(config.feasibility_aux_weight) > 0.0
        ):
            raise ValueError("trajectory/feasibility auxiliaries currently support DDPM/DDIM only, not flow.")
        for name in (
            "trajectory_aux_weight",
            "trajectory_heading_weight",
            "feasibility_aux_weight",
            "aux_alpha_power",
            "tangent_margin_rad",
            "curvature_margin",
            "min_segment_length",
        ):
            if float(getattr(config, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if float(config.trajectory_huber_beta) <= 0.0:
            raise ValueError("trajectory_huber_beta must be positive.")
        if int(config.aux_warmup_epochs) < 0:
            raise ValueError("aux_warmup_epochs must be non-negative.")
        if config.use_last_vla:
            if config.last_vla_cot_condition_layers not in {"all", "cross_attention"}:
                raise ValueError("last_vla_cot_condition_layers must be 'all' or 'cross_attention'.")
            ignored_nondefaults = []
            if int(config.last_vla_cot_num_steps) != 4:
                ignored_nondefaults.append(f"last_vla_cot_num_steps={config.last_vla_cot_num_steps}")
            if float(config.last_vla_vlm_context_dropout_start) != 0.0:
                ignored_nondefaults.append(
                    f"last_vla_vlm_context_dropout_start={config.last_vla_vlm_context_dropout_start}"
                )
            if float(config.last_vla_vlm_context_dropout_end) != 0.7:
                ignored_nondefaults.append(
                    f"last_vla_vlm_context_dropout_end={config.last_vla_vlm_context_dropout_end}"
                )
            if int(config.last_vla_fusion_tokens) != 192:
                ignored_nondefaults.append(f"last_vla_fusion_tokens={config.last_vla_fusion_tokens}")
            if ignored_nondefaults:
                warnings.warn(
                    "The following legacy Last-VLA options are declared but do not alter forward and are ignored: "
                    + ", ".join(ignored_nondefaults),
                    DeprecationWarning,
                    stacklevel=2,
                )
        if config.use_two_expert_slots:
            allowed_two_expert_modes = {
                "horizon_hmef_lite",
                "denoise_hmef_v2",
                "flat_context",
                "prefuse_cross_attention",
            }
            if config.use_last_vla or config.use_last_rd or config.use_expert_features:
                raise ValueError(
                    "two_expert_slot is mutually exclusive with old Last-VLA, Last-RD, and A4 direct expert paths."
                )
            if config.two_expert_slot_mode != "vlm_soft_slots":
                raise ValueError("two_expert_slot_mode must be 'vlm_soft_slots'.")
            if config.two_expert_condition_mode not in allowed_two_expert_modes:
                raise ValueError(f"two_expert_condition_mode must be one of {sorted(allowed_two_expert_modes)}.")
            if config.two_expert_dit_condition_mode not in allowed_two_expert_modes:
                raise ValueError(f"two_expert_dit_condition_mode must be one of {sorted(allowed_two_expert_modes)}.")
            if config.two_expert_denoise_gate_hidden_dim <= 0:
                raise ValueError("two_expert_denoise_gate_hidden_dim must be positive.")
            if config.two_expert_denoise_gate_temperature <= 0.0:
                raise ValueError("two_expert_denoise_gate_temperature must be positive.")
            if int(config.two_expert_prefusion_heads) <= 0:
                raise ValueError("two_expert_prefusion_heads must be positive.")
            if config.input_embedding_dim % int(config.two_expert_prefusion_heads) != 0:
                raise ValueError("two_expert_prefusion_heads must divide input_embedding_dim.")
            for dropout_name in (
                "two_expert_prefusion_dropout",
                "two_expert_prefusion_token_dropout",
                "two_expert_prefusion_condition_dropout",
            ):
                if not 0.0 <= float(getattr(config, dropout_name)) < 1.0:
                    raise ValueError(f"{dropout_name} must be in [0.0, 1.0).")
            if float(config.two_expert_prefusion_scale_init) < 0.0:
                raise ValueError("two_expert_prefusion_scale_init must be non-negative.")
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
            self.two_expert_memory_type_embedding = nn.Parameter(torch.zeros(2, planner_dim))
            self.two_expert_memory_scale = nn.Parameter(
                torch.tensor(float(config.two_expert_memory_scale_init), dtype=torch.float32)
            )
            self.two_expert_prefusion_q_norm = nn.LayerNorm(planner_dim)
            self.two_expert_prefusion_memory_norm = nn.LayerNorm(planner_dim)
            self.two_expert_prefusion_attn = nn.MultiheadAttention(
                planner_dim,
                int(config.two_expert_prefusion_heads),
                dropout=float(config.two_expert_prefusion_dropout),
                batch_first=True,
            )
            self.two_expert_prefusion_out_proj = nn.Linear(planner_dim, planner_dim)
            self.two_expert_prefusion_scale = nn.Parameter(
                torch.tensor(float(config.two_expert_prefusion_scale_init), dtype=torch.float32)
            )
            self.two_expert_denoise_condition_scale = nn.Parameter(
                torch.tensor(float(config.two_expert_denoise_condition_scale_init), dtype=torch.float32)
            )
            gate_hidden_dim = int(config.two_expert_denoise_gate_hidden_dim)
            self.two_expert_denoise_gate = nn.Sequential(
                nn.LayerNorm(planner_dim * 4),
                nn.Linear(planner_dim * 4, gate_hidden_dim),
                nn.GELU(),
                nn.Linear(gate_hidden_dim, 2),
            )
            nn.init.constant_(self.two_expert_denoise_gate[-1].weight, 0.0)
            nn.init.constant_(self.two_expert_denoise_gate[-1].bias, 0.0)
            if config.two_expert_zero_init_deltas:
                nn.init.constant_(self.two_expert_dyn_delta_proj.weight, 0.0)
                nn.init.constant_(self.two_expert_dyn_delta_proj.bias, 0.0)
                nn.init.constant_(self.two_expert_geo_delta_proj.weight, 0.0)
                nn.init.constant_(self.two_expert_geo_delta_proj.bias, 0.0)
            if config.two_expert_prefusion_zero_init:
                nn.init.constant_(self.two_expert_prefusion_out_proj.weight, 0.0)
                nn.init.constant_(self.two_expert_prefusion_out_proj.bias, 0.0)

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
                    use_fs_norm=bool(config.use_fs_norm),
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
            if self.offline_rl_cfg.enabled:
                self._init_offline_rl(self.offline_rl_cfg, config.grpo_cfg)
        elif self.offline_rl_cfg.enabled:
            self._init_offline_rl(self.offline_rl_cfg, config.grpo_cfg)

        self.lfp_grpo_cfg = config.lfp_grpo_cfg
        self.lfp_metric_adapter: Optional[Stage3MetricAdapter] = None
        self.lfp_reference_cache: Optional[Stage3ReferenceCache] = None
        self.lfp_diversity_capacity_cache: Optional[Stage3DiversityCapacityCache] = None
        self.lfp_v2_rollout_evaluator: Optional[OfficialNAVSIMV2MetricEvaluator] = None
        self._lfp_epoch_energy: Dict[str, list[float]] = {}
        if self.stage3_algorithm == "lfp_grpo":
            self.lfp_metric_adapter = Stage3MetricAdapter(self.lfp_grpo_cfg.benchmark)
            self.lfp_reference_cache = Stage3ReferenceCache(
                self.lfp_grpo_cfg.reference_cache_path,
                benchmark=self.lfp_grpo_cfg.benchmark,
            )
            if self.lfp_grpo_cfg.frontier_diversity_capacity_enabled:
                self.lfp_diversity_capacity_cache = Stage3DiversityCapacityCache(
                    self.lfp_grpo_cfg.frontier_diversity_capacity_cache_path
                )
            self.reference_kl_coeff = float(self.lfp_grpo_cfg.reference_kl_coeff)
            if not hasattr(self, "old_policy"):
                raise RuntimeError("LFP-GRPO requires the frozen Stage2 old_policy initialized by _init_grpo().")
            if hasattr(self, "behavior_policy"):
                raise RuntimeError("LFP-GRPO must not initialize a behavior policy.")
            expected_reference_sha = str(
                self.lfp_reference_cache.metadata.get("stage2_checkpoint_sha256", "")
            )
            reference_path = Path(self.stage3_reference_checkpoint_path)
            actual_reference_sha = self._distributed_sha256_file(reference_path)
            if expected_reference_sha and expected_reference_sha != actual_reference_sha:
                raise ValueError(
                    "LFP reference cache was built from a different Stage2 checkpoint: "
                    f"cache={expected_reference_sha}, configured={actual_reference_sha}."
                )
            if not expected_reference_sha:
                warnings.warn(
                    "LFP reference cache metadata has no stage2_checkpoint_sha256; "
                    "checkpoint identity cannot be verified.",
                    RuntimeWarning,
                )
            self.lfp_reference_policy_checkpoint_sha256 = actual_reference_sha
            if int(os.getenv("RANK", "0")) == 0:
                print(
                    "LFP frozen Stage2 reference SHA256: "
                    f"{self.lfp_reference_policy_checkpoint_sha256}"
                )
            if self.lfp_grpo_cfg.benchmark == "navsim_v2":
                missing_adjacent = [
                    token
                    for token, record in self.lfp_reference_cache.records.items()
                    if not record.get("previous_token")
                    or record.get("previous_stage2_trajectory") is None
                ]
                if missing_adjacent:
                    raise KeyError(
                        "NAVSIM v2 LFP references require an adjacent frozen Stage2 trajectory for "
                        f"official EC; missing for {len(missing_adjacent)} token(s), "
                        f"first={missing_adjacent[0]!r}."
                    )
                self.lfp_v2_rollout_evaluator = OfficialNAVSIMV2MetricEvaluator(
                    self.lfp_grpo_cfg,
                    self.config.grpo_cfg.metric_cache_path,
                )

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
        for name in (
            "awac_timestep_sampling",
            "preference_dpo_timestep_sampling",
            "grpo_buffer_distill_timestep_sampling",
            "grpo_buffer_preference_dpo_timestep_sampling",
            "grpo_self_imitation_timestep_sampling",
        ):
            if str(getattr(cfg, name)) not in {"uniform", "ddim", "low_noise", "mid_noise"}:
                raise ValueError(f"offline_rl_cfg.{name} must be uniform, ddim, low_noise, or mid_noise.")
        if str(cfg.grpo_self_imitation_baseline_mode) not in {
            "buffer_gt_il",
            "group_mean",
            "group_leave_one_out",
            "buffer_or_group_mean",
        }:
            raise ValueError(
                "offline_rl_cfg.grpo_self_imitation_baseline_mode must be "
                "buffer_gt_il, group_mean, group_leave_one_out, or buffer_or_group_mean."
            )
        if not (0.0 < float(cfg.grpo_self_imitation_max_target_scene_ratio) <= 1.0):
            raise ValueError("offline_rl_cfg.grpo_self_imitation_max_target_scene_ratio must be in (0, 1].")
        if str(cfg.grpo_self_imitation_batch_cap_score) not in {"reward", "margin"}:
            raise ValueError("offline_rl_cfg.grpo_self_imitation_batch_cap_score must be reward or margin.")
        for name in (
            "grpo_self_imitation_nc_min_absolute",
            "grpo_self_imitation_dac_min_absolute",
            "grpo_self_imitation_ttc_min_absolute",
            "grpo_self_imitation_ddc_min_absolute",
        ):
            if float(getattr(cfg, name)) < 0.0:
                raise ValueError(f"offline_rl_cfg.{name} must be non-negative.")
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
        base_loss_schedules = {"constant", "linear_warmup"}
        decay_loss_schedules = {
            "constant",
            "linear_warmup",
            "linear_warmup_linear_decay",
            "linear_warmup_cosine_decay",
        }
        for name in (
            "awac_loss_schedule",
            "preference_dpo_loss_schedule",
            "grpo_loss_schedule",
            "grpo_buffer_distill_loss_schedule",
            "grpo_buffer_preference_dpo_loss_schedule",
            "grpo_self_imitation_loss_schedule",
        ):
            supported_loss_schedules = (
                decay_loss_schedules
                if name in {"grpo_buffer_distill_loss_schedule", "grpo_self_imitation_loss_schedule"}
                else base_loss_schedules
            )
            if str(getattr(cfg, name)) not in supported_loss_schedules:
                raise ValueError(
                    f"offline_rl_cfg.{name} must be one of {sorted(supported_loss_schedules)}."
                )
        for name in (
            "grpo_buffer_distill_loss_weight_end",
            "grpo_self_imitation_loss_weight_end",
        ):
            if float(getattr(cfg, name)) < 0.0:
                raise ValueError(f"offline_rl_cfg.{name} must be non-negative.")
        for prefix in ("grpo_buffer_distill", "grpo_self_imitation"):
            start_step = int(getattr(cfg, f"{prefix}_decay_start_step"))
            end_step = int(getattr(cfg, f"{prefix}_decay_end_step"))
            schedule = str(getattr(cfg, f"{prefix}_loss_schedule"))
            if start_step < -1 or end_step < -1:
                raise ValueError(f"offline_rl_cfg.{prefix}_decay_*_step must be -1 or non-negative.")
            if schedule in {"linear_warmup_linear_decay", "linear_warmup_cosine_decay"}:
                if start_step < 0 or end_step < 0:
                    raise ValueError(
                        f"offline_rl_cfg.{prefix}_loss_schedule={schedule} requires decay_start_step "
                        "and decay_end_step to be set."
                    )
                if end_step <= start_step:
                    raise ValueError(f"offline_rl_cfg.{prefix}_decay_end_step must be > decay_start_step.")
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
            "grpo_buffer_reward_bonus_weight",
            "grpo_buffer_distill_loss_weight",
            "grpo_buffer_distill_loss_weight_start",
            "grpo_buffer_distill_min_reward_margin",
            "grpo_buffer_preference_dpo_loss_weight",
            "grpo_buffer_preference_dpo_loss_weight_start",
            "grpo_buffer_preference_dpo_beta",
            "grpo_buffer_preference_dpo_min_reward_gap",
            "grpo_buffer_preference_dpo_gap_weight_scale",
            "grpo_buffer_preference_dpo_gap_weight_min",
            "grpo_buffer_preference_dpo_gap_weight_max",
            "grpo_self_imitation_loss_weight",
            "grpo_self_imitation_loss_weight_start",
            "grpo_self_imitation_min_reward",
            "grpo_self_imitation_min_reward_margin",
            "grpo_self_imitation_max_target_scene_ratio",
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
        if not (0.0 <= float(cfg.grpo_buffer_preference_dpo_label_smoothing) < 0.5):
            raise ValueError("offline_rl_cfg.grpo_buffer_preference_dpo_label_smoothing must be in [0, 0.5).")
        if str(cfg.grpo_buffer_preference_dpo_pair_mode) not in {"best_vs_gt_il", "best_vs_low_valid", "best_vs_all"}:
            raise ValueError(
                "offline_rl_cfg.grpo_buffer_preference_dpo_pair_mode must be best_vs_gt_il, "
                "best_vs_low_valid, or best_vs_all."
            )
        if str(cfg.grpo_buffer_preference_dpo_gap_weight_mode) not in {"none", "pair_gap", "winner_advantage"}:
            raise ValueError(
                "offline_rl_cfg.grpo_buffer_preference_dpo_gap_weight_mode must be none, pair_gap, or winner_advantage."
            )
        if float(cfg.grpo_buffer_preference_dpo_gap_weight_scale) <= 0.0:
            raise ValueError("offline_rl_cfg.grpo_buffer_preference_dpo_gap_weight_scale must be positive.")
        if float(cfg.grpo_buffer_preference_dpo_gap_weight_min) > float(cfg.grpo_buffer_preference_dpo_gap_weight_max):
            raise ValueError(
                "offline_rl_cfg.grpo_buffer_preference_dpo_gap_weight_min must be <= gap_weight_max."
            )
        if int(cfg.pairwise_rank_max_pairs_per_scene) < 0:
            raise ValueError("offline_rl_cfg.pairwise_rank_max_pairs_per_scene must be non-negative.")
        if int(cfg.invalid_repulsion_max_pairs_per_scene) < 0:
            raise ValueError("offline_rl_cfg.invalid_repulsion_max_pairs_per_scene must be non-negative.")
        if int(cfg.preference_dpo_max_pairs_per_scene) < 0:
            raise ValueError("offline_rl_cfg.preference_dpo_max_pairs_per_scene must be non-negative.")
        if int(cfg.grpo_buffer_preference_dpo_max_pairs_per_scene) < 0:
            raise ValueError("offline_rl_cfg.grpo_buffer_preference_dpo_max_pairs_per_scene must be non-negative.")
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
        if int(cfg.grpo_buffer_distill_warmup_start_epoch) < 0:
            raise ValueError("offline_rl_cfg.grpo_buffer_distill_warmup_start_epoch must be non-negative.")
        if int(cfg.grpo_buffer_distill_warmup_epochs) <= 0:
            raise ValueError("offline_rl_cfg.grpo_buffer_distill_warmup_epochs must be positive.")
        if int(cfg.grpo_buffer_distill_top_k) <= 0:
            raise ValueError("offline_rl_cfg.grpo_buffer_distill_top_k must be positive.")
        if int(cfg.grpo_buffer_preference_dpo_warmup_start_epoch) < 0:
            raise ValueError("offline_rl_cfg.grpo_buffer_preference_dpo_warmup_start_epoch must be non-negative.")
        if int(cfg.grpo_buffer_preference_dpo_warmup_epochs) <= 0:
            raise ValueError("offline_rl_cfg.grpo_buffer_preference_dpo_warmup_epochs must be positive.")
        if float(cfg.grpo_buffer_reward_bonus_scale_m) <= 0.0:
            raise ValueError("offline_rl_cfg.grpo_buffer_reward_bonus_scale_m must be positive.")
        if int(cfg.grpo_self_imitation_warmup_start_epoch) < 0:
            raise ValueError("offline_rl_cfg.grpo_self_imitation_warmup_start_epoch must be non-negative.")
        if int(cfg.grpo_self_imitation_warmup_epochs) <= 0:
            raise ValueError("offline_rl_cfg.grpo_self_imitation_warmup_epochs must be positive.")
        if int(cfg.grpo_self_imitation_top_k) <= 0:
            raise ValueError("offline_rl_cfg.grpo_self_imitation_top_k must be positive.")
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
        if bool(getattr(cfg, "init_stage3_oracle", True)):
            self._init_stage3_oracle(stage3_cfg)
        elif bool(cfg.build_candidates_online) or str(cfg.missing_buffer_policy) == "fallback_gt":
            raise RuntimeError(
                "offline_rl_cfg.init_stage3_oracle=False is only valid for fully offline support "
                "training. Online candidates or fallback_gt require metric_cache_path/PDM oracle."
            )
        if bool(getattr(cfg, "init_reference_policy", True)) and not hasattr(self, "old_policy"):
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
            and bool(str(self._offline_candidate_buffer_path(cfg)))
            and not bool(cfg.build_candidates_online)
        ):
            self._preload_awac_elite_buffer(cfg)

    @staticmethod
    def _offline_candidate_buffer_path(cfg: OfflineRLConfig) -> str:
        support_path = str(getattr(cfg, "support_archive_path", "") or "")
        if bool(getattr(cfg, "use_dpsi", False)) and support_path:
            return support_path
        return str(getattr(cfg, "elite_buffer_path", "") or "")

    def _reset_awac_elite_record_cache_if_needed(self, buffer_root: Path) -> None:
        root_key = str(Path(buffer_root).resolve())
        if getattr(self, "_awac_elite_record_cache_root", "") != root_key:
            self._awac_elite_record_cache = {}
            self._awac_elite_record_cache_root = root_key

    def _preload_awac_elite_buffer(self, cfg: OfflineRLConfig) -> None:
        buffer_root = Path(self._offline_candidate_buffer_path(cfg))
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
        self.reference_kl_coeff = float(cfg.reference_kl_coeff)
        if self.reference_kl_coeff < 0.0:
            raise ValueError("reference_kl_coeff must be non-negative.")
        self.reference_kl_chunk_size = int(getattr(cfg, "reference_kl_chunk_size", 0))
        if self.reference_kl_chunk_size < 0:
            raise ValueError("reference_kl_chunk_size must be non-negative.")
        if int(getattr(cfg, "ppo_replay_inner_epochs", 1)) <= 0:
            raise ValueError("ppo_replay_inner_epochs must be positive.")
        if int(getattr(cfg, "ppo_replay_minibatch_size", 0)) < 0:
            raise ValueError("ppo_replay_minibatch_size must be non-negative.")
        if float(getattr(cfg, "ppo_replay_max_grad_norm", 1.0)) < 0.0:
            raise ValueError("ppo_replay_max_grad_norm must be non-negative.")
        if float(getattr(cfg, "ppo_replay_min_abs_advantage", 0.0)) < 0.0:
            raise ValueError("ppo_replay_min_abs_advantage must be non-negative.")
        if str(getattr(cfg, "ppo_replay_logprob_mode", "trajectory")) not in {"trajectory", "step"}:
            raise ValueError("ppo_replay_logprob_mode must be either 'trajectory' or 'step'.")
        if str(getattr(cfg, "ppo_replay_step_minibatch_mode", "trajectory_all_steps")) not in {
            "trajectory_all_steps",
            "transition",
        }:
            raise ValueError("ppo_replay_step_minibatch_mode must be 'trajectory_all_steps' or 'transition'.")
        if float(getattr(cfg, "ppo_replay_logprob_clamp_max", 2.0)) < float(
            getattr(cfg, "ppo_replay_logprob_clamp_min", -5.0)
        ):
            raise ValueError("ppo_replay_logprob_clamp_max must be >= ppo_replay_logprob_clamp_min.")
        if str(getattr(cfg, "ppo_replay_step_clip_schedule", "constant")) not in {"constant", "dppo_exp"}:
            raise ValueError("ppo_replay_step_clip_schedule must be 'constant' or 'dppo_exp'.")
        if float(getattr(cfg, "ppo_replay_step_clip_base", 0.001)) < 0.0:
            raise ValueError("ppo_replay_step_clip_base must be non-negative.")
        if float(getattr(cfg, "ppo_replay_step_clip_rate", 3.0)) < 0.0:
            raise ValueError("ppo_replay_step_clip_rate must be non-negative.")
        for name in (
            "reward_mode",
            "use_safety_shaped_reward",
            "hard_gate_nc",
            "hard_gate_dac",
            "hard_gate_ttc",
            "hard_gate_ddc",
            "hard_gate_tlc",
            "safety_advantage_mode",
            "nc_safe_threshold",
            "dac_safe_threshold",
            "ttc_safe_threshold",
            "ddc_safe_threshold",
            "tlc_safe_threshold",
            "progress_bonus_weight",
            "unsafe_reward_floor",
            "unsafe_pdms_scale",
            "soft_safety_penalty_weight",
            "soft_safety_penalty_clip",
            "soft_safety_min_reward",
            "use_core_pareto_grpo",
            "core_ep_weight",
            "core_ttc_weight",
            "core_comfort_weight",
            "core_normalizer",
            "core_pareto_reference_mode",
            "core_pareto_reference_sample_deterministic",
            "core_pareto_use_reference_margin",
            "core_pareto_reference_margin_weight",
            "core_pareto_reference_margin_clip",
            "core_pareto_reference_margin_scale",
            "core_pareto_require_nc",
            "core_pareto_require_dac",
            "core_pareto_ddc_reference_mode",
            "core_pareto_ddc_drop_tolerance",
            "core_pareto_ddc_min_absolute",
            "core_pareto_use_ep_floor",
            "core_pareto_ep_reference_mode",
            "core_pareto_ep_floor_tolerance",
            "core_pareto_slow_penalty_weight",
            "core_pareto_slow_invalid_advantage",
            "core_pareto_use_ttc_tradeoff_penalty",
            "core_pareto_ttc_reference_mode",
            "core_pareto_tradeoff_tolerance",
            "core_pareto_tradeoff_penalty_weight",
            "core_pareto_ttc_floor_tolerance",
            "core_pareto_use_ttc_floor_penalty",
            "core_pareto_ttc_floor_penalty_weight",
            "core_pareto_score_mode",
            "core_pareto_core_margin_weight",
            "core_pareto_normalize_score_per_group",
            "core_pareto_use_pareto_front",
            "core_pareto_pareto_front_bonus",
            "core_pareto_dominated_positive_adv_cap",
            "core_pareto_pareto_objectives",
            "core_pareto_all_valid_objective",
            "core_pareto_all_safe_low_std_group_weight",
            "core_pareto_min_group_reward_std",
            "core_pareto_all_slow_group_weight",
            "core_pareto_use_all_unsafe_rescue_advantage",
            "core_pareto_all_unsafe_base_offset",
            "core_pareto_all_unsafe_rescue_weight",
            "core_pareto_all_unsafe_adv_min",
            "core_pareto_all_unsafe_adv_max",
            "core_pareto_unsafe_advantage_offset",
            "core_pareto_advantage_clip_abs",
            "core_pareto_positive_slow_fail_cap",
            "core_pareto_use_phenotype_bucket_grpo",
            "core_pareto_bucket_by_progress",
            "core_pareto_bucket_by_lateral_endpoint",
            "core_pareto_progress_fast_margin",
            "core_pareto_progress_slow_margin",
            "core_pareto_lateral_bucket_threshold_m",
            "core_pareto_intra_bucket_weight",
            "core_pareto_inter_bucket_weight",
            "core_pareto_inter_bucket_clip",
            "core_pareto_dual_ema",
            "core_pareto_dual_lr",
            "core_pareto_target_slow_rate",
            "core_pareto_target_unsafe_rate",
            "core_pareto_target_ddc_drop_rate",
            "core_pareto_lambda_slow_init",
            "core_pareto_lambda_slow_min",
            "core_pareto_lambda_slow_max",
            "core_pareto_lambda_safety_init",
            "core_pareto_lambda_safety_min",
            "core_pareto_lambda_safety_max",
            "core_pareto_buffer_bonus_enabled",
            "core_pareto_buffer_bonus_weight",
            "core_pareto_buffer_bonus_scale_m",
            "core_pareto_buffer_min_reward_margin",
            "core_pareto_buffer_require_ep_floor",
            "core_pareto_buffer_require_ddc_guard",
            "core_pareto_buffer_max_targets_per_scene",
            "core_pareto_ep_floor",
            "core_pareto_ep_floor_penalty_weight",
            "core_pareto_use_ddc_guard",
            "core_pareto_ddc_guard_threshold",
            "core_pareto_ddc_penalty_weight",
            "core_pareto_ddc_penalty_clip",
            "core_pareto_pareto_bonus",
            "core_pareto_low_ep_adv_scale",
            "core_pareto_infeasible_advantage_offset",
            "core_pareto_min_core_std",
            "core_pareto_advantage_mode",
            "core_pareto_use_phenotype_buckets",
            "core_pareto_ep_ttc_balance_margin",
            "core_pareto_use_adaptive_dual",
            "core_pareto_dual_lr",
            "core_pareto_target_nc",
            "core_pareto_target_dac",
            "core_pareto_target_ddc",
            "use_feasible_pareto_grpo",
            "fp_ep_weight",
            "fp_ttc_weight",
            "fp_ddc_weight",
            "fp_feas_weight",
            "fp_ddc_min_absolute",
            "fp_ddc_ref_tolerance",
            "fp_feas_max",
            "fp_comfort_min",
            "fp_pareto_front_bonus",
            "fp_tradeoff_penalty_weight",
            "fp_tradeoff_ttc_rho",
            "fp_tradeoff_tolerance",
            "fp_regression_negative_advantage",
            "fp_invalid_negative_advantage",
            "fp_dominated_positive_cap",
            "fp_geometry_positive_cap",
            "fp_ddc_regression_positive_cap",
            "fp_offsupport_positive_cap",
            "fp_use_bucketed_advantage",
            "fp_progress_fast_margin",
            "fp_progress_slow_margin",
            "fp_lateral_bucket_threshold_m",
            "fp_feas_bucket_threshold",
            "fp_inter_bucket_weight",
            "fp_inter_bucket_clip",
            "fp_use_pdas",
            "pdas_eps",
            "pdas_alpha",
            "pdas_beta",
            "pdas_gamma",
            "pdas_lambda_bucket",
            "pdas_lambda_regression",
            "pdas_lambda_coverage",
            "pdas_min_support_count_for_coverage",
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
            "advantage_mode",
            "normalize_advantage_batch",
            "advantage_clip_abs",
            "ppo_replay_inner_epochs",
            "ppo_replay_minibatch_size",
            "ppo_replay_max_grad_norm",
            "ppo_replay_min_abs_advantage",
            "ppo_replay_filter_zero_advantage",
            "ppo_replay_sync_behavior_each_batch",
            "ppo_replay_bc_update",
            "ppo_replay_logprob_mode",
            "ppo_replay_step_minibatch_mode",
            "ppo_replay_logprob_clamp_min",
            "ppo_replay_logprob_clamp_max",
            "ppo_replay_step_clip_schedule",
            "ppo_replay_step_clip_base",
            "ppo_replay_step_clip_rate",
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
        if str(self.safety_advantage_mode) not in {"hard", "soft_penalty"}:
            raise ValueError("GRPO safety_advantage_mode must be 'hard' or 'soft_penalty'.")
        if str(self.reward_mode) not in {"safe_diffgrpo", "core_pareto", "feasible_pareto"}:
            raise ValueError("GRPO reward_mode must be 'safe_diffgrpo', 'core_pareto', or 'feasible_pareto'.")
        if str(self.advantage_mode) not in {"safe_zscore", "safe_rpp"}:
            raise ValueError("GRPO advantage_mode must be 'safe_zscore' or 'safe_rpp'.")
        for name in (
            "core_pareto_reference_mode",
            "core_pareto_ddc_reference_mode",
            "core_pareto_ep_reference_mode",
            "core_pareto_ttc_reference_mode",
        ):
            if str(getattr(self, name)) not in {"gt", "il", "max_gt_il"}:
                raise ValueError(f"{name} must be one of 'gt', 'il', or 'max_gt_il'.")
        if str(self.core_pareto_score_mode) not in {
            "pdms_minus_slow",
            "pdms_plus_core_margin_minus_slow",
            "core_minus_slow",
        }:
            raise ValueError("core_pareto_score_mode has an unsupported value.")
        if str(self.core_pareto_all_valid_objective) not in {"core", "score", "pdms"}:
            raise ValueError("core_pareto_all_valid_objective must be 'core', 'score', or 'pdms'.")
        if float(self.core_normalizer) <= 0.0:
            raise ValueError("core_normalizer must be positive.")
        if float(self.core_pareto_reference_margin_scale) <= 0.0:
            raise ValueError("core_pareto_reference_margin_scale must be positive.")
        invalid_objectives = set(self.core_pareto_pareto_objectives) - set(REQUIRED_COMPONENT_KEYS)
        if invalid_objectives:
            raise ValueError(f"Unsupported core_pareto_pareto_objectives: {sorted(invalid_objectives)}")
        if int(self.core_pareto_buffer_max_targets_per_scene) <= 0:
            raise ValueError("core_pareto_buffer_max_targets_per_scene must be positive.")
        if bool(self.core_pareto_buffer_bonus_enabled):
            offline_cfg = getattr(self, "offline_rl_cfg", None)
            buffer_path = "" if offline_cfg is None else str(getattr(offline_cfg, "elite_buffer_path", ""))
            if not buffer_path:
                raise ValueError("core_pareto_buffer_bonus_enabled=True requires an offline RL elite buffer path.")
        if str(self.core_pareto_advantage_mode) not in {"group_zscore", "loo_zscore"}:
            raise ValueError("core_pareto_advantage_mode must be 'group_zscore' or 'loo_zscore'.")
        if float(self.soft_safety_penalty_weight) < 0.0:
            raise ValueError("GRPO soft_safety_penalty_weight must be non-negative.")
        if float(self.soft_safety_penalty_clip) < 0.0:
            raise ValueError("GRPO soft_safety_penalty_clip must be non-negative.")
        for name in (
            "core_pareto_ep_floor",
            "core_pareto_ddc_guard_threshold",
            "core_pareto_target_nc",
            "core_pareto_target_dac",
            "core_pareto_target_ddc",
            "fp_ddc_min_absolute",
            "fp_comfort_min",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        for name in (
            "core_pareto_ep_floor_penalty_weight",
            "core_pareto_ddc_penalty_weight",
            "core_pareto_ddc_penalty_clip",
            "core_pareto_pareto_bonus",
            "core_pareto_low_ep_adv_scale",
            "core_pareto_infeasible_advantage_offset",
            "core_pareto_min_core_std",
            "core_pareto_ep_ttc_balance_margin",
            "core_pareto_dual_lr",
            "core_ep_weight",
            "core_ttc_weight",
            "core_comfort_weight",
            "core_pareto_reference_margin_weight",
            "core_pareto_reference_margin_clip",
            "core_pareto_reference_margin_scale",
            "core_pareto_ddc_drop_tolerance",
            "core_pareto_ddc_min_absolute",
            "core_pareto_ep_floor_tolerance",
            "core_pareto_slow_penalty_weight",
            "core_pareto_tradeoff_tolerance",
            "core_pareto_tradeoff_penalty_weight",
            "core_pareto_ttc_floor_tolerance",
            "core_pareto_ttc_floor_penalty_weight",
            "core_pareto_core_margin_weight",
            "core_pareto_pareto_front_bonus",
            "core_pareto_all_safe_low_std_group_weight",
            "core_pareto_min_group_reward_std",
            "core_pareto_all_slow_group_weight",
            "core_pareto_all_unsafe_base_offset",
            "core_pareto_all_unsafe_rescue_weight",
            "core_pareto_unsafe_advantage_offset",
            "core_pareto_advantage_clip_abs",
            "core_pareto_progress_fast_margin",
            "core_pareto_progress_slow_margin",
            "core_pareto_lateral_bucket_threshold_m",
            "core_pareto_intra_bucket_weight",
            "core_pareto_inter_bucket_weight",
            "core_pareto_inter_bucket_clip",
            "core_pareto_dual_ema",
            "core_pareto_target_slow_rate",
            "core_pareto_target_unsafe_rate",
            "core_pareto_target_ddc_drop_rate",
            "core_pareto_lambda_slow_init",
            "core_pareto_lambda_slow_min",
            "core_pareto_lambda_slow_max",
            "core_pareto_lambda_safety_init",
            "core_pareto_lambda_safety_min",
            "core_pareto_lambda_safety_max",
            "core_pareto_buffer_bonus_weight",
            "core_pareto_buffer_bonus_scale_m",
            "core_pareto_buffer_min_reward_margin",
            "fp_ep_weight",
            "fp_ttc_weight",
            "fp_ddc_weight",
            "fp_feas_weight",
            "fp_ddc_ref_tolerance",
            "fp_pareto_front_bonus",
            "fp_tradeoff_penalty_weight",
            "fp_tradeoff_ttc_rho",
            "fp_tradeoff_tolerance",
            "fp_progress_fast_margin",
            "fp_progress_slow_margin",
            "fp_lateral_bucket_threshold_m",
            "fp_feas_bucket_threshold",
            "fp_inter_bucket_weight",
            "fp_inter_bucket_clip",
            "pdas_eps",
            "pdas_alpha",
            "pdas_beta",
            "pdas_gamma",
            "pdas_lambda_bucket",
            "pdas_lambda_regression",
            "pdas_lambda_coverage",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if int(self.pdas_min_support_count_for_coverage) < 0:
            raise ValueError("pdas_min_support_count_for_coverage must be non-negative.")
        if not (0.0 < float(self.core_pareto_dual_ema) < 1.0):
            raise ValueError("core_pareto_dual_ema must be in (0, 1).")
        if float(self.core_pareto_lambda_slow_max) < float(self.core_pareto_lambda_slow_min):
            raise ValueError("core_pareto_lambda_slow_max must be >= core_pareto_lambda_slow_min.")
        if float(self.core_pareto_lambda_safety_max) < float(self.core_pareto_lambda_safety_min):
            raise ValueError("core_pareto_lambda_safety_max must be >= core_pareto_lambda_safety_min.")
        self.grpo_update_counter = 0
        self.core_pareto_dual_nc = 0.0
        self.core_pareto_dual_dac = 0.0
        self.core_pareto_dual_ddc = 0.0
        self.core_pareto_lambda_slow = float(self.core_pareto_lambda_slow_init)
        self.core_pareto_lambda_safety = float(self.core_pareto_lambda_safety_init)
        self.core_pareto_slow_rate_ema = 0.0
        self.core_pareto_unsafe_rate_ema = 0.0
        self.core_pareto_ddc_drop_rate_ema = 0.0

    def _init_grpo(self, cfg: GRPOConfig):
        """Initializes components and hyperparameters for GRPO training."""
        self._init_stage3_runtime(cfg)

        self._init_stage3_oracle(cfg)

        reference_checkpoint = self._resolve_reference_policy_checkpoint(
            cfg.reference_policy_checkpoint,
            required=True,
        )
        self.stage3_reference_checkpoint_path = str(reference_checkpoint)
        self._safe_load_reference_policy(str(reference_checkpoint))

        behavior_policy = None
        if self.use_gspo_ratio:
            behavior_policy = copy.deepcopy(self)
            self._freeze_policy(behavior_policy)

        self.old_policy = copy.deepcopy(self)
        self._freeze_policy(self.old_policy)

        if behavior_policy is not None:
            self.behavior_policy = behavior_policy

    @staticmethod
    def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _distributed_sha256_file(cls, path: Path) -> str:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return cls._sha256_file(path)
        payload = [cls._sha256_file(path) if torch.distributed.get_rank() == 0 else ""]
        torch.distributed.broadcast_object_list(payload, src=0)
        return str(payload[0])

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

    def set_training_progress(self, epoch: int, total_epochs: int, global_step: Optional[int] = None):
        self.config.current_train_epoch = int(epoch)
        self.config.total_train_epochs = max(1, int(total_epochs))
        if global_step is not None:
            self.config.current_train_step = max(0, int(global_step))
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
                7: "random_slots",
            }.get(code, "normal")
        return str(value)

    def _two_expert_denoise_v2_enabled(self) -> bool:
        return bool(
            self.config.use_two_expert_slots
            and self.config.two_expert_dit_condition_mode == "denoise_hmef_v2"
        )

    def _two_expert_flat_context_enabled(self) -> bool:
        return bool(
            self.config.use_two_expert_slots
            and self.config.two_expert_dit_condition_mode == "flat_context"
        )

    def _two_expert_prefusion_enabled(self) -> bool:
        return bool(
            self.config.use_two_expert_slots
            and self.config.two_expert_dit_condition_mode == "prefuse_cross_attention"
        )

    def _apply_two_expert_prefusion_dropout(
        self,
        expert_memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dropped = expert_memory.new_zeros(())
        token_keep_ratio = expert_memory.new_ones(())
        condition_keep = expert_memory.new_ones(expert_memory.shape[0], 1, 1)
        if not self.training:
            return expert_memory, dropped, token_keep_ratio, condition_keep
        condition_dropout = float(self.config.two_expert_prefusion_condition_dropout)
        if condition_dropout > 0.0:
            keep = torch.rand(expert_memory.shape[0], 1, 1, device=expert_memory.device) >= condition_dropout
            condition_keep = keep.to(dtype=expert_memory.dtype)
            dropped = (~keep).to(dtype=expert_memory.dtype).mean()
            expert_memory = expert_memory * condition_keep
        token_dropout = float(self.config.two_expert_prefusion_token_dropout)
        if token_dropout > 0.0:
            keep = torch.rand(expert_memory.shape[0], expert_memory.shape[1], 1, device=expert_memory.device) >= token_dropout
            token_keep_ratio = keep.to(dtype=expert_memory.dtype).mean()
            expert_memory = expert_memory * keep.to(dtype=expert_memory.dtype) / max(1.0 - token_dropout, 1e-6)
        return expert_memory, dropped, token_keep_ratio, condition_keep

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
        if mode == "random_slots":
            if h_dyn.shape[0] > 1:
                order = torch.roll(torch.arange(h_dyn.shape[0], device=h_dyn.device), shifts=1)
                h_dyn = h_dyn[order]
                h_geo = h_geo[order]
            else:
                h_dyn = torch.randn_like(h_dyn) * h_dyn.detach().float().std().clamp_min(1e-6).to(h_dyn)
                h_geo = torch.randn_like(h_geo) * h_geo.detach().float().std().clamp_min(1e-6).to(h_geo)
        if mode not in {
            "normal",
            "zero_h_dyn",
            "zero_h_geo",
            "zero_all_experts",
            "raw_vlm_only",
            "dyn_only",
            "geo_only",
            "random_slots",
        }:
            raise ValueError(f"Unknown two_expert corruption mode: {mode!r}.")
        dyn_enabled = mode not in {"zero_h_dyn", "zero_all_experts", "raw_vlm_only", "geo_only"}
        geo_enabled = mode not in {"zero_h_geo", "zero_all_experts", "raw_vlm_only", "dyn_only"}

        dyn_proj = self.two_expert_dyn_proj(h_dyn.reshape(h_dyn.shape[0], -1, h_dyn.shape[-1]))
        geo_proj = self.two_expert_geo_proj(h_geo)
        if not dyn_enabled:
            dyn_proj = torch.zeros_like(dyn_proj)
        if not geo_enabled:
            geo_proj = torch.zeros_like(geo_proj)

        expert_memory_tokens = None
        context_tokens = vl_embeds
        flat_context_enabled = self._two_expert_flat_context_enabled()
        prefusion_enabled = self._two_expert_prefusion_enabled()
        prefusion_delta = vl_embeds.new_zeros(vl_embeds.shape)
        prefusion_condition_dropped = vl_embeds.new_zeros(())
        prefusion_token_keep_ratio = vl_embeds.new_ones(())
        if prefusion_enabled:
            type_embedding = self.two_expert_memory_type_embedding.to(device=vl_embeds.device, dtype=vl_embeds.dtype)
            memory_scale = self.two_expert_memory_scale.to(device=vl_embeds.device, dtype=vl_embeds.dtype)
            dyn_memory = (dyn_proj + type_embedding[0].view(1, 1, -1)) * memory_scale
            geo_memory = (geo_proj + type_embedding[1].view(1, 1, -1)) * memory_scale
            if not dyn_enabled:
                dyn_memory = torch.zeros_like(dyn_memory)
            if not geo_enabled:
                geo_memory = torch.zeros_like(geo_memory)
            f_dyn = dyn_proj.new_zeros(vl_embeds.shape[0], self.config.action_horizon, dyn_proj.shape[-1])
            f_geo = geo_proj.new_zeros(vl_embeds.shape[0], self.config.action_horizon, geo_proj.shape[-1])
            dyn_delta = f_dyn.new_zeros(f_dyn.shape)
            geo_delta = f_geo.new_zeros(f_geo.shape)
            expert_step_condition = None
            if mode == "raw_vlm_only":
                expert_memory_tokens = None
                context_tokens = vl_embeds
            else:
                expert_memory_tokens = torch.cat((dyn_memory, geo_memory), dim=1)
                expert_memory_for_attn, prefusion_condition_dropped, prefusion_token_keep_ratio, condition_keep = (
                    self._apply_two_expert_prefusion_dropout(expert_memory_tokens)
                )
                q = self.two_expert_prefusion_q_norm(vl_embeds)
                kv = self.two_expert_prefusion_memory_norm(expert_memory_for_attn)
                prefusion_attn, _ = self.two_expert_prefusion_attn(q, kv, kv, need_weights=False)
                prefusion_delta = self.two_expert_prefusion_out_proj(prefusion_attn)
                prefusion_delta = prefusion_delta * condition_keep.to(dtype=prefusion_delta.dtype)
                prefusion_scale = self.two_expert_prefusion_scale.to(device=vl_embeds.device, dtype=vl_embeds.dtype)
                context_tokens = vl_embeds + prefusion_delta * prefusion_scale
            context_mean = context_tokens.mean(1)
        elif flat_context_enabled:
            type_embedding = self.two_expert_memory_type_embedding.to(device=vl_embeds.device, dtype=vl_embeds.dtype)
            memory_scale = self.two_expert_memory_scale.to(device=vl_embeds.device, dtype=vl_embeds.dtype)
            dyn_memory = (dyn_proj + type_embedding[0].view(1, 1, -1)) * memory_scale
            geo_memory = (geo_proj + type_embedding[1].view(1, 1, -1)) * memory_scale
            if not dyn_enabled:
                dyn_memory = torch.zeros_like(dyn_memory)
            if not geo_enabled:
                geo_memory = torch.zeros_like(geo_memory)
            if mode == "raw_vlm_only":
                expert_memory_tokens = None
                context_tokens = vl_embeds
            else:
                expert_memory_tokens = torch.cat((dyn_memory, geo_memory), dim=1)
                context_tokens = torch.cat((vl_embeds, expert_memory_tokens), dim=1)
            f_dyn = dyn_proj.new_zeros(vl_embeds.shape[0], self.config.action_horizon, dyn_proj.shape[-1])
            f_geo = geo_proj.new_zeros(vl_embeds.shape[0], self.config.action_horizon, geo_proj.shape[-1])
            dyn_delta = f_dyn.new_zeros(f_dyn.shape)
            geo_delta = f_geo.new_zeros(f_geo.shape)
            expert_step_condition = None
            context_mean = context_tokens.mean(1)
        else:
            queries = self.two_expert_horizon_queries.unsqueeze(0).expand(vl_embeds.shape[0], -1, -1).to(vl_embeds)
            f_dyn, _ = self.two_expert_dyn_horizon_attn(queries, dyn_proj, dyn_proj, need_weights=False)
            f_geo, _ = self.two_expert_geo_horizon_attn(queries, geo_proj, geo_proj, need_weights=False)
            if not dyn_enabled:
                f_dyn = torch.zeros_like(f_dyn)
            if not geo_enabled:
                f_geo = torch.zeros_like(f_geo)
            dyn_delta = self.two_expert_dyn_delta_proj(f_dyn)
            geo_delta = self.two_expert_geo_delta_proj(f_geo)
            if not dyn_enabled:
                dyn_delta = torch.zeros_like(dyn_delta)
            if not geo_enabled:
                geo_delta = torch.zeros_like(geo_delta)
            expert_step_condition = dyn_delta + geo_delta
            context_mean = vl_embeds.mean(1)
            if self._two_expert_denoise_v2_enabled() and self.config.two_expert_memory_tokens_to_dit:
                type_embedding = self.two_expert_memory_type_embedding.to(device=vl_embeds.device, dtype=vl_embeds.dtype)
                memory_scale = self.two_expert_memory_scale.to(device=vl_embeds.device, dtype=vl_embeds.dtype)
                dyn_memory = (dyn_proj + type_embedding[0].view(1, 1, -1)) * memory_scale
                geo_memory = (geo_proj + type_embedding[1].view(1, 1, -1)) * memory_scale
                if not dyn_enabled:
                    dyn_memory = torch.zeros_like(dyn_memory)
                if not geo_enabled:
                    geo_memory = torch.zeros_like(geo_memory)
                expert_memory_tokens = torch.cat((dyn_memory, geo_memory), dim=1)
                context_tokens = torch.cat((vl_embeds, expert_memory_tokens), dim=1)
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
                    "random_slots": 7,
                }[mode]
            ),
            "two_expert_raw_vlm_context_used": vl_embeds.new_tensor(1.0),
            "two_expert_denoise_v2_enabled": vl_embeds.new_tensor(float(self._two_expert_denoise_v2_enabled())),
            "two_expert_flat_context_enabled": vl_embeds.new_tensor(float(flat_context_enabled)),
            "two_expert_prefusion_enabled": vl_embeds.new_tensor(float(prefusion_enabled)),
            "two_expert_memory_token_count": vl_embeds.new_tensor(float(0 if expert_memory_tokens is None else expert_memory_tokens.shape[1])),
            "two_expert_context_token_count": vl_embeds.new_tensor(float(context_tokens.shape[1])),
            "two_expert_memory_scale": self.two_expert_memory_scale.detach().to(device=vl_embeds.device, dtype=vl_embeds.dtype),
            "two_expert_prefusion_scale": self.two_expert_prefusion_scale.detach().to(device=vl_embeds.device, dtype=vl_embeds.dtype),
            "two_expert_prefusion_delta_norm": prefusion_delta.detach().float().norm(dim=-1).mean().to(dtype=vl_embeds.dtype),
            "two_expert_prefusion_condition_dropped": prefusion_condition_dropped.to(dtype=vl_embeds.dtype),
            "two_expert_prefusion_token_keep_ratio": prefusion_token_keep_ratio.to(dtype=vl_embeds.dtype),
            "two_expert_prefusion_zero_init": vl_embeds.new_tensor(
                float(
                    torch.count_nonzero(self.two_expert_prefusion_out_proj.weight.detach()).item() == 0
                    and torch.count_nonzero(self.two_expert_prefusion_out_proj.bias.detach()).item() == 0
                )
            ),
            "two_expert_denoise_condition_scale": self.two_expert_denoise_condition_scale.detach().to(
                device=vl_embeds.device,
                dtype=vl_embeds.dtype,
            ),
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
            "context_tokens": context_tokens,
            "context_mean": context_mean,
            "expert_step_condition": expert_step_condition,
            "f_dyn": f_dyn,
            "f_geo": f_geo,
            "dyn_delta": dyn_delta,
            "geo_delta": geo_delta,
            "expert_memory_tokens": expert_memory_tokens,
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
        cached_planning_condition_tokens: Optional[torch.Tensor] = None,
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
        adapter_tokens: Optional[torch.Tensor] = None
        adapter_diagnostics: Dict[str, torch.Tensor] = {}
        if self.planning_adapter is not None and cached_planning_condition_tokens is not None:
            adapter_tokens = cached_planning_condition_tokens.to(vl_embeds)
            adapter_diagnostics = {
                "planning_token_norm": adapter_tokens.detach().float().norm(dim=-1).mean().to(adapter_tokens),
                "planning_token_pairwise_cosine": self.planning_adapter._pairwise_cosine(adapter_tokens.detach()),
                "planning_condition_keep_ratio": adapter_tokens.new_tensor(1.0),
                "planning_adapter_forward_count": adapter_tokens.new_tensor(
                    float(self.planning_adapter.forward_count)
                ),
            }
        elif self.planning_adapter is not None:
            batch_size = int(vl_embeds.shape[0])
            if action_input is None:
                status_feature = vl_embeds.new_zeros((batch_size, 8))
                high_command = vl_embeds.new_zeros((batch_size, 3))
                history = vl_embeds.new_zeros((batch_size, 12))
            else:
                status_feature = action_input.get("status_feature")
                high_command = action_input.get("high_command_one_hot")
                history = action_input.get("his_traj", action_input.get("history_trajectory"))
                if not isinstance(status_feature, torch.Tensor):
                    status_feature = vl_embeds.new_zeros((batch_size, 8))
                if not isinstance(high_command, torch.Tensor):
                    high_command = vl_embeds.new_zeros((batch_size, 3))
                if not isinstance(history, torch.Tensor):
                    history = vl_embeds.new_zeros((batch_size, 12))
            adapter_tokens, adapter_diagnostics = self.planning_adapter(
                vl_embeds,
                status_feature,
                high_command,
                history,
                condition_dropout_enabled=training,
            )

        def finalize_static_planning(payload: Dict[str, Any]) -> Dict[str, Any]:
            context_mean = payload["context_mean"]
            if adapter_tokens is None:
                payload.setdefault("planning_condition_tokens", None)
                payload.setdefault("planning_context_mean_residual", torch.zeros_like(context_mean))
                payload.setdefault("planning_condition_is_static", False)
                return payload
            assert self.planning_adapter is not None
            context_gate = torch.sigmoid(self.planning_adapter.context_gate_logit).to(adapter_tokens)
            context_residual = context_gate * adapter_tokens.mean(dim=1)
            payload["context_mean"] = context_mean + context_residual
            payload["planning_condition_tokens"] = adapter_tokens
            payload["planning_context_mean_residual"] = context_residual
            payload["planning_condition_is_static"] = True
            diagnostics = payload.setdefault("diagnostics", {})
            diagnostics.update(adapter_diagnostics)
            diagnostics["planning_context_gate"] = context_gate.detach().float()
            return payload
        if (
            not self.config.use_expert_features
            and not self.config.use_last_rd
            and not self.config.use_last_vla
            and not self.config.use_two_expert_slots
        ):
            return finalize_static_planning({
                "vl_embeds": vl_embeds,
                "context_tokens": vl_embeds,
                "context_mean": vl_embeds.mean(1),
                "expert_step_condition": None,
                **base_losses,
                "diagnostics": {},
            })

        if self.config.use_two_expert_slots:
            two_expert_context = self._build_two_expert_context(vl_embeds, action_input)
            return finalize_static_planning({
                "vl_embeds": vl_embeds,
                "context_tokens": two_expert_context["context_tokens"],
                "context_mean": two_expert_context["context_mean"],
                "expert_step_condition": two_expert_context["expert_step_condition"],
                "cot_condition_tokens": None,
                **base_losses,
                "diagnostics": two_expert_context["diagnostics"],
                "two_expert_f_dyn": two_expert_context["f_dyn"],
                "two_expert_f_geo": two_expert_context["f_geo"],
                "two_expert_dyn_delta": two_expert_context["dyn_delta"],
                "two_expert_geo_delta": two_expert_context["geo_delta"],
                "two_expert_memory_tokens": two_expert_context["expert_memory_tokens"],
                "selected_target_norm": target_action_norm,
            })

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
                output_bound_fn=(
                    (lambda value: self._bound_output_representation(value, None))
                    if self._uses_fs_norm()
                    else None
                ),
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
            return finalize_static_planning({
                "vl_embeds": vl_embeds,
                "context_tokens": context_tokens,
                "context_mean": context_mean,
                "expert_step_condition": expert["horizon_residual"],
                **base_losses,
                "diagnostics": expert["diagnostics"],
            })

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
        return finalize_static_planning({
            "vl_embeds": vl_embeds,
            "context_tokens": context_tokens,
            "context_mean": context_mean,
            "expert_step_condition": expert_step_condition,
            **base_losses,
            "diagnostics": diagnostics,
            "last_rd_output": last_rd_output,
        })

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
            and not self.config.use_planning_token_adapter
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
            "two_expert_h_dyn",
            "two_expert_h_geo",
            "two_expert_corruption_mode",
        ):
            if key in action_input and isinstance(action_input[key], torch.Tensor):
                data[key] = action_input[key].repeat_interleave(repeat, 0)
            elif key in action_input and isinstance(action_input[key], bool):
                data[key] = action_input[key]
            elif key in action_input and isinstance(action_input[key], str):
                data[key] = action_input[key]
        if self.config.use_expert_features:
            for stream, enabled in (("jepa", self.config.use_jepa), ("vggt", self.config.use_vggt)):
                if not enabled or f"{stream}_context_tokens" in data:
                    continue
                tokens = self._resolve_expert_tokens(action_input, stream, "context", required=True)
                assert tokens is not None
                data[f"{stream}_context_tokens"] = tokens.repeat_interleave(repeat, 0)
        return BatchFeature(data=data)

    def _slice_action_input_batch(
        self,
        action_input: Optional[BatchFeature],
        start: int,
        end: int,
    ) -> Optional[BatchFeature]:
        if action_input is None:
            return None
        data: Dict[str, Any] = {}
        for key, value in action_input.items():
            if isinstance(value, torch.Tensor):
                if value.ndim > 0 and value.shape[0] >= end:
                    data[key] = value[start:end]
                else:
                    data[key] = value
            else:
                data[key] = value
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
                anchor_norm = self._encode_action_target(anchor_value.to(device=reference_norm.device, dtype=reference_norm.dtype))
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

    def _build_training_target(
        self,
        *,
        raw_trajectory: Optional[torch.Tensor],
        selected_repr: Optional[torch.Tensor] = None,
        action_input: Optional[BatchFeature] = None,
        selected_source_code: Optional[torch.Tensor] = None,
        selected_target_is_gt: Optional[torch.Tensor] = None,
        diagnostics: Optional[Dict[str, torch.Tensor]] = None,
    ) -> TrainingTarget:
        if selected_repr is None:
            if raw_trajectory is None:
                raise ValueError("TrainingTarget requires raw_trajectory or selected_repr.")
            selected_repr = self._encode_action_target(raw_trajectory)
        selected = selected_repr.detach()
        if raw_trajectory is None:
            raw = self._decode_action_target(selected).detach()
        else:
            raw = raw_trajectory.to(device=selected.device, dtype=selected.dtype).detach()
        if raw.shape != selected.shape:
            raise ValueError(
                f"TrainingTarget raw/representation shapes must match, got {tuple(raw.shape)} and {tuple(selected.shape)}."
            )
        diffusion_target, residual_alpha, residual_anchor, residual_diag = self._last_vla_diffusion_target_info(
            selected,
            training=self.training,
            action_input=action_input,
        )
        target_diag: Dict[str, torch.Tensor] = dict(diagnostics or {})
        target_diag.update(residual_diag)
        if selected_target_is_gt is None:
            selected_target_is_gt = selected.new_ones((selected.shape[0],))
        gt_ratio = selected_target_is_gt.detach().to(device=selected.device, dtype=torch.float32).mean()
        if selected_source_code is None:
            source_code = selected.new_tensor(1.0)
        else:
            source_code = selected_source_code.detach().to(device=selected.device, dtype=torch.float32).mean()
        target_diag.update(
            {
                "selected_target_is_gt_ratio": gt_ratio.to(selected),
                "selected_target_source_code": source_code.to(selected),
                "residual_alpha": selected.new_tensor(float(residual_alpha)),
            }
        )
        if self._uses_fs_norm():
            abs_target = selected.detach().float().abs()
            target_diag["fs_target_abs_gt3_ratio"] = (abs_target > 3.0).float().mean().to(selected)
            target_diag["fs_target_abs_gt5_ratio"] = (abs_target > 5.0).float().mean().to(selected)
        else:
            target_diag["fs_target_abs_gt3_ratio"] = selected.new_zeros(())
            target_diag["fs_target_abs_gt5_ratio"] = selected.new_zeros(())
        return TrainingTarget(
            raw_trajectory=raw,
            selected_repr=selected,
            diffusion_target_repr=diffusion_target.detach(),
            residual_anchor_repr=None if residual_anchor is None else residual_anchor.detach(),
            residual_alpha=float(residual_alpha),
            diagnostics=target_diag,
        )

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
            teacher_norm = self._encode_action_target(
                action_input["teacher_trajectory"].to(device=gt_actions_norm.device, dtype=gt_actions_norm.dtype)
            )
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

    def _two_expert_denoise_step_condition(
        self,
        action_features: torch.Tensor,
        timesteps: torch.Tensor,
        dit_context: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        base_condition = dit_context.get("expert_step_condition")
        if not self._two_expert_denoise_v2_enabled():
            return base_condition

        f_dyn = dit_context.get("two_expert_f_dyn")
        f_geo = dit_context.get("two_expert_f_geo")
        dyn_delta = dit_context.get("two_expert_dyn_delta")
        geo_delta = dit_context.get("two_expert_geo_delta")
        if not all(isinstance(value, torch.Tensor) for value in (f_dyn, f_geo, dyn_delta, geo_delta)):
            return base_condition

        f_dyn = f_dyn.to(device=action_features.device, dtype=action_features.dtype)
        f_geo = f_geo.to(device=action_features.device, dtype=action_features.dtype)
        dyn_delta = dyn_delta.to(device=action_features.device, dtype=action_features.dtype)
        geo_delta = geo_delta.to(device=action_features.device, dtype=action_features.dtype)
        timestep_features = self.model.timestep_encoder(timesteps.to(device=action_features.device))
        timestep_features = timestep_features.to(device=action_features.device, dtype=action_features.dtype)
        timestep_features = timestep_features.unsqueeze(1).expand(-1, action_features.shape[1], -1)

        gate_input = torch.cat((action_features, timestep_features, f_dyn, f_geo), dim=-1)
        gate_logits = self.two_expert_denoise_gate(gate_input)
        gate_logits = gate_logits / float(self.config.two_expert_denoise_gate_temperature)
        gate = F.softmax(gate_logits, dim=-1)

        scale = self.two_expert_denoise_condition_scale.to(device=action_features.device, dtype=action_features.dtype)
        expert_condition = (gate[..., :1] * dyn_delta + gate[..., 1:2] * geo_delta) * scale
        diagnostics = dit_context.setdefault("diagnostics", {})
        gate_float = gate.detach().float().clamp_min(1e-6)
        entropy = -(gate_float * gate_float.log()).sum(dim=-1).mean()
        diagnostics["two_expert_gate_dyn_mean"] = gate[..., 0].detach().float().mean().to(
            device=action_features.device,
            dtype=action_features.dtype,
        )
        diagnostics["two_expert_gate_geo_mean"] = gate[..., 1].detach().float().mean().to(
            device=action_features.device,
            dtype=action_features.dtype,
        )
        diagnostics["two_expert_gate_confidence"] = gate.detach().float().max(dim=-1).values.mean().to(
            device=action_features.device,
            dtype=action_features.dtype,
        )
        diagnostics["two_expert_gate_entropy"] = entropy.to(device=action_features.device, dtype=action_features.dtype)
        diagnostics["two_expert_denoise_condition_norm"] = expert_condition.detach().float().norm(dim=-1).mean().to(
            device=action_features.device,
            dtype=action_features.dtype,
        )
        diagnostics["two_expert_denoise_action_feature_norm"] = action_features.detach().float().norm(dim=-1).mean().to(
            device=action_features.device,
            dtype=action_features.dtype,
        )
        diagnostics["two_expert_denoise_timestep_feature_norm"] = timestep_features.detach().float().norm(dim=-1).mean().to(
            device=action_features.device,
            dtype=action_features.dtype,
        )
        return expert_condition

    def _apply_two_expert_denoise_condition(
        self,
        fused_input: torch.Tensor,
        action_features: torch.Tensor,
        timesteps: torch.Tensor,
        dit_context: Dict[str, Any],
    ) -> torch.Tensor:
        expert_step_condition = self._two_expert_denoise_step_condition(action_features, timesteps, dit_context)
        if expert_step_condition is None:
            return fused_input
        return fused_input + expert_step_condition.to(device=fused_input.device, dtype=fused_input.dtype)

    def _reset_planning_adapter_forward_count(self) -> None:
        if self.planning_adapter is not None:
            self.planning_adapter.reset_forward_count()

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
        planning_condition_tokens = dit_context.get("planning_condition_tokens")
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
            fused_input = self._apply_two_expert_denoise_condition(
                fused_input,
                action_features,
                timesteps,
                dit_context,
            )
        model_output = self.model(
            hidden_states=fused_input,
            encoder_hidden_states=context_embeds,
            conditioning_features=ego_status_features,
            timesteps=timesteps,
            cot_condition_tokens=cot_condition_tokens,
            planning_condition_tokens=planning_condition_tokens,
            planning_condition_layers=self.config.planning_condition_layers,
            cot_condition_layers=self.config.last_vla_cot_condition_layers,
        )
        if planning_condition_tokens is not None:
            diagnostics = dit_context.setdefault("diagnostics", {})
            diagnostics["planning_delta_norm"] = getattr(
                self.model,
                "last_planning_condition_delta_norm",
                context_embeds.new_zeros(()),
            )
            diagnostics["planning_layer_gate_mean"] = getattr(
                self.model,
                "last_planning_layer_gate_mean",
                context_embeds.new_zeros(()),
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

    def _uses_fs_norm(self) -> bool:
        return self.fs_norm_transform is not None

    def _lfp_runtime_config(self) -> LFPGRPOConfig:
        return getattr(self, "lfp_grpo_cfg", self.config.lfp_grpo_cfg)

    def _lfp_fs_transition_representation_floor(
        self,
        ref: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.stage3_algorithm != "lfp_grpo":
            return None
        cfg = self._lfp_runtime_config()
        if not bool(cfg.fs_transition_std_enabled):
            return None
        if self.fs_norm_transform is None:
            raise RuntimeError("FS-aware LFP transition covariance requires FS-Norm.")
        _, fs_scale = self.fs_norm_transform._center_scale(ref)
        endpoint_std = ref.new_tensor(
            (
                float(cfg.fs_endpoint_std_x_m),
                float(cfg.fs_endpoint_std_y_m),
                float(cfg.fs_endpoint_std_heading_rad),
            )
        )
        return normalized_transition_floor_from_endpoint_std(fs_scale, endpoint_std)

    def _apply_lfp_transition_std_floor(
        self,
        std: torch.Tensor,
        scalar_floor: float,
    ) -> torch.Tensor:
        representation_floor = self._lfp_fs_transition_representation_floor(std)
        return apply_transition_std_floor(std, scalar_floor, representation_floor)

    def _lfp_transition_floor_diagnostics(
        self,
        ref: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.stage3_algorithm != "lfp_grpo" or self.fs_norm_transform is None:
            return {}
        _, fs_scale = self.fs_norm_transform._center_scale(ref)
        representation_floor = self._lfp_fs_transition_representation_floor(ref)
        scalar_floor = float(self.min_sampling_denoising_std)
        effective_floor = apply_transition_std_floor(
            torch.zeros_like(fs_scale),
            scalar_floor,
            representation_floor,
        )
        endpoint_std = endpoint_std_from_normalized_transition_floor(
            effective_floor,
            fs_scale,
        )
        cfg = self._lfp_runtime_config()
        return {
            "lfp_fs_transition_std_enabled": ref.new_tensor(
                float(bool(cfg.fs_transition_std_enabled))
            ),
            "lfp_transition_floor_normalized_mean": effective_floor.mean().detach(),
            "lfp_transition_floor_normalized_max": effective_floor.max().detach(),
            "lfp_transition_floor_endpoint_std_x_m": endpoint_std[0].detach(),
            "lfp_transition_floor_endpoint_std_y_m": endpoint_std[1].detach(),
            "lfp_transition_floor_endpoint_std_heading_rad": endpoint_std[2].detach(),
        }

    def _fs_norm_clip_value(self, field_name: str) -> Optional[float]:
        value = float(getattr(self.config, field_name, -1.0))
        if value >= 0.0:
            return value if value > 0.0 else None
        legacy = float(getattr(self.config, "fs_norm_clip", 5.0))
        return legacy if legacy > 0.0 else None

    def _target_representation_clip_value(self, fallback: Optional[float]) -> Optional[float]:
        if self._uses_fs_norm():
            return None
        return fallback

    def _encode_action_target(self, raw_trajectory: torch.Tensor) -> torch.Tensor:
        if self._uses_fs_norm():
            assert self.fs_norm_transform is not None
            return self.fs_norm_transform.encode(raw_trajectory, apply_clip=False)
        return self.norm_odo(raw_trajectory)

    def _decode_action_target(self, representation: torch.Tensor) -> torch.Tensor:
        if self._uses_fs_norm():
            assert self.fs_norm_transform is not None
            return self.fs_norm_transform.decode(representation)
        return self.denorm_odo(representation)

    def _representation_clip_value(self, fallback: Optional[float]) -> Optional[float]:
        if self._uses_fs_norm():
            return self._fs_norm_clip_value("fs_norm_output_clip")
        return fallback

    def _bound_output_representation(
        self,
        representation: torch.Tensor,
        fallback: Optional[float],
    ) -> torch.Tensor:
        if self._uses_fs_norm() and str(getattr(self.config, "fs_norm_output_clip_mode", "scalar")) == "stats_bounds":
            assert self.fs_norm_transform is not None
            bounds = self.fs_norm_transform._clip_bounds(representation)
            if bounds is None:
                raise ValueError("fs_norm_output_clip_mode='stats_bounds' requires clip_lower/clip_upper statistics.")
            lower, upper = bounds
            return torch.minimum(torch.maximum(representation, lower), upper)
        clip = self._representation_clip_value(fallback)
        if clip is None:
            return representation
        return representation.clamp(-clip, clip)

    def _clip_output_representation(
        self,
        representation: torch.Tensor,
        fallback: Optional[float],
    ) -> torch.Tensor:
        return self._bound_output_representation(representation, fallback)

    def _flow_x0_from_velocity(
        self,
        noisy_actions: torch.Tensor,
        t_cont: torch.Tensor,
        pred_velocity: torch.Tensor,
    ) -> torch.Tensor:
        return noisy_actions + (1.0 - t_cont[:, None, None].to(noisy_actions)) * pred_velocity

    def _low_noise_mask(self, timesteps: torch.Tensor, *, method: str) -> torch.Tensor:
        frac = float(getattr(self.config, "x0_aux_low_noise_frac", 0.5))
        frac = min(max(frac, 0.0), 1.0)
        if frac <= 0.0:
            return torch.zeros_like(timesteps, dtype=torch.bool)
        if method == "flow":
            total = max(int(getattr(self.config.flow_cfg, "num_timestep_buckets", 1000)), 1)
            threshold = int(round((1.0 - frac) * total))
            return timesteps >= threshold
        total = max(int(getattr(self.config.ddpm_cfg, "num_train_timesteps", 100)), 1)
        threshold = int(round(frac * total))
        return timesteps <= threshold

    def _compute_x0_geo_aux_losses(
        self,
        x0_repr: torch.Tensor,
        target_raw: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        method: str,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        zero = x0_repr.new_zeros(())
        x0_weight = float(getattr(self.config, "x0_aux_weight", 0.0))
        delta_weight = float(getattr(self.config, "delta_aux_weight", 0.0))
        geo_weight = float(getattr(self.config, "geo_aux_weight", 0.0))
        per_x0, per_geo, active_mask, diagnostics = self._compute_x0_geo_aux_per_sample_losses(
            x0_repr,
            target_raw,
            timesteps,
            method=method,
        )
        if x0_weight <= 0.0 and delta_weight <= 0.0 and geo_weight <= 0.0:
            return zero, zero, diagnostics
        active = active_mask.to(device=per_x0.device, dtype=per_x0.dtype)
        active_count = active.sum().clamp(min=1.0)
        x0_aux_loss = (per_x0 * active).sum() / active_count
        geo_aux_loss = (per_geo * active).sum() / active_count
        per_delta = diagnostics.get("delta_aux_per_sample")
        if isinstance(per_delta, torch.Tensor):
            diagnostics["delta_aux_loss"] = (per_delta.to(active) * active).sum() / active_count
        else:
            diagnostics["delta_aux_loss"] = zero
        return x0_aux_loss.to(x0_repr), geo_aux_loss.to(x0_repr), diagnostics

    def _compute_x0_geo_aux_per_sample_losses(
        self,
        x0_repr: torch.Tensor,
        target_raw: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        method: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if x0_repr.ndim != 3 or x0_repr.shape[-1] != 3:
            raise ValueError(f"x0_repr must have shape [N, H, 3], got {tuple(x0_repr.shape)}.")
        zero = x0_repr.new_zeros(())
        per_zero = x0_repr.new_zeros((x0_repr.shape[0],))
        diagnostics = {
            "early_kink_rate": zero,
            "tail_reverse_rate": zero,
            "curvature_violation_rate": zero,
        }
        x0_weight = float(getattr(self.config, "x0_aux_weight", 0.0))
        delta_weight = float(getattr(self.config, "delta_aux_weight", 0.0))
        geo_weight = float(getattr(self.config, "geo_aux_weight", 0.0))
        active_mask = self._low_noise_mask(timesteps, method=method).to(device=x0_repr.device)
        if x0_weight <= 0.0 and delta_weight <= 0.0 and geo_weight <= 0.0:
            return per_zero, per_zero, active_mask, diagnostics
        if not bool(active_mask.any().detach().cpu().item()):
            return per_zero, per_zero, active_mask, diagnostics
        pred_raw = self._decode_action_target(x0_repr)
        target = target_raw.to(device=pred_raw.device, dtype=pred_raw.dtype)
        if target.shape != pred_raw.shape:
            raise ValueError(f"target_raw shape {tuple(target.shape)} does not match x0 raw shape {tuple(pred_raw.shape)}.")

        pred_f = pred_raw.float()
        target_f = target.float()
        xy_loss = F.smooth_l1_loss(pred_f[..., :2], target_f[..., :2], reduction="none").mean(dim=(1, 2))
        heading_loss = (
            (torch.sin(pred_f[..., 2]) - torch.sin(target_f[..., 2])).abs().mean(dim=1)
            + (torch.cos(pred_f[..., 2]) - torch.cos(target_f[..., 2])).abs().mean(dim=1)
        )
        x0_aux_loss = (xy_loss + heading_loss).to(dtype=x0_repr.dtype)

        if pred_f.shape[1] >= 2:
            pred_delta_xy = pred_f[:, 1:, :2] - pred_f[:, :-1, :2]
            target_delta_xy = target_f[:, 1:, :2] - target_f[:, :-1, :2]
            pred_delta_heading = torch.atan2(
                torch.sin(pred_f[:, 1:, 2] - pred_f[:, :-1, 2]),
                torch.cos(pred_f[:, 1:, 2] - pred_f[:, :-1, 2]),
            )
            target_delta_heading = torch.atan2(
                torch.sin(target_f[:, 1:, 2] - target_f[:, :-1, 2]),
                torch.cos(target_f[:, 1:, 2] - target_f[:, :-1, 2]),
            )
            pred_delta = torch.cat([pred_delta_xy, pred_delta_heading.unsqueeze(-1)], dim=-1)
            target_delta = torch.cat([target_delta_xy, target_delta_heading.unsqueeze(-1)], dim=-1)
            delta_aux_loss = F.smooth_l1_loss(pred_delta, target_delta, reduction="none").mean(dim=(1, 2)).to(
                dtype=x0_repr.dtype
            )
        else:
            delta_aux_loss = per_zero

        metrics = compute_feasibility_metrics(
            pred_f,
            {
                "w_curv": float(getattr(self.config, "geo_curvature_weight", 1.0)),
                "w_reverse": float(getattr(self.config, "geo_reverse_weight", 1.0)),
                "w_tail_reverse": float(getattr(self.config, "geo_tail_reverse_weight", 2.0)),
                "w_kink": float(getattr(self.config, "geo_early_kink_weight", 2.0)),
                "w_jerk": float(getattr(self.config, "geo_jerk_weight", 0.2)),
                "w_heading": 0.0,
            },
        )
        geo_aux_loss = metrics.feas_cost.reshape(-1).to(device=x0_repr.device, dtype=x0_repr.dtype)
        active_f = active_mask.to(device=x0_repr.device, dtype=x0_repr.dtype)
        x0_aux_loss = torch.where(active_mask, x0_aux_loss, per_zero)
        delta_aux_loss = torch.where(active_mask, delta_aux_loss, per_zero)
        geo_aux_loss = torch.where(active_mask, geo_aux_loss, per_zero)
        active_count = active_f.sum().clamp(min=1.0)
        diagnostics.update(
            {
                "early_kink_rate": (
                    metrics.early_kink_rate.reshape(-1).to(device=x0_repr.device, dtype=x0_repr.dtype) * active_f
                ).sum()
                / active_count,
                "tail_reverse_rate": (
                    metrics.tail_reverse_rate.reshape(-1).to(device=x0_repr.device, dtype=x0_repr.dtype) * active_f
                ).sum()
                / active_count,
                "curvature_violation_rate": (
                    metrics.curvature_violation_rate.reshape(-1).to(device=x0_repr.device, dtype=x0_repr.dtype) * active_f
                ).sum()
                / active_count,
                "delta_aux_per_sample": delta_aux_loss,
                "delta_aux_loss": (delta_aux_loss * active_f).sum() / active_count,
            }
        )
        return x0_aux_loss, geo_aux_loss, active_mask, diagnostics

    def _aux_warmup_ramp(self) -> float:
        warmup = max(int(getattr(self.config, "aux_warmup_epochs", 10)), 1)
        return min((int(getattr(self.config, "current_train_epoch", 0)) + 1) / float(warmup), 1.0)

    def _aux_alpha_weights(self, timesteps: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if self.config.sampling_method == "flow":
            raise ValueError("trajectory/feasibility auxiliaries do not support flow sampling.")
        alpha_bar = self.extract(self.ddpm_alphas_cumprod, timesteps, reference.shape)
        alpha_bar = alpha_bar.reshape(reference.shape[0], -1)[:, 0].detach().float()
        return alpha_bar.clamp(min=0.0, max=1.0).pow(float(self.config.aux_alpha_power)).to(reference)

    def _compute_pta_aux_per_sample_losses(
        self,
        predicted_diffusion_x0_repr: torch.Tensor,
        training_target: TrainingTarget,
        timesteps: torch.Tensor,
        *,
        method: str,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = int(predicted_diffusion_x0_repr.shape[0])
        zeros = predicted_diffusion_x0_repr.new_zeros((batch_size,))
        zero = predicted_diffusion_x0_repr.new_zeros(())
        ramp = self._aux_warmup_ramp()
        diagnostics: Dict[str, torch.Tensor] = {
            "trajectory_aux_loss": zero,
            "feasibility_aux_loss": zero,
            "tangent_excess_loss": zero,
            "curvature_excess_loss": zero,
            "aux_alpha_weight_mean": zero,
            "aux_warmup_ramp": predicted_diffusion_x0_repr.new_tensor(ramp),
            "full_x0_reconstruction_l1": zero,
            "fs_output_bound_hit_ratio": zero,
        }
        if method == "flow":
            if float(self.config.trajectory_aux_weight) > 0.0 or float(self.config.feasibility_aux_weight) > 0.0:
                raise ValueError("trajectory/feasibility auxiliaries support DDPM/DDIM only.")
            return zeros, zeros, diagnostics
        if float(self.config.trajectory_aux_weight) <= 0.0 and float(self.config.feasibility_aux_weight) <= 0.0:
            return zeros, zeros, diagnostics

        full_x0_repr = training_target.reconstruct_full_x0(predicted_diffusion_x0_repr)
        pred_raw = self._decode_action_target(full_x0_repr)
        target_raw = training_target.raw_trajectory.to(device=pred_raw.device, dtype=pred_raw.dtype)
        if pred_raw.shape != target_raw.shape:
            raise ValueError(
                f"PTA auxiliary prediction/target shapes differ: {tuple(pred_raw.shape)} vs {tuple(target_raw.shape)}."
            )
        pred_f = pred_raw.float()
        target_f = target_raw.detach().float()
        xy_loss = F.smooth_l1_loss(
            pred_f[..., :2],
            target_f[..., :2],
            beta=float(self.config.trajectory_huber_beta),
            reduction="none",
        ).mean(dim=(1, 2))
        heading_error = torch.atan2(
            torch.sin(pred_f[..., 2] - target_f[..., 2]),
            torch.cos(pred_f[..., 2] - target_f[..., 2]),
        )
        heading_loss = (1.0 - torch.cos(heading_error)).mean(dim=1)
        trajectory_loss = xy_loss + float(self.config.trajectory_heading_weight) * heading_loss
        tangent_loss, curvature_loss = compute_reference_relative_geometry_components(
            pred_f,
            target_f,
            float(self.config.tangent_margin_rad),
            float(self.config.curvature_margin),
            float(self.config.min_segment_length),
        )
        feasibility_loss = tangent_loss + curvature_loss
        alpha_weight = self._aux_alpha_weights(timesteps, full_x0_repr)
        weighted_trajectory = alpha_weight.float() * trajectory_loss
        weighted_feasibility = alpha_weight.float() * feasibility_loss.float()
        full_l1_per_sample = (
            full_x0_repr.float() - training_target.selected_repr.to(full_x0_repr).float()
        ).abs().mean(dim=(1, 2))
        if self._uses_fs_norm():
            bounded = self._bound_output_representation(full_x0_repr, None)
            bound_hit_per_sample = (
                (bounded.detach().float() - full_x0_repr.detach().float()).abs().gt(1e-7).float().mean(dim=(1, 2))
            )
        else:
            bound_hit_per_sample = zeros
        diagnostics.update(
            {
                "trajectory_aux_loss": weighted_trajectory.mean().to(predicted_diffusion_x0_repr),
                "feasibility_aux_loss": weighted_feasibility.mean().to(predicted_diffusion_x0_repr),
                "tangent_excess_loss": (alpha_weight.float() * tangent_loss.float()).mean().to(
                    predicted_diffusion_x0_repr
                ),
                "curvature_excess_loss": (alpha_weight.float() * curvature_loss.float()).mean().to(
                    predicted_diffusion_x0_repr
                ),
                "aux_alpha_weight_mean": alpha_weight.detach().float().mean().to(predicted_diffusion_x0_repr),
                "full_x0_reconstruction_l1": full_l1_per_sample.detach().mean().to(predicted_diffusion_x0_repr),
                "fs_output_bound_hit_ratio": bound_hit_per_sample.mean().to(predicted_diffusion_x0_repr),
                "_tangent_excess_per_sample": (alpha_weight.float() * tangent_loss.float()).to(
                    predicted_diffusion_x0_repr
                ),
                "_curvature_excess_per_sample": (alpha_weight.float() * curvature_loss.float()).to(
                    predicted_diffusion_x0_repr
                ),
                "_aux_alpha_weight_per_sample": alpha_weight.detach().to(predicted_diffusion_x0_repr),
                "_full_x0_reconstruction_l1_per_sample": full_l1_per_sample.detach().to(
                    predicted_diffusion_x0_repr
                ),
                "_fs_output_bound_hit_per_sample": bound_hit_per_sample.to(predicted_diffusion_x0_repr),
            }
        )
        return (
            weighted_trajectory.to(predicted_diffusion_x0_repr),
            weighted_feasibility.to(predicted_diffusion_x0_repr),
            diagnostics,
        )

    def _attach_training_aux_losses(
        self,
        dit_context: Dict[str, Any],
        predicted_diffusion_x0_repr: torch.Tensor,
        training_target: TrainingTarget,
        timesteps: torch.Tensor,
        *,
        method: str,
    ) -> None:
        x0_aux_loss, geo_aux_loss, geo_diag = self._compute_x0_geo_aux_losses(
            training_target.reconstruct_full_x0(predicted_diffusion_x0_repr),
            training_target.raw_trajectory,
            timesteps,
            method=method,
        )
        trajectory_per_sample, feasibility_per_sample, pta_diag = self._compute_pta_aux_per_sample_losses(
            predicted_diffusion_x0_repr,
            training_target,
            timesteps,
            method=method,
        )
        dit_context["training_target"] = training_target
        dit_context["x0_aux_loss"] = x0_aux_loss
        dit_context["delta_aux_loss"] = geo_diag.get("delta_aux_loss", x0_aux_loss.new_zeros(()))
        dit_context["geo_aux_loss"] = geo_aux_loss
        dit_context["trajectory_aux_loss"] = trajectory_per_sample.mean()
        dit_context["feasibility_aux_loss"] = feasibility_per_sample.mean()
        diagnostics = dit_context.setdefault("diagnostics", {})
        diagnostics.update(training_target.diagnostics)
        diagnostics.update(geo_diag)
        diagnostics.update({key: value for key, value in pta_diag.items() if not key.startswith("_")})

    def _new_auxiliary_weighted_loss(self, dit_context: Dict[str, Any], dtype: torch.dtype) -> torch.Tensor:
        reference = dit_context.get("trajectory_aux_loss")
        if not isinstance(reference, torch.Tensor):
            reference = next(self.parameters()).sum() * 0.0
        ramp = float(self._aux_warmup_ramp())
        return ramp * (
            float(self.config.trajectory_aux_weight) * dit_context.get("trajectory_aux_loss", reference.new_zeros(())).to(dtype=dtype)
            + float(self.config.feasibility_aux_weight)
            * dit_context.get("feasibility_aux_loss", reference.new_zeros(())).to(dtype=dtype)
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
        planning_condition_tokens: Optional[torch.Tensor] = None,
        cot_condition_tokens: Optional[torch.Tensor] = None,
        two_expert_f_dyn: Optional[torch.Tensor] = None,
        two_expert_f_geo: Optional[torch.Tensor] = None,
        two_expert_dyn_delta: Optional[torch.Tensor] = None,
        two_expert_geo_delta: Optional[torch.Tensor] = None,
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
                cached_planning_condition_tokens=(
                    planning_condition_tokens if self.planning_adapter is not None else None
                ),
            )
            context_embeds = latent_context["context_tokens"]
            context_mean = latent_context["context_mean"]
            expert_step_condition = latent_context["expert_step_condition"]
            planning_condition_tokens = latent_context.get("planning_condition_tokens")
            cot_condition_tokens = latent_context.get("cot_condition_tokens")
            two_expert_f_dyn = latent_context.get("two_expert_f_dyn")
            two_expert_f_geo = latent_context.get("two_expert_f_geo")
            two_expert_dyn_delta = latent_context.get("two_expert_dyn_delta")
            two_expert_geo_delta = latent_context.get("two_expert_geo_delta")
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
            step_context = {
                "expert_step_condition": expert_step_condition,
                "two_expert_f_dyn": two_expert_f_dyn,
                "two_expert_f_geo": two_expert_f_geo,
                "two_expert_dyn_delta": two_expert_dyn_delta,
                "two_expert_geo_delta": two_expert_geo_delta,
                "diagnostics": {},
            }
            fused_input = self._apply_two_expert_denoise_condition(
                fused_input,
                action_features,
                t,
                step_context,
            )

        model_output = self.model(
            hidden_states=fused_input,
            encoder_hidden_states=context_embeds,
            conditioning_features=ego_status_features,
            timesteps=t,
            cot_condition_tokens=cot_condition_tokens,
            planning_condition_tokens=planning_condition_tokens,
            planning_condition_layers=self.config.planning_condition_layers,
            cot_condition_layers=self.config.last_vla_cot_condition_layers,
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

        x_recon = self._bound_output_representation(
            x_recon,
            getattr(self, 'denoised_clip_value', 1.0),
        )

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
        self._reset_planning_adapter_forward_count()
        gt_actions = self._encode_action_target(action_input.action)
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

            teacher_ratio = target_diagnostics.get("teacher_traj_used_ratio", selected_target_norm.new_zeros(()))
            selected_is_gt = selected_target_norm.new_full(
                (selected_target_norm.shape[0],),
                1.0 - float(teacher_ratio.detach().float().item()),
            )
            selected_source_code = 1.0 + 10.0 * (1.0 - selected_is_gt)
            training_target = self._build_training_target(
                raw_trajectory=None,
                selected_repr=selected_target_norm,
                action_input=action_input,
                selected_source_code=selected_source_code,
                selected_target_is_gt=selected_is_gt,
                diagnostics=target_diagnostics,
            )
            diffusion_target = training_target.diffusion_target_repr
            residual_alpha = training_target.residual_alpha

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
                pred_velocity = self._denoise_model_output(noisy_actions, t_discrete, dit_context, action_input)
                diffusion_loss = F.mse_loss(pred_velocity, velocity_target, reduction="mean")
                x0_pred = self._flow_x0_from_velocity(noisy_actions, t_cont, pred_velocity)
                self._attach_training_aux_losses(
                    dit_context,
                    x0_pred,
                    training_target,
                    t_discrete,
                    method="flow",
                )
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
                pred_noise = self._denoise_model_output(noisy_actions, t_discrete, dit_context, action_input)
                diffusion_loss = F.mse_loss(pred_noise, noise, reduction="mean")
                x0_pred = self._x0_from_noise(noisy_actions, t_discrete, pred_noise)
                self._attach_training_aux_losses(
                    dit_context,
                    x0_pred,
                    training_target,
                    t_discrete,
                    method="ddpm",
                )
                policy_kd_loss = self._compute_policy_kd_loss(vl_features, action_input, noisy_actions, t_discrete, pred_noise)

            dit_context["policy_kd_loss"] = policy_kd_loss.to(dtype=diffusion_loss.dtype)
            dit_context["diagnostics"].update(target_diagnostics)
            dit_context["diagnostics"].update(self._last_vla_effective_aux_weights(diffusion_loss))
            loss = (
                float(self.config.diffusion_loss_weight) * diffusion_loss
                + self._last_vla_aux_loss(dit_context, diffusion_loss.dtype)
                + float(self.config.policy_kd_loss_weight) * policy_kd_loss.to(dtype=diffusion_loss.dtype)
                + float(self.config.x0_aux_weight) * dit_context["x0_aux_loss"].to(dtype=diffusion_loss.dtype)
                + float(getattr(self.config, "delta_aux_weight", 0.0)) * dit_context["delta_aux_loss"].to(dtype=diffusion_loss.dtype)
                + float(self.config.geo_aux_weight) * dit_context["geo_aux_loss"].to(dtype=diffusion_loss.dtype)
                + self._new_auxiliary_weighted_loss(dit_context, diffusion_loss.dtype)
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

        training_target = self._build_training_target(
            raw_trajectory=action_input.action,
            selected_repr=gt_actions,
            action_input=action_input,
        )
        diffusion_target = training_target.diffusion_target_repr
        residual_alpha = training_target.residual_alpha

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
            pred_velocity = self._denoise_model_output(noisy_actions, t_discrete, dit_context, action_input)
            diffusion_loss = F.mse_loss(pred_velocity, velocity_target, reduction='mean')
            x0_pred = self._flow_x0_from_velocity(noisy_actions, t_cont, pred_velocity)
            self._attach_training_aux_losses(
                dit_context,
                x0_pred,
                training_target,
                t_discrete,
                method="flow",
            )
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
            pred_noise = self._denoise_model_output(noisy_actions, t_discrete, dit_context, action_input)
            diffusion_loss = F.mse_loss(pred_noise, noise, reduction='mean')
            x0_pred = self._x0_from_noise(noisy_actions, t_discrete, pred_noise)
            self._attach_training_aux_losses(
                dit_context,
                x0_pred,
                training_target,
                t_discrete,
                method="ddpm",
            )
            policy_kd_loss = self._compute_policy_kd_loss(vl_features, action_input, noisy_actions, t_discrete, pred_noise)
        dit_context["policy_kd_loss"] = policy_kd_loss.to(dtype=diffusion_loss.dtype)

        jepa_alignment_loss = dit_context["jepa_alignment_loss"].to(dtype=diffusion_loss.dtype)
        vggt_alignment_loss = dit_context["vggt_alignment_loss"].to(dtype=diffusion_loss.dtype)
        loss = (
            float(self.config.diffusion_loss_weight) * diffusion_loss
            + self._stream_alignment_weight("jepa") * jepa_alignment_loss
            + self._stream_alignment_weight("vggt") * vggt_alignment_loss
            + self._last_rd_aux_loss(dit_context, diffusion_loss.dtype)
            + float(self.config.x0_aux_weight) * dit_context["x0_aux_loss"].to(dtype=diffusion_loss.dtype)
            + float(getattr(self.config, "delta_aux_weight", 0.0)) * dit_context["delta_aux_loss"].to(dtype=diffusion_loss.dtype)
            + float(self.config.geo_aux_weight) * dit_context["geo_aux_loss"].to(dtype=diffusion_loss.dtype)
            + self._new_auxiliary_weighted_loss(dit_context, diffusion_loss.dtype)
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
            "x0_aux_loss": dit_context.get("x0_aux_loss", loss.detach().new_tensor(0.0)).to(dtype=loss.dtype),
            "delta_aux_loss": dit_context.get("delta_aux_loss", loss.detach().new_tensor(0.0)).to(dtype=loss.dtype),
            "geo_aux_loss": dit_context.get("geo_aux_loss", loss.detach().new_tensor(0.0)).to(dtype=loss.dtype),
            "trajectory_aux_loss": dit_context.get("trajectory_aux_loss", loss.detach().new_tensor(0.0)).to(dtype=loss.dtype),
            "feasibility_aux_loss": dit_context.get("feasibility_aux_loss", loss.detach().new_tensor(0.0)).to(dtype=loss.dtype),
        }
        diagnostics = dit_context.get("diagnostics", {})
        if diagnostics:
            for key in (
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
            ):
                if key in diagnostics and isinstance(diagnostics[key], torch.Tensor):
                    output[key] = diagnostics[key].to(device=loss.device, dtype=loss.dtype)
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
                "two_expert_denoise_v2_enabled",
                "two_expert_flat_context_enabled",
                "two_expert_prefusion_enabled",
                "two_expert_memory_token_count",
                "two_expert_context_token_count",
                "two_expert_memory_scale",
                "two_expert_prefusion_scale",
                "two_expert_prefusion_delta_norm",
                "two_expert_prefusion_condition_dropped",
                "two_expert_prefusion_token_keep_ratio",
                "two_expert_prefusion_zero_init",
                "two_expert_denoise_condition_scale",
                "two_expert_h_dyn_norm",
                "two_expert_h_geo_norm",
                "two_expert_f_dyn_norm",
                "two_expert_f_geo_norm",
                "two_expert_dyn_expert_delta_norm",
                "two_expert_geo_expert_delta_norm",
                "two_expert_gate_dyn_mean",
                "two_expert_gate_geo_mean",
                "two_expert_gate_confidence",
                "two_expert_gate_entropy",
                "two_expert_denoise_condition_norm",
                "two_expert_denoise_action_feature_norm",
                "two_expert_denoise_timestep_feature_norm",
                "two_expert_zero_init_dyn",
                "two_expert_zero_init_geo",
                "teacher_traj_used_ratio",
                "teacher_traj_mix",
                "teacher_score_mean",
                "early_kink_rate",
                "tail_reverse_rate",
                "curvature_violation_rate",
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
        self._reset_planning_adapter_forward_count()
        self._warn_if_expert_targets_present(action_input, "get_action")
        dit_context = self._prepare_dit_context(vl_features, action_input, training=False, allow_target_tokens=False)
        context_embeds = dit_context["context_tokens"]
        context_mean = dit_context["context_mean"]
        expert_step_condition = dit_context["expert_step_condition"]
        planning_condition_tokens = dit_context.get("planning_condition_tokens")
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
                        cached_planning_condition_tokens=(
                            planning_condition_tokens if self.planning_adapter is not None else None
                        ),
                    )
                    context_embeds = dit_context["context_tokens"]
                    context_mean = dit_context["context_mean"]
                    expert_step_condition = dit_context["expert_step_condition"]
                    planning_condition_tokens = dit_context.get("planning_condition_tokens")
                    cot_condition_tokens = dit_context.get("cot_condition_tokens")

                action_features = self.action_encoder(current_actions, t)
                if hasattr(self, 'position_embedding'):
                    action_features += self.position_embedding(torch.arange(self.config.action_horizon, device=device))
                
                context_mean_features = context_mean.unsqueeze(1).repeat(1, self.config.action_horizon, 1)
                fused_input = self.fusion_projector(
                    torch.cat((history_embeds, context_mean_features, action_features), dim=2)
                )
                if expert_step_condition is not None:
                    fused_input = self._apply_two_expert_denoise_condition(
                        fused_input,
                        action_features,
                        t,
                        dit_context,
                    )
                
                model_output = self.model(
                    fused_input,
                    context_embeds,
                    ego_embeds,
                    t,
                    cot_condition_tokens=cot_condition_tokens,
                    planning_condition_tokens=planning_condition_tokens,
                    planning_condition_layers=self.config.planning_condition_layers,
                    cot_condition_layers=self.config.last_vla_cot_condition_layers,
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
                    planning_condition_tokens=planning_condition_tokens,
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
                
                if i == len(timesteps_to_iterate) - 1:
                    current_actions = self._clip_output_representation(
                        current_actions,
                        getattr(self, "final_action_clip_value", None),
                    )

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
                    planning_condition_tokens=planning_condition_tokens,
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

        current_actions = self._clip_output_representation(
            current_actions,
            getattr(self, "final_action_clip_value", 1.0),
        )

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
            output_data["pred_coarse_traj"] = self._decode_action_target(coarse_prior_norm.to(current_actions))
        if residual_anchor_norm is not None:
            output_data["pred_residual_norm"] = current_actions
            output_data["pred_vlm_text_anchor_traj"] = self._decode_action_target(residual_anchor_norm.to(current_actions))
            output_data["pred_residual_anchor_alpha"] = current_actions.new_tensor(float(residual_alpha))

        final_norm = self._clip_output_representation(
            final_norm,
            getattr(self, "final_action_clip_value", 1.0),
        )

        final_actions = self._decode_action_target(final_norm)
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
        prepared_dit_context: Optional[Dict[str, Any]] = None,
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
        if prepared_dit_context is None:
            self._reset_planning_adapter_forward_count()
            dit_context = self._prepare_dit_context(
                vl_features,
                action_input,
                training=self.training,
                allow_target_tokens=context_allow_target_tokens,
            )
        else:
            dit_context = prepared_dit_context
        context_embeds = dit_context["context_tokens"]
        context_mean = dit_context["context_mean"]
        expert_step_condition = dit_context["expert_step_condition"]
        planning_condition_tokens = dit_context.get("planning_condition_tokens")
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
        lfp_on_policy = self.stage3_algorithm == "lfp_grpo" and not deterministic
        bounded_final_norm: Optional[torch.Tensor] = None

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
                        cached_planning_condition_tokens=(
                            planning_condition_tokens if self.planning_adapter is not None else None
                        ),
                    )
                    context_embeds = dit_context["context_tokens"]
                    context_mean = dit_context["context_mean"]
                    expert_step_condition = dit_context["expert_step_condition"]
                    planning_condition_tokens = dit_context.get("planning_condition_tokens")
                    cot_condition_tokens = dit_context.get("cot_condition_tokens")
                
                action_features = self.action_encoder(current_actions, t_batch)
                if hasattr(self, 'position_embedding'):
                    action_features += self.position_embedding(torch.arange(self.config.action_horizon, device=device))
                
                context_mean_features = context_mean.unsqueeze(1).repeat(1, self.config.action_horizon, 1)
                fused_input = self.fusion_projector(
                    torch.cat((his_traj_features, context_mean_features, action_features), dim=2)
                )
                if expert_step_condition is not None:
                    fused_input = self._apply_two_expert_denoise_condition(
                        fused_input,
                        action_features,
                        t_batch,
                        dit_context,
                    )
                
                model_output = self.model(
                    fused_input,
                    context_embeds,
                    ego_status_features,
                    t_batch,
                    cot_condition_tokens=cot_condition_tokens,
                    planning_condition_tokens=planning_condition_tokens,
                    planning_condition_layers=self.config.planning_condition_layers,
                    cot_condition_layers=self.config.last_vla_cot_condition_layers,
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
                    planning_condition_tokens=planning_condition_tokens,
                    cot_condition_tokens=cot_condition_tokens,
                    vl_features=vl_features if action_input is not None else None,
                    action_input=action_input,
                )

                std = torch.exp(0.5 * logvar).to(dtype)
                noise_sample = torch.randn_like(current_actions)

                if self.config.sampling_method == 'ddim':
                    if deterministic:
                        std.zero_()
                    elif lfp_on_policy:
                        std = self._apply_lfp_transition_std_floor(
                            std,
                            self.min_sampling_denoising_std,
                        )
                    else:
                        std = std.clamp(min=self.min_sampling_denoising_std)
                else: # ddpm
                    if deterministic and t_int == 0:
                        std = torch.zeros_like(std)
                    elif deterministic:
                        std = std.clamp(min=1e-3)
                    elif lfp_on_policy:
                        std = self._apply_lfp_transition_std_floor(
                            std,
                            self.min_sampling_denoising_std,
                        )
                    else:
                        std = std.clamp(min=self.min_sampling_denoising_std)
                
                if (
                    not lfp_on_policy
                    and hasattr(self, 'randn_clip_value')
                    and self.randn_clip_value is not None
                ):
                    noise_sample = noise_sample.clamp_(-self.randn_clip_value, self.randn_clip_value)

                current_actions = mean + std * noise_sample
                
                if i == len(timesteps) - 1:
                    bounded_output = self._clip_output_representation(
                        current_actions,
                        getattr(self, "final_action_clip_value", None),
                    )
                    if lfp_on_policy:
                        # The scored action remains bounded, but the sampled transition stored
                        # in the chain must stay Gaussian for exact on-policy log-prob and KL.
                        bounded_final_norm = bounded_output
                    else:
                        current_actions = bounded_output
                
                denoising_chain.append(current_actions.clone())
        else:
            raise ValueError(f"Unsupported sampling method: {self.config.sampling_method}")

        residual_alpha = self._last_vla_residual_alpha(training=self.training)
        final_norm = bounded_final_norm if bounded_final_norm is not None else current_actions
        if residual_alpha != 0.0:
            residual_anchor_norm, _ = self._last_vla_residual_anchor_norm(
                action_input,
                current_actions,
                required=True,
            )
            assert residual_anchor_norm is not None
            final_norm = current_actions + float(residual_alpha) * residual_anchor_norm.to(current_actions)
            final_norm = self._clip_output_representation(
                final_norm,
                getattr(self, "final_action_clip_value", 1.0),
            )
        final_actions = self._decode_action_target(final_norm)
        chain_tensor = torch.stack(denoising_chain, dim=1)
        
        return chain_tensor.detach(), final_actions.detach()

    def _chain_transition_distribution(
        self,
        vl_features: torch.Tensor,
        his_traj_features: torch.Tensor,
        ego_status_features: torch.Tensor,
        chains: torch.Tensor,
        deterministic: bool = False,
        action_input: Optional[BatchFeature] = None,
        prepared_dit_context: Optional[Dict[str, Any]] = None,
    ) -> Normal:
        """Returns reverse-process transition distributions for a denoising chain."""
        if not self.training:
            self._warn_if_expert_targets_present(action_input, "_chain_transition_distribution")
        B, K1, H, D = chains.shape
        num_denoising_steps = K1 - 1
        
        dit_context = (
            prepared_dit_context
            if prepared_dit_context is not None
            else self._prepare_dit_context(vl_features, action_input, training=self.training)
        )
        context_embeds = dit_context["context_tokens"]
        context_mean = dit_context["context_mean"]
        expert_step_condition = dit_context["expert_step_condition"]
        planning_condition_tokens = dit_context.get("planning_condition_tokens")
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
        if planning_condition_tokens is not None:
            conditioning_embeds['planning_condition_tokens'] = planning_condition_tokens
        for key in ("two_expert_f_dyn", "two_expert_f_geo", "two_expert_dyn_delta", "two_expert_geo_delta"):
            value = dit_context.get(key)
            if isinstance(value, torch.Tensor):
                conditioning_embeds[key] = value
        
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
            planning_condition_tokens=batched_conditioning.get('planning_condition_tokens'),
            cot_condition_tokens=batched_conditioning.get('cot_condition_tokens'),
            two_expert_f_dyn=batched_conditioning.get('two_expert_f_dyn'),
            two_expert_f_geo=batched_conditioning.get('two_expert_f_geo'),
            two_expert_dyn_delta=batched_conditioning.get('two_expert_dyn_delta'),
            two_expert_geo_delta=batched_conditioning.get('two_expert_geo_delta'),
        )

        std = torch.exp(0.5 * logvar)
        if self.stage3_algorithm == "lfp_grpo":
            std = self._apply_lfp_transition_std_floor(
                std,
                self.min_logprob_denoising_std,
            )
        else:
            std = std.clamp(min=self.min_logprob_denoising_std)
        return Normal(mean, std)

    def get_logprobs(
        self,
        vl_features: torch.Tensor,
        his_traj_features: torch.Tensor,
        ego_status_features: torch.Tensor,
        chains: torch.Tensor,
        deterministic: bool = False,
        action_input: Optional[BatchFeature] = None,
        prepared_dit_context: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Calculates the log probability of a full denoising chain."""
        dist = self._chain_transition_distribution(
            vl_features,
            his_traj_features,
            ego_status_features,
            chains,
            deterministic=deterministic,
            action_input=action_input,
            prepared_dit_context=prepared_dit_context,
        )
        x_t_minus_1 = chains[:, 1:].reshape(-1, chains.shape[2], chains.shape[3])
        log_prob = dist.log_prob(x_t_minus_1)
        
        return log_prob

    @staticmethod
    def _slice_prepared_dit_context(
        context: Optional[Dict[str, Any]],
        start: int,
        end: int,
    ) -> Optional[Dict[str, Any]]:
        if context is None:
            return None
        sliced: Dict[str, Any] = {}
        for key, value in context.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] >= end:
                sliced[key] = value[start:end]
            elif key == "diagnostics" and isinstance(value, dict):
                sliced[key] = dict(value)
            else:
                sliced[key] = value
        return sliced

    def _chain_transition_reference_kl(
        self,
        vl_features: torch.Tensor,
        his_traj_features: torch.Tensor,
        ego_status_features: torch.Tensor,
        chains: torch.Tensor,
        action_input: Optional[BatchFeature],
        B: int,
        G: int,
        num_denoising_steps: int,
        discount: torch.Tensor,
        current_dit_context: Optional[Dict[str, Any]] = None,
        reference_dit_context: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Computes exact per-transition KL(current || frozen reference) on sampled chains."""
        if not hasattr(self, "old_policy"):
            raise RuntimeError("reference_kl_coeff > 0 requires a frozen old_policy reference.")
        total_samples = B * G
        if chains.shape[0] != total_samples:
            raise ValueError(f"Expected chains batch {total_samples}, got {chains.shape[0]}.")

        chunk_size = int(getattr(self, "reference_kl_chunk_size", 0))
        if chunk_size <= 0 or chunk_size >= total_samples:
            current_dist = self._chain_transition_distribution(
                vl_features,
                his_traj_features,
                ego_status_features,
                chains,
                deterministic=False,
                action_input=action_input,
                prepared_dit_context=current_dit_context,
            )
            self.old_policy.eval()
            with torch.no_grad():
                reference_dist = self.old_policy._chain_transition_distribution(
                    vl_features,
                    his_traj_features,
                    ego_status_features,
                    chains,
                    deterministic=False,
                    action_input=action_input,
                    prepared_dit_context=reference_dit_context,
                )
            step_kl = kl_divergence(current_dist, reference_dist).mean(dim=(1, 2))
            step_kl = step_kl.view(total_samples, num_denoising_steps)
            if self.trajectory_logprob_reduce == "discounted_mean":
                discount = discount.to(device=step_kl.device, dtype=step_kl.dtype)
                discount_norm = discount / discount.sum().clamp(min=1e-8)
                traj_kl = (step_kl * discount_norm.view(1, num_denoising_steps)).sum(dim=1)
            elif self.trajectory_logprob_reduce == "mean":
                traj_kl = step_kl.mean(dim=1)
            else:
                raise ValueError(f"Unsupported trajectory_logprob_reduce: {self.trajectory_logprob_reduce!r}")
            return traj_kl.mean()

        kl_sum = chains.new_zeros(())
        self.old_policy.eval()
        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)
            action_input_chunk = self._slice_action_input_batch(action_input, start, end)
            current_dist = self._chain_transition_distribution(
                vl_features[start:end],
                his_traj_features[start:end],
                ego_status_features[start:end],
                chains[start:end],
                deterministic=False,
                action_input=action_input_chunk,
                prepared_dit_context=self._slice_prepared_dit_context(current_dit_context, start, end),
            )
            with torch.no_grad():
                reference_dist = self.old_policy._chain_transition_distribution(
                    vl_features[start:end],
                    his_traj_features[start:end],
                    ego_status_features[start:end],
                    chains[start:end],
                    deterministic=False,
                    action_input=action_input_chunk,
                    prepared_dit_context=self._slice_prepared_dit_context(reference_dit_context, start, end),
                )
            step_kl = kl_divergence(current_dist, reference_dist).mean(dim=(1, 2))
            step_kl = step_kl.view(end - start, num_denoising_steps)
            if self.trajectory_logprob_reduce == "discounted_mean":
                discount_chunk = discount.to(device=step_kl.device, dtype=step_kl.dtype)
                discount_norm = discount_chunk / discount_chunk.sum().clamp(min=1e-8)
                traj_kl = (step_kl * discount_norm.view(1, num_denoising_steps)).sum(dim=1)
            elif self.trajectory_logprob_reduce == "mean":
                traj_kl = step_kl.mean(dim=1)
            else:
                raise ValueError(f"Unsupported trajectory_logprob_reduce: {self.trajectory_logprob_reduce!r}")
            kl_sum = kl_sum + traj_kl.sum()
        return kl_sum / float(total_samples)

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

    def _compute_soft_safety_penalty(self, components: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Continuous TTC/DDC penalty used to avoid over-hard progress suppression."""
        pdms = components["pdms"]
        penalty = torch.zeros_like(pdms)
        ttc_threshold = float(self.ttc_safe_threshold)
        if ttc_threshold > 0.0:
            ttc = components["time_to_collision_within_bound"]
            penalty = penalty + (ttc_threshold - ttc).clamp(min=0.0) / max(ttc_threshold, 1e-6)
        ddc_threshold = float(self.ddc_safe_threshold)
        if ddc_threshold > 0.0:
            ddc = components["driving_direction_compliance"]
            penalty = penalty + (ddc_threshold - ddc).clamp(min=0.0) / max(ddc_threshold, 1e-6)
        clip = float(self.soft_safety_penalty_clip)
        if clip > 0.0:
            penalty = penalty.clamp(max=clip)
        return penalty

    def _compute_pdms_core_components(
        self,
        components: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        ep = components["ego_progress"].float()
        ttc = components["time_to_collision_within_bound"].float()
        comfort = components["history_comfort"].float()
        nc = components["no_at_fault_collisions"].float()
        dac = components["drivable_area_compliance"].float()
        ddc = components["driving_direction_compliance"].float()
        core = ((5.0 * ep) + (5.0 * ttc) + (2.0 * comfort)) / 12.0
        core = core.clamp(min=0.0, max=1.0)
        pdms_formula = (nc * dac * core).clamp(min=0.0, max=1.0)
        nc_dac_feasible = (
            (nc >= float(self.nc_safe_threshold))
            & (dac >= float(self.dac_safe_threshold))
        )
        if bool(getattr(self, "core_pareto_use_ddc_guard", True)):
            ddc_guard = ddc >= float(self.core_pareto_ddc_guard_threshold)
        else:
            ddc_guard = torch.ones_like(nc_dac_feasible, dtype=torch.bool)
        guard_mask = nc_dac_feasible & ddc_guard

        ep_floor = float(self.core_pareto_ep_floor)
        ep_floor_gap = (ep_floor - ep).clamp(min=0.0) / max(ep_floor, 1e-6)
        ep_floor_penalty = float(self.core_pareto_ep_floor_penalty_weight) * ep_floor_gap

        ddc_threshold = float(self.core_pareto_ddc_guard_threshold)
        ddc_gap = (ddc_threshold - ddc).clamp(min=0.0) / max(ddc_threshold, 1e-6)
        ddc_gap = ddc_gap.clamp(max=float(self.core_pareto_ddc_penalty_clip))
        ddc_penalty = float(self.core_pareto_ddc_penalty_weight) * ddc_gap

        dual_penalty = torch.zeros_like(core)
        if bool(getattr(self, "core_pareto_use_adaptive_dual", False)):
            dual_lr = float(getattr(self, "core_pareto_dual_lr", 0.0))
            if self.training and dual_lr > 0.0:
                self.core_pareto_dual_nc = max(
                    0.0,
                    float(getattr(self, "core_pareto_dual_nc", 0.0))
                    + dual_lr * (float(self.core_pareto_target_nc) - float(nc.mean().detach().cpu())),
                )
                self.core_pareto_dual_dac = max(
                    0.0,
                    float(getattr(self, "core_pareto_dual_dac", 0.0))
                    + dual_lr * (float(self.core_pareto_target_dac) - float(dac.mean().detach().cpu())),
                )
                self.core_pareto_dual_ddc = max(
                    0.0,
                    float(getattr(self, "core_pareto_dual_ddc", 0.0))
                    + dual_lr * (float(self.core_pareto_target_ddc) - float(ddc.mean().detach().cpu())),
                )
            dual_penalty = (
                float(getattr(self, "core_pareto_dual_nc", 0.0))
                * (float(self.core_pareto_target_nc) - nc).clamp(min=0.0)
                + float(getattr(self, "core_pareto_dual_dac", 0.0))
                * (float(self.core_pareto_target_dac) - dac).clamp(min=0.0)
                + float(getattr(self, "core_pareto_dual_ddc", 0.0))
                * (float(self.core_pareto_target_ddc) - ddc).clamp(min=0.0)
            )

        adjusted_core = (core - ep_floor_penalty - ddc_penalty - dual_penalty).clamp(min=0.0, max=1.0)
        return {
            "core": core.to(components["pdms"]),
            "pdms_formula": pdms_formula.to(components["pdms"]),
            "nc_dac_feasible": nc_dac_feasible,
            "ddc_guard": ddc_guard,
            "guard_mask": guard_mask,
            "ep_floor_gap": ep_floor_gap.to(components["pdms"]),
            "ep_floor_penalty": ep_floor_penalty.to(components["pdms"]),
            "ddc_penalty": ddc_penalty.to(components["pdms"]),
            "dual_penalty": dual_penalty.to(components["pdms"]),
            "adjusted_core": adjusted_core.to(components["pdms"]),
        }

    def _compute_core_pareto_front_mask(
        self,
        ep: torch.Tensor,
        ttc: torch.Tensor,
        comfort: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        values = torch.stack([ep.float(), ttc.float(), comfort.float()], dim=-1)
        vi = values[:, :, None, :]
        vj = values[:, None, :, :]
        dominates = ((vj >= vi - 1e-6).all(dim=-1) & (vj > vi + 1e-6).any(dim=-1))
        pair_valid = valid_mask[:, :, None] & valid_mask[:, None, :]
        dominated = (dominates & pair_valid).any(dim=2)
        return valid_mask & ~dominated

    def _compute_legacy_core_pareto_advantages(
        self,
        rewards_matrix: torch.Tensor,
        hard_safe_matrix: torch.Tensor,
        reward_aux: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        B, G = rewards_matrix.shape
        ep = reward_aux["ego_progress"].reshape(B, G).float()
        ttc = reward_aux["time_to_collision_within_bound"].reshape(B, G).float()
        comfort = reward_aux["history_comfort"].reshape(B, G).float()
        nc = reward_aux["no_at_fault_collisions"].reshape(B, G).float()
        dac = reward_aux["drivable_area_compliance"].reshape(B, G).float()
        ddc = reward_aux["driving_direction_compliance"].reshape(B, G).float()
        core = reward_aux["core_pareto_core"].reshape(B, G).float()
        adjusted_core = reward_aux["core_pareto_adjusted_core"].reshape(B, G).float()
        ep_floor_gap = reward_aux["core_pareto_ep_floor_gap"].reshape(B, G).float()
        valid = hard_safe_matrix.bool()

        valid_f = valid.to(dtype=adjusted_core.dtype)
        valid_count = valid_f.sum(dim=1, keepdim=True)
        fallback_mean = adjusted_core.mean(dim=1, keepdim=True)
        masked_sum = (adjusted_core * valid_f).sum(dim=1, keepdim=True)
        group_mean = torch.where(valid_count > 0, masked_sum / valid_count.clamp(min=1.0), fallback_mean)
        centered = torch.where(valid, adjusted_core - group_mean, torch.zeros_like(adjusted_core))
        var = (centered.square() * valid_f).sum(dim=1, keepdim=True) / valid_count.clamp(min=1.0)
        std = var.sqrt().clamp(min=max(float(self.core_pareto_min_core_std), float(self.advantage_std_floor)))

        if str(getattr(self, "core_pareto_advantage_mode", "group_zscore")) == "loo_zscore" and G > 1:
            loo_count = (valid_count - valid_f).clamp(min=1.0)
            loo_sum = masked_sum - adjusted_core * valid_f
            loo_mean = torch.where(valid_count > 1.0, loo_sum / loo_count, group_mean)
            z = (adjusted_core - loo_mean) / std
        else:
            z = (adjusted_core - group_mean) / std

        pareto_front = self._compute_core_pareto_front_mask(ep, ttc, comfort, valid)
        valid_adv = z + float(self.core_pareto_pareto_bonus) * pareto_front.to(dtype=z.dtype)

        ep_low = ep_floor_gap > 0.0
        low_ep_scale = float(self.core_pareto_low_ep_adv_scale)
        valid_adv = torch.where(ep_low & (valid_adv > 0.0), valid_adv * low_ep_scale, valid_adv)

        invalid_adv = -float(self.core_pareto_infeasible_advantage_offset) + torch.clamp(z, max=0.0)
        advantages = torch.where(valid, valid_adv, invalid_adv)

        flat_advantages = advantages.flatten()
        adv_min = torch.quantile(flat_advantages.float(), float(self.clip_advantage_lower_quantile)).to(advantages)
        adv_max = torch.quantile(flat_advantages.float(), float(self.clip_advantage_upper_quantile)).to(advantages)
        advantages = advantages.clamp(min=adv_min, max=adv_max)

        safe_count = valid.sum(dim=1)
        mixed = (safe_count > 0) & (safe_count < G)
        all_safe = safe_count == G
        all_unsafe = safe_count == 0
        reward_std = std.squeeze(1).to(dtype=rewards_matrix.dtype)

        if self.use_dynamic_group_weight:
            group_weight = torch.ones(B, device=rewards_matrix.device, dtype=rewards_matrix.dtype)
            group_weight[all_safe & (reward_std <= float(self.min_group_reward_std))] = float(
                self.all_safe_low_std_group_weight
            )
            group_weight[all_unsafe] = float(self.all_unsafe_group_weight)
        else:
            group_weight = torch.ones(B, device=rewards_matrix.device, dtype=rewards_matrix.dtype)

        nc_dac_feasible = (nc >= float(self.nc_safe_threshold)) & (dac >= float(self.dac_safe_threshold))
        ddc_guard = ddc >= float(self.core_pareto_ddc_guard_threshold)
        margin = float(self.core_pareto_ep_ttc_balance_margin)
        progress_bucket = valid & (ep >= ttc + margin)
        ttc_bucket = valid & (ttc >= ep + margin)
        balanced_bucket = valid & ((ep - ttc).abs() < margin)

        aux = {
            "reward_std": reward_std.detach(),
            "safe_count": safe_count.detach().to(dtype=rewards_matrix.dtype),
            "mixed_group_ratio": mixed.float().mean().to(dtype=rewards_matrix.dtype),
            "all_safe_group_ratio": all_safe.float().mean().to(dtype=rewards_matrix.dtype),
            "all_unsafe_group_ratio": all_unsafe.float().mean().to(dtype=rewards_matrix.dtype),
            "core_pareto_front_ratio": pareto_front.float().mean().to(dtype=rewards_matrix.dtype),
            "core_pareto_valid_front_ratio": (
                (pareto_front & valid).float().sum() / valid.float().sum().clamp(min=1.0)
            ).to(dtype=rewards_matrix.dtype),
            "core_pareto_ep_low_ratio": ep_low.float().mean().to(dtype=rewards_matrix.dtype),
            "core_pareto_nc_dac_feasible_ratio": nc_dac_feasible.float().mean().to(dtype=rewards_matrix.dtype),
            "core_pareto_ddc_guard_pass_ratio": ddc_guard.float().mean().to(dtype=rewards_matrix.dtype),
            "core_pareto_progress_bucket_ratio": progress_bucket.float().mean().to(dtype=rewards_matrix.dtype),
            "core_pareto_ttc_bucket_ratio": ttc_bucket.float().mean().to(dtype=rewards_matrix.dtype),
            "core_pareto_balanced_bucket_ratio": balanced_bucket.float().mean().to(dtype=rewards_matrix.dtype),
            "core_pareto_core_std": core.std(dim=1, unbiased=False).mean().to(dtype=rewards_matrix.dtype),
        }
        return advantages.reshape(B * G).detach(), group_weight.detach(), aux

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
        soft_safety_penalty = torch.zeros_like(pdms)
        core_aux: Dict[str, torch.Tensor] = {}
        if str(getattr(self, "reward_mode", "safe_diffgrpo")) == "core_pareto":
            core_aux = self._compute_pdms_core_components(components)
            hard_safe_mask = core_aux["guard_mask"]
            unsafe_reward = float(self.unsafe_reward_floor) + float(self.unsafe_pdms_scale) * core_aux["pdms_formula"]
            reward = torch.where(hard_safe_mask, core_aux["adjusted_core"], unsafe_reward.to(pdms))
        elif self.use_safety_shaped_reward:
            safe_reward = pdms + float(self.progress_bonus_weight) * ego_progress
            if str(self.safety_advantage_mode) == "soft_penalty":
                soft_safety_penalty = self._compute_soft_safety_penalty(components)
                soft_reward = (
                    safe_reward
                    - float(self.soft_safety_penalty_weight) * soft_safety_penalty
                ).clamp(min=float(self.soft_safety_min_reward))
                unsafe_reward = float(self.unsafe_reward_floor) + float(self.unsafe_pdms_scale) * pdms
                reward = torch.where(hard_safe_mask, soft_reward, unsafe_reward)
            else:
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
            "soft_safety_penalty": soft_safety_penalty,
            "core_pareto_mode_enabled": pdms.new_tensor(
                float(str(getattr(self, "reward_mode", "safe_diffgrpo")) == "core_pareto")
            ),
        }
        if core_aux:
            aux.update(
                {
                    "core_pareto_core": core_aux["core"],
                    "core_pareto_pdms_formula": core_aux["pdms_formula"],
                    "core_pareto_adjusted_core": core_aux["adjusted_core"],
                    "core_pareto_ep_floor_gap": core_aux["ep_floor_gap"],
                    "core_pareto_ep_floor_penalty": core_aux["ep_floor_penalty"],
                    "core_pareto_ddc_penalty": core_aux["ddc_penalty"],
                    "core_pareto_dual_penalty": core_aux["dual_penalty"],
                    "core_pareto_nc_dac_feasible_mask": core_aux["nc_dac_feasible"].to(dtype=pdms.dtype),
                    "core_pareto_ddc_guard_mask": core_aux["ddc_guard"].to(dtype=pdms.dtype),
                }
            )
        else:
            zero = torch.zeros_like(pdms)
            aux.update(
                {
                    "core_pareto_core": zero,
                    "core_pareto_pdms_formula": zero,
                    "core_pareto_adjusted_core": zero,
                    "core_pareto_ep_floor_gap": zero,
                    "core_pareto_ep_floor_penalty": zero,
                    "core_pareto_ddc_penalty": zero,
                    "core_pareto_dual_penalty": zero,
                    "core_pareto_nc_dac_feasible_mask": zero,
                    "core_pareto_ddc_guard_mask": zero,
                }
            )
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
        reward_aux: Optional[Dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # rewards_matrix/hard_safe_matrix: [B, G]
        assert rewards_matrix.shape == hard_safe_matrix.shape
        B, G = rewards_matrix.shape

        if str(getattr(self, "reward_mode", "safe_diffgrpo")) == "core_pareto" and reward_aux is not None:
            return self._compute_legacy_core_pareto_advantages(rewards_matrix, hard_safe_matrix, reward_aux)

        advantage_mode = str(getattr(self, "advantage_mode", "safe_zscore"))
        mean_r = rewards_matrix.mean(dim=1, keepdim=True)
        if G > 1:
            std_r = rewards_matrix.std(dim=1, keepdim=True).clamp(min=float(self.advantage_std_floor))
            reward_std = rewards_matrix.std(dim=1)
        else:
            std_r = rewards_matrix.new_full((B, 1), float(self.advantage_std_floor))
            reward_std = rewards_matrix.new_zeros(B)
        centered = rewards_matrix - mean_r
        if advantage_mode == "safe_rpp":
            flat_centered = centered.reshape(-1)
            batch_mean = flat_centered.mean()
            batch_std = flat_centered.std(unbiased=False).clamp(min=1e-6)
            base_adv = ((centered - batch_mean) / batch_std).to(dtype=rewards_matrix.dtype)
        else:
            base_adv = centered / std_r

        if self.use_asymmetric_safe_advantage:
            safe_adv = torch.clamp(base_adv, min=0.0) + float(self.safe_negative_adv_scale) * torch.clamp(base_adv, max=0.0)
            unsafe_adv = -float(self.unsafe_advantage_offset) + torch.clamp(base_adv, max=0.0)
            advantages = torch.where(hard_safe_matrix, safe_adv, unsafe_adv)
        else:
            advantages = base_adv

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
            "safe_rpp_advantage_enabled": rewards_matrix.new_tensor(float(advantage_mode == "safe_rpp")),
            "safe_rpp_centered_batch_std": centered.reshape(-1).detach().float().std(unbiased=False).to(dtype=rewards_matrix.dtype),
        }
        return advantages.reshape(B * G).detach(), group_weight.detach(), aux

    def _compute_pdms_core(
        self,
        components: Dict[str, torch.Tensor],
        *,
        ep_weight: Optional[float] = None,
        ttc_weight: Optional[float] = None,
        comfort_weight: Optional[float] = None,
        normalizer: Optional[float] = None,
    ) -> torch.Tensor:
        ep = components["ego_progress"].float()
        ttc = components["time_to_collision_within_bound"].float()
        comfort = components["history_comfort"].float()
        ep_w = float(self.core_ep_weight if ep_weight is None else ep_weight)
        ttc_w = float(self.core_ttc_weight if ttc_weight is None else ttc_weight)
        comfort_w = float(self.core_comfort_weight if comfort_weight is None else comfort_weight)
        denom = float(self.core_normalizer if normalizer is None else normalizer)
        if denom <= 0.0:
            raise ValueError("PDMS core normalizer must be positive.")
        return (ep_w * ep + ttc_w * ttc + comfort_w * comfort) / denom

    def _reshape_components_for_group(
        self,
        components: Dict[str, torch.Tensor],
        B: int,
        G: int,
    ) -> Dict[str, torch.Tensor]:
        grouped: Dict[str, torch.Tensor] = {}
        for key, value in components.items():
            if not isinstance(value, torch.Tensor) or value.numel() != B * G:
                continue
            grouped[key] = value.reshape(B, G)
        return grouped

    def _select_core_pareto_reference_value(
        self,
        gt_value: torch.Tensor,
        il_value: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        if mode == "gt":
            return gt_value
        if mode == "il":
            return il_value
        if mode == "max_gt_il":
            return torch.maximum(gt_value, il_value)
        raise ValueError(f"Unsupported Core-Pareto reference mode: {mode!r}")

    def _core_pareto_reward_kwargs(self) -> Dict[str, Any]:
        offline_cfg = getattr(self, "offline_rl_cfg", None)
        if offline_cfg is None or not bool(getattr(offline_cfg, "enabled", False)):
            return {}
        return {
            "strict_submetrics": bool(offline_cfg.strict_reward_submetrics),
            "required_submetrics": offline_cfg.required_reward_submetrics,
            "missing_submetric_policy": str(offline_cfg.missing_submetric_policy),
            "use_batched_pdm_scoring": bool(offline_cfg.use_batched_pdm_scoring),
            "use_exact_array_pdm_state_conversion": bool(offline_cfg.use_exact_array_pdm_state_conversion),
            "use_fast_pdm_scorer": bool(offline_cfg.use_fast_pdm_scorer),
            "pdm_batch_chunk_size": int(offline_cfg.pdm_batch_chunk_size),
            "pdm_shadow_check": bool(offline_cfg.pdm_shadow_check),
            "pdm_shadow_max_samples": int(offline_cfg.pdm_shadow_max_samples),
            "pdm_shadow_max_abs_diff": float(offline_cfg.pdm_shadow_max_abs_diff),
        }

    def _compute_core_pareto_reference_components(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        tokens_list: list[str],
        metric_cache: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        gt_traj = action_input.action.detach()
        reward_kwargs = self._core_pareto_reward_kwargs()
        gt_pdms, gt_components = self.reward_fn(
            gt_traj,
            tokens_list,
            metric_cache,
            return_components=True,
            **reward_kwargs,
        )
        gt_components = dict(gt_components)
        gt_components["pdms"] = gt_pdms

        if hasattr(self, "old_policy") and self.old_policy is not None:
            self.old_policy.eval()
            with torch.no_grad():
                _, il_traj = self.old_policy.sample_chain(
                    vl_features,
                    action_input.his_traj,
                    action_input.status_feature,
                    deterministic=bool(self.core_pareto_reference_sample_deterministic),
                    action_input=action_input,
                    allow_target_tokens=False,
                )
            il_pdms, il_components = self.reward_fn(
                il_traj.detach(),
                tokens_list,
                metric_cache,
                return_components=True,
                **reward_kwargs,
            )
            il_components = dict(il_components)
            il_components["pdms"] = il_pdms
        else:
            il_pdms = gt_pdms
            il_components = dict(gt_components)

        gt_core = self._compute_pdms_core(gt_components)
        il_core = self._compute_pdms_core(il_components)
        ref_components: Dict[str, torch.Tensor] = {}
        for key in REQUIRED_COMPONENT_KEYS:
            if key in gt_components and key in il_components:
                ref_components[key] = self._select_core_pareto_reference_value(
                    gt_components[key].float(),
                    il_components[key].float(),
                    str(self.core_pareto_reference_mode),
                )
        ref_pdms = self._select_core_pareto_reference_value(
            gt_pdms.float(),
            il_pdms.float(),
            str(self.core_pareto_reference_mode),
        )
        ref_core = self._select_core_pareto_reference_value(
            gt_core.float(),
            il_core.float(),
            str(self.core_pareto_reference_mode),
        )
        ref_ep = self._select_core_pareto_reference_value(
            gt_components["ego_progress"].float(),
            il_components["ego_progress"].float(),
            str(self.core_pareto_ep_reference_mode),
        )
        ref_ttc = self._select_core_pareto_reference_value(
            gt_components["time_to_collision_within_bound"].float(),
            il_components["time_to_collision_within_bound"].float(),
            str(self.core_pareto_ttc_reference_mode),
        )
        ref_ddc = self._select_core_pareto_reference_value(
            gt_components["driving_direction_compliance"].float(),
            il_components["driving_direction_compliance"].float(),
            str(self.core_pareto_ddc_reference_mode),
        )
        ref_comfort = self._select_core_pareto_reference_value(
            gt_components["history_comfort"].float(),
            il_components["history_comfort"].float(),
            str(self.core_pareto_reference_mode),
        )
        ref_nc = self._select_core_pareto_reference_value(
            gt_components["no_at_fault_collisions"].float(),
            il_components["no_at_fault_collisions"].float(),
            str(self.core_pareto_reference_mode),
        )
        ref_dac = self._select_core_pareto_reference_value(
            gt_components["drivable_area_compliance"].float(),
            il_components["drivable_area_compliance"].float(),
            str(self.core_pareto_reference_mode),
        )
        return {
            "gt_components": gt_components,
            "il_components": il_components,
            "ref_components": ref_components,
            "gt_pdms": gt_pdms.float(),
            "il_pdms": il_pdms.float(),
            "ref_pdms": ref_pdms.float(),
            "gt_core": gt_core.float(),
            "il_core": il_core.float(),
            "ref_core": ref_core.float(),
            "ref_ep": ref_ep.float(),
            "ref_ttc": ref_ttc.float(),
            "ref_comfort": ref_comfort.float(),
            "ref_ddc": ref_ddc.float(),
            "ref_nc": ref_nc.float(),
            "ref_dac": ref_dac.float(),
        }

    def _compute_pareto_front_mask(
        self,
        objective_components: Dict[str, torch.Tensor],
        valid_mask: torch.Tensor,
        objectives: tuple[str, ...],
    ) -> torch.Tensor:
        if valid_mask.ndim != 2:
            raise ValueError("valid_mask must be [B, G].")
        values = []
        for key in objectives:
            if key not in objective_components:
                raise KeyError(f"Missing Pareto objective component: {key}")
            values.append(torch.nan_to_num(objective_components[key].float(), nan=-1e6))
        if not values:
            return torch.zeros_like(valid_mask, dtype=torch.bool)
        stacked = torch.stack(values, dim=-1)
        vi = stacked[:, :, None, :]
        vj = stacked[:, None, :, :]
        dominates = ((vj >= vi - 1e-6).all(dim=-1) & (vj > vi + 1e-6).any(dim=-1))
        pair_valid = valid_mask[:, :, None] & valid_mask[:, None, :]
        dominated = (dominates & pair_valid).any(dim=2)
        return valid_mask & ~dominated

    def _compute_phenotype_buckets(
        self,
        trajs_matrix: torch.Tensor,
        components: Dict[str, torch.Tensor],
        ref_ep: torch.Tensor,
    ) -> torch.Tensor:
        B, G = trajs_matrix.shape[:2]
        device = trajs_matrix.device
        if not bool(getattr(self, "core_pareto_use_phenotype_bucket_grpo", False)):
            return torch.zeros((B, G), device=device, dtype=torch.long)

        progress_bucket = torch.ones((B, G), device=device, dtype=torch.long)
        if bool(getattr(self, "core_pareto_bucket_by_progress", True)):
            ep = components["ego_progress"].float()
            progress_bucket = torch.where(
                ep >= ref_ep[:, None] + float(self.core_pareto_progress_fast_margin),
                torch.full_like(progress_bucket, 2),
                progress_bucket,
            )
            progress_bucket = torch.where(
                ep <= ref_ep[:, None] - float(self.core_pareto_progress_slow_margin),
                torch.zeros_like(progress_bucket),
                progress_bucket,
            )

        lateral_bucket = torch.ones((B, G), device=device, dtype=torch.long)
        if bool(getattr(self, "core_pareto_bucket_by_lateral_endpoint", True)):
            final_y = trajs_matrix[:, :, -1, 1]
            threshold = float(self.core_pareto_lateral_bucket_threshold_m)
            lateral_bucket = torch.where(final_y > threshold, torch.full_like(lateral_bucket, 2), lateral_bucket)
            lateral_bucket = torch.where(final_y < -threshold, torch.zeros_like(lateral_bucket), lateral_bucket)
        return progress_bucket * 3 + lateral_bucket

    def _masked_zscore(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        std_floor: float = 0.05,
    ) -> torch.Tensor:
        mask = mask.bool()
        values_f = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
        mask_f = mask.to(dtype=values_f.dtype)
        count = mask_f.sum(dim=1, keepdim=True)
        mean = (values_f * mask_f).sum(dim=1, keepdim=True) / count.clamp(min=1.0)
        centered = torch.where(mask, values_f - mean, torch.zeros_like(values_f))
        var = (centered.square() * mask_f).sum(dim=1, keepdim=True) / count.clamp(min=1.0)
        std = var.sqrt().clamp(min=float(std_floor))
        z = torch.where(mask, centered / std, torch.zeros_like(values_f))
        z = torch.where(count >= 2.0, z, torch.zeros_like(z))
        return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).to(values)

    def _update_core_pareto_dual(
        self,
        slow_violation_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        ddc_drop_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        device = slow_violation_mask.device
        dtype = torch.float32
        if not bool(getattr(self, "core_pareto_use_adaptive_dual", False)):
            return {
                "lambda_slow": torch.tensor(float(self.core_pareto_lambda_slow), device=device, dtype=dtype),
                "lambda_safety": torch.tensor(float(self.core_pareto_lambda_safety), device=device, dtype=dtype),
                "slow_rate_ema": torch.tensor(float(self.core_pareto_slow_rate_ema), device=device, dtype=dtype),
                "unsafe_rate_ema": torch.tensor(float(self.core_pareto_unsafe_rate_ema), device=device, dtype=dtype),
                "ddc_drop_rate_ema": torch.tensor(float(self.core_pareto_ddc_drop_rate_ema), device=device, dtype=dtype),
            }

        with torch.no_grad():
            ema = float(self.core_pareto_dual_ema)
            slow_rate = float(slow_violation_mask.float().mean().detach().cpu())
            unsafe_rate = float(unsafe_mask.float().mean().detach().cpu())
            ddc_drop_rate = float(ddc_drop_mask.float().mean().detach().cpu())
            self.core_pareto_slow_rate_ema = ema * float(self.core_pareto_slow_rate_ema) + (1.0 - ema) * slow_rate
            self.core_pareto_unsafe_rate_ema = ema * float(self.core_pareto_unsafe_rate_ema) + (1.0 - ema) * unsafe_rate
            self.core_pareto_ddc_drop_rate_ema = (
                ema * float(self.core_pareto_ddc_drop_rate_ema) + (1.0 - ema) * ddc_drop_rate
            )
            dual_lr = float(self.core_pareto_dual_lr)
            self.core_pareto_lambda_slow = min(
                float(self.core_pareto_lambda_slow_max),
                max(
                    float(self.core_pareto_lambda_slow_min),
                    float(self.core_pareto_lambda_slow)
                    + dual_lr * (float(self.core_pareto_slow_rate_ema) - float(self.core_pareto_target_slow_rate)),
                ),
            )
            self.core_pareto_lambda_safety = min(
                float(self.core_pareto_lambda_safety_max),
                max(
                    float(self.core_pareto_lambda_safety_min),
                    float(self.core_pareto_lambda_safety)
                    + dual_lr * (float(self.core_pareto_unsafe_rate_ema) - float(self.core_pareto_target_unsafe_rate)),
                ),
            )
        return {
            "lambda_slow": torch.tensor(float(self.core_pareto_lambda_slow), device=device, dtype=dtype),
            "lambda_safety": torch.tensor(float(self.core_pareto_lambda_safety), device=device, dtype=dtype),
            "slow_rate_ema": torch.tensor(float(self.core_pareto_slow_rate_ema), device=device, dtype=dtype),
            "unsafe_rate_ema": torch.tensor(float(self.core_pareto_unsafe_rate_ema), device=device, dtype=dtype),
            "ddc_drop_rate_ema": torch.tensor(float(self.core_pareto_ddc_drop_rate_ema), device=device, dtype=dtype),
        }

    def _compute_core_pareto_advantages(
        self,
        rewards_matrix: torch.Tensor,
        components_matrix: Dict[str, torch.Tensor],
        trajs_matrix: torch.Tensor,
        ref: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        B, G = rewards_matrix.shape
        pdms = components_matrix.get("pdms", rewards_matrix).float()
        ep = components_matrix["ego_progress"].float()
        ttc = components_matrix["time_to_collision_within_bound"].float()
        comfort = components_matrix["history_comfort"].float()
        nc = components_matrix["no_at_fault_collisions"].float()
        dac = components_matrix["drivable_area_compliance"].float()
        ddc = components_matrix["driving_direction_compliance"].float()
        core = self._compute_pdms_core(components_matrix).float()

        ref_pdms = ref["ref_pdms"].to(device=rewards_matrix.device, dtype=torch.float32)
        ref_core = ref["ref_core"].to(device=rewards_matrix.device, dtype=torch.float32)
        ref_ep = ref["ref_ep"].to(device=rewards_matrix.device, dtype=torch.float32)
        ref_ttc = ref["ref_ttc"].to(device=rewards_matrix.device, dtype=torch.float32)
        ref_ddc = ref["ref_ddc"].to(device=rewards_matrix.device, dtype=torch.float32)

        delta_pdms = pdms - ref_pdms[:, None]
        delta_core = core - ref_core[:, None]
        delta_ep = ep - ref_ep[:, None]
        delta_ttc = ttc - ref_ttc[:, None]
        delta_ddc = ddc - ref_ddc[:, None]

        nc_ok = nc >= 1.0 if bool(self.core_pareto_require_nc) else torch.ones_like(nc, dtype=torch.bool)
        dac_ok = dac >= 1.0 if bool(self.core_pareto_require_dac) else torch.ones_like(dac, dtype=torch.bool)
        if bool(self.core_pareto_use_ddc_guard):
            ddc_ok = (
                (ddc >= float(self.core_pareto_ddc_min_absolute))
                | (ddc >= ref_ddc[:, None] - float(self.core_pareto_ddc_drop_tolerance))
            )
        else:
            ddc_ok = torch.ones_like(ddc, dtype=torch.bool)
        valid = nc_ok & dac_ok & ddc_ok

        if bool(self.core_pareto_use_ep_floor):
            ep_floor_ok = ep >= ref_ep[:, None] - float(self.core_pareto_ep_floor_tolerance)
        else:
            ep_floor_ok = torch.ones_like(ep, dtype=torch.bool)
        slow_violation = (ref_ep[:, None] - ep - float(self.core_pareto_ep_floor_tolerance)).clamp(min=0.0)

        tradeoff = delta_ep + delta_ttc
        if bool(self.core_pareto_use_ttc_tradeoff_penalty):
            tradeoff_bad = (-tradeoff - float(self.core_pareto_tradeoff_tolerance)).clamp(min=0.0)
        else:
            tradeoff_bad = torch.zeros_like(ep)
        if bool(self.core_pareto_use_ttc_floor_penalty):
            ttc_violation = (ref_ttc[:, None] - ttc - float(self.core_pareto_ttc_floor_tolerance)).clamp(min=0.0)
        else:
            ttc_violation = torch.zeros_like(ep)

        dual_logs = self._update_core_pareto_dual(
            slow_violation > 0.0,
            ~valid,
            delta_ddc < -float(self.core_pareto_ddc_drop_tolerance),
        )
        lambda_slow = float(dual_logs["lambda_slow"].detach().cpu())
        score_mode = str(self.core_pareto_score_mode)
        if score_mode == "pdms_minus_slow":
            score = pdms - lambda_slow * slow_violation - float(self.core_pareto_tradeoff_penalty_weight) * tradeoff_bad
        elif score_mode == "pdms_plus_core_margin_minus_slow":
            score = (
                pdms
                + float(self.core_pareto_core_margin_weight) * delta_core
                - lambda_slow * slow_violation
                - float(self.core_pareto_tradeoff_penalty_weight) * tradeoff_bad
                - float(self.core_pareto_ttc_floor_penalty_weight) * ttc_violation
            )
        elif score_mode == "core_minus_slow":
            score = (
                core
                - lambda_slow * slow_violation
                - float(self.core_pareto_tradeoff_penalty_weight) * tradeoff_bad
                - float(self.core_pareto_ttc_floor_penalty_weight) * ttc_violation
            )
        else:
            raise ValueError(f"Unsupported core_pareto_score_mode: {score_mode!r}")

        pareto_front = self._compute_pareto_front_mask(
            components_matrix,
            valid & ep_floor_ok,
            tuple(self.core_pareto_pareto_objectives),
        )
        if bool(self.core_pareto_use_pareto_front):
            score = score + float(self.core_pareto_pareto_front_bonus) * pareto_front.float()

        valid_count = valid.sum(dim=1)
        valid_progress = valid & ep_floor_ok
        valid_progress_count = valid_progress.sum(dim=1)
        all_valid = valid_count == G
        mixed = (valid_count > 0) & (valid_count < G)
        all_invalid = valid_count == 0
        all_slow = (valid_count > 0) & (valid_progress_count == 0)

        all_mask = torch.ones_like(valid, dtype=torch.bool)
        std_floor = float(self.core_pareto_min_group_reward_std)
        if bool(self.core_pareto_use_phenotype_bucket_grpo):
            bucket_id = self._compute_phenotype_buckets(trajs_matrix, components_matrix, ref_ep)
            z = torch.zeros_like(score)
            bucket_count_values = []
            for b in range(B):
                unique_buckets = torch.unique(bucket_id[b])
                bucket_count_values.append(float(unique_buckets.numel()))
                bucket_rep_scores = []
                bucket_values = []
                for bucket in unique_buckets:
                    bucket_mask = (bucket_id[b : b + 1] == bucket)
                    bucket_z = self._masked_zscore(score[b : b + 1], bucket_mask, std_floor=std_floor)
                    z[b : b + 1] = torch.where(bucket_mask, bucket_z, z[b : b + 1])
                    rep_mask = bucket_mask & valid_progress[b : b + 1]
                    if bool(rep_mask.any().item()):
                        bucket_rep_scores.append(score[b : b + 1][rep_mask].mean())
                    else:
                        bucket_rep_scores.append(score[b : b + 1][bucket_mask].mean())
                    bucket_values.append(int(bucket.item()))
                if bucket_rep_scores:
                    reps = torch.stack(bucket_rep_scores).view(1, -1)
                    rep_z = self._masked_zscore(reps, torch.ones_like(reps, dtype=torch.bool), std_floor=std_floor)
                    rep_z = rep_z.clamp(
                        min=-float(self.core_pareto_inter_bucket_clip),
                        max=float(self.core_pareto_inter_bucket_clip),
                    )
                    for idx, bucket_value in enumerate(bucket_values):
                        mask_b = bucket_id[b] == bucket_value
                        z[b, mask_b] = (
                            float(self.core_pareto_intra_bucket_weight) * z[b, mask_b]
                            + float(self.core_pareto_inter_bucket_weight) * rep_z[0, idx].to(z)
                        )
            phenotype_bucket_count_mean = score.new_tensor(
                float(sum(bucket_count_values) / max(len(bucket_count_values), 1))
            )
        else:
            score_z = self._masked_zscore(score, all_mask, std_floor=std_floor)
            all_valid_objective = str(self.core_pareto_all_valid_objective)
            if all_valid_objective == "core":
                all_valid_values = core
            elif all_valid_objective == "pdms":
                all_valid_values = pdms
            else:
                all_valid_values = score
            all_valid_z = self._masked_zscore(all_valid_values, all_mask, std_floor=std_floor)
            z = torch.where(all_valid[:, None], all_valid_z, score_z)
            phenotype_bucket_count_mean = score.new_zeros(())

        if score_mode == "core_minus_slow":
            ref_score = ref_core
        else:
            ref_score = ref_pdms
        if bool(self.core_pareto_use_reference_margin):
            reference_margin = (
                (score - ref_score[:, None]) / float(self.core_pareto_reference_margin_scale)
            ).clamp(
                min=-float(self.core_pareto_reference_margin_clip),
                max=float(self.core_pareto_reference_margin_clip),
            )
            reference_margin = float(self.core_pareto_reference_margin_weight) * reference_margin
        else:
            reference_margin = torch.zeros_like(score)

        rescue_z = self._masked_zscore(score, all_mask, std_floor=std_floor)
        all_invalid_adv = (
            -float(self.core_pareto_all_unsafe_base_offset)
            + float(self.core_pareto_all_unsafe_rescue_weight) * rescue_z
        ).clamp(
            min=float(self.core_pareto_all_unsafe_adv_min),
            max=float(self.core_pareto_all_unsafe_adv_max),
        )
        unsafe_adv = -float(self.core_pareto_unsafe_advantage_offset) + torch.clamp(z, max=0.0)
        slow_adv = torch.minimum(
            torch.clamp(z, max=0.0),
            score.new_full(score.shape, float(self.core_pareto_slow_invalid_advantage)),
        )
        positive_adv = z + reference_margin
        adv = torch.where(valid_progress, positive_adv, torch.where(valid, slow_adv, unsafe_adv))
        if bool(self.core_pareto_use_all_unsafe_rescue_advantage):
            adv = torch.where(all_invalid[:, None], all_invalid_adv, adv)
        else:
            adv = torch.where(all_invalid[:, None], unsafe_adv, adv)

        if bool(self.core_pareto_use_pareto_front):
            dominated_valid = valid_progress & (~pareto_front)
            adv = torch.where(
                dominated_valid & (adv > float(self.core_pareto_dominated_positive_adv_cap)),
                score.new_full(score.shape, float(self.core_pareto_dominated_positive_adv_cap)),
                adv,
            )
        slow_fail = valid & (~ep_floor_ok)
        adv = torch.where(
            slow_fail & (adv > float(self.core_pareto_positive_slow_fail_cap)),
            score.new_full(score.shape, float(self.core_pareto_positive_slow_fail_cap)),
            adv,
        )

        advantage_clip = float(self.core_pareto_advantage_clip_abs)
        if advantage_clip > 0.0:
            adv = adv.clamp(min=-advantage_clip, max=advantage_clip)

        group_weight = torch.ones(B, device=rewards_matrix.device, dtype=rewards_matrix.dtype)
        group_weight[all_invalid] = float(getattr(self, "all_unsafe_group_weight", 0.25))
        group_weight[all_slow] = float(self.core_pareto_all_slow_group_weight)
        low_std_values = core if str(self.core_pareto_all_valid_objective) == "core" else score
        low_std = low_std_values.std(dim=1, unbiased=False) < float(self.core_pareto_min_group_reward_std)
        group_weight[all_valid & low_std] = float(self.core_pareto_all_safe_low_std_group_weight)

        positive_advantage = adv > 0.0
        aux = {
            "reward_std": score.std(dim=1, unbiased=False).detach().to(dtype=rewards_matrix.dtype),
            "safe_count": valid_count.detach().to(dtype=rewards_matrix.dtype),
            "mixed_group_ratio": mixed.float().mean().to(dtype=rewards_matrix.dtype),
            "all_safe_group_ratio": all_valid.float().mean().to(dtype=rewards_matrix.dtype),
            "all_unsafe_group_ratio": all_invalid.float().mean().to(dtype=rewards_matrix.dtype),
            "core_pareto_score": score.detach(),
            "core_pareto_valid_mask": valid.detach(),
            "core_pareto_ep_floor_ok_mask": ep_floor_ok.detach(),
            "core_pareto_pareto_front_mask": pareto_front.detach(),
            "core_mean": core.mean().detach().to(dtype=rewards_matrix.dtype),
            "core_ref_mean": ref_core.mean().detach().to(dtype=rewards_matrix.dtype),
            "delta_core_mean": delta_core.mean().detach().to(dtype=rewards_matrix.dtype),
            "delta_pdms_mean": delta_pdms.mean().detach().to(dtype=rewards_matrix.dtype),
            "delta_ep_mean": delta_ep.mean().detach().to(dtype=rewards_matrix.dtype),
            "delta_ttc_mean": delta_ttc.mean().detach().to(dtype=rewards_matrix.dtype),
            "delta_ddc_mean": delta_ddc.mean().detach().to(dtype=rewards_matrix.dtype),
            "ep_floor_pass_ratio": ep_floor_ok.float().mean().detach().to(dtype=rewards_matrix.dtype),
            "ddc_guard_pass_ratio": ddc_ok.float().mean().detach().to(dtype=rewards_matrix.dtype),
            "nc_pass_ratio": nc_ok.float().mean().detach().to(dtype=rewards_matrix.dtype),
            "dac_pass_ratio": dac_ok.float().mean().detach().to(dtype=rewards_matrix.dtype),
            "valid_ratio": valid.float().mean().detach().to(dtype=rewards_matrix.dtype),
            "valid_progress_ratio": valid_progress.float().mean().detach().to(dtype=rewards_matrix.dtype),
            "pareto_front_ratio": pareto_front.float().mean().detach().to(dtype=rewards_matrix.dtype),
            "dominated_ratio": ((valid_progress & (~pareto_front)).float().mean()).detach().to(dtype=rewards_matrix.dtype),
            "positive_advantage_ratio": positive_advantage.float().mean().detach().to(dtype=rewards_matrix.dtype),
            "positive_advantage_slow_fail_ratio": (
                (positive_advantage & slow_fail).float().sum() / slow_fail.float().sum().clamp(min=1.0)
            ).detach().to(dtype=rewards_matrix.dtype),
            "pareto_positive_advantage_ratio": (
                (positive_advantage & pareto_front).float().sum() / pareto_front.float().sum().clamp(min=1.0)
            ).detach().to(dtype=rewards_matrix.dtype),
            "all_valid_group_ratio": all_valid.float().mean().detach().to(dtype=rewards_matrix.dtype),
            "all_invalid_group_ratio": all_invalid.float().mean().detach().to(dtype=rewards_matrix.dtype),
            "all_slow_group_ratio": all_slow.float().mean().detach().to(dtype=rewards_matrix.dtype),
            "effective_group_weight_mean": group_weight.mean().detach().to(dtype=rewards_matrix.dtype),
            "score_mean": score.mean().detach().to(dtype=rewards_matrix.dtype),
            "score_std": score.std(unbiased=False).detach().to(dtype=rewards_matrix.dtype),
            "slow_violation_mean": slow_violation.mean().detach().to(dtype=rewards_matrix.dtype),
            "tradeoff_bad_mean": tradeoff_bad.mean().detach().to(dtype=rewards_matrix.dtype),
            "ttc_violation_mean": ttc_violation.mean().detach().to(dtype=rewards_matrix.dtype),
            "phenotype_bucket_count_mean": phenotype_bucket_count_mean.detach().to(dtype=rewards_matrix.dtype),
            "lambda_slow": dual_logs["lambda_slow"].detach().to(dtype=rewards_matrix.dtype),
            "lambda_safety": dual_logs["lambda_safety"].detach().to(dtype=rewards_matrix.dtype),
            "slow_rate_ema": dual_logs["slow_rate_ema"].detach().to(dtype=rewards_matrix.dtype),
            "unsafe_rate_ema": dual_logs["unsafe_rate_ema"].detach().to(dtype=rewards_matrix.dtype),
            "ddc_drop_rate_ema": dual_logs["ddc_drop_rate_ema"].detach().to(dtype=rewards_matrix.dtype),
            "ref_gt_pdms": ref["gt_pdms"].mean().detach().to(device=rewards_matrix.device, dtype=rewards_matrix.dtype),
            "ref_il_pdms": ref["il_pdms"].mean().detach().to(device=rewards_matrix.device, dtype=rewards_matrix.dtype),
            "ref_gt_core": ref["gt_core"].mean().detach().to(device=rewards_matrix.device, dtype=rewards_matrix.dtype),
            "ref_il_core": ref["il_core"].mean().detach().to(device=rewards_matrix.device, dtype=rewards_matrix.dtype),
        }
        return adv.reshape(B * G).detach(), group_weight.detach(), aux

    def _compute_feasible_pareto_advantages(
        self,
        rewards_matrix: torch.Tensor,
        components_matrix: Dict[str, torch.Tensor],
        trajs_matrix: torch.Tensor,
        ref: Dict[str, torch.Tensor],
        support_batch: Optional[Dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        B, G = rewards_matrix.shape
        dtype = rewards_matrix.dtype
        device = rewards_matrix.device
        pdms = components_matrix.get("pdms", rewards_matrix).float()
        ep = components_matrix["ego_progress"].float()
        ttc = components_matrix["time_to_collision_within_bound"].float()
        comfort = components_matrix["history_comfort"].float()
        nc = components_matrix["no_at_fault_collisions"].float()
        dac = components_matrix["drivable_area_compliance"].float()
        ddc = components_matrix["driving_direction_compliance"].float()

        flat_trajs = trajs_matrix.reshape(B * G, trajs_matrix.shape[-2], trajs_matrix.shape[-1])
        feas_metrics = compute_feasibility_metrics(
            flat_trajs,
            {
                "w_curv": float(getattr(self.config, "geo_curvature_weight", 1.0)),
                "w_reverse": float(getattr(self.config, "geo_reverse_weight", 1.0)),
                "w_tail_reverse": float(getattr(self.config, "geo_tail_reverse_weight", 2.0)),
                "w_kink": float(getattr(self.config, "geo_early_kink_weight", 2.0)),
                "w_jerk": float(getattr(self.config, "geo_jerk_weight", 0.2)),
            },
        )
        feas_cost = feas_metrics.feas_cost.reshape(B, G).to(device=device, dtype=torch.float32)
        early_kink_rate = feas_metrics.early_kink_rate.reshape(B, G).to(device=device, dtype=torch.float32)
        tail_reverse_rate = feas_metrics.tail_reverse_rate.reshape(B, G).to(device=device, dtype=torch.float32)
        curvature_violation_rate = feas_metrics.curvature_violation_rate.reshape(B, G).to(device=device, dtype=torch.float32)

        ref_pdms = ref["ref_pdms"].to(device=device, dtype=torch.float32)
        ref_ep = ref["ref_ep"].to(device=device, dtype=torch.float32)
        ref_ttc = ref["ref_ttc"].to(device=device, dtype=torch.float32)
        ref_ddc = ref["ref_ddc"].to(device=device, dtype=torch.float32)
        ref_utility = (
            float(self.fp_ep_weight) * ref_ep
            + float(self.fp_ttc_weight) * ref_ttc
            + float(self.fp_ddc_weight) * ref_ddc
        )

        if float(self.fp_feas_max) > 0.0:
            feas_threshold = torch.full((B,), float(self.fp_feas_max), device=device, dtype=torch.float32)
        else:
            sorted_feas = torch.sort(feas_cost.detach(), dim=1).values
            q_idx = min(max(int(round(0.9 * max(G - 1, 0))), 0), max(G - 1, 0))
            feas_threshold = sorted_feas[:, q_idx].clamp_min(1e-6)

        ddc_guard = ddc >= torch.maximum(
            ddc.new_full((B, G), float(self.fp_ddc_min_absolute)),
            ref_ddc[:, None] - float(self.fp_ddc_ref_tolerance),
        )
        nc_ok = nc >= 1.0
        dac_ok = dac >= 1.0
        comfort_ok = comfort >= float(self.fp_comfort_min)
        geometry_ok = feas_cost <= feas_threshold[:, None]
        valid = nc_ok & dac_ok & ddc_guard & geometry_ok & comfort_ok

        utility = (
            float(self.fp_ep_weight) * ep
            + float(self.fp_ttc_weight) * ttc
            + float(self.fp_ddc_weight) * ddc
            - float(self.fp_feas_weight) * feas_cost
        )
        tradeoff_bad = (
            -((ep - ref_ep[:, None]) + float(self.fp_tradeoff_ttc_rho) * (ttc - ref_ttc[:, None]))
            - float(self.fp_tradeoff_tolerance)
        ).clamp(min=0.0)
        objective_components = {
            "ego_progress": ep,
            "time_to_collision_within_bound": ttc,
            "driving_direction_compliance": ddc,
            "_neg_feas_cost": -feas_cost,
        }
        pareto_front = self._compute_pareto_front_mask(
            objective_components,
            valid,
            ("ego_progress", "time_to_collision_within_bound", "driving_direction_compliance", "_neg_feas_cost"),
        )
        score = utility + float(self.fp_pareto_front_bonus) * pareto_front.float()
        score = score - float(self.fp_tradeoff_penalty_weight) * tradeoff_bad

        all_mask = torch.ones_like(valid, dtype=torch.bool)
        pdas_metrics = compute_pdas_metrics(
            score.detach(),
            {"ego_progress": ep.detach(), "time_to_collision_within_bound": ttc.detach(), "driving_direction_compliance": ddc.detach()},
            trajs_matrix.detach(),
            {"ref_ep": ref_ep.detach(), "ref_pdms": ref_pdms.detach()},
            self,
        )
        z = self._masked_zscore(score, all_mask)
        bucket_count_mean = score.new_zeros(())
        bucketed_active_ratio = score.new_zeros(())
        bucket_fallback_ratio = score.new_zeros(())
        sampled_buckets = None
        if bool(self.fp_use_bucketed_advantage) or isinstance(support_batch, dict):
            from .pdas import compute_phenotype_buckets

            sampled_buckets = compute_phenotype_buckets(
                trajs_matrix.detach(),
                {
                    "ego_progress": ep.detach(),
                    "time_to_collision_within_bound": ttc.detach(),
                    "driving_direction_compliance": ddc.detach(),
                },
                {"feas_cost": feas_cost.detach()},
                {"ref_ep": ref_ep.detach()},
                self,
            )
        if bool(self.fp_use_bucketed_advantage) and sampled_buckets is not None:
            support_counts = None
            if isinstance(support_batch, dict) and isinstance(support_batch.get("selected_real_mask"), torch.Tensor):
                support_counts = support_batch["selected_real_mask"].to(device=device).bool().sum(dim=1)
            bucket_counts = []
            active_rows = 0
            for b in range(B):
                unique = torch.unique(sampled_buckets[b])
                bucket_counts.append(float(unique.numel()))
                support_ok = True
                if support_counts is not None and int(support_counts.shape[0]) == B:
                    support_ok = bool((support_counts[b] >= int(self.pdas_min_support_count_for_coverage)).item())
                if G < 4 or unique.numel() < 2 or not support_ok:
                    continue
                active_rows += 1
                rep_scores = []
                bucket_values = []
                for bucket in unique:
                    bucket_mask = sampled_buckets[b : b + 1] == bucket
                    z[b : b + 1] = torch.where(
                        bucket_mask,
                        self._masked_zscore(score[b : b + 1], bucket_mask),
                        z[b : b + 1],
                    )
                    rep_mask = bucket_mask & valid[b : b + 1]
                    rep_scores.append(score[b : b + 1][rep_mask].mean() if bool(rep_mask.any().item()) else score[b : b + 1][bucket_mask].mean())
                    bucket_values.append(int(bucket.item()))
                if rep_scores:
                    reps = torch.stack(rep_scores).view(1, -1)
                    rep_z = self._masked_zscore(reps, torch.ones_like(reps, dtype=torch.bool)).clamp(
                        min=-float(self.fp_inter_bucket_clip),
                        max=float(self.fp_inter_bucket_clip),
                    )
                    for idx, bucket_value in enumerate(bucket_values):
                        mask_b = sampled_buckets[b] == bucket_value
                        z[b, mask_b] = z[b, mask_b] + float(self.fp_inter_bucket_weight) * rep_z[0, idx].to(z)
            bucket_count_mean = score.new_tensor(float(sum(bucket_counts) / max(len(bucket_counts), 1)))
            bucketed_active_ratio = score.new_tensor(float(active_rows) / max(float(B), 1.0))
            bucket_fallback_ratio = score.new_tensor(float(B - active_rows) / max(float(B), 1.0))

        unsafe_adv = score.new_full(score.shape, float(self.fp_invalid_negative_advantage))
        adv = torch.where(valid, z, unsafe_adv)
        regression_zero = (ref_pdms[:, None] > 0.0) & (pdms <= 0.0)
        ddc_regression = ~ddc_guard
        geometry_bad = ~geometry_ok
        dominated = valid & (~pareto_front)
        offsupport = (~pareto_front) & (utility <= ref_utility[:, None] + float(self.fp_tradeoff_tolerance))
        cap_mask = torch.zeros_like(valid, dtype=torch.bool)
        adv = torch.where(regression_zero, score.new_full(score.shape, float(self.fp_regression_negative_advantage)), adv)
        for mask, cap in (
            (ddc_regression, float(self.fp_ddc_regression_positive_cap)),
            (geometry_bad, float(self.fp_geometry_positive_cap)),
            (dominated, float(self.fp_dominated_positive_cap)),
            (offsupport, float(self.fp_offsupport_positive_cap)),
            (~comfort_ok, 0.0),
        ):
            active = mask & (adv > cap)
            cap_mask = cap_mask | active
            adv = torch.where(active, score.new_full(score.shape, cap), adv)

        group_weight = torch.ones(B, device=device, dtype=dtype)
        coverage_gap = score.new_zeros((B,), dtype=torch.float32)
        archive_bucket_count = score.new_zeros((B,), dtype=torch.float32)
        sampled_bucket_count = score.new_zeros((B,), dtype=torch.float32)
        if sampled_buckets is not None:
            for b in range(B):
                sampled_bucket_count[b] = float(torch.unique(sampled_buckets[b]).numel())
        if isinstance(support_batch, dict) and isinstance(support_batch.get("selected_trajs"), torch.Tensor):
            support_trajs = support_batch["selected_trajs"].to(device=device)
            support_real = support_batch.get("selected_real_mask")
            support_valid = support_batch.get("selected_valid_mask")
            support_components = support_batch.get("selected_components", {})
            if isinstance(support_real, torch.Tensor) and isinstance(support_valid, torch.Tensor) and isinstance(support_components, dict):
                support_real = support_real.to(device=device).bool()
                support_valid = support_valid.to(device=device).bool()
                support_mask = support_real & support_valid
                support_count = support_real.sum(dim=1)
                support_component_tensors = {
                    key: value.to(device=device).detach()
                    for key, value in support_components.items()
                    if isinstance(value, torch.Tensor)
                }
                if support_trajs.ndim == 4 and int(support_trajs.shape[0]) == B:
                    from .pdas import compute_phenotype_buckets

                    archive_buckets = compute_phenotype_buckets(
                        support_trajs.detach(),
                        support_component_tensors,
                        {},
                        {"ref_ep": ref_ep.detach()},
                        self,
                    )
                    for b in range(B):
                        if support_count[b] < int(self.pdas_min_support_count_for_coverage):
                            continue
                        row_mask = support_mask[b]
                        if not bool(row_mask.any().item()):
                            continue
                        archive_count = float(torch.unique(archive_buckets[b][row_mask]).numel())
                        archive_bucket_count[b] = archive_count
                        if sampled_bucket_count[b] <= 0.0:
                            sampled_bucket_count[b] = float(torch.unique(sampled_buckets[b]).numel()) if sampled_buckets is not None else 0.0
                        coverage_gap[b] = max(
                            0.0,
                            min((archive_count - float(sampled_bucket_count[b].item())) / max(archive_count, 1.0), 1.0),
                        )
        if bool(self.fp_use_pdas):
            coverage_factor = 1.0 + float(self.pdas_lambda_coverage) * coverage_gap.to(device=device, dtype=dtype)
            group_weight = group_weight * pdas_metrics.weight.to(device=device, dtype=dtype) * coverage_factor

        aux = {
            "reward_std": score.std(dim=1, unbiased=False).detach().to(dtype=dtype),
            "safe_count": valid.sum(dim=1).detach().to(dtype=dtype),
            "mixed_group_ratio": ((valid.sum(dim=1) > 0) & (valid.sum(dim=1) < G)).float().mean().to(dtype=dtype),
            "all_safe_group_ratio": (valid.sum(dim=1) == G).float().mean().to(dtype=dtype),
            "all_unsafe_group_ratio": (valid.sum(dim=1) == 0).float().mean().to(dtype=dtype),
            "feasible_pareto_enabled": score.new_tensor(1.0).to(dtype=dtype),
            "feasible_pareto_score": score.detach(),
            "feasible_pareto_valid_mask": valid.detach(),
            "feasible_pareto_pareto_front_mask": pareto_front.detach(),
            "fp_utility_mean": utility.mean().detach().to(dtype=dtype),
            "fp_valid_ratio": valid.float().mean().detach().to(dtype=dtype),
            "fp_pareto_front_ratio": pareto_front.float().mean().detach().to(dtype=dtype),
            "fp_dominated_ratio": dominated.float().mean().detach().to(dtype=dtype),
            "fp_ddc_guard_pass_ratio": ddc_guard.float().mean().detach().to(dtype=dtype),
            "fp_ddc_regression_ratio": ddc_regression.float().mean().detach().to(dtype=dtype),
            "fp_feas_cost_mean": feas_cost.mean().detach().to(dtype=dtype),
            "fp_geometry_bad_ratio": geometry_bad.float().mean().detach().to(dtype=dtype),
            "fp_regression_0_from_positive_ratio": regression_zero.float().mean().detach().to(dtype=dtype),
            "fp_positive_adv_cap_ratio": cap_mask.float().mean().detach().to(dtype=dtype),
            "fp_bucket_count_mean": bucket_count_mean.detach().to(dtype=dtype),
            "fp_bucketed_active_ratio": bucketed_active_ratio.detach().to(dtype=dtype),
            "fp_unique_bucket_count_mean": bucket_count_mean.detach().to(dtype=dtype),
            "fp_bucket_fallback_ratio": bucket_fallback_ratio.detach().to(dtype=dtype),
            "fp_intra_adv_mean": z.mean().detach().to(dtype=dtype),
            "fp_inter_adv_mean": score.new_zeros(()).to(dtype=dtype),
            "pdas_success_rate": pdas_metrics.success_rate.mean().detach().to(dtype=dtype),
            "pdas_learnability": pdas_metrics.learnability.mean().detach().to(dtype=dtype),
            "pdas_reward_std": pdas_metrics.reward_std.mean().detach().to(dtype=dtype),
            "pdas_bon_gap": pdas_metrics.bon_gap.mean().detach().to(dtype=dtype),
            "pdas_bucket_diversity": pdas_metrics.bucket_diversity.mean().detach().to(dtype=dtype),
            "pdas_regression_risk": pdas_metrics.regression_risk.mean().detach().to(dtype=dtype),
            "pdas_weight_mean": pdas_metrics.weight.mean().detach().to(dtype=dtype),
            "pdas_weight_min": pdas_metrics.weight.min().detach().to(dtype=dtype),
            "pdas_weight_max": pdas_metrics.weight.max().detach().to(dtype=dtype),
            "pdas_support_coverage_gap": coverage_gap.mean().detach().to(dtype=dtype),
            "pdas_archive_bucket_count": archive_bucket_count.mean().detach().to(dtype=dtype),
            "pdas_sampled_bucket_count": sampled_bucket_count.mean().detach().to(dtype=dtype),
            "early_kink_rate": early_kink_rate.mean().detach().to(dtype=dtype),
            "tail_reverse_rate": tail_reverse_rate.mean().detach().to(dtype=dtype),
            "curvature_violation_rate": curvature_violation_rate.mean().detach().to(dtype=dtype),
        }
        return adv.reshape(B * G).detach(), group_weight.detach(), aux

    def _stage3_core_pareto_log_metrics(
        self,
        ref: torch.Tensor,
        reward_aux: Dict[str, torch.Tensor],
        advantage_aux: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        zero = ref.new_zeros(())

        def reward_mean(name: str) -> torch.Tensor:
            value = reward_aux.get(name)
            if isinstance(value, torch.Tensor):
                return value.detach().float().mean().to(device=ref.device, dtype=ref.dtype)
            return zero

        def aux_value(name: str) -> torch.Tensor:
            value = advantage_aux.get(name)
            if isinstance(value, torch.Tensor):
                return value.detach().float().mean().to(device=ref.device, dtype=ref.dtype)
            return zero

        return {
            "core_pareto_enabled": reward_mean("core_pareto_mode_enabled"),
            "pdms_core": aux_value("core_mean"),
            "pdms_core_ref": aux_value("core_ref_mean"),
            "delta_core_vs_ref": aux_value("delta_core_mean"),
            "delta_pdms_vs_ref": aux_value("delta_pdms_mean"),
            "delta_ep_vs_ref": aux_value("delta_ep_mean"),
            "delta_ttc_vs_ref": aux_value("delta_ttc_mean"),
            "delta_ddc_vs_ref": aux_value("delta_ddc_mean"),
            "ep_floor_pass_ratio": aux_value("ep_floor_pass_ratio"),
            "ddc_guard_pass_ratio": aux_value("ddc_guard_pass_ratio"),
            "core_pareto_valid_ratio": aux_value("valid_ratio"),
            "core_pareto_valid_progress_ratio": aux_value("valid_progress_ratio"),
            "pareto_front_ratio": aux_value("pareto_front_ratio"),
            "dominated_ratio": aux_value("dominated_ratio"),
            "positive_advantage_slow_fail_ratio": aux_value("positive_advantage_slow_fail_ratio"),
            "pareto_positive_advantage_ratio": aux_value("pareto_positive_advantage_ratio"),
            "core_pareto_mixed_group_ratio": aux_value("mixed_group_ratio"),
            "core_pareto_all_valid_group_ratio": aux_value("all_valid_group_ratio"),
            "core_pareto_all_invalid_group_ratio": aux_value("all_invalid_group_ratio"),
            "core_pareto_all_slow_group_ratio": aux_value("all_slow_group_ratio"),
            "core_pareto_effective_group_weight_mean": aux_value("effective_group_weight_mean"),
            "core_pareto_score_mean": aux_value("score_mean"),
            "core_pareto_score_std": aux_value("score_std"),
            "core_pareto_slow_violation_mean": aux_value("slow_violation_mean"),
            "core_pareto_tradeoff_bad_mean": aux_value("tradeoff_bad_mean"),
            "core_pareto_ttc_violation_mean": aux_value("ttc_violation_mean"),
            "core_pareto_lambda_slow": aux_value("lambda_slow"),
            "core_pareto_lambda_safety": aux_value("lambda_safety"),
            "core_pareto_slow_rate_ema": aux_value("slow_rate_ema"),
            "core_pareto_unsafe_rate_ema": aux_value("unsafe_rate_ema"),
            "core_pareto_ddc_drop_rate_ema": aux_value("ddc_drop_rate_ema"),
            "core_pareto_ref_gt_pdms": aux_value("ref_gt_pdms"),
            "core_pareto_ref_il_pdms": aux_value("ref_il_pdms"),
            "core_pareto_ref_gt_core": aux_value("ref_gt_core"),
            "core_pareto_ref_il_core": aux_value("ref_il_core"),
            "phenotype_bucket_count_mean": aux_value("phenotype_bucket_count_mean"),
            "buffer_bonus_mean": aux_value("buffer_bonus_mean"),
            "buffer_bonus_max": aux_value("buffer_bonus_max"),
            "buffer_bonus_target_ratio": aux_value("buffer_bonus_target_ratio"),
            "buffer_bonus_distance_mean": aux_value("buffer_bonus_distance_mean"),
            "core_pareto_mode_enabled": reward_mean("core_pareto_mode_enabled"),
            "core_pareto_core_mean": reward_mean("core_pareto_core"),
            "core_pareto_adjusted_core_mean": reward_mean("core_pareto_adjusted_core"),
            "core_pareto_pdms_formula_mean": reward_mean("core_pareto_pdms_formula"),
            "core_pareto_ep_floor_gap_mean": reward_mean("core_pareto_ep_floor_gap"),
            "core_pareto_ep_floor_penalty_mean": reward_mean("core_pareto_ep_floor_penalty"),
            "core_pareto_ddc_penalty_mean": reward_mean("core_pareto_ddc_penalty"),
            "core_pareto_dual_penalty_mean": reward_mean("core_pareto_dual_penalty"),
            "core_pareto_nc_dac_feasible_ratio": aux_value("core_pareto_nc_dac_feasible_ratio"),
            "core_pareto_ddc_guard_pass_ratio": (
                aux_value("core_pareto_ddc_guard_pass_ratio") + aux_value("ddc_guard_pass_ratio")
            ),
            "core_pareto_front_ratio": aux_value("core_pareto_front_ratio") + aux_value("pareto_front_ratio"),
            "core_pareto_valid_front_ratio": aux_value("core_pareto_valid_front_ratio"),
            "core_pareto_ep_low_ratio": aux_value("core_pareto_ep_low_ratio"),
            "core_pareto_progress_bucket_ratio": aux_value("core_pareto_progress_bucket_ratio"),
            "core_pareto_ttc_bucket_ratio": aux_value("core_pareto_ttc_bucket_ratio"),
            "core_pareto_balanced_bucket_ratio": aux_value("core_pareto_balanced_bucket_ratio"),
            "core_pareto_core_std": aux_value("core_pareto_core_std"),
            "core_pareto_ep_floor": ref.new_tensor(float(self.core_pareto_ep_floor)),
            "core_pareto_ddc_guard_threshold": ref.new_tensor(float(self.core_pareto_ddc_guard_threshold)),
            "core_pareto_pareto_bonus": ref.new_tensor(float(self.core_pareto_pareto_bonus)),
        }

    def _stage3_feasible_pareto_log_metrics(
        self,
        ref: torch.Tensor,
        advantage_aux: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        zero = ref.new_zeros(())

        def aux_value(name: str) -> torch.Tensor:
            value = advantage_aux.get(name)
            if isinstance(value, torch.Tensor):
                return value.detach().float().mean().to(device=ref.device, dtype=ref.dtype)
            return zero

        keys = (
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
            "fp_bucketed_active_ratio",
            "fp_unique_bucket_count_mean",
            "fp_bucket_fallback_ratio",
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
            "pdas_support_coverage_gap",
            "pdas_archive_bucket_count",
            "pdas_sampled_bucket_count",
            "early_kink_rate",
            "tail_reverse_rate",
            "curvature_violation_rate",
        )
        return {key: aux_value(key) for key in keys}

    def _chain_step_logprobs(
        self,
        log_probs: torch.Tensor,
        B: int,
        G: int,
        K: int,
    ) -> torch.Tensor:
        # raw log_probs: [B * G * K, H, D]
        assert log_probs.shape[0] == B * G * K
        step_logp = log_probs.clamp(min=-5, max=2).mean(dim=[1, 2])
        return step_logp.view(B * G, K)

    def _reduce_chain_logprobs(
        self,
        log_probs: torch.Tensor,
        B: int,
        G: int,
        K: int,
        discount: torch.Tensor,
    ) -> torch.Tensor:
        step_logp = self._chain_step_logprobs(log_probs, B, G, K)

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

    @staticmethod
    def _detach_action_input(action_input: Optional[BatchFeature]) -> Optional[BatchFeature]:
        if action_input is None:
            return None
        data: Dict[str, Any] = {}
        for key, value in action_input.items():
            data[key] = value.detach() if isinstance(value, torch.Tensor) else value
        return BatchFeature(data=data)

    def _stage3_discount(self, num_denoising_steps: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        denoising_indices = torch.arange(num_denoising_steps, device=device)
        return (float(self.gamma_denoising) ** (num_denoising_steps - denoising_indices - 1)).to(
            device=device,
            dtype=dtype,
        )

    def _stage3_reward_kwargs(self) -> Dict[str, Any]:
        offline_cfg = getattr(self, "offline_rl_cfg", None)
        if offline_cfg is None or not bool(offline_cfg.enabled):
            return {}
        return {
            "strict_submetrics": bool(offline_cfg.strict_reward_submetrics),
            "required_submetrics": offline_cfg.required_reward_submetrics,
            "missing_submetric_policy": str(offline_cfg.missing_submetric_policy),
            "use_batched_pdm_scoring": bool(offline_cfg.use_batched_pdm_scoring),
            "use_exact_array_pdm_state_conversion": bool(offline_cfg.use_exact_array_pdm_state_conversion),
            "use_fast_pdm_scorer": bool(offline_cfg.use_fast_pdm_scorer),
            "pdm_batch_chunk_size": int(offline_cfg.pdm_batch_chunk_size),
            "pdm_shadow_check": bool(offline_cfg.pdm_shadow_check),
            "pdm_shadow_max_samples": int(offline_cfg.pdm_shadow_max_samples),
            "pdm_shadow_max_abs_diff": float(offline_cfg.pdm_shadow_max_abs_diff),
        }

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
        end: Optional[float] = None,
        decay_start_step: int = -1,
        decay_end_step: int = -1,
    ) -> float:
        if target <= 0.0:
            return 0.0
        schedule = str(schedule)
        if schedule == "constant":
            return target

        current_epoch = max(0, int(getattr(self.config, "current_train_epoch", 0)))
        if current_epoch < start_epoch:
            return float(start)
        warmup_epochs = max(1, int(warmup_epochs))
        progress = min(float(current_epoch - start_epoch + 1) / float(warmup_epochs), 1.0)
        weight = start + (target - start) * progress
        if schedule == "linear_warmup":
            return float(weight)

        if schedule not in {"linear_warmup_linear_decay", "linear_warmup_cosine_decay"}:
            raise ValueError(f"Unsupported loss schedule: {schedule!r}")
        decay_start_step = int(decay_start_step)
        decay_end_step = int(decay_end_step)
        if decay_start_step < 0 or decay_end_step <= decay_start_step:
            return float(weight)
        current_step = max(0, int(getattr(self.config, "current_train_step", 0)))
        if current_step < decay_start_step:
            return float(weight)
        end_weight = float(target if end is None else end)
        if current_step >= decay_end_step:
            return end_weight
        decay_progress = float(current_step - decay_start_step) / float(decay_end_step - decay_start_step)
        decay_progress = min(max(decay_progress, 0.0), 1.0)
        if schedule == "linear_warmup_cosine_decay":
            decay_progress = 0.5 - 0.5 * math.cos(math.pi * decay_progress)
        return float(weight + (end_weight - weight) * decay_progress)

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

    def _current_grpo_buffer_distill_loss_weight(self, cfg: OfflineRLConfig) -> float:
        return self._current_linear_warmup_loss_weight(
            target=float(cfg.grpo_buffer_distill_loss_weight),
            schedule=str(cfg.grpo_buffer_distill_loss_schedule),
            start=float(cfg.grpo_buffer_distill_loss_weight_start),
            start_epoch=int(cfg.grpo_buffer_distill_warmup_start_epoch),
            warmup_epochs=int(cfg.grpo_buffer_distill_warmup_epochs),
            end=float(cfg.grpo_buffer_distill_loss_weight_end),
            decay_start_step=int(cfg.grpo_buffer_distill_decay_start_step),
            decay_end_step=int(cfg.grpo_buffer_distill_decay_end_step),
        )

    def _current_grpo_buffer_preference_dpo_loss_weight(self, cfg: OfflineRLConfig) -> float:
        return self._current_linear_warmup_loss_weight(
            target=float(cfg.grpo_buffer_preference_dpo_loss_weight),
            schedule=str(cfg.grpo_buffer_preference_dpo_loss_schedule),
            start=float(cfg.grpo_buffer_preference_dpo_loss_weight_start),
            start_epoch=int(cfg.grpo_buffer_preference_dpo_warmup_start_epoch),
            warmup_epochs=int(cfg.grpo_buffer_preference_dpo_warmup_epochs),
        )

    def _current_grpo_self_imitation_loss_weight(self, cfg: OfflineRLConfig) -> float:
        return self._current_linear_warmup_loss_weight(
            target=float(cfg.grpo_self_imitation_loss_weight),
            schedule=str(cfg.grpo_self_imitation_loss_schedule),
            start=float(cfg.grpo_self_imitation_loss_weight_start),
            start_epoch=int(cfg.grpo_self_imitation_warmup_start_epoch),
            warmup_epochs=int(cfg.grpo_self_imitation_warmup_epochs),
            end=float(cfg.grpo_self_imitation_loss_weight_end),
            decay_start_step=int(cfg.grpo_self_imitation_decay_start_step),
            decay_end_step=int(cfg.grpo_self_imitation_decay_end_step),
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
        source = str(source or "").lower()
        if source == "gt" or source.startswith("gt:"):
            return "gt"
        if source == "il" or source.startswith("il:") or source == "recogdrive_stage3" or source.startswith("recogdrive_stage3:"):
            return "il"
        if source.startswith("failure_expand"):
            return "failure_expand"
        if source.startswith("trust_region"):
            return "trust_region"
        if (
            source.startswith("external")
            or source in {"ddv2", "driveor", "drivor", "diffusiondrivev2", "diffusion_drive_v2"}
            or source.startswith(("ddv2:", "driveor:", "drivor:", "diffusiondrivev2:", "diffusion_drive_v2:"))
        ):
            return "external"
        if source.startswith("structured"):
            return "structured"
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
            "external": 7,
            "failure_expand": 8,
            "trust_region": 9,
            "structured": 10,
        }.get(cls._source_bucket(source), 0)

    @staticmethod
    def _dpsi_weight_for_tag(tag: str, source: str, cfg: OfflineRLConfig) -> float:
        tag_l = str(tag or "").lower()
        source_l = str(source or "").lower()
        if tag_l == "":
            return 0.0 if bool(cfg.dpsi_empty_tag_zero) else float(cfg.dpsi_weight_unknown)
        if "safe_ep" in tag_l:
            return float(cfg.dpsi_weight_safe_ep)
        if "vector_pareto" in tag_l or "pareto" in tag_l:
            return float(cfg.dpsi_weight_vector_pareto)
        if "safety_repair" in tag_l:
            return float(cfg.dpsi_weight_safety_repair)
        if "ddc_repair" in tag_l:
            return float(cfg.dpsi_weight_ddc_repair)
        if "best_pdms" in tag_l:
            return float(cfg.dpsi_weight_best_pdms)
        if "diversity" in tag_l:
            return float(cfg.dpsi_weight_diversity)
        if "fallback" in tag_l:
            return float(cfg.dpsi_weight_fallback)
        if "smooth" in tag_l:
            return float(cfg.dpsi_weight_smooth)
        if "il" in tag_l or _dpsi_source_matches(source_l, ("il", "recogdrive_stage3")):
            return float(cfg.dpsi_weight_il)
        if "gt" in tag_l or _dpsi_source_matches(source_l, ("gt",)):
            return float(cfg.dpsi_weight_gt)
        return float(cfg.dpsi_weight_unknown)

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
        target_source_code: Optional[torch.Tensor] = None,
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
        target_norm = policy._encode_action_target(targets)
        flat_source_code = None
        flat_is_gt = None
        if target_source_code is not None:
            if target_source_code.shape != (B, M):
                raise ValueError(f"target_source_code must have shape [B, M], got {tuple(target_source_code.shape)}.")
            flat_source_code = target_source_code.reshape(B * M).to(device=target_norm.device)
            flat_is_gt = (flat_source_code == 1).to(dtype=target_norm.dtype)
        training_target = policy._build_training_target(
            raw_trajectory=targets,
            selected_repr=target_norm,
            action_input=policy._repeat_action_input_for_loss(action_input, M),
            selected_source_code=flat_source_code,
            selected_target_is_gt=flat_is_gt,
        )
        diffusion_target = training_target.diffusion_target_repr
        if noise is None:
            noise = torch.randn_like(diffusion_target)
        else:
            noise = noise.to(device=diffusion_target.device, dtype=diffusion_target.dtype)
        if t_discrete is None:
            t_discrete = policy._sample_offline_rl_timesteps(
                B * M,
                device=diffusion_target.device,
                dtype=diffusion_target.dtype,
                mode=timestep_sampling,
            )
        else:
            t_discrete = t_discrete.to(device=target_norm.device)
        noisy_actions = (
            policy.extract(policy.ddpm_sqrt_alphas_cumprod, t_discrete, diffusion_target.shape) * diffusion_target
            + policy.extract(policy.ddpm_sqrt_one_minus_alphas_cumprod, t_discrete, diffusion_target.shape) * noise
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
        x0_pred = policy._x0_from_noise(noisy_actions, t_discrete, pred_noise)
        x0_aux_per_sample, geo_aux_per_sample, aux_active_mask, geo_diag = policy._compute_x0_geo_aux_per_sample_losses(
            training_target.reconstruct_full_x0(x0_pred),
            training_target.raw_trajectory,
            t_discrete,
            method="ddpm",
        )
        trajectory_aux_per_sample, feasibility_aux_per_sample, pta_diag = policy._compute_pta_aux_per_sample_losses(
            x0_pred,
            training_target,
            t_discrete,
            method="ddpm",
        )
        tangent_per_sample = pta_diag.pop("_tangent_excess_per_sample", trajectory_aux_per_sample.new_zeros(B * M))
        curvature_per_sample = pta_diag.pop(
            "_curvature_excess_per_sample",
            trajectory_aux_per_sample.new_zeros(B * M),
        )
        alpha_weight_per_sample = pta_diag.pop(
            "_aux_alpha_weight_per_sample",
            trajectory_aux_per_sample.new_zeros(B * M),
        )
        full_x0_l1_per_sample = pta_diag.pop(
            "_full_x0_reconstruction_l1_per_sample",
            trajectory_aux_per_sample.new_zeros(B * M),
        )
        fs_bound_hit_per_sample = pta_diag.pop(
            "_fs_output_bound_hit_per_sample",
            trajectory_aux_per_sample.new_zeros(B * M),
        )
        if target_source_code is None:
            source_code_matrix = torch.ones((B, M), device=target_norm.device, dtype=target_norm.dtype)
            target_is_gt_matrix = torch.ones_like(source_code_matrix)
        else:
            source_code_matrix = target_source_code.to(device=target_norm.device, dtype=target_norm.dtype)
            target_is_gt_matrix = (source_code_matrix == 1).to(dtype=target_norm.dtype)
        target_abs = training_target.selected_repr.detach().float().abs()
        if policy._uses_fs_norm():
            fs_abs_gt3_matrix = (target_abs > 3.0).float().mean(dim=(1, 2)).reshape(B, M)
            fs_abs_gt5_matrix = (target_abs > 5.0).float().mean(dim=(1, 2)).reshape(B, M)
        else:
            fs_abs_gt3_matrix = target_abs.new_zeros((B, M))
            fs_abs_gt5_matrix = target_abs.new_zeros((B, M))
        dit_scalar_diag = {
            key: value.detach()
            for key, value in dit_context.get("diagnostics", {}).items()
            if isinstance(value, torch.Tensor) and value.numel() == 1
        }
        diagnostics = {
            "per_sample_loss_mean": per_sample_loss.detach().mean(),
            "target_norm_mean": target_norm.detach().float().abs().mean(),
            "diffusion_timestep_mean": t_discrete.detach().float().mean(),
            "diffusion_timestep_min": t_discrete.detach().float().min(),
            "diffusion_timestep_max": t_discrete.detach().float().max(),
            "x0_aux_loss_matrix": x0_aux_per_sample.reshape(B, M),
            "delta_aux_loss_matrix": geo_diag.get(
                "delta_aux_per_sample",
                x0_aux_per_sample.new_zeros(x0_aux_per_sample.shape),
            ).reshape(B, M),
            "geo_aux_loss_matrix": geo_aux_per_sample.reshape(B, M),
            "trajectory_aux_loss_matrix": trajectory_aux_per_sample.reshape(B, M),
            "feasibility_aux_loss_matrix": feasibility_aux_per_sample.reshape(B, M),
            "tangent_excess_loss_matrix": tangent_per_sample.reshape(B, M),
            "curvature_excess_loss_matrix": curvature_per_sample.reshape(B, M),
            "aux_alpha_weight_matrix": alpha_weight_per_sample.reshape(B, M),
            "full_x0_reconstruction_l1_matrix": full_x0_l1_per_sample.reshape(B, M),
            "fs_output_bound_hit_matrix": fs_bound_hit_per_sample.reshape(B, M),
            "selected_target_is_gt_matrix": target_is_gt_matrix,
            "selected_target_source_code_matrix": source_code_matrix,
            "fs_target_abs_gt3_matrix": fs_abs_gt3_matrix,
            "fs_target_abs_gt5_matrix": fs_abs_gt5_matrix,
            "x0_aux_active_mask": aux_active_mask.reshape(B, M).detach(),
            "early_kink_rate": geo_diag["early_kink_rate"].detach(),
            "tail_reverse_rate": geo_diag["tail_reverse_rate"].detach(),
            "curvature_violation_rate": geo_diag["curvature_violation_rate"].detach(),
            **{key: value.detach() for key, value in training_target.diagnostics.items()},
            **{key: value.detach() for key, value in pta_diag.items()},
            **dit_scalar_diag,
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
        target_source_code: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if target_trajs.ndim != 4 or target_trajs.shape[-1] != 3:
            raise ValueError(f"target_trajs must have shape [B, M, H, 3], got {tuple(target_trajs.shape)}.")
        B, M, H, D = target_trajs.shape
        if weights.shape != (B, M):
            raise ValueError(f"weights must have shape [B, M], got {tuple(weights.shape)}.")
        if target_source_code is not None and target_source_code.shape != (B, M):
            raise ValueError(
                f"target_source_code must have shape [B, M], got {tuple(target_source_code.shape)}."
            )
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
                "x0_aux_loss": zero_loss.detach(),
                "delta_aux_loss": zero_loss.detach(),
                "geo_aux_loss": zero_loss.detach(),
                "trajectory_aux_loss": zero_loss.detach(),
                "feasibility_aux_loss": zero_loss.detach(),
                "tangent_excess_loss": zero_loss.detach(),
                "curvature_excess_loss": zero_loss.detach(),
                "aux_alpha_weight_mean": zero_loss.detach(),
                "aux_warmup_ramp": zero_loss.new_tensor(self._aux_warmup_ramp()).detach(),
                "selected_target_is_gt_ratio": zero_loss.detach(),
                "selected_target_source_code": zero_loss.detach(),
                "residual_alpha": zero_loss.new_tensor(self._last_vla_residual_alpha(training=self.training)).detach(),
                "full_x0_reconstruction_l1": zero_loss.detach(),
                "fs_target_abs_gt3_ratio": zero_loss.detach(),
                "fs_target_abs_gt5_ratio": zero_loss.detach(),
                "fs_output_bound_hit_ratio": zero_loss.detach(),
                "x0_aux_active_weight_sum": zero_loss.detach(),
                "x0_aux_active_ratio": zero_loss.detach(),
                "early_kink_rate": zero_loss.detach(),
                "tail_reverse_rate": zero_loss.detach(),
                "curvature_violation_rate": zero_loss.detach(),
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
            target_source_code=target_source_code,
        )
        x0_aux_matrix = loss_diag.pop("x0_aux_loss_matrix", None)
        delta_aux_matrix = loss_diag.pop("delta_aux_loss_matrix", None)
        geo_aux_matrix = loss_diag.pop("geo_aux_loss_matrix", None)
        trajectory_aux_matrix = loss_diag.pop("trajectory_aux_loss_matrix", None)
        feasibility_aux_matrix = loss_diag.pop("feasibility_aux_loss_matrix", None)
        tangent_excess_matrix = loss_diag.pop("tangent_excess_loss_matrix", None)
        curvature_excess_matrix = loss_diag.pop("curvature_excess_loss_matrix", None)
        alpha_weight_matrix = loss_diag.pop("aux_alpha_weight_matrix", None)
        full_x0_l1_matrix = loss_diag.pop("full_x0_reconstruction_l1_matrix", None)
        fs_bound_hit_matrix = loss_diag.pop("fs_output_bound_hit_matrix", None)
        target_is_gt_matrix = loss_diag.pop("selected_target_is_gt_matrix", None)
        target_source_matrix = loss_diag.pop("selected_target_source_code_matrix", None)
        fs_abs_gt3_matrix = loss_diag.pop("fs_target_abs_gt3_matrix", None)
        fs_abs_gt5_matrix = loss_diag.pop("fs_target_abs_gt5_matrix", None)
        aux_active_mask = loss_diag.pop("x0_aux_active_mask", None)
        per_sample_loss = per_target_loss.reshape(B * M)
        valid_weight_sum = raw_weight_sum.clamp(min=1e-6)
        awac_loss = (flat_weights.to(per_sample_loss) * per_sample_loss).sum() / valid_weight_sum.to(per_sample_loss)
        if not torch.isfinite(awac_loss):
            raise ValueError("AWAC weighted diffusion loss is non-finite.")
        zero = awac_loss.new_zeros(())
        x0_aux_loss = zero
        delta_aux_loss = zero
        geo_aux_loss = zero
        trajectory_aux_loss = zero
        feasibility_aux_loss = zero
        tangent_excess_loss = zero
        curvature_excess_loss = zero
        aux_alpha_weight_mean = zero
        full_x0_reconstruction_l1 = zero
        fs_output_bound_hit_ratio = zero
        selected_target_is_gt_ratio = zero
        selected_target_source_code = zero
        fs_target_abs_gt3_ratio = zero
        fs_target_abs_gt5_ratio = zero
        aux_active_weight_sum = zero
        aux_active_ratio = zero

        def weighted_matrix_mean(matrix: Optional[torch.Tensor]) -> torch.Tensor:
            if matrix is None:
                return zero
            matrix_weights = weights.to(device=matrix.device, dtype=matrix.dtype)
            return (matrix_weights * matrix).sum() / valid_weight_sum.to(device=matrix.device, dtype=matrix.dtype)

        trajectory_aux_loss = weighted_matrix_mean(trajectory_aux_matrix)
        feasibility_aux_loss = weighted_matrix_mean(feasibility_aux_matrix)
        tangent_excess_loss = weighted_matrix_mean(tangent_excess_matrix)
        curvature_excess_loss = weighted_matrix_mean(curvature_excess_matrix)
        aux_alpha_weight_mean = weighted_matrix_mean(alpha_weight_matrix)
        full_x0_reconstruction_l1 = weighted_matrix_mean(full_x0_l1_matrix)
        fs_output_bound_hit_ratio = weighted_matrix_mean(fs_bound_hit_matrix)
        selected_target_is_gt_ratio = weighted_matrix_mean(target_is_gt_matrix)
        selected_target_source_code = weighted_matrix_mean(target_source_matrix)
        fs_target_abs_gt3_ratio = weighted_matrix_mean(fs_abs_gt3_matrix)
        fs_target_abs_gt5_ratio = weighted_matrix_mean(fs_abs_gt5_matrix)
        if x0_aux_matrix is not None and geo_aux_matrix is not None and aux_active_mask is not None:
            aux_weights = weights.to(device=target_trajs.device, dtype=torch.float32) * aux_active_mask.to(
                device=target_trajs.device,
                dtype=torch.float32,
            )
            aux_active_weight_sum = aux_weights.sum().to(device=awac_loss.device, dtype=awac_loss.dtype)
            aux_active_ratio = aux_active_mask.float().mean().to(device=awac_loss.device, dtype=awac_loss.dtype)
            if bool((aux_active_weight_sum.detach() > 0.0).cpu().item()):
                aux_denom = aux_active_weight_sum.clamp(min=1e-6)
                x0_aux_loss = (
                    aux_weights.to(device=x0_aux_matrix.device, dtype=x0_aux_matrix.dtype) * x0_aux_matrix
                ).sum() / aux_denom.to(device=x0_aux_matrix.device, dtype=x0_aux_matrix.dtype)
                if delta_aux_matrix is not None:
                    delta_aux_loss = (
                        aux_weights.to(device=delta_aux_matrix.device, dtype=delta_aux_matrix.dtype) * delta_aux_matrix
                    ).sum() / aux_denom.to(device=delta_aux_matrix.device, dtype=delta_aux_matrix.dtype)
                geo_aux_loss = (
                    aux_weights.to(device=geo_aux_matrix.device, dtype=geo_aux_matrix.dtype) * geo_aux_matrix
                ).sum() / aux_denom.to(device=geo_aux_matrix.device, dtype=geo_aux_matrix.dtype)
        diagnostics = {
            "per_sample_loss_mean": loss_diag["per_sample_loss_mean"].to(device=awac_loss.device, dtype=awac_loss.dtype),
            "target_norm_mean": loss_diag["target_norm_mean"].to(device=awac_loss.device, dtype=awac_loss.dtype),
            "diffusion_timestep_mean": loss_diag["diffusion_timestep_mean"].to(device=awac_loss.device, dtype=awac_loss.dtype),
            "diffusion_timestep_min": loss_diag["diffusion_timestep_min"].to(device=awac_loss.device, dtype=awac_loss.dtype),
            "diffusion_timestep_max": loss_diag["diffusion_timestep_max"].to(device=awac_loss.device, dtype=awac_loss.dtype),
            "effective_weight_sum": valid_weight_sum.detach().to(dtype=awac_loss.dtype),
            "zero_weight_ratio": (flat_weights <= 0).float().mean().to(device=awac_loss.device, dtype=awac_loss.dtype),
            "zero_weight_batch": zero_weight_batch.to(device=awac_loss.device, dtype=awac_loss.dtype),
            "x0_aux_loss": x0_aux_loss.to(device=awac_loss.device, dtype=awac_loss.dtype),
            "delta_aux_loss": delta_aux_loss.to(device=awac_loss.device, dtype=awac_loss.dtype),
            "geo_aux_loss": geo_aux_loss.to(device=awac_loss.device, dtype=awac_loss.dtype),
            "trajectory_aux_loss": trajectory_aux_loss.to(device=awac_loss.device, dtype=awac_loss.dtype),
            "feasibility_aux_loss": feasibility_aux_loss.to(device=awac_loss.device, dtype=awac_loss.dtype),
            "tangent_excess_loss": tangent_excess_loss.to(device=awac_loss.device, dtype=awac_loss.dtype),
            "curvature_excess_loss": curvature_excess_loss.to(device=awac_loss.device, dtype=awac_loss.dtype),
            "aux_alpha_weight_mean": aux_alpha_weight_mean.to(device=awac_loss.device, dtype=awac_loss.dtype),
            "aux_warmup_ramp": awac_loss.new_tensor(self._aux_warmup_ramp()).detach(),
            "selected_target_is_gt_ratio": selected_target_is_gt_ratio.to(
                device=awac_loss.device,
                dtype=awac_loss.dtype,
            ),
            "selected_target_source_code": selected_target_source_code.to(
                device=awac_loss.device,
                dtype=awac_loss.dtype,
            ),
            "residual_alpha": loss_diag.get("residual_alpha", zero).to(
                device=awac_loss.device,
                dtype=awac_loss.dtype,
            ),
            "full_x0_reconstruction_l1": full_x0_reconstruction_l1.to(
                device=awac_loss.device,
                dtype=awac_loss.dtype,
            ),
            "fs_target_abs_gt3_ratio": fs_target_abs_gt3_ratio.to(
                device=awac_loss.device,
                dtype=awac_loss.dtype,
            ),
            "fs_target_abs_gt5_ratio": fs_target_abs_gt5_ratio.to(
                device=awac_loss.device,
                dtype=awac_loss.dtype,
            ),
            "fs_output_bound_hit_ratio": fs_output_bound_hit_ratio.to(
                device=awac_loss.device,
                dtype=awac_loss.dtype,
            ),
            "x0_aux_active_weight_sum": aux_active_weight_sum.detach(),
            "x0_aux_active_ratio": aux_active_ratio.detach(),
            "early_kink_rate": loss_diag["early_kink_rate"].to(device=awac_loss.device, dtype=awac_loss.dtype),
            "tail_reverse_rate": loss_diag["tail_reverse_rate"].to(device=awac_loss.device, dtype=awac_loss.dtype),
            "curvature_violation_rate": loss_diag["curvature_violation_rate"].to(device=awac_loss.device, dtype=awac_loss.dtype),
        }
        for key in (
            "planning_token_norm",
            "planning_token_pairwise_cosine",
            "planning_condition_keep_ratio",
            "planning_context_gate",
            "planning_layer_gate_mean",
            "planning_delta_norm",
            "planning_adapter_forward_count",
        ):
            value = loss_diag.get(key)
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                diagnostics[key] = value.to(device=awac_loss.device, dtype=awac_loss.dtype)
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
                "preference_dpo_logit_mean": zero.detach(),
                "preference_dpo_logit_abs_mean": zero.detach(),
                "preference_dpo_implicit_accuracy": zero.detach(),
                "preference_dpo_current_winner_loss_mean": zero.detach(),
                "preference_dpo_current_loser_loss_mean": zero.detach(),
                "preference_dpo_reference_winner_loss_mean": zero.detach(),
                "preference_dpo_reference_loser_loss_mean": zero.detach(),
                "preference_dpo_current_margin_mean": zero.detach(),
                "preference_dpo_reference_margin_mean": zero.detach(),
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
        dpo_logits = []
        dpo_logit_abs = []
        implicit_accuracies = []
        current_winner_losses = []
        current_loser_losses = []
        reference_winner_losses = []
        reference_loser_losses = []
        current_margins = []
        reference_margins = []
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
            current_winner_loss = current_loss[b, winner_idx].expand_as(current_logratio)
            reference_winner_loss = reference_loss[b, winner_idx].expand_as(reference_logratio)
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
            dpo_logits.append(logits.detach().float().mean())
            dpo_logit_abs.append(logits.detach().float().abs().mean())
            implicit_accuracies.append((logits.detach() > 0).float().mean())
            current_winner_losses.append(current_winner_loss.detach().float().mean())
            current_loser_losses.append(current_loss[b, loser_idx].detach().float().mean())
            reference_winner_losses.append(reference_winner_loss.detach().float().mean())
            reference_loser_losses.append(reference_loss[b, loser_idx].detach().float().mean())
            current_margins.append(current_logratio.detach().float().mean())
            reference_margins.append(reference_logratio.detach().float().mean())
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
                "preference_dpo_logit_mean": zero.detach(),
                "preference_dpo_logit_abs_mean": zero.detach(),
                "preference_dpo_implicit_accuracy": zero.detach(),
                "preference_dpo_current_winner_loss_mean": zero.detach(),
                "preference_dpo_current_loser_loss_mean": zero.detach(),
                "preference_dpo_reference_winner_loss_mean": zero.detach(),
                "preference_dpo_reference_loser_loss_mean": zero.detach(),
                "preference_dpo_current_margin_mean": zero.detach(),
                "preference_dpo_reference_margin_mean": zero.detach(),
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
            "preference_dpo_logit_mean": torch.stack(dpo_logits).mean().to(current_loss),
            "preference_dpo_logit_abs_mean": torch.stack(dpo_logit_abs).mean().to(current_loss),
            "preference_dpo_implicit_accuracy": torch.stack(implicit_accuracies).mean().to(current_loss),
            "preference_dpo_current_winner_loss_mean": torch.stack(current_winner_losses).mean().to(current_loss),
            "preference_dpo_current_loser_loss_mean": torch.stack(current_loser_losses).mean().to(current_loss),
            "preference_dpo_reference_winner_loss_mean": torch.stack(reference_winner_losses).mean().to(current_loss),
            "preference_dpo_reference_loser_loss_mean": torch.stack(reference_loser_losses).mean().to(current_loss),
            "preference_dpo_current_margin_mean": torch.stack(current_margins).mean().to(current_loss),
            "preference_dpo_reference_margin_mean": torch.stack(reference_margins).mean().to(current_loss),
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
        buffer_path = self._offline_candidate_buffer_path(cfg)
        buffer_root = Path(buffer_path)
        if not str(buffer_root):
            raise ValueError(
                "offline_rl_cfg.elite_buffer_path/support_archive_path is empty and build_candidates_online=False."
            )
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

        if bool(getattr(cfg, "use_dpsi", False)) and bool(getattr(cfg, "dpsi_filter_to_support_indices", True)):
            filtered_records = []
            for record in records:
                if "support_indices" not in record:
                    raise KeyError(
                        f"DPSI requires support_indices in v3 archive for token={record.get('token', '')!r}."
                    )
                support_indices = np.asarray(record["support_indices"], dtype=np.int64)
                if support_indices.ndim != 1 or support_indices.size == 0:
                    raise ValueError(
                        f"DPSI requires non-empty support_indices in v3 archive for token={record.get('token', '')!r}."
                    )
                candidate_count = int(np.asarray(record["candidates"]).shape[0])
                if int(support_indices.min()) < 0 or int(support_indices.max()) >= candidate_count:
                    raise IndexError(
                        f"DPSI support_indices out of range for token={record.get('token', '')!r}: "
                        f"candidate_count={candidate_count}, support_indices={support_indices.tolist()}."
                    )
                filtered = dict(record)
                for key in ("candidates", "rewards", "anchor_distance", "valid_mask", "selection_score"):
                    if key in filtered:
                        filtered[key] = np.asarray(filtered[key])[support_indices]
                filtered["sources"] = [str(record["sources"][int(i)]) for i in support_indices.tolist()]
                filtered["support_tags"] = [str(record.get("support_tags", [""] * candidate_count)[int(i)]) for i in support_indices.tolist()]
                if "components" in filtered:
                    filtered["components"] = {
                        key: np.asarray(value)[support_indices]
                        for key, value in dict(record["components"]).items()
                    }
                filtered["support_indices"] = list(range(int(support_indices.size)))
                filtered_records.append(filtered)
            records = filtered_records

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
        selected_support_weight = torch.zeros((B, max_m), device=device, dtype=torch.float32)
        selected_support_tag_code = torch.zeros((B, max_m), device=device, dtype=torch.long)
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
            support_tags = [str(tag) for tag in record.get("support_tags", [""] * m)]
            if len(support_tags) != m:
                raise ValueError(
                    f"Elite buffer token={record['token']!r} support_tags length {len(support_tags)} "
                    f"does not match candidate count {m}."
                )
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
            selected_support_weight[b, :m] = torch.tensor(
                [self._dpsi_weight_for_tag(tag, source, cfg) for tag, source in zip(support_tags, sources)],
                device=device,
                dtype=torch.float32,
            )
            selected_support_tag_code[b, :m] = torch.tensor(
                [int(hashlib.sha1(tag.encode("utf-8")).hexdigest()[:8], 16) % 1000 for tag in support_tags],
                device=device,
                dtype=torch.long,
            )
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
            "selected_support_weight": selected_support_weight,
            "selected_support_tag_code": selected_support_tag_code,
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

    def _load_grpo_buffer_guidance_batch(
        self,
        action_input: BatchFeature,
        tokens_list: list[str],
        cfg: OfflineRLConfig,
    ) -> Dict[str, Any]:
        if not bool(cfg.enabled) or not bool(cfg.grpo_buffer_guidance_enabled):
            return {}
        if not str(self._offline_candidate_buffer_path(cfg)):
            raise ValueError("GRPO buffer guidance requires agent.offline_rl_elite_buffer_path/support_archive_path.")
        if bool(cfg.build_candidates_online):
            raise ValueError("GRPO buffer guidance currently expects a prebuilt offline elite buffer.")
        metric_cache: Dict[str, Any] = {}
        if str(cfg.missing_buffer_policy) == "fallback_gt":
            metric_cache = self._load_metric_cache_for_tokens(tokens_list)
        awac_batch = self._load_awac_buffer_candidates(action_input, tokens_list, metric_cache, cfg)
        selected_rewards = awac_batch["selected_rewards"].float()
        selected_real_mask = awac_batch["selected_real_mask"]
        selected_valid_mask = awac_batch["selected_valid_mask"]
        gt_reward = awac_batch["gt_reward"].to(selected_rewards)
        il_reward = awac_batch["il_reward"].to(selected_rewards)
        baseline = torch.maximum(gt_reward, il_reward)
        eligible = selected_real_mask & selected_valid_mask
        margin = selected_rewards - baseline[:, None]
        eligible &= margin > float(cfg.grpo_buffer_distill_min_reward_margin)

        B, _, H, D = awac_batch["selected_trajs"].shape
        top_k = max(1, int(cfg.grpo_buffer_distill_top_k))
        target_trajs = awac_batch["selected_trajs"].new_zeros((B, top_k, H, D))
        target_weights = selected_rewards.new_zeros((B, top_k))
        target_rewards = selected_rewards.new_zeros((B, top_k))
        target_margin = selected_rewards.new_zeros((B, top_k))
        target_mask = torch.zeros((B, top_k), device=selected_rewards.device, dtype=torch.bool)
        for b in range(B):
            idx = torch.nonzero(eligible[b], as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            row_score = selected_rewards[b, idx].float()
            keep = idx[torch.topk(row_score, k=min(top_k, int(idx.numel())), largest=True).indices]
            m = int(keep.numel())
            target_trajs[b, :m] = awac_batch["selected_trajs"][b, keep]
            target_rewards[b, :m] = selected_rewards[b, keep]
            target_margin[b, :m] = margin[b, keep].clamp(min=0.0)
            weight = target_margin[b, :m].clamp(min=0.0)
            if not bool((weight > 0).any().item()):
                weight = torch.ones_like(weight)
            target_weights[b, :m] = weight / weight.mean().clamp(min=1e-6)
            target_mask[b, :m] = True

        return {
            "target_trajs": target_trajs.detach(),
            "target_weights": target_weights.detach(),
            "target_rewards": target_rewards.detach(),
            "target_margin": target_margin.detach(),
            "target_mask": target_mask.detach(),
            "selected_trajs": awac_batch["selected_trajs"].detach(),
            "selected_rewards": selected_rewards.detach(),
            "selected_components": {
                key: value.detach()
                for key, value in awac_batch["selected_components"].items()
            },
            "selected_real_mask": selected_real_mask.detach(),
            "selected_valid_mask": selected_valid_mask.detach(),
            "selected_source_code": awac_batch["selected_source_code"].detach(),
            "gt_reward": gt_reward.detach(),
            "il_reward": il_reward.detach(),
            "best_valid_reward": awac_batch["best_valid_reward"].detach(),
            "has_target_ratio": target_mask.any(dim=1).float().mean(),
            "selected_valid_ratio": (
                ((selected_valid_mask & selected_real_mask).float().sum())
                / selected_real_mask.float().sum().clamp(min=1.0)
            ).detach(),
            "best_valid_minus_gt_mean": (awac_batch["best_valid_reward"].to(gt_reward) - gt_reward).mean().detach(),
            "best_valid_minus_il_mean": (awac_batch["best_valid_reward"].to(il_reward) - il_reward).mean().detach(),
        }

    def _compute_grpo_buffer_reward_bonus(
        self,
        trajs: torch.Tensor,
        B: int,
        G: int,
        guidance: Dict[str, Any],
        cfg: OfflineRLConfig,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = trajs.new_zeros(())
        if not guidance or "target_trajs" not in guidance:
            return trajs.new_zeros((B * G,)), {
                "grpo_buffer_reward_bonus_mean": zero,
                "grpo_buffer_reward_bonus_max": zero,
                "grpo_buffer_target_distance_mean": zero,
                "grpo_buffer_guidance_target_ratio": zero,
            }
        target_trajs = guidance["target_trajs"].to(device=trajs.device, dtype=trajs.dtype)
        target_mask = guidance["target_mask"].to(device=trajs.device)
        target_margin = guidance["target_margin"].to(device=trajs.device, dtype=trajs.dtype)
        if target_trajs.ndim != 4 or target_trajs.shape[0] != B:
            raise ValueError("GRPO buffer guidance targets must have shape [B, M, H, 3].")
        if not bool(target_mask.any().item()):
            return trajs.new_zeros((B * G,)), {
                "grpo_buffer_reward_bonus_mean": zero,
                "grpo_buffer_reward_bonus_max": zero,
                "grpo_buffer_target_distance_mean": zero,
                "grpo_buffer_guidance_target_ratio": zero,
            }

        generated = trajs.reshape(B, G, trajs.shape[-2], trajs.shape[-1])[..., :2].float()
        targets = target_trajs[..., :2].float()
        distances = torch.linalg.norm(generated[:, :, None] - targets[:, None], dim=-1).mean(dim=-1)
        valid_distances = distances.masked_fill(~target_mask[:, None, :], torch.inf)
        scale = max(float(cfg.grpo_buffer_reward_bonus_scale_m), 1e-6)
        closeness = torch.exp(-(valid_distances / scale).clamp(min=0.0, max=50.0))
        if bool(cfg.grpo_buffer_reward_bonus_use_margin):
            margin_factor = (target_margin / 0.2).clamp(min=0.0, max=1.0)
            closeness = closeness * margin_factor[:, None, :]
        bonus = closeness.max(dim=2).values
        bonus = torch.where(torch.isfinite(valid_distances).any(dim=2), bonus, torch.zeros_like(bonus))
        finite_dist = valid_distances[torch.isfinite(valid_distances)]
        distance_mean = finite_dist.mean().to(trajs) if finite_dist.numel() > 0 else zero
        return bonus.reshape(B * G).to(trajs), {
            "grpo_buffer_reward_bonus_mean": bonus.mean().to(trajs),
            "grpo_buffer_reward_bonus_max": bonus.max().to(trajs),
            "grpo_buffer_target_distance_mean": distance_mean,
            "grpo_buffer_guidance_target_ratio": target_mask.any(dim=1).float().mean().to(trajs),
        }

    @staticmethod
    def _zero_diffusion_dpo_diag(zero: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "preference_dpo_pair_count": zero.detach(),
            "preference_dpo_active_row_ratio": zero.detach(),
            "preference_dpo_reward_gap_mean": zero.detach(),
            "preference_dpo_gap_weight_mean": zero.detach(),
            "preference_dpo_current_logratio_mean": zero.detach(),
            "preference_dpo_reference_logratio_mean": zero.detach(),
            "preference_dpo_logit_mean": zero.detach(),
            "preference_dpo_logit_abs_mean": zero.detach(),
            "preference_dpo_implicit_accuracy": zero.detach(),
            "preference_dpo_current_winner_loss_mean": zero.detach(),
            "preference_dpo_current_loser_loss_mean": zero.detach(),
            "preference_dpo_reference_winner_loss_mean": zero.detach(),
            "preference_dpo_reference_loser_loss_mean": zero.detach(),
            "preference_dpo_current_margin_mean": zero.detach(),
            "preference_dpo_reference_margin_mean": zero.detach(),
            "preference_dpo_timestep_mean": zero.detach(),
            "preference_dpo_timestep_min": zero.detach(),
            "preference_dpo_timestep_max": zero.detach(),
        }

    def _build_grpo_buffer_preference_dpo_targets(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        guidance: Dict[str, Any],
        cfg: OfflineRLConfig,
    ) -> Dict[str, torch.Tensor]:
        if not guidance or "target_trajs" not in guidance:
            zero = vl_features.new_zeros(())
            return {
                "target_trajs": vl_features.new_zeros((vl_features.shape[0], 0, self.config.action_horizon, 3)),
                "ordering_rewards": vl_features.new_zeros((vl_features.shape[0], 0)),
                "valid_mask": torch.zeros((vl_features.shape[0], 0), device=vl_features.device, dtype=torch.bool),
                "real_mask": torch.zeros((vl_features.shape[0], 0), device=vl_features.device, dtype=torch.bool),
                "source_code": torch.zeros((vl_features.shape[0], 0), device=vl_features.device, dtype=torch.long),
                "target_ratio": zero,
                "il_loser_ratio": zero,
            }

        buffer_targets = guidance["target_trajs"].to(device=action_input.action.device, dtype=action_input.action.dtype)
        buffer_rewards = guidance["target_rewards"].to(device=action_input.action.device, dtype=torch.float32)
        buffer_mask = guidance["target_mask"].to(device=action_input.action.device, dtype=torch.bool)
        if buffer_targets.ndim != 4 or buffer_targets.shape[-1] != 3:
            raise ValueError("GRPO buffer DPO target_trajs must have shape [B, K, H, 3].")
        B, K, H, D = buffer_targets.shape
        if action_input.action.shape != (B, H, D):
            raise ValueError(
                f"action_input.action must have shape {(B, H, D)} for GRPO buffer DPO, "
                f"got {tuple(action_input.action.shape)}."
            )
        include_il = bool(cfg.grpo_buffer_preference_dpo_include_il)
        M = K + 1 + int(include_il)
        dpo_trajs = buffer_targets.new_zeros((B, M, H, D))
        ordering_rewards = buffer_rewards.new_zeros((B, M))
        valid_mask = torch.zeros((B, M), device=buffer_targets.device, dtype=torch.bool)
        real_mask = torch.zeros((B, M), device=buffer_targets.device, dtype=torch.bool)
        source_code = torch.zeros((B, M), device=buffer_targets.device, dtype=torch.long)

        dpo_trajs[:, :K] = buffer_targets.detach()
        ordering_rewards[:, :K] = buffer_rewards.detach()
        valid_mask[:, :K] = buffer_mask
        real_mask[:, :K] = buffer_mask

        gt_col = K
        dpo_trajs[:, gt_col] = action_input.action.detach()
        ordering_rewards[:, gt_col] = guidance["gt_reward"].to(device=buffer_targets.device, dtype=torch.float32)
        real_mask[:, gt_col] = True
        source_code[:, gt_col] = self._source_code("gt")

        il_loser_mask = torch.zeros((B,), device=buffer_targets.device, dtype=torch.bool)
        if include_il:
            il_col = K + 1
            il_traj = buffer_targets.new_zeros((B, H, D))
            if all(key in guidance for key in ("selected_trajs", "selected_real_mask", "selected_source_code")):
                selected_trajs = guidance["selected_trajs"].to(device=buffer_targets.device, dtype=buffer_targets.dtype)
                selected_real_mask = guidance["selected_real_mask"].to(device=buffer_targets.device, dtype=torch.bool)
                selected_source_code = guidance["selected_source_code"].to(device=buffer_targets.device, dtype=torch.long)
                for b in range(B):
                    row_il = torch.nonzero(
                        selected_real_mask[b] & (selected_source_code[b] == self._source_code("il")),
                        as_tuple=False,
                    ).flatten()
                    if row_il.numel() > 0:
                        il_traj[b] = selected_trajs[b, row_il[0]]
                        il_loser_mask[b] = True
            # Do not synthesize a fallback IL trajectory here: guidance["il_reward"]
            # belongs to the buffer record, so sampled IL would not have a matched reward.
            dpo_trajs[:, il_col] = il_traj.detach()
            ordering_rewards[:, il_col] = guidance["il_reward"].to(device=buffer_targets.device, dtype=torch.float32)
            real_mask[:, il_col] = il_loser_mask
            source_code[:, il_col] = self._source_code("il")

        return {
            "target_trajs": dpo_trajs.detach(),
            "ordering_rewards": ordering_rewards.detach(),
            "valid_mask": valid_mask.detach(),
            "real_mask": real_mask.detach(),
            "source_code": source_code.detach(),
            "target_ratio": buffer_mask.any(dim=1).float().mean().detach(),
            "il_loser_ratio": il_loser_mask.float().mean().detach(),
        }

    def _build_grpo_self_imitation_targets(
        self,
        trajs: torch.Tensor,
        base_rewards: torch.Tensor,
        hard_safe_mask: torch.Tensor,
        components: Optional[Dict[str, torch.Tensor]],
        B: int,
        G: int,
        guidance: Dict[str, Any],
        cfg: OfflineRLConfig,
    ) -> Dict[str, Any]:
        """Select high-reward on-policy samples as detached diffusion targets."""
        zero = trajs.new_zeros(())
        if trajs.shape[0] != B * G or base_rewards.shape != (B * G,) or hard_safe_mask.shape != (B * G,):
            raise ValueError("GRPO self-imitation expects flattened [B*G] trajectories/rewards/safety masks.")

        sample_trajs = trajs.detach().reshape(B, G, trajs.shape[-2], trajs.shape[-1])
        reward_matrix = base_rewards.detach().reshape(B, G).float()
        safe_matrix = hard_safe_mask.detach().reshape(B, G).bool()
        finite_reward = torch.isfinite(reward_matrix)
        target_safe_matrix = safe_matrix & finite_reward

        component_matrices: Dict[str, torch.Tensor] = {}

        def _component_matrix(name: str) -> torch.Tensor:
            if name not in component_matrices:
                if components is None or name not in components:
                    raise KeyError(
                        f"GRPO self-imitation target filtering requires PDM component '{name}'."
                    )
                value = components[name].detach()
                if value.shape != (B * G,):
                    raise ValueError(
                        f"GRPO self-imitation component '{name}' must have shape {(B * G,)}, "
                        f"got {tuple(value.shape)}."
                    )
                component_matrices[name] = value.reshape(B, G).to(
                    device=reward_matrix.device,
                    dtype=reward_matrix.dtype,
                )
            return component_matrices[name]

        pass_ratios: Dict[str, torch.Tensor] = {
            "nc": trajs.new_ones(()),
            "dac": trajs.new_ones(()),
            "ttc": trajs.new_ones(()),
            "ddc": trajs.new_ones(()),
        }
        if bool(cfg.grpo_self_imitation_require_nc):
            nc_pass = _component_matrix("no_at_fault_collisions") >= float(
                cfg.grpo_self_imitation_nc_min_absolute
            )
            target_safe_matrix &= nc_pass
            pass_ratios["nc"] = nc_pass.float().mean().detach()
        if bool(cfg.grpo_self_imitation_require_dac):
            dac_pass = _component_matrix("drivable_area_compliance") >= float(
                cfg.grpo_self_imitation_dac_min_absolute
            )
            target_safe_matrix &= dac_pass
            pass_ratios["dac"] = dac_pass.float().mean().detach()
        if bool(cfg.grpo_self_imitation_require_ttc):
            ttc_pass = _component_matrix("time_to_collision_within_bound") >= float(
                cfg.grpo_self_imitation_ttc_min_absolute
            )
            target_safe_matrix &= ttc_pass
            pass_ratios["ttc"] = ttc_pass.float().mean().detach()
        if bool(cfg.grpo_self_imitation_require_ddc):
            ddc_pass = _component_matrix("driving_direction_compliance") >= float(
                cfg.grpo_self_imitation_ddc_min_absolute
            )
            target_safe_matrix &= ddc_pass
            pass_ratios["ddc"] = ddc_pass.float().mean().detach()

        mode = str(cfg.grpo_self_imitation_baseline_mode)
        has_buffer_baseline = bool(guidance) and "gt_reward" in guidance and "il_reward" in guidance
        baseline_from_buffer = False
        baseline: Optional[torch.Tensor] = None
        baseline_matrix: Optional[torch.Tensor] = None
        if mode == "buffer_gt_il":
            if not has_buffer_baseline:
                raise RuntimeError(
                    "grpo_self_imitation_baseline_mode=buffer_gt_il requires GRPO buffer guidance "
                    "with gt_reward and il_reward."
                )
            baseline = torch.maximum(
                guidance["gt_reward"].to(device=reward_matrix.device, dtype=reward_matrix.dtype),
                guidance["il_reward"].to(device=reward_matrix.device, dtype=reward_matrix.dtype),
            )
            baseline_from_buffer = True
        elif mode == "group_mean":
            baseline = reward_matrix.masked_fill(~finite_reward, 0.0).sum(dim=1)
            count = finite_reward.float().sum(dim=1).clamp(min=1.0)
            baseline = baseline / count
            baseline_from_buffer = False
        elif mode == "group_leave_one_out":
            reward_finite_zero = reward_matrix.masked_fill(~finite_reward, 0.0)
            count = finite_reward.float().sum(dim=1, keepdim=True)
            group_mean = reward_finite_zero.sum(dim=1, keepdim=True) / count.clamp(min=1.0)
            loo_count = (count - finite_reward.float()).clamp(min=1.0)
            loo_sum = reward_finite_zero.sum(dim=1, keepdim=True) - reward_finite_zero
            baseline_matrix = torch.where(
                (count > 1.0) & finite_reward,
                loo_sum / loo_count,
                group_mean.expand_as(reward_matrix),
            )
            baseline = group_mean.squeeze(1)
            baseline_from_buffer = False
        else:
            if has_buffer_baseline:
                baseline = torch.maximum(
                    guidance["gt_reward"].to(device=reward_matrix.device, dtype=reward_matrix.dtype),
                    guidance["il_reward"].to(device=reward_matrix.device, dtype=reward_matrix.dtype),
                )
                baseline_from_buffer = True
            else:
                baseline = reward_matrix.masked_fill(~finite_reward, 0.0).sum(dim=1)
                count = finite_reward.float().sum(dim=1).clamp(min=1.0)
                baseline = baseline / count
                baseline_from_buffer = False

        if baseline_matrix is None:
            if baseline is None:
                raise RuntimeError("GRPO self-imitation baseline was not initialized.")
            baseline_matrix = baseline[:, None].expand_as(reward_matrix)

        margins = reward_matrix - baseline_matrix
        eligible = (
            target_safe_matrix
            & (reward_matrix >= float(cfg.grpo_self_imitation_min_reward))
            & (margins > float(cfg.grpo_self_imitation_min_reward_margin))
        )

        top_k = max(1, int(cfg.grpo_self_imitation_top_k))
        _, _, H, D = sample_trajs.shape
        target_trajs = sample_trajs.new_zeros((B, top_k, H, D))
        target_weights = reward_matrix.new_zeros((B, top_k))
        target_rewards = reward_matrix.new_zeros((B, top_k))
        target_margin = reward_matrix.new_zeros((B, top_k))
        target_mask = torch.zeros((B, top_k), device=trajs.device, dtype=torch.bool)
        target_component_keys = (
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "time_to_collision_within_bound",
            "ego_progress",
            "history_comfort",
            "driving_direction_compliance",
            "traffic_light_compliance",
        )
        target_components = {
            key: reward_matrix.new_zeros((B, top_k))
            for key in target_component_keys
            if components is not None and key in components
        }

        for b in range(B):
            idx = torch.nonzero(eligible[b], as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            row_score = reward_matrix[b, idx]
            keep = idx[torch.topk(row_score, k=min(top_k, int(idx.numel())), largest=True).indices]
            m = int(keep.numel())
            target_trajs[b, :m] = sample_trajs[b, keep]
            target_rewards[b, :m] = reward_matrix[b, keep]
            target_margin[b, :m] = margins[b, keep].clamp(min=0.0)
            for key in target_components:
                target_components[key][b, :m] = _component_matrix(key)[b, keep]
            weight = target_margin[b, :m].clamp(min=0.0)
            if not bool((weight > 0).any().item()):
                weight = torch.ones_like(weight)
            target_weights[b, :m] = weight / weight.mean().clamp(min=1e-6)
            target_mask[b, :m] = True

        pre_cap_target_ratio = target_mask.any(dim=1).float().mean().detach()
        cap_active = False
        max_scene_ratio = float(cfg.grpo_self_imitation_max_target_scene_ratio)
        if max_scene_ratio < 1.0 and bool(target_mask.any().item()):
            scene_has_target = target_mask.any(dim=1)
            active_scene_idx = torch.nonzero(scene_has_target, as_tuple=False).flatten()
            max_keep = max(1, int(math.ceil(float(B) * max_scene_ratio)))
            if int(active_scene_idx.numel()) > max_keep:
                cap_active = True
                if str(cfg.grpo_self_imitation_batch_cap_score) == "margin":
                    scene_score_source = target_margin
                else:
                    scene_score_source = target_rewards
                scene_scores = scene_score_source.masked_fill(~target_mask, -float("inf")).max(dim=1).values
                keep_rel = torch.topk(scene_scores[active_scene_idx], k=max_keep, largest=True).indices
                keep_scene_idx = active_scene_idx[keep_rel]
                keep_scene_mask = torch.zeros((B,), device=target_mask.device, dtype=torch.bool)
                keep_scene_mask[keep_scene_idx] = True
                drop_scene_mask = scene_has_target & ~keep_scene_mask
                target_mask[drop_scene_mask] = False
                target_weights[drop_scene_mask] = 0.0

        active_rewards = target_rewards[target_mask]
        active_margin = target_margin[target_mask]
        active_component_means: Dict[str, torch.Tensor] = {}
        for key, value in target_components.items():
            active_values = value[target_mask]
            active_component_means[key] = (
                active_values.mean().detach() if active_values.numel() > 0 else zero.detach()
            )
        return {
            "target_trajs": target_trajs.detach(),
            "target_weights": target_weights.detach(),
            "target_rewards": target_rewards.detach(),
            "target_margin": target_margin.detach(),
            "target_mask": target_mask.detach(),
            "candidate_ratio": eligible.float().mean().detach(),
            "safety_candidate_ratio": target_safe_matrix.float().mean().detach(),
            "nc_pass_ratio": pass_ratios["nc"].to(device=trajs.device, dtype=trajs.dtype),
            "dac_pass_ratio": pass_ratios["dac"].to(device=trajs.device, dtype=trajs.dtype),
            "ttc_pass_ratio": pass_ratios["ttc"].to(device=trajs.device, dtype=trajs.dtype),
            "ddc_pass_ratio": pass_ratios["ddc"].to(device=trajs.device, dtype=trajs.dtype),
            "pre_cap_target_ratio": pre_cap_target_ratio,
            "target_ratio": target_mask.any(dim=1).float().mean().detach(),
            "target_scene_cap_ratio": trajs.new_tensor(max_scene_ratio),
            "target_scene_cap_active": trajs.new_tensor(float(cap_active)),
            "target_reward_mean": (
                active_rewards.mean().detach() if active_rewards.numel() > 0 else zero.detach()
            ),
            "target_reward_max": (
                active_rewards.max().detach() if active_rewards.numel() > 0 else zero.detach()
            ),
            "target_margin_mean": (
                active_margin.mean().detach() if active_margin.numel() > 0 else zero.detach()
            ),
            "baseline_mean": baseline.mean().detach(),
            "baseline_from_buffer": trajs.new_tensor(float(baseline_from_buffer)),
            "target_nc_mean": active_component_means.get("no_at_fault_collisions", zero.detach()),
            "target_dac_mean": active_component_means.get("drivable_area_compliance", zero.detach()),
            "target_ttc_mean": active_component_means.get("time_to_collision_within_bound", zero.detach()),
            "target_ep_mean": active_component_means.get("ego_progress", zero.detach()),
            "target_comfort_mean": active_component_means.get("history_comfort", zero.detach()),
            "target_ddc_mean": active_component_means.get("driving_direction_compliance", zero.detach()),
            "target_tlc_mean": active_component_means.get("traffic_light_compliance", zero.detach()),
        }

    def forward_awac_iql(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        tokens_list,
        use_bc_loss: bool = True,
    ) -> BatchFeature:
        self._reset_planning_adapter_forward_count()
        self.set_frozen_modules_to_eval_mode()
        cfg = self.offline_rl_cfg
        if not bool(cfg.enabled):
            raise RuntimeError("forward_awac_iql requires offline_rl_cfg.enabled=True.")
        if tokens_list is None:
            raise ValueError("forward_awac_iql requires tokens_list for train metric-cache reward lookup.")
        token_strs = [str(token) for token in tokens_list]
        use_offline_buffer = bool(str(self._offline_candidate_buffer_path(cfg))) and not bool(cfg.build_candidates_online)
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
        dpsi_weight_mean = weights.new_tensor(1.0)
        dpsi_effective_ratio = weights.new_zeros(())
        if bool(getattr(cfg, "use_dpsi", False)):
            support_weight = awac_batch.get("selected_support_weight")
            if isinstance(support_weight, torch.Tensor):
                support_weight = support_weight.to(device=weights.device, dtype=weights.dtype)
            else:
                support_weight = torch.ones_like(weights)
            weights = weights * support_weight
            real_positive = selected_real_mask & (weights > 0.0)
            dpsi_weight_mean = support_weight[selected_real_mask].mean().to(weights) if bool(selected_real_mask.any().item()) else weights.new_tensor(1.0)
            dpsi_effective_ratio = real_positive.float().sum().to(weights) / selected_real_mask.float().sum().clamp(min=1.0).to(weights)
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
            target_source_code=source_code,
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
            + float(getattr(self.config, "x0_aux_weight", 0.0)) * awac_diag["x0_aux_loss"].to(dtype=awac_loss.dtype)
            + float(getattr(self.config, "delta_aux_weight", 0.0)) * awac_diag["delta_aux_loss"].to(dtype=awac_loss.dtype)
            + float(getattr(self.config, "geo_aux_weight", 0.0)) * awac_diag["geo_aux_loss"].to(dtype=awac_loss.dtype)
            + awac_diag["aux_warmup_ramp"].to(dtype=awac_loss.dtype)
            * (
                float(getattr(self.config, "trajectory_aux_weight", 0.0))
                * awac_diag["trajectory_aux_loss"].to(dtype=awac_loss.dtype)
                + float(getattr(self.config, "feasibility_aux_weight", 0.0))
                * awac_diag["feasibility_aux_loss"].to(dtype=awac_loss.dtype)
            )
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
            "x0_aux_loss": awac_diag["x0_aux_loss"].detach(),
            "delta_aux_loss": awac_diag["delta_aux_loss"].detach(),
            "geo_aux_loss": awac_diag["geo_aux_loss"].detach(),
            "trajectory_aux_loss": awac_diag["trajectory_aux_loss"].detach(),
            "feasibility_aux_loss": awac_diag["feasibility_aux_loss"].detach(),
            "tangent_excess_loss": awac_diag["tangent_excess_loss"].detach(),
            "curvature_excess_loss": awac_diag["curvature_excess_loss"].detach(),
            "aux_alpha_weight_mean": awac_diag["aux_alpha_weight_mean"].detach(),
            "aux_warmup_ramp": awac_diag["aux_warmup_ramp"].detach(),
            "selected_target_is_gt_ratio": awac_diag["selected_target_is_gt_ratio"].detach(),
            "selected_target_source_code": awac_diag["selected_target_source_code"].detach(),
            "residual_alpha": awac_diag["residual_alpha"].detach(),
            "full_x0_reconstruction_l1": awac_diag["full_x0_reconstruction_l1"].detach(),
            "fs_target_abs_gt3_ratio": awac_diag["fs_target_abs_gt3_ratio"].detach(),
            "fs_target_abs_gt5_ratio": awac_diag["fs_target_abs_gt5_ratio"].detach(),
            "fs_output_bound_hit_ratio": awac_diag["fs_output_bound_hit_ratio"].detach(),
            **{
                key: awac_diag[key].detach()
                for key in (
                    "planning_token_norm",
                    "planning_token_pairwise_cosine",
                    "planning_condition_keep_ratio",
                    "planning_context_gate",
                    "planning_layer_gate_mean",
                    "planning_delta_norm",
                    "planning_adapter_forward_count",
                )
                if key in awac_diag and isinstance(awac_diag[key], torch.Tensor) and awac_diag[key].numel() == 1
            },
            "early_kink_rate": awac_diag["early_kink_rate"].detach(),
            "tail_reverse_rate": awac_diag["tail_reverse_rate"].detach(),
            "curvature_violation_rate": awac_diag["curvature_violation_rate"].detach(),
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
            "preference_dpo_logit_mean": dpo_diag["preference_dpo_logit_mean"].to(total_loss).detach(),
            "preference_dpo_logit_abs_mean": dpo_diag["preference_dpo_logit_abs_mean"].to(total_loss).detach(),
            "preference_dpo_implicit_accuracy": dpo_diag["preference_dpo_implicit_accuracy"].to(total_loss).detach(),
            "preference_dpo_current_winner_loss_mean": dpo_diag[
                "preference_dpo_current_winner_loss_mean"
            ].to(total_loss).detach(),
            "preference_dpo_current_loser_loss_mean": dpo_diag[
                "preference_dpo_current_loser_loss_mean"
            ].to(total_loss).detach(),
            "preference_dpo_reference_winner_loss_mean": dpo_diag[
                "preference_dpo_reference_winner_loss_mean"
            ].to(total_loss).detach(),
            "preference_dpo_reference_loser_loss_mean": dpo_diag[
                "preference_dpo_reference_loser_loss_mean"
            ].to(total_loss).detach(),
            "preference_dpo_current_margin_mean": dpo_diag["preference_dpo_current_margin_mean"].to(total_loss).detach(),
            "preference_dpo_reference_margin_mean": dpo_diag[
                "preference_dpo_reference_margin_mean"
            ].to(total_loss).detach(),
            "preference_dpo_timestep_mean": dpo_diag["preference_dpo_timestep_mean"].to(total_loss).detach(),
            "preference_dpo_timestep_min": dpo_diag["preference_dpo_timestep_min"].to(total_loss).detach(),
            "preference_dpo_timestep_max": dpo_diag["preference_dpo_timestep_max"].to(total_loss).detach(),
            "empty_awac_row_ratio": weight_diag["empty_awac_row_ratio"].to(total_loss).detach(),
            "positive_weight_row_ratio": weight_diag["positive_weight_row_ratio"].to(total_loss).detach(),
            "positive_weight_candidate_ratio": weight_diag["positive_weight_candidate_ratio"].to(total_loss).detach(),
            "target_filter_row_ratio": weight_diag["target_filter_row_ratio"].to(total_loss).detach(),
            "target_filter_candidate_ratio": weight_diag["target_filter_candidate_ratio"].to(total_loss).detach(),
            "support_tag_ratios": dpsi_effective_ratio.detach().to(total_loss),
            "dpsi_enabled": total_loss.new_tensor(float(bool(getattr(cfg, "use_dpsi", False)))),
            "dpsi_support_weight_mean": dpsi_weight_mean.detach().to(total_loss),
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

    def forward_dpsi(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        tokens_list,
        use_bc_loss: bool = True,
    ) -> BatchFeature:
        self._reset_planning_adapter_forward_count()
        self.set_frozen_modules_to_eval_mode()
        cfg = self.offline_rl_cfg
        if not bool(cfg.enabled) or not bool(cfg.use_dpsi):
            raise RuntimeError("forward_dpsi requires offline_rl_cfg.enabled=True and use_dpsi=True.")
        if tokens_list is None:
            raise ValueError("forward_dpsi requires tokens_list for support archive lookup.")
        token_strs = [str(token) for token in tokens_list]
        awac_batch = self._load_awac_buffer_candidates(action_input, token_strs, {}, cfg)
        current_epoch = int(getattr(self.config, "current_train_epoch", 0))

        profile, profile_diag = _compute_dpsi_support_profile(
            awac_batch["selected_trajs"],
            awac_batch["selected_rewards"],
            awac_batch["selected_real_mask"],
            awac_batch["selected_source_code"],
            awac_batch["gt_reward"],
            awac_batch["il_reward"],
            awac_batch["selected_valid_mask"],
            cfg,
            current_epoch=current_epoch,
        )
        asmi_weights, asmi_diag = _build_asmi_weights(
            awac_batch["selected_rewards"],
            awac_batch["selected_real_mask"],
            awac_batch["selected_valid_mask"],
            awac_batch["selected_source_code"],
            awac_batch["selected_support_weight"],
            awac_batch["gt_reward"],
            awac_batch["il_reward"],
            profile,
            cfg,
        )
        target_trajs, target_weights, target_mask, sample_diag = _sample_asmi_targets(
            awac_batch["selected_trajs"],
            asmi_weights,
            awac_batch["selected_real_mask"],
            awac_batch["selected_valid_mask"],
            awac_batch["selected_source_code"],
            awac_batch["selected_rewards"],
            cfg,
            current_epoch=current_epoch,
            training=bool(self.training),
        )
        target_source_code = sample_diag.pop("_sampled_target_source_code")
        dpsi_loss, dpsi_diag = self._weighted_diffusion_loss_on_targets(
            vl_features,
            action_input,
            target_trajs,
            target_weights,
            timestep_sampling=cfg.awac_timestep_sampling,
            target_source_code=target_source_code,
        )
        x0_aux = dpsi_diag["x0_aux_loss"].to(dtype=dpsi_loss.dtype)
        delta_aux = dpsi_diag["delta_aux_loss"].to(dtype=dpsi_loss.dtype)
        geo_aux = dpsi_diag["geo_aux_loss"].to(dtype=dpsi_loss.dtype)
        trajectory_aux = dpsi_diag["trajectory_aux_loss"].to(dtype=dpsi_loss.dtype)
        feasibility_aux = dpsi_diag["feasibility_aux_loss"].to(dtype=dpsi_loss.dtype)
        aux_ramp = dpsi_diag["aux_warmup_ramp"].to(dtype=dpsi_loss.dtype)
        total_loss = (
            dpsi_loss
            + float(getattr(self.config, "x0_aux_weight", 0.0)) * x0_aux
            + float(getattr(self.config, "delta_aux_weight", 0.0)) * delta_aux
            + float(getattr(self.config, "geo_aux_weight", 0.0)) * geo_aux
            + aux_ramp
            * (
                float(getattr(self.config, "trajectory_aux_weight", 0.0)) * trajectory_aux
                + float(getattr(self.config, "feasibility_aux_weight", 0.0)) * feasibility_aux
            )
        )
        if not torch.isfinite(total_loss):
            raise ValueError("DPSI/ASMI total loss is non-finite.")

        selected_real = awac_batch["selected_real_mask"]
        selected_rewards = awac_batch["selected_rewards"]
        denom = selected_real.float().sum().clamp(min=1.0)
        selected_reward_mean = (selected_rewards * selected_real.float()).sum() / denom
        zero = total_loss.new_zeros(())
        out = {
            "loss": total_loss,
            "diffusion_loss": dpsi_loss.detach(),
            "policy_loss": dpsi_loss,
            "awac_loss": dpsi_loss,
            "dpsi_loss": dpsi_loss,
            "dpsi_loss_weight_effective": total_loss.new_tensor(1.0).detach(),
            "dpsi_bc_loss": zero.detach(),
            "dpsi_pairwise_rank_loss": zero.detach(),
            "dpsi_invalid_repulsion_loss": zero.detach(),
            "dpsi_preference_dpo_loss": zero.detach(),
            "jepa_alignment_loss": zero.detach(),
            "vggt_alignment_loss": zero.detach(),
            "reward": selected_reward_mean.detach(),
            "reward_mean": selected_reward_mean.detach(),
            "gt_reward_mean": awac_batch["gt_reward"].mean().to(total_loss).detach(),
            "il_reward_mean": awac_batch["il_reward"].mean().to(total_loss).detach(),
            "best_selected_reward_mean": awac_batch["best_selected_reward"].mean().to(total_loss).detach(),
            "x0_aux_loss": x0_aux.detach(),
            "delta_aux_loss": delta_aux.detach(),
            "geo_aux_loss": geo_aux.detach(),
            "trajectory_aux_loss": trajectory_aux.detach(),
            "feasibility_aux_loss": feasibility_aux.detach(),
            "tangent_excess_loss": dpsi_diag["tangent_excess_loss"].detach(),
            "curvature_excess_loss": dpsi_diag["curvature_excess_loss"].detach(),
            "aux_alpha_weight_mean": dpsi_diag["aux_alpha_weight_mean"].detach(),
            "aux_warmup_ramp": aux_ramp.detach(),
            "selected_target_is_gt_ratio": dpsi_diag["selected_target_is_gt_ratio"].detach(),
            "selected_target_source_code": dpsi_diag["selected_target_source_code"].detach(),
            "residual_alpha": dpsi_diag["residual_alpha"].detach(),
            "full_x0_reconstruction_l1": dpsi_diag["full_x0_reconstruction_l1"].detach(),
            "fs_target_abs_gt3_ratio": dpsi_diag["fs_target_abs_gt3_ratio"].detach(),
            "fs_target_abs_gt5_ratio": dpsi_diag["fs_target_abs_gt5_ratio"].detach(),
            "fs_output_bound_hit_ratio": dpsi_diag["fs_output_bound_hit_ratio"].detach(),
            "early_kink_rate": dpsi_diag["early_kink_rate"].detach(),
            "tail_reverse_rate": dpsi_diag["tail_reverse_rate"].detach(),
            "curvature_violation_rate": dpsi_diag["curvature_violation_rate"].detach(),
            "dpsi_enabled": total_loss.new_tensor(1.0).detach(),
            "dpsi_per_sample_loss_mean": dpsi_diag["per_sample_loss_mean"].detach(),
            "dpsi_target_norm_mean": dpsi_diag["target_norm_mean"].detach(),
            "dpsi_timestep_mean": dpsi_diag["diffusion_timestep_mean"].detach(),
            "dpsi_timestep_min": dpsi_diag["diffusion_timestep_min"].detach(),
            "dpsi_timestep_max": dpsi_diag["diffusion_timestep_max"].detach(),
            "dpsi_effective_weight_sum": dpsi_diag["effective_weight_sum"].detach(),
            "dpsi_zero_weight_ratio": dpsi_diag["zero_weight_ratio"].detach(),
            "dpsi_sampled_weight_sum_mean": target_weights.sum(dim=1).mean().detach(),
            "dpsi_sampled_real_ratio": target_mask.float().mean().detach(),
            "valid_candidate_ratio": awac_batch["valid_candidate_ratio"].to(total_loss).detach(),
            "selected_valid_ratio": (
                ((awac_batch["selected_valid_mask"] & selected_real).float().sum())
                / selected_real.float().sum().clamp(min=1.0)
            ).to(total_loss).detach(),
        }
        for key in (
            "planning_token_norm",
            "planning_token_pairwise_cosine",
            "planning_condition_keep_ratio",
            "planning_context_gate",
            "planning_layer_gate_mean",
            "planning_delta_norm",
            "planning_adapter_forward_count",
        ):
            value = dpsi_diag.get(key)
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                out[key] = value.to(total_loss).detach()
        for diag in (profile_diag, asmi_diag, sample_diag):
            out.update({key: value.to(total_loss).detach() for key, value in diag.items()})
        return BatchFeature(data=out)

    def _action_input_index_select(
        self,
        action_input: Optional[BatchFeature],
        indices: torch.Tensor,
        expected_batch: int,
    ) -> Optional[BatchFeature]:
        if action_input is None:
            return None
        data: Dict[str, Any] = {}
        for key, value in action_input.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == expected_batch:
                data[key] = value.index_select(0, indices.to(device=value.device))
            else:
                data[key] = value
        return BatchFeature(data=data)

    def collect_grpo_replay_rollout(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        tokens_list,
        sample_time: Optional[int] = None,
    ) -> BatchFeature:
        """Collects a fixed rollout for PPO-style Stage3 replay updates."""
        self.set_frozen_modules_to_eval_mode()
        B = vl_features.shape[0]
        G = int(sample_time if sample_time is not None else getattr(self, "grpo_sample_time", 8))
        if G <= 0:
            raise ValueError("GRPO replay sample_time must be positive.")

        vl_features_rep = vl_features.detach().repeat_interleave(G, 0)
        his_traj_rep = action_input.his_traj.detach().repeat_interleave(G, 0)
        status_feature_rep = action_input.status_feature.detach().repeat_interleave(G, 0)
        expert_action_input_rep = self._detach_action_input(self._repeat_expert_action_input(action_input, G))

        rollout_policy = self
        sampled_from_behavior_policy = False
        behavior_policy_synced = False
        if bool(getattr(self, "use_gspo_ratio", False)) and bool(getattr(self, "behavior_policy_sample", True)):
            if not hasattr(self, "behavior_policy"):
                raise RuntimeError("grpo_replay with use_gspo_ratio=True requires an initialized behavior_policy.")
            if bool(getattr(self, "ppo_replay_sync_behavior_each_batch", True)):
                self._sync_behavior_policy()
                behavior_policy_synced = True
            rollout_policy = self.behavior_policy
            rollout_policy.eval()
            sampled_from_behavior_policy = True

        with torch.no_grad():
            chains, trajs = rollout_policy.sample_chain(
                vl_features_rep,
                his_traj_rep,
                status_feature_rep,
                deterministic=False,
                action_input=expert_action_input_rep,
            )
            num_denoising_steps = chains.shape[1] - 1
            discount = self._stage3_discount(
                num_denoising_steps,
                device=chains.device,
                dtype=chains.dtype,
            )
            old_log_probs = rollout_policy.get_logprobs(
                vl_features_rep,
                his_traj_rep,
                status_feature_rep,
                chains,
                deterministic=False,
                action_input=expert_action_input_rep,
            )
            old_traj_logp = rollout_policy._reduce_chain_logprobs(
                old_log_probs,
                B,
                G,
                num_denoising_steps,
                discount,
            )
            old_step_logp = rollout_policy._chain_step_logprobs(
                old_log_probs,
                B,
                G,
                num_denoising_steps,
            )

        tokens_rep = [tok for tok in tokens_list for _ in range(G)]
        metric_cache = {}
        for token in set(tokens_list):
            path = self.metric_cache_loader.metric_cache_paths[token]
            with lzma.open(path, 'rb') as f:
                metric_cache[token] = pickle.load(f)

        base_rewards, components = self.reward_fn(
            trajs,
            tokens_rep,
            metric_cache,
            return_components=True,
            **self._stage3_reward_kwargs(),
        )
        assert base_rewards.shape == (B * G,)
        rewards, hard_safe_mask, reward_aux = self._compose_stage3_reward(
            base_rewards,
            components,
            trajs,
            B,
            G,
        )
        rewards_matrix = rewards.view(B, G)
        hard_safe_matrix = hard_safe_mask.view(B, G)
        advantages, group_weight, advantage_aux = self._compute_stage3_advantages(
            rewards_matrix,
            hard_safe_matrix,
            reward_aux,
        )

        adv = advantages * group_weight.repeat_interleave(G)
        adv_before = adv.detach().float()
        if bool(getattr(self, "normalize_advantage_batch", False)):
            adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp(min=1e-6)
        advantage_clip_abs = float(getattr(self, "advantage_clip_abs", 0.0))
        advantage_clip_frac = adv.new_zeros(())
        if advantage_clip_abs > 0.0:
            clip_mask = adv.detach().abs() > advantage_clip_abs
            advantage_clip_frac = clip_mask.float().mean().to(adv)
            adv = adv.clamp(min=-advantage_clip_abs, max=advantage_clip_abs)
        adv_after = adv.detach().float()

        replay_mask = torch.ones_like(adv, dtype=torch.bool)
        if bool(getattr(self, "ppo_replay_filter_zero_advantage", True)):
            replay_mask &= adv_after.abs() > float(getattr(self, "ppo_replay_min_abs_advantage", 1e-6))

        bc_chains = None
        if bool(getattr(self, "ppo_replay_bc_update", True)):
            self.old_policy.eval()
            with torch.no_grad():
                bc_chains, _ = self.old_policy.sample_chain(
                    vl_features.detach(),
                    action_input.his_traj.detach(),
                    action_input.status_feature.detach(),
                    deterministic=False,
                    action_input=self._detach_action_input(action_input),
                )

        zero = rewards.new_zeros(())

        def aux_mean(name: str) -> torch.Tensor:
            value = reward_aux.get(name)
            return value.mean() if isinstance(value, torch.Tensor) else zero

        return BatchFeature(data={
            "replay_rollout": True,
            "B": B,
            "G": G,
            "num_samples": B * G,
            "num_denoising_steps": num_denoising_steps,
            "vl_features": vl_features_rep.detach(),
            "his_traj": his_traj_rep.detach(),
            "status_feature": status_feature_rep.detach(),
            "action_input": expert_action_input_rep,
            "chains": chains.detach(),
            "trajs": trajs.detach(),
            "old_traj_logp": old_traj_logp.detach(),
            "old_step_logp": old_step_logp.detach(),
            "advantages": adv.detach(),
            "raw_advantages": advantages.detach(),
            "replay_mask": replay_mask.detach(),
            "rewards": rewards.detach(),
            "base_rewards": base_rewards.detach(),
            "hard_safe_mask": hard_safe_mask.detach(),
            "group_weight": group_weight.detach(),
            "discount": discount.detach(),
            "bc_vl_features": vl_features.detach(),
            "bc_his_traj": action_input.his_traj.detach(),
            "bc_status_feature": action_input.status_feature.detach(),
            "bc_action_input": self._detach_action_input(action_input),
            "bc_chains": bc_chains.detach() if isinstance(bc_chains, torch.Tensor) else None,
            "loss": zero,
            "reward": rewards.mean().detach(),
            "base_reward": base_rewards.mean().detach(),
            "shaped_reward": rewards.mean().detach(),
            "safe_ratio": hard_safe_mask.detach().float().mean().to(dtype=rewards.dtype),
            "hard_safe_ratio": hard_safe_mask.detach().float().mean().to(dtype=rewards.dtype),
            "mean_ep": aux_mean("ego_progress").detach(),
            "mean_nc": aux_mean("no_at_fault_collisions").detach(),
            "mean_dac": aux_mean("drivable_area_compliance").detach(),
            "mean_ttc": aux_mean("time_to_collision_within_bound").detach(),
            "mean_comfort": aux_mean("history_comfort").detach(),
            "mean_ddc": aux_mean("driving_direction_compliance").detach(),
            "mean_tlc": aux_mean("traffic_light_compliance").detach(),
            "soft_safety_penalty_mean": aux_mean("soft_safety_penalty").detach(),
            "soft_safety_penalty_max": reward_aux["soft_safety_penalty"].detach().max(),
            "soft_safety_penalty_weight": rewards.new_tensor(float(self.soft_safety_penalty_weight)).detach(),
            "soft_safety_mode_enabled": rewards.new_tensor(
                float(str(self.safety_advantage_mode) == "soft_penalty")
            ).detach(),
            "group_reward_std": advantage_aux["reward_std"].mean().detach(),
            "safe_count_mean": advantage_aux["safe_count"].mean().detach(),
            "group_weight_mean": group_weight.mean().detach(),
            "group_weight_min": group_weight.min().detach(),
            "group_weight_max": group_weight.max().detach(),
            "mixed_group_ratio": advantage_aux["mixed_group_ratio"].detach(),
            "all_safe_group_ratio": advantage_aux["all_safe_group_ratio"].detach(),
            "all_unsafe_group_ratio": advantage_aux["all_unsafe_group_ratio"].detach(),
            "safe_rpp_advantage_enabled": advantage_aux.get(
                "safe_rpp_advantage_enabled",
                rewards.new_tensor(0.0),
            ).detach(),
            "safe_rpp_centered_batch_std": advantage_aux.get(
                "safe_rpp_centered_batch_std",
                rewards.new_tensor(0.0),
            ).detach(),
            "mean_advantage": advantages.mean().detach(),
            "mean_abs_advantage": advantages.abs().mean().detach(),
            "grpo_advantage_mean_before_transform": adv_before.mean().detach(),
            "grpo_advantage_std_before_transform": adv_before.std(unbiased=False).detach(),
            "grpo_advantage_mean_after_transform": adv_after.mean().detach(),
            "grpo_advantage_std_after_transform": adv_after.std(unbiased=False).detach(),
            "grpo_advantage_min": adv_after.min().detach(),
            "grpo_advantage_max": adv_after.max().detach(),
            "grpo_advantage_positive_ratio": (adv_after > 0.0).float().mean().detach(),
            "grpo_advantage_zero_ratio": (adv_after.abs() <= 1e-8).float().mean().detach(),
            "grpo_advantage_clip_frac": advantage_clip_frac.detach(),
            "grpo_advantage_batch_normalized": rewards.new_tensor(
                float(bool(getattr(self, "normalize_advantage_batch", False)))
            ),
            "grpo_advantage_clip_abs": rewards.new_tensor(float(advantage_clip_abs)),
            **self._stage3_core_pareto_log_metrics(rewards, reward_aux, advantage_aux),
            **self._stage3_feasible_pareto_log_metrics(rewards, advantage_aux),
            "ppo_replay_valid_ratio": replay_mask.detach().float().mean().to(dtype=rewards.dtype),
            "ppo_replay_valid_count": rewards.new_tensor(float(replay_mask.detach().sum().item())),
            "ppo_replay_inner_epochs": rewards.new_tensor(float(getattr(self, "ppo_replay_inner_epochs", 1))),
            "ppo_replay_minibatch_size": rewards.new_tensor(float(getattr(self, "ppo_replay_minibatch_size", 0))),
            "ppo_replay_step_logprob_mode": rewards.new_tensor(
                float(str(getattr(self, "ppo_replay_logprob_mode", "trajectory")) == "step")
            ),
            "ppo_replay_step_minibatch_mode": rewards.new_tensor(
                float(str(getattr(self, "ppo_replay_step_minibatch_mode", "trajectory_all_steps")) == "transition")
            ),
            "use_gspo_ratio": rewards.new_tensor(float(bool(getattr(self, "use_gspo_ratio", False)))),
            "sampled_from_behavior_policy": rewards.new_tensor(float(bool(sampled_from_behavior_policy))),
            "behavior_policy_synced": rewards.new_tensor(float(bool(behavior_policy_synced))),
            "behavior_policy_sync_interval": rewards.new_tensor(float(getattr(self, "behavior_policy_sync_interval", 0))),
        })

    def _ppo_replay_step_clip_width(
        self,
        step_indices: torch.Tensor,
        num_denoising_steps: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Returns DPPO-style per-denoising-step PPO clip widths."""
        final_width = max(float(self.gspo_clip_low), float(self.gspo_clip_high))
        if str(getattr(self, "ppo_replay_step_clip_schedule", "constant")) != "dppo_exp":
            return torch.full_like(step_indices, final_width, device=device, dtype=dtype)

        base = float(getattr(self, "ppo_replay_step_clip_base", 0.001))
        rate = float(getattr(self, "ppo_replay_step_clip_rate", 3.0))
        step_f = step_indices.to(device=device, dtype=dtype)
        if num_denoising_steps <= 1:
            progress = torch.zeros_like(step_f)
        else:
            progress = step_f / float(num_denoising_steps - 1)
        if rate <= 1e-8:
            return base + (final_width - base) * progress
        denom = math.exp(rate) - 1.0
        scaled = torch.expm1(progress * rate) / max(denom, 1e-8)
        return base + (final_width - base) * scaled

    def compute_grpo_replay_loss(
        self,
        rollout: BatchFeature,
        indices: torch.Tensor,
        denoising_step_indices: Optional[torch.Tensor] = None,
    ) -> BatchFeature:
        """Computes clipped PPO loss on fixed rollout rows."""
        full_n = int(rollout["num_samples"])
        indices = indices.to(device=rollout["chains"].device, dtype=torch.long)
        chains = rollout["chains"].index_select(0, indices)
        vl_features = rollout["vl_features"].index_select(0, indices)
        his_traj = rollout["his_traj"].index_select(0, indices)
        status_feature = rollout["status_feature"].index_select(0, indices)
        action_input = self._action_input_index_select(rollout.get("action_input"), indices, full_n)

        num_denoising_steps = int(rollout["num_denoising_steps"])
        discount = rollout["discount"].to(device=chains.device, dtype=chains.dtype)
        new_log_probs = self.get_logprobs(
            vl_features,
            his_traj,
            status_feature,
            chains,
            deterministic=False,
            action_input=action_input,
        )
        logprob_mode = str(getattr(self, "ppo_replay_logprob_mode", "trajectory"))
        transition_mode = False
        logprob_clamped_frac = chains.new_zeros(())
        new_logprob_clamped_frac = chains.new_zeros(())
        old_logprob_clamped_frac = chains.new_zeros(())
        selected_step_indices_for_logging = None
        if logprob_mode == "step":
            new_step_logp_raw = self._chain_step_logprobs(new_log_probs, indices.numel(), 1, num_denoising_steps)
            old_step_logp_raw = rollout["old_step_logp"].index_select(0, indices).to(new_step_logp_raw)
            clamp_min = float(getattr(self, "ppo_replay_logprob_clamp_min", -5.0))
            clamp_max = float(getattr(self, "ppo_replay_logprob_clamp_max", 2.0))
            new_clamped = (new_step_logp_raw < clamp_min) | (new_step_logp_raw > clamp_max)
            old_clamped = (old_step_logp_raw < clamp_min) | (old_step_logp_raw > clamp_max)
            new_logprob_clamped_frac = new_clamped.detach().float().mean().to(chains)
            old_logprob_clamped_frac = old_clamped.detach().float().mean().to(chains)
            logprob_clamped_frac = (new_clamped | old_clamped).detach().float().mean().to(chains)
            new_step_logp = new_step_logp_raw.clamp(min=clamp_min, max=clamp_max)
            old_step_logp = old_step_logp_raw.clamp(min=clamp_min, max=clamp_max)
            advantages = rollout["advantages"].index_select(0, indices).to(new_step_logp)
            replay_mask = rollout["replay_mask"].index_select(0, indices).to(device=new_step_logp.device)
            discount_norm = discount.to(device=new_step_logp.device, dtype=new_step_logp.dtype)
            discount_norm = discount_norm / discount_norm.sum().clamp(min=1e-8)
            if denoising_step_indices is not None:
                transition_mode = True
                step_idx = denoising_step_indices.to(device=new_step_logp.device, dtype=torch.long)
                if step_idx.shape != indices.shape:
                    raise ValueError(
                        "denoising_step_indices must have the same shape as indices in transition replay mode."
                    )
                if step_idx.numel() > 0 and (
                    int(step_idx.min().item()) < 0 or int(step_idx.max().item()) >= num_denoising_steps
                ):
                    raise ValueError("denoising_step_indices contains an out-of-range denoising step.")
                row_idx = torch.arange(indices.numel(), device=new_step_logp.device)
                log_ratio = new_step_logp[row_idx, step_idx] - old_step_logp[row_idx, step_idx]
                advantage_term = advantages
                valid_f = replay_mask.to(dtype=log_ratio.dtype)
                step_weight = discount_norm.index_select(0, step_idx) * float(num_denoising_steps)
                selected_step_indices_for_logging = step_idx
            else:
                log_ratio = new_step_logp - old_step_logp
                advantage_term = advantages.view(-1, 1)
                valid_f = replay_mask.to(dtype=log_ratio.dtype).view(-1, 1)
                step_weight = discount_norm.view(1, num_denoising_steps)
                selected_step_indices_for_logging = torch.arange(
                    num_denoising_steps,
                    device=new_step_logp.device,
                    dtype=torch.long,
                )
            trajectory_logp_for_logging = (new_step_logp * discount_norm.view(1, num_denoising_steps)).sum(dim=1)
        elif logprob_mode == "trajectory":
            if denoising_step_indices is not None:
                raise ValueError("denoising_step_indices is only supported with ppo_replay_logprob_mode='step'.")
            new_traj_logp = self._reduce_chain_logprobs(
                new_log_probs,
                indices.numel(),
                1,
                num_denoising_steps,
                discount,
            )
            old_traj_logp = rollout["old_traj_logp"].index_select(0, indices).to(new_traj_logp)
            advantages = rollout["advantages"].index_select(0, indices).to(new_traj_logp)
            replay_mask = rollout["replay_mask"].index_select(0, indices).to(device=new_traj_logp.device)
            log_ratio = new_traj_logp - old_traj_logp
            advantage_term = advantages
            valid_f = replay_mask.to(dtype=log_ratio.dtype)
            step_weight = None
            trajectory_logp_for_logging = new_traj_logp
        else:
            raise ValueError(f"Unsupported ppo_replay_logprob_mode: {logprob_mode!r}")

        ratio = torch.exp(log_ratio.clamp(min=-20.0, max=20.0))
        step_clip_values = ratio.detach().new_full(ratio.shape, max(float(self.gspo_clip_low), float(self.gspo_clip_high)))
        if logprob_mode == "step" and str(getattr(self, "ppo_replay_step_clip_schedule", "constant")) == "dppo_exp":
            if transition_mode:
                clip_width = self._ppo_replay_step_clip_width(
                    selected_step_indices_for_logging,
                    num_denoising_steps,
                    dtype=ratio.dtype,
                    device=ratio.device,
                )
            else:
                step_ids = torch.arange(num_denoising_steps, device=ratio.device, dtype=torch.long)
                clip_width = self._ppo_replay_step_clip_width(
                    step_ids,
                    num_denoising_steps,
                    dtype=ratio.dtype,
                    device=ratio.device,
                ).view(1, num_denoising_steps)
            ratio_clip_low = 1.0 - clip_width
            ratio_clip_high = 1.0 + clip_width
            clipped_ratio = torch.minimum(torch.maximum(ratio, ratio_clip_low), ratio_clip_high)
            step_clip_values = clip_width.detach().expand_as(ratio).float()
        else:
            ratio_clip_low = 1.0 - float(self.gspo_clip_low)
            ratio_clip_high = 1.0 + float(self.gspo_clip_high)
            clipped_ratio = ratio.clamp(ratio_clip_low, ratio_clip_high)
        surrogate = torch.minimum(ratio * advantage_term, clipped_ratio * advantage_term)
        if step_weight is not None:
            surrogate = surrogate * step_weight
        valid_count = valid_f.sum()
        if valid_count.detach().item() > 0:
            policy_loss = -(surrogate * valid_f).sum() / valid_count.clamp(min=1.0)
        else:
            policy_loss = (log_ratio * 0.0).sum()

        reference_kl_coeff = float(getattr(self, "reference_kl_coeff", 0.0))
        reference_kl_loss = policy_loss.new_zeros(())
        total_loss = policy_loss
        if reference_kl_coeff > 0.0:
            reference_kl_loss = self._chain_transition_reference_kl(
                vl_features,
                his_traj,
                status_feature,
                chains,
                action_input,
                indices.numel(),
                1,
                num_denoising_steps,
                discount,
            ).to(dtype=policy_loss.dtype)
            total_loss = total_loss + reference_kl_coeff * reference_kl_loss

        ratio_detached = ratio.detach().float()
        log_ratio_detached = log_ratio.detach().float()
        approx_kl_values = ((ratio - 1.0) - log_ratio).detach().float()
        if step_weight is not None:
            valid_float = valid_f.detach().float()
            approx_kl = (approx_kl_values * step_weight.float() * valid_float).sum()
            approx_kl = approx_kl / valid_count.detach().float().clamp(min=1.0)
            reverse_approx_kl = (-log_ratio_detached * step_weight.float() * valid_float).sum()
            reverse_approx_kl = reverse_approx_kl / valid_count.detach().float().clamp(min=1.0)
        else:
            approx_kl = approx_kl_values.mean()
            reverse_approx_kl = (-log_ratio_detached).mean()
        approx_kl = approx_kl.to(total_loss)
        if selected_step_indices_for_logging is None:
            selected_step_stats = total_loss.new_zeros((1,))
        elif transition_mode:
            step_stats_mask = replay_mask.detach().bool()
            if step_stats_mask.any():
                selected_step_stats = selected_step_indices_for_logging.detach()[step_stats_mask].to(total_loss)
            else:
                selected_step_stats = selected_step_indices_for_logging.detach().to(total_loss)
        else:
            selected_step_stats = selected_step_indices_for_logging.detach().to(total_loss)
        selected_transition_count = valid_count
        if logprob_mode == "step" and not transition_mode:
            selected_transition_count = valid_count * float(num_denoising_steps)
        elif logprob_mode != "step":
            selected_transition_count = valid_count.new_zeros(())
        if torch.is_tensor(ratio_clip_low):
            clip_mask = (ratio < ratio_clip_low) | (ratio > ratio_clip_high)
        else:
            clip_mask = (ratio < ratio_clip_low) | (ratio > ratio_clip_high)
        return BatchFeature(data={
            "loss": total_loss,
            "policy_loss": policy_loss.detach(),
            "reference_kl_loss": reference_kl_loss.detach(),
            "reference_kl_coeff": total_loss.new_tensor(reference_kl_coeff),
            "bc_loss": total_loss.new_zeros(()),
            "bc_coeff": total_loss.new_tensor(self._current_bc_coeff()),
            "reward": rollout["reward"].to(total_loss),
            "ppo_replay_minibatch_valid_ratio": valid_f.detach().float().mean().to(total_loss),
            "ppo_replay_minibatch_valid_count": total_loss.new_tensor(float(valid_count.detach().item())),
            "ppo_replay_loss_active": total_loss.new_tensor(float(valid_count.detach().item() > 0)),
            "gspo_ratio_mean": ratio_detached.mean().to(total_loss),
            "gspo_ratio_min": ratio_detached.min().to(total_loss),
            "gspo_ratio_max": ratio_detached.max().to(total_loss),
            "gspo_ratio_clip_frac": clip_mask.detach().float().mean().to(total_loss),
            "gspo_log_ratio_mean": log_ratio_detached.mean().to(total_loss),
            "gspo_log_ratio_std": log_ratio_detached.std(unbiased=False).to(total_loss),
            "gspo_log_ratio_min": log_ratio_detached.min().to(total_loss),
            "gspo_log_ratio_max": log_ratio_detached.max().to(total_loss),
            "gspo_abs_log_ratio_mean": log_ratio_detached.abs().mean().to(total_loss),
            "gspo_approx_kl": approx_kl,
            "gspo_reverse_approx_kl": reverse_approx_kl.to(total_loss),
            "trajectory_logp": trajectory_logp_for_logging.detach().mean().to(total_loss),
            "ppo_replay_step_logprob_mode": total_loss.new_tensor(float(logprob_mode == "step")),
            "ppo_replay_step_minibatch_mode": total_loss.new_tensor(float(transition_mode)),
            "ppo_replay_transition_mode": total_loss.new_tensor(float(transition_mode)),
            "ppo_replay_selected_transition_count": total_loss.new_tensor(float(selected_transition_count.detach().item())),
            "ppo_replay_selected_denoising_step_mean": selected_step_stats.float().mean().to(total_loss),
            "ppo_replay_selected_denoising_step_min": selected_step_stats.float().min().to(total_loss),
            "ppo_replay_selected_denoising_step_max": selected_step_stats.float().max().to(total_loss),
            "ppo_replay_step_clip_mean": step_clip_values.mean().to(total_loss),
            "ppo_replay_step_clip_min": step_clip_values.min().to(total_loss),
            "ppo_replay_step_clip_max": step_clip_values.max().to(total_loss),
            "ppo_replay_logprob_clamped_frac": logprob_clamped_frac.to(total_loss),
            "ppo_replay_new_logprob_clamped_frac": new_logprob_clamped_frac.to(total_loss),
            "ppo_replay_old_logprob_clamped_frac": old_logprob_clamped_frac.to(total_loss),
        })

    def compute_grpo_replay_bc_loss(self, rollout: BatchFeature) -> BatchFeature:
        bc_chains = rollout.get("bc_chains")
        if not isinstance(bc_chains, torch.Tensor):
            zero = rollout["reward"].new_zeros(())
            return BatchFeature(data={"loss": zero, "bc_loss": zero, "bc_coeff": zero})
        bc_logp = self.get_logprobs(
            rollout["bc_vl_features"],
            rollout["bc_his_traj"],
            rollout["bc_status_feature"],
            bc_chains,
            deterministic=False,
            action_input=rollout.get("bc_action_input"),
        )
        K_steps = bc_chains.shape[1] - 1
        bc_logp = bc_logp.clamp(min=-5, max=2)
        bc_logp = bc_logp.view(-1, K_steps, bc_chains.shape[2], bc_chains.shape[3]).mean(dim=[1, 2, 3])
        bc_loss = -bc_logp.mean()
        bc_coeff = self._current_bc_coeff()
        return BatchFeature(data={
            "loss": bc_loss * float(bc_coeff),
            "bc_loss": bc_loss.detach(),
            "bc_coeff": bc_loss.new_tensor(float(bc_coeff)),
        })

    def _accumulate_lfp_epoch_energy(
        self,
        tokens_list: list[str],
        energy: torch.Tensor,
    ) -> None:
        if energy.ndim != 1 or energy.shape[0] != len(tokens_list):
            raise ValueError("LFP frontier energy must have one value per scene token.")
        for token, value in zip(tokens_list, energy.detach().float().cpu().tolist()):
            if not math.isfinite(float(value)):
                raise ValueError(f"Non-finite LFP frontier energy for token {token!r}.")
            stats = self._lfp_epoch_energy.setdefault(str(token), [0.0, 0.0])
            stats[0] += float(value)
            stats[1] += 1.0

    def consume_lfp_epoch_energy(self) -> Dict[str, tuple[float, int]]:
        output = {
            token: (float(values[0]), int(values[1]))
            for token, values in self._lfp_epoch_energy.items()
        }
        self._lfp_epoch_energy.clear()
        return output

    def lfp_runtime_state_dict(self) -> Dict[str, Any]:
        if self.stage3_algorithm != "lfp_grpo" or self.lfp_reference_cache is None:
            return {}
        return {
            "version": 1,
            "benchmark": str(self.lfp_grpo_cfg.benchmark),
            "reference_cache_metadata_hash": self.lfp_reference_cache.metadata_hash,
            "reference_policy_checkpoint_sha256": self.lfp_reference_policy_checkpoint_sha256,
            "diversity_capacity_cache_metadata_hash": (
                self.lfp_diversity_capacity_cache.metadata_hash
                if self.lfp_diversity_capacity_cache is not None
                else ""
            ),
            "epoch_energy": {
                token: (float(values[0]), int(values[1]))
                for token, values in self._lfp_epoch_energy.items()
            },
        }

    def load_lfp_runtime_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not state_dict:
            return
        if self.stage3_algorithm != "lfp_grpo" or self.lfp_reference_cache is None:
            raise ValueError("Checkpoint contains LFP runtime state but current algorithm is not LFP-GRPO.")
        if int(state_dict.get("version", 0)) != 1:
            raise ValueError("Unsupported LFP runtime checkpoint version.")
        if str(state_dict.get("benchmark")) != str(self.lfp_grpo_cfg.benchmark):
            raise ValueError("LFP benchmark changed across checkpoint resume.")
        if str(state_dict.get("reference_cache_metadata_hash")) != self.lfp_reference_cache.metadata_hash:
            raise ValueError("LFP reference cache metadata changed across checkpoint resume.")
        if str(state_dict.get("reference_policy_checkpoint_sha256", "")) != str(
            self.lfp_reference_policy_checkpoint_sha256
        ):
            raise ValueError("Frozen LFP Stage2 reference checkpoint changed across resume.")
        expected_diversity_hash = (
            self.lfp_diversity_capacity_cache.metadata_hash
            if self.lfp_diversity_capacity_cache is not None
            else ""
        )
        if str(state_dict.get("diversity_capacity_cache_metadata_hash", "")) != str(
            expected_diversity_hash
        ):
            raise ValueError("LFP diversity-capacity cache changed across checkpoint resume.")
        self._lfp_epoch_energy = {
            str(token): [float(values[0]), float(values[1])]
            for token, values in dict(state_dict.get("epoch_energy", {})).items()
        }

    def _evaluate_lfp_rollouts(
        self,
        trajectories: torch.Tensor,
        tokens_rep: list[str],
        metric_cache: Dict[str, Any],
        B: int,
        G: int,
    ):
        if self.lfp_metric_adapter is None:
            raise RuntimeError("LFP metric adapter is not initialized.")
        if self.lfp_grpo_cfg.benchmark == "navsim_v2":
            evaluator = self.lfp_v2_rollout_evaluator
            if evaluator is None:
                raise RuntimeError(
                    "NAVSIM v2 LFP training requires an official one-stage EPDMS rollout evaluator. "
                    "The local NAVSIM v1 pdm_score backend cannot supply two_frame_extended_comfort/TLC "
                    "and must not be used as a silent approximation."
                )
            if self.lfp_reference_cache is None:
                raise RuntimeError("NAVSIM v2 LFP scoring requires the coherent reference cache.")
            components = evaluator.score(trajectories, tokens_rep, self.lfp_reference_cache)
            if not isinstance(components, dict):
                raise TypeError("lfp_v2_rollout_evaluator must return a metric component dict.")
        else:
            _, components = self.reward_fn(
                trajectories,
                tokens_rep,
                metric_cache,
                return_components=True,
                strict_submetrics=True,
                required_submetrics=(
                    "pdms",
                    "no_at_fault_collisions",
                    "drivable_area_compliance",
                    "time_to_collision_within_bound",
                    "ego_progress",
                    "history_comfort",
                    "driving_direction_compliance",
                ),
                missing_submetric_policy="error",
                use_batched_pdm_scoring=True,
                use_exact_array_pdm_state_conversion=True,
                use_fast_pdm_scorer=True,
            )
        return self.lfp_metric_adapter.canonicalize(
            components,
            batch_size=B,
            group_size=G,
        )

    def forward_lfp_grpo(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        tokens_list,
        sample_time: Optional[int] = None,
    ) -> BatchFeature:
        """Runs one on-policy trajectory-level LFP-GRPO update."""
        if self.stage3_algorithm != "lfp_grpo":
            raise RuntimeError("forward_lfp_grpo requires stage3_algorithm='lfp_grpo'.")
        if self.lfp_reference_cache is None or self.lfp_metric_adapter is None:
            raise RuntimeError("LFP reference cache/metric adapter is not initialized.")
        self.set_frozen_modules_to_eval_mode()
        B = int(vl_features.shape[0])
        G = int(sample_time if sample_time is not None else self.grpo_sample_time)
        if G <= 0:
            raise ValueError("LFP-GRPO group size must be positive.")
        tokens = [str(token) for token in tokens_list]
        if len(tokens) != B:
            raise ValueError(f"Expected {B} scene tokens, got {len(tokens)}.")

        vl_features_rep = vl_features.repeat_interleave(G, 0)
        his_traj_rep = action_input.his_traj.repeat_interleave(G, 0)
        status_feature_rep = action_input.status_feature.repeat_interleave(G, 0)
        condition_input_rep = self._repeat_expert_action_input(action_input, G)

        self._reset_planning_adapter_forward_count()
        current_dit_context = self._prepare_dit_context(
            vl_features_rep,
            condition_input_rep,
            # Exact KL requires current and frozen policies to see the same condition.
            # DDIM noise remains the sole on-policy exploration source in LFP.
            training=False,
            allow_target_tokens=False,
        )
        with torch.no_grad():
            chains, trajectories = self.sample_chain(
                vl_features_rep,
                his_traj_rep,
                status_feature_rep,
                deterministic=False,
                action_input=condition_input_rep,
                allow_target_tokens=False,
                prepared_dit_context=current_dit_context,
            )

        tokens_rep = [token for token in tokens for _ in range(G)]
        metric_cache: Dict[str, Any] = {}
        if self.lfp_grpo_cfg.benchmark == "navsim_v1":
            for token in set(tokens):
                try:
                    path = self.metric_cache_loader.metric_cache_paths[token]
                except KeyError as error:
                    raise KeyError(f"Stage3 metric cache is missing token {token!r}.") from error
                with lzma.open(path, "rb") as handle:
                    metric_cache[token] = pickle.load(handle)
        metrics = self._evaluate_lfp_rollouts(trajectories, tokens_rep, metric_cache, B, G)
        reference = self.lfp_reference_cache.get(tokens, metrics.scalar.device, metrics.scalar.dtype)
        group_spread = compute_group_trajectory_spread(
            trajectories.reshape(B, G, trajectories.shape[-2], trajectories.shape[-1])
        )
        advantage_output = compute_lfp_advantages(
            metrics,
            reference,
            self.lfp_grpo_cfg,
            group_pairwise_ade_m=group_spread.pairwise_ade_m,
        )
        frontier_energy = advantage_output.energy
        diversity_diagnostics: Dict[str, torch.Tensor] = {}
        if self.lfp_diversity_capacity_cache is not None:
            capacity = self.lfp_diversity_capacity_cache.get(
                tokens,
                metrics.scalar.device,
                metrics.scalar.dtype,
            )
            diversity_frontier = compute_capacity_normalized_frontier_energy(
                frontier_energy,
                advantage_output.feasible,
                group_spread.pairwise_ade_m,
                capacity,
                gap_weight=float(self.lfp_grpo_cfg.frontier_diversity_gap_weight),
                dispersion_floor=float(
                    self.lfp_grpo_cfg.frontier_diversity_dispersion_floor
                ),
            )
            frontier_energy = diversity_frontier.energy
            diversity_diagnostics = {
                "lfp_diversity_mode_capacity_mean": diversity_frontier.mode_capacity.mean().detach(),
                "lfp_diversity_coverage_ratio_mean": diversity_frontier.coverage_ratio.mean().detach(),
                "lfp_diversity_coverage_gap_mean": diversity_frontier.coverage_gap.mean().detach(),
                "lfp_diversity_frontier_bonus_mean": diversity_frontier.bonus.mean().detach(),
                "lfp_diversity_capacity_active_ratio": (
                    diversity_frontier.active_mask.float().mean().detach()
                ),
                "lfp_diversity_frontier_energy_mean": frontier_energy.mean().detach(),
            }
        if bool(self.lfp_grpo_cfg.curriculum_enabled):
            self._accumulate_lfp_epoch_energy(tokens, frontier_energy)

        num_denoising_steps = int(chains.shape[1] - 1)
        discount = self._stage3_discount(
            num_denoising_steps,
            device=chains.device,
            dtype=metrics.scalar.dtype,
        )
        log_probs = self.get_logprobs(
            vl_features_rep,
            his_traj_rep,
            status_feature_rep,
            chains,
            deterministic=False,
            action_input=condition_input_rep,
            prepared_dit_context=current_dit_context,
        )
        trajectory_logp = self._reduce_chain_logprobs(log_probs, B, G, num_denoising_steps, discount)
        policy_loss = trajectory_reinforce_loss(advantage_output.advantages, trajectory_logp)

        self.old_policy.eval()
        with torch.no_grad():
            reference_dit_context = self.old_policy._prepare_dit_context(
                vl_features_rep,
                condition_input_rep,
                training=False,
                allow_target_tokens=False,
            )
        exact_kl = self._chain_transition_reference_kl(
            vl_features_rep,
            his_traj_rep,
            status_feature_rep,
            chains,
            condition_input_rep,
            B,
            G,
            num_denoising_steps,
            discount,
            current_dit_context=current_dit_context,
            reference_dit_context=reference_dit_context,
        ).to(policy_loss)
        kl_coeff = float(self.lfp_grpo_cfg.reference_kl_coeff)
        total_loss = policy_loss + kl_coeff * exact_kl
        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"Non-finite LFP loss: policy={policy_loss.detach().item()}, kl={exact_kl.detach().item()}."
            )

        diagnostics = dict(advantage_output.diagnostics)
        diagnostics.update(diversity_diagnostics)
        diagnostics.update(group_spread.scalar_diagnostics())
        diagnostics.update(self._lfp_transition_floor_diagnostics(trajectories))
        planning_diagnostics = current_dit_context.get("diagnostics", {})
        for key in (
            "planning_token_norm",
            "planning_token_pairwise_cosine",
            "planning_condition_keep_ratio",
            "planning_context_gate",
            "planning_adapter_forward_count",
        ):
            value = planning_diagnostics.get(key)
            if isinstance(value, torch.Tensor):
                diagnostics[key] = value.detach().float().mean().to(total_loss)
        diagnostics.update(
            {
                "lfp_policy_loss": policy_loss.detach(),
                "lfp_exact_kl": exact_kl.detach(),
                "lfp_reference_kl_coeff": total_loss.new_tensor(kl_coeff),
                "lfp_trajectory_logprob_mean": trajectory_logp.detach().mean(),
                "scene_stage_type": total_loss.new_tensor(
                    float(
                        sum(
                            {"first": 1, "followup": 2}.get(
                                str(self.lfp_reference_cache.records[token].get("scene_stage_type", "unknown")),
                                0,
                            )
                            for token in tokens
                        )
                        / max(len(tokens), 1)
                    )
                ),
            }
        )
        zero = total_loss.new_zeros(())
        return BatchFeature(
            data={
                "loss": total_loss,
                "reward": metrics.scalar.mean().detach(),
                "policy_loss": policy_loss,
                "bc_loss": zero,
                "bc_coeff": zero,
                "reference_kl_loss": exact_kl,
                "reference_kl_coeff": total_loss.new_tensor(kl_coeff),
                "trajectory_logp": trajectory_logp.detach().mean(),
                **diagnostics,
            }
        )

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
        if self.stage3_algorithm == "lfp_grpo":
            return self.forward_lfp_grpo(
                vl_features,
                action_input,
                tokens_list,
                sample_time=sample_time,
            )
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
        behavior_policy_synced = False
        if self.use_trajectory_level_objective and self.use_gspo_ratio and self.behavior_policy_sample:
            if not hasattr(self, "behavior_policy"):
                raise RuntimeError("use_gspo_ratio=True requires a frozen behavior_policy initialized in _init_grpo.")
            sync_interval = max(1, int(self.behavior_policy_sync_interval))
            if int(getattr(self, "grpo_update_counter", 0)) % sync_interval == 0:
                self._sync_behavior_policy()
                behavior_policy_synced = True
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
        offline_cfg = getattr(self, "offline_rl_cfg", None)
        use_grpo_buffer_guidance = (
            offline_cfg is not None
            and bool(offline_cfg.enabled)
            and bool(offline_cfg.grpo_buffer_guidance_enabled)
        )
        grpo_buffer_guidance: Dict[str, Any] = {}
        if use_grpo_buffer_guidance:
            grpo_buffer_guidance = self._load_grpo_buffer_guidance_batch(
                action_input,
                [str(token) for token in tokens_list],
                offline_cfg,
            )

        reward_kwargs: Dict[str, Any] = {}
        if offline_cfg is not None and bool(offline_cfg.enabled):
            reward_kwargs = {
                "strict_submetrics": bool(offline_cfg.strict_reward_submetrics),
                "required_submetrics": offline_cfg.required_reward_submetrics,
                "missing_submetric_policy": str(offline_cfg.missing_submetric_policy),
                "use_batched_pdm_scoring": bool(offline_cfg.use_batched_pdm_scoring),
                "use_exact_array_pdm_state_conversion": bool(offline_cfg.use_exact_array_pdm_state_conversion),
                "use_fast_pdm_scorer": bool(offline_cfg.use_fast_pdm_scorer),
                "pdm_batch_chunk_size": int(offline_cfg.pdm_batch_chunk_size),
                "pdm_shadow_check": bool(offline_cfg.pdm_shadow_check),
                "pdm_shadow_max_samples": int(offline_cfg.pdm_shadow_max_samples),
                "pdm_shadow_max_abs_diff": float(offline_cfg.pdm_shadow_max_abs_diff),
            }
        base_rewards, components = self.reward_fn(
            trajs,
            tokens_rep,
            metric_cache,
            return_components=True,
            **reward_kwargs,
        )
        # rewards: [B * G]
        assert base_rewards.shape == (B * G,)
        components = dict(components)
        components["pdms"] = base_rewards
        grpo_buffer_reward_bonus = base_rewards.new_zeros((B * G,))
        grpo_buffer_diag = {
            "grpo_buffer_reward_bonus_mean": base_rewards.new_zeros(()),
            "grpo_buffer_reward_bonus_max": base_rewards.new_zeros(()),
            "grpo_buffer_target_distance_mean": base_rewards.new_zeros(()),
            "grpo_buffer_guidance_target_ratio": base_rewards.new_zeros(()),
        }
        grpo_buffer_bonus_weight = 0.0
        use_core_pareto = bool(getattr(self, "use_core_pareto_grpo", False))
        use_feasible_pareto = (
            bool(getattr(self, "use_feasible_pareto_grpo", False))
            or str(getattr(self, "reward_mode", "safe_diffgrpo")) == "feasible_pareto"
        )
        if use_core_pareto:
            ref = self._compute_core_pareto_reference_components(
                vl_features,
                action_input,
                [str(token) for token in tokens_list],
                metric_cache,
            )
            components_matrix = self._reshape_components_for_group(components, B, G)
            trajs_matrix = trajs.reshape(B, G, trajs.shape[1], trajs.shape[2])
            rewards_matrix = base_rewards.view(B, G)
            advantages, group_weight, advantage_aux = self._compute_core_pareto_advantages(
                rewards_matrix,
                components_matrix,
                trajs_matrix,
                ref,
            )
            hard_safe_matrix = advantage_aux["core_pareto_valid_mask"].to(device=base_rewards.device).bool()
            hard_safe_mask = hard_safe_matrix.reshape(B * G)
            rewards_matrix = advantage_aux["core_pareto_score"].to(device=base_rewards.device, dtype=base_rewards.dtype)
            rewards = rewards_matrix.reshape(B * G)
            if self.log_safe_diversity or self.use_diversity_reward:
                diversity_bonus = self._compute_safe_diversity_bonus(trajs, hard_safe_mask, base_rewards, B, G)
            else:
                diversity_bonus = torch.zeros_like(base_rewards)
            ep_floor_gap = (
                ref["ref_ep"].to(device=base_rewards.device, dtype=base_rewards.dtype)[:, None]
                - components_matrix["ego_progress"].to(base_rewards)
                - float(self.core_pareto_ep_floor_tolerance)
            ).clamp(min=0.0)
            zero = torch.zeros_like(base_rewards)
            reward_aux = {
                "pdms": base_rewards,
                "ego_progress": components["ego_progress"],
                "diversity_bonus": diversity_bonus,
                "soft_safety_penalty": zero,
                "core_pareto_mode_enabled": base_rewards.new_tensor(1.0),
                "core_pareto_core": self._compute_pdms_core(components_matrix).reshape(B * G).to(base_rewards),
                "core_pareto_pdms_formula": base_rewards,
                "core_pareto_adjusted_core": rewards,
                "core_pareto_ep_floor_gap": ep_floor_gap.reshape(B * G),
                "core_pareto_ep_floor_penalty": (
                    float(self.core_pareto_slow_penalty_weight) * ep_floor_gap
                ).reshape(B * G),
                "core_pareto_ddc_penalty": zero,
                "core_pareto_dual_penalty": zero,
                "core_pareto_nc_dac_feasible_mask": (
                    (
                        components_matrix["no_at_fault_collisions"] >= 1.0
                    )
                    & (components_matrix["drivable_area_compliance"] >= 1.0)
                ).reshape(B * G).to(dtype=base_rewards.dtype),
                "core_pareto_ddc_guard_mask": (
                    (
                        components_matrix["driving_direction_compliance"] >= float(self.core_pareto_ddc_min_absolute)
                    )
                    | (
                        components_matrix["driving_direction_compliance"]
                        >= ref["ref_ddc"].to(device=base_rewards.device)[:, None]
                        - float(self.core_pareto_ddc_drop_tolerance)
                    )
                ).reshape(B * G).to(dtype=base_rewards.dtype),
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
                    reward_aux[key] = components[key]
        elif use_feasible_pareto:
            ref = self._compute_core_pareto_reference_components(
                vl_features,
                action_input,
                [str(token) for token in tokens_list],
                metric_cache,
            )
            components_matrix = self._reshape_components_for_group(components, B, G)
            trajs_matrix = trajs.reshape(B, G, trajs.shape[1], trajs.shape[2])
            rewards_matrix = base_rewards.view(B, G)
            advantages, group_weight, advantage_aux = self._compute_feasible_pareto_advantages(
                rewards_matrix,
                components_matrix,
                trajs_matrix,
                ref,
                support_batch=grpo_buffer_guidance if use_grpo_buffer_guidance else None,
            )
            hard_safe_matrix = advantage_aux["feasible_pareto_valid_mask"].to(device=base_rewards.device).bool()
            hard_safe_mask = hard_safe_matrix.reshape(B * G)
            rewards_matrix = advantage_aux["feasible_pareto_score"].to(device=base_rewards.device, dtype=base_rewards.dtype)
            rewards = rewards_matrix.reshape(B * G)
            if self.log_safe_diversity or self.use_diversity_reward:
                diversity_bonus = self._compute_safe_diversity_bonus(trajs, hard_safe_mask, base_rewards, B, G)
            else:
                diversity_bonus = torch.zeros_like(base_rewards)
            zero = torch.zeros_like(base_rewards)
            reward_aux = {
                "pdms": base_rewards,
                "ego_progress": components["ego_progress"],
                "diversity_bonus": diversity_bonus,
                "soft_safety_penalty": zero,
                "core_pareto_mode_enabled": base_rewards.new_tensor(0.0),
                "core_pareto_core": zero,
                "core_pareto_pdms_formula": zero,
                "core_pareto_adjusted_core": zero,
                "core_pareto_ep_floor_gap": zero,
                "core_pareto_ep_floor_penalty": zero,
                "core_pareto_ddc_penalty": zero,
                "core_pareto_dual_penalty": zero,
                "core_pareto_nc_dac_feasible_mask": hard_safe_mask.to(dtype=base_rewards.dtype),
                "core_pareto_ddc_guard_mask": (
                    components_matrix["driving_direction_compliance"]
                    >= torch.maximum(
                        base_rewards.new_full((B, G), float(self.fp_ddc_min_absolute)),
                        ref["ref_ddc"].to(device=base_rewards.device)[:, None] - float(self.fp_ddc_ref_tolerance),
                    )
                ).reshape(B * G).to(dtype=base_rewards.dtype),
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
                    reward_aux[key] = components[key]
        else:
            rewards, hard_safe_mask, reward_aux = self._compose_stage3_reward(
                base_rewards,
                components,
                trajs,
                B,
                G,
            )
            if use_grpo_buffer_guidance:
                grpo_buffer_reward_bonus, grpo_buffer_diag = self._compute_grpo_buffer_reward_bonus(
                    trajs,
                    B,
                    G,
                    grpo_buffer_guidance,
                    offline_cfg,
                )
                grpo_buffer_bonus_weight = float(offline_cfg.grpo_buffer_reward_bonus_weight)
                if grpo_buffer_bonus_weight > 0.0:
                    rewards = rewards + torch.where(
                        hard_safe_mask,
                        grpo_buffer_bonus_weight * grpo_buffer_reward_bonus.to(rewards),
                        torch.zeros_like(rewards),
                    )
            rewards_matrix = rewards.view(B, G)
            hard_safe_matrix = hard_safe_mask.view(B, G)
            advantages, group_weight, advantage_aux = self._compute_stage3_advantages(
                rewards_matrix,
                hard_safe_matrix,
                reward_aux,
            )
        assert rewards.shape == (B * G,)
        assert hard_safe_mask.shape == (B * G,)
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
        adv_before = adv.detach().float()
        grpo_advantage_mean_before = adv_before.mean().to(adv)
        grpo_advantage_std_before = adv_before.std(unbiased=False).to(adv)
        grpo_advantage_clip_frac = adv.new_zeros(())
        if bool(getattr(self, "normalize_advantage_batch", False)):
            adv_mean = adv.mean()
            adv_std = adv.std(unbiased=False).clamp(min=1e-6)
            adv = (adv - adv_mean) / adv_std
        advantage_clip_abs = float(getattr(self, "advantage_clip_abs", 0.0))
        if advantage_clip_abs > 0.0:
            clip_mask = adv.detach().abs() > advantage_clip_abs
            grpo_advantage_clip_frac = clip_mask.float().mean().to(adv)
            adv = adv.clamp(min=-advantage_clip_abs, max=advantage_clip_abs)
        adv_after = adv.detach().float()
        grpo_advantage_mean_after = adv_after.mean().to(adv)
        grpo_advantage_std_after = adv_after.std(unbiased=False).to(adv)
        grpo_advantage_min = adv_after.min().to(adv)
        grpo_advantage_max = adv_after.max().to(adv)
        grpo_advantage_positive_ratio = (adv_after > 0.0).float().mean().to(adv)
        grpo_advantage_zero_ratio = (adv_after.abs() <= 1e-8).float().mean().to(adv)

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
        gspo_log_ratio_mean = new_log_probs.new_zeros(())
        gspo_log_ratio_std = new_log_probs.new_zeros(())
        gspo_log_ratio_min = new_log_probs.new_zeros(())
        gspo_log_ratio_max = new_log_probs.new_zeros(())
        gspo_abs_log_ratio_mean = new_log_probs.new_zeros(())
        gspo_ratio_min = new_log_probs.new_tensor(1.0)
        gspo_ratio_max = new_log_probs.new_tensor(1.0)
        gspo_approx_kl = new_log_probs.new_zeros(())
        gspo_reverse_approx_kl = new_log_probs.new_zeros(())

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
                gspo_ratio_min = ratio.detach().min()
                gspo_ratio_max = ratio.detach().max()
                gspo_approx_kl = (((ratio - 1.0) - log_ratio).detach().float().mean()).to(ratio)
                gspo_reverse_approx_kl = ((old_traj_logp - new_traj_logp).detach().float().mean()).to(ratio)
                log_ratio_detached = log_ratio.detach().float()
                gspo_log_ratio_mean = log_ratio_detached.mean().to(ratio)
                gspo_log_ratio_std = log_ratio_detached.std(unbiased=False).to(ratio)
                gspo_log_ratio_min = log_ratio_detached.min().to(ratio)
                gspo_log_ratio_max = log_ratio_detached.max().to(ratio)
                gspo_abs_log_ratio_mean = log_ratio_detached.abs().mean().to(ratio)
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

        reference_kl_coeff = float(getattr(self, "reference_kl_coeff", 0.0))
        reference_kl_loss = policy_loss.new_zeros(())
        if reference_kl_coeff > 0.0:
            reference_kl_loss = self._chain_transition_reference_kl(
                vl_features_rep,
                his_traj_rep,
                status_feature_rep,
                chains,
                expert_action_input_rep,
                B,
                G,
                num_denoising_steps,
                discount,
            ).to(dtype=policy_loss.dtype)
            total_loss = total_loss + reference_kl_coeff * reference_kl_loss

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

        grpo_buffer_distill_loss = total_loss.new_zeros(())
        grpo_buffer_distill_weight = 0.0
        grpo_buffer_distill_diag = {
            "per_sample_loss_mean": total_loss.new_zeros(()),
            "target_norm_mean": total_loss.new_zeros(()),
            "diffusion_timestep_mean": total_loss.new_zeros(()),
            "diffusion_timestep_min": total_loss.new_zeros(()),
            "diffusion_timestep_max": total_loss.new_zeros(()),
            "effective_weight_sum": total_loss.new_zeros(()),
            "zero_weight_ratio": total_loss.new_zeros(()),
            "zero_weight_batch": total_loss.new_zeros(()),
        }
        if use_grpo_buffer_guidance:
            grpo_buffer_distill_weight = self._current_grpo_buffer_distill_loss_weight(offline_cfg)
            if grpo_buffer_distill_weight > 0.0:
                distill_weights = (
                    grpo_buffer_guidance["target_weights"].to(device=total_loss.device, dtype=torch.float32)
                    * grpo_buffer_guidance["target_mask"].to(device=total_loss.device, dtype=torch.float32)
                )
                grpo_buffer_distill_loss, grpo_buffer_distill_diag = self._weighted_diffusion_loss_on_targets(
                    vl_features,
                    action_input,
                    grpo_buffer_guidance["target_trajs"].to(device=total_loss.device, dtype=action_input.action.dtype),
                    distill_weights,
                    timestep_sampling=str(offline_cfg.grpo_buffer_distill_timestep_sampling),
                )
                total_loss = total_loss + float(grpo_buffer_distill_weight) * grpo_buffer_distill_loss

        grpo_buffer_preference_dpo_loss = total_loss.new_zeros(())
        grpo_buffer_preference_dpo_weight = 0.0
        grpo_buffer_preference_targets: Dict[str, torch.Tensor] = {
            "target_ratio": total_loss.new_zeros(()),
            "il_loser_ratio": total_loss.new_zeros(()),
        }
        grpo_buffer_preference_dpo_diag = self._zero_diffusion_dpo_diag(total_loss.new_zeros(()))
        if use_grpo_buffer_guidance:
            grpo_buffer_preference_dpo_weight = self._current_grpo_buffer_preference_dpo_loss_weight(offline_cfg)
            if grpo_buffer_preference_dpo_weight > 0.0:
                grpo_buffer_preference_targets = self._build_grpo_buffer_preference_dpo_targets(
                    vl_features,
                    action_input,
                    grpo_buffer_guidance,
                    offline_cfg,
                )
                dpo_cfg = copy.copy(offline_cfg)
                dpo_cfg.preference_dpo_loss_weight = float(grpo_buffer_preference_dpo_weight)
                dpo_cfg.preference_dpo_timestep_sampling = str(
                    offline_cfg.grpo_buffer_preference_dpo_timestep_sampling
                )
                dpo_cfg.preference_dpo_beta = float(offline_cfg.grpo_buffer_preference_dpo_beta)
                dpo_cfg.preference_dpo_label_smoothing = float(
                    offline_cfg.grpo_buffer_preference_dpo_label_smoothing
                )
                dpo_cfg.preference_dpo_reference_free = bool(
                    offline_cfg.grpo_buffer_preference_dpo_reference_free
                )
                dpo_cfg.preference_dpo_pair_mode = str(offline_cfg.grpo_buffer_preference_dpo_pair_mode)
                dpo_cfg.preference_dpo_min_reward_gap = float(
                    offline_cfg.grpo_buffer_preference_dpo_min_reward_gap
                )
                dpo_cfg.preference_dpo_max_pairs_per_scene = int(
                    offline_cfg.grpo_buffer_preference_dpo_max_pairs_per_scene
                )
                dpo_cfg.preference_dpo_gap_weight_mode = str(
                    offline_cfg.grpo_buffer_preference_dpo_gap_weight_mode
                )
                dpo_cfg.preference_dpo_gap_weight_scale = float(
                    offline_cfg.grpo_buffer_preference_dpo_gap_weight_scale
                )
                dpo_cfg.preference_dpo_gap_weight_min = float(
                    offline_cfg.grpo_buffer_preference_dpo_gap_weight_min
                )
                dpo_cfg.preference_dpo_gap_weight_max = float(
                    offline_cfg.grpo_buffer_preference_dpo_gap_weight_max
                )
                grpo_buffer_preference_dpo_loss, grpo_buffer_preference_dpo_diag = (
                    self._compute_diffusion_dpo_preference_loss(
                        vl_features,
                        action_input,
                        grpo_buffer_preference_targets["target_trajs"].to(
                            device=total_loss.device,
                            dtype=action_input.action.dtype,
                        ),
                        grpo_buffer_preference_targets["ordering_rewards"].to(
                            device=total_loss.device,
                            dtype=torch.float32,
                        ),
                        grpo_buffer_preference_targets["valid_mask"].to(device=total_loss.device),
                        grpo_buffer_preference_targets["real_mask"].to(device=total_loss.device),
                        grpo_buffer_preference_targets["source_code"].to(device=total_loss.device),
                        dpo_cfg,
                    )
                )
                total_loss = (
                    total_loss
                    + float(grpo_buffer_preference_dpo_weight) * grpo_buffer_preference_dpo_loss
                )

        grpo_self_imitation_loss = total_loss.new_zeros(())
        grpo_self_imitation_weight = 0.0
        grpo_self_imitation_targets: Dict[str, Any] = {}
        grpo_self_imitation_diag = {
            "per_sample_loss_mean": total_loss.new_zeros(()),
            "target_norm_mean": total_loss.new_zeros(()),
            "diffusion_timestep_mean": total_loss.new_zeros(()),
            "diffusion_timestep_min": total_loss.new_zeros(()),
            "diffusion_timestep_max": total_loss.new_zeros(()),
            "effective_weight_sum": total_loss.new_zeros(()),
            "zero_weight_ratio": total_loss.new_zeros(()),
            "zero_weight_batch": total_loss.new_zeros(()),
        }
        if offline_cfg is not None and bool(offline_cfg.enabled):
            grpo_self_imitation_weight = self._current_grpo_self_imitation_loss_weight(offline_cfg)
            if grpo_self_imitation_weight > 0.0:
                grpo_self_imitation_targets = self._build_grpo_self_imitation_targets(
                    trajs,
                    base_rewards,
                    hard_safe_mask,
                    components,
                    B,
                    G,
                    grpo_buffer_guidance,
                    offline_cfg,
                )
                self_imitation_weights = (
                    grpo_self_imitation_targets["target_weights"].to(device=total_loss.device, dtype=torch.float32)
                    * grpo_self_imitation_targets["target_mask"].to(device=total_loss.device, dtype=torch.float32)
                )
                grpo_self_imitation_loss, grpo_self_imitation_diag = self._weighted_diffusion_loss_on_targets(
                    vl_features,
                    action_input,
                    grpo_self_imitation_targets["target_trajs"].to(
                        device=total_loss.device,
                        dtype=action_input.action.dtype,
                    ),
                    self_imitation_weights,
                    timestep_sampling=str(offline_cfg.grpo_self_imitation_timestep_sampling),
                )
                total_loss = total_loss + float(grpo_self_imitation_weight) * grpo_self_imitation_loss

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
            "reference_kl_loss": reference_kl_loss,
            "reference_kl_coeff": total_loss.new_tensor(reference_kl_coeff),
            "reference_kl_chunk_size": total_loss.new_tensor(float(getattr(self, "reference_kl_chunk_size", 0))),
            "grpo_buffer_guidance_enabled": total_loss.new_tensor(float(bool(use_grpo_buffer_guidance))),
            "grpo_buffer_reward_bonus": grpo_buffer_reward_bonus.mean().to(dtype=total_loss.dtype),
            "grpo_buffer_reward_bonus_weight": total_loss.new_tensor(float(grpo_buffer_bonus_weight)),
            "grpo_buffer_reward_bonus_max": grpo_buffer_diag["grpo_buffer_reward_bonus_max"].to(dtype=total_loss.dtype),
            "grpo_buffer_target_distance_mean": grpo_buffer_diag["grpo_buffer_target_distance_mean"].to(dtype=total_loss.dtype),
            "grpo_buffer_guidance_target_ratio": grpo_buffer_diag["grpo_buffer_guidance_target_ratio"].to(dtype=total_loss.dtype),
            "grpo_buffer_selected_valid_ratio": (
                grpo_buffer_guidance.get("selected_valid_ratio", total_loss.new_zeros(()))
                if use_grpo_buffer_guidance else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_buffer_best_valid_minus_gt": (
                grpo_buffer_guidance.get("best_valid_minus_gt_mean", total_loss.new_zeros(()))
                if use_grpo_buffer_guidance else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_buffer_best_valid_minus_il": (
                grpo_buffer_guidance.get("best_valid_minus_il_mean", total_loss.new_zeros(()))
                if use_grpo_buffer_guidance else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_buffer_distill_loss": grpo_buffer_distill_loss,
            "grpo_buffer_distill_weight": total_loss.new_tensor(float(grpo_buffer_distill_weight)),
            "grpo_buffer_distill_weight_sum": grpo_buffer_distill_diag["effective_weight_sum"].to(dtype=total_loss.dtype),
            "grpo_buffer_distill_zero_weight_batch": grpo_buffer_distill_diag["zero_weight_batch"].to(dtype=total_loss.dtype),
            "grpo_buffer_preference_dpo_loss": grpo_buffer_preference_dpo_loss,
            "grpo_buffer_preference_dpo_weight": total_loss.new_tensor(float(grpo_buffer_preference_dpo_weight)),
            "grpo_buffer_preference_dpo_target_ratio": (
                grpo_buffer_preference_targets.get("target_ratio", total_loss.new_zeros(()))
            ).to(dtype=total_loss.dtype),
            "grpo_buffer_preference_dpo_il_loser_ratio": (
                grpo_buffer_preference_targets.get("il_loser_ratio", total_loss.new_zeros(()))
            ).to(dtype=total_loss.dtype),
            "grpo_buffer_preference_dpo_pair_count": (
                grpo_buffer_preference_dpo_diag["preference_dpo_pair_count"].to(total_loss).detach()
            ),
            "grpo_buffer_preference_dpo_active_row_ratio": (
                grpo_buffer_preference_dpo_diag["preference_dpo_active_row_ratio"].to(total_loss).detach()
            ),
            "grpo_buffer_preference_dpo_reward_gap_mean": (
                grpo_buffer_preference_dpo_diag["preference_dpo_reward_gap_mean"].to(total_loss).detach()
            ),
            "grpo_buffer_preference_dpo_gap_weight_mean": (
                grpo_buffer_preference_dpo_diag["preference_dpo_gap_weight_mean"].to(total_loss).detach()
            ),
            "grpo_buffer_preference_dpo_logit_mean": (
                grpo_buffer_preference_dpo_diag["preference_dpo_logit_mean"].to(total_loss).detach()
            ),
            "grpo_buffer_preference_dpo_logit_abs_mean": (
                grpo_buffer_preference_dpo_diag["preference_dpo_logit_abs_mean"].to(total_loss).detach()
            ),
            "grpo_buffer_preference_dpo_implicit_accuracy": (
                grpo_buffer_preference_dpo_diag["preference_dpo_implicit_accuracy"].to(total_loss).detach()
            ),
            "grpo_buffer_preference_dpo_current_margin_mean": (
                grpo_buffer_preference_dpo_diag["preference_dpo_current_margin_mean"].to(total_loss).detach()
            ),
            "grpo_buffer_preference_dpo_reference_margin_mean": (
                grpo_buffer_preference_dpo_diag["preference_dpo_reference_margin_mean"].to(total_loss).detach()
            ),
            "grpo_buffer_preference_dpo_timestep_mean": (
                grpo_buffer_preference_dpo_diag["preference_dpo_timestep_mean"].to(total_loss).detach()
            ),
            "grpo_buffer_preference_dpo_timestep_min": (
                grpo_buffer_preference_dpo_diag["preference_dpo_timestep_min"].to(total_loss).detach()
            ),
            "grpo_buffer_preference_dpo_timestep_max": (
                grpo_buffer_preference_dpo_diag["preference_dpo_timestep_max"].to(total_loss).detach()
            ),
            "grpo_self_imitation_enabled": total_loss.new_tensor(
                float(
                    offline_cfg is not None
                    and bool(offline_cfg.enabled)
                    and float(offline_cfg.grpo_self_imitation_loss_weight) > 0.0
                )
            ),
            "grpo_self_imitation_loss": grpo_self_imitation_loss,
            "grpo_self_imitation_weight": total_loss.new_tensor(float(grpo_self_imitation_weight)),
            "grpo_self_imitation_candidate_ratio": (
                grpo_self_imitation_targets.get("candidate_ratio", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_safety_candidate_ratio": (
                grpo_self_imitation_targets.get("safety_candidate_ratio", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_nc_pass_ratio": (
                grpo_self_imitation_targets.get("nc_pass_ratio", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_dac_pass_ratio": (
                grpo_self_imitation_targets.get("dac_pass_ratio", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_ttc_pass_ratio": (
                grpo_self_imitation_targets.get("ttc_pass_ratio", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_ddc_pass_ratio": (
                grpo_self_imitation_targets.get("ddc_pass_ratio", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_target_ratio": (
                grpo_self_imitation_targets.get("target_ratio", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_pre_cap_target_ratio": (
                grpo_self_imitation_targets.get("pre_cap_target_ratio", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_target_scene_cap_ratio": (
                grpo_self_imitation_targets.get("target_scene_cap_ratio", total_loss.new_ones(()))
                if grpo_self_imitation_targets else total_loss.new_ones(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_target_scene_cap_active": (
                grpo_self_imitation_targets.get("target_scene_cap_active", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_target_reward_mean": (
                grpo_self_imitation_targets.get("target_reward_mean", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_target_reward_max": (
                grpo_self_imitation_targets.get("target_reward_max", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_target_margin_mean": (
                grpo_self_imitation_targets.get("target_margin_mean", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_baseline_mean": (
                grpo_self_imitation_targets.get("baseline_mean", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_baseline_from_buffer": (
                grpo_self_imitation_targets.get("baseline_from_buffer", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_target_nc_mean": (
                grpo_self_imitation_targets.get("target_nc_mean", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_target_dac_mean": (
                grpo_self_imitation_targets.get("target_dac_mean", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_target_ttc_mean": (
                grpo_self_imitation_targets.get("target_ttc_mean", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_target_ep_mean": (
                grpo_self_imitation_targets.get("target_ep_mean", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_target_comfort_mean": (
                grpo_self_imitation_targets.get("target_comfort_mean", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_target_ddc_mean": (
                grpo_self_imitation_targets.get("target_ddc_mean", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_target_tlc_mean": (
                grpo_self_imitation_targets.get("target_tlc_mean", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_weight_sum": grpo_self_imitation_diag["effective_weight_sum"].to(dtype=total_loss.dtype),
            "grpo_self_imitation_zero_weight_batch": grpo_self_imitation_diag["zero_weight_batch"].to(dtype=total_loss.dtype),
            "grpo_self_imitation_timestep_mean": grpo_self_imitation_diag["diffusion_timestep_mean"].to(dtype=total_loss.dtype),
            "safe_ratio": hard_safe_ratio,
            "hard_safe_ratio": hard_safe_ratio,
            "mean_ep": reward_aux["ego_progress"].mean(),
            "mean_nc": reward_aux["no_at_fault_collisions"].mean(),
            "mean_dac": reward_aux["drivable_area_compliance"].mean(),
            "mean_ttc": reward_aux["time_to_collision_within_bound"].mean(),
            "mean_comfort": reward_aux["history_comfort"].mean(),
            "mean_ddc": reward_aux["driving_direction_compliance"].mean(),
            "mean_tlc": reward_aux["traffic_light_compliance"].mean(),
            "soft_safety_penalty_mean": reward_aux["soft_safety_penalty"].mean(),
            "soft_safety_penalty_max": reward_aux["soft_safety_penalty"].max(),
            "soft_safety_penalty_weight": total_loss.new_tensor(float(self.soft_safety_penalty_weight)),
            "soft_safety_mode_enabled": total_loss.new_tensor(
                float(str(self.safety_advantage_mode) == "soft_penalty")
            ),
            "nc_pass_ratio": (
                reward_aux["no_at_fault_collisions"] >= float(self.nc_safe_threshold)
            ).detach().float().mean().to(dtype=total_loss.dtype),
            "dac_pass_ratio": (
                reward_aux["drivable_area_compliance"] >= float(self.dac_safe_threshold)
            ).detach().float().mean().to(dtype=total_loss.dtype),
            "ttc_pass_ratio": (
                reward_aux["time_to_collision_within_bound"] >= float(self.ttc_safe_threshold)
            ).detach().float().mean().to(dtype=total_loss.dtype),
            "ddc_pass_ratio": (
                reward_aux["driving_direction_compliance"] >= float(self.ddc_safe_threshold)
            ).detach().float().mean().to(dtype=total_loss.dtype),
            "tlc_pass_ratio": (
                reward_aux["traffic_light_compliance"] >= float(self.tlc_safe_threshold)
            ).detach().float().mean().to(dtype=total_loss.dtype),
            "diversity_bonus": diversity_bonus.mean(),
            "safe_diversity": safe_diversity,
            "group_reward_std": advantage_aux["reward_std"].mean(),
            "safe_count_mean": advantage_aux["safe_count"].mean(),
            "group_weight_mean": group_weight.mean(),
            "group_weight_min": group_weight.min(),
            "group_weight_max": group_weight.max(),
            "mixed_group_ratio": advantage_aux["mixed_group_ratio"],
            "all_safe_group_ratio": advantage_aux["all_safe_group_ratio"],
            "all_unsafe_group_ratio": advantage_aux["all_unsafe_group_ratio"],
            "safe_rpp_advantage_enabled": advantage_aux.get(
                "safe_rpp_advantage_enabled",
                total_loss.new_tensor(0.0),
            ),
            "safe_rpp_centered_batch_std": advantage_aux.get(
                "safe_rpp_centered_batch_std",
                total_loss.new_tensor(0.0),
            ),
            "mean_advantage": advantages.mean(),
            "mean_abs_advantage": advantages.abs().mean(),
            "grpo_advantage_mean_before_transform": grpo_advantage_mean_before.to(dtype=total_loss.dtype),
            "grpo_advantage_std_before_transform": grpo_advantage_std_before.to(dtype=total_loss.dtype),
            "grpo_advantage_mean_after_transform": grpo_advantage_mean_after.to(dtype=total_loss.dtype),
            "grpo_advantage_std_after_transform": grpo_advantage_std_after.to(dtype=total_loss.dtype),
            "grpo_advantage_min": grpo_advantage_min.to(dtype=total_loss.dtype),
            "grpo_advantage_max": grpo_advantage_max.to(dtype=total_loss.dtype),
            "grpo_advantage_positive_ratio": grpo_advantage_positive_ratio.to(dtype=total_loss.dtype),
            "grpo_advantage_zero_ratio": grpo_advantage_zero_ratio.to(dtype=total_loss.dtype),
            "grpo_advantage_clip_frac": grpo_advantage_clip_frac.to(dtype=total_loss.dtype),
            "grpo_advantage_batch_normalized": total_loss.new_tensor(
                float(bool(getattr(self, "normalize_advantage_batch", False)))
            ),
            "grpo_advantage_clip_abs": total_loss.new_tensor(float(advantage_clip_abs)),
            **self._stage3_core_pareto_log_metrics(total_loss, reward_aux, advantage_aux),
            **self._stage3_feasible_pareto_log_metrics(total_loss, advantage_aux),
            "trajectory_logp": trajectory_logp.mean(),
            "gspo_ratio_mean": gspo_ratio_mean.to(dtype=total_loss.dtype),
            "gspo_ratio_min": gspo_ratio_min.to(dtype=total_loss.dtype),
            "gspo_ratio_max": gspo_ratio_max.to(dtype=total_loss.dtype),
            "gspo_ratio_clip_frac": gspo_ratio_clip_frac.to(dtype=total_loss.dtype),
            "gspo_log_ratio_mean": gspo_log_ratio_mean.to(dtype=total_loss.dtype),
            "gspo_log_ratio_std": gspo_log_ratio_std.to(dtype=total_loss.dtype),
            "gspo_log_ratio_min": gspo_log_ratio_min.to(dtype=total_loss.dtype),
            "gspo_log_ratio_max": gspo_log_ratio_max.to(dtype=total_loss.dtype),
            "gspo_abs_log_ratio_mean": gspo_abs_log_ratio_mean.to(dtype=total_loss.dtype),
            "gspo_approx_kl": gspo_approx_kl.to(dtype=total_loss.dtype),
            "gspo_reverse_approx_kl": gspo_reverse_approx_kl.to(dtype=total_loss.dtype),
            "use_gspo_ratio": total_loss.new_tensor(float(bool(self.use_gspo_ratio))),
            "sampled_from_behavior_policy": total_loss.new_tensor(float(bool(sampled_from_behavior_policy))),
            "behavior_policy_synced": total_loss.new_tensor(float(bool(behavior_policy_synced))),
            "behavior_policy_sync_interval": total_loss.new_tensor(float(getattr(self, "behavior_policy_sync_interval", 0))),
            "gspo_clip_low": total_loss.new_tensor(float(self.gspo_clip_low)),
            "gspo_clip_high": total_loss.new_tensor(float(self.gspo_clip_high)),
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
