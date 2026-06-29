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
import lzma
import pickle
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

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
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
    PDMScorerConfig,
)
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
from .expert_fusion import (
    AlignmentHead,
    ExpertAdapter768,
    HorizonAwareExpertConditioner,
    TeacherTokenProjector,
    branch_logits_from_probs,
    init_logit_from_prob,
    normalized_mse_loss,
)
from .bit_drive import BitRiskHead, ReverseTrajectoryDecoder, RiskTokenEncoder, TargetPathTokenEncoder, TerminalPathHead
from .bit_losses import bit_loss_bundle, build_path_targets

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
    
    metric_cache_path: str = "/path/to/metric_cache_train"
    reference_policy_checkpoint: str = "/path/to/IL_Model.ckpt"
    scorer_config: PDMScorerConfig = field(default_factory=lambda: PDMScorerConfig(
        progress_weight=10.0, ttc_weight=5.0, comfortable_weight=2.0
    ))


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
    vlm_feature_dim: Optional[int] = None
    vlm_adapter_type: str = "linear"
    vlm_adapter_dropout: float = 0.0
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

    use_bit_drive: bool = False
    bit_terminal_loss_weight: float = 0.05
    bit_path_loss_weight: float = 0.05
    bit_end_consistency_loss_weight: float = 0.03
    bit_reverse_loss_weight: float = 0.03
    bit_cycle_loss_weight: float = 0.00
    bit_num_path_anchors: int = 4
    bit_path_anchor_indices: tuple = (1, 3, 5, 7)
    bit_condition_mode: str = "context_tokens"
    bit_use_gt_condition_prob: float = 0.20
    bit_condition_noise_std: float = 0.05
    bit_condition_dropout: float = 0.10
    bit_reverse_decoder_hidden_dim: int = 384
    bit_use_reverse_decoder: bool = False
    bit_use_path_anchors: bool = True
    bit_use_terminal_head: bool = True
    bit_fail_if_gt_condition_in_eval: bool = True
    bit_log_diagnostics: bool = True
    bit_context_condition_strength: float = 0.10
    bit_action_condition_strength: float = 0.00
    bit_learnable_condition_gates: bool = True
    bit_context_gate_init: float = 0.05
    bit_action_gate_init: float = 0.00
    bit_detach_condition: bool = True
    bit_detach_condition_until_step: int = 100000
    bit_use_gt_condition_schedule: str = "constant"
    bit_use_gt_condition_prob_start: float = 0.20
    bit_use_gt_condition_prob_end: float = 0.00
    bit_use_gt_condition_decay_steps: int = 1000
    bit_loss_normalization: str = "norm_odo"
    bit_enable_terminal_condition: bool = True
    bit_enable_path_condition: bool = True
    bit_enable_action_add: bool = False
    bit_enable_context_tokens: bool = True
    bit_condition_coordinate_mode: str = "full"
    bit_condition_axis_scale_x: float = 0.10
    bit_condition_axis_scale_y: float = 1.00
    bit_condition_axis_scale_heading: float = 1.00
    bit_preserve_base_longitudinal: bool = False
    bit_base_longitudinal_loss_weight: float = 0.00
    bit_base_early_points: int = 3
    bit_base_lateral_loss_weight: float = 0.00
    bit_store_base_traj_for_loss: bool = False
    bit_use_safety_fallback: bool = False
    bit_fallback_mode: str = "rule"
    bit_fallback_early_x_delta_threshold: float = 1.0
    bit_fallback_terminal_x_delta_threshold: float = 2.0
    bit_use_risk_head: bool = False
    bit_use_risk_tokens: bool = False
    bit_use_risk_token: bool = False
    bit_risk_loss_weight: float = 0.00
    bit_risk_bce_loss_weight: float = 0.00
    bit_risk_token_strength: float = 0.05
    bit_risk_hidden_dim: int = 512
    bit_risk_positive_weight_zero: float = 1.0
    bit_risk_positive_weight_dac: float = 2.0
    bit_risk_positive_weight_nc: float = 4.0
    bit_risk_positive_weight_ttc: float = 2.0
    bit_use_d5_conservative_loss: bool = False
    bit_safe_kd_loss_weight: float = 0.05
    bit_safe_kd_points: int = 3
    bit_safe_kd_x_weight: float = 1.0
    bit_safe_kd_step_weight: float = 0.5
    bit_safe_kd_heading_weight: float = 0.2
    bit_safe_kd_y_weight: float = 0.0
    bit_safe_kd_apply_on_nc: bool = True
    bit_safe_kd_apply_on_ttc: bool = True
    bit_safe_kd_apply_on_soft_safety: bool = True
    bit_soft_safe_kd_weight_scale: float = 0.25
    bit_dac_path_preserve_weight: float = 0.02
    bit_dac_path_preserve_apply_on_dac_fix: bool = True

    use_risk_vla: bool = False
    risk_vla_num_classes: int = 6
    risk_vla_router_mode: str = "independent"
    risk_vla_use_bit_summary: bool = True
    risk_vla_use_oracle_router: bool = False
    risk_vla_strategy_token_scale: float = 1.0
    risk_vla_horizon_residual_scale: float = 1.0
    risk_vla_risk_loss_weight: float = 0.0
    risk_vla_focal_loss_weight: float = 0.0
    risk_vla_strategy_entropy_weight: float = 0.0
    risk_vla_detach_risk_for_strategy: bool = True
    risk_vla_log_diagnostics: bool = True

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
    use_action_aware_aux: bool = False
    action_aware_aux_weight: float = 0.0
    action_aware_aux_space: str = "norm_odo"
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
    
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    
    flow_cfg: FlowConfig = field(default_factory=FlowConfig)
    ddpm_cfg: DDPMConfig = field(default_factory=DDPMConfig)
    ddim_cfg: DDIMConfig = field(default_factory=DDIMConfig)
    grpo_cfg: GRPOConfig = field(default_factory=GRPOConfig)


class ReCogDriveDiffusionPlanner(nn.Module):
    config_class = ReCogDriveDiffusionPlannerConfig
    expert_target_keys = (
        "jepa_target_tokens",
        "vggt_target_tokens",
        "vggt_geometry_target_tokens",
        "vggt_depth_target_tokens",
        "vggt_pointmap_target_tokens",
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
        vlm_feature_dim = config.vlm_feature_dim
        if vlm_feature_dim is None:
            vlm_feature_dim = 3584 if config.vlm_size == "large" else 1536
        if config.vlm_adapter_type not in {"linear", "pre_ln_linear", "pre_ln_linear_post_ln", "mlp_adapter"}:
            raise ValueError(
                "vlm_adapter_type must be one of 'linear', 'pre_ln_linear', "
                "'pre_ln_linear_post_ln', or 'mlp_adapter'."
            )
        if not 0.0 <= float(config.vlm_adapter_dropout) < 1.0:
            raise ValueError("vlm_adapter_dropout must be in [0, 1).")
        if config.action_aware_aux_space not in {"norm_odo", "raw"}:
            raise ValueError("action_aware_aux_space must be 'norm_odo' or 'raw'.")
        self.feature_encoder = nn.Linear(vlm_feature_dim, config.input_embedding_dim)
        self.vlm_pre_norm = (
            nn.LayerNorm(vlm_feature_dim)
            if config.vlm_adapter_type in {"pre_ln_linear", "pre_ln_linear_post_ln", "mlp_adapter"}
            else nn.Identity()
        )
        self.vlm_post_norm = (
            nn.LayerNorm(config.input_embedding_dim)
            if config.vlm_adapter_type in {"pre_ln_linear_post_ln", "mlp_adapter"}
            else nn.Identity()
        )
        self.vlm_adapter_dropout = (
            nn.Dropout(float(config.vlm_adapter_dropout))
            if float(config.vlm_adapter_dropout) > 0.0
            else nn.Identity()
        )
        self.vlm_adapter_mlp = (
            nn.Sequential(
                nn.GELU(),
                nn.Linear(config.input_embedding_dim, config.input_embedding_dim),
            )
            if config.vlm_adapter_type == "mlp_adapter"
            else nn.Identity()
        )

        if config.alignment_loss_type not in {"normalized_mse", "mse", "cosine"}:
            raise ValueError("alignment_loss_type must be one of 'normalized_mse', 'mse', or 'cosine'.")
        if config.diffusion_loss_weight < 0.0:
            raise ValueError("diffusion_loss_weight must be non-negative.")
        if config.use_last_rd and config.last_rd_stage == "disabled":
            raise ValueError("use_last_rd=True requires last_rd_stage='stage1_5' or 'progressive_sft'.")
        if not config.use_last_rd and config.last_rd_stage != "disabled":
            raise ValueError("last_rd_stage must be 'disabled' when use_last_rd=False.")
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
            "action_aware_aux_weight",
            "bit_terminal_loss_weight",
            "bit_path_loss_weight",
            "bit_end_consistency_loss_weight",
            "bit_reverse_loss_weight",
            "bit_cycle_loss_weight",
            "bit_context_condition_strength",
            "bit_action_condition_strength",
            "bit_base_longitudinal_loss_weight",
            "bit_base_lateral_loss_weight",
            "bit_risk_loss_weight",
            "bit_risk_bce_loss_weight",
            "bit_risk_token_strength",
            "bit_safe_kd_loss_weight",
            "bit_safe_kd_x_weight",
            "bit_safe_kd_step_weight",
            "bit_safe_kd_heading_weight",
            "bit_safe_kd_y_weight",
            "bit_soft_safe_kd_weight_scale",
            "bit_dac_path_preserve_weight",
            "risk_vla_strategy_token_scale",
            "risk_vla_horizon_residual_scale",
            "risk_vla_risk_loss_weight",
            "risk_vla_focal_loss_weight",
            "risk_vla_strategy_entropy_weight",
        ):
            if getattr(config, weight_name) < 0.0:
                raise ValueError(f"{weight_name} must be non-negative.")

        if config.use_bit_drive:
            if config.bit_condition_mode not in {"none", "context_tokens", "action_add", "tokens_and_action"}:
                raise ValueError(
                    "bit_condition_mode must be one of 'none', 'context_tokens', 'action_add', or 'tokens_and_action'."
                )
            if config.bit_num_path_anchors <= 0:
                raise ValueError("bit_num_path_anchors must be positive.")
            if len(tuple(config.bit_path_anchor_indices)) != int(config.bit_num_path_anchors):
                raise ValueError("bit_path_anchor_indices length must match bit_num_path_anchors.")
            if not 0.0 <= float(config.bit_use_gt_condition_prob) <= 1.0:
                raise ValueError("bit_use_gt_condition_prob must be in [0, 1].")
            if config.bit_use_gt_condition_schedule not in {"constant", "linear_decay"}:
                raise ValueError("bit_use_gt_condition_schedule must be 'constant' or 'linear_decay'.")
            if not 0.0 <= float(config.bit_use_gt_condition_prob_start) <= 1.0:
                raise ValueError("bit_use_gt_condition_prob_start must be in [0, 1].")
            if not 0.0 <= float(config.bit_use_gt_condition_prob_end) <= 1.0:
                raise ValueError("bit_use_gt_condition_prob_end must be in [0, 1].")
            if int(config.bit_use_gt_condition_decay_steps) <= 0:
                raise ValueError("bit_use_gt_condition_decay_steps must be positive.")
            if not 0.0 <= float(config.bit_condition_dropout) < 1.0:
                raise ValueError("bit_condition_dropout must be in [0, 1).")
            if float(config.bit_condition_noise_std) < 0.0:
                raise ValueError("bit_condition_noise_std must be non-negative.")
            if config.bit_loss_normalization not in {"norm_odo", "manual_scale", "raw"}:
                raise ValueError("bit_loss_normalization must be one of 'norm_odo', 'manual_scale', or 'raw'.")
            if config.bit_condition_coordinate_mode not in {
                "full",
                "lateral_heading_only",
                "path_lateral_heading_only",
                "terminal_lateral_heading_only",
            }:
                raise ValueError("Unsupported bit_condition_coordinate_mode.")
            if config.bit_fallback_mode not in {"oracle_eval_only", "rule", "learned"}:
                raise ValueError("bit_fallback_mode must be 'oracle_eval_only', 'rule', or 'learned'.")
            if int(config.bit_base_early_points) <= 0:
                raise ValueError("bit_base_early_points must be positive.")
            if int(config.bit_safe_kd_points) <= 0:
                raise ValueError("bit_safe_kd_points must be positive.")
            if int(config.bit_risk_hidden_dim) <= 0:
                raise ValueError("bit_risk_hidden_dim must be positive.")
            if not config.bit_use_terminal_head:
                raise ValueError("BiT Step 1-3 requires bit_use_terminal_head=True.")

            self._bit_train_step = 0
            self.bit_terminal_head = TerminalPathHead(
                planner_dim=config.input_embedding_dim,
                status_dim=8,
                his_dim=12,
                hidden_dim=max(config.hidden_size // 2, config.input_embedding_dim),
                num_anchors=config.bit_num_path_anchors,
            )
            self.bit_condition_encoder = TargetPathTokenEncoder(
                planner_dim=config.input_embedding_dim,
                hidden_dim=max(config.hidden_size // 2, config.input_embedding_dim),
                num_anchors=config.bit_num_path_anchors,
            )
            if config.bit_learnable_condition_gates:
                context_gate_init = min(max(float(config.bit_context_gate_init), 1e-6), 1.0 - 1e-6)
                action_gate_init = min(max(float(config.bit_action_gate_init), 1e-6), 1.0 - 1e-6)
                self.bit_context_gate = nn.Parameter(
                    torch.tensor(init_logit_from_prob(context_gate_init), dtype=torch.float32)
                )
                self.bit_action_gate = nn.Parameter(
                    torch.tensor(init_logit_from_prob(action_gate_init), dtype=torch.float32)
                )
            if config.bit_use_reverse_decoder:
                self.bit_reverse_decoder = ReverseTrajectoryDecoder(
                    planner_dim=config.input_embedding_dim,
                    hidden_dim=config.bit_reverse_decoder_hidden_dim,
                    num_anchors=config.bit_num_path_anchors,
                    reverse_points=max(config.action_horizon - 1, 1),
                )
            if config.bit_use_risk_head:
                self.bit_risk_head = BitRiskHead(
                    planner_dim=config.input_embedding_dim,
                    status_dim=8,
                    his_dim=12,
                    hidden_dim=config.bit_risk_hidden_dim,
                    num_anchors=config.bit_num_path_anchors,
                    num_risks=4,
                )
                if bool(config.bit_use_risk_tokens or config.bit_use_risk_token):
                    self.bit_risk_token_encoder = RiskTokenEncoder(
                        planner_dim=config.input_embedding_dim,
                        hidden_dim=max(config.input_embedding_dim // 2, 128),
                        num_risks=4,
                    )

        if config.use_expert_features:
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

        if config.use_risk_vla:
            if config.risk_vla_router_mode not in {"independent", "learned_softmax"}:
                raise ValueError("risk_vla_router_mode must be 'independent' or 'learned_softmax'.")
            from .risk_vla import RiskConditionedStrategyBank, RiskConditionedStrategyRouter, RiskStateEncoder

            self.risk_state_encoder = RiskStateEncoder(
                planner_dim=config.input_embedding_dim,
                hidden_dim=max(config.hidden_size // 2, config.input_embedding_dim),
                num_risk_classes=config.risk_vla_num_classes,
                action_horizon=config.action_horizon,
                use_bit_summary=config.risk_vla_use_bit_summary,
                use_uncertainty_head=True,
            )
            self.risk_strategy_router = RiskConditionedStrategyRouter(
                planner_dim=config.input_embedding_dim,
                mode=config.risk_vla_router_mode,
                hidden_dim=max(config.hidden_size // 4, 128),
            )
            self.risk_strategy_bank = RiskConditionedStrategyBank(
                planner_dim=config.input_embedding_dim,
                action_horizon=config.action_horizon,
                action_dim=config.action_dim,
                tokens_per_strategy=2,
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
            reference_cfg.use_expert_features = False
            reference_cfg.use_risk_vla = False
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
        self.action_aware_aux_head = nn.Sequential(
            nn.LayerNorm(config.input_embedding_dim),
            nn.Linear(config.input_embedding_dim, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.action_horizon * config.action_dim),
        )
        if not bool(config.use_action_aware_aux) or float(config.action_aware_aux_weight) <= 0.0:
            for parameter in self.action_aware_aux_head.parameters():
                parameter.requires_grad = False
        
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, config.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        
        if self.config.sampling_method == 'flow':
            self._init_flow_sampler(config.flow_cfg)
        elif self.config.sampling_method == 'ddpm':
            self._init_ddpm_sampler(config.ddpm_cfg)
        elif self.config.sampling_method == 'ddim':
            self._init_ddim_sampler(config.ddim_cfg)

        if config.grpo:
            self._init_grpo(config.grpo_cfg)

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

    def _init_grpo(self, cfg: GRPOConfig):
        """Initializes components and hyperparameters for GRPO training."""
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
        
        self.metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
        proposal_sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)
        self.simulator = PDMSimulator(proposal_sampling)
        self.train_scorer = PDMScorer(proposal_sampling, cfg.scorer_config)
        
        self._safe_load_reference_policy(cfg.reference_policy_checkpoint)
        
        self.old_policy = copy.deepcopy(self)
        self.old_policy.eval()
        for param in self.old_policy.parameters():
            param.requires_grad = False

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
            "bit_terminal_head",
            "bit_condition_encoder",
            "bit_reverse_decoder",
            "bit_context_gate",
            "bit_action_gate",
            "bit_risk_head",
            "bit_risk_token_encoder",
            "risk_state_encoder",
            "risk_strategy_router",
            "risk_strategy_bank",
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
                self.vlm_pre_norm.eval()
                self.vlm_post_norm.eval()
                self.vlm_adapter_dropout.eval()
                self.vlm_adapter_mlp.eval()
                self.action_aware_aux_head.eval()
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
                        "bit_terminal_head",
                        "bit_condition_encoder",
                        "bit_reverse_decoder",
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
        embeds = self.feature_encoder(self.vlm_pre_norm(vl_features))
        embeds = self.vlm_adapter_mlp(embeds)
        embeds = self.vlm_adapter_dropout(embeds)
        return self.vlm_post_norm(embeds)

    def _vlm_context_diagnostics(
        self,
        vl_features: torch.Tensor,
        vl_embeds: torch.Tensor,
        context_mean: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            "vlm_hidden_input_norm": vl_features.detach().float().norm(dim=-1).mean(),
            "vlm_context_token_norm": vl_embeds.detach().float().norm(dim=-1).mean(),
            "vlm_context_mean_norm": context_mean.detach().float().norm(dim=-1).mean(),
            "context_token_mean_abs": vl_embeds.detach().float().abs().mean(),
            "context_token_std": vl_embeds.detach().float().std(unbiased=False),
        }

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

    def _prepare_base_context(
        self,
        vl_features: torch.Tensor,
        action_input: Optional[BatchFeature] = None,
    ) -> Dict[str, torch.Tensor]:
        vl_embeds = self._encode_vlm(vl_features)
        return {
            "vl_embeds": vl_embeds,
            "context_tokens": vl_embeds,
            "context_mean": vl_embeds.mean(1),
        }

    def _bit_zero_outputs(self, reference: torch.Tensor) -> Dict[str, Any]:
        zero = reference.new_zeros(())
        return {
            "bit_terminal_loss": zero,
            "bit_path_loss": zero,
            "bit_end_consistency_loss": zero,
            "bit_reverse_loss": zero,
            "bit_cycle_loss": zero,
            "bit_terminal_pred_mean_l2": zero,
            "bit_path_pred_mean_l2": zero,
            "bit_used_gt_condition_rate": zero,
            "bit_context_gate_value": zero,
            "bit_action_gate_value": zero,
            "bit_gt_condition_prob_current": zero,
            "bit_condition_strength_context": zero,
            "bit_condition_strength_action": zero,
            "bit_base_longitudinal_loss": zero,
            "bit_base_lateral_loss": zero,
            "bit_risk_loss": zero,
            "bit_risk_bce_loss": zero,
            "risk_bce_loss": zero,
            "bit_risk_zero_loss": zero,
            "bit_risk_dac_loss": zero,
            "bit_risk_nc_loss": zero,
            "bit_risk_ttc_loss": zero,
            "bit_risk_token_strength": zero,
            "d5_safe_kd_loss": zero,
            "d5_dac_path_preserve_loss": zero,
            "d5_safety_mask_rate": zero,
            "d5_hard_safety_mask_rate": zero,
            "d5_soft_safety_mask_rate": zero,
            "d5_dac_fix_mask_rate": zero,
            "bit_terminal_pred": None,
            "bit_path_anchor_pred": None,
            "bit_risk_logits": None,
            "bit_risk_prob_mean": None,
            "bit_terminal_gt": None,
            "bit_path_gt": None,
            "bit_selected_terminal": None,
            "bit_selected_path_anchors": None,
            "bit_action_condition": None,
            "bit_reverse_points": None,
        }

    def _bit_current_gt_condition_prob(self) -> float:
        if self.config.bit_use_gt_condition_schedule == "constant":
            return float(self.config.bit_use_gt_condition_prob)
        progress = min(
            max(float(getattr(self, "_bit_train_step", 0)) / float(self.config.bit_use_gt_condition_decay_steps), 0.0),
            1.0,
        )
        start = float(self.config.bit_use_gt_condition_prob_start)
        end = float(self.config.bit_use_gt_condition_prob_end)
        return start + (end - start) * progress

    def _bit_gate_value(self, name: str, reference: torch.Tensor) -> torch.Tensor:
        if not self.config.bit_learnable_condition_gates:
            return reference.new_tensor(1.0)
        gate = getattr(self, f"bit_{name}_gate")
        return torch.sigmoid(gate).to(device=reference.device, dtype=reference.dtype)

    def _bit_condition_strength(self, name: str, reference: torch.Tensor) -> torch.Tensor:
        base = float(
            self.config.bit_context_condition_strength
            if name == "context"
            else self.config.bit_action_condition_strength
        )
        return reference.new_tensor(base) * self._bit_gate_value(name, reference)

    def _bit_should_detach_condition(self, training: bool) -> bool:
        if not self.config.bit_detach_condition:
            return False
        if not training:
            return True
        until_step = int(self.config.bit_detach_condition_until_step)
        return until_step <= 0 or int(getattr(self, "_bit_train_step", 0)) < until_step

    def _bit_effective_context_tokens_enabled(self) -> bool:
        mode = self.config.bit_condition_mode
        if mode == "none":
            return False
        return bool(self.config.bit_enable_context_tokens) or mode in {"context_tokens", "tokens_and_action"}

    def _bit_effective_action_add_enabled(self) -> bool:
        mode = self.config.bit_condition_mode
        if mode == "none":
            return False
        return bool(self.config.bit_enable_action_add) or mode in {"action_add", "tokens_and_action"}

    def _bit_loss_space(self, values: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if values is None:
            return None
        mode = self.config.bit_loss_normalization
        if mode == "raw":
            return values
        if mode == "norm_odo":
            return self.norm_odo(values)
        if mode == "manual_scale":
            scale = values.new_tensor([30.0, 10.0, 3.14])
            return values / scale
        raise ValueError(f"Unsupported bit_loss_normalization={mode!r}")

    def _bit_apply_coordinate_mode(
        self,
        terminal: torch.Tensor,
        path_anchors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        axis_scale = terminal.new_tensor([
            float(self.config.bit_condition_axis_scale_x),
            float(self.config.bit_condition_axis_scale_y),
            float(self.config.bit_condition_axis_scale_heading),
        ])
        mode = self.config.bit_condition_coordinate_mode
        terminal_condition = terminal * axis_scale
        path_condition = path_anchors * axis_scale
        if mode == "full":
            return terminal_condition, path_condition
        if mode == "lateral_heading_only":
            return terminal_condition, path_condition
        if mode == "path_lateral_heading_only":
            terminal_full = terminal * terminal.new_tensor([
                min(float(self.config.bit_condition_axis_scale_x), 1.0),
                1.0,
                1.0,
            ])
            return terminal_full, path_condition
        if mode == "terminal_lateral_heading_only":
            path_full = path_anchors * path_anchors.new_tensor([
                min(float(self.config.bit_condition_axis_scale_x), 1.0),
                1.0,
                1.0,
            ])
            return terminal_condition, path_full
        raise ValueError(f"Unsupported bit_condition_coordinate_mode={mode!r}")

    def _build_bit_condition(
        self,
        context_mean: torch.Tensor,
        context_tokens: torch.Tensor,
        action_input: Optional[BatchFeature],
        *,
        gt_actions: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Dict[str, Any]:
        if not self.config.use_bit_drive:
            return self._bit_zero_outputs(context_mean)
        if action_input is None:
            action_input = BatchFeature(data={})

        status_feature = action_input.get("status_feature")
        his_traj = action_input.get("his_traj")
        terminal_pred, path_anchor_pred = self.bit_terminal_head(
            context_mean,
            status_feature=status_feature,
            his_traj=his_traj,
        )

        terminal_gt = None
        path_gt = None
        if gt_actions is None and "action" in action_input and isinstance(action_input["action"], torch.Tensor):
            gt_actions = action_input["action"]
        if gt_actions is not None:
            terminal_gt = gt_actions[:, -1, :].to(device=context_mean.device, dtype=context_mean.dtype)
            path_gt = build_path_targets(
                gt_actions.to(device=context_mean.device, dtype=context_mean.dtype),
                tuple(self.config.bit_path_anchor_indices),
                expected_anchors=int(self.config.bit_num_path_anchors),
            )

        current_gt_prob = self._bit_current_gt_condition_prob() if training else 0.0
        use_gt_mask: Optional[torch.Tensor] = None
        used_gt_condition_rate = context_mean.new_zeros(())
        if training and terminal_gt is not None and path_gt is not None and current_gt_prob > 0.0:
            use_gt_mask = (torch.rand(context_mean.shape[0], 1, device=context_mean.device) < current_gt_prob).to(dtype=context_mean.dtype)
            selected_terminal = torch.where(use_gt_mask.bool(), terminal_gt, terminal_pred)
            selected_path_anchors = torch.where(use_gt_mask.view(-1, 1, 1).bool(), path_gt, path_anchor_pred)
            used_gt_condition_rate = use_gt_mask.mean()
        else:
            selected_terminal = terminal_pred
            selected_path_anchors = path_anchor_pred

        if not training and self.config.bit_fail_if_gt_condition_in_eval and use_gt_mask is not None and bool(use_gt_mask.any().item()):
            raise RuntimeError("BiT evaluation attempted to condition on ground-truth future trajectory.")

        if training and self.config.bit_condition_noise_std > 0.0:
            noise_std = float(self.config.bit_condition_noise_std)
            selected_terminal = selected_terminal + noise_std * torch.randn_like(selected_terminal)
            selected_path_anchors = selected_path_anchors + noise_std * torch.randn_like(selected_path_anchors)
        if training and self.config.bit_condition_dropout > 0.0:
            keep = (
                torch.rand(context_mean.shape[0], 1, device=context_mean.device)
                >= float(self.config.bit_condition_dropout)
            ).to(dtype=context_mean.dtype)
            selected_terminal = selected_terminal * keep
            selected_path_anchors = selected_path_anchors * keep.view(-1, 1, 1)

        if self._bit_should_detach_condition(training):
            selected_terminal_for_injection = selected_terminal.detach()
            selected_path_anchors_for_injection = selected_path_anchors.detach()
        else:
            selected_terminal_for_injection = selected_terminal
            selected_path_anchors_for_injection = selected_path_anchors
        selected_terminal_for_injection, selected_path_anchors_for_injection = self._bit_apply_coordinate_mode(
            selected_terminal_for_injection,
            selected_path_anchors_for_injection,
        )

        terminal_token, path_tokens, target_path_summary = self.bit_condition_encoder(
            selected_terminal_for_injection,
            selected_path_anchors_for_injection,
            use_path_anchors=bool(self.config.bit_use_path_anchors and self.config.bit_enable_path_condition),
        )
        enabled_condition_tokens: list[torch.Tensor] = []
        if bool(self.config.bit_enable_terminal_condition):
            enabled_condition_tokens.append(terminal_token)
        if bool(self.config.bit_enable_path_condition and self.config.bit_use_path_anchors):
            enabled_condition_tokens.append(path_tokens)
        if enabled_condition_tokens:
            condition_tokens = torch.cat(enabled_condition_tokens, dim=1)
            target_path_summary = condition_tokens.mean(dim=1)
        else:
            condition_tokens = context_tokens.new_zeros(context_tokens.shape[0], 0, context_tokens.shape[-1])
            target_path_summary = context_mean.new_zeros(context_mean.shape)

        risk_logits = None
        risk_prob_mean = None
        risk_token = None
        if self.config.bit_use_risk_head and hasattr(self, "bit_risk_head"):
            risk_logits = self.bit_risk_head(
                context_mean,
                terminal_pred,
                path_anchor_pred,
                status_feature=status_feature,
                his_traj=his_traj,
            )
            risk_prob = torch.sigmoid(risk_logits.float()).to(dtype=context_mean.dtype)
            risk_prob_mean = risk_prob.mean(dim=0).detach()
            if bool(self.config.bit_use_risk_tokens or self.config.bit_use_risk_token) and hasattr(self, "bit_risk_token_encoder"):
                risk_token = self.bit_risk_token_encoder(risk_logits)

        context_gate = self._bit_gate_value("context", context_mean)
        action_gate = self._bit_gate_value("action", context_mean)
        context_strength = self._bit_condition_strength("context", context_mean)
        action_strength = self._bit_condition_strength("action", context_mean)

        if self._bit_effective_context_tokens_enabled() and condition_tokens.shape[1] > 0:
            context_tokens = torch.cat([context_tokens, condition_tokens * context_strength], dim=1)
        if risk_token is not None:
            risk_strength = context_mean.new_tensor(float(self.config.bit_risk_token_strength))
            context_tokens = torch.cat([context_tokens, risk_token * risk_strength], dim=1)
        action_condition = (
            target_path_summary * action_strength
            if self._bit_effective_action_add_enabled() and condition_tokens.shape[1] > 0
            else None
        )

        return {
            "context_tokens": context_tokens,
            "bit_action_condition": action_condition,
            "bit_terminal_pred": terminal_pred,
            "bit_path_anchor_pred": path_anchor_pred,
            "bit_terminal_gt": terminal_gt,
            "bit_path_gt": path_gt,
            "bit_selected_terminal": selected_terminal_for_injection,
            "bit_selected_path_anchors": selected_path_anchors_for_injection,
            "bit_used_gt_condition_rate": used_gt_condition_rate,
            "bit_context_gate_value": context_gate.detach(),
            "bit_action_gate_value": action_gate.detach(),
            "bit_gt_condition_prob_current": context_mean.new_tensor(current_gt_prob),
            "bit_condition_strength_context": context_strength.detach(),
            "bit_condition_strength_action": action_strength.detach(),
            "bit_risk_logits": risk_logits,
            "bit_risk_prob_mean": risk_prob_mean,
            "bit_risk_token_strength": context_mean.new_tensor(float(self.config.bit_risk_token_strength)),
        }

    def _augment_with_bit_condition(
        self,
        dit_context: Dict[str, Any],
        action_input: Optional[BatchFeature],
        *,
        training: bool,
        gt_actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if not self.config.use_bit_drive:
            dit_context.update(self._bit_zero_outputs(dit_context["context_mean"]))
            return dit_context
        bit = self._build_bit_condition(
            dit_context["context_mean"],
            dit_context["context_tokens"],
            action_input,
            gt_actions=gt_actions,
            training=training,
        )
        dit_context["context_tokens"] = bit.pop("context_tokens")
        dit_context.update(bit)
        return dit_context

    def _risk_vla_zero_outputs(self, reference: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = reference.new_zeros(())
        return {
            "risk_vla_risk_loss": zero,
            "risk_vla_focal_loss": zero,
            "risk_vla_strategy_entropy_loss": zero,
            "risk_vla_total_aux_loss": zero,
            "risk_vla_prob_low_score": zero,
            "risk_vla_prob_path_dac": zero,
            "risk_vla_prob_interaction_nc": zero,
            "risk_vla_prob_ttc": zero,
            "risk_vla_prob_progress": zero,
            "risk_vla_prob_comfort": zero,
            "risk_vla_weight_base": zero,
            "risk_vla_weight_path_intent": zero,
            "risk_vla_weight_interaction": zero,
            "risk_vla_weight_progress": zero,
            "risk_vla_weight_comfort": zero,
            "risk_vla_strategy_entropy": zero,
        }

    @staticmethod
    def _action_input_tensor(action_input: Optional[BatchFeature], *keys: str) -> Optional[torch.Tensor]:
        if action_input is None:
            return None
        for key in keys:
            value = None
            if isinstance(action_input, dict):
                value = action_input.get(key)
            elif hasattr(action_input, key):
                value = getattr(action_input, key)
            else:
                try:
                    value = action_input[key]
                except Exception:
                    value = None
            if isinstance(value, torch.Tensor):
                return value
        return None

    def _risk_vla_inputs(
        self,
        action_input: Optional[BatchFeature],
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = reference.shape[0]
        dtype = reference.dtype
        device = reference.device
        status = self._action_input_tensor(action_input, "status_feature")
        if status is None:
            status = reference.new_zeros(batch, 8)
        else:
            status = status.to(device=device, dtype=dtype)
        if status.ndim != 2 or status.shape[0] != batch:
            raise ValueError(f"status_feature must have shape [B, 8], got {tuple(status.shape)}.")
        if status.shape[-1] != 8:
            raise ValueError(f"status_feature must have last dimension 8, got {tuple(status.shape)}.")

        history = self._action_input_tensor(action_input, "history_trajectory", "his_traj")
        if history is None:
            history = reference.new_zeros(batch, 12)
        else:
            history = history.to(device=device, dtype=dtype)
        if history.shape[0] != batch or history.ndim not in {2, 3}:
            raise ValueError(f"history_trajectory/his_traj must be [B, 12] or [B, 4, 3], got {tuple(history.shape)}.")

        command = self._action_input_tensor(action_input, "high_command_one_hot")
        if command is None:
            command = reference.new_zeros(batch, 3)
        else:
            command = command.to(device=device, dtype=dtype)
        if command.ndim != 2 or command.shape[0] != batch:
            raise ValueError(f"high_command_one_hot must have shape [B, 3], got {tuple(command.shape)}.")
        if command.shape[-1] != 3:
            raise ValueError(f"high_command_one_hot must have last dimension 3, got {tuple(command.shape)}.")
        return status, history, command

    def _risk_vla_extract_labels(
        self,
        action_input: Optional[BatchFeature],
        reference: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if action_input is None:
            return None
        labels = self._action_input_tensor(action_input, "risk_labels")
        if labels is None:
            generic = self._action_input_tensor(action_input, "generic_risk_labels")
            drivable = self._action_input_tensor(action_input, "drivable_risk_labels")
            ttc = self._action_input_tensor(action_input, "ttc_risk_labels")
            comfort = self._action_input_tensor(action_input, "comfort_risk_labels")
            if all(tensor is not None for tensor in (generic, drivable, ttc, comfort)):
                from .risk_vla import mvp_labels_to_extended

                labels = mvp_labels_to_extended(generic, drivable, ttc, comfort)
        if labels is None:
            return None
        labels = labels.to(device=reference.device, dtype=reference.dtype)
        expected_classes = int(self.config.risk_vla_num_classes)
        if labels.ndim not in {2, 3}:
            raise ValueError(f"risk labels must be [B, K] or [B, H, K], got {tuple(labels.shape)}.")
        if labels.shape[0] != reference.shape[0] or labels.shape[-1] != expected_classes:
            raise ValueError(
                f"risk labels must have batch {reference.shape[0]} and {expected_classes} classes, got {tuple(labels.shape)}."
            )
        return labels.clamp(0.0, 1.0)

    @staticmethod
    def _risk_vla_detach_state(risk_state: Any) -> Any:
        from .risk_vla import RiskState

        return RiskState(
            risk_logits=risk_state.risk_logits.detach(),
            risk_probs=risk_state.risk_probs.detach(),
            risk_embedding=risk_state.risk_embedding.detach(),
            uncertainty=risk_state.uncertainty.detach() if risk_state.uncertainty is not None else None,
            diagnostics={key: value.detach() for key, value in risk_state.diagnostics.items()},
        )

    def _risk_vla_oracle_state(self, risk_state: Any, labels: Optional[torch.Tensor]) -> Any:
        if labels is None:
            raise RuntimeError(
                "risk_vla_use_oracle_router=True requires train/val risk labels in action_input. "
                "Oracle routing is analysis-only and must not be enabled for label-free inference."
            )
        from .risk_vla import RiskState, ensure_scene_level_labels

        probs = ensure_scene_level_labels(labels).to(device=risk_state.risk_probs.device, dtype=risk_state.risk_probs.dtype).clamp(
            1e-4, 1.0 - 1e-4
        )
        logits = torch.logit(probs)
        diagnostics = dict(risk_state.diagnostics)
        for idx, name in enumerate(("low_score", "path_dac", "interaction_nc", "ttc", "progress", "comfort")):
            if idx < probs.shape[-1]:
                diagnostics[f"risk_vla_oracle_prob_{name}"] = probs[:, idx].detach()
        return RiskState(
            risk_logits=logits,
            risk_probs=probs,
            risk_embedding=risk_state.risk_embedding,
            uncertainty=risk_state.uncertainty,
            diagnostics=diagnostics,
        )

    def _augment_with_risk_vla_condition(
        self,
        dit_context: Dict[str, Any],
        action_input: Optional[BatchFeature],
        *,
        training: bool,
    ) -> Dict[str, Any]:
        reference = dit_context["context_tokens"]
        if not self.config.use_risk_vla:
            dit_context.update(self._risk_vla_zero_outputs(reference))
            return dit_context

        from .risk_vla import (
            RiskState,
            extract_bit_intents,
            risk_bce_loss,
            risk_focal_loss,
            strategy_entropy_loss,
        )

        vl_embeds = dit_context["vl_embeds"]
        status_feature, history_trajectory, high_command_one_hot = self._risk_vla_inputs(action_input, vl_embeds)
        labels = self._risk_vla_extract_labels(action_input, vl_embeds)
        bit_terminal, bit_path, bit_diagnostics = extract_bit_intents(
            action_input,
            batch_size=vl_embeds.shape[0],
            action_horizon=int(self.config.action_horizon),
            action_dim=int(self.config.action_dim),
            reference=vl_embeds,
            use_bit_summary=bool(self.config.risk_vla_use_bit_summary),
            strict=bool(training),
        )
        risk_state = self.risk_state_encoder(
            vlm_tokens=vl_embeds,
            status_feature=status_feature,
            history_trajectory=history_trajectory,
            high_command_one_hot=high_command_one_hot,
            bit_terminal=bit_terminal,
            bit_path=bit_path,
        )

        strategy_state = risk_state
        if self.config.risk_vla_use_oracle_router:
            strategy_state = self._risk_vla_oracle_state(strategy_state, labels)
        if self.config.risk_vla_detach_risk_for_strategy:
            strategy_state = self._risk_vla_detach_state(strategy_state)

        strategy_weights = self.risk_strategy_router(strategy_state)
        strategy_output = self.risk_strategy_bank(
            vlm_tokens=vl_embeds,
            risk_state=strategy_state,
            strategy_weights=strategy_weights,
            status_feature=status_feature,
            history_trajectory=history_trajectory,
            high_command_one_hot=high_command_one_hot,
            bit_terminal=bit_terminal,
            bit_path=bit_path,
        )

        token_scale = float(self.config.risk_vla_strategy_token_scale)
        if token_scale > 0.0 and strategy_output.strategy_tokens.shape[1] > 0:
            scaled_tokens = strategy_output.strategy_tokens * strategy_output.strategy_tokens.new_tensor(token_scale)
            dit_context["context_tokens"] = torch.cat([dit_context["context_tokens"], scaled_tokens], dim=1)
            dit_context["context_mean"] = dit_context["context_tokens"].mean(dim=1)

        residual_scale = float(self.config.risk_vla_horizon_residual_scale)
        if residual_scale > 0.0:
            scaled_residual = strategy_output.horizon_residual * strategy_output.horizon_residual.new_tensor(residual_scale)
            current_condition = dit_context.get("expert_step_condition")
            dit_context["expert_step_condition"] = (
                scaled_residual
                if current_condition is None
                else current_condition.to(device=scaled_residual.device, dtype=scaled_residual.dtype) + scaled_residual
            )

        zero = vl_embeds.new_zeros(())
        risk_loss = zero
        focal_loss = zero
        entropy_loss = zero
        if labels is not None and float(self.config.risk_vla_risk_loss_weight) > 0.0:
            risk_loss = risk_bce_loss(risk_state.risk_logits, labels).to(dtype=vl_embeds.dtype)
        if labels is not None and float(self.config.risk_vla_focal_loss_weight) > 0.0:
            focal_loss = risk_focal_loss(risk_state.risk_logits, labels).to(dtype=vl_embeds.dtype)
        if float(self.config.risk_vla_strategy_entropy_weight) > 0.0:
            entropy_loss = strategy_entropy_loss(strategy_weights).to(dtype=vl_embeds.dtype)
        total_aux = (
            float(self.config.risk_vla_risk_loss_weight) * risk_loss
            + float(self.config.risk_vla_focal_loss_weight) * focal_loss
            + float(self.config.risk_vla_strategy_entropy_weight) * entropy_loss
        )

        diagnostics = dit_context.setdefault("diagnostics", {})
        if self.config.risk_vla_log_diagnostics:
            for key, value in bit_diagnostics.items():
                diagnostics[f"risk_vla_{key}"] = value.detach()
            for key, value in risk_state.diagnostics.items():
                diagnostics[f"risk_vla_{key}"] = value.detach()
            for key, value in strategy_weights.diagnostics.items():
                diagnostics[f"risk_vla_{key}"] = value.detach()
            for key, value in strategy_output.diagnostics.items():
                diagnostics[f"risk_vla_{key}"] = value.detach()

        prob_means = risk_state.risk_probs.detach().mean(dim=0)
        weight_means = {
            "base": strategy_weights.base.detach().mean(),
            "path_intent": strategy_weights.path_intent.detach().mean(),
            "interaction": strategy_weights.interaction.detach().mean(),
            "progress": strategy_weights.progress.detach().mean(),
            "comfort": strategy_weights.comfort.detach().mean(),
        }
        strategy_entropy = strategy_weights.diagnostics["strategy_entropy"].detach().mean()
        outputs = self._risk_vla_zero_outputs(vl_embeds)
        outputs.update(
            {
                "risk_vla_risk_loss": risk_loss,
                "risk_vla_focal_loss": focal_loss,
                "risk_vla_strategy_entropy_loss": entropy_loss,
                "risk_vla_total_aux_loss": total_aux,
                "risk_vla_strategy_entropy": strategy_entropy,
                "risk_vla_weight_base": weight_means["base"],
                "risk_vla_weight_path_intent": weight_means["path_intent"],
                "risk_vla_weight_interaction": weight_means["interaction"],
                "risk_vla_weight_progress": weight_means["progress"],
                "risk_vla_weight_comfort": weight_means["comfort"],
                "risk_vla_risk_logits": risk_state.risk_logits,
                "risk_vla_risk_probs": risk_state.risk_probs,
                "risk_vla_labels": labels,
            }
        )
        class_names = ("low_score", "path_dac", "interaction_nc", "ttc", "progress", "comfort")
        for idx, name in enumerate(class_names):
            if idx < prob_means.shape[0]:
                outputs[f"risk_vla_prob_{name}"] = prob_means[idx].to(dtype=vl_embeds.dtype)
        dit_context.update(outputs)
        return dit_context

    def _finalize_dit_context(
        self,
        dit_context: Dict[str, Any],
        action_input: Optional[BatchFeature],
        *,
        training: bool,
        gt_actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        dit_context = self._augment_with_bit_condition(
            dit_context,
            action_input,
            training=training,
            gt_actions=gt_actions,
        )
        return self._augment_with_risk_vla_condition(dit_context, action_input, training=training)

    def _apply_bit_action_condition(self, action_features: torch.Tensor, dit_context: Dict[str, Any]) -> torch.Tensor:
        bit_action_condition = dit_context.get("bit_action_condition")
        if bit_action_condition is None:
            return action_features
        return action_features + bit_action_condition.to(device=action_features.device, dtype=action_features.dtype).unsqueeze(1)

    def _prepare_dit_context(
        self,
        vl_features: torch.Tensor,
        action_input: Optional[BatchFeature],
        training: bool,
        noisy_actions: Optional[torch.Tensor] = None,
        diffusion_timestep: Optional[torch.Tensor] = None,
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
            "action_aware_aux_loss": zero,
            "action_aware_aux_l1": zero,
        }
        base_losses.update(self._bit_zero_outputs(zero))
        base_losses.update(self._risk_vla_zero_outputs(zero))
        if not self.config.use_expert_features and not self.config.use_last_rd:
            context_mean = vl_embeds.mean(1)
            return self._finalize_dit_context({
                "vl_embeds": vl_embeds,
                "context_tokens": vl_embeds,
                "context_mean": context_mean,
                "expert_step_condition": None,
                **base_losses,
                "diagnostics": self._vlm_context_diagnostics(vl_features, vl_embeds, context_mean),
            }, action_input, training=training)

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
            diagnostics = dict(expert["diagnostics"])
            diagnostics.update(self._vlm_context_diagnostics(vl_features, vl_embeds, context_mean))
            return self._finalize_dit_context({
                "vl_embeds": vl_embeds,
                "context_tokens": context_tokens,
                "context_mean": context_mean,
                "expert_step_condition": expert["horizon_residual"],
                **base_losses,
                "diagnostics": diagnostics,
            }, action_input, training=training)

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
        diagnostics.update(self._vlm_context_diagnostics(vl_features, vl_embeds, context_mean))
        for key, value in last_rd_output.losses.items():
            if key in base_losses:
                base_losses[key] = value
        return self._finalize_dit_context({
            "vl_embeds": vl_embeds,
            "context_tokens": context_tokens,
            "context_mean": context_mean,
            "expert_step_condition": expert_step_condition,
            **base_losses,
            "diagnostics": diagnostics,
            "last_rd_output": last_rd_output,
        }, action_input, training=training)

    def _repeat_expert_action_input(
        self,
        action_input: BatchFeature,
        repeat: int,
    ) -> Optional[BatchFeature]:
        if not self.config.use_expert_features and not self.config.use_last_rd:
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
            "vggt_depth_tokens",
            "vggt_pointmap_tokens",
            "vggt_camera_tokens",
        ):
            if key in action_input and isinstance(action_input[key], torch.Tensor):
                data[key] = action_input[key].repeat_interleave(repeat, 0)
        if self.config.use_expert_features:
            for stream, enabled in (("jepa", self.config.use_jepa), ("vggt", self.config.use_vggt)):
                if not enabled or f"{stream}_context_tokens" in data:
                    continue
                tokens = self._resolve_expert_tokens(action_input, stream, "context", required=True)
                assert tokens is not None
                data[f"{stream}_context_tokens"] = tokens.repeat_interleave(repeat, 0)
        return BatchFeature(data=data)

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

    def _compute_action_aware_aux_loss(
        self,
        dit_context: Dict[str, Any],
        action_input: BatchFeature,
        gt_actions_norm: torch.Tensor,
    ) -> torch.Tensor:
        reference = gt_actions_norm
        zero = reference.new_zeros(())
        if (
            not bool(self.config.use_action_aware_aux)
            or float(self.config.action_aware_aux_weight) <= 0.0
            or "action" not in action_input
        ):
            dit_context["action_aware_aux_loss"] = zero
            dit_context["action_aware_aux_l1"] = zero
            return zero

        coarse = self.action_aware_aux_head(dit_context["context_mean"]).view(
            -1,
            int(self.config.action_horizon),
            int(self.config.action_dim),
        )
        if self.config.action_aware_aux_space == "raw":
            target = action_input.action.to(device=coarse.device, dtype=coarse.dtype)
        else:
            target = gt_actions_norm.to(device=coarse.device, dtype=coarse.dtype)
        aux_loss = F.smooth_l1_loss(coarse, target, reduction="mean")
        aux_l1 = (coarse - target).abs().mean()
        dit_context["action_aware_aux_loss"] = aux_loss
        dit_context["action_aware_aux_l1"] = aux_l1
        return float(self.config.action_aware_aux_weight) * aux_loss

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
        his_traj_features = self.his_traj_encoder(
            action_input.his_traj.unsqueeze(1)
        ).repeat(1, self.config.action_horizon, 1)
        ego_status_features = self.ego_status_encoder(action_input.status_feature)

        action_features = self.action_encoder(noisy_actions, timesteps)
        action_features = self._apply_bit_action_condition(action_features, dit_context)
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

    @staticmethod
    def _masked_smooth_l1(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if pred.numel() == 0:
            return reference.new_zeros(())
        mask = mask.to(device=pred.device, dtype=pred.dtype).view(pred.shape[0])
        if mask.sum().item() <= 0:
            return reference.new_zeros(())
        per_item = F.smooth_l1_loss(pred.float(), target.detach().float(), reduction="none")
        per_item = per_item.reshape(pred.shape[0], -1).mean(dim=1).to(dtype=pred.dtype)
        return (per_item * mask).sum().to(dtype=reference.dtype) / mask.sum().clamp_min(1.0).to(dtype=reference.dtype)

    @staticmethod
    def _weighted_smooth_l1(
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if pred.numel() == 0:
            return reference.new_zeros(())
        weights = weights.to(device=pred.device, dtype=pred.dtype).view(pred.shape[0]).clamp_min(0.0)
        active = weights > 0
        if active.sum().item() <= 0:
            return reference.new_zeros(())
        per_item = F.smooth_l1_loss(pred.float(), target.detach().float(), reduction="none")
        per_item = per_item.reshape(pred.shape[0], -1).mean(dim=1).to(dtype=pred.dtype)
        # Divide by active row count so soft rows scale the effective KD weight down.
        denom = active.to(dtype=pred.dtype).sum().clamp_min(1.0)
        return (per_item * weights).sum().to(dtype=reference.dtype) / denom.to(dtype=reference.dtype)

    def _compute_d5_losses(
        self,
        action_input: BatchFeature,
        pred_x0_norm: Optional[torch.Tensor],
        losses: Dict[str, torch.Tensor],
        reference: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        zero = reference.new_zeros(())
        out = {
            "d5_safe_kd_loss": zero,
            "d5_dac_path_preserve_loss": zero,
            "d5_safety_mask_rate": zero,
            "d5_hard_safety_mask_rate": zero,
            "d5_soft_safety_mask_rate": zero,
            "d5_dac_fix_mask_rate": zero,
        }
        if not bool(self.config.bit_use_d5_conservative_loss):
            return out

        def optional_bool_mask(name: str) -> Optional[torch.Tensor]:
            value = action_input.get(name) if action_input is not None else None
            if not isinstance(value, torch.Tensor):
                return None
            return value.to(device=reference.device).bool().view(-1)

        base_safety_mask = optional_bool_mask("bit_safety_regression_mask")
        hard_safety_mask = optional_bool_mask("hard_safety_mask")
        soft_safety_mask = optional_bool_mask("soft_safety_mask")
        nc_mask = optional_bool_mask("bit_nc_regression_mask")
        ttc_mask = optional_bool_mask("bit_ttc_regression_mask")
        dac_fix_mask = action_input.get("bit_dac_fix_mask") if action_input is not None else None
        hard_parts: list[torch.Tensor] = []
        if hard_safety_mask is not None:
            hard_parts.append(hard_safety_mask)
        if bool(self.config.bit_safe_kd_apply_on_nc) and nc_mask is not None:
            hard_parts.append(nc_mask)
        if bool(self.config.bit_safe_kd_apply_on_ttc) and ttc_mask is not None:
            hard_parts.append(ttc_mask)
        if not hard_parts and base_safety_mask is not None and (
            bool(self.config.bit_safe_kd_apply_on_nc) or bool(self.config.bit_safe_kd_apply_on_ttc)
        ):
            hard_parts.append(base_safety_mask)
        hard_mask = torch.stack(hard_parts, dim=0).any(dim=0) if hard_parts else None
        if not bool(self.config.bit_safe_kd_apply_on_soft_safety):
            soft_safety_mask = None
        safety_mask = None
        if hard_mask is not None and soft_safety_mask is not None:
            safety_mask = hard_mask | soft_safety_mask
        elif hard_mask is not None:
            safety_mask = hard_mask
        elif soft_safety_mask is not None:
            safety_mask = soft_safety_mask
        if hard_mask is not None:
            out["d5_hard_safety_mask_rate"] = hard_mask.float().mean().to(dtype=reference.dtype)
        if soft_safety_mask is not None:
            out["d5_soft_safety_mask_rate"] = soft_safety_mask.float().mean().to(dtype=reference.dtype)
        if safety_mask is not None:
            out["d5_safety_mask_rate"] = safety_mask.float().mean().to(dtype=reference.dtype)
        if isinstance(dac_fix_mask, torch.Tensor):
            dac_fix_mask = dac_fix_mask.to(device=reference.device).bool().view(-1)
            out["d5_dac_fix_mask_rate"] = dac_fix_mask.float().mean().to(dtype=reference.dtype)

        safety_weight_tensor = action_input.get("safety_weight") if action_input is not None else None
        if isinstance(safety_weight_tensor, torch.Tensor):
            safety_weights = safety_weight_tensor.to(device=reference.device, dtype=reference.dtype).view(-1).clamp_min(0.0)
            if hard_mask is not None:
                safety_weights = torch.maximum(safety_weights, hard_mask.to(dtype=safety_weights.dtype))
            if soft_safety_mask is not None:
                soft_weights = soft_safety_mask.to(dtype=safety_weights.dtype) * float(self.config.bit_soft_safe_kd_weight_scale)
                safety_weights = torch.maximum(safety_weights, soft_weights)
            if safety_mask is not None:
                safety_weights = safety_weights * safety_mask.to(dtype=safety_weights.dtype)
        elif safety_mask is not None:
            safety_weights = safety_mask.to(dtype=reference.dtype)
            if soft_safety_mask is not None:
                soft_scale = float(self.config.bit_soft_safe_kd_weight_scale)
                safety_weights = torch.where(
                    soft_safety_mask,
                    safety_weights.new_tensor(soft_scale),
                    safety_weights,
                )
            if hard_mask is not None:
                safety_weights = torch.where(hard_mask, safety_weights.new_tensor(1.0), safety_weights)
        else:
            safety_weights = None

        base_pred_traj = action_input.get("base_pred_traj") if action_input is not None else None
        if (
            pred_x0_norm is not None
            and isinstance(base_pred_traj, torch.Tensor)
            and isinstance(safety_weights, torch.Tensor)
            and (safety_weights > 0).any()
        ):
            points = min(int(self.config.bit_safe_kd_points), pred_x0_norm.shape[1], base_pred_traj.shape[1])
            pred_raw = self.denorm_odo(pred_x0_norm[:, :points, :]).to(dtype=reference.dtype)
            base_raw = base_pred_traj[:, :points, :].to(device=pred_raw.device, dtype=pred_raw.dtype)
            safe_loss = zero
            if float(self.config.bit_safe_kd_x_weight) > 0.0:
                safe_loss = safe_loss + float(self.config.bit_safe_kd_x_weight) * self._weighted_smooth_l1(
                    pred_raw[..., 0],
                    base_raw[..., 0],
                    safety_weights,
                    reference,
                )
            if float(self.config.bit_safe_kd_y_weight) > 0.0:
                safe_loss = safe_loss + float(self.config.bit_safe_kd_y_weight) * self._weighted_smooth_l1(
                    pred_raw[..., 1],
                    base_raw[..., 1],
                    safety_weights,
                    reference,
                )
            if float(self.config.bit_safe_kd_heading_weight) > 0.0:
                safe_loss = safe_loss + float(self.config.bit_safe_kd_heading_weight) * self._weighted_smooth_l1(
                    pred_raw[..., 2],
                    base_raw[..., 2],
                    safety_weights,
                    reference,
                )
            if points > 1 and float(self.config.bit_safe_kd_step_weight) > 0.0:
                pred_step = torch.linalg.vector_norm(pred_raw[:, 1:, :2] - pred_raw[:, :-1, :2], dim=-1)
                base_step = torch.linalg.vector_norm(base_raw[:, 1:, :2] - base_raw[:, :-1, :2], dim=-1)
                safe_loss = safe_loss + float(self.config.bit_safe_kd_step_weight) * self._weighted_smooth_l1(
                    pred_step,
                    base_step,
                    safety_weights,
                    reference,
                )
            out["d5_safe_kd_loss"] = safe_loss.to(dtype=reference.dtype)

        if (
            bool(self.config.bit_dac_path_preserve_apply_on_dac_fix)
            and isinstance(dac_fix_mask, torch.Tensor)
            and dac_fix_mask.any()
            and losses.get("bit_path_loss") is not None
        ):
            path_pred = self._bit_loss_space(losses.get("_path_anchor_pred_raw"))
            path_gt = self._bit_loss_space(losses.get("_path_gt_raw"))
            if isinstance(path_pred, torch.Tensor) and isinstance(path_gt, torch.Tensor):
                out["d5_dac_path_preserve_loss"] = self._masked_smooth_l1(
                    path_pred,
                    path_gt,
                    dac_fix_mask,
                    reference,
                )
        return out

    def _compute_bit_losses(
        self,
        dit_context: Dict[str, Any],
        action_input: BatchFeature,
        pred_x0_norm: Optional[torch.Tensor],
    ) -> torch.Tensor:
        reference = pred_x0_norm if pred_x0_norm is not None else dit_context["context_mean"]
        if not self.config.use_bit_drive:
            dit_context.update(self._bit_zero_outputs(reference))
            return reference.new_zeros(())

        selected_terminal = dit_context.get("bit_selected_terminal")
        selected_path_anchors = dit_context.get("bit_selected_path_anchors")
        terminal_condition_norm = None
        pred_x0_terminal_norm = None
        if pred_x0_norm is not None and selected_terminal is not None:
            pred_x0_terminal_norm = pred_x0_norm[:, -1, :]
            terminal_condition_norm = self.norm_odo(selected_terminal.unsqueeze(1)).squeeze(1)

        reverse_points = None
        reverse_gt = None
        cycle_pred_norm = None
        cycle_target_norm = None
        if (
            self.config.bit_use_reverse_decoder
            and hasattr(self, "bit_reverse_decoder")
            and selected_terminal is not None
            and selected_path_anchors is not None
            and "action" in action_input
        ):
            reverse_points = self.bit_reverse_decoder(
                dit_context["context_mean"],
                selected_terminal,
                selected_path_anchors,
                use_path_anchors=bool(self.config.bit_use_path_anchors),
            )
            reverse_gt = action_input.action[:, : reverse_points.shape[1], :].to(
                device=reverse_points.device,
                dtype=reverse_points.dtype,
            )
            if pred_x0_norm is not None and self.config.bit_cycle_loss_weight > 0.0:
                cycle_pred_norm = pred_x0_norm[:, : reverse_points.shape[1], :]
                cycle_target_norm = self.norm_odo(reverse_points)

        base_longitudinal_loss = reference.new_zeros(())
        base_lateral_loss = reference.new_zeros(())
        if (
            self.config.bit_preserve_base_longitudinal
            and pred_x0_norm is not None
            and "base_pred_traj" in action_input
            and isinstance(action_input["base_pred_traj"], torch.Tensor)
        ):
            points = min(int(self.config.bit_base_early_points), pred_x0_norm.shape[1])
            pred_raw = self.denorm_odo(pred_x0_norm[:, :points, :]).to(dtype=reference.dtype)
            base_raw = action_input["base_pred_traj"][:, :points, :].to(device=pred_raw.device, dtype=pred_raw.dtype)
            base_longitudinal_loss = F.smooth_l1_loss(
                pred_raw[..., 0].float(),
                base_raw[..., 0].detach().float(),
                reduction="mean",
            ).to(dtype=reference.dtype)
            if float(self.config.bit_base_lateral_loss_weight) > 0.0:
                base_lateral_loss = F.smooth_l1_loss(
                    pred_raw[..., 1].float(),
                    base_raw[..., 1].detach().float(),
                    reduction="mean",
                ).to(dtype=reference.dtype)

        losses = bit_loss_bundle(
            terminal_pred=self._bit_loss_space(dit_context.get("bit_terminal_pred")),
            path_anchor_pred=self._bit_loss_space(dit_context.get("bit_path_anchor_pred")),
            terminal_gt=self._bit_loss_space(dit_context.get("bit_terminal_gt")),
            path_gt=self._bit_loss_space(dit_context.get("bit_path_gt")),
            pred_x0_terminal_norm=pred_x0_terminal_norm,
            terminal_condition_norm=terminal_condition_norm,
            reverse_points=reverse_points,
            reverse_gt=reverse_gt,
            cycle_pred_norm=cycle_pred_norm,
            cycle_target_norm=cycle_target_norm,
            reference=reference,
        )
        losses["_path_anchor_pred_raw"] = dit_context.get("bit_path_anchor_pred")
        losses["_path_gt_raw"] = dit_context.get("bit_path_gt")
        d5_losses = self._compute_d5_losses(action_input, pred_x0_norm, losses, reference)
        dit_context.update(losses)
        dit_context.update(d5_losses)
        dit_context["bit_base_longitudinal_loss"] = base_longitudinal_loss
        dit_context["bit_base_lateral_loss"] = base_lateral_loss
        dit_context["bit_reverse_points"] = reverse_points
        risk_loss = reference.new_zeros(())
        risk_parts = {
            "bit_risk_zero_loss": reference.new_zeros(()),
            "bit_risk_dac_loss": reference.new_zeros(()),
            "bit_risk_nc_loss": reference.new_zeros(()),
            "bit_risk_ttc_loss": reference.new_zeros(()),
        }
        risk_logits = dit_context.get("bit_risk_logits")
        risk_labels = action_input.get("bit_risk_labels") if action_input is not None else None
        if (
            self.config.bit_use_risk_head
            and risk_logits is not None
            and isinstance(risk_labels, torch.Tensor)
        ):
            labels = risk_labels.to(device=risk_logits.device, dtype=risk_logits.dtype)
            if labels.ndim != 2 or labels.shape[-1] != 4:
                raise ValueError(f"bit_risk_labels must have shape [B, 4], got {tuple(labels.shape)}.")
            valid_mask_bool = torch.isfinite(labels)
            safe_labels = torch.where(valid_mask_bool, labels, torch.zeros_like(labels))
            pos_weight = risk_logits.new_tensor([
                float(self.config.bit_risk_positive_weight_zero),
                float(self.config.bit_risk_positive_weight_dac),
                float(self.config.bit_risk_positive_weight_nc),
                float(self.config.bit_risk_positive_weight_ttc),
            ])
            element_loss = F.binary_cross_entropy_with_logits(
                risk_logits.float(),
                safe_labels.float(),
                pos_weight=pos_weight.float(),
                reduction="none",
            ).to(dtype=reference.dtype)
            valid_mask = valid_mask_bool.to(dtype=element_loss.dtype)
            element_loss = torch.where(valid_mask.bool(), element_loss, torch.zeros_like(element_loss))
            denom = valid_mask.sum(dim=0).clamp_min(1.0)
            per_risk = element_loss.sum(dim=0) / denom
            risk_loss = per_risk.mean()
            risk_parts = {
                "bit_risk_zero_loss": per_risk[0],
                "bit_risk_dac_loss": per_risk[1],
                "bit_risk_nc_loss": per_risk[2],
                "bit_risk_ttc_loss": per_risk[3],
        }
        dit_context["bit_risk_loss"] = risk_loss
        dit_context["bit_risk_bce_loss"] = risk_loss
        dit_context["risk_bce_loss"] = risk_loss
        dit_context.update(risk_parts)
        dit_context["bit_used_gt_condition_rate"] = dit_context.get(
            "bit_used_gt_condition_rate",
            reference.new_zeros(()),
        ).to(device=reference.device, dtype=reference.dtype)
        risk_weight = float(self.config.bit_risk_loss_weight)
        if bool(self.config.bit_use_d5_conservative_loss):
            risk_weight += float(self.config.bit_risk_bce_loss_weight)
        return (
            float(self.config.bit_terminal_loss_weight) * losses["bit_terminal_loss"]
            + float(self.config.bit_path_loss_weight) * losses["bit_path_loss"]
            + float(self.config.bit_end_consistency_loss_weight) * losses["bit_end_consistency_loss"]
            + float(self.config.bit_reverse_loss_weight) * losses["bit_reverse_loss"]
            + float(self.config.bit_cycle_loss_weight) * losses["bit_cycle_loss"]
            + float(self.config.bit_base_longitudinal_loss_weight) * base_longitudinal_loss
            + float(self.config.bit_base_lateral_loss_weight) * base_lateral_loss
            + float(self.config.bit_safe_kd_loss_weight) * d5_losses["d5_safe_kd_loss"]
            + float(self.config.bit_dac_path_preserve_weight) * d5_losses["d5_dac_path_preserve_loss"]
            + risk_weight * risk_loss
        ).to(dtype=reference.dtype)

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
        bit_action_condition: Optional[torch.Tensor] = None,
        vl_features: Optional[torch.Tensor] = None,
        action_input: Optional[BatchFeature] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the mean and log variance of the reverse process p(x_{t-1} | x_t).
        Also returns the predicted x0.
        """
        if self.config.use_last_rd and vl_features is not None and action_input is not None:
            last_rd_context = self._prepare_dit_context(
                vl_features,
                action_input,
                training=False,
                noisy_actions=x,
                diffusion_timestep=t,
            )
            context_embeds = last_rd_context["context_tokens"]
            context_mean = last_rd_context["context_mean"]
            expert_step_condition = last_rd_context["expert_step_condition"]
            bit_action_condition = last_rd_context.get("bit_action_condition")
        model_dtype = next(self.model.parameters()).dtype
        x = x.to(model_dtype)
        action_features = self.action_encoder(x, t)
        if bit_action_condition is not None:
            action_features = action_features + bit_action_condition.to(
                device=action_features.device,
                dtype=action_features.dtype,
            ).unsqueeze(1)
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
            timesteps=t
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
        if self.config.use_bit_drive and self.training:
            self._bit_train_step = int(getattr(self, "_bit_train_step", 0)) + 1
        gt_actions = self.norm_odo(action_input.action)
        allow_target_tokens = self.training or bool(action_input.get("_allow_target_tokens_for_loss", False))
        if not allow_target_tokens:
            self._warn_if_expert_targets_present(action_input, "forward")

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
            action_aware_aux_loss = self._compute_action_aware_aux_loss(dit_context, action_input, gt_actions)
            loss = (
                self._stream_alignment_weight("jepa") * jepa_alignment_loss
                + self._stream_alignment_weight("vggt") * vggt_alignment_loss
                + self._last_rd_aux_loss(dit_context, diffusion_loss.dtype)
                + action_aware_aux_loss.to(dtype=diffusion_loss.dtype)
                + dit_context["risk_vla_total_aux_loss"].to(dtype=diffusion_loss.dtype)
            )
            return self._format_training_output(loss, diffusion_loss, jepa_alignment_loss, vggt_alignment_loss, dit_context)

        if self.config.sampling_method == 'flow':
            noise = torch.randn_like(gt_actions)
            t_cont = self.sample_time(gt_actions.shape[0], device=gt_actions.device, dtype=gt_actions.dtype)
            t_cont_reshaped = t_cont[:, None, None]
            
            noisy_actions = (1 - t_cont_reshaped) * noise + t_cont_reshaped * gt_actions
            velocity_target = gt_actions - noise
            t_discrete = (t_cont * self.num_timestep_buckets).long()
            dit_context = self._prepare_dit_context(
                vl_features,
                action_input,
                training=self.training,
                noisy_actions=noisy_actions,
                diffusion_timestep=t_discrete,
                allow_target_tokens=allow_target_tokens,
            )
            pred_velocity = self._denoise_model_output(noisy_actions, t_discrete, dit_context, action_input)
            diffusion_loss = F.mse_loss(pred_velocity, velocity_target, reduction='mean')
            pred_x0_norm = noisy_actions + (1 - t_cont_reshaped) * pred_velocity
            policy_kd_loss = diffusion_loss.new_zeros(())
        else: 
            noise = torch.randn_like(gt_actions)
            t_discrete = self.sample_time(gt_actions.shape[0], device=gt_actions.device, dtype=gt_actions.dtype)
            
            noisy_actions = (
                self.extract(self.ddpm_sqrt_alphas_cumprod, t_discrete, gt_actions.shape) * gt_actions +
                self.extract(self.ddpm_sqrt_one_minus_alphas_cumprod, t_discrete, gt_actions.shape) * noise
            )
            dit_context = self._prepare_dit_context(
                vl_features,
                action_input,
                training=self.training,
                noisy_actions=noisy_actions,
                diffusion_timestep=t_discrete,
                allow_target_tokens=allow_target_tokens,
            )
            pred_noise = self._denoise_model_output(noisy_actions, t_discrete, dit_context, action_input)
            diffusion_loss = F.mse_loss(pred_noise, noise, reduction='mean')
            pred_x0_norm = self._x0_from_noise(noisy_actions, t_discrete, pred_noise)
            policy_kd_loss = self._compute_policy_kd_loss(vl_features, action_input, noisy_actions, t_discrete, pred_noise)
        dit_context["policy_kd_loss"] = policy_kd_loss.to(dtype=diffusion_loss.dtype)

        jepa_alignment_loss = dit_context["jepa_alignment_loss"].to(dtype=diffusion_loss.dtype)
        vggt_alignment_loss = dit_context["vggt_alignment_loss"].to(dtype=diffusion_loss.dtype)
        bit_aux_loss = self._compute_bit_losses(dit_context, action_input, pred_x0_norm).to(dtype=diffusion_loss.dtype)
        action_aware_aux_loss = self._compute_action_aware_aux_loss(dit_context, action_input, gt_actions).to(
            dtype=diffusion_loss.dtype
        )
        loss = (
            float(self.config.diffusion_loss_weight) * diffusion_loss
            + self._stream_alignment_weight("jepa") * jepa_alignment_loss
            + self._stream_alignment_weight("vggt") * vggt_alignment_loss
            + self._last_rd_aux_loss(dit_context, diffusion_loss.dtype)
            + bit_aux_loss
            + action_aware_aux_loss
            + dit_context["risk_vla_total_aux_loss"].to(dtype=diffusion_loss.dtype)
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
            "action_aware_aux_loss": dit_context["action_aware_aux_loss"].to(dtype=loss.dtype),
            "action_aware_aux_l1": dit_context["action_aware_aux_l1"].to(dtype=loss.dtype),
            "bit_terminal_loss": dit_context["bit_terminal_loss"].to(dtype=loss.dtype),
            "bit_path_loss": dit_context["bit_path_loss"].to(dtype=loss.dtype),
            "bit_end_consistency_loss": dit_context["bit_end_consistency_loss"].to(dtype=loss.dtype),
            "bit_reverse_loss": dit_context["bit_reverse_loss"].to(dtype=loss.dtype),
            "bit_cycle_loss": dit_context["bit_cycle_loss"].to(dtype=loss.dtype),
            "bit_terminal_pred_mean_l2": dit_context["bit_terminal_pred_mean_l2"].to(dtype=loss.dtype),
            "bit_path_pred_mean_l2": dit_context["bit_path_pred_mean_l2"].to(dtype=loss.dtype),
            "bit_used_gt_condition_rate": dit_context["bit_used_gt_condition_rate"].to(dtype=loss.dtype),
            "bit_context_gate_value": dit_context["bit_context_gate_value"].to(dtype=loss.dtype),
            "bit_action_gate_value": dit_context["bit_action_gate_value"].to(dtype=loss.dtype),
            "bit_gt_condition_prob_current": dit_context["bit_gt_condition_prob_current"].to(dtype=loss.dtype),
            "bit_condition_strength_context": dit_context["bit_condition_strength_context"].to(dtype=loss.dtype),
            "bit_condition_strength_action": dit_context["bit_condition_strength_action"].to(dtype=loss.dtype),
            "bit_base_longitudinal_loss": dit_context["bit_base_longitudinal_loss"].to(dtype=loss.dtype),
            "bit_base_lateral_loss": dit_context["bit_base_lateral_loss"].to(dtype=loss.dtype),
            "bit_risk_loss": dit_context["bit_risk_loss"].to(dtype=loss.dtype),
            "bit_risk_bce_loss": dit_context["bit_risk_bce_loss"].to(dtype=loss.dtype),
            "risk_bce_loss": dit_context["risk_bce_loss"].to(dtype=loss.dtype),
            "bit_risk_zero_loss": dit_context["bit_risk_zero_loss"].to(dtype=loss.dtype),
            "bit_risk_dac_loss": dit_context["bit_risk_dac_loss"].to(dtype=loss.dtype),
            "bit_risk_nc_loss": dit_context["bit_risk_nc_loss"].to(dtype=loss.dtype),
            "bit_risk_ttc_loss": dit_context["bit_risk_ttc_loss"].to(dtype=loss.dtype),
            "bit_risk_token_strength": dit_context["bit_risk_token_strength"].to(dtype=loss.dtype),
            "d5_safe_kd_loss": dit_context["d5_safe_kd_loss"].to(dtype=loss.dtype),
            "d5_dac_path_preserve_loss": dit_context["d5_dac_path_preserve_loss"].to(dtype=loss.dtype),
            "d5_safety_mask_rate": dit_context["d5_safety_mask_rate"].to(dtype=loss.dtype),
            "d5_hard_safety_mask_rate": dit_context["d5_hard_safety_mask_rate"].to(dtype=loss.dtype),
            "d5_soft_safety_mask_rate": dit_context["d5_soft_safety_mask_rate"].to(dtype=loss.dtype),
            "d5_dac_fix_mask_rate": dit_context["d5_dac_fix_mask_rate"].to(dtype=loss.dtype),
        }
        if self.config.use_risk_vla:
            for key in (
                "risk_vla_risk_loss",
                "risk_vla_focal_loss",
                "risk_vla_strategy_entropy_loss",
                "risk_vla_total_aux_loss",
                "risk_vla_prob_low_score",
                "risk_vla_prob_path_dac",
                "risk_vla_prob_interaction_nc",
                "risk_vla_prob_ttc",
                "risk_vla_prob_progress",
                "risk_vla_prob_comfort",
                "risk_vla_weight_base",
                "risk_vla_weight_path_intent",
                "risk_vla_weight_interaction",
                "risk_vla_weight_progress",
                "risk_vla_weight_comfort",
                "risk_vla_strategy_entropy",
            ):
                output[key] = dit_context[key].to(dtype=loss.dtype)
        risk_prob_mean = dit_context.get("bit_risk_prob_mean")
        if isinstance(risk_prob_mean, torch.Tensor) and risk_prob_mean.numel() == 4:
            risk_prob_mean = risk_prob_mean.to(device=loss.device, dtype=loss.dtype)
            output["bit_risk_zero_prob_mean"] = risk_prob_mean[0]
            output["bit_risk_dac_prob_mean"] = risk_prob_mean[1]
            output["bit_risk_nc_prob_mean"] = risk_prob_mean[2]
            output["bit_risk_ttc_prob_mean"] = risk_prob_mean[3]
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
            for key in ("last_rd_token_norms", "coarse_traj_l1", "future_jepa_loss_raw", "geometry_mode_code"):
                if key in diagnostics and isinstance(diagnostics[key], torch.Tensor):
                    output[f"last_rd_{key}" if key == "geometry_mode_code" else key] = diagnostics[key].to(
                        device=loss.device,
                        dtype=loss.dtype,
                    )
            for key in (
                "vlm_hidden_input_norm",
                "vlm_context_token_norm",
                "vlm_context_mean_norm",
                "context_token_mean_abs",
                "context_token_std",
            ):
                if key in diagnostics and isinstance(diagnostics[key], torch.Tensor):
                    output[key] = diagnostics[key].to(device=loss.device, dtype=loss.dtype)
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
        dit_context = self._prepare_dit_context(vl_features, action_input, training=False)
        context_embeds = dit_context["context_tokens"]
        context_mean = dit_context["context_mean"]
        expert_step_condition = dit_context["expert_step_condition"]
        bit_action_condition = dit_context.get("bit_action_condition")
        
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
                if self.config.use_last_rd:
                    dit_context = self._prepare_dit_context(
                        vl_features,
                        action_input,
                        training=False,
                        noisy_actions=current_actions,
                        diffusion_timestep=t,
                    )
                    context_embeds = dit_context["context_tokens"]
                    context_mean = dit_context["context_mean"]
                    expert_step_condition = dit_context["expert_step_condition"]
                    bit_action_condition = dit_context.get("bit_action_condition")

                action_features = self.action_encoder(current_actions, t)
                if bit_action_condition is not None:
                    action_features = action_features + bit_action_condition.to(
                        device=action_features.device,
                        dtype=action_features.dtype,
                    ).unsqueeze(1)
                if hasattr(self, 'position_embedding'):
                    action_features += self.position_embedding(torch.arange(self.config.action_horizon, device=device))
                
                context_mean_features = context_mean.unsqueeze(1).repeat(1, self.config.action_horizon, 1)
                fused_input = self.fusion_projector(
                    torch.cat((history_embeds, context_mean_features, action_features), dim=2)
                )
                if expert_step_condition is not None:
                    fused_input = fused_input + expert_step_condition.to(device=fused_input.device, dtype=fused_input.dtype)
                
                model_output = self.model(fused_input, context_embeds, ego_embeds, t)
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
                    bit_action_condition=bit_action_condition,
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
                    bit_action_condition=bit_action_condition,
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

        final_actions = self.denorm_odo(current_actions)

        result = {"pred_traj": final_actions}
        if self.config.use_bit_drive and self.config.bit_log_diagnostics:
            if dit_context.get("bit_terminal_pred") is not None:
                result["bit_terminal_pred"] = dit_context["bit_terminal_pred"].detach()
            if dit_context.get("bit_path_anchor_pred") is not None:
                result["bit_path_anchor_pred"] = dit_context["bit_path_anchor_pred"].detach()
        return BatchFeature(data=result)

    def sample_chain(
        self,
        vl_features: torch.Tensor,
        his_traj_features: torch.Tensor,
        ego_status_features: torch.Tensor,
        init_actions: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        action_input: Optional[BatchFeature] = None,
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
        if not self.training:
            self._warn_if_expert_targets_present(action_input, "sample_chain")
        dit_context = self._prepare_dit_context(vl_features, action_input, training=self.training)
        context_embeds = dit_context["context_tokens"]
        context_mean = dit_context["context_mean"]
        expert_step_condition = dit_context["expert_step_condition"]
        bit_action_condition = dit_context.get("bit_action_condition")
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
                if self.config.use_last_rd and action_input is not None:
                    dit_context = self._prepare_dit_context(
                        vl_features,
                        action_input,
                        training=self.training,
                        noisy_actions=current_actions,
                        diffusion_timestep=t_batch,
                    )
                    context_embeds = dit_context["context_tokens"]
                    context_mean = dit_context["context_mean"]
                    expert_step_condition = dit_context["expert_step_condition"]
                    bit_action_condition = dit_context.get("bit_action_condition")
                
                action_features = self.action_encoder(current_actions, t_batch)
                if bit_action_condition is not None:
                    action_features = action_features + bit_action_condition.to(
                        device=action_features.device,
                        dtype=action_features.dtype,
                    ).unsqueeze(1)
                if hasattr(self, 'position_embedding'):
                    action_features += self.position_embedding(torch.arange(self.config.action_horizon, device=device))
                
                context_mean_features = context_mean.unsqueeze(1).repeat(1, self.config.action_horizon, 1)
                fused_input = self.fusion_projector(
                    torch.cat((his_traj_features, context_mean_features, action_features), dim=2)
                )
                if expert_step_condition is not None:
                    fused_input = fused_input + expert_step_condition.to(device=fused_input.device, dtype=fused_input.dtype)
                
                model_output = self.model(fused_input, context_embeds, ego_status_features, t_batch)
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
                    bit_action_condition=bit_action_condition,
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

        final_actions = self.denorm_odo(current_actions)
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
        bit_action_condition = dit_context.get("bit_action_condition")

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
        if bit_action_condition is not None:
            conditioning_embeds['bit_action_condition'] = bit_action_condition
        
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
            bit_action_condition=batched_conditioning.get('bit_action_condition'),
        )

        std = torch.exp(0.5 * logvar).clamp(min=self.min_logprob_denoising_std)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(x_t_minus_1)
        
        return log_prob

    def forward_grpo(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        tokens_list,
        sample_time: int = 8,
        deterministic=False,
        bc_coeff: float = 0.1,
        use_bc_loss: bool = True
    ) -> BatchFeature:
        """Computes the Diffusion-GRPO loss."""
        self.set_frozen_modules_to_eval_mode()
        B = vl_features.shape[0]
        G = sample_time 

        vl_features_rep = vl_features.repeat_interleave(G, 0)
        his_traj_rep = action_input.his_traj.repeat_interleave(G, 0)
        status_feature_rep = action_input.status_feature.repeat_interleave(G, 0)
        expert_action_input_rep = self._repeat_expert_action_input(action_input, G)

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
        rewards = self.reward_fn(trajs, tokens_rep, metric_cache)

        rewards_matrix = rewards.view(B, G)
        mean_r = rewards_matrix.mean(dim=1, keepdim=True)
        std_r = rewards_matrix.std(dim=1, keepdim=True) + 1e-8
        advantages = ((rewards_matrix - mean_r) / std_r).view(-1).detach()
        
        adv_min = torch.quantile(advantages, self.clip_advantage_lower_quantile)
        adv_max = torch.quantile(advantages, self.clip_advantage_upper_quantile)
        advantages = advantages.clamp(min=adv_min, max=adv_max)

        num_denoising_steps = chains.shape[1] - 1
        denoising_indices = torch.arange(num_denoising_steps, device=advantages.device)
        discount = (self.gamma_denoising ** (num_denoising_steps - denoising_indices - 1))
    
        
        adv_steps = advantages.view(B, G, 1).expand(-1, -1, num_denoising_steps)  
        discount = discount.view(1, 1, num_denoising_steps).expand(B, G, num_denoising_steps)  
        adv_weighted_flat = (adv_steps * discount).reshape(-1)             

        log_probs = self.get_logprobs(
            vl_features_rep,
            his_traj_rep,
            status_feature_rep,
            chains,
            deterministic=False,
            action_input=expert_action_input_rep,
        )
        log_probs = log_probs.clamp(min=-5, max=2).mean(dim=[1, 2])
        
        policy_loss = -torch.mean(log_probs * adv_weighted_flat)
        total_loss = policy_loss

        bc_loss = 0.0
        if use_bc_loss:
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
            total_loss = total_loss + bc_coeff * bc_loss

        zero_loss = total_loss.new_zeros(())
        return BatchFeature(data={
            "loss": total_loss,
            "diffusion_loss": zero_loss,
            "jepa_alignment_loss": zero_loss,
            "vggt_alignment_loss": zero_loss,
            "reward": rewards.mean(),
            "policy_loss": policy_loss,
            "bc_loss": bc_loss,
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

    def reward_fn(
        self,
        pred_traj: torch.Tensor,
        tokens_list,
        cache_dict,
    ) -> torch.Tensor:
        """Calculates PDM scores for a batch of predicted trajectories."""
        pred_np = pred_traj.detach().cpu().numpy()
        rewards = []
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
            rewards.append(asdict(pdm_result)["score"])
        return torch.tensor(rewards, device=pred_traj.device, dtype=pred_traj.dtype).detach()

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
