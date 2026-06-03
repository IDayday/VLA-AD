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
        "teacher_traj_sft",
    ] = "disabled"
    last_vla_cot_num_tokens: int = 32
    last_vla_cot_num_steps: int = 4
    last_vla_vlm_summary_tokens: int = 4
    last_vla_raw_vlm_context_to_dit: bool = False
    last_vla_cot_bottleneck_mode: bool = True
    last_vla_vlm_context_dropout_start: float = 0.0
    last_vla_vlm_context_dropout_end: float = 0.7
    last_vla_use_geometry_step: bool = True
    last_vla_use_dynamic_step: bool = True
    last_vla_use_ego_step: bool = True
    last_vla_use_action_refine_step: bool = True
    last_vla_use_action_conditioned_dynamics: bool = True
    last_vla_use_risk_head: bool = True
    last_vla_require_full_geometry: bool = False
    last_vla_allow_patch_geometry_fallback: bool = False
    last_vla_use_residual_diffusion: bool = True
    last_vla_residual_detach_coarse: bool = True
    last_vla_coarse_prior_clip: float = 1.0
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
    last_vla_vlm_lora_r: int = 16
    last_vla_vlm_lora_alpha: int = 32
    last_vla_vlm_lora_target_modules: str = ""
    
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
        if config.use_last_vla and config.last_vla_stage == "disabled":
            raise ValueError("use_last_vla=True requires last_vla_stage to be non-disabled.")
        if not config.use_last_vla and config.last_vla_stage != "disabled":
            raise ValueError("last_vla_stage must be 'disabled' when use_last_vla=False.")
        if config.last_vla_require_full_geometry and config.last_vla_allow_patch_geometry_fallback:
            raise ValueError("last_vla_require_full_geometry=True is incompatible with patch fallback.")
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
        ):
            if getattr(config, weight_name) < 0.0:
                raise ValueError(f"{weight_name} must be non-negative.")

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
                    vlm_summary_tokens=config.last_vla_vlm_summary_tokens,
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
                    use_residual_diffusion=config.last_vla_use_residual_diffusion,
                    residual_detach_coarse_for_diffusion=config.last_vla_residual_detach_coarse,
                    coarse_prior_clip=config.last_vla_coarse_prior_clip,
                    require_full_geometry=config.last_vla_require_full_geometry,
                    allow_patch_geometry_fallback=config.last_vla_allow_patch_geometry_fallback,
                    geometry_teacher_dim=config.vggt_dim,
                    teacher_traj_mode=config.last_vla_teacher_traj_mode,
                    teacher_traj_mix_start=config.last_vla_teacher_traj_mix_start,
                    teacher_traj_mix_end=config.last_vla_teacher_traj_mix_end,
                )
            )

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
        if not self.config.use_expert_features and not self.config.use_last_rd and not self.config.use_last_vla:
            return {
                "vl_embeds": vl_embeds,
                "context_tokens": vl_embeds,
                "context_mean": vl_embeds.mean(1),
                "expert_step_condition": None,
                **base_losses,
                "diagnostics": {},
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
            return {
                "vl_embeds": vl_embeds,
                "context_tokens": last_vla_output.planner_context_tokens,
                "context_mean": last_vla_output.context_mean,
                "expert_step_condition": last_vla_output.horizon_condition,
                **base_losses,
                "diagnostics": dict(last_vla_output.diagnostics),
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
        if not self.config.use_expert_features and not self.config.use_last_rd and not self.config.use_last_vla:
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
            "risk_labels",
            "generic_risk_labels",
            "drivable_risk_labels",
            "ttc_risk_labels",
            "comfort_risk_labels",
            "last_vla_corrupt_zero_all_cot",
            "last_vla_corrupt_geometry_cot",
            "last_vla_corrupt_dynamic_cot",
            "last_vla_corrupt_ego_cot",
            "last_vla_corrupt_action_refine_cot",
            "last_vla_zero_coarse_prior",
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

    def _last_vla_progress(self) -> float:
        return min(
            max(float(self.config.current_train_epoch) / max(1.0, float(self.config.total_train_epochs)), 0.0),
            1.0,
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
        if self.config.last_vla_stage != "progressive_sft_bottleneck":
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
        return floor + max(float(base) - floor, 0.0) * (1.0 - self._last_vla_progress())

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
        vl_features: Optional[torch.Tensor] = None,
        action_input: Optional[BatchFeature] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the mean and log variance of the reverse process p(x_{t-1} | x_t).
        Also returns the predicted x0.
        """
        if (self.config.use_last_rd or self.config.use_last_vla) and vl_features is not None and action_input is not None:
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
                dit_context["policy_kd_loss"] = zero
                loss = self._last_vla_aux_loss(dit_context, gt_actions.dtype)
                return self._format_training_output(loss, zero, zero, zero, dit_context)

            if self.config.last_vla_use_residual_diffusion:
                prior_context = self._prepare_dit_context(
                    vl_features,
                    action_input,
                    training=self.training,
                    target_action_norm=selected_target_norm,
                    allow_target_tokens=allow_target_tokens,
                )
                prior_output = prior_context["last_vla_output"]
                diffusion_target = prior_output.residual_target_norm
                if diffusion_target is None:
                    coarse = prior_output.coarse_traj_norm
                    base = coarse.detach() if self.config.last_vla_residual_detach_coarse else coarse
                    diffusion_target = selected_target_norm - base
            else:
                diffusion_target = selected_target_norm

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
                pred_noise = self._denoise_model_output(noisy_actions, t_discrete, dit_context, action_input)
                diffusion_loss = F.mse_loss(pred_noise, noise, reduction="mean")
                policy_kd_loss = self._compute_policy_kd_loss(vl_features, action_input, noisy_actions, t_discrete, pred_noise)

            dit_context["policy_kd_loss"] = policy_kd_loss.to(dtype=diffusion_loss.dtype)
            dit_context["diagnostics"].update(target_diagnostics)
            if self.config.last_vla_use_residual_diffusion:
                dit_context["residual_target_norm"] = diffusion_target
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
                "teacher_traj_used_ratio",
                "teacher_traj_mix",
                "teacher_score_mean",
                "gt_score_mean",
                "geometry_mode_code",
                "cot_token_norm",
                "vlm_summary_norm",
                "cot_bottleneck_active",
                "raw_vlm_context_used",
                "coarse_traj_l1",
                "dynamic_loss_raw",
                "geometry_loss_raw",
            ):
                if key in diagnostics and isinstance(diagnostics[key], torch.Tensor):
                    output[f"last_vla_{key}"] = diagnostics[key].to(device=loss.device, dtype=loss.dtype)
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
        coarse_prior_norm = None
        if self.config.use_last_vla and self.config.last_vla_use_residual_diffusion:
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
                if self.config.use_last_rd or self.config.use_last_vla:
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

                action_features = self.action_encoder(current_actions, t)
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

        final_norm = current_actions
        output_data: Dict[str, torch.Tensor] = {}
        if coarse_prior_norm is not None:
            final_norm = coarse_prior_norm.to(current_actions) + current_actions
            output_data["pred_coarse_traj"] = self.denorm_odo(coarse_prior_norm.to(current_actions))
            output_data["pred_residual_norm"] = current_actions.detach()

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
        dit_context = self._prepare_dit_context(vl_features, action_input, training=self.training, allow_target_tokens=self.training)
        context_embeds = dit_context["context_tokens"]
        context_mean = dit_context["context_mean"]
        expert_step_condition = dit_context["expert_step_condition"]
        coarse_prior_norm = None
        if self.config.use_last_vla and self.config.last_vla_use_residual_diffusion:
            coarse_prior_norm = dit_context["last_vla_output"].coarse_traj_norm.detach()
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
                if (self.config.use_last_rd or self.config.use_last_vla) and action_input is not None:
                    dit_context = self._prepare_dit_context(
                        vl_features,
                        action_input,
                        training=self.training,
                        noisy_actions=current_actions,
                        diffusion_timestep=t_batch,
                        allow_target_tokens=self.training,
                    )
                    context_embeds = dit_context["context_tokens"]
                    context_mean = dit_context["context_mean"]
                    expert_step_condition = dit_context["expert_step_condition"]
                
                action_features = self.action_encoder(current_actions, t_batch)
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

        final_norm = current_actions if coarse_prior_norm is None else coarse_prior_norm.to(current_actions) + current_actions
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
