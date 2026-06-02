from typing import Any, List, Dict, Optional, Union
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
        metric_cache_path: Optional[str] = '', 
        reference_policy_checkpoint: Optional[str] = '', 
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
        use_bit_drive: bool = False,
        bit_terminal_loss_weight: float = 0.05,
        bit_path_loss_weight: float = 0.05,
        bit_end_consistency_loss_weight: float = 0.03,
        bit_reverse_loss_weight: float = 0.03,
        bit_cycle_loss_weight: float = 0.00,
        bit_num_path_anchors: int = 4,
        bit_path_anchor_indices: tuple = (1, 3, 5, 7),
        bit_condition_mode: str = "context_tokens",
        bit_use_gt_condition_prob: float = 0.20,
        bit_condition_noise_std: float = 0.05,
        bit_condition_dropout: float = 0.10,
        bit_reverse_decoder_hidden_dim: int = 384,
        bit_use_reverse_decoder: bool = False,
        bit_use_path_anchors: bool = True,
        bit_use_terminal_head: bool = True,
        bit_fail_if_gt_condition_in_eval: bool = True,
        bit_log_diagnostics: bool = True,
        bit_context_condition_strength: float = 0.10,
        bit_action_condition_strength: float = 0.00,
        bit_learnable_condition_gates: bool = True,
        bit_context_gate_init: float = 0.05,
        bit_action_gate_init: float = 0.00,
        bit_detach_condition: bool = True,
        bit_detach_condition_until_step: int = 100000,
        bit_use_gt_condition_schedule: str = "constant",
        bit_use_gt_condition_prob_start: float = 0.20,
        bit_use_gt_condition_prob_end: float = 0.00,
        bit_use_gt_condition_decay_steps: int = 1000,
        bit_loss_normalization: str = "norm_odo",
        bit_enable_terminal_condition: bool = True,
        bit_enable_path_condition: bool = True,
        bit_enable_action_add: bool = False,
        bit_enable_context_tokens: bool = True,
        bit_condition_coordinate_mode: str = "full",
        bit_condition_axis_scale_x: float = 0.10,
        bit_condition_axis_scale_y: float = 1.00,
        bit_condition_axis_scale_heading: float = 1.00,
        bit_preserve_base_longitudinal: bool = False,
        bit_base_longitudinal_loss_weight: float = 0.00,
        bit_base_early_points: int = 3,
        bit_base_lateral_loss_weight: float = 0.00,
        bit_store_base_traj_for_loss: bool = False,
        bit_use_safety_fallback: bool = False,
        bit_fallback_mode: str = "rule",
        bit_fallback_early_x_delta_threshold: float = 1.0,
        bit_fallback_terminal_x_delta_threshold: float = 2.0,
        bit_use_risk_head: bool = False,
        bit_use_risk_tokens: bool = False,
        bit_risk_loss_weight: float = 0.00,
        bit_risk_token_strength: float = 0.05,
        bit_risk_hidden_dim: int = 512,
        bit_risk_positive_weight_zero: float = 1.0,
        bit_risk_positive_weight_dac: float = 2.0,
        bit_risk_positive_weight_nc: float = 4.0,
        bit_risk_positive_weight_ttc: float = 2.0,
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
        lr_action_head: Optional[float] = None,
        lr_expert: Optional[float] = None,
        lr_expert_gate: Optional[float] = None,
        train_expert_only: bool = False,
        freeze_base_action_head: bool = False,
        freeze_expert: bool = False,
        scheduler_epochs: int = 200,
        scheduler_warmup_epochs: int = 3,
        scheduler_min_lr: float = 1e-6,
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
        self.grpo = grpo
        self.backbone = None
        self.metric_cache_path = metric_cache_path
        self.reference_policy_checkpoint = reference_policy_checkpoint
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
        self.use_bit_drive = use_bit_drive
        self.bit_terminal_loss_weight = bit_terminal_loss_weight
        self.bit_path_loss_weight = bit_path_loss_weight
        self.bit_end_consistency_loss_weight = bit_end_consistency_loss_weight
        self.bit_reverse_loss_weight = bit_reverse_loss_weight
        self.bit_cycle_loss_weight = bit_cycle_loss_weight
        self.bit_num_path_anchors = bit_num_path_anchors
        self.bit_path_anchor_indices = bit_path_anchor_indices
        self.bit_condition_mode = bit_condition_mode
        self.bit_use_gt_condition_prob = bit_use_gt_condition_prob
        self.bit_condition_noise_std = bit_condition_noise_std
        self.bit_condition_dropout = bit_condition_dropout
        self.bit_reverse_decoder_hidden_dim = bit_reverse_decoder_hidden_dim
        self.bit_use_reverse_decoder = bit_use_reverse_decoder
        self.bit_use_path_anchors = bit_use_path_anchors
        self.bit_use_terminal_head = bit_use_terminal_head
        self.bit_fail_if_gt_condition_in_eval = bit_fail_if_gt_condition_in_eval
        self.bit_log_diagnostics = bit_log_diagnostics
        self.bit_context_condition_strength = bit_context_condition_strength
        self.bit_action_condition_strength = bit_action_condition_strength
        self.bit_learnable_condition_gates = bit_learnable_condition_gates
        self.bit_context_gate_init = bit_context_gate_init
        self.bit_action_gate_init = bit_action_gate_init
        self.bit_detach_condition = bit_detach_condition
        self.bit_detach_condition_until_step = bit_detach_condition_until_step
        self.bit_use_gt_condition_schedule = bit_use_gt_condition_schedule
        self.bit_use_gt_condition_prob_start = bit_use_gt_condition_prob_start
        self.bit_use_gt_condition_prob_end = bit_use_gt_condition_prob_end
        self.bit_use_gt_condition_decay_steps = bit_use_gt_condition_decay_steps
        self.bit_loss_normalization = bit_loss_normalization
        self.bit_enable_terminal_condition = bit_enable_terminal_condition
        self.bit_enable_path_condition = bit_enable_path_condition
        self.bit_enable_action_add = bit_enable_action_add
        self.bit_enable_context_tokens = bit_enable_context_tokens
        self.bit_condition_coordinate_mode = bit_condition_coordinate_mode
        self.bit_condition_axis_scale_x = bit_condition_axis_scale_x
        self.bit_condition_axis_scale_y = bit_condition_axis_scale_y
        self.bit_condition_axis_scale_heading = bit_condition_axis_scale_heading
        self.bit_preserve_base_longitudinal = bit_preserve_base_longitudinal
        self.bit_base_longitudinal_loss_weight = bit_base_longitudinal_loss_weight
        self.bit_base_early_points = bit_base_early_points
        self.bit_base_lateral_loss_weight = bit_base_lateral_loss_weight
        self.bit_store_base_traj_for_loss = bit_store_base_traj_for_loss
        self.bit_use_safety_fallback = bit_use_safety_fallback
        self.bit_fallback_mode = bit_fallback_mode
        self.bit_fallback_early_x_delta_threshold = bit_fallback_early_x_delta_threshold
        self.bit_fallback_terminal_x_delta_threshold = bit_fallback_terminal_x_delta_threshold
        self.bit_use_risk_head = bit_use_risk_head
        self.bit_use_risk_tokens = bit_use_risk_tokens
        self.bit_risk_loss_weight = bit_risk_loss_weight
        self.bit_risk_token_strength = bit_risk_token_strength
        self.bit_risk_hidden_dim = bit_risk_hidden_dim
        self.bit_risk_positive_weight_zero = bit_risk_positive_weight_zero
        self.bit_risk_positive_weight_dac = bit_risk_positive_weight_dac
        self.bit_risk_positive_weight_nc = bit_risk_positive_weight_nc
        self.bit_risk_positive_weight_ttc = bit_risk_positive_weight_ttc
        self.use_last_rd = use_last_rd
        self.last_rd_stage = last_rd_stage
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
        self.lr_action_head = lr_action_head
        self.lr_expert = lr_expert
        self.lr_expert_gate = lr_expert_gate
        self.train_expert_only = train_expert_only
        self.freeze_base_action_head = freeze_base_action_head
        self.freeze_expert = freeze_expert
        self.scheduler_epochs = scheduler_epochs
        self.scheduler_warmup_epochs = scheduler_warmup_epochs
        self.scheduler_min_lr = scheduler_min_lr
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
        cfg.use_bit_drive = self.use_bit_drive
        cfg.bit_terminal_loss_weight = self.bit_terminal_loss_weight
        cfg.bit_path_loss_weight = self.bit_path_loss_weight
        cfg.bit_end_consistency_loss_weight = self.bit_end_consistency_loss_weight
        cfg.bit_reverse_loss_weight = self.bit_reverse_loss_weight
        cfg.bit_cycle_loss_weight = self.bit_cycle_loss_weight
        cfg.bit_num_path_anchors = self.bit_num_path_anchors
        cfg.bit_path_anchor_indices = self.bit_path_anchor_indices
        cfg.bit_condition_mode = self.bit_condition_mode
        cfg.bit_use_gt_condition_prob = self.bit_use_gt_condition_prob
        cfg.bit_condition_noise_std = self.bit_condition_noise_std
        cfg.bit_condition_dropout = self.bit_condition_dropout
        cfg.bit_reverse_decoder_hidden_dim = self.bit_reverse_decoder_hidden_dim
        cfg.bit_use_reverse_decoder = self.bit_use_reverse_decoder
        cfg.bit_use_path_anchors = self.bit_use_path_anchors
        cfg.bit_use_terminal_head = self.bit_use_terminal_head
        cfg.bit_fail_if_gt_condition_in_eval = self.bit_fail_if_gt_condition_in_eval
        cfg.bit_log_diagnostics = self.bit_log_diagnostics
        cfg.bit_context_condition_strength = self.bit_context_condition_strength
        cfg.bit_action_condition_strength = self.bit_action_condition_strength
        cfg.bit_learnable_condition_gates = self.bit_learnable_condition_gates
        cfg.bit_context_gate_init = self.bit_context_gate_init
        cfg.bit_action_gate_init = self.bit_action_gate_init
        cfg.bit_detach_condition = self.bit_detach_condition
        cfg.bit_detach_condition_until_step = self.bit_detach_condition_until_step
        cfg.bit_use_gt_condition_schedule = self.bit_use_gt_condition_schedule
        cfg.bit_use_gt_condition_prob_start = self.bit_use_gt_condition_prob_start
        cfg.bit_use_gt_condition_prob_end = self.bit_use_gt_condition_prob_end
        cfg.bit_use_gt_condition_decay_steps = self.bit_use_gt_condition_decay_steps
        cfg.bit_loss_normalization = self.bit_loss_normalization
        cfg.bit_enable_terminal_condition = self.bit_enable_terminal_condition
        cfg.bit_enable_path_condition = self.bit_enable_path_condition
        cfg.bit_enable_action_add = self.bit_enable_action_add
        cfg.bit_enable_context_tokens = self.bit_enable_context_tokens
        cfg.bit_condition_coordinate_mode = self.bit_condition_coordinate_mode
        cfg.bit_condition_axis_scale_x = self.bit_condition_axis_scale_x
        cfg.bit_condition_axis_scale_y = self.bit_condition_axis_scale_y
        cfg.bit_condition_axis_scale_heading = self.bit_condition_axis_scale_heading
        cfg.bit_preserve_base_longitudinal = self.bit_preserve_base_longitudinal
        cfg.bit_base_longitudinal_loss_weight = self.bit_base_longitudinal_loss_weight
        cfg.bit_base_early_points = self.bit_base_early_points
        cfg.bit_base_lateral_loss_weight = self.bit_base_lateral_loss_weight
        cfg.bit_store_base_traj_for_loss = self.bit_store_base_traj_for_loss
        cfg.bit_use_safety_fallback = self.bit_use_safety_fallback
        cfg.bit_fallback_mode = self.bit_fallback_mode
        cfg.bit_fallback_early_x_delta_threshold = self.bit_fallback_early_x_delta_threshold
        cfg.bit_fallback_terminal_x_delta_threshold = self.bit_fallback_terminal_x_delta_threshold
        cfg.bit_use_risk_head = self.bit_use_risk_head
        cfg.bit_use_risk_tokens = self.bit_use_risk_tokens
        cfg.bit_risk_loss_weight = self.bit_risk_loss_weight
        cfg.bit_risk_token_strength = self.bit_risk_token_strength
        cfg.bit_risk_hidden_dim = self.bit_risk_hidden_dim
        cfg.bit_risk_positive_weight_zero = self.bit_risk_positive_weight_zero
        cfg.bit_risk_positive_weight_dac = self.bit_risk_positive_weight_dac
        cfg.bit_risk_positive_weight_nc = self.bit_risk_positive_weight_nc
        cfg.bit_risk_positive_weight_ttc = self.bit_risk_positive_weight_ttc
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

        if self.grpo:
            cfg.grpo_cfg.metric_cache_path = self.metric_cache_path
            cfg.grpo_cfg.reference_policy_checkpoint = self.reference_policy_checkpoint
            
        self.action_head = ReCogDriveDiffusionPlanner(cfg).to(device)
        if self.last_rd_adapter_checkpoint:
            self._safe_load_last_rd_adapter(self.last_rd_adapter_checkpoint)
        self._set_trainable_parameters()
        self.num_inference_samples = 1
        self.inference_selection_mode = "median"

    def name(self) -> str:
        return self.__class__.__name__

    def set_training_progress(self, epoch: int, total_epochs: int) -> None:
        if hasattr(self.action_head, "set_training_progress"):
            self.action_head.set_training_progress(epoch, total_epochs)

    def count_trainable_parameters_by_group(self) -> Dict[str, Dict[str, int]]:
        groups = {
            "last_rd": {"trainable": 0, "total": 0},
            "legacy_a4_expert": {"trainable": 0, "total": 0},
            "action_base": {"trainable": 0, "total": 0},
            "backbone": {"trainable": 0, "total": 0},
            "other": {"trainable": 0, "total": 0},
        }
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
            "bit_terminal_head",
            "bit_condition_encoder",
            "bit_reverse_decoder",
            "bit_context_gate",
            "bit_action_gate",
            "bit_risk_head",
            "bit_risk_token_encoder",
        )
        for name, parameter in self.named_parameters():
            count = int(parameter.numel())
            if "action_head.last_rd." in name:
                group = "last_rd"
            elif name.startswith("action_head.") and any(marker in name for marker in legacy_markers):
                group = "legacy_a4_expert"
            elif name.startswith("action_head."):
                group = "action_base"
            elif name.startswith("backbone."):
                group = "backbone"
            else:
                group = "other"
            groups[group]["total"] += count
            if parameter.requires_grad:
                groups[group]["trainable"] += count
        groups["all"] = {
            "total": sum(item["total"] for item in groups.values()),
            "trainable": sum(item["trainable"] for item in groups.values()),
        }
        return groups

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
            allow_expert_target_features=self.allow_expert_target_features and self.training,
            use_jepa=self.use_jepa,
            use_vggt=self.use_vggt,
            jepa_dim=self.jepa_dim,
            vggt_dim=self.vggt_dim,
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

            outputs = self.backbone(pixel_values_cat, questions, num_patches_list=num_patches_list)
            last_hidden_state = outputs.hidden_states[-1]

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
        for key in EXPERT_FEATURE_KEYS:
            if key in features and isinstance(features[key], torch.Tensor):
                if key in EXPERT_TARGET_FEATURE_KEYS and not target_loss_mode:
                    continue
                action_input_data[key] = features[key].to(model_dtype)

        if targets is not None and not self.grpo:
            action_inputs = BatchFeature(
                data={
                    **action_input_data,
                    "action": targets["trajectory"].to(device=action_device, dtype=model_dtype),
                    "_allow_target_tokens_for_loss": True,
                }
            )
            return self.action_head(last_hidden_state, action_inputs)
        elif self.training and self.grpo:
            action_inputs = BatchFeature(
                data={**action_input_data, "action": targets["trajectory"].to(device=action_device, dtype=model_dtype)}
            )
            return self.action_head.forward_grpo(last_hidden_state, action_inputs, tokens_list)
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
            "bit_terminal_head",
            "bit_condition_encoder",
            "bit_reverse_decoder",
            "bit_context_gate",
            "bit_action_gate",
            "bit_risk_head",
            "bit_risk_token_encoder",
        )
        return key.startswith("action_head.") and any(marker in key for marker in expert_markers)

    @staticmethod
    def _is_gate_parameter_key(key: str) -> bool:
        gate_markers = ("jepa_gate", "vggt_gate", "branch_logits", "scene_gate", "timestep_gate", "bit_context_gate", "bit_action_gate")
        return key.startswith("action_head.") and any(marker in key for marker in gate_markers)

    @staticmethod
    def _is_action_head_parameter_key(key: str) -> bool:
        return key.startswith("action_head.")

    def _optimizer_grouping_requested(self) -> bool:
        return (
            self.lr_action_head is not None
            or self.lr_expert is not None
            or self.lr_expert_gate is not None
            or self.train_expert_only
            or self.freeze_base_action_head
            or self.freeze_expert
        )

    def _set_trainable_parameters(self) -> None:
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
            if self.train_expert_only or self.freeze_base_action_head:
                parameter.requires_grad = is_expert and not self.freeze_expert
            elif self.freeze_expert:
                parameter.requires_grad = not is_expert

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
        return optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))

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
                if self._is_expert_parameter_key(mapped_key):
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
        missing_expert = [key for key in missing_keys if self._is_expert_parameter_key(key)]
        missing_other = [key for key in missing_keys if not self._is_expert_parameter_key(key)]

        print(f"Loaded checkpoint from {path} with strict=False.")
        print(f"  loaded keys: {len(filtered_state)}")
        print(f"  missing keys: {len(missing_keys)}")
        if missing_expert:
            print("  expected missing expert keys:")
            for key in missing_expert:
                print(f"    - {key}")
        if missing_other:
            print("  missing non-expert keys:")
            for key in missing_other:
                print(f"    - {key}")
        if unexpected_keys or incompatible.unexpected_keys:
            print("  unexpected keys:")
            for key in [*unexpected_keys, *incompatible.unexpected_keys]:
                print(f"    - {key}")
        if skipped_expert_shape:
            print("  skipped expert shape mismatches:")
            for item in skipped_expert_shape:
                print(f"    - {item}")

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
        if self.training and self.grpo:
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
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=0.0, epochs=10, warmup_epochs=0)
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
