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
import math
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
    use_expert_features: bool = False
    expert_feature_source: Literal['none', 'dummy', 'cache', 'real'] = 'none'
    allow_random_init: bool = True
    allow_dummy_expert_cache: bool = False
    use_jepa: bool = True
    use_vggt: bool = True
    jepa_dim: int = 0
    vggt_dim: int = 0
    expert_dropout: float = 0.0
    expert_fusion_mode: str = "concat_context"
    use_expert_type_embedding: bool = True
    use_expert_gates: bool = True
    expert_alignment_weight: float = 0.0
    jepa_alignment_weight: float = 0.0
    vggt_alignment_weight: float = 0.0
    alignment_loss_type: Literal['mse', 'cosine'] = 'mse'
    
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    
    flow_cfg: FlowConfig = field(default_factory=FlowConfig)
    ddpm_cfg: DDPMConfig = field(default_factory=DDPMConfig)
    ddim_cfg: DDIMConfig = field(default_factory=DDIMConfig)
    grpo_cfg: GRPOConfig = field(default_factory=GRPOConfig)


class ReCogDriveDiffusionPlanner(nn.Module):
    config_class = ReCogDriveDiffusionPlannerConfig
    expert_target_keys = ("jepa_target_tokens", "vggt_target_tokens")

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

        if config.alignment_loss_type not in {"mse", "cosine"}:
            raise ValueError("alignment_loss_type must be either 'mse' or 'cosine'.")
        for weight_name in ("expert_alignment_weight", "jepa_alignment_weight", "vggt_alignment_weight"):
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
                raise ValueError(
                    "use_expert_features=True with use_jepa=True requires positive jepa_dim."
                )
            if config.use_vggt and config.vggt_dim <= 0:
                raise ValueError(
                    "use_expert_features=True with use_vggt=True requires positive vggt_dim."
                )
            if not 0.0 <= config.expert_dropout < 1.0:
                raise ValueError("expert_dropout must be in [0.0, 1.0).")

            if config.use_jepa:
                self.jepa_projector = nn.Linear(config.jepa_dim, config.input_embedding_dim)
            if config.use_vggt:
                self.vggt_projector = nn.Linear(config.vggt_dim, config.input_embedding_dim)

            if config.use_expert_type_embedding:
                self.expert_type_embedding = nn.Embedding(2, config.input_embedding_dim)
                nn.init.normal_(self.expert_type_embedding.weight, mean=0.0, std=0.02)

            if config.use_expert_gates:
                gate_init = math.log(0.1 / 0.9)
                if config.use_jepa:
                    self.jepa_gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
                if config.use_vggt:
                    self.vggt_gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
            
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
            "expert_type_embedding",
            "jepa_gate",
            "vggt_gate",
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
                    if self.config.use_jepa:
                        self.jepa_projector.eval()
                    if self.config.use_vggt:
                        self.vggt_projector.eval()
                    if self.config.use_expert_type_embedding:
                        self.expert_type_embedding.eval()
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

    def _require_expert_tokens(
        self,
        action_input: Optional[BatchFeature],
        key: str,
    ) -> torch.Tensor:
        if action_input is None or key not in action_input:
            raise KeyError(
                f"use_expert_features=True requires action_input['{key}'] for direct expert fusion."
            )
        return action_input[key]

    def _warn_if_expert_targets_present(
        self,
        action_input: Optional[BatchFeature],
        caller: str,
    ) -> None:
        if action_input is None:
            return
        present = [key for key in self.expert_target_keys if key in action_input]
        if present:
            warnings.warn(
                f"{caller} received train-only expert target keys {present}. "
                "They are ignored and are never used for inference conditioning.",
                RuntimeWarning,
            )

    def _project_expert_tokens(
        self,
        tokens: torch.Tensor,
        projector: nn.Module,
        expected_dim: int,
        name: str,
        vl_embeds: torch.Tensor,
        type_index: int,
        training: bool,
    ) -> torch.Tensor:
        embeds = self._adapter_project_expert_tokens(
            tokens=tokens,
            projector=projector,
            expected_dim=expected_dim,
            tensor_name=f"{name}_tokens",
            vl_embeds=vl_embeds,
        )
        return self._apply_expert_context_modifiers(embeds, name, type_index, training)

    def _apply_expert_context_modifiers(
        self,
        embeds: torch.Tensor,
        name: str,
        type_index: int,
        training: bool,
    ) -> torch.Tensor:
        if self.config.use_expert_type_embedding:
            type_embed = self.expert_type_embedding.weight[type_index].view(1, 1, -1)
            embeds = embeds + type_embed.to(device=embeds.device, dtype=embeds.dtype)

        if training and self.config.expert_dropout > 0:
            embeds = F.dropout(embeds, p=self.config.expert_dropout, training=True)

        if self.config.use_expert_gates:
            gate = self.jepa_gate if name == "jepa" else self.vggt_gate
            embeds = embeds * torch.sigmoid(gate).to(device=embeds.device, dtype=embeds.dtype)

        return embeds

    def _adapter_project_expert_tokens(
        self,
        tokens: torch.Tensor,
        projector: nn.Module,
        expected_dim: int,
        tensor_name: str,
        vl_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Projects expert tokens [B, K, D_teacher] into planner space [B, K, D_planner]."""
        if not isinstance(tokens, torch.Tensor):
            raise TypeError(f"action_input.{tensor_name} must be a torch.Tensor, got {type(tokens).__name__}.")
        if tokens.ndim != 3:
            raise ValueError(
                f"action_input.{tensor_name} must have shape [B, K, D], got {tuple(tokens.shape)}."
            )
        if tokens.shape[0] != vl_embeds.shape[0]:
            raise ValueError(
                f"action_input.{tensor_name} batch size {tokens.shape[0]} does not match "
                f"vl_features batch size {vl_embeds.shape[0]}."
            )
        if tokens.shape[-1] != expected_dim:
            raise ValueError(
                f"action_input.{tensor_name} last dimension {tokens.shape[-1]} does not match "
                f"configured expected_dim={expected_dim}."
            )

        tokens = tokens.to(device=vl_embeds.device, dtype=vl_embeds.dtype)
        return projector(tokens)

    def _build_context_embeds(
        self,
        vl_features: torch.Tensor,
        action_input: Optional[BatchFeature],
        training: bool,
        return_expert_embeds: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        vl_embeds = self.feature_encoder(vl_features)

        if not self.config.use_expert_features:
            if return_expert_embeds:
                return vl_embeds, {}
            return vl_embeds

        context_parts = [vl_embeds]
        expert_embeds: Dict[str, torch.Tensor] = {}

        if self.config.use_jepa:
            jepa_adapter_embeds = self._adapter_project_expert_tokens(
                tokens=self._require_expert_tokens(action_input, "jepa_tokens"),
                projector=self.jepa_projector,
                expected_dim=self.config.jepa_dim,
                tensor_name="jepa_tokens",
                vl_embeds=vl_embeds,
            )
            jepa_embeds = self._apply_expert_context_modifiers(
                embeds=jepa_adapter_embeds,
                name="jepa",
                type_index=0,
                training=training,
            )
            context_parts.append(jepa_embeds)
            expert_embeds["jepa"] = jepa_adapter_embeds

        if self.config.use_vggt:
            vggt_adapter_embeds = self._adapter_project_expert_tokens(
                tokens=self._require_expert_tokens(action_input, "vggt_tokens"),
                projector=self.vggt_projector,
                expected_dim=self.config.vggt_dim,
                tensor_name="vggt_tokens",
                vl_embeds=vl_embeds,
            )
            vggt_embeds = self._apply_expert_context_modifiers(
                embeds=vggt_adapter_embeds,
                name="vggt",
                type_index=1,
                training=training,
            )
            context_parts.append(vggt_embeds)
            expert_embeds["vggt"] = vggt_adapter_embeds

        context_embeds = torch.cat(context_parts, dim=1)
        if return_expert_embeds:
            return context_embeds, expert_embeds
        return context_embeds

    def _stream_alignment_weight(self, stream: str) -> float:
        stream_weight = self.config.jepa_alignment_weight if stream == "jepa" else self.config.vggt_alignment_weight
        return self.config.expert_alignment_weight + stream_weight

    def _alignment_loss(
        self,
        current_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
    ) -> torch.Tensor:
        current = current_embeds.float()
        target = target_embeds.detach().float()
        if self.config.alignment_loss_type == "mse":
            return F.mse_loss(current, target, reduction='mean')
        if self.config.alignment_loss_type == "cosine":
            return (1.0 - F.cosine_similarity(current, target, dim=-1)).mean()
        raise ValueError(f"Unsupported alignment_loss_type: {self.config.alignment_loss_type}")

    def _compute_expert_alignment_loss(
        self,
        stream: str,
        action_input: BatchFeature,
        current_embeds: Dict[str, torch.Tensor],
        reference_loss: torch.Tensor,
    ) -> torch.Tensor:
        weight = self._stream_alignment_weight(stream)
        target_key = f"{stream}_target_tokens"
        if (
            weight == 0.0
            or not self.config.use_expert_features
            or not self.training
            or target_key not in action_input
            or stream not in current_embeds
        ):
            return reference_loss.new_zeros(())

        projector = self.jepa_projector if stream == "jepa" else self.vggt_projector
        expected_dim = self.config.jepa_dim if stream == "jepa" else self.config.vggt_dim
        with torch.no_grad():
            target_embeds = self._adapter_project_expert_tokens(
                tokens=action_input[target_key],
                projector=projector,
                expected_dim=expected_dim,
                tensor_name=target_key,
                vl_embeds=current_embeds[stream],
            )
        return self._alignment_loss(current_embeds[stream], target_embeds).to(reference_loss.dtype)

    def _repeat_expert_action_input(
        self,
        action_input: BatchFeature,
        repeat: int,
    ) -> Optional[BatchFeature]:
        if not self.config.use_expert_features:
            return None

        return BatchFeature(data={
            key: self._require_expert_tokens(action_input, key).repeat_interleave(repeat, 0)
            for key, enabled in (("jepa_tokens", self.config.use_jepa), ("vggt_tokens", self.config.use_vggt))
            if enabled
        })


    def p_mean_variance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        index: torch.Tensor,
        context_embeds: torch.Tensor,
        his_traj_features: torch.Tensor,
        ego_status_features: torch.Tensor,
        deterministic: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the mean and log variance of the reverse process p(x_{t-1} | x_t).
        Also returns the predicted x0.
        """
        model_dtype = next(self.model.parameters()).dtype
        x = x.to(model_dtype)
        action_features = self.action_encoder(x, t)
        if hasattr(self, 'position_embedding'):
            pos_ids = torch.arange(action_features.shape[1], device=x.device)
            action_features = action_features + self.position_embedding(pos_ids)

        context_mean = context_embeds.mean(1).unsqueeze(1).repeat(1, self.config.action_horizon, 1)
        fused_input = self.fusion_projector(
            torch.cat((his_traj_features, context_mean, action_features), dim=2)
        )

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
            
            pred_noise = (x - (alpha_t**0.5) * x_recon) / sqrt_one_minus_alpha_t

            eps_clip_value = getattr(self, 'eps_clip_value', None)
            if eps_clip_value is not None:
                pred_noise.clamp_(-eps_clip_value, eps_clip_value)

            if deterministic:
                etas = torch.zeros((x.shape[0], 1, 1)).to(x.device)
            else:
                etas = self.eta(x).unsqueeze(1)

            sigma = (
                etas
                * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)) ** 0.5
            ).clamp_(min=1e-10)

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
        if not self.training:
            self._warn_if_expert_targets_present(action_input, "forward")
        
        context_embeds, expert_embeds = self._build_context_embeds(
            vl_features,
            action_input,
            training=self.training,
            return_expert_embeds=True,
        )
        his_traj_features = self.his_traj_encoder(
            action_input.his_traj.unsqueeze(1)
        ).repeat(1, self.config.action_horizon, 1)
        ego_status_features = self.ego_status_encoder(action_input.status_feature)
        
        gt_actions = self.norm_odo(action_input.action)

        if self.config.sampling_method == 'flow':
            noise = torch.randn_like(gt_actions)
            t_cont = self.sample_time(gt_actions.shape[0], device=gt_actions.device, dtype=gt_actions.dtype)
            t_cont_reshaped = t_cont[:, None, None]
            
            noisy_actions = (1 - t_cont_reshaped) * noise + t_cont_reshaped * gt_actions
            velocity_target = gt_actions - noise
            t_discrete = (t_cont * self.num_timestep_buckets).long()

            action_features = self.action_encoder(noisy_actions, t_discrete)
            if hasattr(self, 'position_embedding'):
                pos_ids = torch.arange(action_features.shape[1], device=gt_actions.device)
                action_features += self.position_embedding(pos_ids)
            
            context_mean = context_embeds.mean(1).unsqueeze(1).repeat(1, self.config.action_horizon, 1)
            fused_input = self.fusion_projector(
                torch.cat((his_traj_features, context_mean, action_features), dim=2)
            )

            model_output = self.model(fused_input, context_embeds, ego_status_features, t_discrete)
            pred_velocity = self.action_decoder(model_output)
            diffusion_loss = F.mse_loss(pred_velocity, velocity_target, reduction='mean')
        else: 
            noise = torch.randn_like(gt_actions)
            t_discrete = self.sample_time(gt_actions.shape[0], device=gt_actions.device, dtype=gt_actions.dtype)
            
            noisy_actions = (
                self.extract(self.ddpm_sqrt_alphas_cumprod, t_discrete, gt_actions.shape) * gt_actions +
                self.extract(self.ddpm_sqrt_one_minus_alphas_cumprod, t_discrete, gt_actions.shape) * noise
            )
            
            action_features = self.action_encoder(noisy_actions, t_discrete)
            if hasattr(self, 'position_embedding'):
                pos_ids = torch.arange(action_features.shape[1], device=gt_actions.device)
                action_features += self.position_embedding(pos_ids)
            
            context_mean = context_embeds.mean(1).unsqueeze(1).repeat(1, self.config.action_horizon, 1)
            fused_input = self.fusion_projector(
                torch.cat((his_traj_features, context_mean, action_features), dim=2)
            )
            
            model_output = self.model(fused_input, context_embeds, ego_status_features, t_discrete)
            pred_noise = self.action_decoder(model_output)
            diffusion_loss = F.mse_loss(pred_noise, noise, reduction='mean')

        jepa_alignment_loss = self._compute_expert_alignment_loss(
            "jepa",
            action_input,
            expert_embeds,
            diffusion_loss,
        )
        vggt_alignment_loss = self._compute_expert_alignment_loss(
            "vggt",
            action_input,
            expert_embeds,
            diffusion_loss,
        )
        loss = (
            diffusion_loss
            + self._stream_alignment_weight("jepa") * jepa_alignment_loss
            + self._stream_alignment_weight("vggt") * vggt_alignment_loss
        )

        return BatchFeature(data={
            "loss": loss,
            "diffusion_loss": diffusion_loss,
            "jepa_alignment_loss": jepa_alignment_loss,
            "vggt_alignment_loss": vggt_alignment_loss,
        })

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
        context_embeds = self._build_context_embeds(vl_features, action_input, training=False)
        
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

                action_features = self.action_encoder(current_actions, t)
                if hasattr(self, 'position_embedding'):
                    action_features += self.position_embedding(torch.arange(self.config.action_horizon, device=device))
                
                context_mean = context_embeds.mean(1).unsqueeze(1).repeat(1, self.config.action_horizon, 1)
                fused_input = self.fusion_projector(
                    torch.cat((history_embeds, context_mean, action_features), dim=2)
                )
                
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
                    current_actions, t_batch, index_batch, context_embeds, history_embeds, ego_embeds, deterministic
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
                    current_actions, t_batch, index_batch, context_embeds, history_embeds, ego_embeds, deterministic
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

        return BatchFeature(data={"pred_traj": final_actions})

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
        context_embeds = self._build_context_embeds(vl_features, action_input, training=self.training)
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
                
                action_features = self.action_encoder(current_actions, t_batch)
                if hasattr(self, 'position_embedding'):
                    action_features += self.position_embedding(torch.arange(self.config.action_horizon, device=device))
                
                context_mean = context_embeds.mean(1).unsqueeze(1).repeat(1, self.config.action_horizon, 1)
                fused_input = self.fusion_projector(
                    torch.cat((his_traj_features, context_mean, action_features), dim=2)
                )
                
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
                    current_actions, t_batch, index_batch, context_embeds, his_traj_features, ego_status_features, deterministic
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
        
        context_embeds = self._build_context_embeds(vl_features, action_input, training=self.training)

        his_traj_features = self.his_traj_encoder(
            his_traj_features.unsqueeze(1)          
        ).repeat(1, self.config.action_horizon, 1) 

        ego_status_features = self.ego_status_encoder(
            ego_status_features      
        )

        conditioning_embeds = {
            'context_embeds': context_embeds,
            'his_traj_features': his_traj_features,
            'ego_status_features': ego_status_features
        }
        
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
            deterministic=deterministic
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
