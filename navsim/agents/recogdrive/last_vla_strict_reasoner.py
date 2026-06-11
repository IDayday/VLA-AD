from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn

from .last_vla_teacher_adapters import DynamicsAdapter, GeometryAdapter, PlanLatentHead


@dataclass
class StrictLastVLAReasonerConfig:
    vlm_hidden_dim: int = 1536
    planner_dim: int = 384
    action_horizon: int = 8
    action_dim: int = 3
    num_dyn_latent_tokens: int = 64
    num_geo_latent_tokens: int = 64
    num_plan_latent_tokens: int = 32
    wm_teacher_source: str = "jepa_dev"
    random_mask_ratio: float = 0.30
    strict_teacher_targets: bool = False


class _LatentProjection(nn.Module):
    def __init__(self, vlm_hidden_dim: int, planner_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(vlm_hidden_dim),
            nn.Linear(vlm_hidden_dim, planner_dim),
            nn.GELU(),
            nn.Linear(planner_dim, planner_dim),
            nn.LayerNorm(planner_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


class StrictLastVLAReasoner(nn.Module):
    """Strict LaST-VLA VLM-side latent-slot reasoner for ReCogDrive.

    This module consumes H_dyn/H_geo/H_plan hidden slots produced by the frozen
    VLM, aligns them to train-time WM/3D teachers, and exposes only latent
    condition tokens to the downstream diffusion planner.
    """

    def __init__(self, config: StrictLastVLAReasonerConfig) -> None:
        super().__init__()
        self.config = config
        self.dynamics_adapter = DynamicsAdapter(
            vlm_hidden_dim=config.vlm_hidden_dim,
            planner_dim=config.planner_dim,
            jepa_dim=1024,
            num_teacher_tokens=128,
            mask_ratio=config.random_mask_ratio,
            teacher_source=config.wm_teacher_source,
            strict_teacher=config.strict_teacher_targets,
        )
        self.geometry_adapter = GeometryAdapter(
            vlm_hidden_dim=config.vlm_hidden_dim,
            planner_dim=config.planner_dim,
            geometry_dim=512,
            num_geometry_tokens=192,
            mask_ratio=config.random_mask_ratio,
            strict_teacher=config.strict_teacher_targets,
        )
        self.plan_head = PlanLatentHead(
            vlm_hidden_dim=config.vlm_hidden_dim,
            planner_dim=config.planner_dim,
            action_horizon=config.action_horizon,
            action_dim=config.action_dim,
        )
        self.latent_projection = _LatentProjection(config.vlm_hidden_dim, config.planner_dim)

    @staticmethod
    def _get(action_input: Optional[Dict[str, Any]], key: str) -> Any:
        if action_input is None:
            return None
        if key in action_input:
            return action_input[key]
        return getattr(action_input, key, None)

    def project_latent_conditions(
        self,
        h_dyn: torch.Tensor,
        h_geo: torch.Tensor,
        h_plan: torch.Tensor,
    ) -> torch.Tensor:
        return self.latent_projection(torch.cat([h_dyn, h_geo, h_plan], dim=1))

    def _validate_slots(self, h_dyn: torch.Tensor, h_geo: torch.Tensor, h_plan: torch.Tensor) -> None:
        expected = (
            ("h_dyn", h_dyn, self.config.num_dyn_latent_tokens),
            ("h_geo", h_geo, self.config.num_geo_latent_tokens),
            ("h_plan", h_plan, self.config.num_plan_latent_tokens),
        )
        batch = h_dyn.shape[0]
        for name, value, count in expected:
            if value.ndim != 3:
                raise ValueError(f"{name} must have shape [B, K, D], got {tuple(value.shape)}.")
            if value.shape[0] != batch:
                raise ValueError(f"{name} batch {value.shape[0]} does not match h_dyn batch {batch}.")
            if value.shape[1] != count:
                raise ValueError(f"{name} token count {value.shape[1]} does not match expected {count}.")
            if value.shape[2] != self.config.vlm_hidden_dim:
                raise ValueError(f"{name} dim {value.shape[2]} does not match expected {self.config.vlm_hidden_dim}.")
            if not torch.isfinite(value.float()).all():
                raise FloatingPointError(f"{name} contains non-finite values.")

    def forward(
        self,
        vlm_outputs: Dict[str, torch.Tensor],
        action_input: Optional[Dict[str, Any]] = None,
        *,
        target_action_norm: Optional[torch.Tensor] = None,
        allow_teacher_targets: bool = True,
        require_teacher_targets: Optional[bool] = None,
    ) -> Dict[str, Any]:
        last_hidden_state = vlm_outputs["last_hidden_state"]
        h_dyn = vlm_outputs["h_dyn"]
        h_geo = vlm_outputs["h_geo"]
        h_plan = vlm_outputs["h_plan"]
        self._validate_slots(h_dyn, h_geo, h_plan)

        image_hidden_states = vlm_outputs.get("image_hidden_states")
        if image_hidden_states is None:
            image_hidden_states = last_hidden_state
        if image_hidden_states.ndim != 3:
            raise ValueError("image_hidden_states must have shape [B, N, D].")

        teacher_required = self.config.strict_teacher_targets if require_teacher_targets is None else bool(require_teacher_targets)
        if not allow_teacher_targets:
            teacher_required = False
        jepa_target = self._get(action_input, "jepa_target_tokens") if allow_teacher_targets else None
        cosmos_target = self._get(action_input, "cosmos_future_features") if allow_teacher_targets else None
        geometry_target = self._get(action_input, "vggt_geometry_tokens") if allow_teacher_targets else None
        if geometry_target is None and allow_teacher_targets:
            geometry_target = self._get(action_input, "vggt_geometry_target_tokens")

        dyn_out = self.dynamics_adapter(
            image_hidden_states,
            h_dyn,
            jepa_target_tokens=jepa_target,
            cosmos_future_features=cosmos_target,
            require_teacher=teacher_required,
        )
        geo_out = self.geometry_adapter(
            image_hidden_states,
            h_geo,
            vggt_geometry_tokens=geometry_target,
            require_teacher=teacher_required,
        )
        if target_action_norm is None:
            target_action_norm = self._get(action_input, "target_action_norm")
        plan_out = self.plan_head(
            h_plan,
            h_dyn=h_dyn,
            h_geo=h_geo,
            target_action_norm=target_action_norm,
            status_feature=self._get(action_input, "status_feature"),
            high_command_one_hot=self._get(action_input, "high_command_one_hot"),
            history_trajectory=self._get(action_input, "history_trajectory"),
        )
        latent_condition_tokens = self.project_latent_conditions(h_dyn, h_geo, h_plan)
        zero = latent_condition_tokens.new_zeros(())
        losses = {
            "wm_loss": dyn_out["loss"],
            "geometry_loss": geo_out["loss"],
            "plan_loss": plan_out["losses"].get("plan_loss", zero),
            "heading_loss": plan_out["losses"].get("heading_loss", zero),
            "progress_loss": plan_out["losses"].get("progress_loss", zero),
        }
        diagnostics = {
            **dyn_out["diagnostics"],
            **geo_out["diagnostics"],
            **plan_out["diagnostics"],
            "h_dyn_norm": h_dyn.detach().float().norm(dim=-1).mean().to(dtype=latent_condition_tokens.dtype),
            "h_geo_norm": h_geo.detach().float().norm(dim=-1).mean().to(dtype=latent_condition_tokens.dtype),
            "h_plan_norm": h_plan.detach().float().norm(dim=-1).mean().to(dtype=latent_condition_tokens.dtype),
            "dyn_slot_count": latent_condition_tokens.new_tensor(float(h_dyn.shape[1])),
            "geo_slot_count": latent_condition_tokens.new_tensor(float(h_geo.shape[1])),
            "plan_slot_count": latent_condition_tokens.new_tensor(float(h_plan.shape[1])),
            "latent_condition_token_count": latent_condition_tokens.new_tensor(float(latent_condition_tokens.shape[1])),
        }
        return {
            "raw_last_hidden_state": last_hidden_state,
            "h_dyn": h_dyn,
            "h_geo": h_geo,
            "h_plan": h_plan,
            "latent_condition_tokens": latent_condition_tokens,
            "pred_coarse_traj_norm": plan_out["coarse_traj_norm"],
            "losses": losses,
            "diagnostics": diagnostics,
        }
