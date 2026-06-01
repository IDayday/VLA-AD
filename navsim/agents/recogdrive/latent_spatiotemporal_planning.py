from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class LastRDConfig:
    planner_dim: int = 384
    jepa_dim: int = 1024
    vggt_dim: int = 2048
    latent_dim: int = 384
    hidden_dim: int = 1024
    action_dim: int = 3
    action_horizon: int = 8

    num_dynamic_tokens: int = 12
    num_geometry_tokens: int = 12
    num_ego_tokens: int = 8
    num_risk_tokens: int = 8

    use_future_jepa_prediction: bool = True
    use_vggt_geometry_tokens: bool = True
    use_ego_trajectory_tokens: bool = True
    use_risk_tokens: bool = True
    use_scene_aware_gate: bool = True
    use_timestep_aware_gate: bool = True

    require_vggt_geometry: bool = False
    allow_patch_geometry_fallback: bool = True

    normalized_target_loss: bool = True
    risk_num_classes: int = 4
    requires_risk_labels: bool = False
    allow_missing_jepa_context: bool = False


@dataclass
class LastRDOutput:
    planning_tokens: torch.Tensor
    dynamic_tokens: Optional[torch.Tensor]
    geometry_tokens: Optional[torch.Tensor]
    ego_tokens: Optional[torch.Tensor]
    risk_tokens: Optional[torch.Tensor]
    coarse_traj_norm: Optional[torch.Tensor]
    predicted_future_jepa: Optional[torch.Tensor]
    predicted_geometry: Optional[torch.Tensor]
    risk_logits: Optional[torch.Tensor]
    group_weights: Optional[torch.Tensor]
    horizon_condition: Optional[torch.Tensor]
    losses: Dict[str, torch.Tensor]
    diagnostics: Dict[str, Any]


GEOMETRY_MODE_TO_CODE = {"no_geometry": 0, "patch_fallback": 1, "full_geometry": 2}
TARGET_KEYS = (
    "jepa_target_tokens",
    "vggt_target_tokens",
    "vggt_geometry_target_tokens",
    "vggt_depth_target_tokens",
    "vggt_pointmap_target_tokens",
)
GROUP_NAMES = ("vlm", "dynamic", "geometry", "ego", "risk")


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor] = None, dim: int = 1) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=dim)
    if mask.dtype != torch.bool:
        mask = mask > 0
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(device=x.device)
    denom = mask.to(dtype=x.dtype).sum(dim=dim).clamp_min(1.0)
    return (x * mask.to(dtype=x.dtype)).sum(dim=dim) / denom


def normalized_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.detach().to(device=pred.device, dtype=pred.dtype)
    pred_norm = F.normalize(pred.float(), p=2, dim=-1, eps=1e-6)
    target_norm = F.normalize(target.float(), p=2, dim=-1, eps=1e-6)
    return F.mse_loss(pred_norm, target_norm)


def smooth_l1_loss_optional(
    pred: torch.Tensor,
    target: Optional[torch.Tensor],
    weight: float = 1.0,
) -> torch.Tensor:
    if target is None:
        return pred.new_zeros(())
    return pred.new_tensor(float(weight)) * F.smooth_l1_loss(pred.float(), target.detach().to(pred).float())


def heading_loss(pred_heading: torch.Tensor, target_heading: torch.Tensor) -> torch.Tensor:
    target_heading = target_heading.detach().to(device=pred_heading.device, dtype=pred_heading.dtype)
    pred_angle = pred_heading.float() * math.pi
    target_angle = target_heading.float() * math.pi
    pred_vec = torch.stack((torch.sin(pred_angle), torch.cos(pred_angle)), dim=-1)
    target_vec = torch.stack((torch.sin(target_angle), torch.cos(target_angle)), dim=-1)
    return F.smooth_l1_loss(pred_vec, target_vec)


def finite_or_raise(name: str, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is not None and not torch.isfinite(tensor.float()).all():
        raise FloatingPointError(f"{name} contains non-finite values.")
    return tensor


def detach_if_tensor_dict(values: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value.detach() if isinstance(value, torch.Tensor) else value for key, value in values.items()}


def _num_heads(dim: int) -> int:
    for heads in (8, 6, 4, 3, 2, 1):
        if dim % heads == 0:
            return heads
    return 1


def _get(action_input: Optional[Dict[str, Any]], key: str, default: Any = None) -> Any:
    if action_input is None:
        return default
    if key in action_input:
        return action_input[key]
    return getattr(action_input, key, default)


def _as_tensor(
    value: Optional[torch.Tensor],
    *,
    reference: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor, got {type(value).__name__}.")
    return value.to(device=reference.device, dtype=dtype or reference.dtype)


def _zeros(batch_size: int, dim: int, reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(batch_size, dim)


def _match_token_count(tokens: torch.Tensor, count: int) -> torch.Tensor:
    if tokens.shape[1] == count:
        return tokens
    return F.interpolate(tokens.transpose(1, 2).float(), size=count, mode="linear", align_corners=False).transpose(1, 2).to(tokens)


def _match_last_dim(tokens: torch.Tensor, dim: int) -> torch.Tensor:
    if tokens.shape[-1] == dim:
        return tokens
    if tokens.shape[-1] > dim:
        return tokens[..., :dim]
    return F.pad(tokens, (0, dim - tokens.shape[-1]))


def _validate_tokens(name: str, tokens: torch.Tensor, *, batch_size: int, dim: Optional[int] = None) -> torch.Tensor:
    if not isinstance(tokens, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tokens).__name__}.")
    if tokens.ndim != 3:
        raise ValueError(f"{name} must have shape [B, K, D], got {tuple(tokens.shape)}.")
    if tokens.shape[0] != batch_size:
        raise ValueError(f"{name} batch size {tokens.shape[0]} does not match B={batch_size}.")
    if dim is not None and tokens.shape[-1] != dim:
        raise ValueError(f"{name} dim {tokens.shape[-1]} does not match expected {dim}.")
    finite_or_raise(name, tokens)
    return tokens


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class QueryCrossAttention(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, _num_heads(dim), batch_first=True)
        self.mlp = ResidualMLP(dim, hidden_dim)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(self.query_norm(query), self.memory_norm(memory), self.memory_norm(memory), need_weights=False)
        return self.mlp(query + attended)


class FutureJEPAPredictor(nn.Module):
    def __init__(self, config: LastRDConfig) -> None:
        super().__init__()
        self.config = config
        self.vlm_proj = nn.Sequential(nn.LayerNorm(config.planner_dim), nn.Linear(config.planner_dim, config.latent_dim))
        self.jepa_proj = nn.Sequential(nn.LayerNorm(config.jepa_dim), nn.Linear(config.jepa_dim, config.latent_dim))
        self.state_proj = nn.Sequential(
            nn.LayerNorm(8 + 3 + 12),
            nn.Linear(8 + 3 + 12, config.latent_dim),
            nn.GELU(),
            nn.Linear(config.latent_dim, config.latent_dim),
        )
        self.dynamic_queries = nn.Parameter(torch.randn(config.num_dynamic_tokens, config.latent_dim) * 0.02)
        self.cross_attn = QueryCrossAttention(config.latent_dim, config.hidden_dim)
        self.dynamic_out = nn.Sequential(nn.LayerNorm(config.latent_dim), nn.Linear(config.latent_dim, config.planner_dim))
        self.future_head = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.jepa_dim),
        )

    def _state_token(
        self,
        ego_status: torch.Tensor,
        high_command_one_hot: Optional[torch.Tensor],
        history_trajectory_flat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = ego_status.shape[0]
        command = high_command_one_hot if high_command_one_hot is not None else _zeros(batch_size, 3, ego_status)
        history = history_trajectory_flat if history_trajectory_flat is not None else _zeros(batch_size, 12, ego_status)
        state = torch.cat([ego_status, command, history], dim=-1)
        return self.state_proj(state).unsqueeze(1)

    def forward(
        self,
        vlm_tokens: torch.Tensor,
        jepa_context_tokens: Optional[torch.Tensor],
        ego_status: torch.Tensor,
        high_command_one_hot: Optional[torch.Tensor] = None,
        history_trajectory_flat: Optional[torch.Tensor] = None,
        jepa_target_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = vlm_tokens.shape[0]
        if jepa_context_tokens is None and not self.config.allow_missing_jepa_context:
            raise KeyError("FutureJEPAPredictor requires jepa_context_tokens unless allow_missing_jepa_context=True.")

        memory_parts = [self.vlm_proj(vlm_tokens)]
        if jepa_context_tokens is not None:
            _validate_tokens("jepa_context_tokens", jepa_context_tokens, batch_size=batch_size, dim=self.config.jepa_dim)
            memory_parts.append(self.jepa_proj(jepa_context_tokens))
        memory_parts.append(self._state_token(ego_status, high_command_one_hot, history_trajectory_flat))
        memory = torch.cat(memory_parts, dim=1)

        queries = self.dynamic_queries.unsqueeze(0).expand(batch_size, -1, -1).to(vlm_tokens)
        latent = self.cross_attn(queries, memory)
        dynamic_tokens = self.dynamic_out(latent)
        predicted_future_jepa = self.future_head(latent)
        finite_or_raise("dynamic_tokens", dynamic_tokens)
        finite_or_raise("predicted_future_jepa", predicted_future_jepa)

        losses = {"future_jepa_loss": vlm_tokens.new_zeros(())}
        if jepa_target_tokens is not None:
            _validate_tokens("jepa_target_tokens", jepa_target_tokens, batch_size=batch_size, dim=self.config.jepa_dim)
            target = _match_token_count(jepa_target_tokens.detach(), self.config.num_dynamic_tokens)
            losses["future_jepa_loss"] = normalized_mse(predicted_future_jepa, target)
        return predicted_future_jepa, dynamic_tokens, losses


class VGGTGeometryEncoder(nn.Module):
    def __init__(self, config: LastRDConfig) -> None:
        super().__init__()
        self.config = config
        self.vlm_proj = nn.Sequential(nn.LayerNorm(config.planner_dim), nn.Linear(config.planner_dim, config.latent_dim))
        self.vggt_proj = nn.Sequential(nn.LayerNorm(config.vggt_dim), nn.Linear(config.vggt_dim, config.latent_dim))
        self.geometry_proj = nn.Sequential(
            nn.LayerNorm(config.vggt_dim),
            nn.Linear(config.vggt_dim, config.latent_dim),
        )
        self.state_proj = nn.Sequential(
            nn.LayerNorm(8 + 3),
            nn.Linear(8 + 3, config.latent_dim),
            nn.GELU(),
            nn.Linear(config.latent_dim, config.latent_dim),
        )
        self.geometry_queries = nn.Parameter(torch.randn(config.num_geometry_tokens, config.latent_dim) * 0.02)
        self.cross_attn = QueryCrossAttention(config.latent_dim, config.hidden_dim)
        self.geometry_out = nn.Sequential(nn.LayerNorm(config.latent_dim), nn.Linear(config.latent_dim, config.planner_dim))
        self.geometry_head = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.vggt_dim),
        )

    def _select_geometry_source(
        self,
        action_input: Optional[Dict[str, Any]],
        vggt_context_tokens: Optional[torch.Tensor],
        *,
        batch_size: int,
    ) -> Tuple[Optional[torch.Tensor], str, bool]:
        for key in ("vggt_geometry_tokens", "vggt_depth_tokens", "vggt_pointmap_tokens", "vggt_camera_tokens"):
            tokens = _get(action_input, key)
            if tokens is not None:
                return _validate_tokens(key, tokens, batch_size=batch_size), "full_geometry", False
        if self.config.require_vggt_geometry:
            raise KeyError(
                "require_vggt_geometry=True requires one of vggt_geometry_tokens, "
                "vggt_depth_tokens, vggt_pointmap_tokens, or vggt_camera_tokens."
            )
        if vggt_context_tokens is not None and self.config.allow_patch_geometry_fallback:
            return _validate_tokens(
                "vggt_context_tokens",
                vggt_context_tokens,
                batch_size=batch_size,
                dim=self.config.vggt_dim,
            ), "patch_fallback", True
        return None, "no_geometry", False

    def _select_geometry_target(
        self,
        action_input: Optional[Dict[str, Any]],
        vggt_target_tokens: Optional[torch.Tensor],
        *,
        batch_size: int,
    ) -> Tuple[Optional[torch.Tensor], str]:
        for key in ("vggt_geometry_target_tokens", "vggt_depth_target_tokens", "vggt_pointmap_target_tokens"):
            tokens = _get(action_input, key)
            if tokens is not None:
                return _validate_tokens(key, tokens, batch_size=batch_size), key
        if vggt_target_tokens is not None:
            return _validate_tokens("vggt_target_tokens", vggt_target_tokens, batch_size=batch_size, dim=self.config.vggt_dim), "vggt_target_tokens"
        return None, ""

    def forward(
        self,
        vggt_context_tokens: Optional[torch.Tensor],
        vlm_tokens: torch.Tensor,
        ego_status: torch.Tensor,
        high_command_one_hot: Optional[torch.Tensor] = None,
        *,
        action_input: Optional[Dict[str, Any]] = None,
        vggt_target_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        batch_size = vlm_tokens.shape[0]
        source_tokens, geometry_mode, source_is_vggt = self._select_geometry_source(
            action_input,
            vggt_context_tokens,
            batch_size=batch_size,
        )
        losses = {
            "vggt_geometry_loss": vlm_tokens.new_zeros(()),
            "vggt_token_loss": vlm_tokens.new_zeros(()),
        }
        diagnostics: Dict[str, Any] = {
            "geometry_mode": geometry_mode,
            "geometry_mode_code": vlm_tokens.new_tensor(float(GEOMETRY_MODE_TO_CODE[geometry_mode])),
        }
        if source_tokens is None:
            return None, None, losses, diagnostics

        source_tokens = source_tokens.to(device=vlm_tokens.device, dtype=vlm_tokens.dtype)
        source_tokens = _match_last_dim(source_tokens, self.config.vggt_dim)
        memory_parts = [self.vlm_proj(vlm_tokens)]
        if source_is_vggt:
            memory_parts.append(self.vggt_proj(source_tokens))
        else:
            memory_parts.append(self.geometry_proj(source_tokens))
        command = high_command_one_hot if high_command_one_hot is not None else _zeros(batch_size, 3, ego_status)
        state = torch.cat([ego_status, command], dim=-1)
        memory_parts.append(self.state_proj(state).unsqueeze(1))
        memory = torch.cat(memory_parts, dim=1)

        queries = self.geometry_queries.unsqueeze(0).expand(batch_size, -1, -1).to(vlm_tokens)
        latent = self.cross_attn(queries, memory)
        geometry_tokens = self.geometry_out(latent)
        predicted_geometry = self.geometry_head(latent)
        finite_or_raise("geometry_tokens", geometry_tokens)
        finite_or_raise("predicted_geometry", predicted_geometry)

        target, target_key = self._select_geometry_target(action_input, vggt_target_tokens, batch_size=batch_size)
        if target is not None:
            target = _match_last_dim(_match_token_count(target.detach().to(predicted_geometry), self.config.num_geometry_tokens), self.config.vggt_dim)
            loss = normalized_mse(predicted_geometry, target)
            if target_key == "vggt_target_tokens":
                losses["vggt_token_loss"] = loss
                losses["vggt_geometry_loss"] = loss
            else:
                losses["vggt_geometry_loss"] = loss
        return geometry_tokens, predicted_geometry, losses, diagnostics


class CoarseTrajectoryHead(nn.Module):
    def __init__(self, config: LastRDConfig) -> None:
        super().__init__()
        self.config = config
        state_dim = config.planner_dim * 2 + 8 + 3 + 12
        self.coarse_head = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.action_horizon * config.action_dim),
            nn.Tanh(),
        )
        self.coarse_step_proj = nn.Linear(config.action_dim, config.planner_dim)
        self.ego_queries = nn.Parameter(torch.randn(config.num_ego_tokens, config.planner_dim) * 0.02)
        self.cross_attn = QueryCrossAttention(config.planner_dim, config.hidden_dim)

    def forward(
        self,
        planning_tokens: torch.Tensor,
        vlm_mean: torch.Tensor,
        ego_status: torch.Tensor,
        high_command_one_hot: Optional[torch.Tensor] = None,
        history_trajectory_flat: Optional[torch.Tensor] = None,
        target_action_norm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = planning_tokens.shape[0]
        command = high_command_one_hot if high_command_one_hot is not None else _zeros(batch_size, 3, planning_tokens)
        history = history_trajectory_flat if history_trajectory_flat is not None else _zeros(batch_size, 12, planning_tokens)
        token_summary = planning_tokens.mean(dim=1)
        state = torch.cat([token_summary, vlm_mean, ego_status, command, history], dim=-1)
        coarse = self.coarse_head(state).view(batch_size, self.config.action_horizon, self.config.action_dim)

        coarse_step_tokens = self.coarse_step_proj(coarse)
        memory = torch.cat([planning_tokens, coarse_step_tokens], dim=1)
        queries = self.ego_queries.unsqueeze(0).expand(batch_size, -1, -1).to(planning_tokens)
        ego_tokens = self.cross_attn(queries, memory)

        losses = {
            "coarse_traj_loss": planning_tokens.new_zeros(()),
            "coarse_heading_loss": planning_tokens.new_zeros(()),
            "coarse_progress_loss": planning_tokens.new_zeros(()),
        }
        if target_action_norm is not None:
            target_action_norm = target_action_norm.detach().to(coarse)
            losses["coarse_traj_loss"] = F.smooth_l1_loss(coarse.float(), target_action_norm.float())
            if self.config.action_dim >= 3:
                losses["coarse_heading_loss"] = heading_loss(coarse[..., 2], target_action_norm[..., 2])
            if self.config.action_dim >= 2 and self.config.action_horizon > 1:
                pred_delta = coarse[:, 1:, :2] - coarse[:, :-1, :2]
                target_delta = target_action_norm[:, 1:, :2] - target_action_norm[:, :-1, :2]
                losses["coarse_progress_loss"] = F.smooth_l1_loss(pred_delta.float(), target_delta.float())
        finite_or_raise("coarse_traj_norm", coarse)
        finite_or_raise("ego_tokens", ego_tokens)
        return coarse, ego_tokens, losses


class RiskTokenHead(nn.Module):
    def __init__(self, config: LastRDConfig) -> None:
        super().__init__()
        self.config = config
        self.risk_queries = nn.Parameter(torch.randn(config.num_risk_tokens, config.planner_dim) * 0.02)
        self.ego_proj = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, config.planner_dim))
        self.coarse_proj = nn.Linear(config.action_dim, config.planner_dim)
        self.cross_attn = QueryCrossAttention(config.planner_dim, config.hidden_dim)
        self.logit_head = nn.Sequential(
            nn.LayerNorm(config.planner_dim + config.action_dim),
            nn.Linear(config.planner_dim + config.action_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.risk_num_classes),
        )

    def _collect_memory(
        self,
        batch_size: int,
        reference: torch.Tensor,
        dynamic_tokens: Optional[torch.Tensor],
        geometry_tokens: Optional[torch.Tensor],
        ego_tokens: Optional[torch.Tensor],
        coarse_traj_norm: torch.Tensor,
        ego_status: torch.Tensor,
    ) -> torch.Tensor:
        parts = []
        for tokens in (dynamic_tokens, geometry_tokens, ego_tokens):
            if tokens is not None:
                parts.append(tokens)
        parts.append(self.coarse_proj(coarse_traj_norm))
        parts.append(self.ego_proj(ego_status).unsqueeze(1))
        if not parts:
            return reference.new_zeros(batch_size, 1, self.config.planner_dim)
        return torch.cat(parts, dim=1)

    def _risk_labels(
        self,
        action_input: Optional[Dict[str, Any]],
        reference: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        labels = _get(action_input, "risk_labels")
        if labels is not None:
            return labels.to(reference)
        columns = []
        for key in ("generic_risk_labels", "drivable_risk_labels", "ttc_risk_labels", "comfort_risk_labels"):
            value = _get(action_input, key)
            columns.append(value.to(reference) if isinstance(value, torch.Tensor) else None)
        if not any(value is not None for value in columns):
            return None
        batch_size = reference.shape[0]
        horizon = self.config.action_horizon
        result = reference.new_zeros(batch_size, horizon, self.config.risk_num_classes)
        for idx, value in enumerate(columns[: self.config.risk_num_classes]):
            if value is None:
                continue
            if value.ndim == 2:
                result[..., idx] = value
            elif value.ndim == 3 and value.shape[-1] == 1:
                result[..., idx] = value[..., 0]
            else:
                raise ValueError(f"Risk label for class {idx} must have shape [B,H] or [B,H,1], got {tuple(value.shape)}.")
        return result

    def forward(
        self,
        dynamic_tokens: Optional[torch.Tensor],
        geometry_tokens: Optional[torch.Tensor],
        ego_tokens: Optional[torch.Tensor],
        coarse_traj_norm: torch.Tensor,
        ego_status: torch.Tensor,
        *,
        action_input: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = coarse_traj_norm.shape[0]
        memory = self._collect_memory(batch_size, coarse_traj_norm, dynamic_tokens, geometry_tokens, ego_tokens, coarse_traj_norm, ego_status)
        queries = self.risk_queries.unsqueeze(0).expand(batch_size, -1, -1).to(memory)
        risk_tokens = self.cross_attn(queries, memory)
        risk_summary = risk_tokens.mean(dim=1).unsqueeze(1).expand(-1, self.config.action_horizon, -1)
        risk_logits = self.logit_head(torch.cat([risk_summary, coarse_traj_norm], dim=-1))
        finite_or_raise("risk_tokens", risk_tokens)
        finite_or_raise("risk_logits", risk_logits)

        losses = {"risk_loss": coarse_traj_norm.new_zeros(())}
        labels = self._risk_labels(action_input, risk_logits)
        if labels is None:
            if self.config.requires_risk_labels:
                raise KeyError("requires_risk_labels=True but no risk_labels or class-specific risk labels were provided.")
        else:
            if labels.shape != risk_logits.shape:
                raise ValueError(f"risk labels shape {tuple(labels.shape)} does not match logits {tuple(risk_logits.shape)}.")
            losses["risk_loss"] = F.binary_cross_entropy_with_logits(risk_logits.float(), labels.detach().float())
        return risk_tokens, risk_logits, losses


class SceneAwareExpertGate(nn.Module):
    def __init__(self, config: LastRDConfig) -> None:
        super().__init__()
        self.config = config
        in_dim = config.planner_dim * 6 + 8 + 3 + 12
        self.scene_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.planner_dim),
            nn.GELU(),
        )
        self.weight_head = nn.Linear(config.planner_dim, len(GROUP_NAMES))

    def forward(
        self,
        vlm_mean: torch.Tensor,
        ego_status: torch.Tensor,
        high_command_one_hot: Optional[torch.Tensor],
        history_motion: Optional[torch.Tensor],
        group_summaries: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = vlm_mean.shape[0]
        command = high_command_one_hot if high_command_one_hot is not None else _zeros(batch_size, 3, vlm_mean)
        history = history_motion if history_motion is not None else _zeros(batch_size, 12, vlm_mean)
        group_summaries = group_summaries or {}
        summaries = []
        for group in GROUP_NAMES:
            value = group_summaries.get(group)
            summaries.append(value if value is not None else _zeros(batch_size, self.config.planner_dim, vlm_mean))
        scene = self.scene_mlp(torch.cat([vlm_mean, *summaries, ego_status, command, history], dim=-1))
        weights = torch.softmax(self.weight_head(scene).float(), dim=-1).to(vlm_mean)
        return weights, scene


class TimestepAwareExpertGate(nn.Module):
    def __init__(self, config: LastRDConfig) -> None:
        super().__init__()
        self.config = config
        self.noisy_proj = nn.Sequential(
            nn.LayerNorm(config.action_dim),
            nn.Linear(config.action_dim, config.planner_dim),
            nn.GELU(),
        )
        self.weight_head = nn.Sequential(
            nn.LayerNorm(config.planner_dim * 3),
            nn.Linear(config.planner_dim * 3, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, len(GROUP_NAMES)),
        )

    def _timestep_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device, dtype=torch.float32)
            * math.log(10000.0)
            / max(half - 1, 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < dim:
            emb = F.pad(emb, (0, dim - emb.shape[-1]))
        return emb

    def forward(
        self,
        scene_embedding: torch.Tensor,
        scene_weights: torch.Tensor,
        diffusion_timestep: Optional[torch.Tensor] = None,
        noisy_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if diffusion_timestep is None or noisy_actions is None:
            return scene_weights
        t_embed = self._timestep_embedding(diffusion_timestep.to(scene_embedding.device), self.config.planner_dim).to(scene_embedding)
        noisy_summary = noisy_actions.to(scene_embedding).mean(dim=1)
        noisy_embed = self.noisy_proj(noisy_summary)
        weights = torch.softmax(self.weight_head(torch.cat([scene_embedding, t_embed, noisy_embed], dim=-1)).float(), dim=-1)
        return weights.to(scene_embedding)


class LatentSpatioTemporalReasoner(nn.Module):
    def __init__(self, config: LastRDConfig) -> None:
        super().__init__()
        self.config = config
        if config.use_future_jepa_prediction:
            self.future_jepa = FutureJEPAPredictor(config)
        if config.use_vggt_geometry_tokens:
            self.geometry_encoder = VGGTGeometryEncoder(config)
        if config.use_ego_trajectory_tokens:
            self.coarse_head = CoarseTrajectoryHead(config)
        if config.use_risk_tokens:
            self.risk_head = RiskTokenHead(config)
        if config.use_scene_aware_gate:
            self.scene_gate = SceneAwareExpertGate(config)
        if config.use_timestep_aware_gate:
            self.timestep_gate = TimestepAwareExpertGate(config)
        self.horizon_queries = nn.Parameter(torch.randn(config.action_horizon, config.planner_dim) * 0.02)
        self.horizon_attn = QueryCrossAttention(config.planner_dim, config.hidden_dim)
        self.current_train_epoch = 0
        self.total_train_epochs = 200
        self._warned_eval_targets = False

    def set_training_progress(self, epoch: int, total_epochs: int) -> None:
        self.current_train_epoch = int(epoch)
        self.total_train_epochs = max(1, int(total_epochs))

    def _resolve_context_tokens(
        self,
        action_input: Optional[Dict[str, Any]],
        vlm_tokens: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        jepa_context = _get(action_input, "jepa_context_tokens", _get(action_input, "jepa_tokens"))
        vggt_context = _get(action_input, "vggt_context_tokens", _get(action_input, "vggt_tokens"))
        jepa_context = _as_tensor(jepa_context, reference=vlm_tokens)
        vggt_context = _as_tensor(vggt_context, reference=vlm_tokens)
        if jepa_context is not None:
            _validate_tokens("jepa_context_tokens", jepa_context, batch_size=vlm_tokens.shape[0], dim=self.config.jepa_dim)
        if vggt_context is not None:
            _validate_tokens("vggt_context_tokens", vggt_context, batch_size=vlm_tokens.shape[0], dim=self.config.vggt_dim)
        return jepa_context, vggt_context

    def _target_tokens(
        self,
        action_input: Optional[Dict[str, Any]],
        vlm_tokens: torch.Tensor,
        *,
        training: bool,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if not training:
            present = [key for key in TARGET_KEYS if _get(action_input, key) is not None]
            if present and not self._warned_eval_targets:
                warnings.warn(
                    f"LaST-RD received train-only target keys during inference/eval: {present}. They are ignored.",
                    RuntimeWarning,
                )
                self._warned_eval_targets = True
            return {key: None for key in TARGET_KEYS}
        return {key: _as_tensor(_get(action_input, key), reference=vlm_tokens) for key in TARGET_KEYS}

    def _scale_tokens(
        self,
        group_weights: Optional[torch.Tensor],
        tokens_by_group: Dict[str, Optional[torch.Tensor]],
        *,
        training: bool,
        action_input: Optional[Dict[str, Any]],
    ) -> Dict[str, Optional[torch.Tensor]]:
        scaled: Dict[str, Optional[torch.Tensor]] = {}
        for idx, group in enumerate(GROUP_NAMES):
            tokens = tokens_by_group.get(group)
            if tokens is None:
                scaled[group] = None
                continue
            out = tokens
            if group_weights is not None:
                out = out * group_weights[:, idx].view(-1, 1, 1).to(out)
            if training and _get(action_input, "last_rd_group_dropout", 0.0):
                p_group = float(_get(action_input, "last_rd_group_dropout", 0.0))
                keep = torch.rand(out.shape[0], 1, 1, device=out.device) >= p_group
                out = out * keep.to(dtype=out.dtype)
            if training and _get(action_input, "last_rd_token_dropout", 0.0):
                p = float(_get(action_input, "last_rd_token_dropout", 0.0))
                out = F.dropout(out, p=p, training=True)
            zero_flag = bool(_get(action_input, f"last_rd_zero_{group}_tokens", False))
            shuffle_flag = bool(_get(action_input, f"last_rd_shuffle_{group}_tokens", False))
            if bool(_get(action_input, "last_rd_zero_all", False)):
                zero_flag = True
            if zero_flag:
                out = torch.zeros_like(out)
            elif shuffle_flag and out.shape[0] > 1:
                out = out[torch.randperm(out.shape[0], device=out.device)]
            scaled[group] = out
        return scaled

    def _group_summaries(
        self,
        vlm_tokens: torch.Tensor,
        dynamic_tokens: Optional[torch.Tensor],
        geometry_tokens: Optional[torch.Tensor],
        ego_tokens: Optional[torch.Tensor],
        risk_tokens: Optional[torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        return {
            "vlm": vlm_tokens.mean(1),
            "dynamic": dynamic_tokens.mean(1) if dynamic_tokens is not None else None,
            "geometry": geometry_tokens.mean(1) if geometry_tokens is not None else None,
            "ego": ego_tokens.mean(1) if ego_tokens is not None else None,
            "risk": risk_tokens.mean(1) if risk_tokens is not None else None,
        }

    def _horizon_condition(self, planning_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = planning_tokens.shape[0]
        queries = self.horizon_queries.unsqueeze(0).expand(batch_size, -1, -1).to(planning_tokens)
        condition = self.horizon_attn(queries, planning_tokens)
        finite_or_raise("horizon_condition", condition)
        return condition

    def forward(
        self,
        vlm_tokens: torch.Tensor,
        action_input: Optional[Dict[str, Any]],
        *,
        training: bool,
        allow_target_tokens: Optional[bool] = None,
        noisy_actions: Optional[torch.Tensor] = None,
        diffusion_timestep: Optional[torch.Tensor] = None,
        norm_odo=None,
        target_action_norm: Optional[torch.Tensor] = None,
    ) -> LastRDOutput:
        batch_size = vlm_tokens.shape[0]
        zero = vlm_tokens.new_zeros(())
        losses: Dict[str, torch.Tensor] = {
            "future_jepa_loss": zero,
            "vggt_geometry_loss": zero,
            "vggt_token_loss": zero,
            "coarse_traj_loss": zero,
            "coarse_heading_loss": zero,
            "coarse_progress_loss": zero,
            "risk_loss": zero,
        }
        diagnostics: Dict[str, Any] = {}

        ego_status = _as_tensor(_get(action_input, "status_feature"), reference=vlm_tokens)
        if ego_status is None:
            raise KeyError("LaST-RD requires action_input.status_feature.")
        high_command = _as_tensor(_get(action_input, "high_command_one_hot"), reference=vlm_tokens)
        if high_command is None and ego_status.shape[-1] >= 3:
            high_command = ego_status[..., :3]
        history = _as_tensor(_get(action_input, "history_trajectory", _get(action_input, "his_traj")), reference=vlm_tokens)
        if history is not None and history.ndim == 3:
            history = history.reshape(history.shape[0], -1)
        if history is None:
            history = _zeros(batch_size, 12, vlm_tokens)

        if target_action_norm is None:
            action = _as_tensor(_get(action_input, "action"), reference=vlm_tokens)
            if action is not None:
                target_action_norm = norm_odo(action) if norm_odo is not None else action
        if target_action_norm is not None:
            target_action_norm = target_action_norm.detach().to(vlm_tokens)

        jepa_context, vggt_context = self._resolve_context_tokens(action_input, vlm_tokens)
        target_training = training if allow_target_tokens is None else bool(allow_target_tokens)
        targets = self._target_tokens(action_input, vlm_tokens, training=target_training)

        dynamic_tokens = predicted_future_jepa = None
        if self.config.use_future_jepa_prediction:
            predicted_future_jepa, dynamic_tokens, dyn_losses = self.future_jepa(
                vlm_tokens,
                jepa_context,
                ego_status,
                high_command,
                history,
                jepa_target_tokens=targets["jepa_target_tokens"],
            )
            losses.update(dyn_losses)

        geometry_tokens = predicted_geometry = None
        if self.config.use_vggt_geometry_tokens:
            geometry_tokens, predicted_geometry, geo_losses, geo_diag = self.geometry_encoder(
                vggt_context,
                vlm_tokens,
                ego_status,
                high_command,
                action_input=action_input if target_training else None,
                vggt_target_tokens=targets["vggt_target_tokens"],
            )
            losses.update(geo_losses)
            diagnostics.update(geo_diag)

        seed_tokens = [vlm_tokens]
        if dynamic_tokens is not None:
            seed_tokens.append(dynamic_tokens)
        if geometry_tokens is not None:
            seed_tokens.append(geometry_tokens)
        preliminary_tokens = torch.cat(seed_tokens, dim=1)

        coarse_traj_norm = ego_tokens = None
        if self.config.use_ego_trajectory_tokens:
            coarse_traj_norm, ego_tokens, coarse_losses = self.coarse_head(
                preliminary_tokens,
                vlm_tokens.mean(1),
                ego_status,
                high_command,
                history,
                target_action_norm=target_action_norm,
            )
            losses.update(coarse_losses)

        risk_tokens = risk_logits = None
        if self.config.use_risk_tokens:
            if coarse_traj_norm is None:
                coarse_traj_norm = vlm_tokens.new_zeros(batch_size, self.config.action_horizon, self.config.action_dim)
            risk_tokens, risk_logits, risk_losses = self.risk_head(
                dynamic_tokens,
                geometry_tokens,
                ego_tokens,
                coarse_traj_norm,
                ego_status,
                action_input=action_input if target_training else None,
            )
            losses.update(risk_losses)

        group_summaries = self._group_summaries(vlm_tokens, dynamic_tokens, geometry_tokens, ego_tokens, risk_tokens)
        group_weights = None
        scene_embedding = vlm_tokens.mean(1)
        if self.config.use_scene_aware_gate:
            group_weights, scene_embedding = self.scene_gate(
                vlm_tokens.mean(1),
                ego_status,
                high_command,
                history,
                group_summaries,
            )
        if self.config.use_timestep_aware_gate:
            base_weights = group_weights if group_weights is not None else vlm_tokens.new_full((batch_size, len(GROUP_NAMES)), 1.0 / len(GROUP_NAMES))
            group_weights = self.timestep_gate(scene_embedding, base_weights, diffusion_timestep, noisy_actions)

        tokens_by_group = {
            "vlm": vlm_tokens,
            "dynamic": dynamic_tokens,
            "geometry": geometry_tokens,
            "ego": ego_tokens,
            "risk": risk_tokens,
        }
        scaled = self._scale_tokens(group_weights, tokens_by_group, training=training, action_input=action_input)
        planning_parts = [tokens for group, tokens in scaled.items() if group != "vlm" and tokens is not None]
        if not planning_parts:
            planning_parts = [vlm_tokens.new_zeros(batch_size, 1, self.config.planner_dim)]
        planning_tokens = torch.cat(planning_parts, dim=1)
        horizon_condition = self._horizon_condition(torch.cat([scaled["vlm"], planning_tokens], dim=1) if scaled["vlm"] is not None else planning_tokens)

        if group_weights is not None:
            for idx, name in enumerate(GROUP_NAMES):
                diagnostics[f"last_rd_group_weight_{name}"] = group_weights[:, idx].detach().mean()
            diagnostics["last_rd_group_weights"] = group_weights.detach()
        diagnostics["last_rd_token_norms"] = planning_tokens.detach().float().norm(dim=-1).mean()
        if coarse_traj_norm is not None and target_action_norm is not None:
            diagnostics["coarse_traj_l1"] = (coarse_traj_norm.detach() - target_action_norm.detach()).abs().mean()
        diagnostics["future_jepa_loss_raw"] = losses["future_jepa_loss"].detach()
        diagnostics.setdefault("geometry_mode", "no_geometry")
        diagnostics.setdefault("geometry_mode_code", vlm_tokens.new_tensor(float(GEOMETRY_MODE_TO_CODE["no_geometry"])))

        for key, value in losses.items():
            finite_or_raise(key, value)
        finite_or_raise("planning_tokens", planning_tokens)

        return LastRDOutput(
            planning_tokens=planning_tokens,
            dynamic_tokens=dynamic_tokens,
            geometry_tokens=geometry_tokens,
            ego_tokens=ego_tokens,
            risk_tokens=risk_tokens,
            coarse_traj_norm=coarse_traj_norm,
            predicted_future_jepa=predicted_future_jepa,
            predicted_geometry=predicted_geometry,
            risk_logits=risk_logits,
            group_weights=group_weights,
            horizon_condition=horizon_condition,
            losses=losses,
            diagnostics=diagnostics,
        )
