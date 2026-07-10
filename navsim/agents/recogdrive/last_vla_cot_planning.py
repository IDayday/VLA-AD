from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class LastVLACoTConfig:
    planner_dim: int = 384
    vlm_dim: int = 384
    jepa_dim: int = 1024
    vggt_dim: int = 2048
    hidden_dim: int = 1024
    action_dim: int = 3
    action_horizon: int = 8

    cot_num_tokens: int = 32
    cot_num_steps: int = 4
    geometry_tokens: int = 12
    dynamic_tokens: int = 12
    ego_tokens: int = 8
    risk_tokens: int = 8

    use_geometry_step: bool = True
    use_dynamic_step: bool = True
    use_ego_step: bool = True
    use_action_refine_step: bool = True
    use_action_conditioned_dynamics: bool = True
    use_cot_risk_head: bool = True

    raw_vlm_context_to_dit: bool = True
    cot_bottleneck_mode: bool = False
    min_cot_context_ratio: float = 0.80
    vlm_context_dropout_start: float = 0.0
    vlm_context_dropout_end: float = 0.7

    use_residual_diffusion: bool = False
    residual_detach_coarse_for_diffusion: bool = True
    coarse_prior_clip: float = 1.0
    use_fs_norm: bool = False

    require_full_geometry: bool = False
    allow_patch_geometry_fallback: bool = False
    geometry_teacher_dim: int = 2048
    geometry_grid_rows: int = 3
    geometry_grid_cols: int = 4

    risk_num_classes: int = 4
    requires_risk_labels: bool = False

    teacher_traj_mode: Literal["none", "gt", "teacher_if_better", "mix"] = "none"
    teacher_traj_mix_start: float = 0.0
    teacher_traj_mix_end: float = 1.0

    normalized_teacher_loss: bool = True
    allow_missing_dynamic_teacher: bool = False

    condition_mode: str = "decoupled_cot_residual"
    use_scene_step: bool = True
    use_parallel_geometry_dynamic: bool = True
    fusion_tokens: int = 192
    dynamic_uses_geometry_memory: bool = True
    cot_condition_zero_init: bool = True
    cot_condition_dropout: float = 0.0
    cot_condition_scale_init: float = 1.0
    cot_condition_trainable_scale: bool = True


@dataclass
class CoTStepOutput:
    name: str
    tokens: torch.Tensor
    teacher_pred: Optional[torch.Tensor]
    teacher_target: Optional[torch.Tensor]
    loss: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]


@dataclass
class LastVLAOutput:
    cot_tokens: torch.Tensor
    cot_tokens_by_step: Dict[str, torch.Tensor]
    raw_vlm_context_tokens: torch.Tensor
    cot_condition_tokens: torch.Tensor
    cot_scene_tokens: torch.Tensor
    cot_geometry_tokens: torch.Tensor
    cot_dynamic_tokens: torch.Tensor
    cot_fusion_tokens: torch.Tensor
    base_context_tokens: torch.Tensor
    planner_context_tokens: torch.Tensor
    context_mean: torch.Tensor
    horizon_condition: torch.Tensor
    coarse_traj_norm: torch.Tensor
    residual_target_norm: Optional[torch.Tensor]
    final_target_norm: Optional[torch.Tensor]
    predicted_future_jepa: Optional[torch.Tensor]
    predicted_geometry: Optional[torch.Tensor]
    risk_logits: Optional[torch.Tensor]
    losses: Dict[str, torch.Tensor]
    diagnostics: Dict[str, torch.Tensor]


GEOMETRY_MODE_TO_CODE = {"missing": -1, "no_geometry": 0, "patch_fallback": 1, "full_geometry": 2}
GEOMETRY_CODE_TO_MODE = {-1: "missing", 0: "no_geometry", 1: "patch_fallback", 2: "full_geometry"}
CONTEXT_MODE_TO_CODE = {"cot_bottleneck": 1, "cot_plus_raw_vlm": 2, "decoupled_cot_residual": 3}
TARGET_KEYS = (
    "jepa_target_tokens",
    "vggt_target_tokens",
    "vggt_geometry_target_tokens",
    "vggt_depth_target_tokens",
    "vggt_pointmap_target_tokens",
)


def finite_or_raise(name: str, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is not None and not torch.isfinite(tensor.float()).all():
        raise FloatingPointError(f"{name} contains non-finite values.")
    return tensor


def normalized_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.detach().to(device=pred.device, dtype=pred.dtype)
    pred_norm = F.normalize(pred.float(), p=2, dim=-1, eps=1e-6)
    target_norm = F.normalize(target.float(), p=2, dim=-1, eps=1e-6)
    return F.mse_loss(pred_norm, target_norm)


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.detach().to(device=pred.device, dtype=pred.dtype)
    return (1.0 - F.cosine_similarity(pred.float(), target.float(), dim=-1)).mean()


def smooth_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred.float(), target.detach().to(device=pred.device, dtype=pred.dtype).float())


def detach_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {key: detach_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(detach_tree(item) for item in value)
    return value


def match_token_count(tokens: torch.Tensor, count: int) -> torch.Tensor:
    if tokens.shape[1] == count:
        return tokens
    return F.interpolate(tokens.transpose(1, 2).float(), size=count, mode="linear", align_corners=False).transpose(1, 2).to(tokens)


def match_last_dim(tokens: torch.Tensor, dim: int) -> torch.Tensor:
    if tokens.shape[-1] == dim:
        return tokens
    if tokens.shape[-1] > dim:
        return tokens[..., :dim]
    return F.pad(tokens, (0, dim - tokens.shape[-1]))


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    if t.ndim == 0:
        t = t.unsqueeze(0)
    half = dim // 2
    exponent = -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    freqs = torch.exp(exponent)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb.to(dtype=torch.float32)


def schedule_linear(start: float, end: float, progress: float) -> float:
    progress = min(max(float(progress), 0.0), 1.0)
    return float(start) + (float(end) - float(start)) * progress


def safe_get(action_input: Optional[Dict[str, Any]], key: str, default: Any = None) -> Any:
    if action_input is None:
        return default
    if key in action_input:
        return action_input[key]
    return getattr(action_input, key, default)


def _bool_flag(action_input: Optional[Dict[str, Any]], key: str) -> bool:
    value = safe_get(action_input, key, False)
    if isinstance(value, torch.Tensor):
        return bool(value.detach().float().max().item() > 0.5)
    return bool(value)


def _num_heads(dim: int) -> int:
    for heads in (8, 6, 4, 3, 2, 1):
        if dim % heads == 0:
            return heads
    return 1


def _infer_token_grid(num_tokens: int) -> tuple[int, int]:
    best = 1
    for rows in range(1, int(num_tokens**0.5) + 1):
        if num_tokens % rows == 0:
            best = rows
    return best, num_tokens // best


def _zero(batch_size: int, dim: int, ref: torch.Tensor) -> torch.Tensor:
    return ref.new_zeros(batch_size, dim)


def _validate_tokens(name: str, tokens: torch.Tensor, *, batch_size: int, dim: Optional[int] = None) -> torch.Tensor:
    if not isinstance(tokens, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tokens).__name__}.")
    if tokens.ndim != 3:
        raise ValueError(f"{name} must have shape [B, K, D], got {tuple(tokens.shape)}.")
    if tokens.shape[0] != batch_size:
        raise ValueError(f"{name} batch {tokens.shape[0]} does not match B={batch_size}.")
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


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, _num_heads(dim), batch_first=True)
        self.mlp = ResidualMLP(dim, hidden_dim)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(self.query_norm(query), self.memory_norm(memory), self.memory_norm(memory), need_weights=False)
        return self.mlp(query + attended)


class LatentCoTStepBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, _num_heads(dim), batch_first=True)
        self.memory_attn = CrossAttentionBlock(dim, hidden_dim)
        self.action_attn = CrossAttentionBlock(dim, hidden_dim)
        self.t_norm = nn.LayerNorm(dim)
        self.timestep_mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2 * dim))
        self.mlp = ResidualMLP(dim, hidden_dim)

    def forward(
        self,
        current_cot_tokens: torch.Tensor,
        memory_tokens: torch.Tensor,
        action_tokens: Optional[torch.Tensor] = None,
        timestep_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attended, _ = self.self_attn(
            self.self_norm(current_cot_tokens),
            self.self_norm(current_cot_tokens),
            self.self_norm(current_cot_tokens),
            need_weights=False,
        )
        x = current_cot_tokens + attended
        x = self.memory_attn(x, memory_tokens)
        if action_tokens is not None:
            x = self.action_attn(x, action_tokens)
        if timestep_embedding is not None:
            scale_shift = self.timestep_mlp(timestep_embedding.to(device=x.device, dtype=x.dtype)).unsqueeze(1)
            scale, shift = scale_shift.chunk(2, dim=-1)
            x = x + self.t_norm(x) * (1.0 + scale) + shift
        x = self.mlp(x)
        finite_or_raise("latent_cot_step", x)
        return x


class ActionTokenEncoder(nn.Module):
    def __init__(self, config: LastVLACoTConfig) -> None:
        super().__init__()
        self.action_proj = nn.Sequential(
            nn.Linear(config.action_dim, config.planner_dim),
            nn.GELU(),
            nn.Linear(config.planner_dim, config.planner_dim),
            nn.LayerNorm(config.planner_dim),
        )
        self.t_proj = nn.Sequential(
            nn.LayerNorm(config.planner_dim),
            nn.Linear(config.planner_dim, config.planner_dim),
            nn.GELU(),
            nn.Linear(config.planner_dim, config.planner_dim),
        )

    def forward(
        self,
        coarse_traj_norm: Optional[torch.Tensor],
        noisy_action_norm: Optional[torch.Tensor] = None,
        timestep_embedding: Optional[torch.Tensor] = None,
        reference: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        parts = []
        if coarse_traj_norm is not None:
            parts.append(self.action_proj(coarse_traj_norm))
        if noisy_action_norm is not None:
            parts.append(self.action_proj(noisy_action_norm))
        if coarse_traj_norm is not None and noisy_action_norm is not None:
            parts.append(self.action_proj(coarse_traj_norm + noisy_action_norm))
        if timestep_embedding is not None:
            parts.append(self.t_proj(timestep_embedding).unsqueeze(1))
        if not parts:
            if reference is None:
                return None
            return reference.new_zeros(reference.shape[0], 1, reference.shape[-1])
        return torch.cat(parts, dim=1)


class GeometryTeacherHead(nn.Module):
    def __init__(self, config: LastVLACoTConfig) -> None:
        super().__init__()
        self.config = config
        self.pred_head = nn.Sequential(
            nn.LayerNorm(config.planner_dim),
            nn.Linear(config.planner_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.geometry_teacher_dim),
        )

    def _mode_from_action_input(self, action_input: Optional[Dict[str, Any]]) -> str:
        raw_code = safe_get(action_input, "vggt_geometry_mode_code")
        if isinstance(raw_code, torch.Tensor):
            codes = raw_code.detach().cpu().view(-1).long().tolist()
            if codes and all(code == GEOMETRY_MODE_TO_CODE["full_geometry"] for code in codes):
                return "full_geometry"
            if any(code == GEOMETRY_MODE_TO_CODE["patch_fallback"] for code in codes):
                return "patch_fallback"
            if any(code == GEOMETRY_MODE_TO_CODE["no_geometry"] for code in codes):
                return "no_geometry"
            return "missing"
        raw = safe_get(action_input, "vggt_geometry_mode")
        if isinstance(raw, str):
            return raw if raw in GEOMETRY_MODE_TO_CODE else "missing"
        return "missing"

    def _target(
        self,
        action_input: Optional[Dict[str, Any]],
        ref: torch.Tensor,
        *,
        training: bool,
        allow_target_tokens: bool,
    ) -> Tuple[Optional[torch.Tensor], str]:
        batch_size = ref.shape[0]
        strict_full_geometry = bool(self.config.require_full_geometry)

        def prepare_tokens(name: str, value: torch.Tensor) -> torch.Tensor:
            tokens = _validate_tokens(
                name,
                value.to(device=ref.device, dtype=ref.dtype),
                batch_size=batch_size,
                dim=self.config.geometry_teacher_dim if strict_full_geometry else None,
            )
            if strict_full_geometry:
                if tokens.shape[1] != self.config.geometry_tokens:
                    raise ValueError(
                        f"{name} token count {tokens.shape[1]} != expected {self.config.geometry_tokens}."
                    )
                return tokens
            return match_token_count(match_last_dim(tokens, self.config.geometry_teacher_dim), self.config.geometry_tokens)

        explicit_mode = self._mode_from_action_input(action_input)
        full_keys = ["vggt_geometry_tokens", "vggt_depth_tokens", "vggt_pointmap_tokens", "vggt_camera_tokens"]
        if training and allow_target_tokens:
            full_keys.append("vggt_geometry_target_tokens")
        for key in full_keys:
            tokens = safe_get(action_input, key)
            if isinstance(tokens, torch.Tensor):
                if explicit_mode == "full_geometry":
                    return prepare_tokens(key, tokens), "full_geometry"
                if explicit_mode == "patch_fallback":
                    if self.config.allow_patch_geometry_fallback:
                        return prepare_tokens(key, tokens), "patch_fallback"
                    raise KeyError("VGGT geometry tokens are marked patch_fallback, but patch fallback is disabled.")
                if explicit_mode == "no_geometry":
                    if self.config.require_full_geometry:
                        raise KeyError("Full geometry is required, but vggt_geometry_mode is no_geometry.")
                    return None, "no_geometry"
                if self.config.require_full_geometry:
                    raise KeyError("Full geometry is required, but vggt_geometry_mode is not full_geometry.")
                if self.config.allow_patch_geometry_fallback:
                    warnings.warn(
                        f"{key} is present without explicit full_geometry mode; treating it as patch_fallback.",
                        RuntimeWarning,
                    )
                    return prepare_tokens(key, tokens), "patch_fallback"
                return None, "missing"
        fallback = safe_get(action_input, "vggt_context_tokens")
        if isinstance(fallback, torch.Tensor) and self.config.allow_patch_geometry_fallback:
            fallback = _validate_tokens("vggt_context_tokens", fallback.to(device=ref.device, dtype=ref.dtype), batch_size=batch_size)
            return match_token_count(match_last_dim(fallback, self.config.geometry_teacher_dim), self.config.geometry_tokens), "patch_fallback"
        if self.config.require_full_geometry:
            raise KeyError(
                "Last-VLA geometry teacher requires full geometry keys "
                "(vggt_geometry_tokens/depth/pointmap/camera or geometry target)."
            )
        return None, "no_geometry"

    def forward(
        self,
        geometry_cot_tokens: torch.Tensor,
        action_input: Optional[Dict[str, Any]],
        *,
        training: bool,
        allow_target_tokens: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        pred_tokens = match_token_count(geometry_cot_tokens, self.config.geometry_tokens)
        predicted = self.pred_head(pred_tokens)
        target, mode = self._target(
            action_input,
            geometry_cot_tokens,
            training=training,
            allow_target_tokens=allow_target_tokens,
        )
        if target is not None:
            loss = normalized_mse(predicted, target) if self.config.normalized_teacher_loss else smooth_l1(predicted, target)
        else:
            loss = predicted.new_zeros(())
        diagnostics = {
            "geometry_mode_code": predicted.new_tensor(float(GEOMETRY_MODE_TO_CODE[mode])),
            "geometry_loss_raw": loss.detach().float(),
        }
        return predicted, target, loss, diagnostics


class DynamicTeacherHead(nn.Module):
    def __init__(self, config: LastVLACoTConfig) -> None:
        super().__init__()
        self.config = config
        self.action_encoder = ActionTokenEncoder(config)
        self.action_attn = CrossAttentionBlock(config.planner_dim, config.hidden_dim)
        self.pred_head = nn.Sequential(
            nn.LayerNorm(config.planner_dim),
            nn.Linear(config.planner_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.jepa_dim),
        )

    def forward(
        self,
        dynamic_cot_tokens: torch.Tensor,
        jepa_target_tokens: Optional[torch.Tensor],
        *,
        training: bool,
        coarse_traj_norm: Optional[torch.Tensor] = None,
        noisy_action_norm: Optional[torch.Tensor] = None,
        diffusion_timestep: Optional[torch.Tensor] = None,
        allow_target_tokens: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        t_embed = None
        if diffusion_timestep is not None:
            t_embed = sinusoidal_timestep_embedding(diffusion_timestep, self.config.planner_dim).to(dynamic_cot_tokens)
        tokens = dynamic_cot_tokens
        if self.config.use_action_conditioned_dynamics:
            action_tokens = self.action_encoder(coarse_traj_norm, noisy_action_norm, t_embed, reference=dynamic_cot_tokens)
            if action_tokens is not None:
                tokens = self.action_attn(tokens, action_tokens)
        pred_tokens = match_token_count(tokens, self.config.dynamic_tokens)
        predicted = self.pred_head(pred_tokens)
        target = None
        if training and allow_target_tokens and isinstance(jepa_target_tokens, torch.Tensor):
            target = _validate_tokens("jepa_target_tokens", jepa_target_tokens.to(predicted), batch_size=predicted.shape[0], dim=self.config.jepa_dim)
            if target.shape[1] != self.config.dynamic_tokens:
                if self.config.dynamic_tokens > 12:
                    raise ValueError(
                        f"jepa_target_tokens token count {target.shape[1]} != expected {self.config.dynamic_tokens}."
                    )
                target = match_token_count(target, self.config.dynamic_tokens)
            loss = normalized_mse(predicted, target) if self.config.normalized_teacher_loss else smooth_l1(predicted, target)
        else:
            loss = predicted.new_zeros(())
        return predicted, target, loss


class CoTCoarseTrajectoryHead(nn.Module):
    def __init__(self, config: LastVLACoTConfig) -> None:
        super().__init__()
        self.config = config
        self.horizon_queries = nn.Parameter(torch.randn(config.action_horizon, config.planner_dim) * 0.02)
        self.ego_queries = nn.Parameter(torch.randn(config.ego_tokens, config.planner_dim) * 0.02)
        self.horizon_attn = CrossAttentionBlock(config.planner_dim, config.hidden_dim)
        self.ego_attn = CrossAttentionBlock(config.planner_dim, config.hidden_dim)
        self.state_proj = nn.Sequential(
            nn.LayerNorm(8 + 3 + 12),
            nn.Linear(8 + 3 + 12, config.planner_dim),
            nn.GELU(),
            nn.Linear(config.planner_dim, config.planner_dim),
        )
        self.traj_head = nn.Sequential(
            nn.LayerNorm(config.planner_dim),
            nn.Linear(config.planner_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )

    def _state_token(
        self,
        ego_status: torch.Tensor,
        high_command_one_hot: Optional[torch.Tensor],
        history_trajectory_flat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = ego_status.shape[0]
        command = high_command_one_hot if high_command_one_hot is not None else _zero(batch_size, 3, ego_status)
        history = history_trajectory_flat if history_trajectory_flat is not None else _zero(batch_size, 12, ego_status)
        return self.state_proj(torch.cat([ego_status, command, history], dim=-1)).unsqueeze(1)

    @staticmethod
    def _heading_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_angle = pred.float() * math.pi
        target_angle = target.detach().to(pred).float() * math.pi
        pred_vec = torch.stack([torch.sin(pred_angle), torch.cos(pred_angle)], dim=-1)
        target_vec = torch.stack([torch.sin(target_angle), torch.cos(target_angle)], dim=-1)
        return F.smooth_l1_loss(pred_vec, target_vec)

    def forward(
        self,
        cot_tokens: torch.Tensor,
        ego_status: torch.Tensor,
        high_command_one_hot: Optional[torch.Tensor],
        history_trajectory_flat: Optional[torch.Tensor],
        target_action_norm: Optional[torch.Tensor] = None,
        output_bound_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = cot_tokens.shape[0]
        memory = torch.cat([cot_tokens, self._state_token(ego_status, high_command_one_hot, history_trajectory_flat)], dim=1)
        horizon_queries = self.horizon_queries.unsqueeze(0).expand(batch_size, -1, -1).to(cot_tokens)
        horizon_hidden = self.horizon_attn(horizon_queries, memory)
        coarse_logits = self.traj_head(horizon_hidden)
        if self.config.use_fs_norm:
            if output_bound_fn is None:
                raise ValueError("FS-Norm Last-VLA coarse prediction requires the planner output-bound function.")
            coarse = output_bound_fn(coarse_logits)
        else:
            coarse = torch.tanh(coarse_logits)
            if self.config.coarse_prior_clip > 0:
                coarse = coarse.clamp(-float(self.config.coarse_prior_clip), float(self.config.coarse_prior_clip))
        ego_queries = self.ego_queries.unsqueeze(0).expand(batch_size, -1, -1).to(cot_tokens)
        ego_tokens = self.ego_attn(ego_queries, torch.cat([memory, horizon_hidden], dim=1))

        zero = coarse.new_zeros(())
        losses = {"coarse_loss": zero, "heading_loss": zero, "progress_loss": zero}
        if target_action_norm is not None:
            target = target_action_norm.detach().to(coarse)
            losses["coarse_loss"] = smooth_l1(coarse, target)
            if not self.config.use_fs_norm and coarse.shape[-1] >= 3:
                losses["heading_loss"] = self._heading_loss(coarse[..., 2], target[..., 2])
            if not self.config.use_fs_norm and coarse.shape[1] > 1 and coarse.shape[-1] >= 2:
                losses["progress_loss"] = smooth_l1(coarse[:, 1:, :2] - coarse[:, :-1, :2], target[:, 1:, :2] - target[:, :-1, :2])
        finite_or_raise("coarse_traj_norm", coarse)
        return coarse, ego_tokens, losses


class RiskTeacherHead(nn.Module):
    def __init__(self, config: LastVLACoTConfig) -> None:
        super().__init__()
        self.config = config
        self.risk_queries = nn.Parameter(torch.randn(config.risk_tokens, config.planner_dim) * 0.02)
        self.cross_attn = CrossAttentionBlock(config.planner_dim, config.hidden_dim)
        self.logit_head = nn.Sequential(
            nn.LayerNorm(config.planner_dim),
            nn.Linear(config.planner_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.risk_num_classes),
        )

    def _labels(self, action_input: Optional[Dict[str, Any]], ref: torch.Tensor) -> Optional[torch.Tensor]:
        labels = safe_get(action_input, "risk_labels")
        if isinstance(labels, torch.Tensor):
            labels = labels.to(ref)
            return labels.unsqueeze(-1) if labels.ndim == 2 else labels
        parts = []
        for key in ("generic_risk_labels", "drivable_risk_labels", "ttc_risk_labels", "comfort_risk_labels"):
            value = safe_get(action_input, key)
            if isinstance(value, torch.Tensor):
                parts.append(value.to(ref).unsqueeze(-1) if value.ndim == 2 else value.to(ref))
        if parts:
            return torch.cat(parts, dim=-1)
        if self.config.requires_risk_labels:
            raise KeyError("Last-VLA risk head requires risk_labels or per-class risk labels.")
        return None

    def forward(self, cot_tokens: torch.Tensor, coarse_traj_norm: torch.Tensor, action_input: Optional[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = cot_tokens.shape[0]
        queries = self.risk_queries.unsqueeze(0).expand(batch_size, -1, -1).to(cot_tokens)
        risk_tokens = self.cross_attn(queries, cot_tokens)
        horizon_tokens = match_token_count(risk_tokens, self.config.action_horizon)
        logits = self.logit_head(horizon_tokens)
        labels = self._labels(action_input, logits)
        if labels is None:
            loss = logits.new_zeros(())
        else:
            labels = match_last_dim(labels.to(logits), self.config.risk_num_classes)
            loss = F.binary_cross_entropy_with_logits(logits.float(), labels.detach().float())
        return risk_tokens, logits, loss


class LastVLACoTTransformer(nn.Module):
    def __init__(self, config: LastVLACoTConfig) -> None:
        super().__init__()
        self.config = config
        if config.condition_mode != "decoupled_cot_residual" or config.cot_bottleneck_mode or not config.raw_vlm_context_to_dit:
            raise ValueError(
                "LastVLACoTTransformer only supports decoupled_cot_residual with raw VLM context preserved."
            )
        if config.require_full_geometry and config.allow_patch_geometry_fallback:
            raise ValueError("require_full_geometry=True is incompatible with allow_patch_geometry_fallback=True.")
        if config.geometry_tokens != int(config.geometry_grid_rows) * int(config.geometry_grid_cols):
            if (int(config.geometry_grid_rows), int(config.geometry_grid_cols)) == (3, 4):
                config.geometry_grid_rows, config.geometry_grid_cols = _infer_token_grid(config.geometry_tokens)
            else:
                raise ValueError(
                    "geometry_tokens must equal geometry_grid_rows * geometry_grid_cols "
                    f"({config.geometry_tokens} != {config.geometry_grid_rows} * {config.geometry_grid_cols})."
                )
        if config.geometry_tokens != int(config.geometry_grid_rows) * int(config.geometry_grid_cols):
            raise ValueError(
                "geometry_tokens must equal geometry_grid_rows * geometry_grid_cols "
                f"({config.geometry_tokens} != {config.geometry_grid_rows} * {config.geometry_grid_cols})."
            )
        if not config.use_cot_risk_head and config.risk_tokens != 0:
            warnings.warn(
                "use_cot_risk_head=False ignores configured risk_tokens; no risk tokens will be appended.",
                RuntimeWarning,
            )
        self.current_epoch = 0
        self.total_epochs = 1

        self.cot_queries = nn.Parameter(torch.randn(config.cot_num_tokens, config.planner_dim) * 0.02)
        self.state_proj = nn.Sequential(
            nn.LayerNorm(8 + 3 + 12),
            nn.Linear(8 + 3 + 12, config.planner_dim),
            nn.GELU(),
            nn.Linear(config.planner_dim, config.planner_dim),
        )
        self.jepa_context_proj = nn.Sequential(nn.LayerNorm(config.jepa_dim), nn.Linear(config.jepa_dim, config.planner_dim))
        self.vggt_context_proj = nn.Sequential(nn.LayerNorm(config.vggt_dim), nn.Linear(config.vggt_dim, config.planner_dim))
        self.geometry_pred_proj = nn.Sequential(nn.LayerNorm(config.geometry_teacher_dim), nn.Linear(config.geometry_teacher_dim, config.planner_dim))
        self.action_encoder = ActionTokenEncoder(config)

        self.scene_step = LatentCoTStepBlock(config.planner_dim, config.hidden_dim)
        self.geometry_step = LatentCoTStepBlock(config.planner_dim, config.hidden_dim)
        self.dynamic_step = LatentCoTStepBlock(config.planner_dim, config.hidden_dim)
        self.fusion_step = LatentCoTStepBlock(config.planner_dim, config.hidden_dim)
        self.ego_step = LatentCoTStepBlock(config.planner_dim, config.hidden_dim)
        self.action_refine_step = LatentCoTStepBlock(config.planner_dim, config.hidden_dim)

        self.geometry_head = GeometryTeacherHead(config)
        self.dynamic_head = DynamicTeacherHead(config)
        self.coarse_head = CoTCoarseTrajectoryHead(config)
        self.risk_head = RiskTeacherHead(config) if config.use_cot_risk_head and config.risk_tokens > 0 else None
        self.horizon_queries = nn.Parameter(torch.randn(config.action_horizon, config.planner_dim) * 0.02)
        self.horizon_attn = CrossAttentionBlock(config.planner_dim, config.hidden_dim)

    def set_training_progress(self, epoch: int, total_epochs: int) -> None:
        self.current_epoch = int(epoch)
        self.total_epochs = max(1, int(total_epochs))

    def _progress(self, current_epoch: Optional[int], total_epochs: Optional[int]) -> float:
        epoch = self.current_epoch if current_epoch is None else int(current_epoch)
        total = self.total_epochs if total_epochs is None else max(1, int(total_epochs))
        return min(max(float(epoch) / float(total), 0.0), 1.0)

    def _state_token(self, ego_status: torch.Tensor, command: Optional[torch.Tensor], history: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size = ego_status.shape[0]
        command = command if command is not None else _zero(batch_size, 3, ego_status)
        history = history if history is not None else _zero(batch_size, 12, ego_status)
        return self.state_proj(torch.cat([ego_status, command, history], dim=-1)).unsqueeze(1)

    def _context_tokens(self, action_input: Optional[Dict[str, Any]], key: str, dim: int, ref: torch.Tensor) -> Optional[torch.Tensor]:
        value = safe_get(action_input, key)
        if not isinstance(value, torch.Tensor):
            return None
        value = _validate_tokens(key, value.to(device=ref.device, dtype=ref.dtype), batch_size=ref.shape[0], dim=dim)
        return value

    def _geometry_memory(self, action_input: Optional[Dict[str, Any]], ref: torch.Tensor) -> Optional[torch.Tensor]:
        mode = self.geometry_head._mode_from_action_input(action_input)
        strict_full_geometry = bool(self.config.require_full_geometry)

        def prepare_geometry_memory(name: str, value: torch.Tensor) -> torch.Tensor:
            tokens = _validate_tokens(
                name,
                value.to(ref),
                batch_size=ref.shape[0],
                dim=self.config.geometry_teacher_dim if strict_full_geometry else None,
            )
            if strict_full_geometry:
                if tokens.shape[1] != self.config.geometry_tokens:
                    raise ValueError(
                        f"{name} token count {tokens.shape[1]} != expected {self.config.geometry_tokens}."
                    )
                return tokens
            return match_token_count(match_last_dim(tokens, self.config.geometry_teacher_dim), self.config.geometry_tokens)

        for key in ("vggt_geometry_tokens", "vggt_depth_tokens", "vggt_pointmap_tokens", "vggt_camera_tokens"):
            value = safe_get(action_input, key)
            if isinstance(value, torch.Tensor):
                if mode == "full_geometry":
                    return self.geometry_pred_proj(prepare_geometry_memory(key, value))
                if mode == "patch_fallback" and self.config.allow_patch_geometry_fallback:
                    return self.geometry_pred_proj(prepare_geometry_memory(key, value))
                if mode == "no_geometry":
                    return None
                if self.config.require_full_geometry:
                    raise KeyError("Full geometry memory is required, but vggt_geometry_mode is not full_geometry.")
                if not self.config.allow_patch_geometry_fallback:
                    return None
                return self.geometry_pred_proj(prepare_geometry_memory(key, value))
        value = self._context_tokens(action_input, "vggt_context_tokens", self.config.vggt_dim, ref)
        if value is not None and self.config.allow_patch_geometry_fallback:
            return self.vggt_context_proj(value)
        return None

    def forward(
        self,
        vlm_tokens: torch.Tensor,
        action_input: Optional[Dict[str, Any]],
        *,
        training: bool,
        target_action_norm: Optional[torch.Tensor] = None,
        noisy_action_norm: Optional[torch.Tensor] = None,
        diffusion_timestep: Optional[torch.Tensor] = None,
        current_epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
        allow_target_tokens: bool = False,
        output_bound_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> LastVLAOutput:
        if self.config.condition_mode != "decoupled_cot_residual" or self.config.cot_bottleneck_mode:
            raise ValueError(
                "Last-VLA v2 supports only decoupled_cot_residual. "
                "Hard bottleneck and summary replacement paths have been removed."
            )
        batch_size = vlm_tokens.shape[0]
        if not training and action_input is not None:
            present_targets = [key for key in TARGET_KEYS if key in action_input]
            if present_targets:
                warnings.warn(
                    f"Last-VLA eval received train-only target keys {present_targets}; they are ignored.",
                    RuntimeWarning,
                )
        ego_status = safe_get(action_input, "status_feature")
        if ego_status is None:
            ego_status = vlm_tokens.new_zeros(batch_size, 8)
        else:
            ego_status = ego_status.to(vlm_tokens)
        command = safe_get(action_input, "high_command_one_hot")
        command = command.to(vlm_tokens) if isinstance(command, torch.Tensor) else None
        history = safe_get(action_input, "his_traj", safe_get(action_input, "history_trajectory"))
        if isinstance(history, torch.Tensor):
            history = history.to(vlm_tokens).view(batch_size, -1)
        else:
            history = None

        raw_vlm_tokens = vlm_tokens
        cot = self.cot_queries.unsqueeze(0).expand(batch_size, -1, -1).to(vlm_tokens)
        state_token = self._state_token(ego_status, command, history)
        cot = cot + state_token
        cot_steps: Dict[str, torch.Tensor] = {}

        t_embed = None
        if diffusion_timestep is not None:
            t_embed = sinusoidal_timestep_embedding(diffusion_timestep, self.config.planner_dim).to(vlm_tokens)
        corrupt_zero_all = _bool_flag(action_input, "last_vla_corrupt_zero_all_cot")
        corrupt_scene = corrupt_zero_all or _bool_flag(action_input, "last_vla_corrupt_zero_scene_cot")
        corrupt_geometry = corrupt_zero_all or _bool_flag(action_input, "last_vla_corrupt_zero_geometry_cot") or _bool_flag(action_input, "last_vla_corrupt_geometry_cot")
        corrupt_dynamic = corrupt_zero_all or _bool_flag(action_input, "last_vla_corrupt_zero_dynamic_cot") or _bool_flag(action_input, "last_vla_corrupt_dynamic_cot")
        corrupt_fusion = corrupt_zero_all or _bool_flag(action_input, "last_vla_corrupt_zero_fusion_cot")
        corrupt_ego = corrupt_zero_all or _bool_flag(action_input, "last_vla_corrupt_zero_ego_cot") or _bool_flag(action_input, "last_vla_corrupt_ego_cot")
        corrupt_action_refine = corrupt_zero_all or _bool_flag(action_input, "last_vla_corrupt_zero_action_refine_cot") or _bool_flag(action_input, "last_vla_corrupt_action_refine_cot")
        corrupt_coarse_prior = corrupt_zero_all or _bool_flag(action_input, "last_vla_corrupt_zero_coarse_prior") or _bool_flag(action_input, "last_vla_zero_coarse_prior")
        corrupt_cot_condition = corrupt_zero_all or _bool_flag(action_input, "last_vla_corrupt_zero_cot_condition_branch") or _bool_flag(action_input, "last_vla_raw_vlm_only")
        cot_only_for_debug = _bool_flag(action_input, "last_vla_cot_only_for_debug_only")

        cot_scene = self.scene_step(cot, raw_vlm_tokens, timestep_embedding=t_embed) if self.config.use_scene_step else cot
        if corrupt_scene:
            cot_scene = torch.zeros_like(cot_scene)
        cot_steps["scene"] = cot_scene

        geometry_memory = self._geometry_memory(action_input, vlm_tokens)
        geometry_memory_parts = [raw_vlm_tokens]
        if geometry_memory is not None:
            geometry_memory_parts.append(geometry_memory)
        geometry_memory_for_step = torch.cat(geometry_memory_parts, dim=1)
        cot_geometry = self.geometry_step(cot_scene, geometry_memory_for_step, timestep_embedding=t_embed) if self.config.use_geometry_step else cot_scene
        if corrupt_geometry:
            cot_geometry = torch.zeros_like(cot_geometry)
        predicted_geometry, geometry_target, geometry_loss, geometry_diag = self.geometry_head(
            cot_geometry,
            action_input,
            training=training,
            allow_target_tokens=allow_target_tokens,
        )
        cot_steps["geometry"] = cot_geometry

        coarse_0, _, _ = self.coarse_head(
            cot_scene,
            ego_status,
            command,
            history,
            target_action_norm,
            output_bound_fn=output_bound_fn,
        )
        jepa_context = self._context_tokens(action_input, "jepa_context_tokens", self.config.jepa_dim, vlm_tokens)
        if jepa_context is not None and self.config.dynamic_tokens > 12 and jepa_context.shape[1] != self.config.dynamic_tokens:
            raise ValueError(
                f"jepa_context_tokens token count {jepa_context.shape[1]} != expected {self.config.dynamic_tokens}."
            )
        dynamic_memory = [raw_vlm_tokens]
        if jepa_context is not None:
            dynamic_memory.append(self.jepa_context_proj(jepa_context))
        if predicted_geometry is not None and self.config.dynamic_uses_geometry_memory:
            dynamic_memory.append(self.geometry_pred_proj(predicted_geometry.to(vlm_tokens)))
        action_tokens = self.action_encoder(coarse_0, noisy_action_norm, t_embed, reference=vlm_tokens)
        cot_dynamic = self.dynamic_step(cot_scene, torch.cat(dynamic_memory, dim=1), action_tokens=action_tokens, timestep_embedding=t_embed) if self.config.use_dynamic_step else cot_scene
        if corrupt_dynamic:
            cot_dynamic = torch.zeros_like(cot_dynamic)
        jepa_target = safe_get(action_input, "jepa_target_tokens") if training and allow_target_tokens else None
        predicted_future_jepa, dynamic_target, dynamic_loss = self.dynamic_head(
            cot_dynamic,
            jepa_target,
            training=training,
            coarse_traj_norm=coarse_0,
            noisy_action_norm=noisy_action_norm,
            diffusion_timestep=diffusion_timestep,
            allow_target_tokens=allow_target_tokens,
        )
        cot_steps["dynamic"] = cot_dynamic

        fusion_memory = torch.cat([cot_scene, cot_geometry, cot_dynamic, raw_vlm_tokens], dim=1)
        cot_fusion = self.fusion_step(cot_scene, fusion_memory, timestep_embedding=t_embed) if self.config.use_parallel_geometry_dynamic else cot_dynamic
        if corrupt_fusion:
            cot_fusion = torch.zeros_like(cot_fusion)
        cot_steps["fusion"] = cot_fusion

        ego_memory = torch.cat([raw_vlm_tokens, state_token, cot_fusion], dim=1)
        ego_action_tokens = self.action_encoder(coarse_0, None, None, reference=vlm_tokens)
        cot_ego = self.ego_step(cot_fusion, ego_memory, action_tokens=ego_action_tokens, timestep_embedding=t_embed) if self.config.use_ego_step else cot_fusion
        if corrupt_ego:
            cot_ego = torch.zeros_like(cot_ego)
        coarse_traj_norm, ego_tokens, coarse_losses = self.coarse_head(
            cot_fusion,
            ego_status,
            command,
            history,
            target_action_norm,
            output_bound_fn=output_bound_fn,
        )
        if corrupt_coarse_prior:
            coarse_traj_norm = torch.zeros_like(coarse_traj_norm)
            ego_tokens = torch.zeros_like(ego_tokens)
        cot_steps["ego"] = cot_ego

        refine_memory = torch.cat([raw_vlm_tokens, cot_geometry, cot_dynamic, ego_tokens], dim=1)
        refine_action_tokens = self.action_encoder(coarse_traj_norm, noisy_action_norm, t_embed, reference=vlm_tokens)
        cot_action = self.action_refine_step(cot_ego, refine_memory, action_tokens=refine_action_tokens, timestep_embedding=t_embed) if self.config.use_action_refine_step else cot_ego
        if corrupt_action_refine:
            cot_action = torch.zeros_like(cot_action)
        cot_steps["action_refine"] = cot_action
        cot_condition_tokens = torch.zeros_like(cot_action) if corrupt_cot_condition else cot_action
        cot_final = cot_action

        if self.config.use_cot_risk_head and self.risk_head is not None:
            risk_tokens, risk_logits, risk_loss = self.risk_head(cot_final, coarse_traj_norm, action_input)
            cot_final = torch.cat([cot_final, risk_tokens], dim=1)
            if corrupt_zero_all:
                cot_final = torch.zeros_like(cot_final)
        else:
            risk_logits = None
            risk_loss = cot_final.new_zeros(())

        base_context_tokens = raw_vlm_tokens
        if cot_only_for_debug:
            base_context_tokens = torch.zeros_like(base_context_tokens)
        planner_context = base_context_tokens
        raw_vlm_used = not cot_only_for_debug
        context_mode = "decoupled_cot_residual"

        horizon_queries = self.horizon_queries.unsqueeze(0).expand(batch_size, -1, -1).to(vlm_tokens)
        horizon_condition = self.horizon_attn(horizon_queries, planner_context)
        final_target = target_action_norm.detach().to(coarse_traj_norm) if target_action_norm is not None else None
        residual_target = None
        if final_target is not None and self.config.use_residual_diffusion:
            base = coarse_traj_norm.detach() if self.config.residual_detach_coarse_for_diffusion else coarse_traj_norm
            residual_target = final_target - base

        cot_consistency = smooth_l1(cot_condition_tokens[:, : cot_ego.shape[1]], cot_ego.detach())
        losses = {
            "geometry_loss": geometry_loss,
            "dynamic_loss": dynamic_loss,
            "coarse_loss": coarse_losses["coarse_loss"],
            "heading_loss": coarse_losses["heading_loss"],
            "progress_loss": coarse_losses["progress_loss"],
            "risk_loss": risk_loss,
            "cot_consistency_loss": cot_consistency,
        }
        if training and allow_target_tokens and self.config.use_dynamic_step and jepa_target is None and not self.config.allow_missing_dynamic_teacher:
            # The planner decides whether this loss is required by assigning a nonzero weight.
            diagnostics_dynamic_missing = cot_final.new_tensor(1.0)
        else:
            diagnostics_dynamic_missing = cot_final.new_tensor(0.0)

        diagnostics = {
            "cot_token_norm": cot_final.detach().float().norm(dim=-1).mean(),
            "geometry_loss_raw": geometry_loss.detach().float(),
            "dynamic_loss_raw": dynamic_loss.detach().float(),
            "coarse_traj_l1": smooth_l1(coarse_traj_norm, final_target).detach().float() if final_target is not None else cot_final.new_zeros(()),
            "geometry_mode_code": geometry_diag["geometry_mode_code"].to(cot_final),
            "context_mode_code": cot_final.new_tensor(float(CONTEXT_MODE_TO_CODE[context_mode])),
            "last_vla_condition_mode_code": cot_final.new_tensor(float(CONTEXT_MODE_TO_CODE[context_mode])),
            "raw_vlm_context_used": cot_final.new_tensor(float(raw_vlm_used)),
            "cot_bottleneck_active": cot_final.new_tensor(0.0),
            "last_vla_no_global_gate": cot_final.new_tensor(1.0),
            "teacher_traj_used_ratio": cot_final.new_zeros(()),
            "dynamic_teacher_missing": diagnostics_dynamic_missing,
            "geometry_teacher_missing": cot_final.new_tensor(float(geometry_target is None)),
            "corruption_zero_all_cot": cot_final.new_tensor(float(corrupt_zero_all)),
            "corruption_zero_coarse_prior": cot_final.new_tensor(float(corrupt_coarse_prior)),
            "last_vla_use_risk_head": cot_final.new_tensor(float(self.config.use_cot_risk_head and self.risk_head is not None)),
            "last_vla_num_risk_tokens": cot_final.new_tensor(float(self.config.risk_tokens if self.config.use_cot_risk_head else 0)),
            "last_vla_context_token_count": cot_final.new_tensor(float(planner_context.shape[1])),
            "last_vla_cot_token_count": cot_final.new_tensor(float(cot_final.shape[1])),
            "raw_vlm_token_count": cot_final.new_tensor(float(raw_vlm_tokens.shape[1])),
            "cot_condition_token_count": cot_final.new_tensor(float(cot_condition_tokens.shape[1])),
            "cot_scene_norm": cot_scene.detach().float().norm(dim=-1).mean(),
            "cot_geometry_norm": cot_geometry.detach().float().norm(dim=-1).mean(),
            "cot_dynamic_norm": cot_dynamic.detach().float().norm(dim=-1).mean(),
            "cot_fusion_norm": cot_fusion.detach().float().norm(dim=-1).mean(),
            "cot_action_norm": cot_action.detach().float().norm(dim=-1).mean(),
            "geometry_dynamic_parallel": cot_final.new_tensor(float(self.config.use_parallel_geometry_dynamic)),
        }
        finite_or_raise("planner_context_tokens", planner_context)
        return LastVLAOutput(
            cot_tokens=cot_final,
            cot_tokens_by_step=cot_steps,
            raw_vlm_context_tokens=raw_vlm_tokens,
            cot_condition_tokens=cot_condition_tokens,
            cot_scene_tokens=cot_scene,
            cot_geometry_tokens=cot_geometry,
            cot_dynamic_tokens=cot_dynamic,
            cot_fusion_tokens=cot_fusion,
            base_context_tokens=base_context_tokens,
            planner_context_tokens=planner_context,
            context_mean=planner_context.mean(dim=1),
            horizon_condition=horizon_condition,
            coarse_traj_norm=coarse_traj_norm,
            residual_target_norm=residual_target,
            final_target_norm=final_target,
            predicted_future_jepa=predicted_future_jepa,
            predicted_geometry=predicted_geometry,
            risk_logits=risk_logits,
            losses=losses,
            diagnostics=diagnostics,
        )
