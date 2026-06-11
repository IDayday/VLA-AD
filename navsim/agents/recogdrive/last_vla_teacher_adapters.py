from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


def normalized_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.detach().to(device=pred.device, dtype=pred.dtype)
    pred_norm = F.normalize(pred.float(), p=2, dim=-1, eps=1e-6)
    target_norm = F.normalize(target.float(), p=2, dim=-1, eps=1e-6)
    return F.mse_loss(pred_norm, target_norm)


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.detach().to(device=pred.device, dtype=pred.dtype)
    return (1.0 - F.cosine_similarity(pred.float(), target.float(), dim=-1)).mean()


def _num_heads(dim: int) -> int:
    for heads in (8, 6, 4, 3, 2, 1):
        if dim % heads == 0:
            return heads
    return 1


class RandomTokenMask(nn.Module):
    def __init__(
        self,
        mask_ratio: float,
        use_learned_mask_token: bool = True,
        token_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= float(mask_ratio) <= 1.0:
            raise ValueError("mask_ratio must be in [0, 1].")
        self.mask_ratio = float(mask_ratio)
        self.use_learned_mask_token = bool(use_learned_mask_token)
        self._token_dim = int(token_dim) if token_dim is not None else None
        if self.use_learned_mask_token and token_dim is not None:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, int(token_dim)))
        else:
            self.register_parameter("mask_token", None)

    def _ensure_mask_token(self, token_dim: int, reference: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.use_learned_mask_token:
            return None
        if self.mask_token is None:
            self._token_dim = int(token_dim)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, int(token_dim), device=reference.device, dtype=reference.dtype))
        if self.mask_token.shape[-1] != token_dim:
            raise ValueError(f"mask_token dim {self.mask_token.shape[-1]} does not match tokens dim {token_dim}.")
        return self.mask_token.to(device=reference.device, dtype=reference.dtype)

    def forward(self, tokens: torch.Tensor, training: Optional[bool] = None) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must have shape [B, N, D], got {tuple(tokens.shape)}.")
        active = self.training if training is None else bool(training)
        if not active or self.mask_ratio <= 0.0:
            return tokens, {
                "mask": torch.zeros(tokens.shape[:2], device=tokens.device, dtype=torch.bool),
                "actual_mask_ratio": tokens.new_zeros(()),
            }

        batch, length, dim = tokens.shape
        mask = torch.rand(batch, length, device=tokens.device) < self.mask_ratio
        if self.mask_ratio > 0.0:
            empty = ~mask.any(dim=1)
            if empty.any():
                forced_index = torch.randint(0, length, (int(empty.sum().item()),), device=tokens.device)
                mask[empty] = False
                mask[empty, forced_index] = True
        if self.use_learned_mask_token:
            replacement = self._ensure_mask_token(dim, tokens).expand(batch, length, dim)
        else:
            replacement = torch.zeros_like(tokens)
        masked_tokens = torch.where(mask.unsqueeze(-1), replacement, tokens)
        return masked_tokens, {
            "mask": mask,
            "actual_mask_ratio": mask.float().mean().to(dtype=tokens.dtype),
        }


class _CrossAttentionDecoder(nn.Module):
    def __init__(self, query_tokens: int, planner_dim: int, output_dim: int) -> None:
        super().__init__()
        self.query_embeddings = nn.Parameter(torch.empty(query_tokens, planner_dim))
        self.query_norm = nn.LayerNorm(planner_dim)
        self.memory_norm = nn.LayerNorm(planner_dim)
        self.attn = nn.MultiheadAttention(planner_dim, _num_heads(planner_dim), batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(planner_dim),
            nn.Linear(planner_dim, planner_dim * 2),
            nn.GELU(),
            nn.Linear(planner_dim * 2, output_dim),
        )
        nn.init.normal_(self.query_embeddings, mean=0.0, std=0.02)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        batch = memory.shape[0]
        query = self.query_embeddings.unsqueeze(0).expand(batch, -1, -1).to(memory)
        decoded, _ = self.attn(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        return self.mlp(query + decoded)


class DynamicsAdapter(nn.Module):
    def __init__(
        self,
        vlm_hidden_dim: int = 1536,
        planner_dim: int = 384,
        jepa_dim: int = 1024,
        num_teacher_tokens: int = 128,
        mask_ratio: float = 0.30,
        teacher_source: str = "jepa_dev",
        loss_type: str = "normalized_mse",
        strict_teacher: bool = True,
    ) -> None:
        super().__init__()
        if teacher_source not in {"cosmos", "jepa_dev"}:
            raise ValueError("teacher_source must be 'cosmos' or 'jepa_dev'.")
        if loss_type not in {"normalized_mse", "mse", "cosine"}:
            raise ValueError("loss_type must be 'normalized_mse', 'mse', or 'cosine'.")
        self.teacher_source = teacher_source
        self.loss_type = loss_type
        self.strict_teacher = bool(strict_teacher)
        self.image_mask = RandomTokenMask(mask_ratio, use_learned_mask_token=True, token_dim=vlm_hidden_dim)
        self.image_proj = nn.Sequential(nn.LayerNorm(vlm_hidden_dim), nn.Linear(vlm_hidden_dim, planner_dim))
        self.h_dyn_proj = nn.Sequential(nn.LayerNorm(vlm_hidden_dim), nn.Linear(vlm_hidden_dim, planner_dim))
        self.decoder = _CrossAttentionDecoder(num_teacher_tokens, planner_dim, jepa_dim)

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "normalized_mse":
            return normalized_mse_loss(pred, target)
        if self.loss_type == "mse":
            return F.mse_loss(pred.float(), target.detach().to(pred).float())
        if self.loss_type == "cosine":
            return cosine_loss(pred, target)
        raise ValueError(f"Unsupported loss_type={self.loss_type!r}.")

    def _select_target(
        self,
        jepa_target_tokens: Optional[torch.Tensor],
        cosmos_future_features: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        missing = reference.new_tensor(0.0)
        if self.teacher_source == "cosmos":
            if cosmos_future_features is None:
                missing = reference.new_tensor(1.0)
                if self.strict_teacher:
                    raise KeyError("teacher_source='cosmos' requires cosmos_future_features in strict mode.")
                return None, missing
            return cosmos_future_features.to(device=reference.device, dtype=reference.dtype), missing
        if jepa_target_tokens is None:
            missing = reference.new_tensor(1.0)
            if self.strict_teacher:
                raise KeyError("teacher_source='jepa_dev' requires jepa_target_tokens in strict mode.")
            return None, missing
        return jepa_target_tokens.to(device=reference.device, dtype=reference.dtype), missing

    def forward(
        self,
        image_hidden_states: torch.Tensor,
        h_dyn: torch.Tensor,
        *,
        jepa_target_tokens: Optional[torch.Tensor] = None,
        cosmos_future_features: Optional[torch.Tensor] = None,
        require_teacher: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if image_hidden_states.ndim != 3 or h_dyn.ndim != 3:
            raise ValueError("image_hidden_states and h_dyn must both have shape [B, K, D].")
        old_strict = self.strict_teacher
        if require_teacher is not None:
            self.strict_teacher = bool(require_teacher)
        try:
            masked_img, mask_info = self.image_mask(image_hidden_states, training=self.training)
            adapter_tokens = torch.cat([self.image_proj(masked_img), self.h_dyn_proj(h_dyn)], dim=1)
            pred = self.decoder(adapter_tokens)
            target, missing = self._select_target(jepa_target_tokens, cosmos_future_features, pred)
        finally:
            self.strict_teacher = old_strict
        if target is None:
            loss = pred.new_zeros(())
        else:
            if target.shape != pred.shape:
                raise ValueError(f"dynamics target shape {tuple(target.shape)} does not match prediction {tuple(pred.shape)}.")
            loss = self._loss(pred, target).to(dtype=pred.dtype)
        return {
            "pred": pred,
            "loss": loss,
            "diagnostics": {
                "wm_mask_ratio": mask_info["actual_mask_ratio"].detach(),
                "wm_teacher_missing": missing.detach(),
            },
        }


class GeometryAdapter(nn.Module):
    def __init__(
        self,
        vlm_hidden_dim: int = 1536,
        planner_dim: int = 384,
        geometry_dim: int = 512,
        num_geometry_tokens: int = 192,
        mask_ratio: float = 0.30,
        loss_type: str = "normalized_mse",
        strict_teacher: bool = True,
    ) -> None:
        super().__init__()
        if loss_type not in {"normalized_mse", "mse", "cosine"}:
            raise ValueError("loss_type must be 'normalized_mse', 'mse', or 'cosine'.")
        self.loss_type = loss_type
        self.strict_teacher = bool(strict_teacher)
        self.image_mask = RandomTokenMask(mask_ratio, use_learned_mask_token=True, token_dim=vlm_hidden_dim)
        self.image_proj = nn.Sequential(nn.LayerNorm(vlm_hidden_dim), nn.Linear(vlm_hidden_dim, planner_dim))
        self.h_geo_proj = nn.Sequential(nn.LayerNorm(vlm_hidden_dim), nn.Linear(vlm_hidden_dim, planner_dim))
        self.decoder = _CrossAttentionDecoder(num_geometry_tokens, planner_dim, geometry_dim)

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "normalized_mse":
            return normalized_mse_loss(pred, target)
        if self.loss_type == "mse":
            return F.mse_loss(pred.float(), target.detach().to(pred).float())
        if self.loss_type == "cosine":
            return cosine_loss(pred, target)
        raise ValueError(f"Unsupported loss_type={self.loss_type!r}.")

    def forward(
        self,
        image_hidden_states: torch.Tensor,
        h_geo: torch.Tensor,
        *,
        vggt_geometry_tokens: Optional[torch.Tensor] = None,
        require_teacher: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if image_hidden_states.ndim != 3 or h_geo.ndim != 3:
            raise ValueError("image_hidden_states and h_geo must both have shape [B, K, D].")
        strict_teacher = self.strict_teacher if require_teacher is None else bool(require_teacher)
        masked_img, mask_info = self.image_mask(image_hidden_states, training=self.training)
        adapter_tokens = torch.cat([self.image_proj(masked_img), self.h_geo_proj(h_geo)], dim=1)
        pred = self.decoder(adapter_tokens)
        missing = pred.new_tensor(0.0)
        if vggt_geometry_tokens is None:
            missing = pred.new_tensor(1.0)
            if strict_teacher:
                raise KeyError("GeometryAdapter requires vggt_geometry_tokens in strict mode.")
            loss = pred.new_zeros(())
        else:
            target = vggt_geometry_tokens.to(device=pred.device, dtype=pred.dtype)
            if target.shape != pred.shape:
                raise ValueError(f"geometry target shape {tuple(target.shape)} does not match prediction {tuple(pred.shape)}.")
            loss = self._loss(pred, target).to(dtype=pred.dtype)
        return {
            "pred": pred,
            "loss": loss,
            "diagnostics": {
                "geometry_mask_ratio": mask_info["actual_mask_ratio"].detach(),
                "geometry_teacher_missing": missing.detach(),
            },
        }


class PlanLatentHead(nn.Module):
    def __init__(
        self,
        vlm_hidden_dim: int = 1536,
        planner_dim: int = 384,
        action_horizon: int = 8,
        action_dim: int = 3,
    ) -> None:
        super().__init__()
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        fused_dim = planner_dim * 3 + 8 + 3 + 12
        self.h_plan_proj = nn.Sequential(nn.LayerNorm(vlm_hidden_dim), nn.Linear(vlm_hidden_dim, planner_dim), nn.GELU())
        self.h_dyn_pool = nn.Sequential(nn.LayerNorm(vlm_hidden_dim), nn.Linear(vlm_hidden_dim, planner_dim), nn.GELU())
        self.h_geo_pool = nn.Sequential(nn.LayerNorm(vlm_hidden_dim), nn.Linear(vlm_hidden_dim, planner_dim), nn.GELU())
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, planner_dim * 2),
            nn.GELU(),
            nn.Linear(planner_dim * 2, action_horizon * action_dim),
        )

    @staticmethod
    def _optional_vector(value: Optional[torch.Tensor], batch: int, width: int, ref: torch.Tensor) -> torch.Tensor:
        if value is None:
            return ref.new_zeros(batch, width)
        tensor = value.to(device=ref.device, dtype=ref.dtype)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] != batch:
            raise ValueError(f"optional vector batch {tensor.shape[0]} does not match {batch}.")
        tensor = tensor.reshape(batch, -1)
        if tensor.shape[1] == width:
            return tensor
        if tensor.shape[1] > width:
            return tensor[:, :width]
        return F.pad(tensor, (0, width - tensor.shape[1]))

    def forward(
        self,
        h_plan: torch.Tensor,
        *,
        h_dyn: Optional[torch.Tensor] = None,
        h_geo: Optional[torch.Tensor] = None,
        target_action_norm: Optional[torch.Tensor] = None,
        status_feature: Optional[torch.Tensor] = None,
        high_command_one_hot: Optional[torch.Tensor] = None,
        history_trajectory: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if h_plan.ndim != 3:
            raise ValueError(f"h_plan must have shape [B, K, D], got {tuple(h_plan.shape)}.")
        batch = h_plan.shape[0]
        plan_summary = self.h_plan_proj(h_plan).mean(dim=1)
        if h_dyn is None:
            dyn_summary = plan_summary.new_zeros(batch, plan_summary.shape[-1])
        else:
            dyn_summary = self.h_dyn_pool(h_dyn.to(h_plan)).mean(dim=1)
        if h_geo is None:
            geo_summary = plan_summary.new_zeros(batch, plan_summary.shape[-1])
        else:
            geo_summary = self.h_geo_pool(h_geo.to(h_plan)).mean(dim=1)
        status = self._optional_vector(status_feature, batch, 8, plan_summary)
        command = self._optional_vector(high_command_one_hot, batch, 3, plan_summary)
        if history_trajectory is not None and history_trajectory.ndim >= 3:
            history = history_trajectory.reshape(batch, -1)
        else:
            history = history_trajectory
        history_vec = self._optional_vector(history, batch, 12, plan_summary)
        fused = torch.cat([plan_summary, dyn_summary, geo_summary, status, command, history_vec], dim=-1)
        coarse = self.head(fused).view(batch, self.action_horizon, self.action_dim)
        zero = coarse.new_zeros(())
        losses: Dict[str, torch.Tensor] = {
            "plan_loss": zero,
            "heading_loss": zero,
            "progress_loss": zero,
        }
        if target_action_norm is not None:
            target = target_action_norm.to(device=coarse.device, dtype=coarse.dtype)
            if target.shape != coarse.shape:
                raise ValueError(f"target_action_norm shape {tuple(target.shape)} does not match coarse {tuple(coarse.shape)}.")
            losses["plan_loss"] = F.smooth_l1_loss(coarse.float(), target.detach().float()).to(dtype=coarse.dtype)
            losses["heading_loss"] = F.smooth_l1_loss(coarse[..., 2].float(), target.detach()[..., 2].float()).to(dtype=coarse.dtype)
            losses["progress_loss"] = F.smooth_l1_loss(coarse[..., 0].float(), target.detach()[..., 0].float()).to(dtype=coarse.dtype)
        return {
            "coarse_traj_norm": coarse,
            "losses": losses,
            "diagnostics": {
                "plan_coarse_norm": coarse.detach().float().norm(dim=-1).mean().to(dtype=coarse.dtype),
            },
        }
