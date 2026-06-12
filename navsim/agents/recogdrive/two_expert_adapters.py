from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


def _num_heads(dim: int) -> int:
    for heads in (8, 6, 4, 3, 2, 1):
        if dim % heads == 0:
            return heads
    return 1


class RandomTokenMask(nn.Module):
    def __init__(
        self,
        mask_ratio: float = 0.3,
        use_learned_mask_token: bool = True,
        token_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= float(mask_ratio) <= 1.0:
            raise ValueError("mask_ratio must be in [0, 1].")
        self.mask_ratio = float(mask_ratio)
        self.use_learned_mask_token = bool(use_learned_mask_token)
        if use_learned_mask_token and token_dim is not None:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, int(token_dim)))
        else:
            self.register_parameter("mask_token", None)

    def _mask_token(self, dim: int, ref: torch.Tensor) -> torch.Tensor:
        if not self.use_learned_mask_token:
            return torch.zeros(1, 1, dim, device=ref.device, dtype=ref.dtype)
        if self.mask_token is None:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, dim, device=ref.device, dtype=ref.dtype))
        if self.mask_token.shape[-1] != dim:
            raise ValueError(f"mask token dim {self.mask_token.shape[-1]} does not match input dim {dim}.")
        return self.mask_token.to(device=ref.device, dtype=ref.dtype)

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
        if length > 0 and self.mask_ratio > 0:
            empty = ~mask.any(dim=1)
            if empty.any():
                forced = torch.randint(0, length, (int(empty.sum().item()),), device=tokens.device)
                mask[empty] = False
                mask[empty, forced] = True
        replacement = self._mask_token(dim, tokens).expand(batch, length, dim)
        return torch.where(mask.unsqueeze(-1), replacement, tokens), {
            "mask": mask,
            "actual_mask_ratio": mask.float().mean().to(dtype=tokens.dtype),
        }


class NormalizedMSELoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.detach().to(device=pred.device, dtype=pred.dtype)
        pred_norm = F.normalize(pred.float(), p=2, dim=-1, eps=1e-6)
        target_norm = F.normalize(target.float(), p=2, dim=-1, eps=1e-6)
        return F.mse_loss(pred_norm, target_norm)


def normalized_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return NormalizedMSELoss()(pred, target)


class _DecoderBlock(nn.Module):
    def __init__(self, adapter_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(adapter_dim)
        self.mlp = nn.Sequential(
            nn.Linear(adapter_dim, adapter_dim * 2),
            nn.GELU(),
            nn.Linear(adapter_dim * 2, adapter_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


class _CrossAttentionDecoder(nn.Module):
    def __init__(self, query_tokens: int, adapter_dim: int, output_dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(query_tokens, adapter_dim))
        self.query_norm = nn.LayerNorm(adapter_dim)
        self.memory_norm = nn.LayerNorm(adapter_dim)
        self.attn = nn.MultiheadAttention(adapter_dim, _num_heads(adapter_dim), batch_first=True)
        self.block = _DecoderBlock(adapter_dim)
        self.out = nn.Sequential(
            nn.LayerNorm(adapter_dim),
            nn.Linear(adapter_dim, adapter_dim * 2),
            nn.GELU(),
            nn.Linear(adapter_dim * 2, output_dim),
        )
        nn.init.normal_(self.query, mean=0.0, std=0.02)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        batch = memory.shape[0]
        query = self.query.unsqueeze(0).expand(batch, -1, -1).to(memory)
        decoded, _ = self.attn(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        decoded = self.block(query + decoded)
        return self.out(decoded)


class JEPADynamicAdapter(nn.Module):
    def __init__(
        self,
        vlm_hidden_dim: int = 1536,
        adapter_dim: int = 384,
        teacher_dim: int = 1024,
        num_dyn_groups: int = 3,
        tokens_per_group: int = 12,
        mask_ratio: float = 0.3,
        loss_type: str = "normalized_mse",
    ) -> None:
        super().__init__()
        if loss_type not in {"normalized_mse", "cosine"}:
            raise ValueError("loss_type must be 'normalized_mse' or 'cosine'.")
        self.num_dyn_groups = int(num_dyn_groups)
        self.tokens_per_group = int(tokens_per_group)
        self.loss_type = loss_type
        self.mask = RandomTokenMask(mask_ratio=mask_ratio, token_dim=vlm_hidden_dim)
        self.image_proj = nn.Sequential(nn.LayerNorm(vlm_hidden_dim), nn.Linear(vlm_hidden_dim, adapter_dim))
        self.h_dyn_proj = nn.Sequential(nn.LayerNorm(vlm_hidden_dim), nn.Linear(vlm_hidden_dim, adapter_dim))
        self.group_decoders = nn.ModuleList(
            [_CrossAttentionDecoder(tokens_per_group, adapter_dim, teacher_dim) for _ in range(num_dyn_groups)]
        )

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "normalized_mse":
            return normalized_mse_loss(pred, target)
        return (1.0 - F.cosine_similarity(pred.float(), target.detach().to(pred).float(), dim=-1)).mean()

    def forward(self, image_hidden: torch.Tensor, h_dyn: torch.Tensor, target: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        if image_hidden.ndim != 3:
            raise ValueError("image_hidden must have shape [B, N_img, D_vlm].")
        if h_dyn.ndim != 4:
            raise ValueError("h_dyn must have shape [B, 3, 12, D_vlm].")
        if h_dyn.shape[1] != self.num_dyn_groups or h_dyn.shape[2] != self.tokens_per_group:
            raise ValueError(
                f"h_dyn shape {tuple(h_dyn.shape)} does not match [B,{self.num_dyn_groups},{self.tokens_per_group},D]."
            )
        masked_image, mask_info = self.mask(image_hidden, training=self.training)
        image_tokens = self.image_proj(masked_image)
        preds = []
        group_losses = []
        for group_idx, decoder in enumerate(self.group_decoders):
            slot_tokens = self.h_dyn_proj(h_dyn[:, group_idx].to(image_hidden))
            memory = torch.cat([image_tokens, slot_tokens], dim=1)
            pred_g = decoder(memory)
            preds.append(pred_g)
            if target is not None:
                group_losses.append(self._loss(pred_g, target[:, group_idx].to(pred_g)))
        pred = torch.stack(preds, dim=1)
        zero = pred.new_zeros(())
        if target is not None:
            target = target.to(device=pred.device, dtype=pred.dtype)
            if target.shape != pred.shape:
                raise ValueError(f"target shape {tuple(target.shape)} does not match pred {tuple(pred.shape)}.")
            loss = self._loss(pred, target).to(dtype=pred.dtype)
        else:
            loss = zero
        names = ("short", "mid", "long")
        diagnostics = {
            "dynamic_pred_norm": pred.detach().float().norm(dim=-1).mean().to(dtype=pred.dtype),
            "dynamic_target_norm": target.detach().float().norm(dim=-1).mean().to(dtype=pred.dtype) if target is not None else zero,
            "dynamic_mask_ratio": mask_info["actual_mask_ratio"].detach(),
        }
        for idx, item in enumerate(group_losses):
            diagnostics[f"dynamic_loss_{names[idx] if idx < len(names) else idx}"] = item.detach().to(dtype=pred.dtype)
        return {"pred": pred, "loss": loss, "diagnostics": diagnostics}


class VGGTFeature23Adapter(nn.Module):
    def __init__(
        self,
        vlm_hidden_dim: int = 1536,
        adapter_dim: int = 384,
        vggt_feature_dim: int = 512,
        num_geo_tokens: int = 12,
        mask_ratio: float = 0.3,
        loss_type: str = "normalized_mse",
    ) -> None:
        super().__init__()
        if loss_type not in {"normalized_mse", "cosine"}:
            raise ValueError("loss_type must be 'normalized_mse' or 'cosine'.")
        self.loss_type = loss_type
        self.num_geo_tokens = int(num_geo_tokens)
        self.mask = RandomTokenMask(mask_ratio=mask_ratio, token_dim=vlm_hidden_dim)
        self.image_proj = nn.Sequential(nn.LayerNorm(vlm_hidden_dim), nn.Linear(vlm_hidden_dim, adapter_dim))
        self.h_geo_proj = nn.Sequential(nn.LayerNorm(vlm_hidden_dim), nn.Linear(vlm_hidden_dim, adapter_dim))
        self.decoder = _CrossAttentionDecoder(num_geo_tokens, adapter_dim, vggt_feature_dim)

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "normalized_mse":
            return normalized_mse_loss(pred, target)
        return (1.0 - F.cosine_similarity(pred.float(), target.detach().to(pred).float(), dim=-1)).mean()

    def forward(self, image_hidden: torch.Tensor, h_geo: torch.Tensor, target: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        if image_hidden.ndim != 3 or h_geo.ndim != 3:
            raise ValueError("image_hidden and h_geo must have shapes [B,N,D] and [B,12,D].")
        if h_geo.shape[1] != self.num_geo_tokens:
            raise ValueError(f"h_geo token count {h_geo.shape[1]} does not match {self.num_geo_tokens}.")
        masked_image, mask_info = self.mask(image_hidden, training=self.training)
        memory = torch.cat([self.image_proj(masked_image), self.h_geo_proj(h_geo.to(image_hidden))], dim=1)
        pred = self.decoder(memory)
        zero = pred.new_zeros(())
        if target is not None:
            target = target.to(device=pred.device, dtype=pred.dtype)
            if target.shape != pred.shape:
                raise ValueError(f"target shape {tuple(target.shape)} does not match pred {tuple(pred.shape)}.")
            loss = self._loss(pred, target).to(dtype=pred.dtype)
        else:
            loss = zero
        return {
            "pred": pred,
            "loss": loss,
            "diagnostics": {
                "geometry_pred_norm": pred.detach().float().norm(dim=-1).mean().to(dtype=pred.dtype),
                "geometry_target_norm": target.detach().float().norm(dim=-1).mean().to(dtype=pred.dtype) if target is not None else zero,
                "geometry_mask_ratio": mask_info["actual_mask_ratio"].detach(),
            },
        }


class TwoExpertTrajectoryProbe(nn.Module):
    def __init__(
        self,
        vlm_hidden_dim: int = 1536,
        adapter_dim: int = 384,
        action_horizon: int = 8,
        action_dim: int = 3,
    ) -> None:
        super().__init__()
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.dyn_pool = nn.Sequential(nn.LayerNorm(vlm_hidden_dim), nn.Linear(vlm_hidden_dim, adapter_dim), nn.GELU())
        self.geo_pool = nn.Sequential(nn.LayerNorm(vlm_hidden_dim), nn.Linear(vlm_hidden_dim, adapter_dim), nn.GELU())
        fused_dim = adapter_dim * 2 + 8 + 3 + 12
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, adapter_dim * 2),
            nn.GELU(),
            nn.Linear(adapter_dim * 2, action_horizon * action_dim),
        )

    @staticmethod
    def _vec(value: Optional[torch.Tensor], batch: int, width: int, ref: torch.Tensor) -> torch.Tensor:
        if value is None:
            return ref.new_zeros(batch, width)
        tensor = value.to(device=ref.device, dtype=ref.dtype).reshape(batch, -1)
        if tensor.shape[1] == width:
            return tensor
        if tensor.shape[1] > width:
            return tensor[:, :width]
        return F.pad(tensor, (0, width - tensor.shape[1]))

    def forward(
        self,
        h_dyn: torch.Tensor,
        h_geo: torch.Tensor,
        *,
        status_feature: Optional[torch.Tensor] = None,
        high_command_one_hot: Optional[torch.Tensor] = None,
        history_trajectory: Optional[torch.Tensor] = None,
        target_action_norm: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if h_dyn.ndim != 4 or h_geo.ndim != 3:
            raise ValueError("h_dyn must be [B,G,T,D] and h_geo must be [B,T,D].")
        batch = h_dyn.shape[0]
        dyn_summary = self.dyn_pool(h_dyn.reshape(batch, -1, h_dyn.shape[-1])).mean(dim=1)
        geo_summary = self.geo_pool(h_geo).mean(dim=1)
        status = self._vec(status_feature, batch, 8, dyn_summary)
        command = self._vec(high_command_one_hot, batch, 3, dyn_summary)
        history = history_trajectory.reshape(batch, -1) if history_trajectory is not None else None
        history_vec = self._vec(history, batch, 12, dyn_summary)
        pred = self.head(torch.cat([dyn_summary, geo_summary, status, command, history_vec], dim=-1)).view(
            batch,
            self.action_horizon,
            self.action_dim,
        )
        zero = pred.new_zeros(())
        losses = {"probe_loss": zero, "probe_heading_loss": zero, "probe_progress_loss": zero}
        if target_action_norm is not None:
            target = target_action_norm.to(device=pred.device, dtype=pred.dtype)
            if target.shape != pred.shape:
                raise ValueError(f"target_action_norm shape {tuple(target.shape)} does not match pred {tuple(pred.shape)}.")
            losses["probe_loss"] = F.smooth_l1_loss(pred.float(), target.detach().float()).to(dtype=pred.dtype)
            losses["probe_heading_loss"] = F.smooth_l1_loss(pred[..., 2].float(), target.detach()[..., 2].float()).to(dtype=pred.dtype)
            losses["probe_progress_loss"] = F.smooth_l1_loss(pred[..., 0].float(), target.detach()[..., 0].float()).to(dtype=pred.dtype)
        return {
            "probe_traj_norm": pred,
            "losses": losses,
            "diagnostics": {"probe_traj_norm": pred.detach().float().norm(dim=-1).mean().to(dtype=pred.dtype)},
        }
