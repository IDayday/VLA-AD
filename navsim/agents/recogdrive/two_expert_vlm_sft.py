from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from .trajectory_normalization import norm_odo
from .two_expert_adapters import JEPADynamicAdapter, TwoExpertTrajectoryProbe, VGGTFeature23Adapter
from .two_expert_slots import TwoExpertSlotConfig, TwoExpertSoftSlots


@dataclass
class TwoExpertVLMSFTConfig:
    vlm_hidden_dim: int = 1536
    adapter_dim: int = 384
    vggt_feature_dim: int = 512
    action_horizon: int = 8
    action_dim: int = 3
    train_mode: str = "lora"
    allow_full_vlm_sft: bool = False
    top_layers: int = 2
    dyn_loss_weight: float = 1.0
    geo_loss_weight: float = 1.0
    probe_traj_loss_weight: float = 0.1
    probe_heading_loss_weight: float = 0.05
    probe_progress_loss_weight: float = 0.05
    hidden_anchor_weight: float = 0.05
    random_mask_ratio: float = 0.3
    loss_type: str = "normalized_mse"

    def __post_init__(self) -> None:
        if self.train_mode not in {"frozen", "lora", "top_layers", "full"}:
            raise ValueError("train_mode must be frozen, lora, top_layers, or full.")
        if self.train_mode == "full" and not self.allow_full_vlm_sft:
            raise ValueError("Full VLM SFT requires allow_full_vlm_sft=True.")
        for name in (
            "dyn_loss_weight",
            "geo_loss_weight",
            "probe_traj_loss_weight",
            "probe_heading_loss_weight",
            "probe_progress_loss_weight",
            "hidden_anchor_weight",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative.")


class TwoExpertVLMSFTModule(nn.Module):
    """Stage1 wrapper for VLM-internal two-expert slot alignment."""

    def __init__(
        self,
        backbone: nn.Module,
        config: Optional[TwoExpertVLMSFTConfig] = None,
        slot_config: Optional[TwoExpertSlotConfig] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config or TwoExpertVLMSFTConfig()
        self.slot_config = slot_config or TwoExpertSlotConfig(
            vlm_hidden_dim=self.config.vlm_hidden_dim,
            planner_dim=self.config.adapter_dim,
        )
        self.two_expert_slots = TwoExpertSoftSlots(self.slot_config)
        self.dynamic_adapter = JEPADynamicAdapter(
            vlm_hidden_dim=self.config.vlm_hidden_dim,
            adapter_dim=self.config.adapter_dim,
            mask_ratio=self.config.random_mask_ratio,
            loss_type=self.config.loss_type,
        )
        self.geometry_adapter = VGGTFeature23Adapter(
            vlm_hidden_dim=self.config.vlm_hidden_dim,
            adapter_dim=self.config.adapter_dim,
            vggt_feature_dim=self.config.vggt_feature_dim,
            mask_ratio=self.config.random_mask_ratio,
            loss_type=self.config.loss_type,
        )
        self.trajectory_probe = TwoExpertTrajectoryProbe(
            vlm_hidden_dim=self.config.vlm_hidden_dim,
            adapter_dim=self.config.adapter_dim,
            action_horizon=self.config.action_horizon,
            action_dim=self.config.action_dim,
        )
        self.apply_trainable_scope()

    def apply_trainable_scope(self) -> None:
        mode = self.config.train_mode
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if mode == "full":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = True
        elif mode == "top_layers":
            self._unfreeze_top_language_layers(self.config.top_layers)
            if self._trainable_count(self.backbone) <= 0:
                raise RuntimeError("train_mode=top_layers did not unfreeze any VLM parameters.")
        elif mode == "lora":
            for name, parameter in self.backbone.named_parameters():
                if "lora_" in name:
                    parameter.requires_grad = True
            if self._trainable_count(self.backbone) <= 0:
                raise RuntimeError("train_mode=lora requested but no LoRA parameters were found on the VLM backbone.")
        for module in (self.two_expert_slots, self.dynamic_adapter, self.geometry_adapter, self.trajectory_probe):
            for parameter in module.parameters():
                parameter.requires_grad = True

    @staticmethod
    def _trainable_count(module: nn.Module) -> int:
        return sum(int(parameter.numel()) for parameter in module.parameters() if parameter.requires_grad)

    def _unfreeze_top_language_layers(self, top_k: int) -> None:
        if top_k <= 0:
            return
        candidates = []
        for name, module in self.backbone.named_modules():
            lowered = name.lower()
            if "vision" in lowered or "visual" in lowered:
                continue
            if any(marker in lowered for marker in ("layer.", "layers.", "blocks.")):
                candidates.append((name, module))
        for _, module in candidates[-int(top_k):]:
            for parameter in module.parameters(recurse=True):
                parameter.requires_grad = True

    @staticmethod
    def _get(batch: Dict[str, Any], key: str) -> Any:
        if key in batch:
            return batch[key]
        return getattr(batch, key, None)

    def _hidden_anchor_loss(self, current: torch.Tensor, frozen: Optional[torch.Tensor]) -> torch.Tensor:
        if frozen is None or self.config.hidden_anchor_weight <= 0.0 or self.config.train_mode == "frozen":
            return current.new_zeros(())
        current_mean = current.float().mean(dim=1)
        frozen_mean = frozen.detach().to(current).float().mean(dim=1)
        return (1.0 - F.cosine_similarity(current_mean, frozen_mean, dim=-1)).mean().to(dtype=current.dtype)

    def _target_action_norm(self, batch: Dict[str, Any], ref: torch.Tensor) -> Optional[torch.Tensor]:
        trajectory_norm = self._get(batch, "trajectory_norm")
        if trajectory_norm is not None:
            return trajectory_norm.to(device=ref.device, dtype=ref.dtype)
        trajectory = self._get(batch, "trajectory")
        if trajectory is None:
            return None
        return norm_odo(trajectory.to(device=ref.device, dtype=ref.dtype))

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        images = self._get(batch, "images")
        prompt_inputs = self._get(batch, "prompt_inputs")
        if prompt_inputs is None:
            prompt_inputs = self._get(batch, "questions")
        if images is None or prompt_inputs is None:
            raise KeyError("TwoExpertVLMSFTModule requires images and prompt_inputs/questions.")
        vlm_out = self.backbone.forward_with_two_expert_slots(
            images,
            prompt_inputs,
            two_expert_slots=self.two_expert_slots,
            return_image_hidden=True,
            return_raw_hidden=True,
            train_vlm_mode=self.config.train_mode,
        )
        jepa_target = self._get(batch, "jepa_dynamic_teacher_tokens")
        vggt_target = self._get(batch, "vggt_feature23_tokens")
        if jepa_target is None:
            raise KeyError("Stage1 two_expert_slot requires jepa_dynamic_teacher_tokens.")
        if vggt_target is None:
            raise KeyError("Stage1 two_expert_slot requires vggt_feature23_tokens.")
        dyn_out = self.dynamic_adapter(vlm_out["image_hidden"], vlm_out["h_dyn"], jepa_target)
        geo_out = self.geometry_adapter(vlm_out["image_hidden"], vlm_out["h_geo"], vggt_target)
        probe_out = self.trajectory_probe(
            vlm_out["h_dyn"],
            vlm_out["h_geo"],
            status_feature=self._get(batch, "status_feature"),
            high_command_one_hot=self._get(batch, "high_command_one_hot"),
            history_trajectory=self._get(batch, "history_trajectory"),
            target_action_norm=self._target_action_norm(batch, vlm_out["h_dyn"]),
        )
        hidden_anchor_loss = self._hidden_anchor_loss(vlm_out["raw_vlm_hidden"], self._get(batch, "frozen_raw_vlm_hidden"))
        probe_losses = probe_out["losses"]
        total = (
            float(self.config.dyn_loss_weight) * dyn_out["loss"]
            + float(self.config.geo_loss_weight) * geo_out["loss"]
            + float(self.config.probe_traj_loss_weight) * probe_losses["probe_loss"]
            + float(self.config.probe_heading_loss_weight) * probe_losses["probe_heading_loss"]
            + float(self.config.probe_progress_loss_weight) * probe_losses["probe_progress_loss"]
            + float(self.config.hidden_anchor_weight) * hidden_anchor_loss
        )
        diagnostics = {
            **dyn_out["diagnostics"],
            **geo_out["diagnostics"],
            **probe_out["diagnostics"],
            "h_dyn_norm": vlm_out["h_dyn"].detach().float().norm(dim=-1).mean().to(dtype=total.dtype),
            "h_geo_norm": vlm_out["h_geo"].detach().float().norm(dim=-1).mean().to(dtype=total.dtype),
        }
        return {
            "loss": total,
            "dyn_loss": dyn_out["loss"],
            "geo_loss": geo_out["loss"],
            "probe_loss": probe_losses["probe_loss"],
            "probe_heading_loss": probe_losses["probe_heading_loss"],
            "probe_progress_loss": probe_losses["probe_progress_loss"],
            "hidden_anchor_loss": hidden_anchor_loss,
            **diagnostics,
        }

    def trainable_parameter_report(self) -> Dict[str, int]:
        groups = {"vlm": 0, "slots": 0, "adapters": 0, "probe": 0}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("backbone."):
                groups["vlm"] += int(parameter.numel())
            elif name.startswith("two_expert_slots."):
                groups["slots"] += int(parameter.numel())
            elif name.startswith(("dynamic_adapter.", "geometry_adapter.")):
                groups["adapters"] += int(parameter.numel())
            elif name.startswith("trajectory_probe."):
                groups["probe"] += int(parameter.numel())
        groups["total"] = sum(groups.values())
        groups["vlm_trainable_param_count"] = groups["vlm"]
        groups["slots_trainable_param_count"] = groups["slots"]
        groups["adapter_trainable_param_count"] = groups["adapters"]
        groups["probe_trainable_param_count"] = groups["probe"]
        return groups
