from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

from .trajectory_normalization import norm_odo
from .two_expert_adapters import JEPADynamicAdapter, TwoExpertTrajectoryProbe, VGGTFeature23Adapter
from .two_expert_slots import TwoExpertSlotConfig, TwoExpertSoftSlots


@dataclass
class TwoExpertVLMSFTConfig:
    vlm_hidden_dim: int = 1536
    adapter_dim: int = 384
    vggt_feature_dim: Optional[int] = None
    action_horizon: int = 8
    action_dim: int = 3
    train_mode: str = "lora"
    allow_full_vlm_sft: bool = False
    top_layers: int = 2
    dyn_loss_weight: float = 1.0
    geo_loss_weight: float = 1.0
    probe_traj_loss_weight: float = 0.1
    probe_fused_loss_weight: Optional[float] = None
    probe_dyn_loss_weight: float = 0.0
    probe_geo_loss_weight: float = 0.0
    probe_heading_loss_weight: float = 0.05
    probe_progress_loss_weight: float = 0.05
    geo_lateral_profile_loss_weight: float = 0.0
    geo_heading_profile_loss_weight: float = 0.0
    slot_only_dyn_loss_weight: float = 0.0
    slot_only_geo_loss_weight: float = 0.0
    contrastive_dyn_loss_weight: float = 0.0
    contrastive_geo_loss_weight: float = 0.0
    contrastive_temperature: float = 0.07
    hidden_anchor_weight: float = 0.05
    hidden_anchor_every_n_steps: int = 1
    random_mask_ratio: float = 0.3
    stage1_image_mask_ratio: Optional[float] = None
    stage1_image_mask_ratio_start: Optional[float] = None
    stage1_image_mask_warmup_fraction: float = 0.25
    stage1_image_dropout_prob: float = 0.0
    stage1_image_dropout_prob_start: float = 0.0
    stage1_image_dropout_warmup_fraction: float = 0.25
    loss_type: str = "normalized_mse"
    slot_only_use_image_memory: bool = False
    recogdrive_replay_ce_loss_weight: float = 0.0
    recogdrive_replay_every_n_steps: int = 1
    recogdrive_replay_source: str = "official"
    recogdrive_replay_max_answer_tokens: int = 256
    recogdrive_replay_train_lora_only: bool = True
    recogdrive_replay_train_slots: bool = False
    contrastive_loss_warmup_fraction: float = 0.30
    replay_ce_loss_warmup_fraction: float = 0.30

    def __post_init__(self) -> None:
        if self.train_mode not in {"frozen", "lora", "top_layers", "full"}:
            raise ValueError("train_mode must be frozen, lora, top_layers, or full.")
        if self.train_mode == "full" and not self.allow_full_vlm_sft:
            raise ValueError("Full VLM SFT requires allow_full_vlm_sft=True.")
        if self.vggt_feature_dim is not None and int(self.vggt_feature_dim) <= 0:
            raise ValueError("vggt_feature_dim must be positive when set.")
        for name in (
            "dyn_loss_weight",
            "geo_loss_weight",
            "probe_traj_loss_weight",
            "probe_dyn_loss_weight",
            "probe_geo_loss_weight",
            "probe_heading_loss_weight",
            "probe_progress_loss_weight",
            "geo_lateral_profile_loss_weight",
            "geo_heading_profile_loss_weight",
            "slot_only_dyn_loss_weight",
            "slot_only_geo_loss_weight",
            "contrastive_dyn_loss_weight",
            "contrastive_geo_loss_weight",
            "hidden_anchor_weight",
            "stage1_image_dropout_prob",
            "stage1_image_dropout_prob_start",
            "recogdrive_replay_ce_loss_weight",
            "stage1_image_mask_warmup_fraction",
            "stage1_image_dropout_warmup_fraction",
            "contrastive_loss_warmup_fraction",
            "replay_ce_loss_warmup_fraction",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if self.probe_fused_loss_weight is not None and float(self.probe_fused_loss_weight) < 0.0:
            raise ValueError("probe_fused_loss_weight must be non-negative when set.")
        if self.contrastive_temperature <= 0.0:
            raise ValueError("contrastive_temperature must be positive.")
        if int(self.hidden_anchor_every_n_steps) <= 0:
            raise ValueError("hidden_anchor_every_n_steps must be positive.")
        if int(self.recogdrive_replay_every_n_steps) <= 0:
            raise ValueError("recogdrive_replay_every_n_steps must be positive.")
        if self.stage1_image_mask_ratio is not None and not 0.0 <= float(self.stage1_image_mask_ratio) <= 1.0:
            raise ValueError("stage1_image_mask_ratio must be in [0, 1].")
        if self.stage1_image_mask_ratio_start is not None and not 0.0 <= float(self.stage1_image_mask_ratio_start) <= 1.0:
            raise ValueError("stage1_image_mask_ratio_start must be in [0, 1].")
        if not 0.0 <= float(self.stage1_image_dropout_prob_start) <= 1.0:
            raise ValueError("stage1_image_dropout_prob_start must be in [0, 1].")
        if not 0.0 <= float(self.stage1_image_dropout_prob) <= 1.0:
            raise ValueError("stage1_image_dropout_prob must be in [0, 1].")
        if self.recogdrive_replay_ce_loss_weight > 0.0 and self.train_mode == "frozen":
            raise ValueError("Replay CE requires train_mode=lora, top_layers, or full; frozen VLM cannot learn replay labels.")

    @property
    def effective_image_mask_ratio(self) -> float:
        if self.stage1_image_mask_ratio is not None:
            return float(self.stage1_image_mask_ratio)
        return float(self.random_mask_ratio)

    @property
    def effective_probe_fused_loss_weight(self) -> float:
        if self.probe_fused_loss_weight is not None:
            return float(self.probe_fused_loss_weight)
        return float(self.probe_traj_loss_weight)


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
        if self.config.vggt_feature_dim is None:
            raise ValueError(
                "TwoExpertVLMSFTConfig.vggt_feature_dim must be resolved from "
                "vggt_feature23 teacher metadata or passed explicitly."
            )
        self.slot_config = slot_config or TwoExpertSlotConfig(
            vlm_hidden_dim=self.config.vlm_hidden_dim,
            planner_dim=self.config.adapter_dim,
        )
        self.two_expert_slots = TwoExpertSoftSlots(self.slot_config)
        self.dynamic_adapter = JEPADynamicAdapter(
            vlm_hidden_dim=self.config.vlm_hidden_dim,
            adapter_dim=self.config.adapter_dim,
            mask_ratio=self.config.effective_image_mask_ratio,
            image_dropout_prob=self.config.stage1_image_dropout_prob,
            loss_type=self.config.loss_type,
        )
        self.geometry_adapter = VGGTFeature23Adapter(
            vlm_hidden_dim=self.config.vlm_hidden_dim,
            adapter_dim=self.config.adapter_dim,
            vggt_feature_dim=int(self.config.vggt_feature_dim),
            mask_ratio=self.config.effective_image_mask_ratio,
            image_dropout_prob=self.config.stage1_image_dropout_prob,
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

    @staticmethod
    def _scalar_zero(ref: torch.Tensor) -> torch.Tensor:
        return ref.new_zeros(())

    @staticmethod
    def _global_step(batch: Dict[str, Any]) -> int:
        value = TwoExpertVLMSFTModule._get(batch, "global_step")
        if isinstance(value, torch.Tensor):
            return int(value.detach().cpu().view(-1)[0].item())
        if value is None:
            return 0
        return int(value)

    @staticmethod
    def _every(batch: Dict[str, Any], interval: int) -> bool:
        return (TwoExpertVLMSFTModule._global_step(batch) % max(1, int(interval))) == 0

    @staticmethod
    def _fraction(batch: Dict[str, Any], key: str, default: float = 0.0) -> float:
        value = TwoExpertVLMSFTModule._get(batch, key)
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().view(-1)[0].cpu().item())
        if value is None:
            return float(default)
        return float(value)

    def _progress_fraction(self, batch: Dict[str, Any]) -> float:
        explicit = self._get(batch, "train_progress_fraction")
        if explicit is not None:
            return max(0.0, min(1.0, self._fraction(batch, "train_progress_fraction")))
        total = self._get(batch, "total_forward_steps")
        if total is None:
            return 1.0
        if isinstance(total, torch.Tensor):
            total_value = int(total.detach().cpu().view(-1)[0].item())
        else:
            total_value = int(total)
        if total_value <= 0:
            return 1.0
        return max(0.0, min(1.0, float(self._global_step(batch)) / float(total_value)))

    @staticmethod
    def _linear_ramp(start: float, end: float, progress: float, warmup_fraction: float) -> float:
        if warmup_fraction <= 0.0:
            return float(end)
        alpha = max(0.0, min(1.0, float(progress) / float(warmup_fraction)))
        return float(start) + (float(end) - float(start)) * alpha

    def _scheduled_image_mask_ratio(self, progress: float) -> float:
        end = self.config.effective_image_mask_ratio
        start = end if self.config.stage1_image_mask_ratio_start is None else float(self.config.stage1_image_mask_ratio_start)
        return self._linear_ramp(start, end, progress, float(self.config.stage1_image_mask_warmup_fraction))

    def _scheduled_image_dropout_prob(self, progress: float) -> float:
        return self._linear_ramp(
            float(self.config.stage1_image_dropout_prob_start),
            float(self.config.stage1_image_dropout_prob),
            progress,
            float(self.config.stage1_image_dropout_warmup_fraction),
        )

    def _scheduled_loss_weight(self, final_weight: float, progress: float, warmup_fraction: float) -> float:
        return self._linear_ramp(0.0, float(final_weight), progress, float(warmup_fraction))

    def _hidden_anchor_loss(self, current: torch.Tensor, frozen: Optional[torch.Tensor]) -> torch.Tensor:
        if frozen is None or self.config.hidden_anchor_weight <= 0.0 or self.config.train_mode == "frozen":
            return current.new_zeros(())
        current_mean = current.float().mean(dim=1)
        frozen_mean = frozen.detach().to(current).float().mean(dim=1)
        return (1.0 - F.cosine_similarity(current_mean, frozen_mean, dim=-1)).mean().to(dtype=current.dtype)

    def _online_frozen_raw_hidden(
        self,
        images: torch.Tensor,
        prompt_inputs: Any,
    ) -> Optional[torch.Tensor]:
        if self.config.train_mode != "lora":
            return None
        model = getattr(self.backbone, "model", None)
        disable_adapter = getattr(model, "disable_adapter", None)
        if disable_adapter is None:
            return None
        with torch.no_grad(), disable_adapter():
            out = self.backbone.forward_with_two_expert_slots(
                images,
                prompt_inputs,
                two_expert_slots=self.two_expert_slots,
                return_image_hidden=False,
                return_raw_hidden=True,
                train_vlm_mode=self.config.train_mode,
            )
        frozen = out.get("raw_vlm_hidden")
        return frozen.detach() if isinstance(frozen, torch.Tensor) else None

    def _contrastive_loss(self, pred: torch.Tensor, target: torch.Tensor, *, kind: str) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = pred.new_zeros(())
        batch = int(pred.shape[0])
        pred_pool = pred.float().reshape(batch, -1, pred.shape[-1]).mean(dim=1)
        target_pool = target.detach().to(pred).float().reshape(batch, -1, target.shape[-1]).mean(dim=1)
        pred_pool = F.normalize(pred_pool, dim=-1, eps=1e-6)
        target_pool = F.normalize(target_pool, dim=-1, eps=1e-6)
        global_target = target_pool
        label_offset = 0
        global_count = batch
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            target_parts = [torch.zeros_like(target_pool) for _ in range(world_size)]
            dist.all_gather(target_parts, target_pool.detach())
            global_target = torch.cat(target_parts, dim=0)
            label_offset = rank * batch
            global_count = int(global_target.shape[0])
        if global_count < 2:
            return zero, {f"contrastive_{kind}_skipped": pred.new_tensor(1.0), f"contrastive_{kind}_top1": zero}
        logits = pred_pool @ global_target.t() / float(self.config.contrastive_temperature)
        labels = torch.arange(batch, device=logits.device) + int(label_offset)
        loss = F.cross_entropy(logits, labels).to(dtype=pred.dtype)
        top1 = (logits.argmax(dim=1) == labels).float().mean().to(dtype=pred.dtype)
        row_index = torch.arange(batch, device=logits.device)
        pos = logits[row_index, labels]
        neg_mask = torch.zeros_like(logits, dtype=torch.bool)
        neg_mask[row_index, labels] = True
        hardest_neg = logits.masked_fill(neg_mask, -1e9).max(dim=1).values
        return loss, {
            f"contrastive_{kind}_skipped": zero,
            f"contrastive_{kind}_top1": top1,
            f"contrastive_{kind}_positive_margin": (pos - hardest_neg).mean().to(dtype=pred.dtype),
        }

    def _replay_ce_loss(
        self,
        images: torch.Tensor,
        batch: Dict[str, Any],
        ref: torch.Tensor,
        *,
        effective_weight: float,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = self._scalar_zero(ref)
        if self.config.recogdrive_replay_ce_loss_weight <= 0.0 or effective_weight <= 0.0:
            return zero, {
                "recogdrive_replay_ce_loss": zero,
                "recogdrive_replay_token_count": zero,
                "recogdrive_replay_active": zero,
            }
        if not self._every(batch, self.config.recogdrive_replay_every_n_steps):
            return zero, {
                "recogdrive_replay_ce_loss": zero,
                "recogdrive_replay_token_count": zero,
                "recogdrive_replay_active": zero,
            }
        replay_inputs = self._get(batch, "replay_prompt_inputs")
        if replay_inputs is None:
            raise KeyError("recogdrive_replay_ce_loss_weight>0 requires replay_prompt_inputs in the Stage1 batch.")
        out = self.backbone.forward_replay_ce(
            images,
            replay_inputs,
            train_vlm_mode=self.config.train_mode,
            max_length=int(self.config.recogdrive_replay_max_answer_tokens) + 2800,
        )
        loss = out["loss"].to(dtype=ref.dtype)
        parse_ok = self._get(batch, "replay_parse_ok")
        if isinstance(parse_ok, torch.Tensor):
            parse_ratio = parse_ok.to(device=ref.device, dtype=ref.dtype).float().mean()
        else:
            parse_ratio = ref.new_tensor(1.0)
        official = self._get(batch, "replay_official_recogdrive_stage1")
        if isinstance(official, torch.Tensor):
            official_ratio = official.to(device=ref.device, dtype=ref.dtype).float().mean()
        else:
            official_ratio = ref.new_tensor(1.0 if self.config.recogdrive_replay_source == "official" else 0.0)
        return loss, {
            "recogdrive_replay_ce_loss": loss.detach(),
            "recogdrive_replay_token_count": out["token_count"].detach().to(dtype=ref.dtype),
            "recogdrive_replay_parse_ok_ratio": parse_ratio.detach(),
            "recogdrive_replay_active": ref.new_tensor(1.0),
            "recogdrive_replay_source_is_official": official_ratio.detach(),
            "recogdrive_replay_answer_length_mean": out["answer_length_mean"].detach().to(dtype=ref.dtype),
        }

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
        progress = self._progress_fraction(batch)
        image_mask_ratio = self._scheduled_image_mask_ratio(progress)
        image_dropout_prob = self._scheduled_image_dropout_prob(progress)
        contrastive_dyn_weight = self._scheduled_loss_weight(
            float(self.config.contrastive_dyn_loss_weight),
            progress,
            float(self.config.contrastive_loss_warmup_fraction),
        )
        contrastive_geo_weight = self._scheduled_loss_weight(
            float(self.config.contrastive_geo_loss_weight),
            progress,
            float(self.config.contrastive_loss_warmup_fraction),
        )
        replay_ce_weight = self._scheduled_loss_weight(
            float(self.config.recogdrive_replay_ce_loss_weight),
            progress,
            float(self.config.replay_ce_loss_warmup_fraction),
        )
        dyn_out = self.dynamic_adapter(
            vlm_out["image_hidden"],
            vlm_out["h_dyn"],
            jepa_target,
            image_mask_ratio=image_mask_ratio,
            image_dropout_prob=image_dropout_prob,
        )
        geo_out = self.geometry_adapter(
            vlm_out["image_hidden"],
            vlm_out["h_geo"],
            vggt_target,
            image_mask_ratio=image_mask_ratio,
            image_dropout_prob=image_dropout_prob,
        )
        slot_only_dyn_out = self.dynamic_adapter(
            None if not self.config.slot_only_use_image_memory else torch.zeros_like(vlm_out["image_hidden"]),
            vlm_out["h_dyn"],
            jepa_target,
            use_image_memory=bool(self.config.slot_only_use_image_memory),
        )
        slot_only_geo_out = self.geometry_adapter(
            None if not self.config.slot_only_use_image_memory else torch.zeros_like(vlm_out["image_hidden"]),
            vlm_out["h_geo"],
            vggt_target,
            use_image_memory=bool(self.config.slot_only_use_image_memory),
        )
        contrastive_dyn_loss, contrastive_dyn_diag = self._contrastive_loss(dyn_out["pred"], jepa_target, kind="dyn")
        contrastive_geo_loss, contrastive_geo_diag = self._contrastive_loss(geo_out["pred"], vggt_target, kind="geo")
        probe_out = self.trajectory_probe(
            vlm_out["h_dyn"],
            vlm_out["h_geo"],
            status_feature=self._get(batch, "status_feature"),
            high_command_one_hot=self._get(batch, "high_command_one_hot"),
            history_trajectory=self._get(batch, "history_trajectory"),
            target_action_norm=self._target_action_norm(batch, vlm_out["h_dyn"]),
        )
        frozen_raw = self._get(batch, "frozen_raw_vlm_hidden")
        hidden_anchor_active = vlm_out["raw_vlm_hidden"].new_zeros(())
        if (
            frozen_raw is None
            and self.config.hidden_anchor_weight > 0.0
            and self._every(batch, self.config.hidden_anchor_every_n_steps)
        ):
            frozen_raw = self._online_frozen_raw_hidden(images, prompt_inputs)
        if frozen_raw is not None:
            hidden_anchor_active = vlm_out["raw_vlm_hidden"].new_tensor(1.0)
        hidden_anchor_loss = self._hidden_anchor_loss(vlm_out["raw_vlm_hidden"], frozen_raw)
        replay_ce_loss, replay_diag = self._replay_ce_loss(
            images,
            batch,
            vlm_out["h_dyn"],
            effective_weight=replay_ce_weight,
        )
        probe_losses = probe_out["losses"]
        total = (
            float(self.config.dyn_loss_weight) * dyn_out["loss"]
            + float(self.config.geo_loss_weight) * geo_out["loss"]
            + float(self.config.slot_only_dyn_loss_weight) * slot_only_dyn_out["loss"]
            + float(self.config.slot_only_geo_loss_weight) * slot_only_geo_out["loss"]
            + contrastive_dyn_weight * contrastive_dyn_loss
            + contrastive_geo_weight * contrastive_geo_loss
            + float(self.config.effective_probe_fused_loss_weight) * probe_losses["probe_loss"]
            + float(self.config.probe_dyn_loss_weight) * probe_losses["probe_dyn_loss"]
            + float(self.config.probe_geo_loss_weight) * probe_losses["probe_geo_loss"]
            + float(self.config.probe_heading_loss_weight) * probe_losses["probe_heading_loss"]
            + float(self.config.probe_progress_loss_weight) * probe_losses["probe_progress_loss"]
            + float(self.config.geo_lateral_profile_loss_weight) * probe_losses["geo_lateral_profile_loss"]
            + float(self.config.geo_heading_profile_loss_weight) * probe_losses["geo_heading_profile_loss"]
            + float(self.config.hidden_anchor_weight) * hidden_anchor_loss
            + replay_ce_weight * replay_ce_loss
        )
        diagnostics = {
            **dyn_out["diagnostics"],
            **geo_out["diagnostics"],
            **contrastive_dyn_diag,
            **contrastive_geo_diag,
            **replay_diag,
            "slot_only_dyn_loss": slot_only_dyn_out["loss"].detach().to(dtype=total.dtype),
            "slot_only_geo_loss": slot_only_geo_out["loss"].detach().to(dtype=total.dtype),
            "contrastive_dyn_loss": contrastive_dyn_loss.detach().to(dtype=total.dtype),
            "contrastive_geo_loss": contrastive_geo_loss.detach().to(dtype=total.dtype),
            "probe_dyn_loss": probe_losses["probe_dyn_loss"].detach().to(dtype=total.dtype),
            "probe_geo_loss": probe_losses["probe_geo_loss"].detach().to(dtype=total.dtype),
            "geo_lateral_profile_loss": probe_losses["geo_lateral_profile_loss"].detach().to(dtype=total.dtype),
            "geo_heading_profile_loss": probe_losses["geo_heading_profile_loss"].detach().to(dtype=total.dtype),
            "hidden_anchor_active": hidden_anchor_active.detach().to(dtype=total.dtype),
            "train_progress_fraction": total.new_tensor(float(progress)),
            "stage1_effective_image_mask_ratio": total.new_tensor(float(image_mask_ratio)),
            "stage1_effective_image_dropout_prob": total.new_tensor(float(image_dropout_prob)),
            "contrastive_dyn_loss_weight_effective": total.new_tensor(float(contrastive_dyn_weight)),
            "contrastive_geo_loss_weight_effective": total.new_tensor(float(contrastive_geo_weight)),
            "recogdrive_replay_ce_loss_weight_effective": total.new_tensor(float(replay_ce_weight)),
            **probe_out["diagnostics"],
            "h_dyn_norm": vlm_out["h_dyn"].detach().float().norm(dim=-1).mean().to(dtype=total.dtype),
            "h_geo_norm": vlm_out["h_geo"].detach().float().norm(dim=-1).mean().to(dtype=total.dtype),
        }
        return {
            "loss": total,
            "dyn_loss": dyn_out["loss"],
            "geo_loss": geo_out["loss"],
            "slot_only_dyn_loss": slot_only_dyn_out["loss"],
            "slot_only_geo_loss": slot_only_geo_out["loss"],
            "contrastive_dyn_loss": contrastive_dyn_loss,
            "contrastive_geo_loss": contrastive_geo_loss,
            "probe_loss": probe_losses["probe_loss"],
            "probe_dyn_loss": probe_losses["probe_dyn_loss"],
            "probe_geo_loss": probe_losses["probe_geo_loss"],
            "probe_heading_loss": probe_losses["probe_heading_loss"],
            "probe_progress_loss": probe_losses["probe_progress_loss"],
            "geo_lateral_profile_loss": probe_losses["geo_lateral_profile_loss"],
            "geo_heading_profile_loss": probe_losses["geo_heading_profile_loss"],
            "hidden_anchor_loss": hidden_anchor_loss,
            "recogdrive_replay_ce_loss": replay_ce_loss,
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
