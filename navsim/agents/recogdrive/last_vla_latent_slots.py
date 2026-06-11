from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import torch
from torch import nn


StructuredMaskMode = Literal["stage1_alignment", "stage2_sft"]


@dataclass
class LastVLALatentSlotConfig:
    num_dyn_latent_tokens: int = 64
    num_geo_latent_tokens: int = 64
    num_plan_latent_tokens: int = 32
    vlm_hidden_dim: int = 1536
    planner_dim: int = 384
    use_soft_latent_slots: bool = True
    use_answer_tokens: bool = False
    structured_mask_mode: StructuredMaskMode = "stage1_alignment"

    def __post_init__(self) -> None:
        if self.structured_mask_mode not in {"stage1_alignment", "stage2_sft"}:
            raise ValueError(
                "structured_mask_mode must be 'stage1_alignment' or 'stage2_sft', "
                f"got {self.structured_mask_mode!r}."
            )
        for name in ("num_dyn_latent_tokens", "num_geo_latent_tokens", "num_plan_latent_tokens"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.vlm_hidden_dim <= 0 or self.planner_dim <= 0:
            raise ValueError("vlm_hidden_dim and planner_dim must be positive.")


@dataclass
class LastVLASlotSpans:
    dyn_start: int
    dyn_end: int
    geo_start: int
    geo_end: int
    plan_start: int
    plan_end: int
    answer_start: Optional[int] = None
    answer_end: Optional[int] = None

    @property
    def latent_start(self) -> int:
        return self.dyn_start

    @property
    def latent_end(self) -> int:
        return self.plan_end


@dataclass
class LastVLASlotBatch:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: Optional[torch.Tensor]
    slot_spans: LastVLASlotSpans
    token_type_map: Optional[torch.Tensor] = None
    structured_attention_mask: Optional[torch.Tensor] = None


class LastVLASoftLatentSlots(nn.Module):
    """Soft VLM-side H_dyn/H_geo/H_plan latent slots.

    The slots are embeddings appended to the language-model input embeddings,
    rather than tokenizer special tokens. They are trainable while the VLM base
    can remain frozen.
    """

    def __init__(self, config: LastVLALatentSlotConfig) -> None:
        super().__init__()
        self.config = config
        dim = int(config.vlm_hidden_dim)
        self.dyn_slot_embeddings = nn.Parameter(torch.empty(config.num_dyn_latent_tokens, dim))
        self.geo_slot_embeddings = nn.Parameter(torch.empty(config.num_geo_latent_tokens, dim))
        self.plan_slot_embeddings = nn.Parameter(torch.empty(config.num_plan_latent_tokens, dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in (
            self.dyn_slot_embeddings,
            self.geo_slot_embeddings,
            self.plan_slot_embeddings,
        ):
            nn.init.normal_(parameter, mean=0.0, std=0.02)

    def build_inputs_embeds(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        answer_embeddings: Optional[torch.Tensor] = None,
        build_structured_mask: bool = False,
    ) -> LastVLASlotBatch:
        if not self.config.use_soft_latent_slots:
            raise NotImplementedError("Strict LaST-VLA uses soft latent slots; tokenizer special-token mode is disabled.")
        if token_embeddings.ndim != 3:
            raise ValueError(f"token_embeddings must have shape [B, N, D], got {tuple(token_embeddings.shape)}.")
        if attention_mask.ndim != 2:
            raise ValueError(f"attention_mask must have shape [B, N], got {tuple(attention_mask.shape)}.")
        if token_embeddings.shape[:2] != attention_mask.shape:
            raise ValueError(
                "token_embeddings and attention_mask disagree on [B, N]: "
                f"{tuple(token_embeddings.shape[:2])} vs {tuple(attention_mask.shape)}."
            )
        if token_embeddings.shape[-1] != self.config.vlm_hidden_dim:
            raise ValueError(
                f"token_embeddings dim {token_embeddings.shape[-1]} does not match "
                f"vlm_hidden_dim={self.config.vlm_hidden_dim}."
            )
        if answer_embeddings is not None and not self.config.use_answer_tokens:
            raise ValueError("answer_embeddings were provided but use_answer_tokens=False.")
        if answer_embeddings is not None:
            if answer_embeddings.ndim != 3 or answer_embeddings.shape[0] != token_embeddings.shape[0]:
                raise ValueError("answer_embeddings must have shape [B, N_answer, D].")
            if answer_embeddings.shape[-1] != token_embeddings.shape[-1]:
                raise ValueError("answer_embeddings last dim must match token_embeddings.")

        batch = token_embeddings.shape[0]
        slot_parts = [
            self.dyn_slot_embeddings.unsqueeze(0).expand(batch, -1, -1),
            self.geo_slot_embeddings.unsqueeze(0).expand(batch, -1, -1),
            self.plan_slot_embeddings.unsqueeze(0).expand(batch, -1, -1),
        ]
        base_len = token_embeddings.shape[1]
        dyn_start = base_len
        dyn_end = dyn_start + self.config.num_dyn_latent_tokens
        geo_start = dyn_end
        geo_end = geo_start + self.config.num_geo_latent_tokens
        plan_start = geo_end
        plan_end = plan_start + self.config.num_plan_latent_tokens
        answer_start = None
        answer_end = None
        pieces = [token_embeddings, *slot_parts]
        if answer_embeddings is not None:
            answer_start = plan_end
            answer_end = answer_start + answer_embeddings.shape[1]
            pieces.append(answer_embeddings)

        inputs_embeds = torch.cat(pieces, dim=1)
        latent_mask = torch.ones(
            batch,
            self.config.num_dyn_latent_tokens + self.config.num_geo_latent_tokens + self.config.num_plan_latent_tokens,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        masks = [attention_mask, latent_mask]
        if answer_embeddings is not None:
            masks.append(torch.ones(batch, answer_embeddings.shape[1], device=attention_mask.device, dtype=attention_mask.dtype))
        extended_attention_mask = torch.cat(masks, dim=1)

        extended_position_ids = None
        if position_ids is not None:
            if position_ids.shape != attention_mask.shape:
                raise ValueError(
                    f"position_ids shape {tuple(position_ids.shape)} does not match attention_mask {tuple(attention_mask.shape)}."
                )
            start = position_ids.max(dim=1, keepdim=True).values + 1
            extra_len = inputs_embeds.shape[1] - token_embeddings.shape[1]
            offsets = torch.arange(extra_len, device=position_ids.device, dtype=position_ids.dtype).unsqueeze(0)
            extended_position_ids = torch.cat([position_ids, start + offsets], dim=1)

        token_type_map = torch.zeros(
            batch,
            inputs_embeds.shape[1],
            device=attention_mask.device,
            dtype=torch.long,
        )
        token_type_map[:, dyn_start:dyn_end] = 1
        token_type_map[:, geo_start:geo_end] = 2
        token_type_map[:, plan_start:plan_end] = 3
        if answer_start is not None and answer_end is not None:
            token_type_map[:, answer_start:answer_end] = 4

        spans = LastVLASlotSpans(
            dyn_start=dyn_start,
            dyn_end=dyn_end,
            geo_start=geo_start,
            geo_end=geo_end,
            plan_start=plan_start,
            plan_end=plan_end,
            answer_start=answer_start,
            answer_end=answer_end,
        )
        slot_batch = LastVLASlotBatch(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            position_ids=extended_position_ids,
            slot_spans=spans,
            token_type_map=token_type_map,
        )
        if build_structured_mask:
            slot_batch.structured_attention_mask = StructuredCausalMaskBuilder().build(
                slot_batch,
                mode=self.config.structured_mask_mode,
            )
        return slot_batch

    def extract_slot_hidden(
        self,
        last_hidden_state: torch.Tensor,
        slot_spans: LastVLASlotSpans,
    ) -> Dict[str, torch.Tensor]:
        if last_hidden_state.ndim != 3:
            raise ValueError(f"last_hidden_state must have shape [B, N, D], got {tuple(last_hidden_state.shape)}.")
        if slot_spans.plan_end > last_hidden_state.shape[1]:
            raise ValueError(
                f"slot spans end at {slot_spans.plan_end}, but hidden sequence length is {last_hidden_state.shape[1]}."
            )
        return {
            "h_dyn": last_hidden_state[:, slot_spans.dyn_start:slot_spans.dyn_end],
            "h_geo": last_hidden_state[:, slot_spans.geo_start:slot_spans.geo_end],
            "h_plan": last_hidden_state[:, slot_spans.plan_start:slot_spans.plan_end],
        }


class StructuredCausalMaskBuilder:
    """Builds strict 4D additive masks for latent-slot VLM forwards."""

    def build(
        self,
        slot_batch: LastVLASlotBatch,
        mode: StructuredMaskMode,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if mode not in {"stage1_alignment", "stage2_sft"}:
            raise ValueError(f"Unknown structured mask mode: {mode!r}.")
        attention_mask = slot_batch.attention_mask
        if attention_mask.ndim != 2:
            raise ValueError("slot_batch.attention_mask must have shape [B, N].")
        device = attention_mask.device
        batch, seq_len = attention_mask.shape
        mask_dtype = dtype or torch.float32
        blocked = torch.finfo(mask_dtype).min
        allowed = torch.zeros(seq_len, seq_len, device=device, dtype=mask_dtype)
        blocked_matrix = torch.full((seq_len, seq_len), blocked, device=device, dtype=mask_dtype)

        spans = slot_batch.slot_spans
        base_end = spans.dyn_start
        answer_start = spans.answer_start
        answer_end = spans.answer_end
        causal_allowed = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        allow = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)

        # Base image/text tokens keep normal visibility inside the prompt only.
        allow[:base_end, :base_end] = True

        # Latent slots are VLM hidden tokens: they can read the prompt and each
        # other, but never read answer tokens.
        allow[spans.dyn_start:spans.plan_end, :spans.plan_end] = True

        if mode == "stage2_sft":
            # Plan slots may use H_dyn/H_geo and previous plan slots, but not
            # answer future. This keeps plan latent generation before answer.
            allow[spans.plan_start:spans.plan_end, :spans.plan_end] = True

        if answer_start is not None and answer_end is not None:
            # Answer tokens read image/text, H_dyn/H_geo/H_plan, and previous
            # answer tokens only. Future answer leakage is blocked.
            answer_rows = slice(answer_start, answer_end)
            allow[answer_rows, :answer_start] = True
            allow[answer_rows, answer_start:answer_end] = causal_allowed[answer_rows, answer_start:answer_end]

        allowed = torch.where(allow, allowed, blocked_matrix)
        key_padding_block = attention_mask == 0
        mask = allowed.unsqueeze(0).unsqueeze(0).expand(batch, 1, seq_len, seq_len).clone()
        mask = mask.masked_fill(key_padding_block[:, None, None, :], blocked)
        return mask

    @staticmethod
    def require_4d_attention_mask_support(model: nn.Module) -> None:
        forward = getattr(model, "forward", None)
        if forward is None:
            raise NotImplementedError("Strict LaST-VLA structured masks require a model.forward method.")
        # Many HuggingFace wrappers accept attention_mask via **kwargs. We cannot
        # prove 4D support statically, so wrappers must opt in with this flag.
        if not getattr(model, "supports_4d_attention_mask", False):
            raise NotImplementedError(
                "Strict LaST-VLA structured masks require a model API that accepts a custom 4D attention_mask. "
                "Set model.supports_4d_attention_mask=True only after verifying the wrapper forwards it to the LLM."
            )
