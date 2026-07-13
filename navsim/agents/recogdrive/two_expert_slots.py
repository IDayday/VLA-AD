from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class TwoExpertSlotConfig:
    vlm_hidden_dim: int = 1536
    planner_dim: int = 384

    num_dyn_groups: int = 3
    num_dyn_tokens_per_group: int = 12
    num_geo_tokens: int = 12

    slot_init_std: float = 0.02
    slot_dropout: float = 0.0
    use_soft_slots: bool = True
    use_tokenizer_special_tokens: bool = False

    use_dyn_group_embeddings: bool = True
    use_geo_type_embedding: bool = True

    def __post_init__(self) -> None:
        if not self.use_soft_slots:
            raise NotImplementedError("two_expert_slot only supports soft slots.")
        if self.use_tokenizer_special_tokens:
            raise ValueError("two_expert_slot must not add tokenizer special tokens or resize embeddings.")
        for name in ("vlm_hidden_dim", "planner_dim", "num_dyn_groups", "num_dyn_tokens_per_group", "num_geo_tokens"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if not 0.0 <= float(self.slot_dropout) < 1.0:
            raise ValueError("slot_dropout must be in [0, 1).")

    @property
    def num_dyn_tokens(self) -> int:
        return int(self.num_dyn_groups) * int(self.num_dyn_tokens_per_group)


class TwoExpertSoftSlots(nn.Module):
    """Soft expert slots inserted into the VLM embedding sequence.

    These are not tokenizer tokens. They are trainable embeddings that must pass
    through the VLM transformer and are later extracted as H_dyn and H_geo.
    """

    def __init__(self, config: TwoExpertSlotConfig) -> None:
        super().__init__()
        self.config = config
        dim = int(config.vlm_hidden_dim)
        self.dyn_slots = nn.Parameter(torch.empty(config.num_dyn_groups, config.num_dyn_tokens_per_group, dim))
        self.geo_slots = nn.Parameter(torch.empty(config.num_geo_tokens, dim))
        if config.use_dyn_group_embeddings:
            self.dyn_group_embeddings = nn.Parameter(torch.empty(config.num_dyn_groups, dim))
        else:
            self.register_parameter("dyn_group_embeddings", None)
        if config.use_geo_type_embedding:
            self.geo_type_embedding = nn.Parameter(torch.empty(dim))
        else:
            self.register_parameter("geo_type_embedding", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = float(self.config.slot_init_std)
        nn.init.normal_(self.dyn_slots, mean=0.0, std=std)
        nn.init.normal_(self.geo_slots, mean=0.0, std=std)
        if self.dyn_group_embeddings is not None:
            nn.init.normal_(self.dyn_group_embeddings, mean=0.0, std=std)
        if self.geo_type_embedding is not None:
            nn.init.normal_(self.geo_type_embedding, mean=0.0, std=std)

    def _metadata(self, dyn_start: int = 0) -> Dict[str, Any]:
        cfg = self.config
        dyn_group_spans: List[Tuple[int, int]] = []
        cursor = int(dyn_start)
        for _ in range(cfg.num_dyn_groups):
            dyn_group_spans.append((cursor, cursor + cfg.num_dyn_tokens_per_group))
            cursor += cfg.num_dyn_tokens_per_group
        geo_span = (cursor, cursor + cfg.num_geo_tokens)
        return {
            "num_dyn_groups": int(cfg.num_dyn_groups),
            "num_dyn_tokens_per_group": int(cfg.num_dyn_tokens_per_group),
            "num_geo_tokens": int(cfg.num_geo_tokens),
            "dyn_group_spans": dyn_group_spans,
            "geo_span": geo_span,
            "slot_mode": "vlm_soft_slots",
        }

    def get_slots(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        cfg = self.config
        dyn = self.dyn_slots
        if self.dyn_group_embeddings is not None:
            dyn = dyn + self.dyn_group_embeddings[:, None, :]
        geo = self.geo_slots
        if self.geo_type_embedding is not None:
            geo = geo + self.geo_type_embedding[None, :]

        dyn = dyn.reshape(cfg.num_dyn_tokens, cfg.vlm_hidden_dim).unsqueeze(0).expand(batch_size, -1, -1)
        geo = geo.unsqueeze(0).expand(batch_size, -1, -1)
        dyn = dyn.to(device=device, dtype=dtype)
        geo = geo.to(device=device, dtype=dtype)
        if self.training and cfg.slot_dropout > 0.0:
            dyn = F.dropout(dyn, p=float(cfg.slot_dropout), training=True)
            geo = F.dropout(geo, p=float(cfg.slot_dropout), training=True)
        return dyn, geo, self._metadata(dyn_start=0)


def extract_two_expert_slot_hidden(
    full_hidden_state: torch.Tensor,
    metadata: Dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract H_dyn [B,G,T,D] and H_geo [B,T,D] from full hidden states."""
    if full_hidden_state.ndim != 3:
        raise ValueError(f"full_hidden_state must have shape [B, N, D], got {tuple(full_hidden_state.shape)}.")
    dyn_spans = metadata.get("absolute_dyn_group_spans") or metadata.get("dyn_group_spans")
    geo_span = metadata.get("absolute_geo_span") or metadata.get("geo_span")
    if not dyn_spans or geo_span is None:
        raise KeyError("slot metadata must include dyn_group_spans and geo_span.")
    dyn_groups = []
    for start, end in dyn_spans:
        dyn_groups.append(full_hidden_state[:, int(start):int(end)])
    h_dyn = torch.stack(dyn_groups, dim=1)
    h_geo = full_hidden_state[:, int(geo_span[0]):int(geo_span[1])]
    return h_dyn, h_geo
