from __future__ import annotations

import pytest
import torch
from torch import nn

from navsim.agents.recogdrive.last_vla_latent_slots import (
    LastVLALatentSlotConfig,
    LastVLASoftLatentSlots,
    StructuredCausalMaskBuilder,
)


def test_strict_last_vla_latent_slots_shapes_and_extract():
    cfg = LastVLALatentSlotConfig(
        num_dyn_latent_tokens=3,
        num_geo_latent_tokens=4,
        num_plan_latent_tokens=2,
        vlm_hidden_dim=16,
        planner_dim=8,
        structured_mask_mode="stage2_sft",
    )
    slots = LastVLASoftLatentSlots(cfg)
    token_embeddings = torch.randn(2, 5, 16)
    attention_mask = torch.ones(2, 5, dtype=torch.long)
    batch = slots.build_inputs_embeds(token_embeddings, attention_mask, build_structured_mask=True)

    assert batch.inputs_embeds.shape == (2, 14, 16)
    assert batch.attention_mask.shape == (2, 14)
    assert batch.structured_attention_mask.shape == (2, 1, 14, 14)

    hidden = torch.randn(2, 14, 16)
    extracted = slots.extract_slot_hidden(hidden, batch.slot_spans)
    assert extracted["h_dyn"].shape == (2, 3, 16)
    assert extracted["h_geo"].shape == (2, 4, 16)
    assert extracted["h_plan"].shape == (2, 2, 16)


def test_strict_last_vla_structured_mask_blocks_future_answer():
    cfg = LastVLALatentSlotConfig(
        num_dyn_latent_tokens=1,
        num_geo_latent_tokens=1,
        num_plan_latent_tokens=1,
        vlm_hidden_dim=8,
        use_answer_tokens=True,
        structured_mask_mode="stage2_sft",
    )
    slots = LastVLASoftLatentSlots(cfg)
    batch = slots.build_inputs_embeds(
        torch.randn(1, 2, 8),
        torch.ones(1, 2, dtype=torch.long),
        answer_embeddings=torch.randn(1, 3, 8),
        build_structured_mask=True,
    )
    mask = batch.structured_attention_mask[0, 0]
    answer_start = batch.slot_spans.answer_start
    assert answer_start is not None
    assert mask[answer_start, answer_start + 1] < -1e20
    assert mask[answer_start + 2, answer_start] == 0


def test_strict_last_vla_requires_4d_mask_support():
    with pytest.raises(NotImplementedError, match="custom 4D attention_mask"):
        StructuredCausalMaskBuilder.require_4d_attention_mask_support(nn.Linear(1, 1))
