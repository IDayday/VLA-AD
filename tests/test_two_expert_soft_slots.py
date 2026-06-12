from __future__ import annotations

import torch
import pytest

from navsim.agents.recogdrive.two_expert_slots import TwoExpertSlotConfig, TwoExpertSoftSlots


def test_two_expert_soft_slot_shapes_and_metadata():
    slots = TwoExpertSoftSlots(TwoExpertSlotConfig(vlm_hidden_dim=16, planner_dim=8))
    dyn, geo, metadata = slots.get_slots(2, device="cpu", dtype=torch.float32)

    assert dyn.shape == (2, 36, 16)
    assert geo.shape == (2, 12, 16)
    assert metadata["num_dyn_groups"] == 3
    assert metadata["num_dyn_tokens_per_group"] == 12
    assert metadata["num_geo_tokens"] == 12
    assert metadata["dyn_group_spans"] == [(0, 12), (12, 24), (24, 36)]
    assert metadata["geo_span"] == (36, 48)


def test_group_embeddings_affect_dynamic_slots():
    cfg = TwoExpertSlotConfig(vlm_hidden_dim=8, use_dyn_group_embeddings=True)
    slots = TwoExpertSoftSlots(cfg)
    with torch.no_grad():
        slots.dyn_slots.zero_()
        slots.dyn_group_embeddings[0].fill_(1.0)
        slots.dyn_group_embeddings[1].fill_(2.0)
        slots.dyn_group_embeddings[2].fill_(3.0)
    dyn, _, _ = slots.get_slots(1, device="cpu", dtype=torch.float32)

    assert torch.allclose(dyn[0, :12], torch.ones(12, 8))
    assert torch.allclose(dyn[0, 12:24], torch.full((12, 8), 2.0))
    assert torch.allclose(dyn[0, 24:36], torch.full((12, 8), 3.0))


def test_slot_dropout_only_training():
    torch.manual_seed(5)
    slots = TwoExpertSoftSlots(TwoExpertSlotConfig(vlm_hidden_dim=8, slot_dropout=0.9))
    slots.eval()
    dyn_eval, geo_eval, _ = slots.get_slots(1, device="cpu", dtype=torch.float32)
    slots.train()
    dyn_train, geo_train, _ = slots.get_slots(1, device="cpu", dtype=torch.float32)

    assert not torch.allclose(dyn_eval, dyn_train)
    assert not torch.allclose(geo_eval, geo_train)


def test_no_tokenizer_special_tokens_allowed():
    with pytest.raises(ValueError, match="must not add tokenizer"):
        TwoExpertSlotConfig(use_tokenizer_special_tokens=True)
