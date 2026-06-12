from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

_install_dependency_stubs()

from navsim.agents.recogdrive.recogdrive_backbone import RecogDriveBackbone
from navsim.agents.recogdrive.two_expert_slots import TwoExpertSlotConfig, TwoExpertSoftSlots


class _MockTokenizer:
    padding_side = "left"

    def __call__(self, queries, return_tensors=None, padding=None, max_length=None):
        batch = len(queries)
        input_ids = torch.ones(batch, 6, dtype=torch.long)
        input_ids[:, 1:3] = 5
        attention_mask = torch.ones(batch, 6, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _MockLanguage(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(16, hidden_dim)

    def get_input_embeddings(self):
        return self.emb


class _MockInternVL(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.language_model = _MockLanguage(hidden_dim)
        self.seen_inputs_embeds = False

    def forward(self, pixel_values=None, inputs_embeds=None, **kwargs):
        assert inputs_embeds is not None
        self.seen_inputs_embeds = True
        return SimpleNamespace(hidden_states=[inputs_embeds + 1.0])


def _mock_backbone(hidden_dim: int = 16) -> RecogDriveBackbone:
    backbone = RecogDriveBackbone.__new__(RecogDriveBackbone)
    nn.Module.__init__(backbone)
    backbone.model = _MockInternVL(hidden_dim)
    backbone.tokenizer = _MockTokenizer()
    backbone.model_type = "internvl"
    backbone.device = "cpu"
    backbone.num_image_token = 2
    backbone.img_context_token_id = 5
    return backbone


def test_forward_with_two_expert_slots_passes_soft_slots_through_vlm():
    backbone = _mock_backbone()
    slots = TwoExpertSoftSlots(TwoExpertSlotConfig(vlm_hidden_dim=16, planner_dim=8))
    out = backbone.forward_with_two_expert_slots(
        torch.randn(2, 3, 4, 4),
        {"questions": ["<image>\nq"], "num_patches_list": [2]},
        slots,
    )

    assert backbone.model.seen_inputs_embeds
    assert out["raw_vlm_hidden"].shape == (1, 6, 16)
    assert out["image_hidden"].shape == (1, 2, 16)
    assert out["h_dyn"].shape == (1, 3, 12, 16)
    assert out["h_geo"].shape == (1, 12, 16)
    assert out["full_hidden_state"].shape[1] == 6 + 36 + 12


def test_forward_with_two_expert_slots_fail_fast_without_inputs_embeds():
    class NoInputsEmbeds(_MockInternVL):
        def forward(
            self,
            pixel_values=None,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            image_flags=None,
            output_hidden_states=None,
            return_dict=None,
        ):
            return SimpleNamespace(hidden_states=[torch.zeros(1, 1, 16)])

    backbone = _mock_backbone()
    backbone.model = NoInputsEmbeds(16)
    slots = TwoExpertSoftSlots(TwoExpertSlotConfig(vlm_hidden_dim=16, planner_dim=8))

    try:
        backbone.forward_with_two_expert_slots(torch.randn(1, 3, 4, 4), ["q"], slots)
    except NotImplementedError as exc:
        assert "inputs_embeds" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError")
