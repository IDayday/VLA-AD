from __future__ import annotations

import torch

from scripts.training.build_recogdrive_stage3_awac_elite_buffer import _candidate_records_from_awac_row


def _awac_batch() -> dict:
    candidates = torch.arange(1 * 3 * 8 * 3, dtype=torch.float32).reshape(1, 3, 8, 3)
    components = {"pdms": torch.ones((1, 3), dtype=torch.float32)}
    return {
        "candidate_sources": ["gt", "progress_endpoint", "ddv2"],
        "candidate_trajs": candidates,
        "candidate_rewards": torch.tensor([[0.9, 0.95, 0.96]]),
        "candidate_selection_score": torch.tensor([[0.9, 0.94, 0.95]]),
        "candidate_components": components,
        "selected_real_mask": torch.tensor([[True]]),
        "selected_source_index": torch.tensor([[0]]),
        "selected_trajs": candidates[:, :1],
        "selected_rewards": torch.tensor([[0.9]]),
        "selected_selection_score": torch.tensor([[0.9]]),
        "selected_components": {"pdms": torch.ones((1, 1), dtype=torch.float32)},
    }


def test_v4_builder_uses_raw_candidates_before_awac_topk() -> None:
    awac = _awac_batch()

    legacy = _candidate_records_from_awac_row("scene", awac, 0)
    raw = _candidate_records_from_awac_row("scene", awac, 0, use_raw_candidates=True)

    assert [record.source for record in legacy] == ["gt"]
    assert [record.source for record in raw] == ["gt", "progress_endpoint", "ddv2"]
