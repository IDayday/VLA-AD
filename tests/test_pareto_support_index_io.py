from __future__ import annotations

from pathlib import Path

import torch

from navsim.agents.recogdrive.pareto_support import (
    DESCRIPTOR_NAMES,
    FREE_BUCKET_ID,
    load_pareto_support_index,
    lookup_support_batch,
    assign_support_buckets,
)


def _write_index(path: Path) -> dict:
    support_trajectories = torch.zeros(2, 3, 8, 3)
    support_trajectories[0, 0, :, 0] = torch.linspace(0.0, 8.0, 8)
    support_trajectories[0, 1, :, 0] = torch.linspace(0.0, 8.0, 8)
    support_trajectories[0, 1, :, 1] = 3.0
    support_trajectories[1, 0, :, 0] = torch.linspace(0.0, 4.0, 8)
    support_mask = torch.tensor([[True, True, False], [True, False, False]])
    support_weights = torch.tensor([[0.7, 0.3, 0.0], [1.0, 0.0, 0.0]])
    support_scores = torch.tensor([[0.95, 0.93, 0.0], [0.7, 0.0, 0.0]])
    payload = {
        "version": 1,
        "tokens": ["tok-a", "tok-b"],
        "token_to_row": {"tok-a": 0, "tok-b": 1},
        "support_trajectories": support_trajectories,
        "support_mask": support_mask,
        "support_weights": support_weights,
        "support_scores": support_scores,
        "descriptor_mean": torch.zeros(len(DESCRIPTOR_NAMES)),
        "descriptor_std": torch.ones(len(DESCRIPTOR_NAMES)),
        "summary": {"unit_test": True},
    }
    torch.save(payload, path)
    return payload


def test_load_and_lookup_pareto_support_index(tmp_path: Path):
    index_path = tmp_path / "support.pt"
    _write_index(index_path)

    payload = load_pareto_support_index(index_path)
    batch = lookup_support_batch(["tok-b", "missing"], payload, missing_policy="zeros")

    assert batch["support_trajectories"].shape == (2, 3, 8, 3)
    assert batch["support_mask"].tolist() == [[True, False, False], [False, False, False]]
    assert batch["support_missing_mask"].tolist() == [False, True]
    assert torch.isclose(batch["support_weights"][0].sum(), torch.tensor(1.0))


def test_assign_support_buckets_uses_nearest_and_free_bucket(tmp_path: Path):
    index_path = tmp_path / "support.pt"
    payload = _write_index(index_path)

    trajectories = torch.zeros(1, 3, 8, 3)
    trajectories[0, 0] = payload["support_trajectories"][0, 0]
    trajectories[0, 1] = payload["support_trajectories"][0, 1]
    trajectories[0, 2, :, 0] = torch.linspace(0.0, 30.0, 8)
    support = payload["support_trajectories"][0:1]
    support_mask = payload["support_mask"][0:1]

    buckets = assign_support_buckets(
        trajectories,
        support,
        support_mask,
        payload["descriptor_mean"],
        payload["descriptor_std"],
        free_distance=1.0,
    )

    assert buckets["bucket_id"][0, 0].item() == 0
    assert buckets["bucket_id"][0, 1].item() == 1
    assert buckets["bucket_id"][0, 2].item() == FREE_BUCKET_ID
