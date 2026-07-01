from __future__ import annotations

import torch

from navsim.agents.recogdrive.pareto_support import (
    DESCRIPTOR_NAMES,
    FREE_BUCKET_ID,
    MISSING_BUCKET_ID,
    assign_support_buckets,
)


def test_assign_support_buckets_nearest_free_and_missing() -> None:
    trajectories = torch.zeros(2, 3, 8, 3)
    support_trajectories = torch.zeros(2, 3, 8, 3)

    support_trajectories[0, 0, :, 0] = torch.linspace(0.0, 8.0, 8)
    support_trajectories[0, 1, :, 0] = torch.linspace(0.0, 8.0, 8)
    support_trajectories[0, 1, :, 1] = 3.0

    trajectories[0, 0] = support_trajectories[0, 0]
    trajectories[0, 1] = support_trajectories[0, 1]
    trajectories[0, 2, :, 0] = torch.linspace(0.0, 40.0, 8)
    trajectories[1] = trajectories[0]

    support_mask = torch.tensor(
        [
            [True, True, False],
            [False, False, False],
        ]
    )

    buckets = assign_support_buckets(
        trajectories,
        support_trajectories,
        support_mask,
        torch.zeros(len(DESCRIPTOR_NAMES)),
        torch.ones(len(DESCRIPTOR_NAMES)),
        free_distance=1.0,
    )

    assert buckets["bucket_id"][0].tolist() == [0, 1, FREE_BUCKET_ID]
    assert buckets["bucket_id"][1].tolist() == [MISSING_BUCKET_ID, MISSING_BUCKET_ID, MISSING_BUCKET_ID]
    assert buckets["has_support"].tolist() == [True, False]
