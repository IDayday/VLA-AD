from __future__ import annotations

import torch

from navsim.agents.recogdrive.pareto_support import (
    DESCRIPTOR_NAMES,
    select_adaptive_pareto_supports,
    trajectory_descriptor,
)


def test_select_adaptive_pareto_supports_best_first_then_farther():
    candidates = torch.zeros(4, 8, 3)
    candidates[0, :, 0] = torch.linspace(0.0, 8.0, 8)
    candidates[1, :, 0] = torch.linspace(0.0, 8.1, 8)
    candidates[1, :, 1] = 0.05
    candidates[2, :, 0] = torch.linspace(0.0, 8.0, 8)
    candidates[2, :, 1] = torch.linspace(0.0, 3.0, 8)
    candidates[3] = float("nan")
    scores = torch.tensor([0.95, 0.94, 0.93, 1.0])
    valid = torch.tensor([True, True, True, True])
    descriptors = trajectory_descriptor(candidates[:3])
    descriptor_mean = descriptors.mean(dim=0)
    descriptor_std = descriptors.std(dim=0, unbiased=False).clamp(min=1e-3)
    descriptors = torch.cat([descriptors, torch.zeros(1, len(DESCRIPTOR_NAMES))], dim=0)

    selection = select_adaptive_pareto_supports(
        candidates,
        scores,
        valid,
        descriptors=descriptors,
        descriptor_mean=descriptor_mean,
        descriptor_std=descriptor_std,
        quality_band=0.03,
        min_descriptor_distance=0.5,
        gt_candidate_index=0,
        sources=["gt", "stage2_stochastic", "structured_perturbation", "bad"],
    )

    assert selection.mask.tolist()[:2] == [True, True]
    assert int(selection.indices[0]) == 0
    assert int(selection.indices[1]) == 2
    assert 3 not in selection.indices.tolist()
    assert 1 <= int(selection.mask.sum().item()) <= 3
    assert torch.isclose(selection.weights.sum(), torch.tensor(1.0))
    assert selection.fallback_mode == "none"


def test_select_adaptive_pareto_supports_uses_gt_fallback_without_valid_positive():
    candidates = torch.zeros(2, 8, 3)
    candidates[0, :, 0] = torch.linspace(0.0, 2.0, 8)
    candidates[1, :, 0] = torch.linspace(0.0, 4.0, 8)
    scores = torch.tensor([0.2, 0.8])
    valid = torch.tensor([False, False])

    selection = select_adaptive_pareto_supports(
        candidates,
        scores,
        valid,
        gt_candidate_index=0,
        deterministic_il_index=1,
    )

    assert selection.mask.tolist() == [True, False, False]
    assert int(selection.indices[0]) == 0
    assert selection.weights.tolist() == [1.0, 0.0, 0.0]
    assert selection.fallback_mode == "gt_fallback"
