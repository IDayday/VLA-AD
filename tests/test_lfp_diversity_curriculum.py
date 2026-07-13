from __future__ import annotations

import pytest
import torch

from navsim.agents.recogdrive.stage3_diversity_curriculum import (
    DiversityCapacityBatch,
    Stage3DiversityCapacityCache,
    compute_capacity_normalized_frontier_energy,
)
from navsim.agents.recogdrive.stage3_lfp_grpo import LFPGRPOConfig


def _capacity(dispersion=(1.0, 1.0), modes=(4.0, 1.0)) -> DiversityCapacityBatch:
    return DiversityCapacityBatch(
        support_pairwise_ade_m=torch.tensor(dispersion),
        reference_mode_count=torch.tensor(modes),
    )


def test_capacity_cache_loads_scene_records_and_rejects_missing_token() -> None:
    cache = Stage3DiversityCapacityCache(
        payload={
            "metadata": {"version": 2, "archive_fingerprint": "test"},
            "records": {
                "a": {"support_pairwise_ade_m": 0.8, "reference_mode_count": 4},
                "b": {"support_pairwise_ade_m": 0.0, "reference_mode_count": 1},
            },
        }
    )
    batch = cache.get(["b", "a"], "cpu", torch.float32)
    assert torch.equal(batch.support_pairwise_ade_m, torch.tensor([0.0, 0.8]))
    assert torch.equal(batch.reference_mode_count, torch.tensor([1.0, 4.0]))
    with pytest.raises(KeyError, match="missing 1 requested token"):
        cache.get(["missing"], "cpu", torch.float32)


def test_capacity_normalized_energy_only_boosts_uncovered_all_feasible_multimodal_scene() -> None:
    base = torch.tensor([0.10, 0.20])
    feasible = torch.ones((2, 4), dtype=torch.bool)
    output = compute_capacity_normalized_frontier_energy(
        base,
        feasible,
        torch.tensor([0.20, 0.20]),
        _capacity(),
        gap_weight=0.05,
        dispersion_floor=0.05,
    )
    expected_coverage = (0.20 + 0.05) / (1.0 + 0.05)
    expected_gap = (1.0 - 1.0 / 4.0) * (1.0 - expected_coverage)
    assert output.coverage_gap[0].item() == pytest.approx(expected_gap)
    assert output.energy[0].item() == pytest.approx(0.10 * (1.0 + 0.05 * expected_gap))
    assert output.energy[1].item() == pytest.approx(0.20)
    assert output.active_mask.tolist() == [True, False]


def test_capacity_gap_cannot_create_frontier_energy_without_policy_credit() -> None:
    output = compute_capacity_normalized_frontier_energy(
        torch.tensor([0.0]),
        torch.ones((1, 4), dtype=torch.bool),
        torch.tensor([0.05]),
        _capacity(dispersion=(1.0,), modes=(8.0,)),
        gap_weight=1.0,
        dispersion_floor=0.05,
    )

    assert output.coverage_gap.item() > 0.0
    assert output.active_mask.item()
    assert output.energy.item() == 0.0
    assert output.bonus.item() == 0.0


def test_capacity_normalized_energy_does_not_override_safety_first() -> None:
    base = torch.tensor([0.10])
    feasible = torch.tensor([[True, True, False, True]])
    output = compute_capacity_normalized_frontier_energy(
        base,
        feasible,
        torch.tensor([0.05]),
        _capacity(dispersion=(1.0,), modes=(8.0,)),
        gap_weight=1.0,
        dispersion_floor=0.05,
    )
    assert torch.equal(output.energy, base)
    assert output.bonus.item() == 0.0
    assert not output.active_mask.item()


def test_capacity_normalized_energy_is_zero_when_group_matches_support_width() -> None:
    base = torch.tensor([0.10])
    output = compute_capacity_normalized_frontier_energy(
        base,
        torch.ones((1, 4), dtype=torch.bool),
        torch.tensor([1.20]),
        _capacity(dispersion=(1.0,), modes=(5.0,)),
        gap_weight=0.5,
        dispersion_floor=0.05,
    )
    assert output.coverage_ratio.item() == 1.0
    assert output.coverage_gap.item() == 0.0
    assert torch.equal(output.energy, base)


def test_capacity_weight_zero_is_bit_exact() -> None:
    base = torch.tensor([0.0, 0.3], dtype=torch.float64)
    output = compute_capacity_normalized_frontier_energy(
        base,
        torch.ones((2, 3), dtype=torch.bool),
        torch.tensor([0.0, 0.8], dtype=torch.float64),
        _capacity(),
        gap_weight=0.0,
        dispersion_floor=0.05,
    )
    assert torch.equal(output.energy, base)


def test_capacity_curriculum_config_requires_curriculum_and_cache() -> None:
    with pytest.raises(ValueError, match="requires curriculum_enabled"):
        LFPGRPOConfig(
            frontier_diversity_capacity_enabled=True,
            frontier_diversity_capacity_cache_path="capacity.pt",
            curriculum_enabled=False,
        ).validate()
    with pytest.raises(ValueError, match="requires frontier_diversity_capacity_cache_path"):
        LFPGRPOConfig(
            frontier_diversity_capacity_enabled=True,
            curriculum_enabled=True,
        ).validate()
