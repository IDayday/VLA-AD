from __future__ import annotations

import torch

from navsim.agents.recogdrive.recogdrive_diffusion_planner import (
    OfflineRLConfig,
    build_frontier_curriculum_mask,
)


def _cfg(enabled: bool = True) -> OfflineRLConfig:
    return OfflineRLConfig(
        dpsi_frontier_curriculum_enabled=enabled,
        dpsi_frontier_difficulty_start=0.45,
        dpsi_frontier_difficulty_end=1.0,
        dpsi_frontier_difficulty_warmup_epochs=40,
    )


def test_frontier_curriculum_exposes_easy_modes_without_per_scene_quota() -> None:
    support = torch.tensor([[True, True, True], [True, True, False]])
    difficulty = torch.tensor([[0.30, 0.60, 0.90], [0.70, 0.80, 0.0]])

    active, diagnostics = build_frontier_curriculum_mask(support, difficulty, _cfg(), current_epoch=0)

    assert torch.equal(active, torch.tensor([[True, False, False], [False, False, False]]))
    assert torch.isclose(diagnostics["dpsi_frontier_difficulty_threshold"], torch.tensor(0.45))
    assert diagnostics["dpsi_frontier_deferred_scene_ratio"] == 1.0
    assert diagnostics["dpsi_frontier_all_modes_deferred_scene_ratio"] == 0.5


def test_frontier_curriculum_recovers_complete_uniform_support_at_ramp_end() -> None:
    support = torch.tensor([[True, True, True], [True, False, False]])
    difficulty = torch.tensor([[0.30, 0.60, 0.90], [0.99, 0.0, 0.0]])

    active, diagnostics = build_frontier_curriculum_mask(support, difficulty, _cfg(), current_epoch=40)

    assert torch.equal(active, support)
    assert diagnostics["dpsi_frontier_active_mode_ratio"] == 1.0
    assert diagnostics["dpsi_frontier_deferred_scene_ratio"] == 0.0
    assert diagnostics["dpsi_frontier_all_modes_deferred_scene_ratio"] == 0.0


def test_disabled_frontier_curriculum_preserves_all_support_modes() -> None:
    support = torch.tensor([[True, True, False]])
    difficulty = torch.tensor([[0.20, 0.95, 0.0]])

    active, diagnostics = build_frontier_curriculum_mask(support, difficulty, _cfg(False), current_epoch=0)

    assert torch.equal(active, support)
    assert diagnostics["dpsi_frontier_curriculum_enabled"] == 0.0
