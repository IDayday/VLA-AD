#!/usr/bin/env python
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.offline_rl_buffer import load_elite_record_path
from navsim.agents.recogdrive.pareto_support.bpdas import compute_pdas_metrics
from navsim.agents.recogdrive.pareto_support.dataclasses import TrajectoryCandidate
from navsim.agents.recogdrive.pareto_support.fs_norm import (
    delta_to_trajectory,
    fit_fs_norm_stats,
    normalize_trajectory,
    denormalize_trajectory,
    trajectory_to_delta,
)
from navsim.agents.recogdrive.pareto_support.mining_pipeline import mine_scene
from navsim.agents.recogdrive.pareto_support.pareto_archive import build_archive_record, build_support_set
from navsim.agents.recogdrive.pareto_support.vector_scorer import ParetoVectorScorer
from navsim.agents.recogdrive.offline_rl_buffer import save_elite_record


def _traj(offset: float = 0.0) -> np.ndarray:
    x = np.linspace(0.2, 4.0 + offset, 8, dtype=np.float32)
    return np.stack([x, 0.05 * offset * np.ones_like(x), np.zeros_like(x)], axis=-1)


def _metrics(pdms: float, ep: float) -> dict[str, float]:
    return {
        "pdms": pdms,
        "no_at_fault_collisions": 1.0,
        "drivable_area_compliance": 1.0,
        "time_to_collision_within_bound": 1.0,
        "ego_progress": ep,
        "history_comfort": 1.0,
        "lane_keeping": 1.0,
        "driving_direction_compliance": 1.0,
        "traffic_light_compliance": 1.0,
        "feas_cost": 0.0,
    }


def main() -> None:
    trajs = np.stack([_traj(0.0), _traj(0.3), _traj(0.6)], axis=0)
    stats = fit_fs_norm_stats(trajs, robust=True, clip=100.0)
    recon = denormalize_trajectory(normalize_trajectory(trajs, stats), stats)
    assert np.max(np.abs(recon - trajs)) < 1e-4
    assert np.max(np.abs(delta_to_trajectory(trajectory_to_delta(trajs)) - trajs)) < 1e-5

    candidates = [
        TrajectoryCandidate("scene", "scene:gt:0", "gt", trajectory=_traj(0.0), true_metrics=_metrics(0.70, 0.70)),
        TrajectoryCandidate("scene", "scene:recogdrive_stage3:0", "recogdrive_stage3", trajectory=_traj(0.2), true_metrics=_metrics(0.75, 0.72)),
        TrajectoryCandidate("scene", "scene:ddv2:0", "ddv2", trajectory=_traj(0.5), true_metrics=_metrics(0.82, 0.80)),
        TrajectoryCandidate("scene", "scene:driveor:0", "driveor", trajectory=_traj(0.3), true_metrics=_metrics(0.78, 0.76)),
    ]
    archive = build_support_set(candidates, candidates[1], {"support_top_m": 4})
    assert len(archive.support_set) >= 3

    with tempfile.TemporaryDirectory() as tmpdir:
        legacy_all = [c.to_legacy_record() for c in archive.evaluated_candidates]
        legacy_selected = [c.to_legacy_record() for c in archive.support_set]
        record = build_archive_record("scene", legacy_all, legacy_selected, ref=archive.reference["ref_metrics"], cfg={})
        save_elite_record(Path(tmpdir), "scene", record)
        loaded = load_elite_record_path(next(Path(tmpdir).glob("*.pkl.xz")))
        assert int(loaded["version"]) == 3
        assert loaded["candidates"].shape[-2:] == (8, 3)

    rewards = torch.tensor([[0.7, 0.8, 0.6, 0.9]], dtype=torch.float32)
    traj_tensor = torch.as_tensor(trajs[[0, 1, 2, 1]], dtype=torch.float32).unsqueeze(0)
    pdas = compute_pdas_metrics(
        rewards,
        {"pdms": rewards, "ego_progress": rewards, "driving_direction_compliance": torch.ones_like(rewards), "feas_cost": torch.zeros_like(rewards)},
        traj_tensor,
        {"reward": torch.tensor([0.7]), "pdms": torch.tensor([0.7]), "ego_progress": torch.tensor([0.7]), "driving_direction_compliance": torch.tensor([1.0])},
    )
    assert torch.isfinite(pdas.weight).all()
    assert float(pdas.weight.std(unbiased=False)) >= 0.0

    scorer = ParetoVectorScorer()
    outputs = scorer(None, None, None, torch.as_tensor(trajs, dtype=torch.float32), torch.zeros(3, dtype=torch.long))
    assert "pdms" in outputs and torch.isfinite(outputs["pdms"]).all()

    smoke_archive = mine_scene(
        {
            "token": "scene_smoke",
            "trajectory": _traj(0.0),
            "trajectory_metrics": _metrics(0.70, 0.70),
            "il_trajectory": _traj(0.2),
            "il_metrics": _metrics(0.75, 0.72),
        },
        {"recogdrive_stage3_enabled": True, "recogdrive_stage3_checkpoint_path": "cache", "support_top_m": 4, "allow_unverified_smoke": True},
        metric_context=None,
        scorer=None,
        round_id="all",
    )
    assert smoke_archive.scene_token == "scene_smoke"
    print("SG-FPS core smoke passed")


if __name__ == "__main__":
    main()
