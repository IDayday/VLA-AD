from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from navsim.agents.recogdrive.offline_rl_buffer import load_elite_record_path
from navsim.agents.recogdrive.pareto_support.mining_pipeline import mine_cache


def _traj(offset: float = 0.0) -> np.ndarray:
    x = np.linspace(0.2, 4.0 + offset, 8, dtype=np.float32)
    return np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=-1)


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
    }


def test_mining_pipeline_writes_full_archive_and_v3_buffer(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.pt"
    torch.save(
        [
            {
                "token": "scene",
                "trajectory": _traj(0.0),
                "trajectory_metrics": _metrics(0.7, 0.7),
                "il_trajectory": _traj(0.2),
                "il_metrics": _metrics(0.75, 0.72),
            }
        ],
        cache_path,
    )
    out = tmp_path / "out"
    summary = mine_cache(
        cache_path,
        out,
        {"recogdrive_stage3_enabled": True, "recogdrive_stage3_checkpoint_path": "cache", "support_top_m": 4, "allow_unverified_smoke": True},
        limit_scenes=1,
        overwrite=True,
    )
    assert summary["scene_count"] == 1
    root_records = list(out.glob("*.pkl.xz"))
    full_records = list((out / "full_archive").glob("*.pkl.xz"))
    assert len(root_records) == 1
    assert len(full_records) == 1
    record = load_elite_record_path(root_records[0])
    assert int(record["version"]) == 3
    assert record["candidates"].shape[-2:] == (8, 3)
