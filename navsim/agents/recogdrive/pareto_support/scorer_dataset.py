from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .io import load_archive


SOURCE_IDS = {
    "gt": 0,
    "recogdrive_stage3": 1,
    "ddv2": 2,
    "driveor": 3,
    "path_speed_recombine": 4,
    "progress_tune": 5,
    "yield_delay": 6,
    "ddc_repair": 7,
    "feas_repair": 8,
    "control_expand": 9,
}


class ParetoScorerDataset(Dataset):
    def __init__(self, archive_dir: str | Path):
        self.paths = sorted(Path(archive_dir).glob("*.pkl.xz"))
        self.index: list[tuple[int, int]] = []
        self._archives: dict[int, Any] = {}
        for path_idx, path in enumerate(self.paths):
            archive = load_archive(path)
            self._archives[path_idx] = archive
            for cand_idx in range(len(archive.evaluated_candidates)):
                self.index.append((path_idx, cand_idx))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        path_idx, cand_idx = self.index[item]
        archive = self._archives[path_idx]
        candidate = archive.evaluated_candidates[cand_idx]
        metrics = candidate.true_metrics or {}
        target = {
            "trajectory": torch.as_tensor(candidate.trajectory, dtype=torch.float32),
            "source_id": torch.tensor(SOURCE_IDS.get(candidate.source, 15), dtype=torch.long),
        }
        for key in ("pdms", "time_to_collision_within_bound", "ego_progress", "history_comfort", "driving_direction_compliance", "feas_cost"):
            short = {
                "time_to_collision_within_bound": "ttc",
                "ego_progress": "ep",
                "history_comfort": "comfort",
                "driving_direction_compliance": "ddc",
            }.get(key, key)
            target[short] = torch.tensor(float(metrics.get(key, 0.0)), dtype=torch.float32)
        target["nc"] = torch.tensor(float(metrics.get("no_at_fault_collisions", 0.0)), dtype=torch.float32)
        target["dac"] = torch.tensor(float(metrics.get("drivable_area_compliance", 0.0)), dtype=torch.float32)
        target["tlc"] = torch.tensor(float(metrics.get("traffic_light_compliance", 0.0)), dtype=torch.float32)
        target["pareto_front"] = torch.tensor(float(candidate.pareto_front), dtype=torch.float32)
        target["utility"] = torch.tensor(float(metrics.get("utility", 0.0)), dtype=torch.float32)
        scene = np.asarray(metrics.get("scene_feature", np.zeros((256,), dtype=np.float32)), dtype=np.float32)
        target["scene_tokens"] = torch.as_tensor(scene, dtype=torch.float32)
        return target
