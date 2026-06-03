#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch

A0_OFFICIAL_BASELINE_PDMS = 0.864891
PDM_COMPONENT_KEYS = ("PDMS", "NC", "DAC", "TTC", "comfort", "EP", "DDC")


def proxy_score(trajectory: torch.Tensor, gt: Optional[torch.Tensor] = None) -> float:
    traj = trajectory.detach().float()
    progress = traj[-1, 0] - traj[0, 0]
    lateral = traj[:, 1].abs().mean()
    heading = traj[:, 2].abs().mean()
    comfort = traj[1:, :2].diff(dim=0).norm(dim=-1).mean() if traj.shape[0] > 2 else traj.new_tensor(0.0)
    score = 0.10 * progress - 0.05 * lateral - 0.02 * heading - 0.01 * comfort
    if gt is not None:
        score = score - 0.03 * (traj - gt.detach().float()).abs().mean()
    return float(score.item())


def _trajectory_l1(trajectory: torch.Tensor, sample: Optional[Dict[str, Any]]) -> Optional[float]:
    if sample is None:
        return None
    target = sample.get("trajectory")
    if not isinstance(target, torch.Tensor):
        return None
    return float((trajectory.detach().float().cpu() - target.detach().float().cpu()).abs().mean().item())


def _pdm_components_from_result(result_dict: Dict[str, Any]) -> Dict[str, Optional[float]]:
    return {
        "PDMS": float(result_dict["score"]),
        "NC": float(result_dict["no_at_fault_collisions"]),
        "DAC": float(result_dict["drivable_area_compliance"]),
        "TTC": float(result_dict["time_to_collision_within_bound"]),
        "comfort": float(result_dict["comfort"]),
        "EP": float(result_dict["ego_progress"]),
        "DDC": float(result_dict["driving_direction_compliance"]),
    }


class TrajectoryScorer:
    """Scores candidate trajectories with either real NAVSIM PDM or explicit proxy scoring."""

    def __init__(
        self,
        score_mode: str,
        metric_cache_dir: Optional[Path] = None,
        *,
        pdm_backend: Optional[Any] = None,
    ) -> None:
        if score_mode not in {"pdm", "proxy"}:
            raise ValueError(f"score_mode must be 'pdm' or 'proxy', got {score_mode!r}.")
        self.score_mode = score_mode
        self.metric_cache_dir = Path(metric_cache_dir) if metric_cache_dir is not None else None
        self._pdm_backend = pdm_backend
        self._metric_cache_loader = None
        self._pdm_tools = None
        if self.score_mode == "pdm":
            if self._pdm_backend is None:
                if self.metric_cache_dir is None:
                    raise ValueError("--score-mode pdm requires --metric-cache-dir.")
                if not self.metric_cache_dir.exists():
                    raise FileNotFoundError(f"Metric cache directory does not exist: {self.metric_cache_dir}")
                from scripts import eval_recogdrive_expert_pdm as base

                self._metric_cache_loader = base.build_metric_cache_loader(self.metric_cache_dir)
                self._pdm_tools = base.build_pdm_tools()

    @property
    def pdm_scoring_active(self) -> bool:
        return self.score_mode == "pdm"

    @property
    def proxy_scoring_active(self) -> bool:
        return self.score_mode == "proxy"

    def score(self, sample: Optional[Dict[str, Any]], sample_token: str, trajectory: torch.Tensor) -> Dict[str, Any]:
        traj = trajectory.detach().float().cpu()
        l1 = _trajectory_l1(traj, sample)
        if self.score_mode == "proxy":
            gt = sample.get("trajectory") if isinstance(sample, dict) else None
            score = proxy_score(traj, gt if isinstance(gt, torch.Tensor) else None)
            return {
                "score": score,
                "PDMS": None,
                "NC": None,
                "DAC": None,
                "TTC": None,
                "comfort": None,
                "EP": None,
                "DDC": None,
                "trajectory_l1": l1,
                "score_mode": "proxy",
                "proxy_score": score,
            }
        if self._pdm_backend is not None:
            result = self._pdm_backend(sample, sample_token, traj)
            if not isinstance(result, dict):
                raise TypeError("Mock PDM backend must return a dict.")
            components = {key: result.get(key) for key in PDM_COMPONENT_KEYS}
            if components["PDMS"] is None:
                components["PDMS"] = result.get("score")
            if components["PDMS"] is None:
                raise KeyError("PDM backend result must include PDMS or score.")
            score = float(components["PDMS"])
            return {
                "score": score,
                **{key: (float(value) if value is not None else None) for key, value in components.items()},
                "trajectory_l1": l1 if result.get("trajectory_l1") is None else result.get("trajectory_l1"),
                "score_mode": "pdm",
            }

        assert self._metric_cache_loader is not None and self._pdm_tools is not None
        metric_paths = getattr(self._metric_cache_loader, "metric_cache_paths", {}) or {}
        if sample_token not in metric_paths:
            raise KeyError(f"Metric cache missing sample_token={sample_token}")
        from navsim.common.dataclasses import Trajectory
        from navsim.evaluate.pdm_score import pdm_score

        metric_cache = self._metric_cache_loader.get_from_token(sample_token)
        future_sampling, simulator, scorer = self._pdm_tools
        result = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=Trajectory(poses=traj.numpy()),
            future_sampling=future_sampling,
            simulator=simulator,
            scorer=scorer,
        )
        result_dict = asdict(result)
        components = _pdm_components_from_result(result_dict)
        return {
            "score": components["PDMS"],
            **components,
            "trajectory_l1": l1,
            "score_mode": "pdm",
        }


def average_score_dicts(results: list[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    output: Dict[str, Optional[float]] = {}
    for key in ("score", *PDM_COMPONENT_KEYS, "trajectory_l1", "proxy_score"):
        values = [float(item[key]) for item in results if item.get(key) is not None]
        output[key] = float(torch.tensor(values, dtype=torch.float32).mean().item()) if values else None
    return output


def json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)
