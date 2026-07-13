from __future__ import annotations

import json
import lzma
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .dataclasses import TrajectoryCandidate, ensure_trajectory_array
from .metrics import cfg_value


def _as_trajectory_stack(value: Any) -> np.ndarray | None:
    if hasattr(value, "poses"):
        value = getattr(value, "poses")
    if isinstance(value, Mapping) and "poses" in value:
        value = value["poses"]
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return None
    if arr.ndim == 2 and arr.shape[-1] == 3:
        return arr[None, ...]
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr
    return None


def _load_payload(path: Path) -> Iterable[Any]:
    if not path.exists():
        return []
    if path.is_dir():
        payloads: list[Any] = []
        for suffix in ("*.pkl.xz", "*.pkl", "*.npz", "*.pt", "*.pth", "*.jsonl", "*.json"):
            for item in sorted(path.rglob(suffix)):
                payloads.extend(_load_payload(item))
        return payloads
    if path.name.endswith(".pkl.xz"):
        with lzma.open(path, "rb") as f:
            return [pickle.load(f)]
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            return [pickle.load(f)]
    if path.suffix in {".pt", ".pth"}:
        return [torch.load(path, map_location="cpu", weights_only=False)]
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return [{k: data[k] for k in data.files}]
    if path.suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return [json.load(f)]
    return []


def _records_from_payload(payload: Any, source: str, token_hint: str | None = None, top_k: int | None = None) -> list[TrajectoryCandidate]:
    out: list[TrajectoryCandidate] = []
    if isinstance(payload, list):
        for item in payload:
            out.extend(_records_from_payload(item, source, token_hint, top_k))
        return out
    if not isinstance(payload, Mapping):
        if token_hint is None:
            return []
        arr = _as_trajectory_stack(payload)
        if arr is None:
            return []
        for idx in range(min(arr.shape[0], top_k or arr.shape[0])):
            out.append(
                TrajectoryCandidate(
                    scene_token=token_hint,
                    candidate_id=f"{token_hint}:{source}:{idx}",
                    source=source,
                    operator="precomputed",
                    trajectory=arr[idx],
                    scorer_pred={},
                    source_metadata={"format": "array"},
                )
            )
        return out

    if "predictions" in payload and isinstance(payload["predictions"], list):
        for item in payload["predictions"]:
            out.extend(_records_from_payload(item, source, token_hint, top_k))
        return out
    if "token" in payload or "scene_token" in payload:
        token = str(payload.get("token", payload.get("scene_token")))
        if token_hint is not None and token != str(token_hint):
            return []
        trajs = payload.get("trajectories", payload.get("trajectory", payload.get("candidates", payload.get("poses"))))
        arr = _as_trajectory_stack(trajs)
        if arr is None:
            return []
        scores = np.asarray(payload.get("scores", payload.get("score", payload.get("rewards", np.zeros(arr.shape[0])))), dtype=np.float32)
        if scores.ndim == 0:
            scores = np.full((arr.shape[0],), float(scores), dtype=np.float32)
        metrics_payload = payload.get("metrics", payload.get("components", None))
        for idx in range(min(arr.shape[0], top_k or arr.shape[0])):
            true_metrics = None
            if isinstance(metrics_payload, Mapping):
                row = {}
                for key, value in metrics_payload.items():
                    values = np.asarray(value)
                    row[str(key)] = float(values[idx] if values.shape[:1] == (arr.shape[0],) else values)
                true_metrics = row
            out.append(
                TrajectoryCandidate(
                    scene_token=token,
                    candidate_id=f"{token}:{source}:{idx}",
                    source=source,
                    operator="precomputed",
                    trajectory=arr[idx],
                    scorer_pred={"source_score": float(scores[min(idx, scores.shape[0] - 1)])},
                    true_metrics=true_metrics,
                    source_metadata={k: str(v) for k, v in payload.get("metadata", {}).items()} if isinstance(payload.get("metadata"), Mapping) else {},
                )
            )
        return out

    metadata_keys = {"team_name", "authors", "email", "institution", "country / region", "country", "version", "metadata"}
    for token, value in payload.items():
        if str(token) in metadata_keys:
            continue
        if token_hint is not None and str(token) != str(token_hint):
            continue
        if isinstance(value, Mapping):
            nested = dict(value)
            nested.setdefault("token", str(token))
            out.extend(_records_from_payload(nested, source, str(token), top_k))
            continue
        arr = _as_trajectory_stack(value)
        if arr is None:
            continue
        for idx in range(min(arr.shape[0], top_k or arr.shape[0])):
            out.append(
                TrajectoryCandidate(
                    scene_token=str(token),
                    candidate_id=f"{token}:{source}:{idx}",
                    source=source,
                    operator="precomputed",
                    trajectory=arr[idx],
                    scorer_pred={},
                    source_metadata={"format": "token_map"},
                )
            )
    return out


class BaseTrajectoryGenerator:
    source: str = "unknown"

    def __init__(self, cfg: Any = None):
        self.cfg = cfg or {}

    def generate(self, scene_token: str, features: Mapping[str, Any], targets: Mapping[str, Any], num_candidates: int) -> list[TrajectoryCandidate]:
        raise NotImplementedError


class GTGenerator(BaseTrajectoryGenerator):
    source = "gt"

    def generate(self, scene_token: str, features: Mapping[str, Any], targets: Mapping[str, Any], num_candidates: int = 1) -> list[TrajectoryCandidate]:
        traj = targets.get("trajectory", targets.get("gt_trajectory"))
        if traj is None:
            return []
        return [
            TrajectoryCandidate(
                scene_token=scene_token,
                candidate_id=f"{scene_token}:gt:0",
                source="gt",
                operator="dataset",
                trajectory=ensure_trajectory_array(traj),
                true_metrics=targets.get("trajectory_metrics") if isinstance(targets.get("trajectory_metrics"), Mapping) else None,
            )
        ]


class RecogDriveStage3Generator(BaseTrajectoryGenerator):
    source = "recogdrive_stage3"

    def __init__(self, cfg: Any = None):
        super().__init__(cfg)
        self.checkpoint_path = str(cfg_value(cfg, "recogdrive_stage3_checkpoint_path", cfg_value(cfg, "recogdrive_stage3_checkpoint", "")))
        self.precomputed_dir = str(cfg_value(cfg, "recogdrive_stage3_precomputed_dir", ""))
        self.allow_cache_il = bool(cfg_value(cfg, "recogdrive_stage3_allow_cache_il", True))
        if (
            bool(cfg_value(cfg, "recogdrive_stage3_enabled", True))
            and not self.checkpoint_path
            and not self.precomputed_dir
            and not self.allow_cache_il
        ):
            raise FileNotFoundError("recogdrive_stage3 generator is enabled but no checkpoint/precomputed path was provided.")
        self._precomputed = PrecomputedTrajectoryGenerator(
            {
                "source": "recogdrive_stage3",
                "precomputed_dir": self.precomputed_dir,
                "top_k": int(cfg_value(cfg, "recogdrive_stage3_top_k", 1)),
            }
        ) if self.precomputed_dir else None

    def generate(self, scene_token: str, features: Mapping[str, Any], targets: Mapping[str, Any], num_candidates: int = 1) -> list[TrajectoryCandidate]:
        il = targets.get("il_trajectory", targets.get("expert_trajectory", targets.get("recogdrive_stage3_trajectory")))
        if il is not None:
            return [
                TrajectoryCandidate(
                    scene_token=scene_token,
                    candidate_id=f"{scene_token}:recogdrive_stage3:cache0",
                    source="recogdrive_stage3",
                    operator="cache",
                    trajectory=ensure_trajectory_array(il),
                    true_metrics=targets.get("il_metrics") if isinstance(targets.get("il_metrics"), Mapping) else None,
                )
            ]
        if self._precomputed is not None:
            return self._precomputed.generate(scene_token, features, targets, num_candidates)
        raise RuntimeError(
            "Direct RecogDrive Stage3 checkpoint sampling requires planner context; "
            "use precomputed mode or include il_trajectory/expert_trajectory in the cache sample."
        )


class PrecomputedTrajectoryGenerator(BaseTrajectoryGenerator):
    def __init__(self, cfg: Any = None):
        super().__init__(cfg)
        self.source = str(cfg_value(cfg, "source", cfg_value(cfg, "precomputed_source", "unknown")))
        self.root = Path(str(cfg_value(cfg, "precomputed_dir", cfg_value(cfg, f"{self.source}_precomputed_dir", ""))))
        self.top_k = int(cfg_value(cfg, "top_k", cfg_value(cfg, f"{self.source}_top_k", 8)))
        self.missing_policy = str(cfg_value(cfg, "missing_policy", "error"))
        self._index: dict[str, list[TrajectoryCandidate]] | None = None

    def _build_index(self) -> dict[str, list[TrajectoryCandidate]]:
        index: dict[str, list[TrajectoryCandidate]] = defaultdict(list)
        if not self.root:
            return {}
        if not self.root.exists():
            if self.missing_policy == "error":
                raise FileNotFoundError(f"{self.source} precomputed dir does not exist: {self.root}")
            return {}
        for payload in _load_payload(self.root):
            for record in _records_from_payload(payload, self.source, top_k=self.top_k):
                index[record.scene_token].append(record)
        return dict(index)

    def generate(self, scene_token: str, features: Mapping[str, Any], targets: Mapping[str, Any], num_candidates: int = 8) -> list[TrajectoryCandidate]:
        if self._index is None:
            self._index = self._build_index()
        records = list(self._index.get(str(scene_token), []))
        if not records:
            direct_paths = []
            if self.root.is_dir():
                for suffix in (".pkl.xz", ".pkl", ".npz", ".pt", ".pth", ".json"):
                    direct_paths.append(self.root / f"{scene_token}{suffix}")
            for path in direct_paths:
                for payload in _load_payload(path):
                    records.extend(_records_from_payload(payload, self.source, token_hint=str(scene_token), top_k=self.top_k))
        if not records and self.missing_policy == "error":
            raise FileNotFoundError(f"No {self.source} precomputed trajectories found for token={scene_token!r}.")
        return records[: min(num_candidates, self.top_k)]


class DDV2Generator(PrecomputedTrajectoryGenerator):
    def __init__(self, cfg: Any = None):
        mode = str(cfg_value(cfg, "ddv2_mode", "precomputed"))
        if mode == "direct_model":
            raise RuntimeError("DDV2 direct_model adapter is not initialized in this process; export precomputed trajectories first.")
        super().__init__(
            {
                "source": "ddv2",
                "precomputed_dir": cfg_value(cfg, "ddv2_precomputed_dir", ""),
                "top_k": cfg_value(cfg, "ddv2_top_k", 8),
                "missing_policy": cfg_value(cfg, "missing_external_policy", "error"),
            }
        )


class DriveORGenerator(PrecomputedTrajectoryGenerator):
    def __init__(self, cfg: Any = None):
        mode = str(cfg_value(cfg, "driveor_mode", "precomputed"))
        if mode == "direct_model":
            raise RuntimeError("DriveOR direct_model adapter is not initialized in this process; export precomputed trajectories first.")
        super().__init__(
            {
                "source": "driveor",
                "precomputed_dir": cfg_value(cfg, "driveor_precomputed_dir", ""),
                "top_k": cfg_value(cfg, "driveor_top_k", 8),
                "missing_policy": cfg_value(cfg, "missing_external_policy", "error"),
            }
        )


def build_generators(cfg: Any) -> list[BaseTrajectoryGenerator]:
    generators: list[BaseTrajectoryGenerator] = []
    if bool(cfg_value(cfg, "gt_enabled", True)):
        generators.append(GTGenerator(cfg))
    if bool(cfg_value(cfg, "recogdrive_stage3_enabled", False)):
        generators.append(RecogDriveStage3Generator(cfg))
    if bool(cfg_value(cfg, "ddv2_enabled", False)):
        generators.append(DDV2Generator(cfg))
    if bool(cfg_value(cfg, "driveor_enabled", False)):
        generators.append(DriveORGenerator(cfg))
    return generators
