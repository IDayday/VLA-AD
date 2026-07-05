from __future__ import annotations

import hashlib
import json
import lzma
import os
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .dataclasses import ParetoSupportArchive, TrajectoryCandidate


ARCHIVE_VERSION = 3


def stable_config_hash(config: Any) -> str:
    try:
        payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    except TypeError:
        payload = repr(config).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def archive_path(root: str | Path, scene_token: str) -> Path:
    key = hashlib.sha1(str(scene_token).encode("utf-8")).hexdigest()
    return Path(root) / f"{key}.pkl.xz"


def _candidate_to_dict(candidate: TrajectoryCandidate) -> dict[str, Any]:
    data = asdict(candidate)
    data["trajectory"] = candidate.trajectory.astype("float32")
    return data


def _candidate_from_dict(data: dict[str, Any]) -> TrajectoryCandidate:
    return TrajectoryCandidate(**data)


def archive_to_dict(archive: ParetoSupportArchive) -> dict[str, Any]:
    return {
        "version": ARCHIVE_VERSION,
        "scene_token": archive.scene_token,
        "reference": archive.reference,
        "seed_candidates": [_candidate_to_dict(c) for c in archive.seed_candidates],
        "generated_candidates": [_candidate_to_dict(c) for c in archive.generated_candidates],
        "evaluated_candidates": [_candidate_to_dict(c) for c in archive.evaluated_candidates],
        "support_set": [_candidate_to_dict(c) for c in archive.support_set],
        "hard_negatives": [_candidate_to_dict(c) for c in archive.hard_negatives],
        "operator_stats": archive.operator_stats,
        "archive_hypervolume": float(archive.archive_hypervolume),
        "oracle_best": archive.oracle_best,
        "build_metadata": archive.build_metadata,
    }


def archive_from_dict(data: dict[str, Any]) -> ParetoSupportArchive:
    version = int(data.get("version", 1))
    if version > ARCHIVE_VERSION:
        raise ValueError(f"Unsupported archive version {version}; max supported is {ARCHIVE_VERSION}.")
    return ParetoSupportArchive(
        scene_token=str(data["scene_token"]),
        reference=dict(data.get("reference", {})),
        seed_candidates=[_candidate_from_dict(c) for c in data.get("seed_candidates", [])],
        generated_candidates=[_candidate_from_dict(c) for c in data.get("generated_candidates", [])],
        evaluated_candidates=[_candidate_from_dict(c) for c in data.get("evaluated_candidates", [])],
        support_set=[_candidate_from_dict(c) for c in data.get("support_set", [])],
        hard_negatives=[_candidate_from_dict(c) for c in data.get("hard_negatives", [])],
        operator_stats=dict(data.get("operator_stats", {})),
        archive_hypervolume=float(data.get("archive_hypervolume", 0.0)),
        oracle_best=dict(data.get("oracle_best", {})),
        build_metadata=dict(data.get("build_metadata", {})),
    )


def save_archive(root: str | Path, archive: ParetoSupportArchive, *, overwrite: bool = False) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = archive_path(root, archive.scene_token)
    if path.exists() and not overwrite:
        return path
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with lzma.open(tmp, "wb") as f:
            pickle.dump(archive_to_dict(archive), f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def load_archive(path_or_root: str | Path, scene_token: str | None = None) -> ParetoSupportArchive:
    path = archive_path(path_or_root, scene_token) if scene_token is not None else Path(path_or_root)
    with lzma.open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, ParetoSupportArchive):
        return data
    if not isinstance(data, dict):
        raise TypeError(f"Archive payload must be a dict, got {type(data).__name__}: {path}")
    return archive_from_dict(data)


def should_skip_existing(root: str | Path, scene_token: str, config_hash: str, *, overwrite: bool = False) -> bool:
    if overwrite:
        return False
    path = archive_path(root, scene_token)
    if not path.is_file():
        return False
    try:
        archive = load_archive(path)
    except Exception:
        return False
    return str(archive.build_metadata.get("config_hash", "")) == str(config_hash)
