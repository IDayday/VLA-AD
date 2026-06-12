from __future__ import annotations

import hashlib
import lzma
import os
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np

REQUIRED_COMPONENT_KEYS = (
    "pdms",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "history_comfort",
    "lane_keeping",
    "driving_direction_compliance",
    "traffic_light_compliance",
)


def token_to_buffer_key(token: str) -> str:
    return hashlib.sha1(str(token).encode("utf-8")).hexdigest()


def _record_path(buffer_root: Path, token: str) -> Path:
    return Path(buffer_root) / f"{token_to_buffer_key(token)}.pkl.xz"


def _validate_record(record: Dict[str, Any]) -> None:
    version = int(record.get("version", 1))
    required = {
        "token",
        "candidates",
        "rewards",
        "components",
        "sources",
        "anchor_distance",
        "gt_reward",
        "il_reward",
        "best_reward",
        "best_source",
        "version",
    }
    if version >= 2:
        required.update(
            {
                "valid_mask",
                "selection_score",
                "best_raw_reward",
                "best_valid_reward",
                "best_selected_reward",
                "best_raw_source",
                "best_valid_source",
                "best_selected_source",
                "has_valid_candidate",
            }
        )
    missing = sorted(required.difference(record))
    if missing:
        raise KeyError(f"Elite buffer record is missing keys: {missing}")

    candidates = np.asarray(record["candidates"])
    rewards = np.asarray(record["rewards"])
    anchor_distance = np.asarray(record["anchor_distance"])
    if candidates.ndim != 3 or candidates.shape[-1] != 3:
        raise ValueError(f"record['candidates'] must have shape [K, H, 3], got {candidates.shape}.")
    if rewards.shape != (candidates.shape[0],):
        raise ValueError(f"record['rewards'] shape {rewards.shape} does not match K={candidates.shape[0]}.")
    if anchor_distance.shape != (candidates.shape[0],):
        raise ValueError(
            f"record['anchor_distance'] shape {anchor_distance.shape} does not match K={candidates.shape[0]}."
        )
    if len(record["sources"]) != candidates.shape[0]:
        raise ValueError(f"record['sources'] length {len(record['sources'])} does not match K={candidates.shape[0]}.")
    if version < 1:
        raise ValueError(f"record['version'] must be >= 1, got {version}.")
    if "valid_mask" in record:
        valid_mask = np.asarray(record["valid_mask"])
        if valid_mask.shape != (candidates.shape[0],):
            raise ValueError(f"record['valid_mask'] shape {valid_mask.shape} does not match K={candidates.shape[0]}.")
    if "selection_score" in record:
        selection_score = np.asarray(record["selection_score"])
        if selection_score.shape != (candidates.shape[0],):
            raise ValueError(
                f"record['selection_score'] shape {selection_score.shape} does not match K={candidates.shape[0]}."
            )

    components = record["components"]
    if not isinstance(components, dict):
        raise TypeError("record['components'] must be a dict.")
    missing_components = sorted(set(REQUIRED_COMPONENT_KEYS).difference(components))
    if missing_components:
        raise KeyError(f"Elite buffer record is missing component keys: {missing_components}")
    for key in REQUIRED_COMPONENT_KEYS:
        values = np.asarray(components[key])
        if values.shape != (candidates.shape[0],):
            raise ValueError(f"component {key!r} shape {values.shape} does not match K={candidates.shape[0]}.")


def save_elite_record(buffer_root: Path, token: str, record: Dict[str, Any]) -> None:
    buffer_root = Path(buffer_root)
    buffer_root.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload["token"] = str(token)
    payload["candidates"] = np.asarray(payload["candidates"], dtype=np.float32)
    payload["rewards"] = np.asarray(payload["rewards"], dtype=np.float32)
    payload["anchor_distance"] = np.asarray(payload["anchor_distance"], dtype=np.float32)
    payload["valid_mask"] = np.asarray(payload["valid_mask"], dtype=np.bool_)
    payload["selection_score"] = np.asarray(payload["selection_score"], dtype=np.float32)
    payload["components"] = {
        key: np.asarray(value, dtype=np.float32) for key, value in dict(payload["components"]).items()
    }
    payload["sources"] = [str(source) for source in payload["sources"]]
    payload["gt_reward"] = float(payload["gt_reward"])
    payload["il_reward"] = float(payload["il_reward"])
    payload["best_reward"] = float(payload["best_reward"])
    payload["best_source"] = str(payload["best_source"])
    payload["best_raw_reward"] = float(payload["best_raw_reward"])
    payload["best_valid_reward"] = float(payload["best_valid_reward"])
    payload["best_selected_reward"] = float(payload["best_selected_reward"])
    payload["best_raw_source"] = str(payload["best_raw_source"])
    payload["best_valid_source"] = str(payload["best_valid_source"])
    payload["best_selected_source"] = str(payload["best_selected_source"])
    payload["has_valid_candidate"] = bool(payload["has_valid_candidate"])
    payload["version"] = 2
    _validate_record(payload)
    path = _record_path(buffer_root, token)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with lzma.open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_elite_record(buffer_root: Path, token: str) -> Dict[str, Any]:
    path = _record_path(Path(buffer_root), token)
    if not path.is_file():
        raise FileNotFoundError(f"Elite buffer record not found for token={token!r}: {path}")
    with lzma.open(path, "rb") as f:
        record = pickle.load(f)
    if not isinstance(record, dict):
        raise TypeError(f"Elite buffer record must be a dict, got {type(record).__name__}: {path}")
    _validate_record(record)
    record_token = str(record["token"])
    if record_token != str(token):
        raise ValueError(
            f"Elite buffer record token mismatch for {path}: requested={token!r}, record={record_token!r}."
        )
    return record
