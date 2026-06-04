from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import torch


CHUNK_VERSION = "recogdrive2b_expert768_chunk_v2"
CONTEXT_SCHEMA = {
    "jepa_context_tokens": (12, 1024),
    "jepa_target_tokens": (12, 1024),
    "vggt_context_tokens": (12, 2048),
    "vggt_target_tokens": (12, 2048),
    "vggt_geometry_tokens": (12, 2048),
    "vggt_geometry_target_tokens": (12, 2048),
    "vggt_depth_tokens": (12, 2048),
    "vggt_pointmap_tokens": (12, 2048),
    "vggt_camera_tokens": (12, 2048),
    "teacher_trajectory": (8, 3),
    "teacher_trajectory_norm": (8, 3),
}
LEGACY_CONTEXT_ALIASES = {
    "jepa_tokens": "jepa_context_tokens",
    "vggt_tokens": "vggt_context_tokens",
}


@dataclass(frozen=True)
class ExpertCacheMetadata:
    version: str = CHUNK_VERSION
    is_dummy: bool = False
    contains_vlm_hidden: bool = False
    contains_jepa: bool = True
    contains_vggt: bool = True
    contains_vggt_geometry: bool = False
    target_tokens_are_train_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "is_dummy": self.is_dummy,
            "contains_vlm_hidden": self.contains_vlm_hidden,
            "contains_jepa": self.contains_jepa,
            "contains_vggt": self.contains_vggt,
            "contains_vggt_geometry": self.contains_vggt_geometry,
            "target_tokens_are_train_only": self.target_tokens_are_train_only,
        }


def atomic_torch_save(payload: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, tmp_path)
    try:
        tmp_path.replace(path)
    except FileNotFoundError:
        if path.exists():
            return
        raise


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    try:
        tmp_path.replace(path)
    except FileNotFoundError:
        if path.exists():
            return
        raise


def load_metadata(cache_dir: str | Path) -> Dict[str, Any]:
    path = Path(cache_dir) / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Chunk cache metadata not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if not isinstance(metadata, dict):
        raise TypeError(f"Chunk cache metadata must be a JSON object, got {type(metadata).__name__}.")
    return metadata


def normalize_sample_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    for old_key, new_key in LEGACY_CONTEXT_ALIASES.items():
        if new_key not in payload and old_key in payload:
            payload[new_key] = payload[old_key]
    return payload


def validate_token_tensor(payload: Dict[str, Any], key: str, expected_shape: tuple[int, int]) -> torch.Tensor:
    if key not in payload:
        raise KeyError(f"Expert cache sample missing required key '{key}'.")
    value = payload[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expert cache key '{key}' must be a torch.Tensor, got {type(value).__name__}.")
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"Expert cache key '{key}' shape {tuple(value.shape)} != {expected_shape}.")
    if not torch.isfinite(value.float()).all():
        raise ValueError(f"Expert cache key '{key}' contains non-finite values.")
    return value.detach().cpu()


def validate_sample_payload(
    payload: Dict[str, Any],
    *,
    require_jepa: bool = True,
    require_vggt: bool = True,
    require_targets: bool = False,
    use_last_rd: bool = False,
    future_jepa_loss_weight: float = 0.0,
    vggt_geometry_loss_weight: float = 0.0,
    require_vggt_geometry: bool = False,
    allow_patch_geometry_fallback: bool = True,
    use_last_vla: bool = False,
    last_vla_stage: str = "disabled",
    last_vla_teacher_traj_mode: str = "none",
) -> Dict[str, Any]:
    payload = normalize_sample_payload(dict(payload))
    if require_jepa:
        validate_token_tensor(payload, "jepa_context_tokens", CONTEXT_SCHEMA["jepa_context_tokens"])
        if require_targets or (use_last_rd and future_jepa_loss_weight > 0.0):
            validate_token_tensor(payload, "jepa_target_tokens", CONTEXT_SCHEMA["jepa_target_tokens"])
    if require_vggt:
        validate_token_tensor(payload, "vggt_context_tokens", CONTEXT_SCHEMA["vggt_context_tokens"])
        if require_targets:
            validate_token_tensor(payload, "vggt_target_tokens", CONTEXT_SCHEMA["vggt_target_tokens"])
    if use_last_rd and require_vggt:
        has_geometry = any(key in payload for key in ("vggt_geometry_tokens", "vggt_depth_tokens", "vggt_pointmap_tokens", "vggt_camera_tokens"))
        if require_vggt_geometry and not has_geometry:
            raise KeyError("LaST-RD cache requires full VGGT geometry tokens but none were found.")
        if vggt_geometry_loss_weight > 0.0 and not allow_patch_geometry_fallback and "vggt_geometry_target_tokens" not in payload:
            raise KeyError("LaST-RD cache requires vggt_geometry_target_tokens because geometry fallback is disabled.")
        for key in ("vggt_geometry_tokens", "vggt_geometry_target_tokens", "vggt_depth_tokens", "vggt_pointmap_tokens", "vggt_camera_tokens"):
            if key in payload:
                validate_token_tensor(payload, key, CONTEXT_SCHEMA[key])
    if use_last_vla:
        for key in ("teacher_trajectory", "teacher_trajectory_norm"):
            if key in payload:
                validate_token_tensor(payload, key, CONTEXT_SCHEMA[key])
        if last_vla_stage == "teacher_traj_sft" and last_vla_teacher_traj_mode != "none":
            if "teacher_trajectory" not in payload and "teacher_trajectory_norm" not in payload:
                raise KeyError("Last-VLA teacher_traj_sft requires teacher_trajectory or teacher_trajectory_norm.")
        for key in ("teacher_score", "gt_score", "oracle_best_of_k_score", "candidate_count"):
            if key in payload:
                value = payload[key]
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"Expert cache key '{key}' must be a torch.Tensor, got {type(value).__name__}.")
                if value.ndim > 1:
                    raise ValueError(f"Expert cache key '{key}' must be scalar or [1], got {tuple(value.shape)}.")
                if not torch.isfinite(value.float()).all():
                    raise ValueError(f"Expert cache key '{key}' contains non-finite values.")
    if "last_hidden_state" in payload:
        value = payload["last_hidden_state"]
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[-1] != 1536:
            raise ValueError(
                "last_hidden_state must be a tensor with shape [Nv, 1536], "
                f"got {tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__}."
            )
    return payload


def _indexed_chunk_dirs(cache_dir: Path) -> list[Path]:
    if (cache_dir / "index.jsonl").is_file():
        return [cache_dir]
    if not cache_dir.is_dir():
        return []
    return sorted(
        child for child in cache_dir.iterdir()
        if child.is_dir() and (child / "index.jsonl").is_file()
    )


def find_indexed_sample_path(
    cache_dir: str | Path,
    sample_token: str,
    scene_token: Optional[str] = None,
) -> Optional[Path]:
    cache_dir = Path(cache_dir)
    for chunk_dir in _indexed_chunk_dirs(cache_dir):
        for record in iter_index(chunk_dir):
            path = Path(record.get("path", ""))
            token = str(record.get("sample_token") or path.stem)
            if token != str(sample_token):
                continue
            record_scene_token = record.get("scene_token")
            if scene_token and record_scene_token and str(record_scene_token) != str(scene_token):
                continue
            if path.is_file():
                return path
    return None


def sample_cache_path(cache_dir: str | Path, sample_token: str, scene_token: Optional[str] = None) -> Path:
    cache_dir = Path(cache_dir)
    candidates = []
    if scene_token:
        candidates.extend([
            cache_dir / "samples" / f"{scene_token}.pt",
            cache_dir / scene_token / f"{sample_token}.pt",
            cache_dir / scene_token / sample_token / "expert_features.pt",
        ])
    candidates.extend([
        cache_dir / "samples" / f"{sample_token}.pt",
        cache_dir / f"{sample_token}.pt",
        cache_dir / sample_token / "expert_features.pt",
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    indexed_path = find_indexed_sample_path(cache_dir, sample_token=sample_token, scene_token=scene_token)
    if indexed_path is not None:
        return indexed_path

    indexed_dirs = _indexed_chunk_dirs(cache_dir)
    raise FileNotFoundError(
        f"Expert chunk sample '{sample_token}' not found under {cache_dir}. Tried: "
        + ", ".join(str(path) for path in candidates)
        + (
            f". Also scanned {len(indexed_dirs)} indexed chunk dirs."
            if indexed_dirs else ""
        )
    )


def load_sample(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expert cache sample {path} must contain a dict, got {type(payload).__name__}.")
    return normalize_sample_payload(payload)


def iter_index(cache_dir: str | Path) -> Iterator[Dict[str, Any]]:
    index_path = Path(cache_dir) / "index.jsonl"
    if not index_path.is_file():
        for path in sorted(Path(cache_dir).glob("**/*.pt")):
            yield {"path": str(path), "sample_token": path.stem}
        return
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                if "path" in record and not Path(record["path"]).is_absolute():
                    record["path"] = str(Path(cache_dir) / record["path"])
                yield record


def write_index(cache_dir: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "index.jsonl"
    tmp_path = index_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True))
            f.write("\n")
    tmp_path.replace(index_path)
