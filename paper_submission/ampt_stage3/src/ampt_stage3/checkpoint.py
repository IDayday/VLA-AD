"""Read-only integrity checks for a supplied ReCogDrive evaluation weight."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_artifact(path: str | Path) -> str:
    """Hash one file or a directory tree without embedding its local path."""

    source = Path(path).expanduser().resolve()
    if source.is_file():
        return sha256_file(source)
    if not source.is_dir():
        raise FileNotFoundError(f"evaluation artifact does not exist: {source}")
    files = sorted(item for item in source.rglob("*") if item.is_file())
    if not files:
        raise ValueError("evaluation artifact directory is empty")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(source).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CheckpointReport:
    artifact_id: str
    sha256: str
    state_keys: int
    action_head_keys: int
    checkpoint_kind: str


def inspect_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str = "",
) -> CheckpointReport:
    """Validate a ReCogDrive checkpoint without changing it."""

    import torch

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    digest = sha256_file(checkpoint_path)
    if expected_sha256 and digest != expected_sha256.lower():
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected {expected_sha256}, found {digest}"
        )
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    state: Any = payload.get("state_dict", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(state, Mapping) or not state:
        raise TypeError("checkpoint must contain a non-empty state_dict mapping")
    keys = [str(key) for key in state]
    action_keys = [key for key in keys if "action_head." in key or key.startswith("model.")]
    if not action_keys:
        raise ValueError("checkpoint has no recognizable ReCogDrive action-head weights")
    return CheckpointReport(
        artifact_id=f"weight-{digest[:12]}",
        sha256=digest,
        state_keys=len(keys),
        action_head_keys=len(action_keys),
        checkpoint_kind="recogdrive_policy",
    )


def write_checkpoint_report(report: CheckpointReport, output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
