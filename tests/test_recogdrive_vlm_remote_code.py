from __future__ import annotations

from pathlib import Path

import pytest

from scripts.bench2drive.materialize_recogdrive_vlm_remote_code import (
    REMOTE_CODE_FILES,
    materialize,
)


def _checkpoint(root: Path) -> Path:
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"weights")
    (root / "config.json").write_text("{}\n", encoding="utf-8")
    return root


def _source(root: Path) -> Path:
    root.mkdir()
    for name in REMOTE_CODE_FILES:
        (root / name).write_text(f"# {name}\n", encoding="utf-8")
    return root


def test_materialize_makes_trainer_checkpoint_self_contained(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    source = _source(tmp_path / "source")
    report = materialize(checkpoint, source)
    assert report["copied"] == list(REMOTE_CODE_FILES)
    assert all((checkpoint / name).read_bytes() == (source / name).read_bytes() for name in REMOTE_CODE_FILES)

    second = materialize(checkpoint, source)
    assert second["copied"] == []


def test_materialize_refuses_different_existing_remote_code(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    source = _source(tmp_path / "source")
    (checkpoint / REMOTE_CODE_FILES[0]).write_text("# local modification\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Refusing to replace"):
        materialize(checkpoint, source)
