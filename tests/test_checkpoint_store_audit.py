from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from scripts.checkpoints.audit_checkpoint_store import audit_checkpoint_store


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_checkpoint_store_audit_passes_for_inventory_and_top5(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    object_dir = run_root / "checkpoint_store" / "objects"
    object_dir.mkdir(parents=True)
    obj = object_dir / "placeholder.ckpt"
    obj.write_bytes(b"checkpoint-bytes")
    sha = _sha(obj)
    obj = obj.rename(object_dir / f"{sha}.ckpt")
    row = {
        "checkpoint_id": "epoch_001",
        "sha256": sha,
        "source_path": str(run_root / "checkpoints" / "raw" / "epoch_001.ckpt"),
        "object_path": str(obj),
        "size_bytes": str(obj.stat().st_size),
    }
    _write_tsv(run_root / "checkpoint_store" / "inventory.tsv", [row], list(row))
    top5_row = {
        "rank": "1",
        "checkpoint_id": "epoch_001",
        "sha256": sha,
        "object_path": str(obj),
        "source_path": row["source_path"],
        "pdms": "0.9",
    }
    _write_tsv(run_root / "rankings" / "val6000" / "current_top5.tsv", [top5_row], list(top5_row))

    payload = audit_checkpoint_store(run_root, ["val6000", "navtest"])

    assert payload["passed"] is True
    assert payload["inventory_rows"] == 1
    assert payload["top5_entries"] == 1
    assert payload["failures"] == []


def test_checkpoint_store_audit_reports_temp_files_without_failing(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    object_dir = run_root / "checkpoint_store" / "objects"
    object_dir.mkdir(parents=True)
    obj = object_dir / "object.ckpt"
    obj.write_bytes(b"checkpoint-bytes")
    sha = _sha(obj)
    obj = obj.rename(object_dir / f"{sha}.ckpt")
    temp = object_dir / f".{sha}.123.tmp"
    temp.write_bytes(b"partial-or-stale-temp")
    row = {
        "checkpoint_id": "epoch_001",
        "sha256": sha,
        "source_path": str(run_root / "checkpoints" / "raw" / "epoch_001.ckpt"),
        "object_path": str(obj),
        "size_bytes": str(obj.stat().st_size),
    }
    _write_tsv(run_root / "checkpoint_store" / "inventory.tsv", [row], list(row))

    payload = audit_checkpoint_store(run_root, ["val6000"])

    assert payload["passed"] is True
    assert payload["temp_files"] == 1
    assert payload["temp_file_paths"] == [str(temp)]
    assert payload["failures"] == []
    assert any("temp_file_present" in warning for warning in payload["warnings"])


def test_checkpoint_store_audit_rejects_top5_symlink(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    object_dir = run_root / "checkpoint_store" / "objects"
    object_dir.mkdir(parents=True)
    obj = object_dir / "object.ckpt"
    obj.write_bytes(b"checkpoint-bytes")
    sha = _sha(obj)
    obj = obj.rename(object_dir / f"{sha}.ckpt")
    row = {
        "checkpoint_id": "epoch_001",
        "sha256": sha,
        "source_path": str(run_root / "checkpoints" / "raw" / "epoch_001.ckpt"),
        "object_path": str(obj),
        "size_bytes": str(obj.stat().st_size),
    }
    _write_tsv(run_root / "checkpoint_store" / "inventory.tsv", [row], list(row))
    link = run_root / "rankings" / "val6000" / "linked.ckpt"
    link.parent.mkdir(parents=True)
    link.symlink_to(obj)
    top5_row = {
        "rank": "1",
        "checkpoint_id": "epoch_001",
        "sha256": sha,
        "object_path": str(link),
        "source_path": row["source_path"],
        "pdms": "0.9",
    }
    _write_tsv(run_root / "rankings" / "val6000" / "current_top5.tsv", [top5_row], list(top5_row))

    payload = audit_checkpoint_store(run_root, ["val6000"])

    assert payload["passed"] is False
    assert any("object_path_is_symlink" in failure for failure in payload["failures"])
