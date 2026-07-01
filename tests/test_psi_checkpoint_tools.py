from __future__ import annotations

import csv
import hashlib
import os
import json
import subprocess
import sys
from pathlib import Path

from scripts.checkpoints import archive_checkpoint_immutable as archive_mod


REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = REPO_ROOT / "scripts" / "checkpoints" / "archive_checkpoint_immutable.py"
RANK = REPO_ROOT / "scripts" / "checkpoints" / "rank_eval_checkpoints.py"
VERIFY = REPO_ROOT / "scripts" / "checkpoints" / "verify_checkpoint_store.py"
UPDATE_EVAL = REPO_ROOT / "scripts" / "checkpoints" / "update_checkpoint_inventory_eval_metrics.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_archive_deduplicates_same_content_and_preserves_sources(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    a = tmp_path / "a.ckpt"
    b = tmp_path / "b.ckpt"
    a.write_bytes(b"same checkpoint")
    b.write_bytes(b"same checkpoint")

    for checkpoint in (a, b):
        subprocess.run(
            [sys.executable, str(ARCHIVE), "--run-root", str(run_root), "--checkpoint", str(checkpoint), "--stage", "stage2"],
            cwd=REPO_ROOT,
            check=True,
        )

    rows = _read_tsv(run_root / "checkpoint_store" / "inventory.tsv")
    assert len(rows) == 2
    assert rows[0]["sha256"] == rows[1]["sha256"] == _sha(a)
    objects = list((run_root / "checkpoint_store" / "objects").glob("*.ckpt"))
    assert len(objects) == 1
    assert a.is_file() and b.is_file()


def test_archive_cleans_temp_when_concurrent_hardlink_object_exists(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    ckpt = tmp_path / "epoch_001.ckpt"
    ckpt.write_bytes(b"same-inode race checkpoint")
    sha = _sha(ckpt)
    object_path = run_root / "checkpoint_store" / "objects" / f"{sha}.ckpt"

    def racing_hardlink_or_copy(src: Path, dst: Path) -> str:
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dst)
        os.link(src, object_path)
        return "hardlink"

    monkeypatch.setattr(archive_mod, "hardlink_or_copy", racing_hardlink_or_copy)

    archive_mod.archive_checkpoint(
        ckpt,
        run_root=run_root,
        stage="stage2",
        checkpoint_id=None,
        config_hash="",
        stable_seconds=0,
    )

    assert object_path.is_file()
    assert not list((run_root / "checkpoint_store" / "objects").glob(".*.tmp"))


def test_rank_marks_incomplete_when_fewer_than_top5(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    ckpt = tmp_path / "step_00000300.ckpt"
    ckpt.write_bytes(b"checkpoint")
    subprocess.run(
        [sys.executable, str(ARCHIVE), "--run-root", str(run_root), "--checkpoint", str(ckpt), "--stage", "stage3"],
        cwd=REPO_ROOT,
        check=True,
    )
    summary = tmp_path / "summary.tsv"
    summary.write_text(
        "checkpoint_id\tpdms_mean\tcore_mean\tnc_mean\tdac_mean\tttc_mean\tep_mean\n"
        "step_00000300\t0.91\t0.88\t1.0\t1.0\t0.9\t0.8\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(RANK),
            "--run-root",
            str(run_root),
            "--split",
            "val6000",
            "--eval-summary",
            str(summary),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    payload = json.loads((run_root / "rankings" / "val6000" / "current_top5.json").read_text())
    assert payload["complete"] is False
    assert len(payload["entries"]) == 1
    assert payload["entries"][0]["object_path"]
    subprocess.run([sys.executable, str(VERIFY), "--run-root", str(run_root)], cwd=REPO_ROOT, check=True)


def test_existing_object_hash_mismatch_fails_without_inventory(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    ckpt = tmp_path / "bad.ckpt"
    ckpt.write_bytes(b"real")
    sha = _sha(ckpt)
    object_dir = run_root / "checkpoint_store" / "objects"
    object_dir.mkdir(parents=True)
    (object_dir / f"{sha}.ckpt").write_bytes(b"corrupt")

    proc = subprocess.run(
        [sys.executable, str(ARCHIVE), "--run-root", str(run_root), "--checkpoint", str(ckpt), "--stage", "stage2"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert not (run_root / "checkpoint_store" / "inventory.tsv").exists()


def test_update_inventory_eval_metrics_from_summary(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    ckpt = tmp_path / "epoch_001.ckpt"
    ckpt.write_bytes(b"checkpoint for eval metrics")
    sha = _sha(ckpt)

    subprocess.run(
        [sys.executable, str(ARCHIVE), "--run-root", str(run_root), "--checkpoint", str(ckpt), "--stage", "stage2"],
        cwd=REPO_ROOT,
        check=True,
    )
    summary = tmp_path / "summary.tsv"
    summary.write_text(
        "checkpoint_id\teval_dir\tpdms_mean\tcore_mean\tnc_mean\tdac_mean\tttc_mean\tep_mean\n"
        f"{sha}\t/tmp/eval\t0.8123\t0.776\t1.0\t0.99\t0.88\t0.77\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(UPDATE_EVAL),
            "--run-root",
            str(run_root),
            "--split",
            "val6000",
            "--sha256",
            sha,
            "--state",
            "done",
            "--summary-tsv",
            str(summary),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    rows = _read_tsv(run_root / "checkpoint_store" / "inventory.tsv")
    assert len(rows) == 1
    assert rows[0]["val6000_state"] == "done"
    assert rows[0]["val6000_pdms"] == "0.8123"
    assert rows[0]["val6000_core"] == "0.776"
    assert rows[0]["val6000_nc"] == "1.0"
    assert rows[0]["val6000_dac"] == "0.99"
    assert rows[0]["val6000_ttc"] == "0.88"
    assert rows[0]["val6000_ep"] == "0.77"
    assert rows[0]["navtest_state"] == "pending"

    subprocess.run(
        [
            sys.executable,
            str(UPDATE_EVAL),
            "--run-root",
            str(run_root),
            "--split",
            "navtest",
            "--sha256",
            sha,
            "--state",
            "failed",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    rows = _read_tsv(run_root / "checkpoint_store" / "inventory.tsv")
    assert rows[0]["navtest_state"] == "failed"
    assert rows[0]["navtest_pdms"] == ""
    subprocess.run([sys.executable, str(VERIFY), "--run-root", str(run_root)], cwd=REPO_ROOT, check=True)
