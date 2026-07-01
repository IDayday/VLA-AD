from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = REPO_ROOT / "scripts" / "psi_drive" / "summarize_psi_run.py"


def _write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_summarize_psi_run_handles_incomplete_eval_and_top5(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    (run_root / "checkpoints" / "raw").mkdir(parents=True)
    (run_root / "checkpoints" / "raw" / "epoch_001.ckpt").write_bytes(b"ckpt")
    (run_root / "data_report.json").write_text(
        json.dumps(
            {
                "loader_mode": "official-cache-loader-all-cache-train-log-val",
                "cache_train_all_records": True,
                "train_val_overlap_count": 10,
                "train": {
                    "num_records": 100,
                    "stage2_target_source": "pareto_support",
                    "stage2_pareto_support_index_records": 100,
                },
                "val": {"num_records": 10},
            }
        ),
        encoding="utf-8",
    )
    _write_tsv(
        run_root / "checkpoint_store" / "inventory.tsv",
        [
            {
                "checkpoint_id": "epoch_001",
                "sha256": "abc",
                "source_path": "/tmp/epoch_001.ckpt",
                "object_path": "/tmp/abc.ckpt",
                "size_bytes": "4",
                "mtime": "1",
                "stage": "stage2",
                "train_step": "",
                "epoch": "001",
                "config_hash": "",
                "git_commit": "commit",
                "val6000_state": "done",
                "navtest_state": "pending",
                "first_seen_at": "now",
                "archived_at": "now",
            }
        ],
        [
            "checkpoint_id",
            "sha256",
            "source_path",
            "object_path",
            "size_bytes",
            "mtime",
            "stage",
            "train_step",
            "epoch",
            "config_hash",
            "git_commit",
            "val6000_state",
            "navtest_state",
            "first_seen_at",
            "archived_at",
        ],
    )
    (run_root / "checkpoint_store" / "audit_latest.json").write_text(
        json.dumps({"passed": True, "failures": [], "inventory_rows": 1, "object_files": 1}),
        encoding="utf-8",
    )
    _write_tsv(
        run_root / "eval" / "val6000" / "checkpoint_eval_status.tsv",
        [{"timestamp": "now", "checkpoint_id": "epoch_001", "sha256": "abc", "state": "done"}],
        ["timestamp", "checkpoint_id", "sha256", "state"],
    )
    (run_root / "rankings" / "val6000").mkdir(parents=True)
    (run_root / "rankings" / "val6000" / "current_top5.json").write_text(
        json.dumps(
            {
                "complete": False,
                "num_ranked": 1,
                "entries": [
                    {
                        "checkpoint_id": "epoch_001",
                        "sha256": "abc",
                        "pdms": 0.9,
                        "object_path": "/tmp/abc.ckpt",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output_json = tmp_path / "summary.json"
    output_md = tmp_path / "summary.md"
    subprocess.run(
        [
            sys.executable,
            str(SUMMARY),
            "--run-root",
            str(run_root),
            "--stage",
            "stage2",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["raw_checkpoints"]["count"] == 1
    assert payload["inventory"]["rows"] == 1
    assert payload["eval_status"]["val6000"]["latest_state_counts"] == {"done": 1}
    assert payload["top5"]["val6000"]["complete"] is False
    assert payload["top5"]["val6000"]["pdms"]["mean"] == 0.9
    assert "Remaining Gates" in output_md.read_text(encoding="utf-8")
