#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair NAVSIM sensor files from audit report alternate candidates.")
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def iter_issues(report: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for kind, key in (("image", "bad_images"), ("lidar", "bad_lidar"), ("log", "bad_logs")):
        for issue in report.get(key, []) or []:
            item = dict(issue)
            item["kind"] = kind
            yield item


def pick_source(issue: Dict[str, Any]) -> Optional[Path]:
    for candidate in issue.get("alternate_candidates", []) or []:
        if candidate.get("ok") and candidate.get("path"):
            return Path(candidate["path"])
    return None


def backup_existing(target: Path, backup_dir: Path) -> Optional[Path]:
    if not target.exists():
        return None
    try:
        rel = target.relative_to("/")
    except ValueError:
        rel = Path(str(target).lstrip("/"))
    backup_path = backup_dir / rel
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if backup_path.exists():
        suffix = f".{int(time.time())}"
        backup_path = backup_path.with_name(backup_path.name + suffix)
    shutil.copy2(target, backup_path)
    return backup_path


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp-repair-{os.getpid()}")
    if tmp.exists():
        tmp.unlink()
    shutil.copy2(source, tmp)
    tmp.replace(target)


def main() -> int:
    args = parse_args()
    report = json.load(args.audit_json.open("r", encoding="utf-8"))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.backup_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for issue in iter_issues(report):
        target_value = issue.get("path")
        source = pick_source(issue)
        status = "missing_target_path"
        backup_path: Optional[Path] = None
        if target_value and source is None:
            status = "no_ok_alternate"
        elif target_value and source is not None:
            target = Path(target_value)
            if not source.is_file():
                status = "source_not_found"
            elif args.dry_run:
                status = "dry_run"
            else:
                try:
                    if target.exists() and issue.get("reason") != "missing_file":
                        backup_path = backup_existing(target, args.backup_dir)
                    atomic_copy(source, target)
                    status = "copied"
                except Exception as exc:
                    status = "copy_failed"
                    issue["copy_error_type"] = type(exc).__name__
                    issue["copy_error"] = str(exc)[:500]
        counts[status] = counts.get(status, 0) + 1
        rows.append(
            {
                "kind": issue.get("kind"),
                "reason": issue.get("reason"),
                "log_name": issue.get("log_name"),
                "split": issue.get("split"),
                "target": target_value,
                "source": str(source) if source is not None else None,
                "backup": str(backup_path) if backup_path is not None else None,
                "status": status,
            }
        )

    output = {
        "audit_json": str(args.audit_json),
        "backup_dir": str(args.backup_dir),
        "dry_run": bool(args.dry_run),
        "counts": counts,
        "rows": rows,
    }
    args.out_json.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"counts": counts, "out_json": str(args.out_json)}, indent=2, sort_keys=True))
    return 0 if set(counts) <= {"copied", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
