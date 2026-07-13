#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


CKPT_RE = re.compile(r"epoch[=_](\d+)-step[=_](\d+)\.ckpt")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_name(text: str) -> str:
    chars = []
    for char in text:
        chars.append(char if char.isalnum() or char in ("-", "_", ".") else "_")
    return "".join(chars).strip("._") or "item"


def normalize_ckpt_name(text: str) -> str | None:
    match = CKPT_RE.search(text)
    if not match:
        return None
    return f"epoch={match.group(1)}-step={match.group(2)}.ckpt"


def read_val_rows(val_roots: Sequence[Path]) -> List[Dict[str, object]]:
    by_name: Dict[str, Dict[str, object]] = {}
    for root in val_roots:
        path = root / "summary" / "val6000_summary.csv"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                name = row.get("checkpoint_name") or normalize_ckpt_name(row.get("checkpoint") or "")
                if not name:
                    continue
                try:
                    pdms = float(row.get("PDMS") or "nan")
                except ValueError:
                    continue
                if pdms != pdms:
                    continue
                payload = dict(row)
                payload["checkpoint_name"] = name
                payload["PDMS_float"] = pdms
                current = by_name.get(name)
                if current is None or pdms > float(current["PDMS_float"]):
                    by_name[name] = payload
    return sorted(by_name.values(), key=lambda row: float(row["PDMS_float"]), reverse=True)


def read_navtest_rows(navtest_out_root: Path) -> List[Dict[str, object]]:
    by_name: Dict[str, Dict[str, object]] = {}
    path = navtest_out_root / "summary" / "navtest_summary.csv"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("checkpoint_name") or normalize_ckpt_name(row.get("checkpoint") or "")
            if not name:
                continue
            try:
                pdms = float(row.get("PDMS") or "nan")
            except ValueError:
                continue
            if pdms != pdms:
                continue
            payload = dict(row)
            payload["checkpoint_name"] = name
            payload["PDMS_float"] = pdms
            current = by_name.get(name)
            if current is None or pdms > float(current["PDMS_float"]):
                by_name[name] = payload
    return sorted(by_name.values(), key=lambda row: float(row["PDMS_float"]), reverse=True)


def host_ps(host: str) -> List[Tuple[int, str]]:
    command = "ps -eo pid=,cmd= | grep -E 'eval_recogdrive_expert_pdm|stable_gpu_launcher' | grep -v grep || true"
    if host == "local":
        proc = subprocess.run(["bash", "-lc", command], text=True, capture_output=True)
    else:
        proc = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, command],
            text=True,
            capture_output=True,
        )
    rows: List[Tuple[int, str]] = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        rows.append((int(parts[0]), parts[1]))
    return rows


def _command_in_scope(args: argparse.Namespace, cmd: str) -> bool:
    roots = [str(root) for root in args.val_roots]
    roots.append(str(args.navtest_out_root))
    return any(root in cmd for root in roots)


def active_by_split(args: argparse.Namespace) -> Tuple[Set[str], Set[str], Dict[str, List[int]]]:
    active_val: Set[str] = set()
    active_navtest: Set[str] = set()
    navtest_pids_by_name: Dict[str, List[int]] = {}
    for host in args.remote_hosts:
        for pid, cmd in host_ps(host):
            if not _command_in_scope(args, cmd):
                continue
            name = normalize_ckpt_name(cmd)
            if not name:
                continue
            if "--split val6000" in cmd:
                active_val.add(name)
            if "--split navtest" in cmd or "/navtest/" in cmd:
                active_navtest.add(name)
                navtest_pids_by_name.setdefault(f"{host}:{name}", []).append(pid)
    return active_val, active_navtest, navtest_pids_by_name


def kill_pids(host: str, pids: Iterable[int]) -> None:
    ids = sorted({int(pid) for pid in pids})
    if not ids:
        return
    joined = " ".join(str(pid) for pid in ids)
    script = f"kill -TERM {joined} 2>/dev/null || true; sleep 5; kill -KILL {joined} 2>/dev/null || true"
    if host == "local":
        subprocess.run(["bash", "-lc", script])
    else:
        subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, script])


def append_eliminated(path: Path, names: Iterable[str]) -> None:
    existing: Set[str] = set()
    if path.is_file():
        existing = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    merged = sorted(existing | set(names))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(merged) + ("\n" if merged else ""), encoding="utf-8")


def read_protected_names(path: Path | None) -> Set[str]:
    if path is None or not path.is_file():
        return set()
    names: Set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        names.add(normalize_ckpt_name(text) or text)
    return names


def remove_path(path: Path, removed: List[str]) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        removed.append(str(path))
    elif path.exists():
        path.unlink()
        removed.append(str(path))


def prune_artifacts(args: argparse.Namespace, eliminated: Set[str], protected: Set[str]) -> List[str]:
    removed: List[str] = []
    for name in sorted(eliminated):
        if name in protected:
            continue
        key = safe_name(name)
        if args.ckpt_mirror_root is not None:
            remove_path(args.ckpt_mirror_root / name, removed)
        remove_path(args.navtest_out_root / "checkpoint_snapshots" / key, removed)
        run_dir = args.navtest_out_root / "navtest" / key
        if run_dir.is_dir() and not (run_dir / "metrics.json").is_file():
            remove_path(run_dir, removed)
        for val_root in args.val_roots:
            snapshot_root = val_root / "checkpoint_snapshots"
            if snapshot_root.is_dir():
                for path in snapshot_root.glob(f"*{key}*"):
                    remove_path(path, removed)
    return removed


def log(args: argparse.Namespace, message: str) -> None:
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_once(args: argparse.Namespace) -> None:
    ranked = read_val_rows(args.val_roots)
    navtest_ranked = read_navtest_rows(args.navtest_out_root)
    top = {str(row["checkpoint_name"]) for row in ranked[: args.top_k]}
    protected_navtest = {
        str(row["checkpoint_name"])
        for row in navtest_ranked[: max(args.protect_navtest_top_k, 0)]
    }
    protected = top | protected_navtest | read_protected_names(args.protect_names_file)
    completed = {str(row["checkpoint_name"]) for row in ranked}
    active_val, active_navtest, navtest_pids = active_by_split(args)
    eliminated = completed - protected - active_val
    append_eliminated(args.eliminated_file, eliminated)

    killed: List[str] = []
    for key, pids in navtest_pids.items():
        host, name = key.split(":", 1)
        if name in eliminated:
            kill_pids(host, pids)
            killed.append(f"{host}:{name}:{','.join(str(pid) for pid in sorted(set(pids)))}")

    removed = prune_artifacts(args, eliminated, protected)
    top_text = ", ".join(f"{row['checkpoint_name']}={float(row['PDMS_float']):.6f}" for row in ranked[: args.top_k])
    navtest_top_text = ", ".join(
        f"{row['checkpoint_name']}={float(row['PDMS_float']):.6f}"
        for row in navtest_ranked[: args.protect_navtest_top_k]
    )
    log(
        args,
        "top="
        + (top_text or "none")
        + " protected_navtest="
        + (navtest_top_text or "none")
        + f" completed={len(completed)} active_val={sorted(active_val)} active_navtest={sorted(active_navtest)}"
        + f" eliminated={sorted(eliminated)} killed={killed} removed={len(removed)}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune eval checkpoint copies that have fallen out of current val top-k.")
    parser.add_argument("--val-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--navtest-out-root", type=Path, required=True)
    parser.add_argument("--ckpt-mirror-root", type=Path, default=None)
    parser.add_argument("--eliminated-file", type=Path, required=True)
    parser.add_argument("--remote-hosts", nargs="+", default=["local"])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--protect-navtest-top-k", type=int, default=3)
    parser.add_argument("--protect-names-file", type=Path, default=None)
    parser.add_argument("--poll-seconds", type=float, default=120.0)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    while True:
        run_once(args)
        if args.once:
            return 0
        time.sleep(max(args.poll_seconds, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())
