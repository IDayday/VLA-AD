#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


CKPT_RE = re.compile(r"epoch=(?P<epoch>\d+)-step=(?P<step>\d+)\.ckpt$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuously monitor Lightning top-k checkpoint files.")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--monitor-root", type=Path, default=None)
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--eval-root", type=Path, default=None)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def ckpt_record(path: Path) -> Dict[str, Any]:
    st = path.stat()
    match = CKPT_RE.search(path.name)
    return {
        "path": str(path),
        "name": path.name,
        "epoch": int(match.group("epoch")) if match else None,
        "step": int(match.group("step")) if match else None,
        "size": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
    }


def collect_line(run_root: Path) -> List[Dict[str, Any]]:
    ckpts = [
        ckpt_record(path)
        for path in run_root.glob("lightning_logs/version_*/checkpoints/epoch=*-step=*.ckpt")
        if path.is_file()
    ]
    return sorted(ckpts, key=lambda item: (item["epoch"] or -1, item["step"] or -1, item["path"]))


def evaluated_runs(eval_root: Path | None) -> Dict[str, List[str]]:
    if eval_root is None:
        return {"A": [], "B": []}
    result: Dict[str, List[str]] = {"A": [], "B": []}
    for line in ("A", "B"):
        line_root = eval_root / "eval" / line
        if not line_root.is_dir():
            continue
        seen = set()
        for metrics in line_root.glob("**/metrics.json"):
            run_dir = metrics.parent.parent if metrics.parent.name.startswith("shard_") else metrics.parent
            if run_dir.parent != line_root:
                continue
            if run_dir.name in seen:
                continue
            seen.add(run_dir.name)
            result[line].append(run_dir.name)
        result[line].sort()
    return result


def state(args: argparse.Namespace) -> Dict[str, Any]:
    a_root = args.out_root / "A_frozen_vlm_stage2_progressive"
    b_root = args.out_root / "B_lora_stage2_progressive"
    return {
        "updated_at": utc_now(),
        "out_root": str(args.out_root),
        "eval_root": str(args.eval_root) if args.eval_root else None,
        "lines": {
            "A": {
                "run_root": str(a_root),
                "topk_count": len(collect_line(a_root)),
                "topk": collect_line(a_root),
            },
            "B": {
                "run_root": str(b_root),
                "topk_count": len(collect_line(b_root)),
                "topk": collect_line(b_root),
            },
        },
        "evaluated_runs": evaluated_runs(args.eval_root),
    }


def signature(payload: Dict[str, Any]) -> Dict[str, List[str]]:
    return {
        line: [item["path"] for item in payload["lines"][line]["topk"]]
        for line in ("A", "B")
    }


def main() -> int:
    args = parse_args()
    monitor_root = args.monitor_root or args.out_root / "top5_ckpt_monitor"
    monitor_root.mkdir(parents=True, exist_ok=True)
    current_path = monitor_root / "top5_current.json"
    history_path = monitor_root / "top5_history.jsonl"
    log_path = monitor_root / "top5_monitor.log"
    (monitor_root / "monitor.pid").write_text(f"{Path('/proc/self').resolve().name}\n", encoding="utf-8")

    previous: Dict[str, List[str]] | None = None
    while True:
        payload = state(args)
        sig = signature(payload)
        changed = previous != sig
        payload["changed"] = changed
        current_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if changed:
            with history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, sort_keys=True) + "\n")
            with log_path.open("a", encoding="utf-8") as f:
                counts = ", ".join(f"{line}={len(sig[line])}" for line in ("A", "B"))
                f.write(f"[{payload['updated_at']}] top-k changed: {counts}\n")
                for line in ("A", "B"):
                    names = ", ".join(Path(path).name for path in sig[line]) or "none"
                    f.write(f"  {line}: {names}\n")
        previous = sig
        if args.once:
            break
        time.sleep(max(args.poll_seconds, 1.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
