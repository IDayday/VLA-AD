#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional


DEFAULT_OUT_ROOT = Path("/mnt/project/VLA-AD/outputs/last_vla_v2/no_residual_stage2_ab_top5val_latest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor Last-VLA v2 A/B stage2 training and top-5 eval progress.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--remote-host", default="training-rl-zt2")
    parser.add_argument("--watch", type=int, default=0, help="Refresh every N seconds when >0.")
    parser.add_argument("--lines", type=int, default=20)
    return parser.parse_args()


def run(cmd: List[str], timeout: int = 8) -> str:
    try:
        return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout).stdout.strip()
    except Exception as exc:
        return f"<error: {exc}>"


def remote(remote_host: str, script: str, timeout: int = 10) -> str:
    if not remote_host:
        return ""
    return run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "LogLevel=ERROR", remote_host, script],
        timeout=timeout,
    )


def tail(path: Path, n: int) -> List[str]:
    if not path.is_file():
        return []
    try:
        data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return [f"<read error: {exc}>"]
    return data[-n:]


def find_progress(lines: Iterable[str]) -> List[str]:
    patterns = (
        re.compile(r"Epoch\s+\d+"),
        re.compile(r"\btrain/loss\b|\bval/loss\b|global_step|it/s|Starting Training|Num training samples|Num validation samples"),
        re.compile(r"Error executing job|Traceback|Disk quota exceeded|CUDA out of memory", re.I),
    )
    matched = []
    for line in lines:
        if any(p.search(line) for p in patterns):
            matched.append(line)
    return matched[-8:]


def ckpt_rows(run_root: Path) -> List[str]:
    rows = []
    for path in sorted(run_root.glob("*.ckpt")):
        rows.append(fmt_file(path))
    for path in sorted(run_root.glob("lightning_logs/version_*/checkpoints/*.ckpt")):
        rows.append(fmt_file(path))
    return rows[-12:]


def fmt_file(path: Path) -> str:
    try:
        stat = path.stat()
    except OSError as exc:
        return f"{path.name}: <stat error {exc}>"
    gib = stat.st_size / 1024**3
    mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%m-%d %H:%M")
    return f"{path.name:34s} {gib:6.2f} GiB  {mtime}"


def topk_count(run_root: Path) -> int:
    return len(list(run_root.glob("lightning_logs/version_*/checkpoints/epoch=*-step=*.ckpt")))


def latest_metrics(eval_root: Path) -> List[str]:
    rows = []
    for metrics_path in sorted(eval_root.glob("*/*/metrics.json")):
        try:
            data = json.loads(metrics_path.read_text())
        except Exception:
            continue
        pdms = data.get("PDMS", data.get("pdm_score"))
        valid = data.get("num_pdm_valid")
        rows.append(f"{metrics_path.parent.parent.name}/{metrics_path.parent.name}: PDMS={pdms} valid={valid}")
    return rows[-20:]


def process_summary(out_root: Path, remote_host: str) -> List[str]:
    needle = str(out_root)
    ps_cmd = f"ps -eo pid,stat,etime,pcpu,pmem,cmd | grep -F {needle!r} | grep -v grep"
    local = summarize_ps(run(["bash", "-lc", ps_cmd]))
    remote_out = summarize_ps(remote(remote_host, ps_cmd))
    rows = ["local processes:", *local]
    if remote_host:
        rows.extend([f"remote processes ({remote_host}):", *remote_out])
    return rows


def gpu_summary(remote_host: str) -> List[str]:
    cmd = "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
    rows = ["local GPUs:", run(["bash", "-lc", cmd]) or "  unavailable"]
    if remote_host:
        rows.extend([f"remote GPUs ({remote_host}):", remote(remote_host, cmd) or "  unavailable"])
    return rows


def summarize_ps(text: str) -> List[str]:
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("<error:")]
    if not lines:
        return ["  none"]
    torchrun = sum("torchrun" in line for line in lines)
    training = sum("run_training_recogdrive.py" in line for line in lines)
    rows = [f"  total={len(lines)} torchrun={torchrun} training_like={training}"]
    for line in lines[:6]:
        rows.append("  " + line[:220])
    if len(lines) > 6:
        rows.append(f"  ... {len(lines) - 6} more")
    return rows


def render(args: argparse.Namespace) -> str:
    out_root = args.out_root
    a_root = out_root / "A_frozen_vlm_stage2_progressive"
    b_root = out_root / "B_lora_stage2_progressive"
    eval_root = out_root / "decoupled_highcap_no_risk_eval"
    parts: List[str] = []
    parts.append(f"Last-VLA v2 A/B Stage2 Monitor  {datetime.now().isoformat(timespec='seconds')}")
    parts.append(f"OUT_ROOT: {out_root}")
    parts.append("")
    parts.extend(process_summary(out_root, args.remote_host))
    parts.append("")
    parts.extend(gpu_summary(args.remote_host))
    parts.append("")
    for name, root in (("A", a_root), ("B", b_root)):
        parts.append(f"{name} run: {root}")
        parts.append(f"  exists={root.is_dir()} topk_ckpts={topk_count(root)} latest={fmt_file(root / 'latest.ckpt') if (root / 'latest.ckpt').exists() else 'missing'}")
        ckpts = ckpt_rows(root)
        parts.extend([f"  {row}" for row in ckpts] or ["  no checkpoints yet"])
        log_lines: List[str] = []
        log_lines.extend(tail(root / "run_training_recogdrive.log", args.lines))
        log_lines.extend(tail(out_root / "logs" / ("A_stage2_local.log" if name == "A" else "B_stage2_remote.log"), args.lines))
        progress = find_progress(log_lines)
        parts.extend(["  progress:"] + [f"    {line}" for line in progress] if progress else ["  progress: no matching log lines yet"])
        parts.append("")
    watcher_log = out_root / "logs" / "ab_top5_eval_watcher.log"
    parts.append("Top-5 eval watcher:")
    parts.extend([f"  {line}" for line in tail(watcher_log, 8)] or ["  no watcher log yet"])
    parts.append("")
    parts.append("Completed eval metrics:")
    parts.extend([f"  {row}" for row in latest_metrics(eval_root)] or ["  no completed metrics yet"])
    parts.append("")
    parts.append("Disk:")
    parts.append(run(["bash", "-lc", "df -h /mnt/project /mnt/navsim 2>/dev/null || true"]))
    return "\n".join(parts)


def main() -> int:
    args = parse_args()
    while True:
        if args.watch > 0:
            print("\033[2J\033[H", end="")
        print(render(args), flush=True)
        if args.watch <= 0:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
