#!/usr/bin/env python3
"""Summarize a ReCogDrive Stage3 GRPO run and its checkpoint eval watchers."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Iterable

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_OUTPUTS_ROOT = Path("/mnt/project/VLA-AD/outputs")
DEFAULT_WATCHERS = (
    "unique_lock_watch_on_vla_zt2_4gpu",
    "secondary_watch_on_rl_zt3_memfit_4gpu",
)

SCALAR_TAGS = (
    "epoch",
    "train/reward_step",
    "train/base_reward_step",
    "train/shaped_reward_step",
    "train/loss_step",
    "train/policy_loss_step",
    "train/bc_loss_step",
    "train/bc_coeff_step",
    "train/reference_kl_loss_step",
    "train/reference_kl_coeff_step",
    "train/reference_kl_chunk_size_step",
    "train/safe_ratio_step",
    "train/hard_safe_ratio_step",
    "train/mean_ep_step",
    "train/mean_ttc_step",
    "train/mean_comfort_step",
    "train/group_reward_std_step",
    "train/mixed_group_ratio_step",
    "train/all_safe_group_ratio_step",
    "train/all_unsafe_group_ratio_step",
    "train/mean_advantage_step",
    "train/mean_abs_advantage_step",
    "train/gspo_ratio_mean_step",
    "train/gspo_ratio_clip_frac_step",
    "lr-AdamW",
)


def _utc(ts: float | int | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if ts is None else ts))


def _tail(path: Path, lines: int) -> list[str]:
    if not path.exists():
        return []
    data = path.read_text(errors="replace").splitlines()
    return data[-lines:]


def _latest_file(paths: Iterable[Path]) -> Path | None:
    candidates = [p for p in paths if p.exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_status(run_root: Path) -> dict:
    status = run_root / "status" / "stage3_rl_2b.json"
    if not status.exists():
        status = _latest_file((run_root / "status").glob("*.json"))
    if status is None or not status.exists():
        return {}
    try:
        return json.loads(status.read_text())
    except Exception as exc:
        return {"state": "unreadable", "error": f"{type(exc).__name__}: {exc}"}


def _load_event_scalars(run_root: Path) -> tuple[Path | None, dict[str, list]]:
    event_file = _latest_file(run_root.glob("**/events.out.tfevents.*"))
    if event_file is None:
        return None, {}
    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 100000})
    accumulator.Reload()
    tags = set(accumulator.Tags().get("scalars", []))
    scalars = {}
    for tag in SCALAR_TAGS:
        if tag in tags:
            scalars[tag] = accumulator.Scalars(tag)
    return event_file, scalars


def _format_recent(values: list, limit: int = 5) -> str:
    items = []
    for event in values[-limit:]:
        items.append(f"{event.step}:{event.value:.6g}")
    return ", ".join(items)


def _last_scalar_value(scalars: dict[str, list], tag: str) -> float | None:
    values = scalars.get(tag)
    if not values:
        return None
    return float(values[-1].value)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _print_status(run_root: Path, epoch_step_target: int) -> None:
    print(f"now: {_utc()}")
    print(f"run_root: {run_root}")

    status = _load_status(run_root)
    if status:
        print(f"training_state: {status.get('state', 'unknown')} alive={status.get('alive', 'unknown')}")
        print(f"pid: {status.get('pid', '-')}")
    else:
        print("training_state: missing")

    ckpts = sorted(run_root.glob("train/**/*.ckpt"), key=lambda p: p.stat().st_mtime)
    print(f"checkpoints: {len(ckpts)}")
    for ckpt in ckpts[-5:]:
        print(f"  {_utc(ckpt.stat().st_mtime)} {ckpt.stat().st_size} {ckpt}")

    event_file, scalars = _load_event_scalars(run_root)
    if event_file is None:
        print("event_file: missing")
        return

    print(f"event_file: {event_file}")
    print(f"event_age_sec: {time.time() - event_file.stat().st_mtime:.1f}")
    print("scalars:")
    for tag in SCALAR_TAGS:
        values = scalars.get(tag)
        if not values:
            continue
        last = values[-1]
        print(f"  {tag}: step={last.step} value={last.value:.8g} n={len(values)} recent=[{_format_recent(values)}]")

    reward_values = scalars.get("train/reward_step", [])
    if len(reward_values) >= 2:
        prev, last = reward_values[-2], reward_values[-1]
        step_delta = max(1, last.step - prev.step)
        sec_per_step = (last.wall_time - prev.wall_time) / step_delta
        remaining_steps = max(0, epoch_step_target - last.step)
        eta = time.time() + remaining_steps * sec_per_step
        print(
            "eta: "
            f"target_step={epoch_step_target} sec_per_step={sec_per_step:.3f} "
            f"remaining_hours={remaining_steps * sec_per_step / 3600:.3f} eta_utc={_utc(eta)}"
        )

    if scalars.get("train/reward_step"):
        rewards = [event.value for event in scalars["train/reward_step"][-5:]]
        print(f"diagnostic_reward_recent_mean: {sum(rewards) / len(rewards):.6f}")
    if scalars.get("train/mixed_group_ratio_step"):
        mixed = _last_scalar_value(scalars, "train/mixed_group_ratio_step")
        all_unsafe = _last_scalar_value(scalars, "train/all_unsafe_group_ratio_step")
        group_std = _last_scalar_value(scalars, "train/group_reward_std_step")
        print(
            "diagnostic_group_signal: "
            f"mixed={mixed if mixed is not None else 'missing'} "
            f"all_unsafe={all_unsafe if all_unsafe is not None else 'missing'} "
            f"group_reward_std={group_std if group_std is not None else 'missing'}"
        )


def _print_watcher(run_root: Path, watcher_name: str, tail_lines: int) -> None:
    watcher_dir = run_root / watcher_name
    print(f"\nwatcher: {watcher_name}")
    if not watcher_dir.exists():
        print("  missing")
        return

    pid_path = watcher_dir / "watcher.pid"
    print(f"  pid_file: {pid_path.read_text().strip() if pid_path.exists() else '-'}")
    for line in _tail(watcher_dir / "watcher.log", tail_lines):
        print(f"  log: {line}")

    summary_rows = _read_tsv(watcher_dir / "checkpoint_eval_summary.tsv")
    print(f"  summary_rows: {len(summary_rows)}")
    for row in summary_rows[-5:]:
        print(
            "  summary: "
            f"{row.get('timestamp', '-')} {row.get('checkpoint_id', '-')} "
            f"{row.get('state', '-')} rc={row.get('rc', '-')}"
        )

    submetric_rows = _read_tsv(watcher_dir / "checkpoint_eval_submetrics.tsv")
    print(f"  submetric_rows: {len(submetric_rows)}")
    for row in submetric_rows[-5:]:
        pdms = row.get("pdms_mean") or row.get("pdms") or row.get("score") or "-"
        print(
            "  submetric: "
            f"{row.get('checkpoint_id', '-')} pdms={pdms} "
            f"csv={row.get('csv_path', row.get('csv', '-'))}"
        )

    best = watcher_dir / "best_checkpoint_by_pdms.txt"
    if best.exists():
        print("  best:")
        for line in best.read_text(errors="replace").splitlines():
            print(f"    {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    run_group = parser.add_mutually_exclusive_group(required=True)
    run_group.add_argument("--run-root", type=Path, help="Absolute output root for the Stage3 run.")
    run_group.add_argument("--run-name", help="Run directory name under --outputs-root.")
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--epoch-step-target", type=int, default=1330)
    parser.add_argument("--watcher", action="append", default=None)
    parser.add_argument("--watcher-log-lines", type=int, default=8)
    args = parser.parse_args()

    run_root = args.run_root if args.run_root is not None else args.outputs_root / args.run_name
    if not run_root.exists():
        raise SystemExit(f"Run root does not exist: {run_root}")

    _print_status(run_root, args.epoch_step_target)
    watcher_names = args.watcher if args.watcher is not None else list(DEFAULT_WATCHERS)
    for watcher_name in watcher_names:
        _print_watcher(run_root, watcher_name, args.watcher_log_lines)


if __name__ == "__main__":
    main()
