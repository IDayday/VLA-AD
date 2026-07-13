#!/usr/bin/env python3
"""Gate a ReCogDrive Stage3 run on early checkpoint PDMS.

This script is intentionally read-only: it reports whether a run should keep
training, wait for evaluation, or be stopped by an operator. It does not kill
training processes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any


PDMS_FIELDS = ("pdms_mean", "seed_mean_pdms_mean", "pdms", "score", "pdm_score")
SUBMETRIC_FIELDS = ("nc_mean", "dac_mean", "ttc_mean", "ep_mean", "comfort_mean", "ddc_mean", "tlc_mean")


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _checkpoint_sort_key(checkpoint_id: str, row_index: int) -> tuple[int, int, int]:
    text = str(checkpoint_id)
    step_matches = [int(match) for match in re.findall(r"step[_=-]?(\d+)", text)]
    epoch_matches = [int(match) for match in re.findall(r"epoch[_=-]?(\d+)", text)]
    step = step_matches[-1] if step_matches else 10**12
    epoch = epoch_matches[-1] if epoch_matches else 10**9
    return step, epoch, row_index


def _extract_pdms(row: dict[str, str]) -> float | None:
    for field in PDMS_FIELDS:
        value = _float_or_none(row.get(field))
        if value is not None:
            return value
    return None


def _load_eval_rows(run_root: Path, extra_summary_tsv: list[Path]) -> list[dict[str, Any]]:
    summary_paths = sorted(run_root.rglob("checkpoint_eval_submetrics.tsv")) if run_root.exists() else []
    summary_paths.extend(extra_summary_tsv)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    row_index = 0
    for path in summary_paths:
        for row in _read_tsv(path):
            pdms = _extract_pdms(row)
            checkpoint_id = str(row.get("checkpoint_id") or "").strip()
            if pdms is None or not checkpoint_id:
                continue
            eval_dir = str(row.get("eval_dir") or "").strip()
            key = (checkpoint_id, eval_dir)
            if key in seen:
                continue
            seen.add(key)
            out: dict[str, Any] = {
                "checkpoint_id": checkpoint_id,
                "pdms": pdms,
                "summary_tsv": str(path),
                "eval_dir": eval_dir,
                "csv_path": row.get("csv_path", ""),
                "num_valid_rows": row.get("num_valid_rows", ""),
                "sort_key": _checkpoint_sort_key(checkpoint_id, row_index),
            }
            for field in SUBMETRIC_FIELDS:
                value = _float_or_none(row.get(field))
                if value is not None:
                    out[field.removesuffix("_mean")] = value
            rows.append(out)
            row_index += 1
    return sorted(rows, key=lambda item: item["sort_key"])


def _load_status(run_root: Path, status_file: Path | None) -> dict[str, Any]:
    candidates: list[Path] = []
    if status_file is not None:
        candidates.append(status_file)
    candidates.append(run_root / "status" / "stage3_rl_2b.json")
    candidates.extend(sorted((run_root / "status").glob("*.json")) if (run_root / "status").exists() else [])
    for path in candidates:
        try:
            return json.loads(path.read_text())
        except Exception:
            continue
    return {}


def _pgid_for_pid(pid: int | None) -> int | None:
    if pid is None or pid <= 0:
        return None
    try:
        return os.getpgid(pid)
    except OSError:
        return None


def _training_alive_from_status(status: dict[str, Any]) -> bool | None:
    if "alive" in status:
        return bool(status["alive"])
    state = str(status.get("state", "")).strip().lower()
    if state in {"running", "started"}:
        return True
    if state in {"done", "failed", "launch_failed", "stopped"}:
        return False
    return None


def _row_step(row: dict[str, Any]) -> int | None:
    sort_key = row.get("sort_key")
    if not isinstance(sort_key, tuple) or not sort_key:
        return None
    step = sort_key[0]
    if not isinstance(step, int) or step >= 10**12:
        return None
    return step


def decide(
    rows: list[dict[str, Any]],
    threshold: float,
    margin: float,
    use_best: bool,
    min_stop_step: int = 0,
) -> dict[str, Any]:
    if not rows:
        return {
            "action": "wait",
            "reason": "No evaluated checkpoint rows were found yet.",
            "evaluated_checkpoints": 0,
        }
    selected = max(rows, key=lambda row: row["pdms"]) if use_best else rows[0]
    pdms = float(selected["pdms"])
    if pdms >= threshold:
        action = "continue"
        reason = f"Selected checkpoint PDMS {pdms:.6f} is at or above threshold {threshold:.6f}."
    elif pdms < threshold - margin:
        selected_step = _row_step(selected)
        if min_stop_step > 0 and (selected_step is None or selected_step < min_stop_step):
            action = "watch"
            step_text = "unknown" if selected_step is None else str(selected_step)
            reason = (
                f"Selected checkpoint PDMS {pdms:.6f} is below threshold {threshold:.6f} "
                f"by more than margin {margin:.6f}, but checkpoint step {step_text} is "
                f"below min_stop_step {min_stop_step}; report only and wait for a later checkpoint."
            )
        else:
            action = "stop"
            reason = (
                f"Selected checkpoint PDMS {pdms:.6f} is below threshold {threshold:.6f} "
                f"by more than margin {margin:.6f}."
            )
    else:
        action = "watch"
        reason = (
            f"Selected checkpoint PDMS {pdms:.6f} is below threshold {threshold:.6f} "
            f"but within margin {margin:.6f}; wait for the next early checkpoint."
        )
    return {
        "action": action,
        "reason": reason,
        "evaluated_checkpoints": len(rows),
        "selected_checkpoint": selected,
        "selected_checkpoint_step": _row_step(selected),
        "best_checkpoint": max(rows, key=lambda row: row["pdms"]),
        "best_checkpoint_step": _row_step(max(rows, key=lambda row: row["pdms"])),
        "latest_checkpoint": rows[-1],
        "latest_checkpoint_step": _row_step(rows[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--summary-tsv", type=Path, action="append", default=[])
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--threshold", type=float, default=0.88)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument(
        "--use-best",
        action="store_true",
        help="Gate on best evaluated checkpoint instead of the earliest evaluated checkpoint.",
    )
    parser.add_argument(
        "--min-stop-step",
        type=int,
        default=0,
        help="Do not return action=stop for evaluated step checkpoints below this step. Use 0 to disable.",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    rows = _load_eval_rows(args.run_root, args.summary_tsv)
    result = decide(
        rows,
        threshold=args.threshold,
        margin=args.margin,
        use_best=args.use_best,
        min_stop_step=max(0, args.min_stop_step),
    )
    status = _load_status(args.run_root, args.status_file)
    pid = int(status.get("pid") or 0) if str(status.get("pid") or "").isdigit() else None
    pgid = _pgid_for_pid(pid)
    suggested_stop_command = f"kill -TERM -{pgid}" if result["action"] == "stop" and pgid is not None else ""
    result.update(
        {
            "run_root": str(args.run_root),
            "threshold": args.threshold,
            "margin": args.margin,
            "min_stop_step": max(0, args.min_stop_step),
            "gate_mode": "best" if args.use_best else "earliest",
            "training_state": status.get("state", ""),
            "training_alive": _training_alive_from_status(status),
            "train_pid": pid,
            "train_pgid": pgid,
            "suggested_stop_command": suggested_stop_command,
        }
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print(f"action={result['action']}")
    print(f"reason={result['reason']}")
    if result.get("selected_checkpoint"):
        selected = result["selected_checkpoint"]
        print(f"selected_checkpoint={selected['checkpoint_id']} pdms={selected['pdms']:.6f}")
        submetrics = " ".join(
            f"{field}={selected[field]:.6f}" for field in ("nc", "dac", "ttc", "ep", "comfort", "ddc", "tlc") if field in selected
        )
        if submetrics:
            print(submetrics)
    if result["action"] == "stop" and result["suggested_stop_command"]:
        print(f"suggested_stop_command={result['suggested_stop_command']}")


if __name__ == "__main__":
    main()
