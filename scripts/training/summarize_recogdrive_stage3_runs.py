#!/usr/bin/env python3
"""Summarize ReCogDrive Stage3 training/eval runs across output folders."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Iterable

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception:  # pragma: no cover - optional in lightweight envs
    EventAccumulator = None


DEFAULT_OUTPUTS_ROOT = Path("/mnt/project/VLA-AD/outputs")
DEFAULT_RUN_GLOBS = (
    "stage3_*",
    "*stage3*",
)
TRAIN_SCALAR_TAGS = (
    "train/reward_step",
    "train/base_reward_step",
    "train/shaped_reward_step",
    "train/safe_ratio_step",
    "train/mean_ep_step",
    "train/mean_ttc_step",
    "train/mean_comfort_step",
    "train/group_reward_std_step",
    "train/grpo_self_imitation_candidate_ratio_step",
    "train/grpo_self_imitation_safety_candidate_ratio_step",
    "train/grpo_self_imitation_nc_pass_ratio_step",
    "train/grpo_self_imitation_dac_pass_ratio_step",
    "train/grpo_self_imitation_ttc_pass_ratio_step",
    "train/grpo_self_imitation_ddc_pass_ratio_step",
    "train/grpo_self_imitation_pre_cap_target_ratio_step",
    "train/grpo_self_imitation_target_ratio_step",
    "train/grpo_self_imitation_target_scene_cap_ratio_step",
    "train/grpo_self_imitation_target_scene_cap_active_step",
    "train/grpo_self_imitation_target_reward_mean_step",
    "train/grpo_self_imitation_target_margin_mean_step",
    "train/grpo_self_imitation_target_nc_mean_step",
    "train/grpo_self_imitation_target_dac_mean_step",
    "train/grpo_self_imitation_target_ttc_mean_step",
    "train/grpo_self_imitation_target_ep_mean_step",
    "train/grpo_self_imitation_target_comfort_mean_step",
    "train/grpo_self_imitation_target_ddc_mean_step",
    "train/mixed_group_ratio_step",
    "train/all_safe_group_ratio_step",
    "train/all_unsafe_group_ratio_step",
    "train/reference_kl_loss_step",
    "train/reference_kl_coeff_step",
    "train/bc_coeff_step",
    "train/use_gspo_ratio_step",
    "train/sampled_from_behavior_policy_step",
    "lr-AdamW",
)
EVAL_PDMS_FIELDS = (
    "pdms_mean",
    "pdms",
    "score",
    "pdm_score",
    "seed_mean_pdms_mean",
)
METRIC_ALIASES = {
    "pdms": ("pdms", "pdm_score", "score", "final_score"),
    "nc": ("nc", "no_at_fault_collisions"),
    "dac": ("dac", "drivable_area_compliance", "drivable_area"),
    "ttc": ("ttc", "time_to_collision_within_bound", "time_to_collision"),
    "ep": ("ep", "ego_progress", "progress"),
    "comfort": ("comfort", "history_comfort", "comfortable"),
    "ddc": ("ddc", "driving_direction_compliance"),
    "tlc": ("tlc", "traffic_light_compliance"),
}
SUBMETRIC_FIELDS = (
    "nc_mean",
    "dac_mean",
    "ttc_mean",
    "ep_mean",
    "comfort_mean",
    "ddc_mean",
    "tlc_mean",
)
OUTPUT_FIELDS = (
    "run_name",
    "run_root",
    "mtime_utc",
    "training_state",
    "training_alive",
    "train_pid",
    "checkpoint_count",
    "latest_checkpoint",
    "latest_checkpoint_utc",
    "event_file",
    "event_age_sec",
    "latest_step",
    "train_reward",
    "base_reward",
    "shaped_reward",
    "safe_ratio",
    "mean_ep",
    "mean_ttc",
    "mean_comfort",
    "group_reward_std",
    "grpo_self_imitation_candidate_ratio",
    "grpo_self_imitation_safety_candidate_ratio",
    "grpo_self_imitation_nc_pass_ratio",
    "grpo_self_imitation_dac_pass_ratio",
    "grpo_self_imitation_ttc_pass_ratio",
    "grpo_self_imitation_ddc_pass_ratio",
    "grpo_self_imitation_pre_cap_target_ratio",
    "grpo_self_imitation_target_ratio",
    "grpo_self_imitation_target_scene_cap_ratio",
    "grpo_self_imitation_target_scene_cap_active",
    "grpo_self_imitation_target_reward_mean",
    "grpo_self_imitation_target_margin_mean",
    "grpo_self_imitation_target_nc_mean",
    "grpo_self_imitation_target_dac_mean",
    "grpo_self_imitation_target_ttc_mean",
    "grpo_self_imitation_target_ep_mean",
    "grpo_self_imitation_target_comfort_mean",
    "grpo_self_imitation_target_ddc_mean",
    "mixed_group_ratio",
    "all_safe_group_ratio",
    "all_unsafe_group_ratio",
    "reference_kl_loss",
    "reference_kl_coeff",
    "bc_coeff",
    "use_gspo_ratio",
    "sampled_from_behavior_policy",
    "lr",
    "eval_rows",
    "best_pdms",
    "best_checkpoint_id",
    "best_watcher",
    "best_eval_dir",
    "best_csv_path",
    "best_nc",
    "best_dac",
    "best_ttc",
    "best_ep",
    "best_comfort",
    "best_ddc",
    "best_tlc",
    "recommendation",
)


def _utc(ts: float | int | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if ts is None else ts))


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


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception as exc:
        return {"state": "unreadable", "error": f"{type(exc).__name__}: {exc}"}


def _read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with path.open(newline="") as f:
            return list(csv.DictReader(f, delimiter="\t"))
    except Exception:
        return []


def _find_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    normalized = {str(column).lower(): column for column in columns}
    for alias in aliases:
        if alias.lower() in normalized:
            return normalized[alias.lower()]
    return None


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _latest_file(paths: Iterable[Path]) -> Path | None:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def _load_status(run_root: Path) -> dict:
    status = run_root / "status" / "stage3_rl_2b.json"
    if status.exists():
        return _read_json(status)
    latest = _latest_file((run_root / "status").glob("*.json"))
    return _read_json(latest) if latest is not None else {}


def _load_train_scalars(run_root: Path) -> tuple[Path | None, dict[str, tuple[int, float]]]:
    event_file = _latest_file(run_root.glob("**/events.out.tfevents.*"))
    if event_file is None or EventAccumulator is None:
        return event_file, {}
    try:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 100000})
        accumulator.Reload()
        tags = set(accumulator.Tags().get("scalars", []))
    except Exception:
        return event_file, {}

    scalars: dict[str, tuple[int, float]] = {}
    for tag in TRAIN_SCALAR_TAGS:
        if tag not in tags:
            continue
        values = accumulator.Scalars(tag)
        if values:
            scalars[tag] = (int(values[-1].step), float(values[-1].value))
    return event_file, scalars


def _candidate_metric_csvs(eval_dir: Path) -> list[Path]:
    if not eval_dir.exists():
        return []
    return sorted(eval_dir.rglob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)


def _summarize_metric_csv(path: Path) -> dict[str, object] | None:
    try:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if _find_column(fieldnames, METRIC_ALIASES["pdms"]) is None:
                return None
            columns = {metric: _find_column(fieldnames, aliases) for metric, aliases in METRIC_ALIASES.items()}
            valid_column = _find_column(fieldnames, ("valid",))
            token_column = _find_column(fieldnames, ("token",))
            sums = {metric: 0.0 for metric in METRIC_ALIASES}
            counts = {metric: 0 for metric in METRIC_ALIASES}
            total_rows = 0
            valid_rows = 0
            for row in reader:
                total_rows += 1
                if token_column is not None and str(row.get(token_column, "")).strip().lower() == "average":
                    continue
                if valid_column is not None and not _truthy(row.get(valid_column)):
                    continue
                valid_rows += 1
                for metric, column in columns.items():
                    if column is None:
                        continue
                    value = _float_or_none(row.get(column))
                    if value is None:
                        continue
                    sums[metric] += value
                    counts[metric] += 1
    except Exception:
        return None

    if counts["pdms"] == 0:
        return None
    summary: dict[str, object] = {
        "csv_path": str(path),
        "num_rows": total_rows,
        "num_valid_rows": valid_rows,
    }
    for metric in METRIC_ALIASES:
        if counts[metric] > 0:
            summary[f"{metric}_mean"] = sums[metric] / counts[metric]
    return summary


def _summarize_eval_dir_csv(eval_dir: Path) -> dict[str, object]:
    for path in _candidate_metric_csvs(eval_dir):
        summary = _summarize_metric_csv(path)
        if summary is not None:
            return summary
    return {}


def _first_float(row: dict[str, str], fields: Iterable[str]) -> float | None:
    for field in fields:
        value = _float_or_none(row.get(field))
        if value is not None:
            return value
    return None


def _collect_eval_rows(run_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[Path] = set()
    for path in sorted(run_root.rglob("checkpoint_eval_submetrics.tsv")):
        if path in seen:
            continue
        seen.add(path)
        for row in _read_tsv(path):
            enriched = dict(row)
            enriched["_watcher"] = path.parent.name
            enriched["_summary_path"] = str(path)
            rows.append(enriched)

    for path in sorted(run_root.rglob("checkpoint_eval_summary.tsv")):
        if path in seen:
            continue
        seen.add(path)
        for row in _read_tsv(path):
            if row.get("state") not in {"done", "success", "completed", ""} and not _first_float(row, EVAL_PDMS_FIELDS):
                continue
            enriched = dict(row)
            if _first_float(enriched, EVAL_PDMS_FIELDS) is None and enriched.get("eval_dir"):
                csv_summary = _summarize_eval_dir_csv(Path(enriched["eval_dir"]))
                for key, value in csv_summary.items():
                    enriched.setdefault(key, str(value))
            enriched["_watcher"] = path.parent.name
            enriched["_summary_path"] = str(path)
            rows.append(enriched)
    return rows


def _best_eval_row(rows: list[dict[str, str]]) -> tuple[dict[str, str] | None, float | None]:
    best_row = None
    best_pdms = None
    for row in rows:
        pdms = _first_float(row, EVAL_PDMS_FIELDS)
        if pdms is None:
            continue
        if best_pdms is None or pdms > best_pdms:
            best_pdms = pdms
            best_row = row
    return best_row, best_pdms


def _collect_checkpoints(run_root: Path) -> list[Path]:
    return sorted(run_root.glob("train/**/*.ckpt"), key=lambda p: p.stat().st_mtime)


def _recommend(row: dict[str, object], target_pdms: float) -> str:
    eval_rows = int(row.get("eval_rows") or 0)
    state = str(row.get("training_state") or "")
    best_pdms = _float_or_none(row.get("best_pdms"))
    checkpoint_count = int(row.get("checkpoint_count") or 0)
    reward = _float_or_none(row.get("train_reward"))
    all_unsafe = _float_or_none(row.get("all_unsafe_group_ratio"))
    group_std = _float_or_none(row.get("group_reward_std"))
    event_age = _float_or_none(row.get("event_age_sec"))

    if state == "running" and checkpoint_count == 0:
        return "wait_for_first_checkpoint"
    if state == "running" and eval_rows == 0:
        return "wait_for_checkpoint_eval"
    if event_age is not None and event_age > 1800 and state == "running":
        return "inspect_training_progress_stale_event_file"
    if best_pdms is not None and best_pdms >= target_pdms and state == "running":
        return "keep_and_extend_best_run"
    if best_pdms is not None and best_pdms >= target_pdms:
        return "use_as_historical_baseline"
    if best_pdms is not None and state == "running":
        return "continue_until_next_epoch_then_compare"
    if reward is not None and reward < 0.70:
        return "review_reward_signal_or_exploration"
    if all_unsafe is not None and all_unsafe > 0.10:
        return "tighten_safety_or_reference_trust_region"
    if group_std is not None and group_std < 0.05:
        return "increase_exploration_or_group_diversity"
    if eval_rows > 0 and best_pdms is not None and best_pdms < target_pdms:
        return "below_target_try_next_algorithm_variant"
    return "monitor"


def summarize_run(run_root: Path, target_pdms: float) -> dict[str, object]:
    status = _load_status(run_root)
    ckpts = _collect_checkpoints(run_root)
    event_file, scalars = _load_train_scalars(run_root)
    eval_rows = _collect_eval_rows(run_root)
    best_eval, best_pdms = _best_eval_row(eval_rows)

    latest_step = ""
    for _, (step, _) in scalars.items():
        latest_step = max(int(latest_step or 0), step)

    row: dict[str, object] = {
        "run_name": run_root.name,
        "run_root": str(run_root),
        "mtime_utc": _utc(run_root.stat().st_mtime),
        "training_state": status.get("state", ""),
        "training_alive": status.get("alive", ""),
        "train_pid": status.get("pid", ""),
        "checkpoint_count": len(ckpts),
        "latest_checkpoint": str(ckpts[-1]) if ckpts else "",
        "latest_checkpoint_utc": _utc(ckpts[-1].stat().st_mtime) if ckpts else "",
        "event_file": str(event_file) if event_file is not None else "",
        "event_age_sec": f"{time.time() - event_file.stat().st_mtime:.1f}" if event_file is not None else "",
        "latest_step": latest_step,
        "eval_rows": len(eval_rows),
        "best_pdms": best_pdms if best_pdms is not None else "",
        "best_checkpoint_id": "",
        "best_watcher": "",
        "best_eval_dir": "",
        "best_csv_path": "",
    }

    scalar_map = {
        "train_reward": "train/reward_step",
        "base_reward": "train/base_reward_step",
        "shaped_reward": "train/shaped_reward_step",
        "safe_ratio": "train/safe_ratio_step",
        "mean_ep": "train/mean_ep_step",
        "mean_ttc": "train/mean_ttc_step",
        "mean_comfort": "train/mean_comfort_step",
        "group_reward_std": "train/group_reward_std_step",
        "grpo_self_imitation_candidate_ratio": "train/grpo_self_imitation_candidate_ratio_step",
        "grpo_self_imitation_safety_candidate_ratio": "train/grpo_self_imitation_safety_candidate_ratio_step",
        "grpo_self_imitation_nc_pass_ratio": "train/grpo_self_imitation_nc_pass_ratio_step",
        "grpo_self_imitation_dac_pass_ratio": "train/grpo_self_imitation_dac_pass_ratio_step",
        "grpo_self_imitation_ttc_pass_ratio": "train/grpo_self_imitation_ttc_pass_ratio_step",
        "grpo_self_imitation_ddc_pass_ratio": "train/grpo_self_imitation_ddc_pass_ratio_step",
        "grpo_self_imitation_pre_cap_target_ratio": "train/grpo_self_imitation_pre_cap_target_ratio_step",
        "grpo_self_imitation_target_ratio": "train/grpo_self_imitation_target_ratio_step",
        "grpo_self_imitation_target_scene_cap_ratio": "train/grpo_self_imitation_target_scene_cap_ratio_step",
        "grpo_self_imitation_target_scene_cap_active": "train/grpo_self_imitation_target_scene_cap_active_step",
        "grpo_self_imitation_target_reward_mean": "train/grpo_self_imitation_target_reward_mean_step",
        "grpo_self_imitation_target_margin_mean": "train/grpo_self_imitation_target_margin_mean_step",
        "grpo_self_imitation_target_nc_mean": "train/grpo_self_imitation_target_nc_mean_step",
        "grpo_self_imitation_target_dac_mean": "train/grpo_self_imitation_target_dac_mean_step",
        "grpo_self_imitation_target_ttc_mean": "train/grpo_self_imitation_target_ttc_mean_step",
        "grpo_self_imitation_target_ep_mean": "train/grpo_self_imitation_target_ep_mean_step",
        "grpo_self_imitation_target_comfort_mean": "train/grpo_self_imitation_target_comfort_mean_step",
        "grpo_self_imitation_target_ddc_mean": "train/grpo_self_imitation_target_ddc_mean_step",
        "mixed_group_ratio": "train/mixed_group_ratio_step",
        "all_safe_group_ratio": "train/all_safe_group_ratio_step",
        "all_unsafe_group_ratio": "train/all_unsafe_group_ratio_step",
        "reference_kl_loss": "train/reference_kl_loss_step",
        "reference_kl_coeff": "train/reference_kl_coeff_step",
        "bc_coeff": "train/bc_coeff_step",
        "use_gspo_ratio": "train/use_gspo_ratio_step",
        "sampled_from_behavior_policy": "train/sampled_from_behavior_policy_step",
        "lr": "lr-AdamW",
    }
    for output_field, tag in scalar_map.items():
        row[output_field] = scalars.get(tag, ("", ""))[1]

    if best_eval is not None:
        row["best_checkpoint_id"] = best_eval.get("checkpoint_id") or best_eval.get("checkpoint") or ""
        row["best_watcher"] = best_eval.get("_watcher", "")
        row["best_eval_dir"] = best_eval.get("eval_dir", "")
        row["best_csv_path"] = best_eval.get("csv_path") or best_eval.get("csv") or ""
        for metric in SUBMETRIC_FIELDS:
            row[f"best_{metric.removesuffix('_mean')}"] = best_eval.get(metric, "")

    row["recommendation"] = _recommend(row, target_pdms)
    return row


def _discover_runs(outputs_root: Path, run_globs: list[str], max_runs: int) -> list[Path]:
    found: dict[Path, None] = {}
    for pattern in run_globs:
        for path in outputs_root.glob(pattern):
            if path.is_dir():
                found[path] = None
    runs = sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[:max_runs] if max_runs > 0 else runs


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in OUTPUT_FIELDS})


def _write_json(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--run-glob", action="append", default=None)
    parser.add_argument("--max-runs", type=int, default=40)
    parser.add_argument("--target-pdms", type=float, default=0.9055)
    parser.add_argument("--include-dryruns", action="store_true")
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--output-tsv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--print-limit", type=int, default=20)
    args = parser.parse_args()

    run_globs = args.run_glob if args.run_glob is not None else list(DEFAULT_RUN_GLOBS)
    runs = _discover_runs(args.outputs_root, run_globs, args.max_runs)
    rows = []
    for run in runs:
        if not args.include_dryruns and "dryrun" in run.name.lower():
            continue
        row = summarize_run(run, args.target_pdms)
        has_evidence = bool(
            row.get("training_state")
            or row.get("event_file")
            or int(row.get("checkpoint_count") or 0) > 0
            or int(row.get("eval_rows") or 0) > 0
        )
        if not args.include_empty and not has_evidence:
            continue
        rows.append(row)

    if args.output_tsv is not None:
        _write_tsv(args.output_tsv, rows)
    if args.output_json is not None:
        _write_json(args.output_json, rows)

    print(
        "\t".join(
            [
                "run_name",
                "state",
                "ckpts",
                "step",
                "reward",
                "safe",
                "eval_rows",
                "best_pdms",
                "best_ckpt",
                "recommendation",
            ]
        )
    )
    for row in rows[: args.print_limit]:
        print(
            "\t".join(
                str(value)
                for value in [
                    row.get("run_name", ""),
                    row.get("training_state", ""),
                    row.get("checkpoint_count", ""),
                    row.get("latest_step", ""),
                    row.get("train_reward", ""),
                    row.get("safe_ratio", ""),
                    row.get("eval_rows", ""),
                    row.get("best_pdms", ""),
                    row.get("best_checkpoint_id", ""),
                    row.get("recommendation", ""),
                ]
            )
        )


if __name__ == "__main__":
    main()
