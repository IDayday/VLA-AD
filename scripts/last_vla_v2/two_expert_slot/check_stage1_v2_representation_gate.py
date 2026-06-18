#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import write_json  # noqa: E402


DEFAULTS = {
    "trained_dyn_loss_random_max_fraction": 0.60,
    "trained_geo_loss_random_max_fraction": 0.30,
    "image_only_dyn_ratio_min": 1.10,
    "image_only_geo_ratio_min": 1.20,
    "slot_only_dyn_gain_min": 1.15,
    "slot_only_geo_gain_min": 1.25,
    "dyn_top1_min": 0.06,
    "geo_top1_min": 0.06,
    "dyn_top5_min": 0.20,
    "geo_top5_min": 0.20,
    "positive_margin_min": 0.0,
    "zero_dyn_probe_ratio_min": 5.0,
    "zero_geo_probe_ratio_min": 5.0,
    "direct_traj_parse_ok_min": 0.95,
    "direct_traj_l1_max_relative_to_base": 1.05,
    "hidden_drift_cosine_min": 0.90,
    "old_stage1_image_only_ratio_delta_min": 0.05,
}


def _load(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _get_float(container: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = container.get(key, default)
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _metric(result: Dict[str, Any], key: str, default: float = 0.0) -> float:
    metrics = result.get("metrics", {}) if isinstance(result.get("metrics"), dict) else {}
    return _get_float(metrics, key, default)


def _comparison(result: Dict[str, Any], key: str, default: float = 0.0) -> float:
    comparisons = result.get("comparisons", {}) if isinstance(result.get("comparisons"), dict) else {}
    return _get_float(comparisons, key, default)


def _retrieval(result: Dict[str, Any], route: str, key: str, default: float = 0.0) -> float:
    retrieval = result.get("retrieval", {}) if isinstance(result.get("retrieval"), dict) else {}
    item = retrieval.get(route, {}) if isinstance(retrieval.get(route), dict) else {}
    return _get_float(item, key, default)


def _section(result: Dict[str, Any], name: str, key: str, default: float = 0.0) -> float:
    section = result.get(name, {}) if isinstance(result.get(name), dict) else {}
    return _get_float(section, key, default)


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def check(
    eval_json: Path,
    base_direct_traj_eval_json: Path | None = None,
    thresholds: Optional[Dict[str, float]] = None,
    *,
    bootstrap_json: Path | None = None,
    old_stage1_eval_json: Path | None = None,
) -> Dict[str, Any]:
    thresholds = dict(DEFAULTS if thresholds is None else thresholds)
    result = _load(eval_json)
    old = _load(old_stage1_eval_json) if old_stage1_eval_json is not None and old_stage1_eval_json.is_file() else {}
    bootstrap = _load(bootstrap_json) if bootstrap_json is not None and bootstrap_json.is_file() else {}
    checks: Dict[str, Dict[str, Any]] = {}

    def add(name: str, value: float, op: str, threshold: float, group: str, hard: bool = True) -> None:
        if op == ">=":
            ok = value >= threshold
        elif op == ">":
            ok = value > threshold
        elif op == "<=":
            ok = value <= threshold
        elif op == "<":
            ok = value < threshold
        else:
            raise ValueError(f"Unknown gate op: {op}")
        checks[name] = {
            "ok": bool(ok),
            "value": value,
            "op": op,
            "threshold": threshold,
            "group": group,
            "hard": bool(hard),
        }

    trained_dyn = _metric(result, "trained_dyn_loss")
    trained_geo = _metric(result, "trained_geo_loss")
    random_dyn = _metric(result, "random_dyn_loss")
    random_geo = _metric(result, "random_geo_loss")
    add(
        "teacher_dyn_trained_vs_random",
        trained_dyn / max(random_dyn, 1e-12),
        "<=",
        thresholds["trained_dyn_loss_random_max_fraction"],
        "teacher",
    )
    add(
        "teacher_geo_trained_vs_random",
        trained_geo / max(random_geo, 1e-12),
        "<=",
        thresholds["trained_geo_loss_random_max_fraction"],
        "teacher",
    )

    add("image_only_dyn_ratio", _comparison(result, "image_only_dyn_loss_over_trained_dyn_loss"), ">=", thresholds["image_only_dyn_ratio_min"], "anti_shortcut", hard=False)
    add("image_only_geo_ratio", _comparison(result, "image_only_geo_loss_over_trained_geo_loss"), ">=", thresholds["image_only_geo_ratio_min"], "anti_shortcut", hard=False)
    add("slot_only_dyn_gain", _comparison(result, "slot_only_dyn_gain"), ">=", thresholds["slot_only_dyn_gain_min"], "slot_information")
    add("slot_only_geo_gain", _comparison(result, "slot_only_geo_gain"), ">=", thresholds["slot_only_geo_gain_min"], "slot_information")

    add("dyn_top1", _retrieval(result, "trained_dyn", "top1"), ">=", thresholds["dyn_top1_min"], "retrieval")
    add("geo_top1", _retrieval(result, "trained_geo", "top1"), ">=", thresholds["geo_top1_min"], "retrieval")
    add("dyn_top5", _retrieval(result, "trained_dyn", "top5"), ">=", thresholds["dyn_top5_min"], "retrieval")
    add("geo_top5", _retrieval(result, "trained_geo", "top5"), ">=", thresholds["geo_top5_min"], "retrieval")
    add("dyn_positive_margin", _retrieval(result, "trained_dyn", "positive_margin"), ">", thresholds["positive_margin_min"], "retrieval")
    add("geo_positive_margin", _retrieval(result, "trained_geo", "positive_margin"), ">", thresholds["positive_margin_min"], "retrieval")

    add("zero_dyn_probe_ratio", _comparison(result, "zero_dyn_probe_ratio"), ">=", thresholds["zero_dyn_probe_ratio_min"], "planning")
    add("zero_geo_probe_ratio", _comparison(result, "zero_geo_probe_ratio"), ">=", thresholds["zero_geo_probe_ratio_min"], "planning")
    add("dyn_only_beats_no_signal", _metric(result, "dyn_only_probe_loss"), "<=", _metric(result, "no_signal_probe_loss"), "planning")
    add("geo_only_beats_no_signal", _metric(result, "geo_only_probe_loss"), "<=", _metric(result, "no_signal_probe_loss"), "planning")

    direct_count = _section(result, "direct", "direct_eval_count")
    if direct_count > 0:
        add("direct_traj_parse_ok", _section(result, "direct", "direct_traj_parse_ok_ratio"), ">=", thresholds["direct_traj_parse_ok_min"], "replay", hard=False)
        base_l1 = _section(result, "direct", "base_direct_traj_l1")
        direct_l1 = _section(result, "direct", "direct_traj_l1")
        if base_l1 > 0 and _finite(direct_l1):
            add("direct_traj_l1_relative_to_base", direct_l1 / base_l1, "<=", thresholds["direct_traj_l1_max_relative_to_base"], "replay", hard=False)
        else:
            add("direct_traj_l1_finite", 1.0 if _finite(direct_l1) else 0.0, ">=", 1.0, "replay", hard=False)
    else:
        add("direct_traj_available", 0.0, ">=", 1.0, "replay", hard=False)

    add("hidden_drift_cosine", _section(result, "hidden", "hidden_drift_cosine"), ">=", thresholds["hidden_drift_cosine_min"], "hidden", hard=False)
    add("hidden_anchor_loss_finite", 1.0 if _finite(_section(result, "hidden", "hidden_anchor_loss")) else 0.0, ">=", 1.0, "hidden", hard=False)

    if old:
        add(
            "old_stage1_image_only_dyn_ratio_delta",
            _comparison(result, "image_only_dyn_loss_over_trained_dyn_loss") - _comparison(old, "image_only_dyn_loss_over_trained_dyn_loss"),
            ">=",
            thresholds["old_stage1_image_only_ratio_delta_min"],
            "old_stage1",
            hard=False,
        )
        add(
            "old_stage1_image_only_geo_ratio_delta",
            _comparison(result, "image_only_geo_loss_over_trained_geo_loss") - _comparison(old, "image_only_geo_loss_over_trained_geo_loss"),
            ">=",
            thresholds["old_stage1_image_only_ratio_delta_min"],
            "old_stage1",
            hard=False,
        )
        add("old_stage1_slot_only_dyn_gain", _comparison(result, "slot_only_dyn_gain") - _comparison(old, "slot_only_dyn_gain"), ">=", 0.0, "old_stage1", hard=False)
        add("old_stage1_slot_only_geo_gain", _comparison(result, "slot_only_geo_gain") - _comparison(old, "slot_only_geo_gain"), ">=", 0.0, "old_stage1", hard=False)

    hard_ok = all(item["ok"] for item in checks.values() if item["hard"])
    teacher_ok = all(item["ok"] for item in checks.values() if item["group"] == "teacher")
    planning_ok = all(item["ok"] for item in checks.values() if item["group"] == "planning")
    all_ok = all(item["ok"] for item in checks.values())
    if all_ok:
        status = "READY"
    elif teacher_ok and planning_ok and hard_ok:
        status = "CONDITIONAL"
    else:
        status = "NOT_READY"
    return {
        "ok": status == "READY",
        "status": status,
        "eval_json": str(eval_json),
        "bootstrap_json": str(bootstrap_json) if bootstrap_json else None,
        "old_stage1_eval_json": str(old_stage1_eval_json) if old_stage1_eval_json else None,
        "thresholds": thresholds,
        "checks": checks,
        "bootstrap_summary_present": bool(bootstrap),
        "recommendation": (
            "May generate hidden-cache smoke commands only; do not start Stage2 until hidden-cache smoke passes."
            if status in {"READY", "CONDITIONAL"}
            else "Do not enter hidden cache / Stage2; inspect failed Stage1-v2 checks."
        ),
    }


def write_markdown(path: Path, result: Dict[str, Any]) -> None:
    lines = [
        "# Stage1-v2 Representation Gate",
        "",
        f"- status: `{result['status']}`",
        f"- ok: `{result['ok']}`",
        f"- recommendation: {result['recommendation']}",
        "",
        "| check | group | hard | ok | value | threshold |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for name, item in sorted(result["checks"].items()):
        lines.append(
            f"| `{name}` | `{item['group']}` | `{item['hard']}` | `{item['ok']}` | "
            f"{item['value']:.6g} | {item['op']} {item['threshold']:.6g} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Stage1-v2 representation gate.")
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--bootstrap-json", type=Path, default=None)
    parser.add_argument("--old-stage1-eval-json", type=Path, default=None)
    parser.add_argument("--base-direct-eval-json", "--base-direct-traj-eval-json", dest="base_direct_traj_eval_json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    for name, default in DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=default)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = {name: float(getattr(args, name)) for name in DEFAULTS}
    result = check(
        args.eval_json,
        args.base_direct_traj_eval_json,
        thresholds,
        bootstrap_json=args.bootstrap_json,
        old_stage1_eval_json=args.old_stage1_eval_json,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "stage1_v2_gate.json", result)
    md_path = args.output_dir / "stage1_v2_gate.md"
    write_markdown(md_path, result)
    if args.output_dir.name == "json":
        sibling = args.output_dir.parent / "md" / "stage1_v2_gate.md"
        write_markdown(sibling, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"READY", "CONDITIONAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
