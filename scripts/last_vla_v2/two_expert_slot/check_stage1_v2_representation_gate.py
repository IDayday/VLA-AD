#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import write_json  # noqa: E402


DEFAULTS = {
    "image_only_dyn_ratio_min": 1.15,
    "image_only_geo_ratio_min": 1.30,
    "slot_only_dyn_gain_min": 1.20,
    "slot_only_geo_gain_min": 1.30,
    "dyn_top1_min": 0.08,
    "geo_top1_min": 0.08,
    "dyn_top5_min": 0.25,
    "geo_top5_min": 0.25,
    "positive_margin_min": 0.0,
    "direct_traj_parse_ok_min": 0.95,
    "direct_traj_l1_max_relative_to_base": 1.05,
    "hidden_drift_cosine_min": 0.90,
}


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _get_float(container: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(container.get(key, default))
    except (TypeError, ValueError):
        return default


def check(eval_json: Path, base_direct_traj_eval_json: Path | None, thresholds: Dict[str, float]) -> Dict[str, Any]:
    result = _load(eval_json)
    metrics = result.get("metrics", {}) if isinstance(result.get("metrics"), dict) else {}
    comparisons = result.get("comparisons", {}) if isinstance(result.get("comparisons"), dict) else {}
    retrieval = result.get("retrieval", {}) if isinstance(result.get("retrieval"), dict) else {}
    trained_dyn = retrieval.get("trained_dyn", {}) if isinstance(retrieval.get("trained_dyn"), dict) else {}
    trained_geo = retrieval.get("trained_geo", {}) if isinstance(retrieval.get("trained_geo"), dict) else {}

    checks: Dict[str, Dict[str, Any]] = {}
    def add(name: str, value: float, op: str, threshold: float) -> None:
        ok = value >= threshold if op == ">=" else value <= threshold
        checks[name] = {"ok": bool(ok), "value": value, "op": op, "threshold": threshold}

    add("image_only_dyn_ratio", _get_float(comparisons, "image_only_dyn_loss_over_trained_dyn_loss"), ">=", thresholds["image_only_dyn_ratio_min"])
    add("image_only_geo_ratio", _get_float(comparisons, "image_only_geo_loss_over_trained_geo_loss"), ">=", thresholds["image_only_geo_ratio_min"])
    slot_dyn_gain = _get_float(metrics, "no_signal_dyn_loss") / max(_get_float(metrics, "slot_only_dyn_loss"), 1e-12)
    slot_geo_gain = _get_float(metrics, "no_signal_geo_loss") / max(_get_float(metrics, "slot_only_geo_loss"), 1e-12)
    add("slot_only_dyn_gain", slot_dyn_gain, ">=", thresholds["slot_only_dyn_gain_min"])
    add("slot_only_geo_gain", slot_geo_gain, ">=", thresholds["slot_only_geo_gain_min"])
    add("dyn_top1", _get_float(trained_dyn, "top1"), ">=", thresholds["dyn_top1_min"])
    add("geo_top1", _get_float(trained_geo, "top1"), ">=", thresholds["geo_top1_min"])
    add("dyn_top5", _get_float(trained_dyn, "top5"), ">=", thresholds["dyn_top5_min"])
    add("geo_top5", _get_float(trained_geo, "top5"), ">=", thresholds["geo_top5_min"])
    add("dyn_positive_margin", _get_float(trained_dyn, "positive_margin"), ">=", thresholds["positive_margin_min"])
    add("geo_positive_margin", _get_float(trained_geo, "positive_margin"), ">=", thresholds["positive_margin_min"])

    direct = result.get("replay", {}) if isinstance(result.get("replay"), dict) else {}
    if direct:
        add("direct_traj_parse_ok", _get_float(direct, "direct_traj_parse_ok_ratio"), ">=", thresholds["direct_traj_parse_ok_min"])
        if base_direct_traj_eval_json is not None:
            base = _load(base_direct_traj_eval_json)
            base_replay = base.get("replay", {}) if isinstance(base.get("replay"), dict) else {}
            base_l1 = _get_float(base_replay, "direct_traj_l1", 0.0)
            if base_l1 > 0.0:
                add("direct_traj_l1_relative_to_base", _get_float(direct, "direct_traj_l1") / base_l1, "<=", thresholds["direct_traj_l1_max_relative_to_base"])
    hidden = result.get("hidden", {}) if isinstance(result.get("hidden"), dict) else {}
    if hidden:
        add("hidden_drift_cosine", _get_float(hidden, "hidden_drift_cosine"), ">=", thresholds["hidden_drift_cosine_min"])

    ok = all(item["ok"] for item in checks.values())
    return {
        "ok": bool(ok),
        "status": "READY" if ok else "FAIL",
        "eval_json": str(eval_json),
        "thresholds": thresholds,
        "checks": checks,
        "recommendation": "Stage1-v2 may proceed to hidden cache / Stage2 command planning." if ok else "Do not enter Stage2; inspect failed Stage1-v2 checks.",
    }


def write_markdown(path: Path, result: Dict[str, Any]) -> None:
    lines = [
        "# Stage1-v2 Representation Gate",
        "",
        f"- status: `{result['status']}`",
        f"- ok: `{result['ok']}`",
        f"- recommendation: {result['recommendation']}",
        "",
        "| check | ok | value | threshold |",
        "| --- | --- | ---: | ---: |",
    ]
    for name, item in sorted(result["checks"].items()):
        lines.append(f"| `{name}` | `{item['ok']}` | {item['value']:.6g} | {item['op']} {item['threshold']:.6g} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Stage1-v2 representation gate.")
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--base-direct-traj-eval-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    for name, default in DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=default)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = {name: float(getattr(args, name)) for name in DEFAULTS}
    result = check(args.eval_json, args.base_direct_traj_eval_json, thresholds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "stage1_v2_gate.json", result)
    write_markdown(args.output_dir / "stage1_v2_gate.md", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
