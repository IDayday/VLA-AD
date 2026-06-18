#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import write_json  # noqa: E402


MetricFn = Callable[[List[Dict[str, Any]]], float]


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return data


def _mean(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows if key in row and row[key] is not None]
    return sum(vals) / max(1, len(vals))


def _ratio(rows: List[Dict[str, Any]], num: str, den: str) -> float:
    return _mean(rows, num) / max(_mean(rows, den), 1e-12)


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    pos = (len(values) - 1) * float(q)
    lo = int(pos)
    hi = min(len(values) - 1, lo + 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def _bootstrap(rows: List[Dict[str, Any]], fn: MetricFn, *, n: int, rng: random.Random) -> Dict[str, float]:
    if not rows:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0.0}
    actual = fn(rows)
    draws = []
    count = len(rows)
    for _ in range(int(n)):
        sample = [rows[rng.randrange(count)] for _ in range(count)]
        draws.append(fn(sample))
    return {
        "mean": float(actual),
        "ci95_low": float(_percentile(draws, 0.025)),
        "ci95_high": float(_percentile(draws, 0.975)),
        "n": float(count),
    }


def build(args: argparse.Namespace) -> Dict[str, Any]:
    data = _load(args.eval_json)
    rows = data.get("per_sample", [])
    if not isinstance(rows, list):
        rows = []
    rng = random.Random(int(args.seed))
    specs: Dict[str, MetricFn] = {
        "normal_dyn_loss": lambda r: _mean(r, "normal_dyn_loss"),
        "normal_geo_loss": lambda r: _mean(r, "normal_geo_loss"),
        "image_only_dyn_ratio": lambda r: _ratio(r, "image_only_dyn_loss", "normal_dyn_loss"),
        "image_only_geo_ratio": lambda r: _ratio(r, "image_only_geo_loss", "normal_geo_loss"),
        "slot_only_dyn_gain": lambda r: _ratio(r, "no_signal_dyn_loss", "slot_only_dyn_loss"),
        "slot_only_geo_gain": lambda r: _ratio(r, "no_signal_geo_loss", "slot_only_geo_loss"),
        "dyn_top1": lambda r: _mean(r, "trained_dyn_top1"),
        "geo_top1": lambda r: _mean(r, "trained_geo_top1"),
        "dyn_top5": lambda r: _mean(r, "trained_dyn_top5"),
        "geo_top5": lambda r: _mean(r, "trained_geo_top5"),
        "dyn_mrr": lambda r: _mean(r, "trained_dyn_rr"),
        "geo_mrr": lambda r: _mean(r, "trained_geo_rr"),
        "probe_fused_loss": lambda r: _mean(r, "normal_probe_loss"),
        "zero_dyn_probe_ratio": lambda r: _ratio(r, "zero_dyn_probe_loss", "normal_probe_loss"),
        "zero_geo_probe_ratio": lambda r: _ratio(r, "zero_geo_probe_loss", "normal_probe_loss"),
        "direct_traj_parse_ok_ratio": lambda r: float(data.get("direct", {}).get("direct_traj_parse_ok_ratio", 0.0)),
        "direct_traj_l1": lambda r: float(data.get("direct", {}).get("direct_traj_l1", float("nan"))),
        "hidden_drift_cosine": lambda r: float(data.get("hidden", {}).get("hidden_drift_cosine", float("nan"))),
    }
    result = {
        "eval_json": str(args.eval_json),
        "num_bootstrap": int(args.num_bootstrap),
        "seed": int(args.seed),
        "sample_count": len(rows),
        "metrics": {
            name: _bootstrap(rows, fn, n=int(args.num_bootstrap), rng=rng)
            for name, fn in specs.items()
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "stage1_v2_bootstrap_ci.json", result)
    write_markdown(args.output_dir / "stage1_v2_bootstrap_ci.md", result)
    return result


def write_markdown(path: Path, result: Dict[str, Any]) -> None:
    lines = [
        "# Stage1-v2 Bootstrap Confidence Intervals",
        "",
        f"- eval_json: `{result['eval_json']}`",
        f"- sample_count: `{result['sample_count']}`",
        f"- num_bootstrap: `{result['num_bootstrap']}`",
        "",
        "| metric | mean | ci95_low | ci95_high |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, item in sorted(result["metrics"].items()):
        lines.append(f"| `{name}` | {item['mean']:.6g} | {item['ci95_low']:.6g} | {item['ci95_high']:.6g} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap Stage1-v2 eval metrics.")
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    result = build(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
