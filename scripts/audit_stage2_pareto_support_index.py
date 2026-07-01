from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Set

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.pareto_support import load_pareto_support_index


def _read_tokens(path: str) -> Set[str]:
    if not path:
        return set()
    token_path = Path(path)
    if not token_path.is_file():
        raise FileNotFoundError(f"token file not found: {token_path}")
    return {line.strip() for line in token_path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(args: argparse.Namespace) -> Dict[str, object]:
    index_path = Path(args.index)
    payload = load_pareto_support_index(index_path)
    tokens = [str(token) for token in payload["tokens"]]
    token_set = set(tokens)
    support_mask = payload["support_mask"].bool()
    support_weights = payload["support_weights"].float()
    support_trajs = payload["support_trajectories"].float()
    support_scores = payload["support_scores"].float()
    support_count = support_mask.sum(dim=1)
    weight_sum = (support_weights * support_mask.float()).sum(dim=1)
    finite_traj = torch.isfinite(support_trajs).all().item()
    finite_scores = torch.isfinite(support_scores).all().item()
    finite_weights = torch.isfinite(support_weights).all().item()

    source_hist: Counter[str] = Counter()
    for row_sources in payload.get("support_sources", []):
        for source in row_sources:
            if str(source):
                source_hist[str(source)] += 1

    metadata_fallback: Counter[str] = Counter()
    for item in payload.get("support_metadata", []):
        mode = str(item.get("fallback_mode", "none"))
        if mode != "none":
            metadata_fallback[mode] += 1

    val_tokens = _read_tokens(args.val_tokens)
    navtest_tokens = _read_tokens(args.navtest_tokens)
    train_tokens = _read_tokens(args.train_tokens)
    report = {
        "index": str(index_path),
        "index_sha256": _sha256(index_path),
        "num_tokens": len(tokens),
        "unique_tokens": len(token_set),
        "duplicate_tokens": len(tokens) - len(token_set),
        "support_count_hist": dict(sorted(Counter(str(int(v)) for v in support_count.tolist()).items())),
        "support_count_min": int(support_count.min().item()) if support_count.numel() else 0,
        "support_count_max": int(support_count.max().item()) if support_count.numel() else 0,
        "weights_sum_min": float(weight_sum.min().item()) if weight_sum.numel() else None,
        "weights_sum_max": float(weight_sum.max().item()) if weight_sum.numel() else None,
        "weights_sum_allclose_1": bool(torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-4)),
        "finite_trajectories": bool(finite_traj),
        "finite_scores": bool(finite_scores),
        "finite_weights": bool(finite_weights),
        "source_hist": dict(sorted(source_hist.items())),
        "fallback_hist": dict(sorted(metadata_fallback.items())),
        "overlap_train_tokens": len(token_set & train_tokens) if train_tokens else None,
        "overlap_val_tokens": len(token_set & val_tokens) if val_tokens else None,
        "overlap_navtest_tokens": len(token_set & navtest_tokens) if navtest_tokens else None,
        "summary": payload.get("summary", {}),
    }
    failures = []
    if report["duplicate_tokens"]:
        failures.append("duplicate_tokens")
    if report["support_count_min"] < 1 or report["support_count_max"] > 3:
        failures.append("support_count_out_of_range")
    if not report["weights_sum_allclose_1"]:
        failures.append("weights_not_sum_to_1")
    if not (report["finite_trajectories"] and report["finite_scores"] and report["finite_weights"]):
        failures.append("non_finite_values")
    if navtest_tokens and report["overlap_navtest_tokens"]:
        failures.append("navtest_overlap")
    if args.fail_on_val_overlap and val_tokens and report["overlap_val_tokens"]:
        failures.append("val_overlap")
    report["failures"] = failures
    report["passed"] = not failures
    return report


def _write_md(report: Dict[str, object], path: Path) -> None:
    lines = [
        "# Stage2 Pareto Support Index Audit",
        "",
        f"- index: `{report['index']}`",
        f"- sha256: `{report['index_sha256']}`",
        f"- passed: `{report['passed']}`",
        f"- tokens: `{report['num_tokens']}`",
        f"- unique_tokens: `{report['unique_tokens']}`",
        f"- duplicate_tokens: `{report['duplicate_tokens']}`",
        f"- support_count_min/max: `{report['support_count_min']}` / `{report['support_count_max']}`",
        f"- weights_sum_min/max: `{report['weights_sum_min']}` / `{report['weights_sum_max']}`",
        f"- overlap_navtest_tokens: `{report['overlap_navtest_tokens']}`",
        "",
        "## Failures",
        "",
    ]
    failures = report.get("failures", [])
    if failures:
        lines.extend(f"- {item}" for item in failures)
    else:
        lines.append("- none")
    lines.extend(["", "## Support Count", ""])
    for key, value in dict(report["support_count_hist"]).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Sources", ""])
    for key, value in dict(report["source_hist"]).items():
        lines.append(f"- {key}: {value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a PSI-Drive Stage2 Pareto support index.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--train-tokens", default="")
    parser.add_argument("--val-tokens", default="")
    parser.add_argument("--navtest-tokens", default="")
    parser.add_argument("--fail-on-val-overlap", action="store_true")
    args = parser.parse_args()

    report = audit(args)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    _write_md(report, output_md)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
