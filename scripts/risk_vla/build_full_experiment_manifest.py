#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml


def first_matching(rows: Sequence[Dict[str, Any]], *, split: Optional[str] = None, method: Optional[str] = None, min_count: int = 0) -> Optional[Dict[str, Any]]:
    for row in rows:
        if split is not None and row.get("split") != split:
            continue
        if method is not None and row.get("method") != method:
            continue
        count = int(row.get("row_count") or row.get("row_or_token_count") or 0)
        if count < min_count:
            continue
        return row
    return None


def paths_for_split(chunks: Sequence[Dict[str, Any]], split: str) -> List[str]:
    return [str(row["path"]) for row in chunks if row.get("split") == split]


def checkpoint_by_method(checkpoints: Sequence[Dict[str, Any]], method: str) -> Optional[str]:
    non_debug = [row for row in checkpoints if row.get("method") == method and not bool(row.get("is_debug"))]
    rows = non_debug or [row for row in checkpoints if row.get("method") == method]
    return str(rows[0]["path"]) if rows else None


def build_manifest(discovery: Dict[str, Any], *, min_real_samples: int) -> Dict[str, Any]:
    pdms = discovery.get("pdm_csvs", [])
    candidates = discovery.get("candidate_files", [])
    chunks = discovery.get("chunks", [])
    checkpoints = discovery.get("checkpoints", [])
    metric_caches = discovery.get("metric_caches", [])
    train_pdms = [row for row in pdms if row.get("split") == "train" and int(row.get("row_count") or 0) >= min_real_samples]
    val_pdms = [row for row in pdms if row.get("split") == "val" and int(row.get("row_count") or 0) >= min_real_samples]
    navtest_pdms = [row for row in pdms if row.get("split") == "navtest" and int(row.get("row_count") or 0) >= min_real_samples]
    train_candidates = [row for row in candidates if row.get("split") == "train" and int(row.get("row_or_token_count") or 0) >= min_real_samples]
    navtest_candidates = [row for row in candidates if row.get("split") == "navtest" and int(row.get("row_or_token_count") or 0) >= min_real_samples]
    blockers = list(discovery.get("readiness", {}).get("blockers", []))
    if len(train_pdms) < 2:
        blockers.append("Need at least two train split PDM tables with >=10k rows to build strategy utility labels.")
    if not val_pdms:
        blockers.append("Need held-out val PDM/candidate assets for tuning; navtest cannot be used.")
    return {
        "round": "risk_vla_v3_full",
        "git_commit": discovery.get("git_commit"),
        "scale_policy": {
            "min_real_samples": min_real_samples,
            "debug_runs_are_not_evidence": True,
            "navtest_labels_analysis_only": True,
        },
        "roots": discovery.get("paths", {}),
        "chunk_caches": {
            "train_chunks": paths_for_split(chunks, "train"),
            "val_chunks": paths_for_split(chunks, "val"),
            "navtest_chunks_analysis_only": paths_for_split(chunks, "navtest"),
        },
        "metric_caches": {
            "train": [row["path"] for row in metric_caches if row.get("split") == "train"],
            "val": [row["path"] for row in metric_caches if row.get("split") == "val"],
            "navtest_analysis_only": [row["path"] for row in metric_caches if row.get("split") == "navtest"],
        },
        "checkpoints": {
            "a0_base": checkpoint_by_method(checkpoints, "a0_base"),
            "bit": checkpoint_by_method(checkpoints, "bit"),
            "d5": checkpoint_by_method(checkpoints, "d5"),
            "risk_vla": checkpoint_by_method(checkpoints, "risk_vla"),
        },
        "pdm_tables": {
            "train": train_pdms,
            "val": val_pdms,
            "navtest_analysis_only": navtest_pdms,
        },
        "candidate_assets": {
            "train": train_candidates,
            "navtest_analysis_only": navtest_candidates,
        },
        "expert_caches": discovery.get("expert_caches", []),
        "readiness": discovery.get("readiness", {}),
        "blockers": blockers,
        "launchable": {
            "train_labels_10k": len(train_pdms) >= 2,
            "critic_10k": bool(train_candidates) and len(train_pdms) >= 2,
            "router_10k": len(train_pdms) >= 2,
            "navtest_10k_analysis": bool(paths_for_split(chunks, "navtest")),
            "full_training": bool(paths_for_split(chunks, "train")) and not blockers,
        },
    }


def write_markdown(manifest: Dict[str, Any], path: Path) -> None:
    lines = [
        "# RISK-VLA v3 Full Experiment Manifest Summary",
        "",
        f"Git commit: `{manifest.get('git_commit')}`",
        f"Minimum real samples: `{manifest['scale_policy']['min_real_samples']}`",
        "",
        "## Launchability",
        "",
    ]
    for key, value in manifest["launchable"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Blockers", ""])
    if manifest["blockers"]:
        lines.extend(f"- {item}" for item in manifest["blockers"])
    else:
        lines.append("_None._")
    lines.extend(["", "## Leakage Guard", "", "Manifest keeps navtest assets under `*_analysis_only` keys and never under training keys."])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build explicit RISK-VLA v3 full experiment manifest from discovery JSON.")
    parser.add_argument("--discovery-json", type=Path, required=True)
    parser.add_argument("--output-yaml", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--min-real-samples", type=int, default=10000)
    args = parser.parse_args(argv)
    discovery = json.loads(args.discovery_json.read_text(encoding="utf-8"))
    manifest = build_manifest(discovery, min_real_samples=args.min_real_samples)
    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.output_yaml.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    if args.output_md:
        write_markdown(manifest, args.output_md)
    print(json.dumps({"output_yaml": str(args.output_yaml), "blockers": manifest["blockers"], "launchable": manifest["launchable"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
