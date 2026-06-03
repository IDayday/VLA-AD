#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Optional


def read_optional(path: Optional[Path]) -> str:
    if path is None:
        return "_Not provided._"
    if not path.is_file():
        return f"_Missing: `{path}`_"
    return path.read_text(encoding="utf-8")


def git_info() -> tuple[str, str]:
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return branch, commit
    except Exception:
        return "<unknown>", "<unknown>"


def build_evidence_report(
    *,
    output_md: Path,
    input_discovery_md: Optional[Path] = None,
    input_manifest_md: Optional[Path] = None,
    candidate_search_md: Optional[Path] = None,
    dryrun_summary_md: Optional[Path] = None,
    r0_diagnostics_md: Optional[Path] = None,
    go_no_go_md: Optional[Path] = None,
    first_result_summary_md: Optional[Path] = None,
    r1_results: Optional[Path] = None,
    r2_results: Optional[Path] = None,
) -> None:
    branch, commit = git_info()
    lines = [
        "# RISK-VLA Round 1 Evidence Report",
        "",
        "This is small-scale diagnostic evidence for low-score risk scenarios / critical-risk subsets. It is not a final PDM performance claim.",
        "",
        "BiT is treated as a path/terminal intent strategy inside the RISK-VLA strategy bank. The RISK-VLA framework is risk state -> strategy routing/modulation -> diffusion planning.",
        "",
        "Oracle-router results are analysis-only. Navtest/test labels must not be used for train-time supervision.",
        "",
        "PDM CSVs may be missing because Stage 0-6 intentionally built analysis/training harnesses and ran only safe dry-runs/smoke tests. Candidate search looks only for already-created PDM result CSVs in safe experiment roots; it does not create them. Creating PDM CSVs requires running NAVSIM PDM evaluation with real checkpoints, metric cache, logs, and sensor blobs. If no small-scale eval has been executed and no CSV paths were explicitly registered, no valid PDM CSVs exist to discover.",
        "",
        "## 1. Current Branch / Commit",
        "",
        f"- branch: `{branch}`",
        f"- commit: `{commit}`",
        "",
        "## 2. Inputs Found And Missing",
        "",
        read_optional(input_discovery_md),
        "",
        "## Input Candidate Search",
        "",
        read_optional(candidate_search_md),
        "",
        "## Explicit Manifest Status",
        "",
        read_optional(input_manifest_md),
        "",
        "## 3. Risk Label Distribution Summary",
        "",
        "See input discovery and diagnostic aggregation artifacts for label coverage. Do not proceed with training if leakage checks fail.",
        "",
        "## 4. R0 Risk-Head Diagnostic Results",
        "",
        read_optional(r0_diagnostics_md),
        "",
        "## 5. Strategy Activation Sanity Check",
        "",
        "Router weights are diagnostic signals. Strategy scales may be zero in R0, but router activations should still be observable.",
        "",
        "## 6. GO/NO-GO Decision",
        "",
        read_optional(go_no_go_md),
        "",
        "## First-Result Tables",
        "",
        read_optional(first_result_summary_md),
        "",
        "## R1/R2 Permission",
        "",
        "R1 oracle-router may proceed only as analysis. R2 predicted-router should be blocked unless the GO/NO-GO decision is `GO_R1_R2`.",
        "",
        "## 7. R1 Oracle-Router Pilot Status",
        "",
        read_optional(r1_results),
        "",
        "## 8. R2 Predicted-Router Pilot Status",
        "",
        read_optional(r2_results),
        "",
        "## 9. Interpretation Under RISK-VLA Framing",
        "",
        "Risk diagnostic, oracle-router, and predicted-router results must be interpreted separately. R1 estimates routing upper bound; R2 estimates predicted risk routing behavior.",
        "",
        "## 10. Next Recommended Experiment",
        "",
        "Use the GO/NO-GO decision: improve labels/calibration if R0 is weak, run R1 only for analysis if needed, and run R2 only when diagnostic evidence is strong enough.",
        "",
        "## Dry-Run Summary",
        "",
        read_optional(dryrun_summary_md),
        "",
    ]
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a RISK-VLA Round 1 evidence report from partial artifacts.")
    parser.add_argument("--input-discovery-md", type=Path, default=None)
    parser.add_argument("--input-manifest-md", type=Path, default=None)
    parser.add_argument("--candidate-search-md", type=Path, default=None)
    parser.add_argument("--dryrun-summary-md", type=Path, default=None)
    parser.add_argument("--r0-diagnostics-md", type=Path, default=None)
    parser.add_argument("--go-no-go-md", type=Path, default=None)
    parser.add_argument("--first-result-summary-md", type=Path, default=None)
    parser.add_argument("--r1-results", type=Path, default=None)
    parser.add_argument("--r2-results", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    build_evidence_report(
        input_discovery_md=args.input_discovery_md,
        input_manifest_md=args.input_manifest_md,
        candidate_search_md=args.candidate_search_md,
        dryrun_summary_md=args.dryrun_summary_md,
        r0_diagnostics_md=args.r0_diagnostics_md,
        go_no_go_md=args.go_no_go_md,
        first_result_summary_md=args.first_result_summary_md,
        r1_results=args.r1_results,
        r2_results=args.r2_results,
        output_md=args.output_md,
    )
    print(f"Wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
