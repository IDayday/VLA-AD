#!/usr/bin/env python3
"""Assemble a compact, repository-relative reproduction manifest with hashes."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pipeline_common import (
    ANALYSIS_SEED,
    atomic_write_json,
    configure_logging,
    environment_versions,
    git_revision,
    load_json,
    relative_display,
    repository_root,
    sha256_file,
    supplementary_root,
)


DEFAULT_INCLUDE_DIRS = [
    "scripts",
    "configs",
    "raw_manifest",
    "derived",
    "audit",
    "tables/sources",
    "tables/generated",
    "figures/sources",
    "figures/generated",
    "launch",
    "sections",
]


def parse_args() -> argparse.Namespace:
    supp = supplementary_root()
    parser = argparse.ArgumentParser(
        description="Hash supplementary inputs, scripts, outputs, and recorded commands."
    )
    parser.add_argument("--supplementary-root", type=Path, default=supp)
    parser.add_argument(
        "--include-dir",
        action="append",
        default=[],
        help="Supplementary-relative directory to include; repeatable.",
    )
    parser.add_argument("--output", type=Path, default=supp / "reproduction_manifest.json")
    parser.add_argument("--seed", type=int, default=ANALYSIS_SEED)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    supp = args.supplementary_root.resolve()
    repo = repository_root().resolve()
    try:
        supp.relative_to(repo)
    except ValueError as exc:
        raise ValueError("--supplementary-root must be inside the repository") from exc

    include_dirs = args.include_dir or DEFAULT_INCLUDE_DIRS
    files: set[Path] = set()
    for name in include_dirs:
        directory = (supp / name).resolve()
        try:
            directory.relative_to(supp)
        except ValueError as exc:
            raise ValueError(f"include directory escapes supplementary root: {name}") from exc
        if not directory.exists():
            continue
        files.update(
            path for path in directory.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
    for name in (
        "supplementary.tex",
        "supplementary.bib",
        "supplementary.pdf",
        "README.md",
        "MISSING_EVIDENCE.md",
        "MAIN_TEXT_SUGGESTIONS.md",
        "Makefile",
        "make_supplement.sh",
    ):
        path = supp / name
        if path.is_file():
            files.add(path)
    completion_report = repo / "supplementary_completion_report.md"
    if completion_report.is_file():
        files.add(completion_report)
    files.discard(args.output.resolve())

    records = [
        {
            "path": relative_display(path, repo),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(files)
    ]
    provenance_records = []
    for path in sorted((supp / "derived").glob("*_provenance.json")):
        try:
            payload = load_json(path)
        except Exception as exc:
            provenance_records.append(
                {
                    "path": relative_display(path, repo),
                    "status": "unreadable",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        provenance_records.append(
            {
                "path": relative_display(path, repo),
                "status": payload.get("status"),
                "analysis_seed": payload.get("analysis_seed"),
                "command": payload.get("command"),
                "inputs": payload.get("inputs", []),
                "outputs": payload.get("outputs", []),
            }
        )

    payload = {
        "schema_version": 1,
        "repository": ".",
        "analysis_seed": args.seed,
        "analysis_seed_scope": "resampling and plotting only; not a training/evaluation seed",
        "git": git_revision(repo),
        "environment": environment_versions(),
        "files": records,
        "pipeline_runs": provenance_records,
    }
    atomic_write_json(args.output, payload)
    logging.info("wrote %s with %d file hashes", relative_display(args.output, repo), len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
