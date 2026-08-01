#!/usr/bin/env python3
"""Inventory evidence-bearing repository files and create a SHA-256 manifest.

The default scan intentionally skips model weights, caches, Python environments,
and other heavyweight trees.  Add explicit ``--scan-root`` values when an
otherwise excluded result directory must be audited.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from pipeline_common import (
    ANALYSIS_SEED,
    atomic_write_json,
    atomic_write_text,
    build_provenance,
    configure_logging,
    markdown_table,
    relative_display,
    repository_root,
    sha256_file,
    supplementary_root,
)


DEFAULT_SCAN_ROOTS = [
    "docs",
    "configs",
    "scripts",
    "reports",
    "experiments",
    "outputs",
    "navsim",
    "external",
]
DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "cache",
    "checkpoints",
    "hf_home",
    "modelscope_cache",
    "node_modules",
    "site-packages",
    "wandb",
    "env",
    "venv",
    ".venv",
}
EVIDENCE_SUFFIXES = {
    ".bib",
    ".cfg",
    ".ckpt",
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".out",
    ".parquet",
    ".pdf",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".py",
    ".safetensors",
    ".sh",
    ".tex",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}


@dataclass
class FileRecord:
    path: str
    category: str
    suffix: str
    bytes: int
    modified_utc: str
    sha256: str | None
    hash_status: str


def parse_args() -> argparse.Namespace:
    root = repository_root()
    supp = supplementary_root()
    parser = argparse.ArgumentParser(
        description="Inventory paper/code/result evidence and write a SHA-256 manifest."
    )
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument(
        "--scan-root",
        action="append",
        dest="scan_roots",
        help="Repository-relative directory to scan; repeatable (defaults to curated roots).",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Directory basename to exclude in addition to heavyweight defaults.",
    )
    parser.add_argument(
        "--include-all-files",
        action="store_true",
        help="Include files without a known evidence extension.",
    )
    parser.add_argument(
        "--hash-max-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="Do not hash files larger than this many bytes (default: 64 MiB).",
    )
    parser.add_argument("--hash-workers", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=250_000)
    parser.add_argument("--markdown-limit", type=int, default=2_000)
    parser.add_argument(
        "--output-md", type=Path, default=supp / "audit/file_inventory_auto.md"
    )
    parser.add_argument(
        "--output-json", type=Path, default=supp / "raw_manifest/repository_inventory.json"
    )
    parser.add_argument(
        "--hash-manifest", type=Path, default=supp / "raw_manifest/file_hashes.jsonl"
    )
    parser.add_argument(
        "--provenance", type=Path, default=supp / "derived/inventory_provenance.json"
    )
    parser.add_argument("--seed", type=int, default=ANALYSIS_SEED)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def is_evidence_file(path: Path, include_all: bool) -> bool:
    if include_all:
        return True
    name = path.name.lower()
    return path.suffix.lower() in EVIDENCE_SUFFIXES or "tfevents" in name


def category(path: Path) -> str:
    name = path.name.lower()
    full = path.as_posix().lower()
    suffix = path.suffix.lower()
    if suffix == ".bib":
        return "bibliography"
    if suffix in {".tex", ".pdf"} and any(token in full for token in ("author", "paper", "main", "supp")):
        return "manuscript"
    if "tfevents" in name:
        return "tensorboard_export"
    if "wandb" in full:
        return "wandb_export"
    if suffix in {".yaml", ".yml", ".toml", ".cfg"} or "config" in name:
        return "configuration"
    if suffix == ".sh" or any(token in name for token in ("launch", "train", "eval")):
        return "launch_or_evaluation_code"
    if any(token in name for token in ("checkpoint", "trainer_state", "state_dict")):
        return "checkpoint_metadata_or_weight"
    if suffix in {".csv", ".json", ".jsonl", ".parquet", ".pkl", ".pickle"}:
        if any(token in full for token in ("scene", "metric", "eval", "result", "score", "rollout")):
            return "evaluation_or_scene_result"
        return "structured_data"
    if any(token in full for token in ("trajectory", "rollout", "teacher", "candidate")):
        return "trajectory_or_teacher_artifact"
    if suffix == ".py":
        return "source_code"
    if suffix in {".md", ".txt", ".log", ".out"}:
        return "report_or_log"
    return "other_evidence"


def discover_files(args: argparse.Namespace) -> tuple[list[Path], list[str]]:
    repo = args.repo_root.resolve()
    excluded = DEFAULT_EXCLUDED_DIRS | set(args.exclude_dir)
    requested = args.scan_roots or DEFAULT_SCAN_ROOTS
    warnings: list[str] = []
    discovered: set[Path] = set()

    # Root-level evidence is cheap and prevents missing a main manuscript.
    for path in repo.iterdir():
        if path.is_file() and is_evidence_file(path, args.include_all_files):
            discovered.add(path)

    for item in requested:
        scan_root = (repo / item).resolve() if not Path(item).is_absolute() else Path(item).resolve()
        try:
            scan_root.relative_to(repo)
        except ValueError as exc:
            raise ValueError(f"scan root must be inside repository: {scan_root}") from exc
        if not scan_root.exists():
            warnings.append(f"scan root does not exist: {relative_display(scan_root, repo)}")
            continue
        if scan_root.is_file():
            if is_evidence_file(scan_root, args.include_all_files):
                discovered.add(scan_root)
            continue
        for current, dirs, files in os.walk(scan_root):
            dirs[:] = sorted(directory for directory in dirs if directory not in excluded)
            current_path = Path(current)
            for filename in sorted(files):
                path = current_path / filename
                if is_evidence_file(path, args.include_all_files):
                    discovered.add(path)
                    if len(discovered) > args.max_files:
                        raise RuntimeError(
                            f"inventory exceeds --max-files={args.max_files}; narrow --scan-root"
                        )
    return sorted(discovered), warnings


def make_record(path: Path, repo: Path, hash_max_bytes: int) -> FileRecord:
    stat = path.stat()
    modified = __import__("datetime").datetime.fromtimestamp(
        stat.st_mtime, tz=__import__("datetime").timezone.utc
    ).isoformat()
    if stat.st_size > hash_max_bytes:
        digest = None
        status = f"skipped_above_{hash_max_bytes}_bytes"
    else:
        try:
            digest = sha256_file(path)
            status = "hashed"
        except OSError as exc:
            digest = None
            status = f"read_error:{type(exc).__name__}"
    return FileRecord(
        path=relative_display(path, repo),
        category=category(path.relative_to(repo)),
        suffix=path.suffix.lower(),
        bytes=stat.st_size,
        modified_utc=modified,
        sha256=digest,
        hash_status=status,
    )


def render_markdown(
    records: list[FileRecord], warnings: list[str], args: argparse.Namespace
) -> str:
    counts = Counter(record.category for record in records)
    hashed = sum(record.sha256 is not None for record in records)
    lines = [
        "# Automatically Generated File Inventory",
        "",
        "This is a reproducible filesystem inventory, not a claim-verification report. "
        "Paths are repository-relative; heavyweight caches and environments are excluded by default.",
        "",
        "## Summary",
        "",
        f"- Evidence-bearing files indexed: {len(records)}",
        f"- Files with SHA-256 hashes: {hashed}",
        f"- Files whose hashes were skipped or failed: {len(records) - hashed}",
        f"- Analysis seed (tooling only): {args.seed}",
        "",
        markdown_table(
            ["Category", "Files"],
            [(name, count) for name, count in sorted(counts.items())],
        ),
        "",
        "## Scope notes",
        "",
        "- Default excluded directory basenames: "
        + ", ".join(f"`{name}`" for name in sorted(DEFAULT_EXCLUDED_DIRS))
        + ".",
        "- The complete machine-readable inventory and hash records live under `raw_manifest/`.",
    ]
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(["", "## Indexed files", ""])
    visible = records[: args.markdown_limit]
    lines.append(
        markdown_table(
            ["Path", "Category", "Bytes", "SHA-256", "Hash status"],
            [
                (
                    f"`{record.path}`",
                    record.category,
                    record.bytes,
                    record.sha256 or "",
                    record.hash_status,
                )
                for record in visible
            ],
        )
    )
    if len(records) > len(visible):
        lines.extend(
            [
                "",
                f"Markdown listing truncated at {len(visible)} files; the JSON inventory contains all {len(records)} records.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.hash_workers < 1 or args.hash_max_bytes < 0 or args.max_files < 1:
        raise ValueError("hash workers/max bytes/max files must be positive (max bytes may be zero)")
    repo = args.repo_root.resolve()
    if not (repo / ".git").exists():
        raise ValueError(f"--repo-root is not a Git worktree: {repo}")

    files, warnings = discover_files(args)
    logging.info("indexing %d evidence-bearing files", len(files))
    with ThreadPoolExecutor(max_workers=args.hash_workers) as executor:
        records = list(executor.map(lambda path: make_record(path, repo, args.hash_max_bytes), files))

    payload = {
        "schema_version": 1,
        "repository": ".",
        "scan_roots": args.scan_roots or DEFAULT_SCAN_ROOTS,
        "excluded_directory_basenames": sorted(DEFAULT_EXCLUDED_DIRS | set(args.exclude_dir)),
        "warnings": warnings,
        "files": [asdict(record) for record in records],
    }
    atomic_write_json(args.output_json, payload)
    args.hash_manifest.parent.mkdir(parents=True, exist_ok=True)
    hash_lines = [json.dumps(asdict(record), ensure_ascii=False) for record in records]
    atomic_write_text(args.hash_manifest, "\n".join(hash_lines) + ("\n" if hash_lines else ""))
    atomic_write_text(args.output_md, render_markdown(records, warnings, args))

    provenance = build_provenance(
        seed=args.seed,
        # Per-file hashes already live in output_json/hash_manifest. Re-hashing
        # every repository file here would double audit time and duplicate a
        # potentially very large manifest inside provenance.
        inputs=[],
        outputs=[args.output_md, args.output_json, args.hash_manifest],
        status="complete",
        notes=warnings
        + ["Per-file input paths and SHA-256 values are stored in the inventory outputs."],
        root=repo,
    )
    atomic_write_json(args.provenance, provenance)
    logging.info("wrote %s", relative_display(args.output_md, repo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
