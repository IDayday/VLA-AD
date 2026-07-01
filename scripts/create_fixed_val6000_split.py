#!/usr/bin/env python3
"""Create a deterministic fixed navtrain val6000 token split for PSI-Drive."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{line}\n" for line in lines)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def read_token_file(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    tokens = set()
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            token = raw.strip()
            if token:
                tokens.add(token)
    return tokens


def load_support_tokens(index_path: Path) -> list[str]:
    payload = torch.load(index_path, map_location="cpu")
    tokens = [str(token) for token in payload.get("tokens", [])]
    if not tokens:
        raise ValueError(f"No tokens found in support index: {index_path}")
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"Duplicate tokens found in support index: {index_path}")
    return sorted(tokens)


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/splits"))
    parser.add_argument("--seed", type=int, default=260306049)
    parser.add_argument("--num-val", type=int, default=6000)
    parser.add_argument("--prefix", default="navtrain_val6000_seed260306049")
    parser.add_argument("--navtest-tokens", type=Path, default=None)
    parser.add_argument("--fail-on-navtest-overlap", action="store_true")
    args = parser.parse_args()

    tokens = load_support_tokens(args.support_index)
    if args.num_val <= 0 or args.num_val >= len(tokens):
        raise ValueError(f"--num-val must be in [1, {len(tokens) - 1}], got {args.num_val}")

    rng = random.Random(args.seed)
    shuffled = list(tokens)
    rng.shuffle(shuffled)
    val_tokens = sorted(shuffled[: args.num_val])
    train_tokens = sorted(shuffled[args.num_val :])
    if len(val_tokens) != args.num_val or len(set(val_tokens)) != args.num_val:
        raise RuntimeError("val token selection is not unique or has wrong size")
    train_val_overlap = sorted(set(train_tokens) & set(val_tokens))
    if train_val_overlap:
        raise RuntimeError(f"Train/val overlap is not allowed: {train_val_overlap[:5]}")

    navtest_tokens = read_token_file(args.navtest_tokens)
    navtest_overlap = None
    if navtest_tokens is not None:
        navtest_overlap = sorted(set(val_tokens) & navtest_tokens)
        if navtest_overlap and args.fail_on_navtest_overlap:
            raise RuntimeError(f"Val/navtest overlap is not allowed: {navtest_overlap[:5]}")

    output_dir = args.output_dir
    val_path = output_dir / f"{args.prefix}.txt"
    train_path = output_dir / f"{args.prefix}_train_complement.txt"
    write_lines(val_path, val_tokens)
    write_lines(train_path, train_tokens)
    val_sha = sha256_file(val_path)
    train_sha = sha256_file(train_path)
    (val_path.with_suffix(val_path.suffix + ".sha256")).write_text(f"{val_sha}  {val_path.name}\n", encoding="utf-8")
    (train_path.with_suffix(train_path.suffix + ".sha256")).write_text(
        f"{train_sha}  {train_path.name}\n", encoding="utf-8"
    )

    audit = {
        "created_at": utc_now(),
        "git_branch": git_value(["branch", "--show-current"]),
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "support_index": str(args.support_index),
        "seed": args.seed,
        "num_source_tokens": len(tokens),
        "num_val_tokens": len(val_tokens),
        "num_train_complement_tokens": len(train_tokens),
        "val_token_file": str(val_path),
        "val_token_file_sha256": val_sha,
        "train_complement_file": str(train_path),
        "train_complement_file_sha256": train_sha,
        "train_val_overlap_count": len(train_val_overlap),
        "navtest_token_file": str(args.navtest_tokens) if args.navtest_tokens else None,
        "navtest_overlap_count": None if navtest_overlap is None else len(navtest_overlap),
        "navtest_overlap_tokens": None if navtest_overlap is None else navtest_overlap[:20],
    }
    audit_json = output_dir / f"{args.prefix}_audit.json"
    audit_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_md = output_dir / f"{args.prefix}_audit.md"
    lines = [
        "# PSI-Drive Fixed Val6000 Split Audit",
        "",
        f"- support_index: `{args.support_index}`",
        f"- seed: `{args.seed}`",
        f"- source tokens: `{len(tokens)}`",
        f"- val tokens: `{len(val_tokens)}`",
        f"- train complement tokens: `{len(train_tokens)}`",
        f"- val sha256: `{val_sha}`",
        f"- train complement sha256: `{train_sha}`",
        f"- train/val overlap: `{len(train_val_overlap)}`",
        f"- navtest overlap: `{audit['navtest_overlap_count']}`",
        "",
        "Use the `.txt` file with evaluator `--sample-token-file`; do not replace it with `--max-samples 6000`.",
    ]
    audit_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
