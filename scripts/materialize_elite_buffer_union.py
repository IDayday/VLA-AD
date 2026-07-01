from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List


def _iter_records(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Elite buffer directory does not exist: {root}")
    yield from sorted(path for path in root.iterdir() if path.is_file() and path.name.endswith(".pkl.xz"))


def collect_union(source_roots: List[Path]) -> Dict[str, Path]:
    selected: Dict[str, Path] = {}
    for root in source_roots:
        for path in _iter_records(root):
            selected[path.name] = path.resolve()
    return selected


def _safe_unlink(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.exists():
        raise IsADirectoryError(f"Refusing to replace non-file path: {path}")


def materialize_union(source_roots: List[Path], output_root: Path, *, overwrite: bool = False) -> Dict[str, object]:
    selected = collect_union(source_roots)
    output_root.mkdir(parents=True, exist_ok=True)
    linked = 0
    skipped_existing = 0
    replaced = 0
    for name, src in selected.items():
        dst = output_root / name
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() and Path(os.readlink(dst)) == src:
                skipped_existing += 1
                continue
            if not overwrite:
                raise FileExistsError(f"Output path already exists and differs from selected source: {dst}")
            _safe_unlink(dst)
            replaced += 1
        os.symlink(src, dst)
        linked += 1
    source_counts = {str(root): sum(1 for _ in _iter_records(root)) for root in source_roots}
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_roots": [str(root) for root in source_roots],
        "source_counts": source_counts,
        "output_root": str(output_root),
        "union_records": len(selected),
        "linked": linked,
        "replaced": replaced,
        "skipped_existing": skipped_existing,
        "precedence": "later source_roots override earlier roots when file names collide",
    }
    (output_root / "union_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize a symlink union of AWAC/elite buffer record directories.")
    parser.add_argument("--source-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if len(args.source_root) < 1:
        raise ValueError("At least one --source-root is required.")
    summary = materialize_union(args.source_root, args.output_root, overwrite=bool(args.overwrite))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
