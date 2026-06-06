#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


PDM_COLUMNS = (
    "score",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
)
TOKEN_COLUMNS = ("token", "sample_token", "scene_token")
CHECKPOINT_SUFFIXES = (".ckpt", ".pt", ".pth", ".safetensors", ".bin")
CANDIDATE_SUFFIXES = (".jsonl", ".csv", ".npz", ".parquet")
SKIP_DIRS = {".git", "__pycache__", "wandb", "hf_home", "modelscope_cache", "node_modules", "vggt_py310_conda"}


def line_count(path: Path, limit: Optional[int] = None) -> int:
    count = 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for count, _line in enumerate(handle, start=1):
                if limit is not None and count >= limit:
                    break
    except OSError:
        return 0
    return count


def infer_split(path: Path) -> str:
    text = str(path).lower()
    if "navtest" in text or "test" in text:
        return "navtest"
    if "val" in text:
        return "val"
    if "train" in text or "backfill" in text:
        return "train"
    return "unknown"


def infer_method(path: Path) -> str:
    text = str(path).lower()
    if "risk_vla" in text or "riskvla" in text:
        return "risk_vla"
    if "d5" in text:
        return "d5"
    if "bit" in text or "b3" in text or "b5" in text or "b6" in text:
        return "bit"
    if "a0" in text or "base" in text or "recogdrive-2b-il" in text:
        return "a0_base"
    if "jepa" in text:
        return "jepa"
    if "vggt" in text:
        return "vggt"
    if "last" in text:
        return "last_rd"
    return "unknown"


def iter_files(roots: Sequence[Path], suffixes: Sequence[str], *, max_depth: int) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        stack = [(root.resolve(), 0)]
        while stack:
            current, depth = stack.pop()
            if depth > max_depth:
                continue
            try:
                entries = sorted(current.iterdir())
            except OSError:
                continue
            for entry in entries:
                if entry.is_dir():
                    if entry.name in SKIP_DIRS:
                        continue
                    stack.append((entry, depth + 1))
                elif entry.suffix.lower() in suffixes:
                    resolved = entry.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        yield resolved


def discover_chunk_roots(roots: Sequence[Path], *, max_depth: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Path] = set()
    for index_path in iter_files(roots, (".jsonl",), max_depth=max_depth):
        if index_path.name != "index.jsonl":
            continue
        chunk_dir = index_path.parent
        if chunk_dir in seen:
            continue
        seen.add(chunk_dir)
        count = line_count(index_path)
        rows.append(
            {
                "path": str(chunk_dir),
                "index_path": str(index_path),
                "split": infer_split(chunk_dir),
                "token_count": count,
                "sample_dir_exists": (chunk_dir / "samples").is_dir(),
                "metadata_path": str(chunk_dir / "metadata.json") if (chunk_dir / "metadata.json").is_file() else None,
            }
        )
    rows.sort(key=lambda item: (item["split"], item["path"]))
    return rows


def csv_header(path: Path) -> List[str]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.reader(handle)
            return [str(item) for item in next(reader, [])]
    except Exception:
        return []


def inspect_pdm_csv(path: Path, *, count_limit: int) -> Optional[Dict[str, Any]]:
    header = csv_header(path)
    if not header:
        return None
    token_col = next((column for column in TOKEN_COLUMNS if column in header), None)
    present = [column for column in PDM_COLUMNS if column in header]
    if token_col is None and len(present) < 3:
        return None
    return {
        "path": str(path),
        "split": infer_split(path),
        "method": infer_method(path),
        "row_count": max(0, line_count(path, limit=count_limit + 1) - 1),
        "row_count_is_lower_bound": line_count(path, limit=count_limit + 2) >= count_limit + 2,
        "token_column": token_col,
        "metric_columns": present,
        "missing_metric_columns": [column for column in PDM_COLUMNS if column not in header],
    }


def discover_pdm_csvs(roots: Sequence[Path], *, max_depth: int, count_limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in iter_files(roots, (".csv",), max_depth=max_depth):
        item = inspect_pdm_csv(path, count_limit=count_limit)
        if item is not None:
            rows.append(item)
    rows.sort(key=lambda item: (item["split"], -int(item["row_count"]), item["path"]))
    return rows


def inspect_candidate_file(path: Path, *, count_limit: int) -> Dict[str, Any]:
    count: Optional[int] = None
    lower_bound = False
    text = str(path).lower()
    should_count = not ("prediction" in text and "candidate" not in text and "counterfactual" not in text and "utility" not in text)
    if should_count and path.suffix.lower() in {".jsonl", ".csv"}:
        header_offset = 1 if path.suffix.lower() == ".csv" else 0
        count = max(0, line_count(path, limit=count_limit + header_offset) - header_offset)
        lower_bound = line_count(path, limit=count_limit + header_offset + 1) >= count_limit + header_offset + 1
    elif path.suffix.lower() == ".npz":
        try:
            import numpy as np

            with np.load(path, allow_pickle=False) as data:
                for key in ("sample_tokens", "tokens", "trajectories"):
                    if key in data:
                        count = int(data[key].shape[0])
                        break
        except Exception:
            count = None
    return {
        "path": str(path),
        "split": infer_split(path),
        "method": infer_method(path),
        "row_or_token_count": count,
        "row_or_token_count_is_lower_bound": lower_bound,
        "kind": path.suffix.lower().lstrip("."),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def discover_candidate_files(roots: Sequence[Path], *, max_depth: int, count_limit: int) -> List[Dict[str, Any]]:
    keywords = ("candidate", "counterfactual", "prediction", "trajectory", "utility")
    rows = []
    for path in iter_files(roots, CANDIDATE_SUFFIXES, max_depth=max_depth):
        text = str(path).lower()
        if any(keyword in text for keyword in keywords):
            rows.append(inspect_candidate_file(path, count_limit=count_limit))
    rows.sort(key=lambda item: (item["split"], -(item["row_or_token_count"] or 0), item["path"]))
    return rows


def discover_checkpoints(roots: Sequence[Path], *, max_depth: int) -> List[Dict[str, Any]]:
    rows = []
    for path in iter_files(roots, CHECKPOINT_SUFFIXES, max_depth=max_depth):
        rows.append(
            {
                "path": str(path),
                "method": infer_method(path),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "is_debug": any(token in str(path).lower() for token in ("debug", "smoke", "pilot256", "subset1024")),
            }
        )
    rows.sort(key=lambda item: (item["method"], item["is_debug"], item["path"]))
    return rows


def discover_metric_caches(roots: Sequence[Path], *, max_depth: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        stack = [(root.resolve(), 0)]
        while stack:
            current, depth = stack.pop()
            if depth > max_depth:
                continue
            name = current.name.lower()
            if "metric_cache" in name:
                metadata = sorted((current / "metadata").glob("*.csv")) if (current / "metadata").is_dir() else []
                rows.append(
                    {
                        "path": str(current),
                        "split": infer_split(current),
                        "metadata_csvs": [str(item) for item in metadata[:5]],
                        "metadata_rows": sum(max(0, line_count(item) - 1) for item in metadata),
                    }
                )
                continue
            try:
                for child in sorted(current.iterdir()):
                    if child.is_dir() and child.name not in SKIP_DIRS:
                        stack.append((child, depth + 1))
            except OSError:
                continue
    rows.sort(key=lambda item: (item["split"], -int(item["metadata_rows"]), item["path"]))
    return rows


def discover_expert_caches(roots: Sequence[Path], *, max_depth: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        candidates = [
            root / "last_vla_v2",
            root / "last_vla_v2" / "highcap_no_risk" / "train_geometry192_overlay",
            root / "last_vla_v2" / "highcap_no_risk" / "train_jepa128_overlay",
            root / "last_vla_v2" / "highcap_no_risk" / "train_full_highcap_chunks",
            root / "teachers",
            root / "recogdrive_expert_chunks",
        ]
        for current in candidates:
            if not current.is_dir():
                continue
            index_count = int((current / "index.jsonl").is_file())
            try:
                index_count += sum(1 for child in current.iterdir() if child.is_dir() and (child / "index.jsonl").is_file())
            except OSError:
                pass
            rows.append({"path": str(current), "kind": infer_method(current), "index_file_count": index_count})
    unique: Dict[str, Dict[str, Any]] = {row["path"]: row for row in rows}
    return sorted(unique.values(), key=lambda item: item["path"])


def system_info(paths: Sequence[Path]) -> Dict[str, Any]:
    gpus: List[Dict[str, Any]] = []
    try:
        import torch

        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                gpus.append({"index": idx, "name": torch.cuda.get_device_name(idx), "total_memory_gb": round(props.total_memory / 1e9, 2)})
    except Exception as exc:
        gpus.append({"error": repr(exc)})
    disks = []
    seen: set[str] = set()
    for path in paths:
        target = path if path.exists() else path.parent
        try:
            usage = shutil.disk_usage(target)
            key = str(target.resolve())
            if key in seen:
                continue
            seen.add(key)
            disks.append(
                {
                    "path": str(target),
                    "total_gb": round(usage.total / 1e9, 2),
                    "used_gb": round(usage.used / 1e9, 2),
                    "free_gb": round(usage.free / 1e9, 2),
                }
            )
        except OSError:
            continue
    return {"cpu_count": os.cpu_count(), "gpus": gpus, "disks": disks}


def summarize_chunks(chunks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_split: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"chunk_count": 0, "token_count": 0, "paths": []})
    for chunk in chunks:
        split = str(chunk["split"])
        by_split[split]["chunk_count"] += 1
        by_split[split]["token_count"] += int(chunk["token_count"])
        by_split[split]["paths"].append(chunk["path"])
    return dict(by_split)


def readiness(report: Dict[str, Any], min_samples: int) -> Dict[str, Any]:
    chunks = report["chunk_summary"]
    pdms = report["pdm_csvs"]
    candidates = report["candidate_files"]
    checkpoints = report["checkpoints"]
    train_tokens = int(chunks.get("train", {}).get("token_count", 0))
    val_tokens = int(chunks.get("val", {}).get("token_count", 0))
    navtest_tokens = int(chunks.get("navtest", {}).get("token_count", 0))
    train_pdm = [row for row in pdms if row["split"] == "train" and row["row_count"] >= min_samples]
    val_pdm = [row for row in pdms if row["split"] == "val" and row["row_count"] >= min_samples]
    navtest_pdm = [row for row in pdms if row["split"] == "navtest" and row["row_count"] >= min_samples]
    large_train_candidates = [row for row in candidates if row["split"] == "train" and (row["row_or_token_count"] or 0) >= min_samples]
    base_ckpt = [row for row in checkpoints if row["method"] == "a0_base"]
    bit_ckpt = [row for row in checkpoints if row["method"] == "bit" and not row["is_debug"]]
    risk_ckpt = [row for row in checkpoints if row["method"] == "risk_vla" and not row["is_debug"]]
    blockers = []
    if train_tokens < min_samples:
        blockers.append(f"train chunk tokens below {min_samples}: {train_tokens}")
    if val_tokens < min_samples:
        blockers.append(f"held-out val chunk tokens below {min_samples}: {val_tokens}; do not substitute navtest for tuning")
    if navtest_tokens < min_samples:
        blockers.append(f"navtest chunk tokens below {min_samples}: {navtest_tokens}")
    if not train_pdm:
        blockers.append("no train split PDM CSV with >=10k rows found for utility-label construction")
    if not val_pdm:
        blockers.append("no val split PDM CSV with >=10k rows found for held-out tuning")
    if not base_ckpt:
        blockers.append("A0/Base checkpoint missing")
    return {
        "min_real_samples": min_samples,
        "train_10k_possible": train_tokens >= min_samples,
        "val_10k_possible": val_tokens >= min_samples,
        "navtest_10k_possible": navtest_tokens >= min_samples,
        "large_train_utility_labels_possible": len(train_pdm) >= 2,
        "large_val_utility_labels_possible": len(val_pdm) >= 2,
        "large_candidate_assets_found": bool(large_train_candidates),
        "base_checkpoint_found": bool(base_ckpt),
        "bit_checkpoint_found": bool(bit_ckpt),
        "risk_vla_checkpoint_found": bool(risk_ckpt),
        "navtest_analysis_pdm_found": bool(navtest_pdm),
        "blockers": blockers,
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# RISK-VLA v3 Full Input Discovery Report",
        "",
        f"Git commit: `{report['git_commit']}`",
        f"Minimum real scale: `{report['readiness']['min_real_samples']}` tokens",
        "",
        "## Chunk Caches",
        "",
        "| Split | Chunks | Tokens | >=10k |",
        "| --- | ---: | ---: | --- |",
    ]
    for split, item in sorted(report["chunk_summary"].items()):
        lines.append(f"| {split} | {item['chunk_count']} | {item['token_count']} | {item['token_count'] >= report['readiness']['min_real_samples']} |")
    lines.extend(["", "## Metric Caches", ""])
    for row in report["metric_caches"][:20]:
        lines.append(f"- `{row['path']}` split=`{row['split']}` metadata_rows=`{row['metadata_rows']}`")
    lines.extend(["", "## PDM CSV Candidates", ""])
    for row in report["pdm_csvs"][:40]:
        lines.append(f"- `{row['path']}` split=`{row['split']}` method=`{row['method']}` rows=`{row['row_count']}`")
    lines.extend(["", "## Candidate / Counterfactual Assets", ""])
    for row in report["candidate_files"][:40]:
        lines.append(f"- `{row['path']}` split=`{row['split']}` method=`{row['method']}` count=`{row['row_or_token_count']}`")
    lines.extend(["", "## Checkpoints", ""])
    for row in report["checkpoints"][:50]:
        lines.append(f"- `{row['path']}` method=`{row['method']}` debug=`{row['is_debug']}`")
    lines.extend(["", "## Expert / World Token Caches", ""])
    for row in report["expert_caches"][:40]:
        lines.append(f"- `{row['path']}` kind=`{row['kind']}` index_files=`{row['index_file_count']}`")
    lines.extend(["", "## System", ""])
    lines.append(f"- CPU count: `{report['system']['cpu_count']}`")
    for gpu in report["system"]["gpus"]:
        lines.append(f"- GPU: `{gpu}`")
    for disk in report["system"]["disks"]:
        lines.append(f"- Disk: `{disk}`")
    lines.extend(["", "## Readiness", ""])
    for key, value in report["readiness"].items():
        if key != "blockers":
            lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Blockers", ""])
    blockers = report["readiness"]["blockers"]
    if blockers:
        lines.extend(f"- {item}" for item in blockers)
    else:
        lines.append("_None._")
    lines.extend(
        [
            "",
            "## Leakage Guard",
            "",
            "Navtest/test PDM labels are analysis-only. Discovery reports them, but the manifest builder must not mark them as training labels, router supervision, threshold tuning, hard-negative mining, or VLM instruction data.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
    except Exception:
        return "unknown"


def discover(args: argparse.Namespace) -> Dict[str, Any]:
    roots = [Path(item).expanduser() for item in args.search_root]
    if not roots:
        roots = [args.project_root, args.work_root]
    cache_roots = [args.project_root / "cache", args.work_root / "cache", args.cache_root]
    exp_roots = [args.project_root / "experiments", args.work_root / "experiments", args.exp_root]
    checkpoint_roots = [args.project_root / "checkpoints", args.work_root / "checkpoints"]
    chunk_roots = [
        args.project_root / "cache" / "recogdrive_expert_chunks",
        args.project_root / "cache" / "last_vla_v2" / "highcap_no_risk" / "train_full_highcap_chunks",
        args.work_root / "cache",
        args.cache_root,
    ]
    chunks = discover_chunk_roots(chunk_roots, max_depth=args.chunk_max_depth)
    report: Dict[str, Any] = {
        "git_commit": git_commit(args.project_root),
        "paths": {
            "project_root": str(args.project_root),
            "work_root": str(args.work_root),
            "exp_root": str(args.exp_root),
            "cache_root": str(args.cache_root),
            "search_roots": [str(root) for root in roots],
        },
        "chunks": chunks,
        "chunk_summary": summarize_chunks(chunks),
        "metric_caches": discover_metric_caches(cache_roots + exp_roots, max_depth=args.max_depth),
        "pdm_csvs": discover_pdm_csvs(exp_roots + [args.project_root / "outputs"], max_depth=args.max_depth, count_limit=int(args.min_real_samples) + 2),
        "candidate_files": discover_candidate_files(exp_roots, max_depth=args.max_depth, count_limit=int(args.min_real_samples) + 2),
        "checkpoints": discover_checkpoints(checkpoint_roots + exp_roots, max_depth=args.max_depth),
        "expert_caches": discover_expert_caches(cache_roots + checkpoint_roots, max_depth=args.max_depth),
        "system": system_info([args.project_root, args.work_root, args.exp_root, args.cache_root]),
    }
    report["readiness"] = readiness(report, int(args.min_real_samples))
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover full-scale RISK-VLA v3 train/val/navtest inputs.")
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/project/VLA-AD"))
    parser.add_argument("--work-root", type=Path, default=Path(os.getenv("BIT_WORK_ROOT", "/mnt/project/bit_drive_left_tail")))
    parser.add_argument("--exp-root", type=Path, default=Path(os.getenv("BIT_EXP_ROOT", "/mnt/project/bit_drive_left_tail/experiments/risk_vla_v3")))
    parser.add_argument("--cache-root", type=Path, default=Path(os.getenv("BIT_CACHE_ROOT", "/mnt/project/bit_drive_left_tail/cache/risk_vla_v3")))
    parser.add_argument("--search-root", action="append", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--chunk-max-depth", type=int, default=4)
    parser.add_argument("--min-real-samples", type=int, default=10000)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = discover(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, args.output_md)
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md), "readiness": report["readiness"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
