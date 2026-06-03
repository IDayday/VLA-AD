#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.bit_safety_selector import trajectory_delta_features  # noqa: E402
from scripts.build_bit_counterfactual_dataset import (  # noqa: E402
    compact_metrics,
    resolve_checkpoint_path,
    score_prediction,
    split_chunk_pattern,
)
from scripts.eval_bit_drive_pdm import (  # noqa: E402
    build_metric_cache_loader,
    build_pdm_tools,
    build_planner,
    chunk_dirs,
    dtype_from_precision,
    load_checkpoint,
    load_sample,
    load_yaml,
    make_batch,
    sample_paths,
    seed_everything,
)


METRIC_EPS = 1e-9
DEFAULT_BIT_WORK_ROOT = Path(os.environ.get("BIT_WORK_ROOT", "/mnt/project/bit_drive_left_tail"))
DEFAULT_BIT_EXP_ROOT = Path(os.environ.get("BIT_EXP_ROOT", DEFAULT_BIT_WORK_ROOT / "experiments/bit_drive"))


class CombinedMetricCacheLoader:
    def __init__(self, loaders: List[Any]) -> None:
        self.loaders = loaders
        self.metric_cache_paths: Dict[str, Any] = {}
        for loader in loaders:
            self.metric_cache_paths.update(getattr(loader, "metric_cache_paths", {}) or {})

    def get_from_token(self, token: str) -> Any:
        for loader in self.loaders:
            if token in getattr(loader, "metric_cache_paths", {}):
                return loader.get_from_token(token)
        raise KeyError(token)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Actively mine non-test Base-vs-BiT counterfactual safety cases.")
    parser.add_argument("--navsim-root", type=Path, default=Path("/mnt/navsim"))
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--bit-config", type=Path, required=True)
    parser.add_argument("--bit-checkpoint", type=Path, required=True)
    parser.add_argument("--split-aliases", default="navtrain,navval")
    parser.add_argument("--chunk-cache-root", type=Path, default=Path("/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1"))
    parser.add_argument("--chunk-name-pattern", default=None)
    parser.add_argument("--metric-cache-dir", type=Path, action="append", default=None)
    parser.add_argument("--source-counterfactual-jsonl", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-nc-regressions", type=int, default=100)
    parser.add_argument("--target-ttc-regressions", type=int, default=300)
    parser.add_argument("--target-dac-fixes", type=int, default=300)
    parser.add_argument("--max-total-samples", type=int, default=30000)
    parser.add_argument(
        "--sampling-mode",
        choices=("random", "stratified_by_command", "high_curvature", "high_speed", "endpoint_delta", "mixed"),
        default="mixed",
    )
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="fp32")
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def configure_navsim_env(navsim_root: Path) -> None:
    root = navsim_root.resolve()
    os.environ["NAVSIM_DATA_ROOT"] = str(root)
    os.environ.setdefault("OPENSCENE_DATA_ROOT", str(root / "openscene"))
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(root / "maps"))


def tensorish_flat_list(value: Any) -> List[float]:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        out: List[float] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                out.extend(tensorish_flat_list(item))
            else:
                try:
                    out.append(float(item))
                except (TypeError, ValueError):
                    pass
        return out
    return []


def history_xy(sample: Dict[str, Any]) -> Optional[torch.Tensor]:
    value = sample.get("history_trajectory")
    if not isinstance(value, torch.Tensor):
        return None
    value = value.detach().float()
    if value.ndim != 2 or value.shape[0] < 2 or value.shape[1] < 2:
        return None
    return value[:, :2]


def history_speed_score(sample: Dict[str, Any]) -> float:
    hist = history_xy(sample)
    if hist is not None:
        steps = torch.linalg.vector_norm(hist[1:] - hist[:-1], dim=-1)
        if torch.isfinite(steps).all() and steps.numel() > 0:
            return float(steps.mean().item())
    status = tensorish_flat_list(sample.get("status_feature"))
    if len(status) >= 2 and math.isfinite(status[0]) and math.isfinite(status[1]):
        return float(math.hypot(status[0], status[1]))
    return 0.0


def history_curvature_score(sample: Dict[str, Any]) -> float:
    hist = history_xy(sample)
    if hist is None or hist.shape[0] < 3:
        return 0.0
    delta = hist[1:] - hist[:-1]
    if delta.shape[0] < 2:
        return 0.0
    angles = torch.atan2(delta[:, 1], delta[:, 0])
    turn = torch.diff(angles)
    turn = torch.atan2(torch.sin(turn), torch.cos(turn)).abs()
    if not torch.isfinite(turn).all() or turn.numel() == 0:
        return 0.0
    return float(turn.mean().item())


def endpoint_complexity_score(sample: Dict[str, Any]) -> float:
    traj = sample.get("trajectory")
    if isinstance(traj, torch.Tensor) and traj.ndim == 2 and traj.shape[0] > 0 and traj.shape[1] >= 2:
        endpoint = traj.detach().float()[-1, :2]
        if torch.isfinite(endpoint).all():
            return float(torch.linalg.vector_norm(endpoint).item())
    hist = history_xy(sample)
    if hist is not None:
        endpoint = hist[-1] - hist[0]
        if torch.isfinite(endpoint).all():
            return float(torch.linalg.vector_norm(endpoint).item())
    return 0.0


def command_bucket(sample: Optional[Dict[str, Any]], record: Dict[str, Any]) -> str:
    if sample is not None and isinstance(sample.get("high_command_one_hot"), torch.Tensor):
        command = sample["high_command_one_hot"].detach().float().reshape(-1)
        if command.numel() > 0 and torch.isfinite(command).all():
            return f"cmd_{int(torch.argmax(command).item())}"
    for key in ("command", "high_command", "log_name"):
        value = (sample or {}).get(key, record.get(key))
        if value is not None:
            return str(value)
    return "unknown"


def scored_candidates(
    items: List[Tuple[Path, Path, Dict[str, Any], str]],
    *,
    max_scan: int,
) -> List[Tuple[Tuple[Path, Path, Dict[str, Any], str], Dict[str, Any]]]:
    scored: List[Tuple[Tuple[Path, Path, Dict[str, Any], str], Dict[str, Any]]] = []
    for item in items[:max_scan]:
        _chunk, path, record, _split = item
        try:
            sample = load_sample(path)
        except Exception as exc:
            scored.append((item, {"error": repr(exc), "command": command_bucket(None, record), "speed": 0.0, "curvature": 0.0, "endpoint": 0.0}))
            continue
        scored.append(
            (
                item,
                {
                    "command": command_bucket(sample, record),
                    "speed": history_speed_score(sample),
                    "curvature": history_curvature_score(sample),
                    "endpoint": endpoint_complexity_score(sample),
                },
            )
        )
    return scored


def round_robin_by_key(
    scored: List[Tuple[Tuple[Path, Path, Dict[str, Any], str], Dict[str, Any]]],
    key: str,
) -> List[Tuple[Path, Path, Dict[str, Any], str]]:
    groups: Dict[str, List[Tuple[Path, Path, Dict[str, Any], str]]] = defaultdict(list)
    for item, meta in scored:
        groups[str(meta.get(key, "unknown"))].append(item)
    ordered: List[Tuple[Path, Path, Dict[str, Any], str]] = []
    while groups:
        for group_key in sorted(list(groups)):
            group = groups[group_key]
            if group:
                ordered.append(group.pop(0))
            if not group:
                groups.pop(group_key, None)
    return ordered


def unique_take(
    items: Iterable[Tuple[Path, Path, Dict[str, Any], str]],
    limit: int,
) -> List[Tuple[Path, Path, Dict[str, Any], str]]:
    out: List[Tuple[Path, Path, Dict[str, Any], str]] = []
    seen: set[str] = set()
    for item in items:
        token = str(item[2].get("sample_token") or item[1])
        if token in seen:
            continue
        seen.add(token)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def metric_value(metrics: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not metrics or metrics.get(key) is None:
        return None
    return float(metrics[key])


def tag_counterfactual(base_metrics: Optional[Dict[str, Any]], bit_metrics: Optional[Dict[str, Any]]) -> List[str]:
    tags: List[str] = []
    if base_metrics is None or bit_metrics is None:
        return ["unusable"]
    base_pdm = metric_value(base_metrics, "pdm")
    bit_pdm = metric_value(bit_metrics, "pdm")
    base_dac = metric_value(base_metrics, "dac")
    bit_dac = metric_value(bit_metrics, "dac")
    base_nc = metric_value(base_metrics, "nc")
    bit_nc = metric_value(bit_metrics, "nc")
    base_ttc = metric_value(base_metrics, "ttc")
    bit_ttc = metric_value(bit_metrics, "ttc")
    if base_dac is not None and bit_dac is not None and base_dac <= METRIC_EPS and bit_dac > METRIC_EPS:
        tags.append("bit_dac_fix")
    if base_nc is not None and bit_nc is not None and base_nc > METRIC_EPS and bit_nc <= METRIC_EPS:
        tags.append("bit_nc_regression")
    if base_ttc is not None and bit_ttc is not None and base_ttc > METRIC_EPS and bit_ttc <= METRIC_EPS:
        tags.append("bit_ttc_regression")
    if base_pdm is not None and bit_pdm is not None and base_pdm <= METRIC_EPS and bit_pdm > METRIC_EPS:
        tags.append("bit_zero_fix")
    if base_pdm is not None and bit_pdm is not None and bit_pdm + METRIC_EPS < base_pdm:
        tags.append("bit_bad")
    return tags or ["neutral"]


def delta_metrics(base_metrics: Optional[Dict[str, Any]], bit_metrics: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    delta: Dict[str, Optional[float]] = {}
    for key in ("pdm", "dac", "nc", "ttc", "ego", "comfort"):
        base_value = metric_value(base_metrics, key)
        bit_value = metric_value(bit_metrics, key)
        delta[key] = None if base_value is None or bit_value is None else bit_value - base_value
    return delta


def add_tags_and_features(row: Dict[str, Any], split: str) -> Dict[str, Any]:
    base_metrics = row.get("base_metrics")
    bit_metrics = row.get("bit_metrics")
    base_traj = row.get("base_pred_traj")
    bit_traj = row.get("bit_pred_traj")
    tags = tag_counterfactual(base_metrics, bit_metrics)
    delta = delta_metrics(base_metrics, bit_metrics)
    out = dict(row)
    out["split"] = split
    out["tags"] = tags
    out["delta"] = delta
    out["delta_metrics"] = delta
    out.setdefault("feature_dict", {})
    out.setdefault("command_ego_history_features", out.get("feature_dict") or {})
    if base_traj is not None and bit_traj is not None:
        out["features"] = trajectory_delta_features(base_traj, bit_traj)
    out["label_use_bit"] = int(
        "bit_bad" not in tags
        and "bit_nc_regression" not in tags
        and "bit_ttc_regression" not in tags
        and (out["delta"].get("pdm") is not None and out["delta"]["pdm"] > 0.0)
    )
    return out


def auto_metric_cache_dirs(output_dir: Path, split_aliases: List[str]) -> List[Path]:
    roots = [
        output_dir,
        DEFAULT_BIT_EXP_ROOT / "select",
        DEFAULT_BIT_WORK_ROOT / "cache",
        Path("/mnt/project/VLA-AD/experiments/bit_drive/select"),
        Path("/mnt/project/VLA-AD/cache"),
    ]
    split_terms = {alias.lower().replace("nav", "") for alias in split_aliases}
    split_terms.update(alias.lower() for alias in split_aliases)
    paths: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("metric_cache*"):
            if not path.is_dir():
                continue
            name = str(path).lower()
            if "navtest" in name or "/test" in name:
                continue
            if any(term and term in name for term in split_terms):
                paths.append(path)
    return sorted(set(paths))


def metric_loader_from_dirs(paths: List[Path]) -> Optional[CombinedMetricCacheLoader]:
    loaders = []
    for path in paths:
        try:
            loaders.append(build_metric_cache_loader(path))
        except Exception:
            continue
    if not loaders:
        return None
    return CombinedMetricCacheLoader(loaders)


def rows_from_source_jsonl(paths: List[Path], split: str, max_rows: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    seen_tokens: set[str] = set()
    for path in paths:
        metadata_path = path.parent / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            source_split = str(metadata.get("split", split)).lower()
            if "test" in source_split or metadata.get("training_allowed") is False or metadata.get("analysis_only"):
                rejected.append({"path": str(path), "error": "test_or_analysis_source"})
                continue
        for row in read_jsonl(path):
            if len(rows) >= max_rows:
                return rows, rejected
            row_split = str(row.get("split") or split)
            if "test" in row_split.lower():
                rejected.append({"sample_token": row.get("sample_token"), "split": row_split, "error": "test_split_row"})
                continue
            if row.get("base_pred_traj") is None or row.get("bit_pred_traj") is None:
                rejected.append({"sample_token": row.get("sample_token"), "error": "missing_trajectory"})
                continue
            if row.get("base_metrics") is None or row.get("bit_metrics") is None:
                rejected.append({"sample_token": row.get("sample_token"), "error": "missing_metrics"})
                continue
            sample_token = str(row.get("sample_token") or "")
            if sample_token:
                if sample_token in seen_tokens:
                    rejected.append({"sample_token": sample_token, "error": "duplicate_sample_token"})
                    continue
                seen_tokens.add(sample_token)
            rows.append(add_tags_and_features(row, split=row_split))
    return rows, rejected


def candidate_samples(args: argparse.Namespace, split_aliases: List[str]) -> List[Tuple[Path, Path, Dict[str, Any], str]]:
    items: List[Tuple[Path, Path, Dict[str, Any], str]] = []
    for split in split_aliases:
        local_args = argparse.Namespace(
            chunk_cache_dir=None,
            chunk_cache_root=args.chunk_cache_root,
            chunk_name_pattern=args.chunk_name_pattern or split_chunk_pattern(split),
            max_samples=None,
        )
        try:
            for chunk, path, record in sample_paths(chunk_dirs(local_args), None):
                items.append((chunk, path, record, split))
        except FileNotFoundError:
            continue
    rng = random.Random(args.seed)
    rng.shuffle(items)
    if not items:
        return []
    if args.sampling_mode == "random":
        return items[: args.max_total_samples]

    max_scan = min(len(items), args.max_total_samples)
    scored = scored_candidates(items, max_scan=max_scan)
    if args.sampling_mode == "stratified_by_command":
        return unique_take(round_robin_by_key(scored, "command"), args.max_total_samples)
    if args.sampling_mode == "high_curvature":
        ordered = [item for item, _meta in sorted(scored, key=lambda pair: float(pair[1].get("curvature", 0.0)), reverse=True)]
        return unique_take(ordered, args.max_total_samples)
    if args.sampling_mode == "high_speed":
        ordered = [item for item, _meta in sorted(scored, key=lambda pair: float(pair[1].get("speed", 0.0)), reverse=True)]
        return unique_take(ordered, args.max_total_samples)
    if args.sampling_mode == "endpoint_delta":
        ordered = [item for item, _meta in sorted(scored, key=lambda pair: float(pair[1].get("endpoint", 0.0)), reverse=True)]
        return unique_take(ordered, args.max_total_samples)

    random_order = [item for item, _meta in scored]
    command_order = round_robin_by_key(scored, "command")
    curvature_order = [item for item, _meta in sorted(scored, key=lambda pair: float(pair[1].get("curvature", 0.0)), reverse=True)]
    speed_order = [item for item, _meta in sorted(scored, key=lambda pair: float(pair[1].get("speed", 0.0)), reverse=True)]
    endpoint_order = [item for item, _meta in sorted(scored, key=lambda pair: float(pair[1].get("endpoint", 0.0)), reverse=True)]
    mixed: List[Tuple[Path, Path, Dict[str, Any], str]] = []
    cursors = [0, 0, 0, 0, 0]
    pools = [random_order, command_order, curvature_order, speed_order, endpoint_order]
    while len(mixed) < args.max_total_samples and any(cursor < len(pool) for cursor, pool in zip(cursors, pools)):
        for pool_index, pool in enumerate(pools):
            if cursors[pool_index] < len(pool):
                mixed.append(pool[cursors[pool_index]])
                cursors[pool_index] += 1
    return unique_take(mixed, args.max_total_samples)


def command_history_features(sample: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ("high_command_one_hot", "command", "status_feature", "history_trajectory"):
        value = sample.get(key)
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().float().cpu().reshape(-1).tolist()
        elif value is not None:
            out[key] = value
    return out


def rows_by_running_models(args: argparse.Namespace, metric_loader: CombinedMetricCacheLoader) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_precision(args.precision, device)
    base_planner = build_planner(load_yaml(args.base_config)).to(device)
    bit_planner = build_planner(load_yaml(args.bit_config)).to(device)
    if dtype != torch.float32:
        base_planner = base_planner.to(dtype=dtype)
        bit_planner = bit_planner.to(dtype=dtype)
    load_checkpoint(base_planner, resolve_checkpoint_path(args.base_checkpoint))
    load_checkpoint(bit_planner, resolve_checkpoint_path(args.bit_checkpoint))
    base_planner.eval()
    bit_planner.eval()
    pdm_tools = build_pdm_tools()

    rows: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    tag_counts: Counter[str] = Counter()
    split_aliases = [item.strip() for item in args.split_aliases.split(",") if item.strip()]
    for chunk, path, index_record, split in candidate_samples(args, split_aliases):
        sample = load_sample(path)
        sample_token = str(sample.get("sample_token", index_record.get("sample_token", path.stem)))
        scene_token = str(sample.get("scene_token", index_record.get("scene_token", path.stem)))
        if sample_token not in metric_loader.metric_cache_paths:
            rejected.append({"sample_token": sample_token, "scene_token": scene_token, "split": split, "error": "missing_metric_cache"})
            continue
        vl_features, action_input = make_batch(sample, device, dtype)
        with torch.no_grad():
            base_output = base_planner.get_action(vl_features, action_input, deterministic=args.deterministic)
            bit_output = bit_planner.get_action(vl_features, action_input, deterministic=args.deterministic)
        base_traj = base_output["pred_traj"].detach().float().cpu().squeeze(0)
        bit_traj = bit_output["pred_traj"].detach().float().cpu().squeeze(0)
        if not torch.isfinite(base_traj).all() or not torch.isfinite(bit_traj).all():
            rejected.append({"sample_token": sample_token, "scene_token": scene_token, "split": split, "error": "non_finite_prediction"})
            continue
        base_metrics = compact_metrics(score_prediction(sample_token=sample_token, pred=base_traj, metric_cache_loader=metric_loader, pdm_tools=pdm_tools))
        bit_metrics = compact_metrics(score_prediction(sample_token=sample_token, pred=bit_traj, metric_cache_loader=metric_loader, pdm_tools=pdm_tools))
        command_features = command_history_features(sample)
        row = add_tags_and_features(
            {
                "scene_token": scene_token,
                "sample_token": sample_token,
                "base_pred_traj": base_traj.tolist(),
                "bit_pred_traj": bit_traj.tolist(),
                "base_metrics": base_metrics,
                "bit_metrics": bit_metrics,
                "chunk": chunk.name,
                "feature_dict": command_features,
                "command_ego_history_features": command_features,
            },
            split=split,
        )
        rows.append(row)
        tag_counts.update(row["tags"])
        if (
            tag_counts["bit_nc_regression"] >= args.target_nc_regressions
            and tag_counts["bit_ttc_regression"] >= args.target_ttc_regressions
            and tag_counts["bit_dac_fix"] >= args.target_dac_fixes
        ):
            break
    return rows, rejected


def summarize(rows: List[Dict[str, Any]], rejected: List[Dict[str, Any]], args: argparse.Namespace, metric_dirs: List[Path]) -> Dict[str, Any]:
    tag_counts = Counter(tag for row in rows for tag in row.get("tags", []))
    split_counts = Counter(str(row.get("split")) for row in rows)
    split_scene_tokens: Dict[str, List[str]] = {}
    for split in sorted(split_counts):
        split_scene_tokens[split] = sorted(
            {str(row.get("scene_token")) for row in rows if str(row.get("split")) == split and row.get("scene_token")}
        )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_allowed": all("test" not in str(row.get("split", "")).lower() for row in rows),
        "split_aliases": [item.strip() for item in args.split_aliases.split(",") if item.strip()],
        "exact_splits": sorted(split_counts),
        "sampling_mode": args.sampling_mode,
        "total_rows": len(rows),
        "rejected_rows": len(rejected),
        "target_nc_regressions": args.target_nc_regressions,
        "target_ttc_regressions": args.target_ttc_regressions,
        "target_dac_fixes": args.target_dac_fixes,
        "tag_counts": dict(tag_counts),
        "split_counts": dict(split_counts),
        "scene_tokens": sorted({str(row.get("scene_token")) for row in rows if row.get("scene_token")}),
        "split_scene_tokens": split_scene_tokens,
        "sample_tokens": sorted({str(row.get("sample_token")) for row in rows if row.get("sample_token")}),
        "metric_cache_dirs": [str(path) for path in metric_dirs],
        "navsim_root": str(args.navsim_root),
        "base_config": str(args.base_config),
        "base_checkpoint": str(args.base_checkpoint),
        "bit_config": str(args.bit_config),
        "bit_checkpoint": str(args.bit_checkpoint),
        "max_total_samples": args.max_total_samples,
        "seed": args.seed,
        "deterministic": bool(args.deterministic),
    }


def write_summary_markdown(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# BiT Counterfactual Safety Mining Summary",
        "",
        f"Created at: `{summary['created_at']}`",
        f"Training allowed: `{summary['training_allowed']}`",
        f"Rows: `{summary['total_rows']}`",
        f"Rejected/unusable: `{summary['rejected_rows']}`",
        "",
        "| Tag | Count | Target |",
        "| --- | ---: | ---: |",
        f"| bit_nc_regression | {summary['tag_counts'].get('bit_nc_regression', 0)} | {summary['target_nc_regressions']} |",
        f"| bit_ttc_regression | {summary['tag_counts'].get('bit_ttc_regression', 0)} | {summary['target_ttc_regressions']} |",
        f"| bit_dac_fix | {summary['tag_counts'].get('bit_dac_fix', 0)} | {summary['target_dac_fixes']} |",
        f"| bit_zero_fix | {summary['tag_counts'].get('bit_zero_fix', 0)} | - |",
        f"| bit_bad | {summary['tag_counts'].get('bit_bad', 0)} | - |",
        f"| neutral | {summary['tag_counts'].get('neutral', 0)} | - |",
        "",
        "## Split Distribution",
        "",
        "```json",
        json.dumps(summary["split_counts"], indent=2, sort_keys=True),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.stochastic:
        args.deterministic = False
    configure_navsim_env(args.navsim_root)
    seed_everything(args.seed)
    split_aliases = [item.strip() for item in args.split_aliases.split(",") if item.strip()]
    if any("test" in split.lower() for split in split_aliases):
        raise RuntimeError("Refusing to mine training counterfactual labels from navtest/test splits.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]]
    rejected: List[Dict[str, Any]]
    metric_dirs = args.metric_cache_dir or auto_metric_cache_dirs(args.output_dir, split_aliases)
    if args.source_counterfactual_jsonl:
        rows, rejected = rows_from_source_jsonl(args.source_counterfactual_jsonl, split=split_aliases[0], max_rows=args.max_total_samples)
    else:
        metric_loader = metric_loader_from_dirs(metric_dirs)
        if metric_loader is None:
            raise RuntimeError(
                "No non-test metric cache is available. Provide --metric-cache-dir or --source-counterfactual-jsonl, "
                "or build a cache with scripts/run_metric_cache_for_chunk_subset.py."
            )
        rows, rejected = rows_by_running_models(args, metric_loader)

    summary = summarize(rows, rejected, args, metric_dirs)
    metadata = {
        **summary,
        "split": ",".join(summary["split_aliases"]),
        "source_counterfactual_jsonl": [str(path) for path in (args.source_counterfactual_jsonl or [])],
    }
    if not metadata["training_allowed"]:
        metadata["analysis_only"] = True
    else:
        metadata["analysis_only"] = False
    write_jsonl(args.output_dir / "counterfactual_mined.jsonl", rows)
    write_jsonl(args.output_dir / "rejected_or_unusable.jsonl", rejected)
    (args.output_dir / "mining_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary_markdown(args.output_dir / "mining_summary.md", summary)
    compact_summary = {key: value for key, value in summary.items() if key not in {"scene_tokens", "sample_tokens"}}
    print(json.dumps(compact_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
