#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import lzma
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SOURCE_COLORS = {
    "gt": "#111111",
    "il": "#1f77b4",
    "recogdrive_stage3": "#1f77b4",
    "ddv2": "#ff7f0e",
    "driveor": "#2ca02c",
    "progress": "#9467bd",
    "lateral": "#17becf",
    "timing": "#8c564b",
    "fallback": "#7f7f7f",
}


TAG_COLORS = {
    "gt_anchor": "#111111",
    "il_anchor": "#1f77b4",
    "best_pdms": "#d62728",
    "safe_ep_improver": "#2ca02c",
    "safety_repair": "#bcbd22",
    "ddc_repair": "#17becf",
    "smooth_feasible": "#9467bd",
    "vector_pareto": "#ff7f0e",
    "diversity_max": "#e377c2",
    "fallback_best": "#7f7f7f",
}


@dataclass
class SceneScore:
    category: str
    token: str
    path: Path
    score: float
    payload: dict[str, Any]


def _load_payload(path: Path) -> dict[str, Any]:
    with lzma.open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(payload).__name__}")
    if "token" not in payload or "candidates" not in payload:
        raise ValueError(f"Unsupported support archive payload: {path}")
    return payload


def _path_for_token(archive_dir: Path, token: str) -> Path:
    return archive_dir / f"{hashlib.sha1(str(token).encode('utf-8')).hexdigest()}.pkl.xz"


def _iter_archive_paths(archive_dir: Path, tokens: Iterable[str], max_archives: int) -> Iterable[Path]:
    seen: set[Path] = set()
    for token in tokens:
        if not token:
            continue
        path = _path_for_token(archive_dir, token)
        if path.is_file() and path not in seen:
            seen.add(path)
            yield path
    count = 0
    for path in sorted(archive_dir.glob("*.pkl.xz")):
        if path in seen:
            continue
        yield path
        count += 1
        if max_archives > 0 and count >= max_archives:
            break


def _array(payload: dict[str, Any], key: str, default: float = 0.0) -> np.ndarray:
    value = payload.get(key)
    if value is None:
        return np.zeros(len(payload["candidates"]), dtype=np.float32) + default
    return np.asarray(value)


def _component(payload: dict[str, Any], key: str, default: float = 0.0) -> np.ndarray:
    return np.asarray(payload.get("components", {}).get(key, np.zeros(len(payload["candidates"])) + default), dtype=np.float32)


def _feas(payload: dict[str, Any], key: str, default: float = 0.0) -> np.ndarray:
    return np.asarray(payload.get("feasibility", {}).get(key, np.zeros(len(payload["candidates"])) + default), dtype=np.float32)


def _selected_indices(payload: dict[str, Any]) -> np.ndarray:
    tags = np.asarray([str(x or "") for x in payload.get("support_tags", [])], dtype=object)
    if tags.size == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.flatnonzero(tags != "")


def _source_bucket(source: str) -> str:
    source = str(source)
    if source in SOURCE_COLORS:
        return source
    if source.startswith("ddv2"):
        return "ddv2"
    if source.startswith("driveor"):
        return "driveor"
    if "progress" in source:
        return "progress"
    if "lateral" in source or "endpoint" in source:
        return "lateral"
    if "timing" in source or "delay" in source or "slow" in source:
        return "timing"
    return "fallback"


def _selected_valid_ratio(payload: dict[str, Any]) -> float:
    selected = _selected_indices(payload)
    if len(selected) == 0:
        return 0.0
    valid = np.asarray(payload.get("valid_mask", np.zeros(len(payload["candidates"]), dtype=bool)), dtype=bool)
    return float(valid[selected].mean())


def _best_selected_reward(payload: dict[str, Any]) -> float:
    selected = _selected_indices(payload)
    if len(selected) == 0:
        return float("-inf")
    rewards = np.asarray(payload.get("rewards", np.zeros(len(payload["candidates"]))), dtype=np.float32)
    return float(np.max(rewards[selected]))


def _score_payload(path: Path, payload: dict[str, Any]) -> list[SceneScore]:
    token = str(payload["token"])
    selected = _selected_indices(payload)
    rewards = np.asarray(payload.get("rewards", np.zeros(len(payload["candidates"]))), dtype=np.float32)
    valid = np.asarray(payload.get("valid_mask", np.zeros(len(payload["candidates"]), dtype=bool)), dtype=bool)
    tags = [str(x or "") for x in payload.get("support_tags", [])]
    sources = [str(x) for x in payload.get("sources", [])]
    gt_reward = float(payload.get("gt_reward", rewards[0] if len(rewards) else 0.0))
    il_reward = float(payload.get("il_reward", 0.0))
    ddc = _component(payload, "driving_direction_compliance")
    ep = _component(payload, "ego_progress")
    ttc = _component(payload, "time_to_collision_within_bound")
    feas = _feas(payload, "feas_cost")

    rows: list[SceneScore] = []
    if len(selected):
        selected_valid_idx = selected[valid[selected]] if len(valid) == len(rewards) else np.array([], dtype=np.int64)
        positive_idx = selected_valid_idx
        best_sel = float(np.max(rewards[selected]))
        best_valid_sel = float(np.max(rewards[selected_valid_idx])) if len(selected_valid_idx) else float("-inf")
        if len(selected_valid_idx):
            rows.append(SceneScore("best_gt_improver", token, path, best_valid_sel - gt_reward, payload))
        external_mask = np.array([_source_bucket(s) in {"ddv2", "driveor"} for s in sources], dtype=bool)
        external_sel = positive_idx[external_mask[positive_idx]] if len(sources) == len(rewards) else np.array([], dtype=np.int64)
        if len(external_sel):
            rows.append(SceneScore("external_support", token, path, float(np.max(rewards[external_sel]) - gt_reward), payload))
        fallback_count = sum(1 for i in selected if tags[i] == "fallback_best")
        rows.append(SceneScore("fallback_heavy", token, path, float(fallback_count), payload))
        selected_valid = float(valid[selected].mean()) if len(valid) == len(rewards) else 0.0
        rows.append(SceneScore("low_selected_valid", token, path, -selected_valid, payload))
        ddc_repair_idx = [int(i) for i in positive_idx if tags[i] == "ddc_repair"]
        if ddc_repair_idx:
            ddc_gain = float(np.max(ddc[ddc_repair_idx]) - ddc[0]) if len(ddc) == len(rewards) and len(rewards) else 0.0
            rows.append(SceneScore("ddc_repair", token, path, ddc_gain + 0.01 * float(np.max(rewards[selected])), payload))
        smooth_idx = [int(i) for i in positive_idx if tags[i] == "smooth_feasible"]
        if smooth_idx:
            rows.append(SceneScore("smooth_feasible", token, path, -float(np.min(feas[smooth_idx])), payload))
        if gt_reward >= 0.99 and best_valid_sel >= 0.99 and len(selected_valid_idx):
            spread = float(np.ptp(ep[selected_valid_idx]) + np.ptp(ttc[selected_valid_idx]) + np.ptp(ddc[selected_valid_idx]))
            rows.append(SceneScore("high_quality_diverse", token, path, spread, payload))
        if il_reward > 0 and len(selected_valid_idx):
            rows.append(SceneScore("beats_il", token, path, best_valid_sel - il_reward, payload))
    return rows


def _top_by_category(paths: Iterable[Path], categories: set[str], limit_per_category: int) -> dict[str, list[SceneScore]]:
    selected: dict[str, list[SceneScore]] = {category: [] for category in categories}
    for path in paths:
        try:
            payload = _load_payload(path)
        except Exception as exc:
            print(f"[WARN] failed to load {path}: {exc}", file=sys.stderr)
            continue
        for score in _score_payload(path, payload):
            if score.category not in selected:
                continue
            bucket = selected[score.category]
            bucket.append(score)
            bucket.sort(key=lambda item: item.score, reverse=True)
            del bucket[limit_per_category:]
    return selected


def _format_float(value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.3f}"


def _metric_row(payload: dict[str, Any], index: int) -> dict[str, Any]:
    tags = payload.get("support_tags", [])
    sources = payload.get("sources", [])
    rewards = np.asarray(payload.get("rewards", np.zeros(len(payload["candidates"]))), dtype=np.float32)
    valid = np.asarray(payload.get("valid_mask", np.zeros(len(payload["candidates"]), dtype=bool)), dtype=bool)
    return {
        "index": index,
        "tag": str(tags[index] if index < len(tags) else ""),
        "source": str(sources[index] if index < len(sources) else ""),
        "valid": bool(valid[index]) if index < len(valid) else False,
        "pdms": float(rewards[index]) if index < len(rewards) else 0.0,
        "ep": float(_component(payload, "ego_progress")[index]),
        "ttc": float(_component(payload, "time_to_collision_within_bound")[index]),
        "ddc": float(_component(payload, "driving_direction_compliance")[index]),
        "nc": float(_component(payload, "no_at_fault_collisions")[index]),
        "dac": float(_component(payload, "drivable_area_compliance")[index]),
        "feas": float(_feas(payload, "feas_cost")[index]),
    }


def plot_archive(payload: dict[str, Any], out_path: Path, title: str) -> list[dict[str, Any]]:
    candidates = np.asarray(payload["candidates"], dtype=np.float32)
    sources = [str(x) for x in payload.get("sources", [""] * len(candidates))]
    tags = [str(x or "") for x in payload.get("support_tags", [""] * len(candidates))]
    valid = np.asarray(payload.get("valid_mask", np.zeros(len(candidates), dtype=bool)), dtype=bool)
    selected = _selected_indices(payload)
    selected_set = set(int(i) for i in selected)

    fig = plt.figure(figsize=(13.5, 8.0), dpi=150)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])
    ax = fig.add_subplot(grid[0, 0])
    info_ax = fig.add_subplot(grid[0, 1])
    info_ax.axis("off")

    for idx, traj in enumerate(candidates):
        if idx in selected_set:
            continue
        style = "--" if idx < len(valid) and not valid[idx] else "-"
        ax.plot(traj[:, 1], traj[:, 0], color="#c8c8c8", alpha=0.18, linewidth=0.8, linestyle=style, zorder=1)

    table_rows = []
    for order, idx in enumerate(selected):
        traj = candidates[idx]
        tag = tags[idx] if idx < len(tags) else ""
        source = sources[idx] if idx < len(sources) else ""
        color = TAG_COLORS.get(tag, SOURCE_COLORS.get(_source_bucket(source), "#555555"))
        linestyle = "--" if idx < len(valid) and not valid[idx] else "-"
        linewidth = 3.0 if tag in {"gt_anchor", "best_pdms", "safe_ep_improver"} else 2.0
        ax.plot(
            traj[:, 1],
            traj[:, 0],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=0.95,
            label=f"{order}:{tag or source}",
            zorder=3,
        )
        ax.scatter(traj[-1, 1], traj[-1, 0], color=color, s=28, edgecolor="black", linewidth=0.4, zorder=4)
        ax.text(traj[-1, 1], traj[-1, 0], f"{order}", color=color, fontsize=8, weight="bold")
        row = _metric_row(payload, int(idx))
        row["plot_id"] = order
        table_rows.append(row)

    ax.scatter([0.0], [0.0], marker="^", color="black", s=70, label="ego", zorder=5)
    ax.axhline(0.0, color="#eeeeee", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="#eeeeee", linewidth=0.8, zorder=0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("ego y / lateral (m)")
    ax.set_ylabel("ego x / forward (m)")
    ax.set_title(title)

    metrics_text = [
        f"token: {payload.get('token')}",
        f"candidates: {len(candidates)}  support: {len(selected)}  selected_valid: {_selected_valid_ratio(payload):.3f}",
        f"gt={float(payload.get('gt_reward', 0.0)):.3f}  il={float(payload.get('il_reward', 0.0)):.3f}  best_sel={_best_selected_reward(payload):.3f}",
        "",
        "id tag/source                    v  pdms   ep   ttc   ddc  feas",
    ]
    for row in table_rows:
        label = row["tag"] or row["source"]
        source = row["source"]
        if row["tag"] and row["source"]:
            label = f"{row['tag']}|{source}"
        label = label[:27]
        metrics_text.append(
            f"{row['plot_id']:>2} {label:<27} {int(row['valid'])} "
            f"{_format_float(row['pdms']):>5} {_format_float(row['ep']):>5} "
            f"{_format_float(row['ttc']):>5} {_format_float(row['ddc']):>5} {_format_float(row['feas']):>5}"
        )
    info_ax.text(0.0, 1.0, "\n".join(metrics_text), va="top", ha="left", family="monospace", fontsize=8.4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return table_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize SG-FPS Pareto support trajectories from v3 support archives.")
    parser.add_argument("--archive-dir", required=True, help="Directory containing *.pkl.xz support archive records.")
    parser.add_argument("--output-dir", required=True, help="Directory for PNG visualizations and manifest files.")
    parser.add_argument("--audit-json", default="", help="Optional audit JSON; example tokens are loaded first.")
    parser.add_argument("--max-archives", type=int, default=20000, help="Max archives to scan after explicit audit tokens; <=0 scans all.")
    parser.add_argument("--limit-per-category", type=int, default=2)
    parser.add_argument(
        "--categories",
        default="best_gt_improver,external_support,ddc_repair,smooth_feasible,high_quality_diverse,fallback_heavy,low_selected_valid,beats_il",
    )
    parser.add_argument("--tokens", default="", help="Comma-separated extra scene tokens to include first.")
    args = parser.parse_args()

    archive_dir = Path(args.archive_dir)
    output_dir = Path(args.output_dir)
    categories = {item.strip() for item in args.categories.split(",") if item.strip()}
    tokens = [item.strip() for item in args.tokens.split(",") if item.strip()]
    if args.audit_json:
        audit = json.load(open(args.audit_json, "r", encoding="utf-8"))
        examples = audit.get("examples", {})
        for rows in examples.values():
            for row in rows:
                token = row.get("token")
                if token:
                    tokens.append(str(token))

    scores = _top_by_category(_iter_archive_paths(archive_dir, tokens, args.max_archives), categories, args.limit_per_category)
    manifest_rows: list[dict[str, Any]] = []
    used_tokens: set[str] = set()
    for category in sorted(categories):
        for rank, item in enumerate(scores.get(category, []), start=1):
            token = item.token
            filename = f"{category}/{rank:02d}_{token}.png"
            out_path = output_dir / filename
            title = f"{category} rank={rank} score={item.score:.4f}"
            table_rows = plot_archive(item.payload, out_path, title)
            used_tokens.add(token)
            manifest_rows.append(
                {
                    "category": category,
                    "rank": rank,
                    "token": token,
                    "score": item.score,
                    "png": filename,
                    "candidate_count": len(item.payload["candidates"]),
                    "support_count": len(_selected_indices(item.payload)),
                    "selected_valid_ratio": _selected_valid_ratio(item.payload),
                    "gt_reward": float(item.payload.get("gt_reward", 0.0)),
                    "il_reward": float(item.payload.get("il_reward", 0.0)),
                    "best_selected_reward": _best_selected_reward(item.payload),
                    "support_tags": ";".join(str(row["tag"]) for row in table_rows),
                    "support_sources": ";".join(str(row["source"]) for row in table_rows),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "category",
            "rank",
            "token",
            "score",
            "png",
            "candidate_count",
            "support_count",
            "selected_valid_ratio",
            "gt_reward",
            "il_reward",
            "best_selected_reward",
            "support_tags",
            "support_sources",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    with open(output_dir / "README.md", "w", encoding="utf-8") as f:
        f.write("# SG-FPS Pareto Support Visual Cases\n\n")
        f.write(f"- archive: `{archive_dir}`\n")
        f.write(f"- scanned max archives after audit tokens: `{args.max_archives}`\n")
        f.write(f"- unique visualized tokens: `{len(used_tokens)}`\n\n")
        f.write("Each PNG plots ego-local trajectories with x-forward on the vertical axis and y-lateral on the horizontal axis. ")
        f.write("Grey lines are non-selected evaluated candidates; colored lines are selected support trajectories. ")
        f.write("Dashed selected lines are invalid according to archive `valid_mask`.\n\n")
        f.write("See `manifest.csv` for token/category/source/tag details.\n")
    print(json.dumps({"output_dir": str(output_dir), "visualization_count": len(manifest_rows), "unique_tokens": len(used_tokens)}, indent=2))


if __name__ == "__main__":
    main()
