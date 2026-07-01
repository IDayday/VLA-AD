#!/usr/bin/env python3
"""Analyze ReCogDrive navtest scenes against hidden-cache navtrain scenes.

The script avoids torch.load on the full hidden-cache files. It reads only the
small storage blobs needed for scene descriptors from the PyTorch zip archive:
history, command, status, target trajectory, and two-expert dyn/geo slot tokens.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


STRUCT_FEATURE_NAMES: Tuple[str, ...] = (
    "cmd_0",
    "cmd_1",
    "cmd_2",
    "cmd_3",
    "status_0",
    "status_1",
    "status_2",
    "status_3",
    "status_4",
    "status_5",
    "status_6",
    "status_7",
    "hist_last_x",
    "hist_last_y",
    "hist_last_heading",
    "hist_dx_total",
    "hist_dy_total",
    "hist_path_len",
    "traj_final_x",
    "traj_final_y",
    "traj_abs_final_y",
    "traj_final_heading",
    "traj_path_len",
    "traj_displacement",
    "traj_progress_ratio",
    "traj_max_abs_y",
    "traj_mean_abs_y",
    "traj_turn_amount",
    "traj_curvature_proxy",
    "traj_reverse_frac",
)

COMPONENT_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "score": ("score", "pdms", "pdm_score"),
    "nc": ("no_at_fault_collisions", "nc"),
    "dac": ("drivable_area_compliance", "dac"),
    "ep": ("ego_progress", "ep", "progress"),
    "ttc": ("time_to_collision_within_bound", "ttc"),
    "comfort": ("comfort", "history_comfort"),
    "ddc": ("driving_direction_compliance", "ddc"),
    "tlc": ("traffic_light_compliance", "tlc"),
}


def _read_index(index_path: Path, root: Path, limit: int | None = None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(rows) >= limit:
                break
            item = json.loads(line)
            rows.append(
                {
                    "token": item["sample_token"],
                    "scene_token": item.get("scene_token", ""),
                    "log_name": item.get("log_name", ""),
                    "path": str(root / item["path"]),
                }
            )
    return rows


def _zip_storage_name(names: Sequence[str], storage_idx: int) -> str:
    suffix = f"/data/{storage_idx}"
    for name in names:
        if name.endswith(suffix):
            return name
    raise KeyError(f"Missing storage {storage_idx}; available={names[:8]}")


def _read_tensor_from_zip(zf: zipfile.ZipFile, names: Sequence[str], storage_idx: int, dtype: torch.dtype, shape: Tuple[int, ...]) -> np.ndarray:
    raw = bytearray(zf.read(_zip_storage_name(names, storage_idx)))
    tensor = torch.frombuffer(raw, dtype=dtype)
    return tensor.reshape(shape).float().numpy()


def _wrap_angle(x: np.ndarray | float) -> np.ndarray | float:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _path_length(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xy.astype(np.float64), axis=0), axis=1).sum())


def _trajectory_descriptor(traj: np.ndarray, prefix: str) -> Dict[str, float]:
    if traj.ndim != 2 or traj.shape[1] < 3:
        traj = np.zeros((1, 3), dtype=np.float32)
    xy = traj[:, :2].astype(np.float64)
    heading = np.unwrap(traj[:, 2].astype(np.float64))
    path_len = _path_length(xy)
    final = xy[-1]
    displacement = float(np.linalg.norm(final - xy[0]))
    heading_delta = float(_wrap_angle(heading[-1] - heading[0]))
    turn_amount = float(np.abs(np.diff(heading)).sum()) if len(heading) > 1 else 0.0
    reverse_frac = float((np.diff(xy[:, 0]) < -1e-3).mean()) if len(xy) > 1 else 0.0
    return {
        f"{prefix}_final_x": float(final[0]),
        f"{prefix}_final_y": float(final[1]),
        f"{prefix}_abs_final_y": float(abs(final[1])),
        f"{prefix}_final_heading": heading_delta,
        f"{prefix}_path_len": path_len,
        f"{prefix}_displacement": displacement,
        f"{prefix}_progress_ratio": float(final[0] / max(path_len, 1e-6)),
        f"{prefix}_max_abs_y": float(np.max(np.abs(xy[:, 1]))),
        f"{prefix}_mean_abs_y": float(np.mean(np.abs(xy[:, 1]))),
        f"{prefix}_turn_amount": turn_amount,
        f"{prefix}_curvature_proxy": float(turn_amount / max(path_len, 1e-6)),
        f"{prefix}_reverse_frac": reverse_frac,
    }


def _pool_slot(slot: np.ndarray, out_dim: int = 128) -> np.ndarray:
    # slot is [*, 1536]. Block-pool to a compact deterministic descriptor.
    flat = slot.reshape(-1, slot.shape[-1]).astype(np.float32)
    mean = flat.mean(axis=0)
    usable = (mean.shape[0] // out_dim) * out_dim
    pooled = mean[:usable].reshape(out_dim, usable // out_dim).mean(axis=1)
    norm = np.linalg.norm(pooled)
    if norm > 1e-6:
        pooled = pooled / norm
    return pooled.astype(np.float32)


def _extract_hidden_one(item: Dict[str, str]) -> Tuple[Dict[str, str], np.ndarray, np.ndarray, Dict[str, float]]:
    path = Path(item["path"])
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            history = _read_tensor_from_zip(zf, names, 0, torch.float32, (4, 3))
            command = _read_tensor_from_zip(zf, names, 1, torch.float32, (4,))
            status = _read_tensor_from_zip(zf, names, 2, torch.float32, (8,))
            traj = _read_tensor_from_zip(zf, names, 3, torch.float64, (8, 3))
            dyn = _read_tensor_from_zip(zf, names, 6, torch.bfloat16, (3, 12, 1536))
            geo = _read_tensor_from_zip(zf, names, 7, torch.bfloat16, (12, 1536))
        visual = np.concatenate([_pool_slot(dyn), _pool_slot(geo)], axis=0).astype(np.float32)
        traj_desc = _trajectory_descriptor(traj, "traj")
        hist_xy = history[:, :2].astype(np.float64)
        hist_path = _path_length(hist_xy)
        struct_values: Dict[str, float] = {
            **{f"cmd_{i}": float(command[i]) for i in range(4)},
            **{f"status_{i}": float(status[i]) for i in range(8)},
            "hist_last_x": float(history[-1, 0]),
            "hist_last_y": float(history[-1, 1]),
            "hist_last_heading": float(history[-1, 2]),
            "hist_dx_total": float(history[-1, 0] - history[0, 0]),
            "hist_dy_total": float(history[-1, 1] - history[0, 1]),
            "hist_path_len": hist_path,
            **traj_desc,
        }
        struct = np.asarray([struct_values[name] for name in STRUCT_FEATURE_NAMES], dtype=np.float32)
        return item, visual, struct, {"ok": 1.0, "error": ""}
    except Exception as exc:  # noqa: BLE001
        visual = np.zeros((256,), dtype=np.float32)
        struct = np.zeros((len(STRUCT_FEATURE_NAMES),), dtype=np.float32)
        return item, visual, struct, {"ok": 0.0, "error": repr(exc)}


def _save_feature_cache(
    out_path: Path,
    rows: Sequence[Dict[str, str]],
    visual: np.ndarray,
    struct: np.ndarray,
    ok: np.ndarray,
    errors: Sequence[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        token=np.asarray([r["token"] for r in rows], dtype="U32"),
        scene_token=np.asarray([r["scene_token"] for r in rows], dtype="U32"),
        log_name=np.asarray([r["log_name"] for r in rows], dtype="U96"),
        path=np.asarray([r["path"] for r in rows], dtype="U256"),
        visual=visual.astype(np.float32),
        struct=struct.astype(np.float32),
        ok=ok.astype(np.bool_),
        error=np.asarray(list(errors), dtype="U256"),
        struct_feature_names=np.asarray(STRUCT_FEATURE_NAMES, dtype="U64"),
    )


def build_or_load_feature_cache(
    *,
    hidden_root: Path,
    output_npz: Path,
    limit: int | None,
    workers: int,
    rebuild: bool,
) -> Dict[str, np.ndarray]:
    if output_npz.is_file() and not rebuild:
        return dict(np.load(output_npz, allow_pickle=False))
    rows = _read_index(hidden_root / "index.jsonl", hidden_root, limit)
    visual = np.zeros((len(rows), 256), dtype=np.float32)
    struct = np.zeros((len(rows), len(STRUCT_FEATURE_NAMES)), dtype=np.float32)
    ok = np.zeros((len(rows),), dtype=np.bool_)
    errors: List[str] = [""] * len(rows)
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for idx, (item, v, s, meta) in enumerate(pool.map(_extract_hidden_one, rows, chunksize=64)):
            rows[idx] = item
            visual[idx] = v
            struct[idx] = s
            ok[idx] = bool(meta["ok"])
            errors[idx] = str(meta["error"])
            if (idx + 1) % 5000 == 0 or idx + 1 == len(rows):
                print(f"extracted hidden features {idx + 1}/{len(rows)} -> {output_npz}", flush=True)
    _save_feature_cache(output_npz, rows, visual, struct, ok, errors)
    return dict(np.load(output_npz, allow_pickle=False))


def _component_column(df: pd.DataFrame, canonical: str) -> pd.Series:
    for col in COMPONENT_ALIASES[canonical]:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=np.float64)


def read_eval(path: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    if "token" not in df.columns:
        raise ValueError(f"Missing token column in {path}")
    if "valid" in df.columns:
        df = df[df["valid"].astype(str).str.lower().isin({"true", "1"})].copy()
    out = pd.DataFrame({"token": df["token"].astype(str)})
    for key in ("score", "nc", "dac", "ep", "ttc", "comfort", "ddc", "tlc"):
        out[f"{prefix}_{key}"] = _component_column(df, key)
    out[f"{prefix}_core"] = (5.0 * out[f"{prefix}_ep"] + 5.0 * out[f"{prefix}_ttc"] + 2.0 * out[f"{prefix}_comfort"]) / 12.0
    return out.drop_duplicates("token")


def _standardize(train: np.ndarray, other: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    return (
        np.nan_to_num((train - mean) / std, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num((other - mean) / std, nan=0.0, posinf=0.0, neginf=0.0),
    )


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-6)


def _combined_embedding(
    train_cache: Mapping[str, np.ndarray],
    nav_cache: Mapping[str, np.ndarray],
    *,
    visual_weight: float,
    struct_weight: float,
) -> Tuple[np.ndarray, np.ndarray]:
    train_visual = _l2_normalize(train_cache["visual"].astype(np.float32))
    nav_visual = _l2_normalize(nav_cache["visual"].astype(np.float32))
    train_struct, nav_struct = _standardize(train_cache["struct"].astype(np.float32), nav_cache["struct"].astype(np.float32))
    train_emb = np.concatenate([visual_weight * train_visual, struct_weight * train_struct], axis=1).astype(np.float32)
    nav_emb = np.concatenate([visual_weight * nav_visual, struct_weight * nav_struct], axis=1).astype(np.float32)
    return _l2_normalize(train_emb).astype(np.float32), _l2_normalize(nav_emb).astype(np.float32)


def _topk_cosine_cpu(train_emb: np.ndarray, query_emb: np.ndarray, k: int, block: int) -> Tuple[np.ndarray, np.ndarray]:
    k = min(k, train_emb.shape[0])
    indices = np.zeros((query_emb.shape[0], k), dtype=np.int64)
    scores = np.zeros((query_emb.shape[0], k), dtype=np.float32)
    train_t = np.ascontiguousarray(train_emb.T)
    for start in range(0, query_emb.shape[0], block):
        end = min(start + block, query_emb.shape[0])
        sim = np.asarray(query_emb[start:end] @ train_t, dtype=np.float32)
        part = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        indices[start:end] = np.take_along_axis(part, order, axis=1)
        scores[start:end] = np.take_along_axis(part_scores, order, axis=1)
        print(f"queried nearest train scenes {end}/{query_emb.shape[0]}", flush=True)
    return indices, scores


def _topk_cosine_gpu_worker(
    *,
    device: str,
    train_emb: np.ndarray,
    query_emb: np.ndarray,
    row_offset: int,
    k: int,
    block: int,
) -> Tuple[int, np.ndarray, np.ndarray]:
    torch_device = torch.device(device)
    train_t = torch.from_numpy(np.ascontiguousarray(train_emb.T)).to(torch_device)
    local_idx = np.zeros((query_emb.shape[0], k), dtype=np.int64)
    local_scores = np.zeros((query_emb.shape[0], k), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, query_emb.shape[0], block):
            end = min(start + block, query_emb.shape[0])
            q = torch.from_numpy(np.ascontiguousarray(query_emb[start:end])).to(torch_device)
            sim = q @ train_t
            score_t, idx_t = torch.topk(sim, k=k, dim=1, largest=True, sorted=True)
            local_idx[start:end] = idx_t.cpu().numpy()
            local_scores[start:end] = score_t.float().cpu().numpy()
    return row_offset, local_idx, local_scores


def _topk_cosine_gpu(
    train_emb: np.ndarray,
    query_emb: np.ndarray,
    k: int,
    block: int,
    devices: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    k = min(k, train_emb.shape[0])
    clean_devices = [device.strip() for device in devices if device.strip()]
    if not clean_devices:
        return _topk_cosine_cpu(train_emb, query_emb, k, block)
    if not torch.cuda.is_available():
        raise RuntimeError("--gpu-devices was set but CUDA is not available")
    indices = np.zeros((query_emb.shape[0], k), dtype=np.int64)
    scores = np.zeros((query_emb.shape[0], k), dtype=np.float32)
    splits = np.array_split(np.arange(query_emb.shape[0]), len(clean_devices))
    with ThreadPoolExecutor(max_workers=len(clean_devices)) as pool:
        futures = []
        for device, rows in zip(clean_devices, splits):
            if len(rows) == 0:
                continue
            start, end = int(rows[0]), int(rows[-1]) + 1
            futures.append(
                pool.submit(
                    _topk_cosine_gpu_worker,
                    device=device,
                    train_emb=train_emb,
                    query_emb=query_emb[start:end],
                    row_offset=start,
                    k=k,
                    block=block,
                )
            )
        done_rows = 0
        for future in as_completed(futures):
            offset, local_idx, local_scores = future.result()
            end = offset + local_idx.shape[0]
            indices[offset:end] = local_idx
            scores[offset:end] = local_scores
            done_rows += local_idx.shape[0]
            print(f"queried nearest train scenes {done_rows}/{query_emb.shape[0]} with CUDA", flush=True)
    return indices, scores


def _topk_cosine(
    train_emb: np.ndarray,
    query_emb: np.ndarray,
    k: int,
    block: int,
    gpu_devices: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    if gpu_devices:
        return _topk_cosine_gpu(train_emb, query_emb, k, block, gpu_devices.split(","))
    return _topk_cosine_cpu(train_emb, query_emb, k, block)


def _score_bucket(score: float) -> str:
    if not math.isfinite(score):
        return "missing"
    if score == 0.0:
        return "zero"
    if score <= 0.5:
        return "low_0_0p5"
    if score <= 0.85:
        return "mid_0p5_0p85"
    if score <= 0.95:
        return "good_0p85_0p95"
    return "high_0p95_1"


def _failure_reason(row: pd.Series, prefix: str = "sota") -> str:
    score = float(row.get(f"{prefix}_score", np.nan))
    nc = float(row.get(f"{prefix}_nc", np.nan))
    dac = float(row.get(f"{prefix}_dac", np.nan))
    ep = float(row.get(f"{prefix}_ep", np.nan))
    ttc = float(row.get(f"{prefix}_ttc", np.nan))
    ddc = float(row.get(f"{prefix}_ddc", np.nan))
    if score == 0.0:
        if nc < 1.0 and dac < 1.0:
            return "zero_nc_and_dac"
        if nc < 1.0:
            return "zero_nc_collision"
        if dac < 1.0:
            return "zero_dac_offroad"
        return "zero_other"
    tags: List[str] = []
    if ep < 0.75:
        tags.append("low_ep")
    if ttc < 0.9:
        tags.append("low_ttc")
    if ddc < 0.95:
        tags.append("low_ddc")
    return "+".join(tags) if tags else "nonzero_ok"


def _motion_phenotype(struct_row: np.ndarray) -> str:
    names = {name: i for i, name in enumerate(STRUCT_FEATURE_NAMES)}
    final_x = float(struct_row[names["traj_final_x"]])
    final_y = float(struct_row[names["traj_final_y"]])
    heading = abs(float(struct_row[names["traj_final_heading"]]))
    turn = float(struct_row[names["traj_turn_amount"]])
    path_len = float(struct_row[names["traj_path_len"]])
    if path_len < 3.0 or final_x < 3.0:
        return "stop_or_creep"
    if heading > 0.45 or turn > 0.70:
        return "turning"
    if abs(final_y) > 1.2:
        return "lateral_shift"
    if final_x > 20.0:
        return "fast_straight"
    return "straight"


def _delta_bucket(delta: float) -> str:
    if not math.isfinite(delta):
        return "no_baseline"
    if delta >= 0.10:
        return "large_gain_ge_0p10"
    if delta >= 0.02:
        return "gain_0p02_0p10"
    if delta <= -0.10:
        return "large_drop_le_minus_0p10"
    if delta <= -0.02:
        return "drop_minus_0p10_0p02"
    return "flat_abs_lt_0p02"


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def _fmt(value: Any, nd: int = 4) -> str:
    try:
        f = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(f):
        return "nan"
    return f"{f:.{nd}f}"


def _nearest_columns(
    nav: pd.DataFrame,
    train_cache: Mapping[str, np.ndarray],
    nn_idx: np.ndarray,
    nn_score: np.ndarray,
    k: int,
) -> pd.DataFrame:
    train_token = train_cache["token"]
    train_log = train_cache["log_name"]
    train_scene = train_cache["scene_token"]
    out = nav.copy()
    for rank in range(k):
        out[f"nn{rank + 1}_train_token"] = train_token[nn_idx[:, rank]]
        out[f"nn{rank + 1}_train_scene_token"] = train_scene[nn_idx[:, rank]]
        out[f"nn{rank + 1}_train_log_name"] = train_log[nn_idx[:, rank]]
        out[f"nn{rank + 1}_similarity"] = nn_score[:, rank]
    out["nn1_similarity"] = nn_score[:, 0]
    out["nn5_mean_similarity"] = nn_score[:, : min(5, k)].mean(axis=1)
    out["nn10_mean_similarity"] = nn_score[:, : min(10, k)].mean(axis=1)
    return out


def _category_train_mapping(nav: pd.DataFrame, k: int, topn: int = 20) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["score_bucket", "failure_reason", "motion_phenotype", "delta_bucket"]
    for keys, group in nav.groupby(group_cols, dropna=False):
        counter: Counter[str] = Counter()
        log_counter: Counter[str] = Counter()
        for _, row in group.iterrows():
            for rank in range(1, min(k, 10) + 1):
                tok = str(row.get(f"nn{rank}_train_token", ""))
                log = str(row.get(f"nn{rank}_train_log_name", ""))
                if tok:
                    counter[tok] += 1
                if log:
                    log_counter[log] += 1
        top_train = [f"{tok}:{count}" for tok, count in counter.most_common(topn)]
        top_logs = [f"{log}:{count}" for log, count in log_counter.most_common(10)]
        rows.append(
            {
                "score_bucket": keys[0],
                "failure_reason": keys[1],
                "motion_phenotype": keys[2],
                "delta_bucket": keys[3],
                "navtest_count": int(len(group)),
                "sota_score_mean": float(group["sota_score"].mean()),
                "sota_core_mean": float(group["sota_core"].mean()),
                "delta_score_mean": float(group["delta_score"].mean()) if "delta_score" in group else float("nan"),
                "nn1_similarity_mean": float(group["nn1_similarity"].mean()),
                "nn10_similarity_mean": float(group["nn10_mean_similarity"].mean()),
                "top_train_tokens": ";".join(top_train),
                "top_train_logs": ";".join(top_logs),
            }
        )
    return pd.DataFrame(rows).sort_values(["score_bucket", "failure_reason", "navtest_count"], ascending=[True, True, False])


def _write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    nav: pd.DataFrame,
    category_map: pd.DataFrame,
    baseline_name: str | None,
    train_rows: int,
    nav_rows: int,
) -> None:
    score_rows = []
    for bucket, group in nav.groupby("score_bucket"):
        score_rows.append(
            [
                bucket,
                len(group),
                _fmt(group["sota_score"].mean()),
                _fmt(group["sota_core"].mean()),
                _fmt(group["sota_nc"].mean()),
                _fmt(group["sota_dac"].mean()),
                _fmt(group["sota_ep"].mean()),
                _fmt(group["sota_ttc"].mean()),
                _fmt(group["sota_ddc"].mean()),
                _fmt(group["nn1_similarity"].mean()),
                _fmt(group["nn10_mean_similarity"].mean()),
            ]
        )
    failure_rows = []
    for reason, group in nav.groupby("failure_reason"):
        failure_rows.append([reason, len(group), _fmt(group["sota_score"].mean()), _fmt(group["nn10_mean_similarity"].mean())])
    failure_rows.sort(key=lambda r: (-int(r[1]), str(r[0])))
    delta_rows: List[List[Any]] = []
    if baseline_name:
        for bucket, group in nav.groupby("delta_bucket"):
            delta_rows.append(
                [
                    bucket,
                    len(group),
                    _fmt(group["delta_score"].mean()),
                    _fmt(group["delta_core"].mean()),
                    _fmt(group["delta_ep"].mean()),
                    _fmt(group["delta_ttc"].mean()),
                    _fmt(group["delta_nc"].mean()),
                    _fmt(group["delta_dac"].mean()),
                ]
            )
    zero = nav[nav["sota_score"] == 0.0]
    lines = [
        "# ReCogDrive Stage3 Scene Similarity And Navtest Error Analysis",
        "",
        "## Data Sources",
        f"- SOTA eval CSV: `{args.sota_eval_csv}`",
        f"- Baseline eval CSV: `{args.baseline_eval_csv}`" if baseline_name else "- Baseline eval CSV: not provided",
        f"- Train hidden cache: `{args.train_hidden_root}`",
        f"- Navtest hidden cache: `{args.navtest_hidden_root}`",
        f"- Train feature rows used: {train_rows}",
        f"- Navtest rows joined with SOTA eval: {nav_rows}",
        "",
        "## Similarity Definition",
        "This report does not rely on trajectory shape alone. The current nearest-neighbor descriptor combines:",
        "- two-expert dynamic/geometric slot embeddings block-pooled from hidden cache, as a lightweight visual/semantic proxy;",
        "- future trajectory shape from the cached expert target;",
        "- ego history, high-level command, and status features;",
        "- evaluation submetrics only for failure classification, not for nearest-neighbor distance.",
        "",
        "Important limitation: raw map topology, actor counts, traffic-light state, and drivable-area geometry are not yet in this hidden-cache descriptor. They should be added from metric cache in a second pass, but the raw metric cache loader is much slower.",
        "",
        "## SOTA Navtest Summary",
        _md_table(
            ["bucket", "count", "PDMS", "Core", "NC", "DAC", "EP", "TTC", "DDC", "NN1 sim", "NN10 sim"],
            score_rows,
        ),
        "",
        f"Zero-score scenes: {len(zero)} / {len(nav)} ({len(zero) / max(len(nav), 1):.2%}).",
        "",
        "## Failure Reasons",
        _md_table(["failure_reason", "count", "PDMS", "NN10 sim"], failure_rows[:20]),
    ]
    if delta_rows:
        lines.extend(
            [
                "",
                f"## SOTA vs {baseline_name}",
                _md_table(["delta_bucket", "count", "delta_PDMS", "delta_Core", "delta_EP", "delta_TTC", "delta_NC", "delta_DAC"], delta_rows),
                "",
                "Interpretation: this is a per-token comparison against the available local baseline CSV. The documented original 0.9054/0.9055 per-token CSV was not found locally during this pass, so do not treat the 91.04 development CSV as the original 90.55 baseline.",
            ]
        )
    top_categories = category_map.sort_values("navtest_count", ascending=False).head(20)
    category_rows = [
        [
            r.score_bucket,
            r.failure_reason,
            r.motion_phenotype,
            r.delta_bucket,
            int(r.navtest_count),
            _fmt(r.sota_score_mean),
            _fmt(r.nn10_similarity_mean),
            str(r.top_train_tokens)[:180],
        ]
        for r in top_categories.itertuples(index=False)
    ]
    lines.extend(
        [
            "",
            "## Category To Train Mapping",
            _md_table(
                ["score", "failure", "motion", "delta", "navtest_n", "PDMS", "NN10 sim", "top train token:freq"],
                category_rows,
            ),
            "",
            "Full files:",
            f"- navtest scene mapping: `{args.output_dir}/navtest_scene_category_train_mapping.csv`",
            f"- category-to-train mapping: `{args.output_dir}/category_to_train_scene_mapping.csv`",
            f"- feature caches: `{args.output_dir}/hidden_features_navtrain.npz`, `{args.output_dir}/hidden_features_navtest.npz`",
            "",
            "## Practical Use",
            "- Use `failure_reason` to pick what the policy failed on: collision, off-road, low EP, low TTC, or low DDC.",
            "- Use `motion_phenotype` to avoid mixing stop/turn/lateral-shift/straight scenes in one training bucket.",
            "- Use nearest train tokens/logs as scene-level candidates for focused replay, oracle search, or GRPO diagnostics.",
            "- If a navtest failure bucket has low NN similarity, it is more likely a coverage/novelty issue; if similarity is high, it is more likely a learning/objective issue.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_cache = build_or_load_feature_cache(
        hidden_root=Path(args.train_hidden_root),
        output_npz=out_dir / "hidden_features_navtrain.npz",
        limit=args.max_train,
        workers=args.workers,
        rebuild=args.rebuild_features,
    )
    nav_cache = build_or_load_feature_cache(
        hidden_root=Path(args.navtest_hidden_root),
        output_npz=out_dir / "hidden_features_navtest.npz",
        limit=args.max_navtest,
        workers=args.workers,
        rebuild=args.rebuild_features,
    )
    train_ok = train_cache["ok"].astype(bool)
    nav_ok = nav_cache["ok"].astype(bool)
    train_cache = {key: value[train_ok] if getattr(value, "shape", (0,))[0] == len(train_ok) else value for key, value in train_cache.items()}
    nav_cache = {key: value[nav_ok] if getattr(value, "shape", (0,))[0] == len(nav_ok) else value for key, value in nav_cache.items()}
    train_emb, nav_emb = _combined_embedding(
        train_cache,
        nav_cache,
        visual_weight=float(args.visual_weight),
        struct_weight=float(args.struct_weight),
    )
    nn_idx, nn_score = _topk_cosine(
        train_emb,
        nav_emb,
        int(args.knn_k),
        int(args.query_block),
        gpu_devices=str(args.gpu_devices),
    )

    nav_df = pd.DataFrame(
        {
            "token": nav_cache["token"].astype(str),
            "scene_token": nav_cache["scene_token"].astype(str),
            "log_name": nav_cache["log_name"].astype(str),
        }
    )
    nav_df["motion_phenotype"] = [_motion_phenotype(row) for row in nav_cache["struct"]]
    cmd_idx = np.argmax(nav_cache["struct"][:, :4], axis=1)
    nav_df["command_bucket"] = [f"cmd_{int(i)}" for i in cmd_idx]

    sota = read_eval(Path(args.sota_eval_csv), "sota")
    nav = nav_df.merge(sota, on="token", how="inner")
    if args.baseline_eval_csv:
        baseline_name = args.baseline_name or Path(args.baseline_eval_csv).stem
        base = read_eval(Path(args.baseline_eval_csv), "base")
        nav = nav.merge(base, on="token", how="left")
        for key in ("score", "core", "nc", "dac", "ep", "ttc", "comfort", "ddc"):
            nav[f"delta_{key}"] = nav[f"sota_{key}"] - nav[f"base_{key}"]
        nav["delta_bucket"] = nav["delta_score"].map(_delta_bucket)
    else:
        baseline_name = None
        nav["delta_bucket"] = "no_baseline"
        for key in ("score", "core", "nc", "dac", "ep", "ttc", "comfort", "ddc"):
            nav[f"delta_{key}"] = np.nan

    # Align nearest-neighbor arrays to the joined nav rows.
    nav_pos = {token: i for i, token in enumerate(nav_df["token"].astype(str))}
    keep = np.asarray([nav_pos[token] for token in nav["token"].astype(str)], dtype=np.int64)
    nav = _nearest_columns(nav.reset_index(drop=True), train_cache, nn_idx[keep], nn_score[keep], int(args.knn_k))
    nav["score_bucket"] = nav["sota_score"].map(_score_bucket)
    nav["failure_reason"] = nav.apply(_failure_reason, axis=1)

    nav_out = out_dir / "navtest_scene_category_train_mapping.csv"
    nav.to_csv(nav_out, index=False)

    category = _category_train_mapping(nav, int(args.knn_k), topn=int(args.category_topn))
    category_out = out_dir / "category_to_train_scene_mapping.csv"
    category.to_csv(category_out, index=False)

    zero = nav[nav["sota_score"] == 0.0].copy()
    zero.to_csv(out_dir / "sota_zero_score_scenes_with_train_neighbors.csv", index=False)

    summary = {
        "sota_rows": int(len(nav)),
        "train_rows_used": int(len(train_cache["token"])),
        "sota_score_mean": float(nav["sota_score"].mean()),
        "sota_core_mean": float(nav["sota_core"].mean()),
        "zero_score_count": int((nav["sota_score"] == 0.0).sum()),
        "zero_score_ratio": float((nav["sota_score"] == 0.0).mean()),
        "nn1_similarity_mean": float(nav["nn1_similarity"].mean()),
        "nn10_similarity_mean": float(nav["nn10_mean_similarity"].mean()),
    }
    if baseline_name:
        summary.update(
            {
                "baseline_name": baseline_name,
                "baseline_rows_joined": int(nav["base_score"].notna().sum()),
                "delta_score_mean": float(nav["delta_score"].mean()),
                "delta_score_gain_ge_0p02_ratio": float((nav["delta_score"] >= 0.02).mean()),
                "delta_score_drop_le_minus_0p02_ratio": float((nav["delta_score"] <= -0.02).mean()),
            }
        )
    (out_dir / "scene_similarity_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    _write_report(
        out_dir / "recogdrive_stage3_scene_similarity_report.md",
        args=args,
        nav=nav,
        category_map=category,
        baseline_name=baseline_name,
        train_rows=len(train_cache["token"]),
        nav_rows=len(nav),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-hidden-root", required=True)
    parser.add_argument("--navtest-hidden-root", required=True)
    parser.add_argument("--sota-eval-csv", required=True)
    parser.add_argument("--baseline-eval-csv", default="")
    parser.add_argument("--baseline-name", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-navtest", type=int, default=None)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--query-block", type=int, default=128)
    parser.add_argument("--gpu-devices", default="", help="Comma-separated CUDA devices for exact cosine top-k, e.g. cuda:0,cuda:1")
    parser.add_argument("--visual-weight", type=float, default=1.0)
    parser.add_argument("--struct-weight", type=float, default=0.35)
    parser.add_argument("--category-topn", type=int, default=20)
    return parser


def main() -> None:
    analyze(build_argparser().parse_args())


if __name__ == "__main__":
    main()
