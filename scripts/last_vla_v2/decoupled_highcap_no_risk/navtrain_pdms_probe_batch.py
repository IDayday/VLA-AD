#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import sys
import threading
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import load_sample  # noqa: E402
from navsim.common.dataclasses import PDMResults, Trajectory  # noqa: E402
from navsim.common.dataloader import MetricCacheLoader  # noqa: E402
from navsim.evaluate.pdm_score import get_trajectory_as_array, pdm_score, transform_trajectory  # noqa: E402
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex  # noqa: E402
from scripts.eval_recogdrive_expert_pdm import (  # noqa: E402
    build_metric_cache_loader,
    build_pdm_tools,
    build_planner,
    chunk_dirs,
    dtype_from_precision,
    load_sample_token_filter,
    load_vlm_text_anchor_index,
    load_yaml,
    make_batch,
    mean_or_none,
    sample_paths,
    write_pdm_csv,
)


PROJECT_ROOT = Path("/mnt/project/VLA-AD_last_vla_dev")
DEFAULT_CHUNK_ROOT = Path("/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/train_full_highcap_chunks")
DEFAULT_METRIC_CACHE = Path("/mnt/project/VLA-AD/cache/metric_cache_train_full")
DEFAULT_ANCHOR_CACHE = Path(
    "/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/vlm_text_anchor_2bbase_train_20260609T052233Z"
)
METRIC_KEYS = ("PDMS", "NC", "DAC", "TTC", "comfort", "EP", "DDC")
_PDM_THREAD_LOCAL = threading.local()


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def enabled(row: Dict[str, str]) -> bool:
    return row.get("enabled", "1") not in {"0", "false", "False"}


def safe_name(text: str) -> str:
    keep = []
    for char in text:
        if char.isalnum() or char in ("-", "_", "."):
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("._") or "item"


def parse_probe_filter(raw: str) -> Optional[set[str]]:
    items = {item.strip() for item in raw.split(",") if item.strip()}
    return items or None


def probe_rows(args: argparse.Namespace) -> List[Dict[str, str]]:
    requested = parse_probe_filter(args.probes)
    rows = []
    for row in read_csv(args.probe_manifest):
        if not enabled(row):
            continue
        if args.host_key and row.get("host_key") != args.host_key:
            continue
        if requested is not None and row.get("name") not in requested:
            continue
        rows.append(row)
    if not rows:
        raise RuntimeError("No probe rows matched the requested filter.")
    return rows


def shard_done(out_root: Path, probe: Dict[str, str], shard_index: int) -> bool:
    return (out_root / "eval" / safe_name(probe["name"]) / f"shard_{shard_index:02d}" / "metrics.json").is_file()


def metric_mean(items: Iterable[Dict[str, Any]], key: str, weight_key: str) -> Optional[float]:
    total = 0.0
    weight = 0
    for item in items:
        value = item.get(key)
        w = int(item.get(weight_key) or 0)
        if value is not None and w > 0:
            total += float(value) * w
            weight += w
    return total / weight if weight else None


def load_eval_states(args: argparse.Namespace, rows: List[Dict[str, str]], device: torch.device, dtype: torch.dtype) -> List[Dict[str, Any]]:
    states = []
    for row in rows:
        cfg = load_yaml(args.project_root / row["config"])
        planner = build_planner(cfg).to(device)
        if dtype != torch.float32:
            planner = planner.to(dtype=dtype)
        from scripts.eval_recogdrive_expert_pdm import load_checkpoint

        load_checkpoint(planner, Path(row["checkpoint"]))
        planner.eval()
        states.append(
            {
                "probe": row,
                "planner": planner,
                "predictions": [],
                "pdm_rows": [],
                "l1_values": [],
                "missing_metric_cache": 0,
                "failed_pdm": 0,
                "pdm_metric_values": {
                    "score": [],
                    "no_at_fault_collisions": [],
                    "drivable_area_compliance": [],
                    "ego_progress": [],
                    "time_to_collision_within_bound": [],
                    "comfort": [],
                    "driving_direction_compliance": [],
                },
            }
        )
    return states


def build_fast_metric_cache_loader(cache_path: Path):
    try:
        loader = MetricCacheLoader(cache_path)
        if not getattr(loader, "metric_cache_paths", None):
            raise RuntimeError(f"No metric cache paths loaded from metadata under {cache_path}")
        return loader
    except Exception as exc:
        warnings.warn(
            f"Could not use metric-cache metadata from {cache_path}: {exc!r}; falling back to recursive scan.",
            RuntimeWarning,
        )
        return build_metric_cache_loader(cache_path)


def batch_cache_key(planner: Any) -> Tuple[Any, ...]:
    cfg = planner.config
    keys = (
        "use_expert_features",
        "use_last_rd",
        "use_last_vla",
        "use_jepa",
        "use_vggt",
        "num_jepa_tokens",
        "num_vggt_tokens",
        "num_geometry_tokens",
        "last_vla_geometry_teacher_dim",
        "last_vla_require_full_geometry",
        "last_vla_allow_patch_geometry_fallback",
        "last_vla_use_residual_diffusion",
        "last_vla_residual_anchor_source",
        "last_vla_require_residual_anchor",
    )
    return tuple(getattr(cfg, key, None) for key in keys)


def pairwise_pdm_result_from_scorer(scorer: Any, pred_idx: int) -> PDMResults:
    multiplicative = scorer._multi_metrics.prod(axis=0)
    pair = np.asarray([0, pred_idx], dtype=np.int64)
    raw_progress = scorer._progress_raw[pair] * multiplicative[pair]
    max_raw_progress = float(np.max(raw_progress))
    if max_raw_progress > scorer._config.progress_distance_threshold:
        ego_progress = float(raw_progress[1] / max_raw_progress)
    else:
        ego_progress = 1.0 if float(multiplicative[pred_idx]) != 0.0 else 0.0

    weighted_metrics = scorer._weighted_metrics[:, pred_idx].astype(np.float64, copy=True)
    weighted_metrics[WeightedMetricIndex.PROGRESS] = ego_progress
    weighted_metrics_array = scorer._config.weighted_metrics_array
    weighted_score = float((weighted_metrics * weighted_metrics_array).sum() / weighted_metrics_array.sum())
    score = float(multiplicative[pred_idx] * weighted_score)

    return PDMResults(
        float(scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, pred_idx]),
        float(scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, pred_idx]),
        ego_progress,
        float(scorer._weighted_metrics[WeightedMetricIndex.TTC, pred_idx]),
        float(scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, pred_idx]),
        float(scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION, pred_idx]),
        score,
    )


def pdm_score_many_pairwise(
    metric_cache: Any,
    model_trajectories: Sequence[Trajectory],
    future_sampling: Any,
    simulator: Any,
    scorer: Any,
) -> List[PDMResults]:
    if not model_trajectories:
        return []
    initial_ego_state = metric_cache.ego_state
    pdm_states = get_trajectory_as_array(metric_cache.trajectory, future_sampling, initial_ego_state.time_point)
    pred_states = [
        get_trajectory_as_array(transform_trajectory(trajectory, initial_ego_state), future_sampling, initial_ego_state.time_point)
        for trajectory in model_trajectories
    ]
    trajectory_states = np.concatenate([pdm_states[None, ...], np.stack(pred_states, axis=0)], axis=0)
    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)
    scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
    )
    return [pairwise_pdm_result_from_scorer(scorer, pred_idx) for pred_idx in range(1, len(model_trajectories) + 1)]


def apply_pdm_result(state: Dict[str, Any], record: Dict[str, Any], pdm_row: Dict[str, Any], result: PDMResults) -> None:
    result_dict = asdict(result)
    record["pdm_valid"] = True
    record["pdm"] = result_dict
    pdm_row.update({"valid": True, **result_dict})
    for key, value in result_dict.items():
        state["pdm_metric_values"][key].append(float(value))


def get_thread_pdm_tools():
    tools = getattr(_PDM_THREAD_LOCAL, "tools", None)
    if tools is None:
        tools = build_pdm_tools()
        _PDM_THREAD_LOCAL.tools = tools
    return tools


def score_sample_items(
    sample_items: Sequence[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], torch.Tensor]],
    metric_cache: Any,
    metric_error: Optional[str],
    pdm_tools: Optional[Tuple[Any, Any, Any]],
) -> List[Dict[str, Any]]:
    updates: List[Dict[str, Any]] = []
    if metric_cache is None:
        for state, record, pdm_row, _pred in sample_items:
            updates.append(
                {
                    "state": state,
                    "record": record,
                    "pdm_row": pdm_row,
                    "missing": True,
                    "error": metric_error or "missing_metric_cache",
                }
            )
        return updates

    future_sampling, simulator, scorer = pdm_tools if pdm_tools is not None else get_thread_pdm_tools()
    model_trajectories = [Trajectory(poses=pred.numpy()) for _state, _record, _pdm_row, pred in sample_items]
    try:
        results = pdm_score_many_pairwise(
            metric_cache=metric_cache,
            model_trajectories=model_trajectories,
            future_sampling=future_sampling,
            simulator=simulator,
            scorer=scorer,
        )
        for (state, record, pdm_row, _pred), result in zip(sample_items, results):
            updates.append({"state": state, "record": record, "pdm_row": pdm_row, "result": result})
    except Exception:
        # Keep the shard robust and preserve the original single-checkpoint semantics on rare scorer failures.
        for state, record, pdm_row, pred in sample_items:
            try:
                model_trajectory = Trajectory(poses=pred.numpy())
                result = pdm_score(
                    metric_cache=metric_cache,
                    model_trajectory=model_trajectory,
                    future_sampling=future_sampling,
                    simulator=simulator,
                    scorer=scorer,
                )
                updates.append({"state": state, "record": record, "pdm_row": pdm_row, "result": result})
            except Exception as exc:
                updates.append({"state": state, "record": record, "pdm_row": pdm_row, "failed": True, "error": repr(exc)})
    return updates


def apply_pdm_updates(updates: Sequence[Dict[str, Any]]) -> None:
    for update in updates:
        state = update["state"]
        record = update["record"]
        pdm_row = update["pdm_row"]
        if update.get("missing"):
            state["missing_metric_cache"] += 1
            record["pdm_valid"] = False
            record["pdm_error"] = update["error"]
            pdm_row.update({"valid": False, "error": record["pdm_error"]})
        elif update.get("failed"):
            state["failed_pdm"] += 1
            record["pdm_valid"] = False
            record["pdm_error"] = update["error"]
            pdm_row.update({"valid": False, "error": record["pdm_error"]})
        else:
            apply_pdm_result(state, record, pdm_row, update["result"])
        state["pdm_rows"].append(pdm_row)
        state["predictions"].append(record)


def batch_feature_data(action_input: BatchFeature) -> Dict[str, Any]:
    data = getattr(action_input, "data", None)
    if data is None:
        data = dict(action_input)
    return data


def action_input_signature(action_input: BatchFeature) -> Tuple[Tuple[str, Tuple[int, ...], str], ...]:
    signature = []
    for key, value in sorted(batch_feature_data(action_input).items()):
        if torch.is_tensor(value):
            signature.append((key, tuple(value.shape[1:]), str(value.dtype)))
        else:
            signature.append((key, (), type(value).__name__))
    return tuple(signature)


def stack_model_inputs(items: Sequence[Tuple[int, torch.Tensor, BatchFeature]]) -> Tuple[torch.Tensor, BatchFeature]:
    if not items:
        raise ValueError("Cannot stack an empty model-input list.")
    vl_features = torch.cat([item[1] for item in items], dim=0)
    keys = list(batch_feature_data(items[0][2]).keys())
    data: Dict[str, Any] = {}
    for key in keys:
        values = [batch_feature_data(item[2])[key] for item in items]
        if torch.is_tensor(values[0]):
            data[key] = torch.cat(values, dim=0)
        else:
            data[key] = values
    return vl_features, BatchFeature(data=data)


def split_batches(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    if batch_size <= 1:
        for item in items:
            yield [item]
        return
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def write_probe_shard_output(
    args: argparse.Namespace,
    probe: Dict[str, str],
    state: Dict[str, Any],
    chunk_dirs_used: Sequence[Path],
) -> None:
    out_dir = args.out_root / "eval" / safe_name(probe["name"]) / f"shard_{args.shard_index:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdm_values = state["pdm_metric_values"]
    pdm_averages = {key: mean_or_none(values) for key, values in pdm_values.items()}
    metrics = {
        "split": "navtrain_pdms_probe",
        "feature_source": "chunk",
        "checkpoint": probe["checkpoint"],
        "precision": args.precision,
        "trajectory_output_key": args.trajectory_output_key,
        "online_vlm_path": None,
        "online_vlm_lora_adapter_dir": None,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "chunk_cache_dirs": [str(path) for path in chunk_dirs_used],
        "metric_cache_dir": str(args.metric_cache_dir) if args.metric_cache_dir is not None else None,
        "num_samples": len(state["predictions"]),
        "target_teacher_tokens_disabled_in_eval": True,
        "trajectory_l1": sum(state["l1_values"]) / len(state["l1_values"]) if state["l1_values"] else None,
        "num_pdm_valid": len(pdm_values["score"]),
        "num_pdm_missing_metric_cache": state["missing_metric_cache"],
        "num_pdm_failed": state["failed_pdm"],
        "pdm_score": pdm_averages["score"],
        "PDMS": pdm_averages["score"],
        "NC": pdm_averages["no_at_fault_collisions"],
        "DAC": pdm_averages["drivable_area_compliance"],
        "TTC": pdm_averages["time_to_collision_within_bound"],
        "comfort": pdm_averages["comfort"],
        "EP": pdm_averages["ego_progress"],
        "DDC": pdm_averages["driving_direction_compliance"],
        "batch_probe_eval": True,
    }
    if args.metric_cache_dir is not None and metrics["num_pdm_valid"] == 0:
        raise RuntimeError(f"Probe {probe['name']} produced zero valid PDM rows.")
    (out_dir / "predictions.json").write_text(json.dumps(state["predictions"], indent=2) + "\n", encoding="utf-8")
    write_json(out_dir / "metrics.json", metrics)
    if state["pdm_rows"]:
        write_pdm_csv(out_dir / "pdm_results.csv", state["pdm_rows"], pdm_averages)
    report_lines = [
        "# ReCogDrive Batch Probe Evaluation Report",
        "",
        f"Config: `{probe['config']}`",
        f"Checkpoint: `{probe['checkpoint']}`",
        f"Probe: `{probe['name']}`",
        f"Shard: {args.shard_index}/{args.num_shards}",
        f"Samples: {len(state['predictions'])}",
        f"PDM score: {metrics.get('pdm_score')}",
    ]
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def run_shard(args: argparse.Namespace) -> int:
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards).")
    rows = probe_rows(args)
    if args.skip_done:
        rows = [row for row in rows if not shard_done(args.out_root, row, args.shard_index)]
    if not rows:
        print(json.dumps({"state": "all_shards_already_done", "shard_index": args.shard_index}))
        return 0

    shim_args = argparse.Namespace(
        chunk_cache_dir=None,
        chunk_cache_root=args.chunk_cache_root,
        chunk_name_pattern=args.chunk_name_pattern,
        max_samples=args.max_samples,
    )
    chunks = chunk_dirs(shim_args)
    sample_token_filter = load_sample_token_filter(args.sample_token_file)
    paths = sample_paths(chunks, None if sample_token_filter is not None else args.max_samples)
    if sample_token_filter is not None:
        paths = [item for item in paths if str(item[2].get("sample_token") or item[1].stem) in sample_token_filter]
        if args.max_samples is not None:
            paths = paths[: args.max_samples]
    paths = [item for idx, item in enumerate(paths) if idx % args.num_shards == args.shard_index]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_precision(args.precision) if device.type == "cuda" else torch.float32
    torch.set_grad_enabled(False)
    states = load_eval_states(args, rows, device, dtype)
    anchor_index = load_vlm_text_anchor_index(args.vlm_text_anchor_cache_root)
    metric_cache_loader = build_fast_metric_cache_loader(args.metric_cache_dir) if args.metric_cache_dir is not None else None
    use_async_pdm = bool(metric_cache_loader is not None and args.pdm_workers > 0)
    pdm_tools = build_pdm_tools() if metric_cache_loader is not None and not use_async_pdm else None
    pdm_executor = (
        concurrent.futures.ThreadPoolExecutor(max_workers=args.pdm_workers, thread_name_prefix="pdm")
        if use_async_pdm
        else None
    )
    pending_pdm: List[concurrent.futures.Future] = []
    pdm_queue_size = max(int(args.pdm_queue_size), int(args.pdm_workers), 1)

    def drain_pdm(*, wait_for_one: bool = False, wait_all: bool = False) -> None:
        nonlocal pending_pdm
        if not pending_pdm:
            return
        if wait_all:
            done = list(pending_pdm)
            pending_pdm = []
        else:
            timeout = None if wait_for_one else 0
            done_set, pending_set = concurrent.futures.wait(
                pending_pdm,
                timeout=timeout,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            done = list(done_set)
            pending_pdm = list(pending_set)
        for future in done:
            apply_pdm_updates(future.result())

    def submit_or_apply_pdm(
        sample_items: List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], torch.Tensor]],
        metric_cache: Any,
        metric_error: Optional[str],
    ) -> None:
        if metric_cache_loader is None:
            for state, record, _pdm_row, _pred in sample_items:
                state["predictions"].append(record)
            return
        if pdm_executor is None:
            apply_pdm_updates(score_sample_items(sample_items, metric_cache, metric_error, pdm_tools))
            return
        pending_pdm.append(pdm_executor.submit(score_sample_items, list(sample_items), metric_cache, metric_error, None))
        if len(pending_pdm) >= pdm_queue_size:
            drain_pdm(wait_for_one=True)

    print(
        json.dumps(
            {
                "state": "started",
                "host_key": args.host_key,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "num_probes": len(states),
                "num_samples": len(paths),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    last_progress_bucket = 0
    for batch_start, path_batch in enumerate(split_batches(paths, max(1, args.inference_batch_size))):
        entries = []
        for chunk_dir, path, index_record in path_batch:
            sample = load_sample(path)
            token = str(sample.get("sample_token", index_record.get("sample_token", path.stem)))
            metric_cache = None
            metric_error = None
            if metric_cache_loader is not None:
                if token not in metric_cache_loader.metric_cache_paths:
                    metric_error = "missing_metric_cache"
                else:
                    try:
                        metric_cache = metric_cache_loader.get_from_token(token)
                    except Exception as exc:
                        metric_error = repr(exc)
            entries.append(
                {
                    "chunk_dir": chunk_dir,
                    "path": path,
                    "index_record": index_record,
                    "sample": sample,
                    "token": token,
                    "metric_cache": metric_cache,
                    "metric_error": metric_error,
                    "batch_cache": {},
                }
            )

        preds_by_state: List[List[Optional[torch.Tensor]]] = [[None for _entry in entries] for _state in states]
        for state_index, state in enumerate(states):
            probe = state["probe"]
            planner = state["planner"]
            cache_key = batch_cache_key(planner)
            grouped_inputs: Dict[Tuple[Tuple[str, Tuple[int, ...], str], ...], List[Tuple[int, torch.Tensor, BatchFeature]]] = {}
            for entry_index, entry in enumerate(entries):
                if cache_key not in entry["batch_cache"]:
                    entry["batch_cache"][cache_key] = make_batch(
                        entry["sample"], planner, device, dtype, anchor_index=anchor_index
                    )
                vl_features, action_input = entry["batch_cache"][cache_key]
                grouped_inputs.setdefault(action_input_signature(action_input), []).append((entry_index, vl_features, action_input))
            for group_items in grouped_inputs.values():
                vl_features, action_input = stack_model_inputs(group_items)
                output = planner.get_action(vl_features, action_input, deterministic=args.deterministic)
                if args.trajectory_output_key not in output:
                    available = ", ".join(sorted(output.keys()))
                    raise KeyError(f"{probe['name']} missing output {args.trajectory_output_key!r}; available keys: {available}")
                pred_batch = output[args.trajectory_output_key].detach().float().cpu()
                if pred_batch.ndim == 2:
                    pred_batch = pred_batch.unsqueeze(0)
                if pred_batch.shape[0] != len(group_items):
                    raise RuntimeError(
                        f"{probe['name']} returned batch {pred_batch.shape[0]} predictions for {len(group_items)} inputs."
                    )
                if not torch.isfinite(pred_batch).all():
                    raise RuntimeError(f"Non-finite prediction for {probe['name']} batch_start={batch_start}")
                for row_index, (entry_index, _vl_features, _action_input) in enumerate(group_items):
                    preds_by_state[state_index][entry_index] = pred_batch[row_index]

        for entry_index, entry in enumerate(entries):
            sample = entry["sample"]
            path = entry["path"]
            chunk_dir = entry["chunk_dir"]
            token = entry["token"]
            sample_items: List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], torch.Tensor]] = []
            for state_index, state in enumerate(states):
                pred = preds_by_state[state_index][entry_index]
                if pred is None:
                    raise RuntimeError(f"Missing prediction for {state['probe']['name']} sample={path}")
                record = {
                    "path": str(path),
                    "chunk": str(chunk_dir),
                    "scene_token": str(sample.get("scene_token", path.stem)),
                    "sample_token": token,
                    "trajectory_output_key": args.trajectory_output_key,
                    "pred_traj": pred.tolist(),
                    "pdm_valid": None,
                }
                pdm_row: Dict[str, Any] = {
                    "sample_token": token,
                    "scene_token": record["scene_token"],
                    "chunk": chunk_dir.name,
                    "valid": None,
                }
                if "trajectory" in sample:
                    target = sample["trajectory"].float()
                    l1 = torch.nn.functional.l1_loss(pred, target).item()
                    record["trajectory_l1"] = l1
                    pdm_row["trajectory_l1"] = l1
                    state["l1_values"].append(l1)
                sample_items.append((state, record, pdm_row, pred))

            submit_or_apply_pdm(sample_items, entry["metric_cache"], entry["metric_error"])

        drain_pdm()

        done_count = min((batch_start + 1) * max(1, args.inference_batch_size), len(paths))
        progress_bucket = done_count // args.progress_every if args.progress_every > 0 else 0
        if args.progress_every > 0 and progress_bucket > last_progress_bucket:
            last_progress_bucket = progress_bucket
            print(
                json.dumps(
                    {
                        "state": "progress",
                        "shard_index": args.shard_index,
                        "done": done_count,
                        "total": len(paths),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    drain_pdm(wait_all=True)
    if pdm_executor is not None:
        pdm_executor.shutdown(wait=True)

    for state in states:
        write_probe_shard_output(args, state["probe"], state, chunks)
    print(json.dumps({"state": "done", "shard_index": args.shard_index, "num_probes": len(states)}, sort_keys=True), flush=True)
    return 0


def aggregate_one(args: argparse.Namespace, probe: Dict[str, str]) -> Optional[Dict[str, Any]]:
    run_dir = args.out_root / "eval" / safe_name(probe["name"])
    shard_metrics = []
    for shard in range(args.num_shards):
        metrics_path = run_dir / f"shard_{shard:02d}" / "metrics.json"
        if not metrics_path.is_file():
            return None
        shard_metrics.append(json.loads(metrics_path.read_text(encoding="utf-8")))
    payload: Dict[str, Any] = {
        "probe": probe["name"],
        "family": probe["family"],
        "config": probe["config"],
        "checkpoint": probe["checkpoint"],
        "navtest_pdms": float(probe["navtest_pdms"]),
        "navtest_l1": float(probe.get("navtest_l1") or float("nan")),
        "num_shards": args.num_shards,
        "num_samples": sum(int(item.get("num_samples") or 0) for item in shard_metrics),
        "num_pdm_valid": sum(int(item.get("num_pdm_valid") or 0) for item in shard_metrics),
        "num_pdm_missing_metric_cache": sum(int(item.get("num_pdm_missing_metric_cache") or 0) for item in shard_metrics),
        "num_pdm_failed": sum(int(item.get("num_pdm_failed") or 0) for item in shard_metrics),
        "trajectory_l1": metric_mean(shard_metrics, "trajectory_l1", "num_samples"),
        "shards": [str(run_dir / f"shard_{idx:02d}") for idx in range(args.num_shards)],
        "completed_at": utc_now(),
        "batch_probe_eval": True,
    }
    for key in METRIC_KEYS:
        payload[key] = metric_mean(shard_metrics, key, "num_pdm_valid")
    payload["pdm_score"] = payload["PDMS"]
    write_json(run_dir / "metrics.json", payload)
    return payload


def write_summary(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> None:
    path = args.out_root / "summary" / "probe_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "probe",
        "family",
        "checkpoint",
        "navtest_pdms",
        "PDMS",
        "trajectory_l1",
        "num_samples",
        "num_pdm_valid",
        "num_pdm_missing_metric_cache",
        "num_pdm_failed",
        "completed_at",
        "metrics_path",
    ]
    existing = []
    if path.is_file():
        existing = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    by_probe = {row.get("probe"): row for row in existing if row.get("probe")}
    for row in rows:
        by_probe[row["probe"]] = row
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(by_probe.values(), key=lambda item: str(item.get("probe"))):
            writer.writerow({key: row.get(key) for key in fields})


def aggregate(args: argparse.Namespace) -> int:
    rows = probe_rows(args)
    aggregated = []
    pending = []
    for probe in rows:
        payload = aggregate_one(args, probe)
        if payload is None:
            pending.append(probe["name"])
        else:
            aggregated.append({**payload, "metrics_path": str(args.out_root / "eval" / safe_name(probe["name"]) / "metrics.json")})
    if aggregated:
        write_summary(args, aggregated)
    print(json.dumps({"aggregated": len(aggregated), "pending": pending}, indent=2, sort_keys=True))
    return 0 if not pending else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch navtrain PDMS probe evaluator that reuses shard I/O across checkpoints.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out-root", type=Path, required=True)
    common.add_argument("--probe-manifest", type=Path, required=True)
    common.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    common.add_argument("--host-key", default="")
    common.add_argument("--probes", default="", help="Optional comma-separated probe names.")
    common.add_argument("--num-shards", type=int, default=8)
    common.add_argument("--trajectory-output-key", choices=("pred_traj", "pred_coarse_traj"), default="pred_traj")

    p_shard = sub.add_parser("shard", parents=[common])
    p_shard.add_argument("--shard-index", type=int, required=True)
    p_shard.add_argument("--chunk-cache-root", type=Path, default=DEFAULT_CHUNK_ROOT)
    p_shard.add_argument("--chunk-name-pattern", default="train_*")
    p_shard.add_argument("--metric-cache-dir", type=Path, default=DEFAULT_METRIC_CACHE)
    p_shard.add_argument("--sample-token-file", type=Path, required=True)
    p_shard.add_argument("--vlm-text-anchor-cache-root", type=Path, default=DEFAULT_ANCHOR_CACHE)
    p_shard.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="fp32")
    p_shard.add_argument("--inference-batch-size", type=int, default=1)
    p_shard.add_argument("--pdm-workers", type=int, default=0)
    p_shard.add_argument("--pdm-queue-size", type=int, default=32)
    p_shard.add_argument("--max-samples", type=int, default=None)
    p_shard.add_argument("--deterministic", action="store_true", default=True)
    p_shard.add_argument("--skip-done", action="store_true")
    p_shard.add_argument("--progress-every", type=int, default=500)
    p_shard.set_defaults(func=run_shard)

    p_agg = sub.add_parser("aggregate", parents=[common])
    p_agg.set_defaults(func=aggregate)
    return parser


def main() -> int:
    warnings.filterwarnings("default")
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
