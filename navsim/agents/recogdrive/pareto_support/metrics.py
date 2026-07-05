from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..trajectory_feasibility import compute_feasibility_metrics as _torch_feasibility


REQUIRED_METRIC_KEYS = (
    "pdms",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "history_comfort",
    "lane_keeping",
    "driving_direction_compliance",
    "traffic_light_compliance",
)


def cfg_value(cfg: Any, name: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def metric_value(metrics: Mapping[str, Any] | None, *names: str, default: float = 0.0) -> float:
    if not metrics:
        return float(default)
    for name in names:
        if name in metrics:
            return float(metrics[name])
    return float(default)


def compute_feasibility_metrics(traj: np.ndarray | torch.Tensor, cfg: Any = None) -> dict[str, float]:
    tensor = torch.as_tensor(traj, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    metrics = _torch_feasibility(tensor, cfg or {})
    first = 0
    traj_np = tensor.detach().cpu().numpy()[first]
    final = traj_np[-1] if traj_np.shape[0] else np.zeros(3, dtype=np.float32)
    out = {
        "feas_cost": float(metrics.feas_cost.reshape(-1)[first]),
        "curvature_violation": float(metrics.curvature_violation.reshape(-1)[first]),
        "curvature_excess_mean": float(metrics.curvature_violation.reshape(-1)[first]),
        "early_kink_violation": float(metrics.early_kink_violation.reshape(-1)[first]),
        "early_kink_cost": float(metrics.early_kink_violation.reshape(-1)[first]),
        "tail_reverse_violation": float(metrics.tail_reverse_violation.reshape(-1)[first]),
        "tail_reverse_cost": float(metrics.tail_reverse_violation.reshape(-1)[first]),
        "total_reverse_cost": float(metrics.reverse_violation.reshape(-1)[first]),
        "heading_jump_cost": float(metrics.heading_jump_violation.reshape(-1)[first]),
        "jerk_cost": float(metrics.jerk_violation.reshape(-1)[first]),
        "repair_distance": 0.0,
        "final_progress": float(final[0]),
        "endpoint_x": float(final[0]),
        "endpoint_y": float(final[1]),
        "early_kink_rate": float(metrics.early_kink_rate.reshape(-1)[first]),
        "tail_reverse_rate": float(metrics.tail_reverse_rate.reshape(-1)[first]),
        "curvature_violation_rate": float(metrics.curvature_violation_rate.reshape(-1)[first]),
    }
    return out


def normalize_feas_cost(feas_cost: float, cfg: Any = None) -> float:
    denom = float(cfg_value(cfg, "feas_cost_norm", cfg_value(cfg, "fpv3_feas_cost_max", 0.10)))
    if denom <= 0.0:
        denom = 1.0
    return float(feas_cost) / denom


def compute_metric_utility(metrics: Mapping[str, Any], cfg: Any = None) -> float:
    ep_w = float(cfg_value(cfg, "fpv3_ep_weight", cfg_value(cfg, "fp_ep_weight", 0.60)))
    ttc_w = float(cfg_value(cfg, "fpv3_ttc_weight", cfg_value(cfg, "fp_ttc_weight", 0.25)))
    ddc_w = float(cfg_value(cfg, "fpv3_ddc_weight", cfg_value(cfg, "fp_ddc_weight", 0.10)))
    feas_w = float(cfg_value(cfg, "fpv3_feas_weight", cfg_value(cfg, "fp_feas_weight", 0.05)))
    ep = metric_value(metrics, "ego_progress", "ep")
    ttc = metric_value(metrics, "time_to_collision_within_bound", "ttc")
    ddc = metric_value(metrics, "driving_direction_compliance", "ddc")
    feas = normalize_feas_cost(metric_value(metrics, "feas_cost", default=0.0), cfg)
    return float(ep_w * ep + ttc_w * ttc + ddc_w * ddc - feas_w * feas)


def is_valid_metrics(metrics: Mapping[str, Any], ref_metrics: Mapping[str, Any] | None = None, cfg: Any = None) -> bool:
    ref_metrics = ref_metrics or {}
    nc = metric_value(metrics, "no_at_fault_collisions", "nc")
    dac = metric_value(metrics, "drivable_area_compliance", "dac")
    ddc = metric_value(metrics, "driving_direction_compliance", "ddc")
    comfort = metric_value(metrics, "history_comfort", "comfort", default=1.0)
    ref_ddc = metric_value(ref_metrics, "driving_direction_compliance", "ddc", default=ddc)
    ref_comfort = metric_value(ref_metrics, "history_comfort", "comfort", default=comfort)
    feas_cost = metric_value(metrics, "feas_cost", default=0.0)
    ref_feas_cost = metric_value(ref_metrics, "feas_cost", default=feas_cost)
    ddc_min_absolute = float(cfg_value(cfg, "fpv3_ddc_min_absolute", cfg_value(cfg, "ddc_min_absolute", 0.95)))
    ddc_ref_floor = ref_ddc - float(cfg_value(cfg, "fpv3_ddc_drop_tolerance", cfg_value(cfg, "ddc_drop_tolerance", 0.01)))
    ddc_gate_mode = str(cfg_value(cfg, "support_ddc_gate_mode", "")).lower()
    if not ddc_gate_mode:
        ddc_gate_mode = "relax_ref_below_min" if bool(cfg_value(cfg, "support_relax_ddc_when_ref_below_min", False)) else "absolute_and_ref"
    if ddc_gate_mode in {"ref", "ref_relative", "gt_relative", "relative_to_ref"}:
        ddc_min = ddc_ref_floor
    elif ddc_gate_mode in {"absolute", "absolute_only"}:
        ddc_min = ddc_min_absolute
    elif ddc_gate_mode in {"relax_ref_below_min", "adaptive_ref"} and ref_ddc < ddc_min_absolute:
        ddc_min = ddc_ref_floor
    else:
        ddc_min = max(ddc_min_absolute, ddc_ref_floor)
    feas_max = float(cfg_value(cfg, "fpv3_feas_cost_max", cfg_value(cfg, "feas_cost_max", 0.10)))
    feas_gate_mode = str(cfg_value(cfg, "support_feas_gate_mode", "absolute")).lower()
    feas_tolerance = float(cfg_value(cfg, "support_feas_cost_tolerance", 0.03))
    if feas_gate_mode in {"disabled", "none", "off"}:
        feas_ok = True
    elif feas_gate_mode in {"ref", "ref_relative", "gt_relative", "relative_to_ref"}:
        feas_ok = feas_cost <= ref_feas_cost + feas_tolerance
    elif feas_gate_mode in {"relax_ref_above_max", "adaptive_ref"} and ref_feas_cost > feas_max:
        feas_ok = feas_cost <= ref_feas_cost + feas_tolerance
    else:
        feas_ok = feas_cost <= feas_max
    comfort_min_absolute = float(cfg_value(cfg, "fpv3_comfort_min", cfg_value(cfg, "comfort_min", 0.95)))
    comfort_gate_mode = str(cfg_value(cfg, "support_comfort_gate_mode", "absolute")).lower()
    comfort_drop_tolerance = float(
        cfg_value(cfg, "support_comfort_drop_tolerance", max(0.0, 1.0 - comfort_min_absolute))
    )
    if comfort_gate_mode in {"disabled", "none", "off"}:
        comfort_ok = True
    elif comfort_gate_mode in {"ref", "ref_relative", "gt_relative", "relative_to_ref"}:
        comfort_ok = comfort >= ref_comfort - comfort_drop_tolerance
    elif comfort_gate_mode in {"relax_ref_below_min", "adaptive_ref"} and ref_comfort < comfort_min_absolute:
        comfort_ok = comfort >= ref_comfort - comfort_drop_tolerance
    else:
        comfort_ok = comfort >= comfort_min_absolute
    return (
        nc >= 1.0
        and dac >= 1.0
        and ddc >= ddc_min
        and feas_ok
        and comfort_ok
    )


def merge_true_and_feasibility_metrics(
    true_metrics: Mapping[str, Any] | None,
    traj: np.ndarray | torch.Tensor,
    cfg: Any = None,
) -> dict[str, float]:
    out = {str(k): float(v) for k, v in dict(true_metrics or {}).items()}
    out.update(compute_feasibility_metrics(traj, cfg))
    out["utility"] = compute_metric_utility(out, cfg)
    return out


def _extract_pdm_row(result: Any) -> dict[str, float]:
    if isinstance(result, Mapping):
        raw = result
    else:
        raw = getattr(result, "__dict__", {})
    row: dict[str, float] = {}
    for key in REQUIRED_METRIC_KEYS:
        if key in raw:
            row[key] = float(raw[key])
            continue
        if key == "pdms":
            for alt in ("score", "pdm_score"):
                if alt in raw:
                    row[key] = float(raw[alt])
                    break
        if key not in row:
            row[key] = 0.0
    return row


def compute_true_metrics_batch(
    trajs: Sequence[np.ndarray] | np.ndarray,
    tokens: Sequence[str],
    metric_cache: Any,
    cfg: Any = None,
) -> list[dict[str, float]]:
    """Evaluate candidate trajectories when a real NAVSIM scoring context is provided.

    Supported contexts:
    - ``metric_cache`` is callable: called as ``metric_cache(trajs, tokens, cfg)``.
    - ``metric_cache`` is a dict with precomputed ``{token: [metrics...]}``.
    - ``cfg`` provides ``future_sampling``, ``simulator`` and ``scorer`` plus a token cache dict.
    """
    traj_arr = np.asarray(trajs, dtype=np.float32)
    if traj_arr.ndim == 2:
        traj_arr = traj_arr[None, ...]
    if len(tokens) != traj_arr.shape[0]:
        raise ValueError(f"tokens length {len(tokens)} does not match trajectories K={traj_arr.shape[0]}.")
    if callable(metric_cache):
        rows = metric_cache(traj_arr, list(tokens), cfg)
        return [merge_true_and_feasibility_metrics(row, traj_arr[idx], cfg) for idx, row in enumerate(rows)]
    if isinstance(metric_cache, Mapping) and not all(k in metric_cache for k in ("cache_dict", "future_sampling", "simulator", "scorer")):
        out = []
        counters: dict[str, int] = {}
        for idx, token in enumerate(tokens):
            rows = metric_cache.get(str(token), [])
            if isinstance(rows, Mapping):
                row = rows
            else:
                pos = counters.get(str(token), 0)
                counters[str(token)] = pos + 1
                if pos >= len(rows):
                    raise KeyError(f"No precomputed metric row {pos} for token={token!r}.")
                row = rows[pos]
            out.append(merge_true_and_feasibility_metrics(row, traj_arr[idx], cfg))
        return out

    context = metric_cache
    if isinstance(metric_cache, Mapping):
        context = metric_cache
    cache_dict = context.get("cache_dict") if isinstance(context, Mapping) else getattr(context, "cache_dict", None)
    future_sampling = context.get("future_sampling") if isinstance(context, Mapping) else getattr(context, "future_sampling", None)
    simulator = context.get("simulator") if isinstance(context, Mapping) else getattr(context, "simulator", None)
    scorer = context.get("scorer") if isinstance(context, Mapping) else getattr(context, "scorer", None)
    if cache_dict is None or future_sampling is None or simulator is None or scorer is None:
        raise RuntimeError("True evaluator context is unavailable; provide precomputed evaluator metrics or NAVSIM scoring objects.")

    from navsim.evaluate.pdm_score_batch import pdm_score_batch_same_cache

    token_to_indices: dict[str, list[int]] = {}
    for idx, token in enumerate(tokens):
        token_to_indices.setdefault(str(token), []).append(idx)
    rows: list[dict[str, float] | None] = [None] * traj_arr.shape[0]
    for token, indices in token_to_indices.items():
        pdm_rows = pdm_score_batch_same_cache(
            metric_cache=cache_dict[token],
            model_trajectories=traj_arr[indices],
            future_sampling=future_sampling,
            simulator=simulator,
            scorer=scorer,
            use_exact_array_conversion=bool(cfg_value(cfg, "use_exact_array_pdm_state_conversion", False)),
        )
        if len(pdm_rows) != len(indices):
            raise RuntimeError(f"PDM scorer returned {len(pdm_rows)} rows for {len(indices)} trajectories.")
        for local_idx, result in enumerate(pdm_rows):
            global_idx = indices[local_idx]
            rows[global_idx] = merge_true_and_feasibility_metrics(_extract_pdm_row(result), traj_arr[global_idx], cfg)
    if any(row is None for row in rows):
        raise RuntimeError("True evaluator did not produce all rows.")
    return [row for row in rows if row is not None]
