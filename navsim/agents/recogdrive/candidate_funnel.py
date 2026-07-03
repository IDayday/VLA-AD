from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .offline_action_explorer import (
    build_structured_perturbations,
    clamp_heading_steps,
    enforce_forward_monotonic_x,
    generate_delay,
    generate_endpoint_extension,
    generate_lateral_offsets,
    generate_slow_first,
    smooth_trajectory,
)
from .pareto_support import CandidateRecord


def _cfg_value(cfg: Any, name: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _default_components() -> dict[str, float]:
    return {}


class ExternalCandidateLoader:
    def __init__(self, source_roots: dict[str, str]):
        self.source_roots = {str(k): Path(v) for k, v in dict(source_roots).items()}
        self._payload_cache: dict[Path, list[Any]] = {}
        self._indexed_roots: dict[tuple[str, Path], dict[str, list[CandidateRecord]]] = {}

    def _load_payloads(self, path: Path) -> Iterable[Any]:
        if not path.exists():
            return []
        cache_key = path.resolve()
        if cache_key in self._payload_cache:
            return self._payload_cache[cache_key]
        if path.is_dir():
            payloads = []
            for suffix in ("*.pkl", "*.npz", "*.pt", "*.pth", "*.pkl.xz"):
                for item in sorted(path.rglob(suffix)):
                    payloads.extend(self._load_payloads(item))
            self._payload_cache[cache_key] = payloads
            return payloads
        if path.suffix == ".xz" and path.name.endswith(".pkl.xz"):
            import lzma

            with lzma.open(path, "rb") as f:
                payloads = [pickle.load(f)]
            self._payload_cache[cache_key] = payloads
            return payloads
        if path.suffix == ".pkl":
            with open(path, "rb") as f:
                payloads = [pickle.load(f)]
            self._payload_cache[cache_key] = payloads
            return payloads
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=True) as data:
                payloads = [{key: data[key] for key in data.files}]
            self._payload_cache[cache_key] = payloads
            return payloads
        if path.suffix in {".pt", ".pth"}:
            payloads = [torch.load(path, map_location="cpu")]
            self._payload_cache[cache_key] = payloads
            return payloads
        return []

    def _index_root(self, source: str, root: Path) -> dict[str, list[CandidateRecord]]:
        cache_key = (str(source), root.resolve())
        indexed = self._indexed_roots.get(cache_key)
        if indexed is not None:
            return indexed
        out: dict[str, list[CandidateRecord]] = defaultdict(list)
        for payload in self._load_payloads(root):
            for record in _records_from_any_payload(payload, source=str(source)):
                out[str(record.token)].append(record)
        indexed = dict(out)
        self._indexed_roots[cache_key] = indexed
        return indexed

    def load(self, token: str) -> list[CandidateRecord]:
        records: list[CandidateRecord] = []
        for source, root in self.source_roots.items():
            payloads: list[Any] = []
            if root.is_dir():
                for suffix in (".pkl", ".npz", ".pt", ".pth", ".pkl.xz"):
                    payloads.extend(self._load_payloads(root / f"{token}{suffix}"))
            if not payloads:
                indexed = self._index_root(source, root)
                records.extend(indexed.get(str(token), []))
                continue
            for payload in payloads:
                records.extend(_records_from_payload(payload, token=str(token), source=source))
        return records


def _trajectory_array(value: Any) -> np.ndarray | None:
    if hasattr(value, "poses"):
        value = getattr(value, "poses")
    elif isinstance(value, dict) and "poses" in value:
        value = value["poses"]
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return None
    if arr.ndim == 2 and arr.shape[-1] == 3:
        return arr[None, ...]
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr
    return None


def _records_from_any_payload(payload: Any, source: str) -> list[CandidateRecord]:
    """Index token-keyed candidate payloads, including NAVSIM submission.pkl files."""
    if isinstance(payload, list):
        out: list[CandidateRecord] = []
        for item in payload:
            out.extend(_records_from_any_payload(item, source))
        return out
    if isinstance(payload, dict):
        if "predictions" in payload and isinstance(payload["predictions"], list):
            out: list[CandidateRecord] = []
            for prediction_map in payload["predictions"]:
                out.extend(_records_from_any_payload(prediction_map, source))
            return out
        if "token" in payload:
            return _records_from_payload(payload, token=str(payload["token"]), source=source)
        out: list[CandidateRecord] = []
        metadata_keys = {
            "team_name",
            "authors",
            "email",
            "institution",
            "country / region",
            "country",
            "version",
            "metadata",
        }
        for token, value in payload.items():
            if str(token) in metadata_keys:
                continue
            arr = _trajectory_array(value)
            if arr is None:
                continue
            for idx in range(arr.shape[0]):
                out.append(
                    CandidateRecord(
                        trajectory=arr[idx],
                        source=source,
                        token=str(token),
                        components=_default_components(),
                        reward=0.0,
                        feas={},
                        selection_score=0.0,
                    )
                )
        return out
    return []


def _records_from_payload(payload: Any, token: str, source: str) -> list[CandidateRecord]:
    if isinstance(payload, list):
        out: list[CandidateRecord] = []
        for item in payload:
            out.extend(_records_from_payload(item, token, source))
        return out
    if isinstance(payload, dict):
        if "token" in payload and str(payload["token"]) != str(token):
            return []
        trajs = payload.get("trajectories", payload.get("trajectory", payload.get("candidates")))
        if trajs is None:
            arr = _trajectory_array(payload)
            if arr is None:
                return []
            trajs = arr
        arr = _trajectory_array(trajs)
        if arr is None:
            return []
        components = payload.get("components", {})
        rewards = np.asarray(payload.get("rewards", payload.get("reward", np.zeros(arr.shape[0]))), dtype=np.float32)
        if rewards.ndim == 0:
            rewards = np.full((arr.shape[0],), float(rewards), dtype=np.float32)
        records = []
        for idx in range(arr.shape[0]):
            comp = _default_components()
            if isinstance(components, dict):
                for key, value in components.items():
                    values = np.asarray(value)
                    comp[key] = float(values[idx] if values.shape[:1] == (arr.shape[0],) else values)
            records.append(
                CandidateRecord(
                    trajectory=arr[idx],
                    source=source,
                    token=token,
                    components=comp,
                    reward=float(rewards[min(idx, rewards.shape[0] - 1)]),
                    feas={},
                    selection_score=float(rewards[min(idx, rewards.shape[0] - 1)]),
                )
            )
        return records
    arr = np.asarray(payload, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return []
    return [
        CandidateRecord(
            trajectory=arr[idx],
            source=source,
            token=token,
            components=_default_components(),
            reward=0.0,
            feas={},
            selection_score=0.0,
        )
        for idx in range(arr.shape[0])
    ]


def _candidate(traj: np.ndarray | torch.Tensor, source: str, token: str, parent_id: str = "") -> CandidateRecord:
    arr = traj.detach().cpu().numpy() if isinstance(traj, torch.Tensor) else np.asarray(traj, dtype=np.float32)
    return CandidateRecord(
        trajectory=arr.astype(np.float32),
        source=source,
        token=str(token),
        components=_default_components(),
        reward=0.0,
        feas={},
        selection_score=0.0,
        parent_id=parent_id,
    )


def build_seed_anchors(
    gt: np.ndarray | torch.Tensor | None,
    il: np.ndarray | torch.Tensor | None,
    current_policy: np.ndarray | torch.Tensor | None,
    external_candidates: Iterable[CandidateRecord],
    cfg: Any,
) -> list[CandidateRecord]:
    token = str(_cfg_value(cfg, "token", ""))
    records: list[CandidateRecord] = []
    if gt is not None:
        records.append(_candidate(gt, "gt", token))
    if il is not None:
        records.append(_candidate(il, "il", token))
    if current_policy is not None:
        arr = current_policy.detach().cpu().numpy() if isinstance(current_policy, torch.Tensor) else np.asarray(current_policy)
        if arr.ndim == 2:
            arr = arr[None, ...]
        for idx in range(arr.shape[0]):
            records.append(_candidate(arr[idx], "current_policy", token, parent_id=f"policy:{idx}"))
    records.extend(list(external_candidates))
    return records


def pareto_nms(candidates: list[CandidateRecord], scorer_or_proxy: Any, cfg: Any) -> list[CandidateRecord]:
    if not candidates:
        return []
    threshold = float(_cfg_value(cfg, "pareto_nms_endpoint_threshold_m", 0.5))
    limit = int(_cfg_value(cfg, "max_candidates_per_scene", len(candidates)))
    scored = []
    for cand in candidates:
        score = float(cand.selection_score)
        if callable(scorer_or_proxy):
            score = float(scorer_or_proxy(cand))
        scored.append((score, cand))
    selected: list[CandidateRecord] = []
    endpoints: list[np.ndarray] = []
    for _, cand in sorted(scored, key=lambda item: item[0], reverse=True):
        end = np.asarray(cand.trajectory, dtype=np.float32)[-1, :2]
        if endpoints and min(float(np.linalg.norm(end - prev)) for prev in endpoints) < threshold:
            continue
        selected.append(cand)
        endpoints.append(end)
        if len(selected) >= limit:
            break
    return selected


def _expand_with_variants(anchor: CandidateRecord, variants: torch.Tensor, source: str) -> list[CandidateRecord]:
    if variants.ndim == 4:
        variants = variants.reshape(-1, variants.shape[-2], variants.shape[-1])
    return [_candidate(variants[idx], source, anchor.token, parent_id=anchor.parent_id or anchor.source) for idx in range(variants.shape[0])]


def failure_conditioned_expand(anchor: CandidateRecord, ref: dict, cfg: Any) -> list[CandidateRecord]:
    base = torch.as_tensor(anchor.trajectory, dtype=torch.float32).unsqueeze(0)
    variants = []
    variants.append(generate_endpoint_extension(base, _cfg_value(cfg, "progress_endpoint_deltas_m", (0.5, 1.0, 2.0))))
    variants.append(generate_slow_first(base, _cfg_value(cfg, "timing_slow_first_scales", (0.7, 0.85))))
    variants.append(generate_delay(base, _cfg_value(cfg, "timing_delay_strengths", (0.15, 0.3))))
    variants.append(generate_lateral_offsets(base, _cfg_value(cfg, "lateral_offsets_m", (-0.5, 0.5))))
    out = []
    for idx, var in enumerate(variants):
        out.extend(_expand_with_variants(anchor, var, f"failure_expand_{idx}"))
    return out


def trust_region_control_expand(anchor: CandidateRecord, cfg: Any) -> list[CandidateRecord]:
    base = torch.as_tensor(anchor.trajectory, dtype=torch.float32)
    variants = [
        smooth_trajectory(base),
        enforce_forward_monotonic_x(base),
        clamp_heading_steps(base, float(_cfg_value(cfg, "max_heading_step_rad", 0.25))),
    ]
    return [_candidate(var, f"trust_region_{idx}", anchor.token, parent_id=anchor.parent_id or anchor.source) for idx, var in enumerate(variants)]


def build_candidate_pool_for_scene(
    token: str,
    gt: np.ndarray | torch.Tensor | None,
    il: np.ndarray | torch.Tensor | None,
    current_policy: np.ndarray | torch.Tensor | None,
    external_candidates: Iterable[CandidateRecord],
    cfg: Any,
) -> list[CandidateRecord]:
    if isinstance(cfg, dict):
        cfg = dict(cfg)
        cfg.setdefault("token", token)
    anchors = build_seed_anchors(gt, il, current_policy, external_candidates, cfg)
    pool = list(anchors)
    for anchor in anchors:
        pool.extend(failure_conditioned_expand(anchor, {}, cfg))
        pool.extend(trust_region_control_expand(anchor, cfg))
    if gt is not None or il is not None:
        tensor_anchors = []
        sources = []
        for rec in anchors:
            if rec.source in {"gt", "il"}:
                tensor_anchors.append(torch.as_tensor(rec.trajectory, dtype=torch.float32))
                sources.append(rec.source)
        if tensor_anchors:
            stacked = torch.stack(tensor_anchors, dim=0).unsqueeze(0)
            variants, variant_sources, _ = build_structured_perturbations(stacked, sources, cfg)
            if variants.numel() > 0:
                for idx in range(variants.shape[1]):
                    pool.append(_candidate(variants[0, idx], variant_sources[idx], token))
    return pareto_nms(pool, None, cfg)
