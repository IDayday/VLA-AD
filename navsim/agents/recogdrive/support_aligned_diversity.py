from __future__ import annotations

from dataclasses import dataclass
import hashlib
import lzma
import math
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SupportAlignedDiversityConfig:
    """Configuration for model-agnostic, support-relative policy evaluation."""

    endpoint_scale_m: float = 3.0
    mean_xy_scale_m: float = 1.5
    progress_scale_m: float = 1.5
    lateral_scale_m: float = 0.8
    heading_scale_rad: float = 0.35
    endpoint_weight: float = 0.30
    mean_xy_weight: float = 0.30
    progress_weight: float = 0.15
    lateral_weight: float = 0.15
    heading_weight: float = 0.10
    bandwidths: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
    density_bandwidth: float = 0.40
    mode_threshold: float = 0.40
    hard_match_threshold: float = 0.50
    dispersion_floor: float = 0.05

    def __post_init__(self) -> None:
        scales = (
            self.endpoint_scale_m,
            self.mean_xy_scale_m,
            self.progress_scale_m,
            self.lateral_scale_m,
            self.heading_scale_rad,
        )
        weights = (
            self.endpoint_weight,
            self.mean_xy_weight,
            self.progress_weight,
            self.lateral_weight,
            self.heading_weight,
        )
        if any(value <= 0.0 for value in scales):
            raise ValueError("All trajectory-distance scales must be positive.")
        if any(value < 0.0 for value in weights) or sum(weights) <= 0.0:
            raise ValueError("Trajectory-distance weights must be non-negative with a positive sum.")
        if not self.bandwidths or any(value <= 0.0 for value in self.bandwidths):
            raise ValueError("bandwidths must contain positive values.")
        if self.density_bandwidth <= 0.0 or self.mode_threshold <= 0.0:
            raise ValueError("density_bandwidth and mode_threshold must be positive.")
        if self.hard_match_threshold <= 0.0 or self.dispersion_floor <= 0.0:
            raise ValueError("hard_match_threshold and dispersion_floor must be positive.")


@dataclass(frozen=True)
class SupportReference:
    token: str
    trajectories: np.ndarray
    tags: tuple[str, ...]
    rewards: np.ndarray
    archive_version: int


def _as_trajectory_batch(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3 or array.shape[-1] != 3 or array.shape[-2] < 1:
        raise ValueError(f"{name} must have shape [N, H, 3], got {array.shape}.")
    if array.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one trajectory.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf.")
    return array


def _normalized_weights(config: SupportAlignedDiversityConfig) -> np.ndarray:
    weights = np.asarray(
        [
            config.endpoint_weight,
            config.mean_xy_weight,
            config.progress_weight,
            config.lateral_weight,
            config.heading_weight,
        ],
        dtype=np.float64,
    )
    return weights / weights.sum()


def trajectory_distance_matrix(
    lhs: np.ndarray,
    rhs: np.ndarray,
    config: SupportAlignedDiversityConfig = SupportAlignedDiversityConfig(),
) -> np.ndarray:
    """Returns the fixed, dimensionless trajectory distance for every pair."""

    left = _as_trajectory_batch(lhs, name="lhs")
    right = _as_trajectory_batch(rhs, name="rhs")
    if left.shape[1] != right.shape[1]:
        raise ValueError(f"Trajectory horizons differ: {left.shape[1]} vs {right.shape[1]}.")

    delta_xy = left[:, None, :, :2] - right[None, :, :, :2]
    xy_distance = np.linalg.norm(delta_xy, axis=-1)
    heading_delta = left[:, None, :, 2] - right[None, :, :, 2]
    wrapped_heading = np.abs(np.arctan2(np.sin(heading_delta), np.cos(heading_delta)))
    components = np.stack(
        [
            xy_distance[..., -1],
            xy_distance.mean(axis=-1),
            np.abs(delta_xy[..., 0]).mean(axis=-1),
            np.abs(delta_xy[..., 1]).mean(axis=-1),
            wrapped_heading.mean(axis=-1),
        ],
        axis=-1,
    )
    scales = np.asarray(
        [
            config.endpoint_scale_m,
            config.mean_xy_scale_m,
            config.progress_scale_m,
            config.lateral_scale_m,
            config.heading_scale_rad,
        ],
        dtype=np.float64,
    )
    return np.sum((components / scales) * _normalized_weights(config), axis=-1)


def density_balanced_support_weights(
    support_distance: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """Gives near-duplicate support points a fixed total mass instead of extra votes."""

    distance = np.asarray(support_distance, dtype=np.float64)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1] or distance.shape[0] < 1:
        raise ValueError(f"support_distance must be a non-empty square matrix, got {distance.shape}.")
    if not np.isfinite(distance).all() or bandwidth <= 0.0:
        raise ValueError("support_distance must be finite and bandwidth must be positive.")
    similarity = np.exp(-0.5 * np.square(distance / float(bandwidth)))
    inverse_density = 1.0 / similarity.sum(axis=1).clip(min=1e-12)
    return inverse_density / inverse_density.sum().clip(min=1e-12)


def complete_linkage_mode_medoids(distance: np.ndarray, threshold: float) -> list[int]:
    """Returns deterministic medoids for an interpretable mode-count diagnostic."""

    matrix = np.asarray(distance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError(f"distance must be a non-empty square matrix, got {matrix.shape}.")
    clusters: list[list[int]] = [[index] for index in range(matrix.shape[0])]
    while len(clusters) > 1:
        candidates: list[tuple[float, int, int]] = []
        for left_index in range(len(clusters)):
            for right_index in range(left_index + 1, len(clusters)):
                linkage = max(
                    matrix[left_member, right_member]
                    for left_member in clusters[left_index]
                    for right_member in clusters[right_index]
                )
                candidates.append((float(linkage), left_index, right_index))
        linkage, left_index, right_index = min(candidates)
        if linkage > threshold:
            break
        clusters[left_index].extend(clusters[right_index])
        clusters[left_index].sort()
        clusters.pop(right_index)

    medoids: list[int] = []
    for cluster in clusters:
        medoids.append(
            min(
                cluster,
                key=lambda index: (float(matrix[index, cluster].mean()), index),
            )
        )
    return medoids


def _mean_off_diagonal(distance: np.ndarray, weights: np.ndarray | None = None) -> float:
    matrix = np.asarray(distance, dtype=np.float64)
    if matrix.shape[0] < 2:
        return 0.0
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    if weights is None:
        return float(matrix[mask].mean())
    probability = np.outer(weights, weights)
    denominator = float(probability[mask].sum())
    if denominator <= 1e-12:
        return 0.0
    return float((matrix[mask] * probability[mask]).sum() / denominator)


def _effective_mode_metrics(
    prediction_to_support: np.ndarray,
    support_distance: np.ndarray,
    config: SupportAlignedDiversityConfig,
) -> tuple[float, float, float]:
    medoids = complete_linkage_mode_medoids(support_distance, config.mode_threshold)
    medoid_distance = prediction_to_support[:, medoids]
    nearest_mode = medoid_distance.argmin(axis=1)
    nearest_distance = medoid_distance.min(axis=1)
    matched = nearest_distance <= config.hard_match_threshold
    if not matched.any():
        return float(len(medoids)), 0.0, 0.0
    counts = np.bincount(nearest_mode[matched], minlength=len(medoids)).astype(np.float64)
    probabilities = counts / counts.sum()
    effective_modes = float(1.0 / np.square(probabilities).sum().clip(min=1e-12))
    normalized = min(effective_modes / max(float(len(medoids)), 1.0), 1.0)
    return float(len(medoids)), effective_modes, float(normalized)


def _kernel_effective_modes(
    distance: np.ndarray,
    weights: np.ndarray,
    bandwidth: float,
) -> float:
    """Returns a density-aware soft mode count using the kernel participation ratio."""

    matrix = np.asarray(distance, dtype=np.float64)
    probability = np.asarray(weights, dtype=np.float64).reshape(-1)
    if matrix.shape != (probability.size, probability.size):
        raise ValueError("Kernel distance and probability shapes differ.")
    probability = probability / probability.sum().clip(min=1e-12)
    squared_similarity = np.exp(-np.square(matrix / float(bandwidth)))
    collision_probability = float(probability @ squared_similarity @ probability)
    return float(1.0 / max(collision_probability, 1e-12))


def compute_support_aligned_diversity(
    predictions: np.ndarray,
    support: np.ndarray,
    config: SupportAlignedDiversityConfig = SupportAlignedDiversityConfig(),
) -> dict[str, float]:
    """Computes scene-level metrics before any cross-scene averaging."""

    policy = _as_trajectory_batch(predictions, name="predictions")
    reference = _as_trajectory_batch(support, name="support")
    prediction_to_support = trajectory_distance_matrix(policy, reference, config)
    support_distance = trajectory_distance_matrix(reference, reference, config)
    policy_distance = trajectory_distance_matrix(policy, policy, config)
    support_weights = density_balanced_support_weights(support_distance, config.density_bandwidth)

    metrics: dict[str, float] = {
        "num_predictions": float(policy.shape[0]),
        "num_supports": float(reference.shape[0]),
        "nearest_support_distance_mean": float(prediction_to_support.min(axis=1).mean()),
        "nearest_support_distance_p95": float(np.quantile(prediction_to_support.min(axis=1), 0.95)),
        "hard_on_support_rate": float(
            np.mean(prediction_to_support.min(axis=1) <= config.hard_match_threshold)
        ),
        "hard_support_coverage": float(
            np.sum(
                support_weights
                * (prediction_to_support.min(axis=0) <= config.hard_match_threshold).astype(np.float64)
            )
        ),
    }

    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    effective_mode_ratio_values: list[float] = []
    policy_uniform_weights = np.full(policy.shape[0], 1.0 / float(policy.shape[0]), dtype=np.float64)
    for bandwidth in config.bandwidths:
        affinity = np.exp(-0.5 * np.square(prediction_to_support / float(bandwidth)))
        precision = float(affinity.max(axis=1).mean())
        recall = float(np.sum(support_weights * affinity.max(axis=0)))
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        suffix = str(bandwidth).replace(".", "p")
        metrics[f"support_precision_bw_{suffix}"] = precision
        metrics[f"support_recall_bw_{suffix}"] = recall
        metrics[f"support_f1_bw_{suffix}"] = f1
        reference_effective_modes = _kernel_effective_modes(
            support_distance,
            support_weights,
            bandwidth,
        )
        policy_effective_modes = _kernel_effective_modes(
            policy_distance,
            policy_uniform_weights,
            bandwidth,
        )
        effective_mode_ratio = min(policy_effective_modes / max(reference_effective_modes, 1e-12), 1.0)
        metrics[f"reference_kernel_effective_modes_bw_{suffix}"] = reference_effective_modes
        metrics[f"policy_kernel_effective_modes_bw_{suffix}"] = policy_effective_modes
        metrics[f"kernel_effective_mode_ratio_bw_{suffix}"] = float(effective_mode_ratio)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        effective_mode_ratio_values.append(float(effective_mode_ratio))

    precision_auc = float(np.mean(precision_values))
    recall_auc = float(np.mean(recall_values))
    f1_auc = float(np.mean(f1_values))
    effective_mode_coverage_auc = float(np.mean(effective_mode_ratio_values))
    support_dispersion = _mean_off_diagonal(support_distance, support_weights)
    policy_dispersion = _mean_off_diagonal(policy_distance)
    dispersion_ratio = (policy_dispersion + config.dispersion_floor) / (
        support_dispersion + config.dispersion_floor
    )
    dispersion_calibration = float(math.exp(-abs(math.log(max(dispersion_ratio, 1e-12)))))
    mode_count, effective_modes, normalized_effective_modes = _effective_mode_metrics(
        prediction_to_support,
        support_distance,
        config,
    )

    policy_xy = policy[..., :2]
    pairwise_xy = np.linalg.norm(policy_xy[:, None] - policy_xy[None, :], axis=-1)
    off_diagonal = ~np.eye(policy.shape[0], dtype=bool)
    raw_pairwise_ade = float(pairwise_xy.mean(axis=-1)[off_diagonal].mean()) if policy.shape[0] > 1 else 0.0
    raw_pairwise_fde = float(pairwise_xy[..., -1][off_diagonal].mean()) if policy.shape[0] > 1 else 0.0

    metrics.update(
        {
            "support_precision_auc": precision_auc,
            "density_corrected_support_recall_auc": recall_auc,
            "support_f1_auc": f1_auc,
            "kernel_effective_mode_coverage_auc": effective_mode_coverage_auc,
            "support_dispersion": support_dispersion,
            "policy_dispersion": policy_dispersion,
            "relative_dispersion_ratio": float(dispersion_ratio),
            "relative_dispersion_calibration": dispersion_calibration,
            "reference_mode_count": mode_count,
            "policy_effective_mode_count": effective_modes,
            "normalized_effective_mode_coverage": normalized_effective_modes,
            "raw_pairwise_ade_m": raw_pairwise_ade,
            "raw_pairwise_fde_m": raw_pairwise_fde,
            "snsad_v1": float((precision_auc * recall_auc * dispersion_calibration) ** (1.0 / 3.0)),
            "snsad": float(
                (
                    precision_auc
                    * recall_auc
                    * effective_mode_coverage_auc
                    * dispersion_calibration
                )
                ** 0.25
            ),
        }
    )
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("Support-aligned diversity metrics contain NaN or Inf.")
    return metrics


def support_archive_path(root: str | Path, token: str) -> Path:
    digest = hashlib.sha1(str(token).encode("utf-8")).hexdigest()
    return Path(root) / f"{digest}.pkl.xz"


def _load_payload(path: Path) -> Mapping[str, Any]:
    with lzma.open(path, "rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Support archive must contain a mapping, got {type(payload).__name__}: {path}")
    return payload


def load_support_reference(path_or_root: str | Path, token: str | None = None) -> SupportReference:
    """Loads both the deployed legacy v3 array schema and dataclass-style archives."""

    path = support_archive_path(path_or_root, token) if token is not None else Path(path_or_root)
    payload = _load_payload(path)
    version = int(payload.get("version", 1))
    if "candidates" in payload and "support_indices" in payload:
        all_trajectories = _as_trajectory_batch(payload["candidates"], name="candidates")
        indices = np.asarray(payload["support_indices"], dtype=np.int64).reshape(-1)
        if indices.size < 1 or (indices < 0).any() or (indices >= all_trajectories.shape[0]).any():
            raise ValueError(f"Invalid support_indices in {path}.")
        tags_all = list(payload.get("support_tags", [""] * all_trajectories.shape[0]))
        rewards_all = np.asarray(payload.get("rewards", np.zeros(all_trajectories.shape[0])), dtype=np.float64)
        return SupportReference(
            token=str(payload.get("token", token or "")),
            trajectories=all_trajectories[indices],
            tags=tuple(str(tags_all[index] or "") for index in indices),
            rewards=rewards_all[indices],
            archive_version=version,
        )

    raw_support = payload.get("support_set", [])
    if not isinstance(raw_support, Sequence) or not raw_support:
        raise ValueError(f"Support archive contains no positive support trajectories: {path}")
    trajectories: list[np.ndarray] = []
    tags: list[str] = []
    rewards: list[float] = []
    for item in raw_support:
        if isinstance(item, Mapping):
            trajectories.append(np.asarray(item["trajectory"], dtype=np.float64))
            tags.append(str(item.get("support_category", "") or ""))
            true_metrics = item.get("true_metrics") or {}
            rewards.append(float(true_metrics.get("pdms", true_metrics.get("score", 0.0))))
        else:
            trajectories.append(np.asarray(item.trajectory, dtype=np.float64))
            tags.append(str(getattr(item, "support_category", "") or ""))
            true_metrics = getattr(item, "true_metrics", None) or {}
            rewards.append(float(true_metrics.get("pdms", true_metrics.get("score", 0.0))))
    return SupportReference(
        token=str(payload.get("scene_token", token or "")),
        trajectories=_as_trajectory_batch(trajectories, name="support_set"),
        tags=tuple(tags),
        rewards=np.asarray(rewards, dtype=np.float64),
        archive_version=version,
    )
