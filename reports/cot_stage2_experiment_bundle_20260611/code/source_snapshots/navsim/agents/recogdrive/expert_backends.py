from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import hashlib
import warnings

import torch


EXPERT_FEATURE_SOURCE_CHOICES = ("none", "dummy", "chunk", "disk", "online", "cache", "real")
EXPERT_CONTEXT_KEYS = ("jepa_context_tokens", "vggt_context_tokens")
EXPERT_LEGACY_CONTEXT_KEYS = ("jepa_tokens", "vggt_tokens")
EXPERT_GEOMETRY_CONTEXT_KEYS = (
    "vggt_geometry_tokens",
    "vggt_depth_tokens",
    "vggt_pointmap_tokens",
    "vggt_camera_tokens",
    "vggt_geometry_mode_code",
)
EXPERT_TARGET_KEYS = (
    "jepa_target_tokens",
    "vggt_target_tokens",
    "vggt_geometry_target_tokens",
    "vggt_depth_target_tokens",
    "vggt_pointmap_target_tokens",
)
EXPERT_RISK_KEYS = (
    "risk_labels",
    "generic_risk_labels",
    "drivable_risk_labels",
    "ttc_risk_labels",
    "comfort_risk_labels",
)
EXPERT_LAST_VLA_TEACHER_KEYS = (
    "teacher_trajectory",
    "teacher_trajectory_norm",
    "teacher_score",
    "gt_score",
    "oracle_best_of_k_score",
    "candidate_count",
)
EXPERT_LAST_VLA_TEXT_ANCHOR_KEYS = (
    "vlm_text_trajectory",
    "vlm_text_trajectory_norm",
    "vlm_text_parse_ok",
)
EXPERT_ALL_KEYS = (
    *EXPERT_CONTEXT_KEYS,
    *EXPERT_GEOMETRY_CONTEXT_KEYS,
    *EXPERT_TARGET_KEYS,
    *EXPERT_RISK_KEYS,
    *EXPERT_LAST_VLA_TEACHER_KEYS,
    *EXPERT_LAST_VLA_TEXT_ANCHOR_KEYS,
    *EXPERT_LEGACY_CONTEXT_KEYS,
)
DUMMY_EXPERT_WARNING = (
    "Dummy expert features are for computation-flow validation only. "
    "They are not training data and must not be used for performance claims."
)


def normalize_expert_feature_source(
    source: Optional[str],
    *,
    use_expert_features: bool,
    expert_cache_dir: Optional[str] = None,
) -> str:
    """Normalizes expert feature source while preserving old cache-dir behavior."""
    if not use_expert_features:
        return "none"

    normalized = (source or "none").strip().lower()
    if normalized == "cache":
        normalized = "chunk"
    elif normalized == "real":
        normalized = "online"
    if normalized == "none" and expert_cache_dir:
        normalized = "chunk"
    if normalized not in EXPERT_FEATURE_SOURCE_CHOICES:
        raise ValueError(
            f"expert_feature_source={source!r} is invalid. "
            f"Expected one of {EXPERT_FEATURE_SOURCE_CHOICES}."
        )
    return normalized


def _stable_offset(sample_key: object, seed: int) -> float:
    key = f"{seed}:{sample_key}".encode("utf-8")
    digest = hashlib.sha1(key).digest()
    integer = int.from_bytes(digest[:4], byteorder="little", signed=False)
    return (integer % 10000) / 997.0


@dataclass
class DummyExpertBackend:
    """Deterministic fake JEPA/VGGT tensor backend for smoke tests."""

    num_jepa_tokens: int = 12
    num_vggt_tokens: int = 12
    jepa_dim: int = 1024
    vggt_dim: int = 2048
    use_jepa: bool = True
    use_vggt: bool = True
    seed: int = 0

    def _tokens(
        self,
        shape: tuple[int, int, int],
        *,
        sample_keys: Optional[Iterable[object]],
        device: torch.device,
        dtype: torch.dtype,
        scale: float,
        offset: float,
    ) -> torch.Tensor:
        batch_size, num_tokens, dim = shape
        base = torch.arange(num_tokens * dim, device=device, dtype=torch.float32).view(1, num_tokens, dim)
        if sample_keys is None:
            offsets = torch.arange(batch_size, device=device, dtype=torch.float32).view(batch_size, 1, 1) * 0.173
        else:
            offsets = torch.tensor(
                [_stable_offset(key, self.seed) for key in sample_keys],
                device=device,
                dtype=torch.float32,
            ).view(batch_size, 1, 1)
        values = torch.sin(base * 0.013 + offsets + offset + self.seed * 0.019) * scale
        return values.to(dtype=dtype)

    def generate(
        self,
        batch_size: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        include_targets: bool = False,
        sample_keys: Optional[Iterable[object]] = None,
    ) -> Dict[str, torch.Tensor]:
        warnings.warn(DUMMY_EXPERT_WARNING, RuntimeWarning)
        device = torch.device(device)
        sample_keys_list = list(sample_keys) if sample_keys is not None else None
        if sample_keys_list is not None and len(sample_keys_list) != batch_size:
            raise ValueError(f"sample_keys length {len(sample_keys_list)} does not match batch_size={batch_size}.")

        features: Dict[str, torch.Tensor] = {}
        if self.use_jepa:
            jepa_tokens = self._tokens(
                (batch_size, self.num_jepa_tokens, self.jepa_dim),
                sample_keys=sample_keys_list,
                device=device,
                dtype=dtype,
                scale=0.2,
                offset=0.3,
            )
            features["jepa_context_tokens"] = jepa_tokens
            if include_targets:
                features["jepa_target_tokens"] = jepa_tokens + self._tokens(
                    (batch_size, self.num_jepa_tokens, self.jepa_dim),
                    sample_keys=sample_keys_list,
                    device=device,
                    dtype=dtype,
                    scale=0.02,
                    offset=1.3,
                )

        if self.use_vggt:
            vggt_tokens = self._tokens(
                (batch_size, self.num_vggt_tokens, self.vggt_dim),
                sample_keys=sample_keys_list,
                device=device,
                dtype=dtype,
                scale=0.15,
                offset=0.7,
            )
            features["vggt_context_tokens"] = vggt_tokens
            features["vggt_geometry_tokens"] = vggt_tokens
            features["vggt_geometry_mode_code"] = torch.tensor(1, device=device, dtype=torch.int64)
            if include_targets:
                features["vggt_target_tokens"] = vggt_tokens + self._tokens(
                    (batch_size, self.num_vggt_tokens, self.vggt_dim),
                    sample_keys=sample_keys_list,
                    device=device,
                    dtype=dtype,
                    scale=0.02,
                    offset=1.7,
                )
                features["vggt_geometry_target_tokens"] = features["vggt_target_tokens"]

        return features

    def generate_single(
        self,
        *,
        sample_key: object,
        include_targets: bool = False,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Dict[str, torch.Tensor]:
        return {
            key: value.squeeze(0)
            for key, value in self.generate(
                1,
                device=device,
                dtype=dtype,
                include_targets=include_targets,
                sample_keys=[sample_key],
            ).items()
        }


def build_dummy_expert_backend(
    *,
    num_jepa_tokens: int,
    num_vggt_tokens: int,
    jepa_dim: int,
    vggt_dim: int,
    use_jepa: bool = True,
    use_vggt: bool = True,
    seed: int = 0,
) -> DummyExpertBackend:
    return DummyExpertBackend(
        num_jepa_tokens=num_jepa_tokens,
        num_vggt_tokens=num_vggt_tokens,
        jepa_dim=jepa_dim,
        vggt_dim=vggt_dim,
        use_jepa=use_jepa,
        use_vggt=use_vggt,
        seed=seed,
    )
