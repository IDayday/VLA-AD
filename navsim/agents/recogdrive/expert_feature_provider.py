from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .expert_backends import DummyExpertBackend, build_dummy_expert_backend
from .expert_cache import load_sample, sample_cache_path, validate_sample_payload


class BaseExpertFeatureProvider:
    def get(self, *, sample_token: str, scene_token: Optional[str] = None, include_targets: bool = False) -> Dict[str, Any]:
        raise NotImplementedError


class DummyExpertFeatureProvider(BaseExpertFeatureProvider):
    def __init__(
        self,
        *,
        num_jepa_tokens: int = 12,
        num_vggt_tokens: int = 12,
        jepa_dim: int = 1024,
        vggt_dim: int = 2048,
        use_jepa: bool = True,
        use_vggt: bool = True,
        seed: int = 0,
    ) -> None:
        self.backend: DummyExpertBackend = build_dummy_expert_backend(
            num_jepa_tokens=num_jepa_tokens,
            num_vggt_tokens=num_vggt_tokens,
            jepa_dim=jepa_dim,
            vggt_dim=vggt_dim,
            use_jepa=use_jepa,
            use_vggt=use_vggt,
            seed=seed,
        )

    def get(self, *, sample_token: str, scene_token: Optional[str] = None, include_targets: bool = False) -> Dict[str, Any]:
        sample_key = f"{scene_token or ''}/{sample_token}"
        return self.backend.generate_single(sample_key=sample_key, include_targets=include_targets)


class ChunkExpertFeatureProvider(BaseExpertFeatureProvider):
    def __init__(self, cache_dir: str | Path, *, require_jepa: bool = True, require_vggt: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.require_jepa = require_jepa
        self.require_vggt = require_vggt

    def get(self, *, sample_token: str, scene_token: Optional[str] = None, include_targets: bool = False) -> Dict[str, Any]:
        path = sample_cache_path(self.cache_dir, sample_token=sample_token, scene_token=scene_token)
        payload = load_sample(path)
        return validate_sample_payload(
            payload,
            require_jepa=self.require_jepa,
            require_vggt=self.require_vggt,
            require_targets=include_targets,
        )


class DiskExpertFeatureProvider(ChunkExpertFeatureProvider):
    """Alias for larger on-disk caches with the same per-sample payload schema."""


class OnlineExpertFeatureProvider(BaseExpertFeatureProvider):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def get(self, *, sample_token: str, scene_token: Optional[str] = None, include_targets: bool = False) -> Dict[str, Any]:
        raise NotImplementedError(
            "OnlineExpertFeatureProvider is a placeholder. Build a chunk cache first, "
            "or wire V-JEPA2/VGGT extraction into the NAVSIM data path explicitly."
        )


def build_expert_feature_provider(
    source: str,
    *,
    cache_dir: Optional[str | Path] = None,
    allow_dummy_cache: bool = False,
    **kwargs: Any,
) -> Optional[BaseExpertFeatureProvider]:
    source = (source or "none").lower()
    if source in {"none", ""}:
        return None
    if source == "dummy":
        if not allow_dummy_cache:
            raise RuntimeError("Dummy expert features require allow_dummy_cache=True.")
        return DummyExpertFeatureProvider(**kwargs)
    if source in {"chunk", "cache"}:
        if cache_dir is None:
            raise ValueError("chunk expert feature source requires cache_dir.")
        return ChunkExpertFeatureProvider(cache_dir, require_jepa=kwargs.get("use_jepa", True), require_vggt=kwargs.get("use_vggt", True))
    if source == "disk":
        if cache_dir is None:
            raise ValueError("disk expert feature source requires cache_dir.")
        return DiskExpertFeatureProvider(cache_dir, require_jepa=kwargs.get("use_jepa", True), require_vggt=kwargs.get("use_vggt", True))
    if source in {"online", "real"}:
        return OnlineExpertFeatureProvider(**kwargs)
    raise ValueError(f"Unknown expert feature source: {source!r}")


def move_expert_tensors(payload: Dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor) and key.endswith("tokens"):
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value
    return moved
