from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DynamicTokenizerMetadata:
    output_tokens: int
    input_shape: tuple[int, ...]
    temporal_bins: int
    spatial_tokens_per_bin: int
    tokenizer_mode: str


class DynamicTokenPacker:
    """Deterministic high-cap tokenizer for dense V-JEPA/V-JEPA2 tokens."""

    def __init__(
        self,
        output_tokens: int = 128,
        temporal_bins: int = 8,
        spatial_tokens_per_bin: int = 16,
        *,
        teacher_dim: int = 1024,
        strict: bool = True,
    ) -> None:
        self.output_tokens = int(output_tokens)
        self.temporal_bins = int(temporal_bins)
        self.spatial_tokens_per_bin = int(spatial_tokens_per_bin)
        self.teacher_dim = int(teacher_dim)
        self.strict = bool(strict)
        if self.output_tokens <= 0:
            raise ValueError("output_tokens must be positive.")
        if self.teacher_dim <= 0:
            raise ValueError("teacher_dim must be positive.")
        if self.temporal_bins <= 0 or self.spatial_tokens_per_bin <= 0:
            raise ValueError("temporal_bins and spatial_tokens_per_bin must be positive.")
        if self.temporal_bins * self.spatial_tokens_per_bin != self.output_tokens:
            raise ValueError(
                "output_tokens must equal temporal_bins * spatial_tokens_per_bin "
                f"({self.temporal_bins} * {self.spatial_tokens_per_bin})."
            )
        self._last_metadata: Optional[DynamicTokenizerMetadata] = None

    @staticmethod
    def _reshape_sequence(tokens: torch.Tensor, temporal_size: int, spatial_hw: Optional[tuple[int, int]]) -> torch.Tensor:
        batch, num_tokens, dim = tokens.shape
        if spatial_hw is not None:
            spatial = int(spatial_hw[0]) * int(spatial_hw[1])
            if spatial > 0 and num_tokens % spatial == 0:
                temporal = num_tokens // spatial
                return tokens.view(batch, temporal, int(spatial_hw[0]), int(spatial_hw[1]), dim)
        if num_tokens % temporal_size == 0:
            spatial_tokens = num_tokens // temporal_size
            side = int(spatial_tokens**0.5)
            if side * side == spatial_tokens:
                return tokens.view(batch, temporal_size, side, side, dim)
            return tokens.view(batch, temporal_size, spatial_tokens, 1, dim)
        return tokens

    def _metadata(self, input_shape: tuple[int, ...], mode: str) -> Dict[str, Any]:
        meta = DynamicTokenizerMetadata(
            output_tokens=self.output_tokens,
            input_shape=input_shape,
            temporal_bins=self.temporal_bins,
            spatial_tokens_per_bin=self.spatial_tokens_per_bin,
            tokenizer_mode=mode,
        )
        self._last_metadata = meta
        return asdict(meta)

    def metadata(self) -> Dict[str, Any]:
        if self._last_metadata is None:
            return {
                "output_tokens": self.output_tokens,
                "input_shape": (),
                "temporal_bins": self.temporal_bins,
                "spatial_tokens_per_bin": self.spatial_tokens_per_bin,
                "tokenizer_mode": "uninitialized",
            }
        return asdict(self._last_metadata)

    def pack(
        self,
        dense_tokens: torch.Tensor,
        *,
        spatial_hw: Optional[tuple[int, int]] = None,
        temporal_size: int = 8,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        if not isinstance(dense_tokens, torch.Tensor):
            raise TypeError(f"dense_tokens must be a torch.Tensor, got {type(dense_tokens).__name__}.")
        if dense_tokens.shape[-1] != self.teacher_dim:
            raise ValueError(f"JEPA token dim {dense_tokens.shape[-1]} != expected {self.teacher_dim}.")
        input_shape = tuple(int(item) for item in dense_tokens.shape)
        if dense_tokens.ndim == 2:
            dense_tokens = dense_tokens.unsqueeze(0)
        if dense_tokens.ndim == 3 and dense_tokens.shape[1] <= 12 and self.output_tokens > dense_tokens.shape[1]:
            if self.strict:
                raise ValueError(
                    "High-cap JEPA requires dense or sufficiently many JEPA tokens; old 12-token cache is insufficient."
                )
        if dense_tokens.ndim == 3:
            reshaped = self._reshape_sequence(dense_tokens.float(), temporal_size, spatial_hw)
            if reshaped.ndim == 5:
                x = reshaped.permute(0, 4, 1, 2, 3).contiguous()
                pooled = F.adaptive_avg_pool3d(x, (self.temporal_bins, 1, self.spatial_tokens_per_bin))
                output = pooled.permute(0, 2, 3, 4, 1).reshape(reshaped.shape[0], self.output_tokens, self.teacher_dim)
                metadata = self._metadata(input_shape, "temporal_spatial_pool")
            else:
                x = dense_tokens.float().transpose(1, 2)
                output = F.adaptive_avg_pool1d(x, self.output_tokens).transpose(1, 2)
                metadata = self._metadata(input_shape, "dense_adaptive_pool")
        elif dense_tokens.ndim == 4:
            batch, temporal, spatial, dim = dense_tokens.shape
            side = int(spatial**0.5)
            if side * side == spatial:
                dense_tokens = dense_tokens.view(batch, temporal, side, side, dim)
                x = dense_tokens.float().permute(0, 4, 1, 2, 3).contiguous()
                pooled = F.adaptive_avg_pool3d(x, (self.temporal_bins, 1, self.spatial_tokens_per_bin))
                output = pooled.permute(0, 2, 3, 4, 1).reshape(batch, self.output_tokens, self.teacher_dim)
                metadata = self._metadata(input_shape, "temporal_spatial_pool")
            else:
                output = F.adaptive_avg_pool1d(dense_tokens.reshape(batch, -1, dim).float().transpose(1, 2), self.output_tokens).transpose(1, 2)
                metadata = self._metadata(input_shape, "dense_adaptive_pool")
        elif dense_tokens.ndim == 5:
            x = dense_tokens.float().permute(0, 4, 1, 2, 3).contiguous()
            pooled = F.adaptive_avg_pool3d(x, (self.temporal_bins, 1, self.spatial_tokens_per_bin))
            output = pooled.permute(0, 2, 3, 4, 1).reshape(dense_tokens.shape[0], self.output_tokens, self.teacher_dim)
            metadata = self._metadata(input_shape, "temporal_spatial_pool")
        else:
            raise ValueError(
                "dense_tokens must have shape [B,N,D], [B,T,N,D], or [B,T,H,W,D], "
                f"got {input_shape}."
            )
        if tuple(output.shape[1:]) != (self.output_tokens, self.teacher_dim):
            raise RuntimeError(f"Dynamic tokenizer produced invalid shape {tuple(output.shape)}.")
        if not torch.isfinite(output.float()).all():
            raise ValueError("Dynamic tokenizer produced non-finite tokens.")
        return output.contiguous(), metadata
