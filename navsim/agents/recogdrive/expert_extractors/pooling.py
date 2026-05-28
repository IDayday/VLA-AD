from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
import torch.nn.functional as F


def expand_four_frames_to_eight(frames: Sequence[object]) -> list[object]:
    if len(frames) != 4:
        raise ValueError(f"Expected exactly 4 frames, got {len(frames)}.")
    expanded: list[object] = []
    for frame in frames:
        expanded.extend([frame, frame])
    return expanded


def _infer_square_hw(num_tokens: int) -> int:
    side = int(num_tokens**0.5)
    if side * side != num_tokens:
        raise ValueError(f"Cannot infer square grid from {num_tokens} tokens.")
    return side


def _infer_factor_hw(num_tokens: int) -> tuple[int, int]:
    if num_tokens <= 0:
        raise ValueError(f"Cannot infer grid from {num_tokens} tokens.")
    best_h = 1
    for h in range(1, int(num_tokens**0.5) + 1):
        if num_tokens % h == 0:
            best_h = h
    return best_h, num_tokens // best_h


def _reshape_vjepa_sequence(
    dense_tokens: torch.Tensor,
    *,
    teacher_dim: int,
    temporal_size: int,
    spatial_hw: Optional[tuple[int, int]],
) -> torch.Tensor:
    batch, num_tokens, dim = dense_tokens.shape
    if dim != teacher_dim:
        raise ValueError(f"V-JEPA2 dim {dim} != {teacher_dim}.")
    if spatial_hw is not None:
        h, w = spatial_hw
        spatial_tokens = h * w
        if spatial_tokens > 0 and num_tokens % spatial_tokens == 0:
            temporal = num_tokens // spatial_tokens
            return dense_tokens.view(batch, temporal, h, w, dim)
    if num_tokens % temporal_size != 0:
        raise ValueError(f"Cannot split {num_tokens} V-JEPA2 tokens over T={temporal_size}.")
    spatial_tokens = num_tokens // temporal_size
    try:
        side = _infer_square_hw(spatial_tokens)
        h = w = side
    except ValueError:
        h, w = _infer_factor_hw(spatial_tokens)
    return dense_tokens.view(batch, temporal_size, h, w, dim)


def pool_vjepa2_tokens(
    dense_tokens: torch.Tensor,
    *,
    teacher_dim: int = 1024,
    output_size: tuple[int, int, int] = (4, 1, 3),
    temporal_size: int = 8,
    spatial_hw: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    """Pools V-JEPA2 dense tokens to [B, 12, 1024] using adaptive_avg_pool3d."""
    if dense_tokens.ndim == 3:
        dense_tokens = _reshape_vjepa_sequence(
            dense_tokens, teacher_dim=teacher_dim, temporal_size=temporal_size, spatial_hw=spatial_hw
        )
    elif dense_tokens.ndim == 4:
        if dense_tokens.shape[-1] != teacher_dim:
            raise ValueError(f"V-JEPA2 dim {dense_tokens.shape[-1]} != {teacher_dim}.")
        # Common real-model shape: [B, T, N, D]. Convert N to HxW.
        if spatial_hw is not None and dense_tokens.shape[2] == spatial_hw[0] * spatial_hw[1]:
            dense_tokens = dense_tokens.view(
                dense_tokens.shape[0], dense_tokens.shape[1], spatial_hw[0], spatial_hw[1], teacher_dim
            )
        elif dense_tokens.shape[2] != teacher_dim:
            batch, temporal, spatial_tokens, dim = dense_tokens.shape
            try:
                side = _infer_square_hw(spatial_tokens)
                h = w = side
            except ValueError:
                h, w = _infer_factor_hw(spatial_tokens)
            dense_tokens = dense_tokens.view(batch, temporal, h, w, dim)
        else:
            # Shape [T, H, W, D].
            dense_tokens = dense_tokens.unsqueeze(0)
    elif dense_tokens.ndim == 5:
        if dense_tokens.shape[-1] != teacher_dim:
            raise ValueError(f"V-JEPA2 dim {dense_tokens.shape[-1]} != {teacher_dim}.")
    else:
        raise ValueError(
            "V-JEPA2 dense tokens must have shape [B,N,D], [B,T,N,D], [T,H,W,D], or [B,T,H,W,D], "
            f"got {tuple(dense_tokens.shape)}."
        )

    x = dense_tokens.permute(0, 4, 1, 2, 3).contiguous()
    pooled = F.adaptive_avg_pool3d(x.float(), output_size=output_size)
    pooled = pooled.permute(0, 2, 3, 4, 1).reshape(dense_tokens.shape[0], -1, teacher_dim)
    expected_tokens = output_size[0] * output_size[1] * output_size[2]
    if pooled.shape[1:] != (expected_tokens, teacher_dim):
        raise RuntimeError(f"Pooled V-JEPA2 shape {tuple(pooled.shape)} is invalid.")
    return pooled


def pool_vggt_tokens(
    patch_tokens: torch.Tensor,
    *,
    teacher_dim: int = 2048,
    grid_size: tuple[int, int] = (37, 37),
    output_size: tuple[int, int] = (3, 4),
) -> torch.Tensor:
    """Pools VGGT patch tokens to [B, 12, 2048] using adaptive_avg_pool2d."""
    if patch_tokens.ndim == 2:
        patch_tokens = patch_tokens.unsqueeze(0)
    if patch_tokens.ndim == 3:
        batch, num_tokens, dim = patch_tokens.shape
        if dim != teacher_dim:
            raise ValueError(f"VGGT dim {dim} != {teacher_dim}.")
        expected = grid_size[0] * grid_size[1]
        if num_tokens != expected:
            raise ValueError(f"VGGT patch token count {num_tokens} != {expected} for grid {grid_size}.")
        patch_tokens = patch_tokens.view(batch, grid_size[0], grid_size[1], dim)
    elif patch_tokens.ndim == 4:
        if patch_tokens.shape[-1] != teacher_dim:
            raise ValueError(f"VGGT dim {patch_tokens.shape[-1]} != {teacher_dim}.")
    else:
        raise ValueError(f"VGGT patch tokens must have shape [B,N,D] or [B,H,W,D], got {tuple(patch_tokens.shape)}.")

    x = patch_tokens.permute(0, 3, 1, 2).contiguous()
    pooled = F.adaptive_avg_pool2d(x.float(), output_size=output_size)
    pooled = pooled.permute(0, 2, 3, 1).reshape(patch_tokens.shape[0], -1, teacher_dim)
    expected_tokens = output_size[0] * output_size[1]
    if pooled.shape[1:] != (expected_tokens, teacher_dim):
        raise RuntimeError(f"Pooled VGGT shape {tuple(pooled.shape)} is invalid.")
    return pooled
