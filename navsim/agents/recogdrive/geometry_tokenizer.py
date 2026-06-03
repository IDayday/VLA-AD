from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GeometryTokenizerMetadata:
    output_dim: int
    num_tokens: int
    grid: Tuple[int, int]
    fourier_bands: int
    seed: int
    projection_input_dim: int


class GeometryTokenPacker:
    """Deterministic offline tokenization for VGGT full geometry tensors."""

    def __init__(
        self,
        num_tokens: int = 12,
        output_dim: int = 512,
        grid: Tuple[int, int] = (3, 4),
        fourier_bands: int = 8,
        seed: int = 2026,
    ) -> None:
        self.num_tokens = int(num_tokens)
        self.output_dim = int(output_dim)
        self.grid = (int(grid[0]), int(grid[1]))
        self.fourier_bands = int(fourier_bands)
        self.seed = int(seed)
        if self.num_tokens != self.grid[0] * self.grid[1]:
            raise ValueError(f"num_tokens={self.num_tokens} must equal grid cells {self.grid[0] * self.grid[1]}.")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive.")

    @staticmethod
    def _as_hw(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        value = tensor.detach().float().cpu()
        while value.ndim > 3:
            value = value[0]
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.ndim == 3 and value.shape[-1] == 1:
            value = value[..., 0]
        if value.ndim != 2:
            return None
        return value

    @staticmethod
    def _as_hwc(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        value = tensor.detach().float().cpu()
        while value.ndim > 4:
            value = value[0]
        if value.ndim == 4:
            value = value[0]
        if value.ndim == 3 and value.shape[0] in (2, 3, 4) and value.shape[-1] not in (2, 3, 4):
            value = value.permute(1, 2, 0).contiguous()
        if value.ndim != 3:
            return None
        return value

    @staticmethod
    def _safe_stats(values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return torch.zeros(5, dtype=torch.float32)
        finite = torch.isfinite(values)
        valid_ratio = finite.float().mean()
        if not bool(finite.any()):
            return torch.tensor([0.0, 0.0, 0.0, 0.0, float(valid_ratio.item())], dtype=torch.float32)
        v = values[finite].float()
        return torch.stack([v.mean(), v.std(unbiased=False), v.min(), v.max(), valid_ratio])

    @staticmethod
    def _gradient_stats(depth: Optional[torch.Tensor], rows: slice, cols: slice) -> torch.Tensor:
        if depth is None:
            return torch.zeros(4, dtype=torch.float32)
        patch = depth[rows, cols]
        if patch.numel() <= 1:
            return torch.zeros(4, dtype=torch.float32)
        gy = patch[1:, :] - patch[:-1, :] if patch.shape[0] > 1 else patch.new_zeros(0)
        gx = patch[:, 1:] - patch[:, :-1] if patch.shape[1] > 1 else patch.new_zeros(0)
        gy_stats = GeometryTokenPacker._safe_stats(gy)[0:2] if gy.numel() else torch.zeros(2)
        gx_stats = GeometryTokenPacker._safe_stats(gx)[0:2] if gx.numel() else torch.zeros(2)
        return torch.cat([gx_stats, gy_stats], dim=0).float()

    def _cell_slices(self, height: int, width: int, index: int) -> tuple[slice, slice]:
        row = index // self.grid[1]
        col = index % self.grid[1]
        r0 = int(round(row * height / self.grid[0]))
        r1 = int(round((row + 1) * height / self.grid[0]))
        c0 = int(round(col * width / self.grid[1]))
        c1 = int(round((col + 1) * width / self.grid[1]))
        return slice(max(r0, 0), max(r1, r0 + 1)), slice(max(c0, 0), max(c1, c0 + 1))

    def _raw_stats(
        self,
        depth: Optional[torch.Tensor],
        point_map: Optional[torch.Tensor],
        camera: Optional[torch.Tensor],
        tracks: Optional[torch.Tensor],
    ) -> torch.Tensor:
        height = width = 1
        if depth is not None:
            height, width = int(depth.shape[0]), int(depth.shape[1])
        elif point_map is not None:
            height, width = int(point_map.shape[0]), int(point_map.shape[1])

        camera_flat = camera.detach().float().cpu().view(-1) if isinstance(camera, torch.Tensor) else torch.zeros(0)
        camera_summary = self._safe_stats(camera_flat)[:4]
        tracks_flat = tracks.detach().float().cpu().view(-1) if isinstance(tracks, torch.Tensor) else torch.zeros(0)
        tracks_summary = self._safe_stats(tracks_flat)[:4]
        xy_centers = []
        rows = []
        for idx in range(self.num_tokens):
            row_slice, col_slice = self._cell_slices(height, width, idx)
            stats = []
            if depth is not None:
                stats.append(self._safe_stats(depth[row_slice, col_slice]))
                stats.append(self._gradient_stats(depth, row_slice, col_slice))
            else:
                stats.extend([torch.zeros(5), torch.zeros(4)])
            if point_map is not None:
                point_patch = point_map[row_slice, col_slice]
                channels = []
                for channel in range(min(3, point_patch.shape[-1])):
                    channels.append(self._safe_stats(point_patch[..., channel])[:2])
                while len(channels) < 3:
                    channels.append(torch.zeros(2))
                stats.append(torch.cat(channels, dim=0))
            else:
                stats.append(torch.zeros(6))
            stats.extend([camera_summary, tracks_summary])
            y = ((row_slice.start + row_slice.stop) * 0.5 / max(height, 1)) * 2.0 - 1.0
            x = ((col_slice.start + col_slice.stop) * 0.5 / max(width, 1)) * 2.0 - 1.0
            xy_centers.append(torch.tensor([x, y], dtype=torch.float32))
            rows.append(torch.cat([*stats, xy_centers[-1]], dim=0).float())
        return torch.stack(rows, dim=0)

    def _fourier(self, stats: torch.Tensor) -> torch.Tensor:
        if self.fourier_bands <= 0:
            return stats
        bands = torch.arange(self.fourier_bands, dtype=torch.float32).view(1, 1, -1)
        scales = (2.0 ** bands) * torch.pi
        expanded = stats.unsqueeze(-1) * scales
        return torch.cat([stats, torch.sin(expanded).flatten(1), torch.cos(expanded).flatten(1)], dim=1)

    def _projection(self, input_dim: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(self.seed + input_dim * 97 + self.output_dim)
        proj = torch.randn(input_dim, self.output_dim, generator=generator, dtype=torch.float32)
        return proj / max(float(input_dim), 1.0) ** 0.5

    def pack(
        self,
        *,
        depth: Optional[torch.Tensor] = None,
        point_map: Optional[torch.Tensor] = None,
        camera: Optional[torch.Tensor] = None,
        tracks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        depth_hw = self._as_hw(depth)
        point_hwc = self._as_hwc(point_map)
        raw = self._raw_stats(depth_hw, point_hwc, camera, tracks)
        features = self._fourier(raw)
        tokens = torch.tanh(features @ self._projection(features.shape[-1]))
        if tokens.shape != (self.num_tokens, self.output_dim):
            tokens = F.adaptive_avg_pool1d(tokens.T.unsqueeze(0), self.num_tokens).squeeze(0).T
            tokens = tokens[:, : self.output_dim]
        if not torch.isfinite(tokens).all():
            raise ValueError("GeometryTokenPacker produced non-finite tokens.")
        return tokens.contiguous().float()

    def metadata(self) -> Dict[str, Any]:
        stats_dim = 5 + 4 + 6 + 4 + 4 + 2
        projected_dim = stats_dim * (1 + 2 * max(self.fourier_bands, 0))
        return asdict(
            GeometryTokenizerMetadata(
                output_dim=self.output_dim,
                num_tokens=self.num_tokens,
                grid=self.grid,
                fourier_bands=self.fourier_bands,
                seed=self.seed,
                projection_input_dim=projected_dim,
            )
        )
