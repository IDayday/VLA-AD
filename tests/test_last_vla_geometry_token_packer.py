from __future__ import annotations

import torch

from navsim.agents.recogdrive.geometry_tokenizer import GeometryTokenPacker


def test_geometry_token_packer_is_deterministic_finite_and_shaped():
    packer = GeometryTokenPacker(num_tokens=12, output_dim=512, grid=(3, 4), seed=2026)
    depth = torch.linspace(0, 1, 24 * 32).view(24, 32)
    point_map = torch.stack([depth, depth + 1, depth + 2], dim=-1)
    camera = torch.eye(4)

    first = packer.pack(depth=depth, point_map=point_map, camera=camera)
    second = packer.pack(depth=depth, point_map=point_map, camera=camera)

    assert first.shape == (12, 512)
    assert torch.isfinite(first).all()
    assert torch.allclose(first, second)
    assert packer.metadata()["output_dim"] == 512
