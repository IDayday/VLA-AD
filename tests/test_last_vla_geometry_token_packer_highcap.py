from __future__ import annotations

import torch

from navsim.agents.recogdrive.geometry_tokenizer import GeometryTokenPacker


def test_geometry_token_packer_highcap_grid_shape_and_metadata():
    torch.manual_seed(512)
    depth = torch.linspace(0.0, 1.0, steps=48 * 64).view(48, 64)
    point_map = torch.randn(48, 64, 3)
    camera = torch.eye(4)
    packer = GeometryTokenPacker(num_tokens=192, output_dim=512, grid=(12, 16), seed=77)

    tokens_a = packer.pack(depth=depth, point_map=point_map, camera=camera)
    metadata = packer.metadata()
    tokens_b = GeometryTokenPacker(num_tokens=192, output_dim=512, grid=(12, 16), seed=77).pack(
        depth=depth,
        point_map=point_map,
        camera=camera,
    )

    assert tokens_a.shape == (192, 512)
    assert torch.allclose(tokens_a, tokens_b)
    assert metadata["grid"] == (12, 16)
    assert metadata["output_dim"] == 512
    assert metadata["num_tokens"] == 192
    assert metadata["seed"] == 77
    assert metadata["source_fields_used"] == ("depth", "point_map", "camera")
