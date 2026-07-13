from __future__ import annotations

import torch
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.last_vla_cot_planning import GEOMETRY_MODE_TO_CODE, LastVLACoTConfig, LastVLACoTTransformer


def _small_model() -> LastVLACoTTransformer:
    return LastVLACoTTransformer(
        LastVLACoTConfig(
            planner_dim=64,
            vlm_dim=64,
            jepa_dim=32,
            vggt_dim=48,
            hidden_dim=128,
            cot_num_tokens=8,
            geometry_tokens=8,
            dynamic_tokens=8,
            ego_tokens=2,
            risk_tokens=0,
            use_cot_risk_head=False,
            raw_vlm_context_to_dit=True,
            cot_bottleneck_mode=False,
            geometry_teacher_dim=16,
            geometry_grid_rows=2,
            geometry_grid_cols=4,
            require_full_geometry=True,
            allow_patch_geometry_fallback=False,
        )
    )


def _input(batch: int = 1, zero_geometry: bool = False) -> BatchFeature:
    return BatchFeature(
        data={
            "status_feature": torch.randn(batch, 8),
            "history_trajectory": torch.randn(batch, 4, 3),
            "high_command_one_hot": torch.eye(3)[torch.arange(batch) % 3].float(),
            "jepa_context_tokens": torch.randn(batch, 8, 32),
            "vggt_geometry_tokens": torch.randn(batch, 8, 16),
            "vggt_geometry_mode_code": torch.full((batch,), GEOMETRY_MODE_TO_CODE["full_geometry"]),
            "last_vla_corrupt_zero_geometry_cot": zero_geometry,
        }
    )


def test_geometry_dynamic_are_parallel_not_sequentially_zeroed():
    torch.manual_seed(11)
    model = _small_model().eval()
    vlm = torch.randn(1, 5, 64)
    with torch.no_grad():
        normal = model(vlm, _input(zero_geometry=False), training=False)
        zero_geometry = model(vlm, _input(zero_geometry=True), training=False)

    assert normal.diagnostics["geometry_dynamic_parallel"].item() == 1.0
    assert zero_geometry.cot_geometry_tokens.abs().sum().item() == 0.0
    assert zero_geometry.cot_dynamic_tokens.abs().sum().item() > 0.0
    assert zero_geometry.raw_vlm_context_tokens.shape == normal.raw_vlm_context_tokens.shape
