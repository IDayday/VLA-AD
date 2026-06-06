from __future__ import annotations

import torch
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.last_vla_cot_planning import GEOMETRY_MODE_TO_CODE, LastVLACoTConfig, LastVLACoTTransformer


def _model() -> LastVLACoTTransformer:
    return LastVLACoTTransformer(
        LastVLACoTConfig(
            planner_dim=384,
            vlm_dim=384,
            jepa_dim=1024,
            vggt_dim=2048,
            hidden_dim=384,
            cot_num_tokens=192,
            cot_num_steps=5,
            geometry_tokens=192,
            dynamic_tokens=128,
            ego_tokens=8,
            risk_tokens=0,
            use_cot_risk_head=False,
            raw_vlm_context_to_dit=True,
            cot_bottleneck_mode=False,
            geometry_teacher_dim=512,
            geometry_grid_rows=12,
            geometry_grid_cols=16,
            require_full_geometry=True,
            allow_patch_geometry_fallback=False,
        )
    )


def _batch(batch: int = 2) -> BatchFeature:
    return BatchFeature(
        data={
            "status_feature": torch.randn(batch, 8),
            "high_command_one_hot": torch.eye(3)[torch.arange(batch) % 3].float(),
            "history_trajectory": torch.randn(batch, 4, 3),
            "jepa_context_tokens": torch.randn(batch, 128, 1024),
            "jepa_target_tokens": torch.randn(batch, 128, 1024),
            "vggt_geometry_tokens": torch.randn(batch, 192, 512),
            "vggt_geometry_mode_code": torch.full((batch,), GEOMETRY_MODE_TO_CODE["full_geometry"]),
        }
    )


def test_decoupled_cot_shapes_no_summary():
    torch.manual_seed(7)
    model = _model()
    batch = 2
    out = model(
        torch.randn(batch, 11, 384),
        _batch(batch),
        training=True,
        target_action_norm=torch.randn(batch, 8, 3),
        noisy_action_norm=torch.randn(batch, 8, 3),
        diffusion_timestep=torch.tensor([1, 2]),
        allow_target_tokens=True,
    )

    assert out.cot_scene_tokens.shape == (batch, 192, 384)
    assert out.cot_geometry_tokens.shape == (batch, 192, 384)
    assert out.cot_dynamic_tokens.shape == (batch, 192, 384)
    assert out.cot_condition_tokens.shape == (batch, 192, 384)
    assert out.raw_vlm_context_tokens.shape == (batch, 11, 384)
    assert out.planner_context_tokens.shape == (batch, 11, 384)
    assert not hasattr(out, "vlm_summary_tokens")
    assert out.diagnostics["raw_vlm_context_used"].item() == 1.0
    assert out.diagnostics["cot_condition_token_count"].item() == 192.0
