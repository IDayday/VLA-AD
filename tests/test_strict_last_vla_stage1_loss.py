from __future__ import annotations

import torch

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

_install_dependency_stubs()

from transformers.feature_extraction_utils import BatchFeature

from tests.test_strict_last_vla_stage2_context import make_strict_planner, make_strict_batch


def test_strict_last_vla_stage1_loss_includes_wm_geo_plan():
    torch.manual_seed(12)
    planner = make_strict_planner(stage="stage1_latent_alignment", diffusion_loss_weight=0.0)
    planner.train()
    vl_features, action_input = make_strict_batch(batch=1)
    out = planner(vl_features, BatchFeature(data={**action_input, "_allow_target_tokens_for_loss": True}))

    assert out["diffusion_loss"] == 0
    assert out["last_vla_dynamic_loss"] > 0
    assert out["last_vla_geometry_loss"] > 0
    assert out["last_vla_coarse_loss"] > 0
    assert out["last_vla_h_dyn_norm"] > 0
