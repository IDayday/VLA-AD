from __future__ import annotations

import warnings
import torch

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

_install_dependency_stubs()

from transformers.feature_extraction_utils import BatchFeature
from tests.test_expert_fusion_shapes import _planner


def test_get_action_ignores_nan_target_tokens():
    planner = _planner(True)
    planner.eval()
    b = 2
    vl = torch.randn(b, 7, 1536)
    action_input = BatchFeature(data={
        "his_traj": torch.randn(b, 12),
        "status_feature": torch.randn(b, 8),
        "jepa_context_tokens": torch.randn(b, 12, 1024),
        "vggt_context_tokens": torch.randn(b, 12, 2048),
        "jepa_target_tokens": torch.full((b, 12, 1024), float("nan")),
        "vggt_target_tokens": torch.full((b, 12, 2048), float("nan")),
    })
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.no_grad():
            out = planner.get_action(vl, action_input, init_actions=torch.zeros(b, 8, 3), deterministic=True)
    assert out["pred_traj"].shape == (b, 8, 3)
    assert torch.isfinite(out["pred_traj"]).all()
    assert any("train-only expert target keys" in str(item.message) for item in caught)
