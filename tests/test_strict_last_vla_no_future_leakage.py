from __future__ import annotations

import torch

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

_install_dependency_stubs()

from transformers.feature_extraction_utils import BatchFeature

from tests.test_strict_last_vla_stage2_context import make_strict_batch, make_strict_planner


def test_strict_last_vla_eval_ignores_teacher_targets():
    torch.manual_seed(13)
    planner = make_strict_planner()
    planner.eval()
    vl_features, data = make_strict_batch()
    context = {key: value for key, value in data.items() if key not in {"action", "jepa_target_tokens", "vggt_geometry_tokens"}}
    with_targets = dict(context)
    with_targets["jepa_target_tokens"] = torch.randn(2, 128, 1024)
    with_targets["vggt_geometry_tokens"] = torch.randn(2, 192, 512)
    init_actions = torch.zeros(2, 8, 3)

    with torch.no_grad():
        out_context = planner.get_action(vl_features, BatchFeature(data=context), init_actions=init_actions.clone(), deterministic=True)
        out_targets = planner.get_action(vl_features, BatchFeature(data=with_targets), init_actions=init_actions.clone(), deterministic=True)

    assert torch.allclose(out_context["pred_traj"], out_targets["pred_traj"], atol=1e-6, rtol=0.0)
