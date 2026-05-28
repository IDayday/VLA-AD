from __future__ import annotations

import torch

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

_install_dependency_stubs()

from tests.test_expert_fusion_shapes import _planner

EXPERT_MARKERS = (
    "jepa_projector", "vggt_projector", "jepa_adapter", "vggt_adapter",
    "jepa_alignment_head", "vggt_alignment_head", "jepa_type_embedding", "vggt_type_embedding",
    "z_jepa_type_embedding", "z_vggt_type_embedding", "jepa_gate", "vggt_gate", "branch_logits",
)


def test_base_il_like_state_loads_with_missing_expert_keys():
    baseline = _planner(False)
    expert = _planner(True)
    baseline_state = baseline.state_dict()
    expert_state = expert.state_dict()
    filtered = {key: value for key, value in baseline_state.items() if key in expert_state and tuple(value.shape) == tuple(expert_state[key].shape)}
    incompatible = expert.load_state_dict(filtered, strict=False)
    missing_expert = [key for key in incompatible.missing_keys if any(marker in key for marker in EXPERT_MARKERS)]
    missing_other = [key for key in incompatible.missing_keys if key not in missing_expert]
    assert missing_expert
    assert not missing_other
