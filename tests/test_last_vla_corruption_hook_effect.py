from __future__ import annotations

import torch
from transformers.feature_extraction_utils import BatchFeature

from tests.test_last_vla_helpers import make_last_vla_batch, make_last_vla_planner


def _context(planner, vl_features, action_input):
    return planner._prepare_dit_context(
        vl_features,
        action_input,
        training=False,
        noisy_actions=torch.zeros(vl_features.shape[0], 8, 3),
        diffusion_timestep=torch.zeros(vl_features.shape[0], dtype=torch.long),
        allow_target_tokens=False,
    )


def test_zero_all_cot_zeros_condition_but_keeps_raw_context():
    torch.manual_seed(11)
    planner = make_last_vla_planner()
    planner.eval()
    vl_features, action_input = make_last_vla_batch(include_targets=False)
    base_data = dict(action_input)

    normal = _context(planner, vl_features, BatchFeature(data=base_data))
    zero = _context(
        planner,
        vl_features,
        BatchFeature(data={**base_data, "last_vla_corrupt_zero_all_cot": True}),
    )

    assert torch.allclose(zero["last_vla_output"].cot_tokens, torch.zeros_like(zero["last_vla_output"].cot_tokens))
    assert torch.allclose(zero["last_vla_output"].cot_condition_tokens, torch.zeros_like(zero["last_vla_output"].cot_condition_tokens))
    assert torch.allclose(normal["context_tokens"], zero["context_tokens"])


def test_zero_coarse_prior_affects_residual_prior_path():
    torch.manual_seed(12)
    planner = make_last_vla_planner()
    planner.eval()
    vl_features, action_input = make_last_vla_batch(include_targets=False)
    base_data = dict(action_input)

    normal = _context(planner, vl_features, BatchFeature(data=base_data))
    zero = _context(
        planner,
        vl_features,
        BatchFeature(data={**base_data, "last_vla_zero_coarse_prior": True}),
    )

    assert torch.allclose(
        zero["last_vla_output"].coarse_traj_norm,
        torch.zeros_like(zero["last_vla_output"].coarse_traj_norm),
    )
    assert not torch.allclose(normal["last_vla_output"].coarse_traj_norm, zero["last_vla_output"].coarse_traj_norm)
