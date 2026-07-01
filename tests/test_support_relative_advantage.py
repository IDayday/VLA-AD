from __future__ import annotations

from pathlib import Path

import torch

from navsim.agents.recogdrive.pareto_support import DESCRIPTOR_NAMES
from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner


def _support_index(path: Path) -> None:
    support = torch.zeros(1, 3, 8, 3)
    support[0, 0, :, 0] = torch.linspace(0.0, 8.0, 8)
    support[0, 1, :, 0] = torch.linspace(0.0, 8.0, 8)
    support[0, 1, :, 1] = 3.0
    payload = {
        "version": 1,
        "tokens": ["tok-a"],
        "token_to_row": {"tok-a": 0},
        "support_trajectories": support,
        "support_mask": torch.tensor([[True, True, False]]),
        "support_weights": torch.tensor([[0.7, 0.3, 0.0]]),
        "support_scores": torch.tensor([[0.9, 0.85, 0.0]]),
        "descriptor_mean": torch.zeros(len(DESCRIPTOR_NAMES)),
        "descriptor_std": torch.ones(len(DESCRIPTOR_NAMES)),
    }
    torch.save(payload, path)


def _planner(index_path: Path):
    planner = ReCogDriveDiffusionPlanner.__new__(ReCogDriveDiffusionPlanner)
    planner.grpo_support_index_path = str(index_path)
    planner._grpo_support_index = None
    planner._grpo_support_index_path_loaded = None
    planner.grpo_support_free_distance = 1.5
    planner.grpo_support_rank_margin = 0.01
    planner.grpo_support_std_floor = 0.005
    planner.grpo_support_novel_margin = 0.01
    planner.grpo_support_novel_positive_cap = 0.2
    planner.grpo_support_intra_weight = 1.0
    planner.grpo_support_inter_weight = 0.15
    planner.grpo_support_inter_clip = 0.30
    planner.grpo_support_positive_only_inter = True
    planner.grpo_support_low_rank_group_weight = 0.25
    planner.core_pareto_dominated_positive_adv_cap = 0.0
    planner.core_pareto_positive_slow_fail_cap = 0.0
    return planner


def test_support_relative_advantage_updates_supported_scene_and_fallbacks_missing(tmp_path: Path):
    index_path = tmp_path / "support.pt"
    _support_index(index_path)
    planner = _planner(index_path)

    B, G = 2, 3
    trajs = torch.zeros(B, G, 8, 3)
    trajs[0, 0, :, 0] = torch.linspace(0.0, 8.0, 8)
    trajs[0, 1, :, 0] = torch.linspace(0.0, 8.2, 8)
    trajs[0, 2, :, 0] = torch.linspace(0.0, 8.0, 8)
    trajs[0, 2, :, 1] = 3.0
    trajs[1] = trajs[0]
    base_adv = torch.tensor([0.0, 0.0, 0.0, 5.0, 5.0, 5.0])
    base_group = torch.ones(B)
    score = torch.tensor([[0.80, 0.90, 0.95], [0.10, 0.20, 0.30]])
    valid = torch.ones(B, G, dtype=torch.bool)
    ep_ok = torch.ones(B, G, dtype=torch.bool)
    pareto = torch.ones(B, G, dtype=torch.bool)
    reference_margin = torch.zeros(B, G)
    ref_score = torch.tensor([0.70, 0.0])

    adv, group_weight, aux = planner._compute_support_relative_pareto_advantages(
        base_adv,
        base_group,
        score,
        trajs,
        valid & ep_ok,
        valid,
        ep_ok,
        pareto,
        reference_margin,
        ref_score,
        ["tok-a", "missing-token"],
    )

    assert not torch.allclose(adv[:3], base_adv[:3])
    assert torch.allclose(adv[3:], base_adv[3:])
    assert group_weight[0].item() == 1.0
    assert aux["support_missing_ratio"].item() == 0.5
    assert aux["rankable_support_bucket_count"].item() > 0.0
