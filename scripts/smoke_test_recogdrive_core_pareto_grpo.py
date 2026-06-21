#!/usr/bin/env python3
"""CPU smoke test for ReCogDrive Stage3 Core-Pareto GRPO v2 helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.recogdrive_diffusion_planner import (  # noqa: E402
    GRPOConfig,
    ReCogDriveDiffusionPlanner,
)


def _make_planner() -> ReCogDriveDiffusionPlanner:
    planner = object.__new__(ReCogDriveDiffusionPlanner)
    cfg = GRPOConfig(
        use_core_pareto_grpo=True,
        core_pareto_use_ddc_guard=True,
        core_pareto_ddc_drop_tolerance=0.01,
        core_pareto_ddc_min_absolute=0.95,
        core_pareto_use_ep_floor=True,
        core_pareto_ep_floor_tolerance=0.02,
        core_pareto_use_ttc_tradeoff_penalty=True,
        core_pareto_tradeoff_tolerance=0.01,
        core_pareto_tradeoff_penalty_weight=0.2,
        core_pareto_use_pareto_front=True,
        core_pareto_pareto_front_bonus=0.2,
        core_pareto_dominated_positive_adv_cap=0.0,
        core_pareto_all_slow_group_weight=0.25,
        core_pareto_advantage_clip_abs=5.0,
    )
    ReCogDriveDiffusionPlanner._init_stage3_runtime(planner, cfg)
    planner.training = True
    return planner


def _components() -> dict[str, torch.Tensor]:
    # B=2, G=5. Row 0 is mixed. Row 1 is all valid by NC/DAC/DDC but all fail EP floor.
    ep = torch.tensor(
        [
            [0.82, 0.79, 0.82, 0.86, 0.80],
            [0.78, 0.79, 0.80, 0.79, 0.78],
        ],
        dtype=torch.float32,
    )
    ttc = torch.tensor(
        [
            [1.00, 1.00, 0.96, 0.85, 0.99],
            [0.98, 0.97, 0.96, 0.95, 0.94],
        ],
        dtype=torch.float32,
    )
    comfort = torch.ones_like(ep)
    nc = torch.ones_like(ep)
    dac = torch.ones_like(ep)
    ddc = torch.tensor(
        [
            [0.965, 0.965, 0.94, 0.98, 0.965],
            [0.97, 0.97, 0.97, 0.97, 0.97],
        ],
        dtype=torch.float32,
    )
    core = (5.0 * ep + 5.0 * ttc + 2.0 * comfort) / 12.0
    pdms = nc * dac * core
    return {
        "pdms": pdms,
        "no_at_fault_collisions": nc,
        "drivable_area_compliance": dac,
        "time_to_collision_within_bound": ttc,
        "ego_progress": ep,
        "history_comfort": comfort,
        "lane_keeping": torch.ones_like(ep),
        "driving_direction_compliance": ddc,
        "traffic_light_compliance": torch.ones_like(ep),
    }


def _ref() -> dict[str, torch.Tensor]:
    ref_ep = torch.tensor([0.83, 0.83], dtype=torch.float32)
    ref_ttc = torch.tensor([0.95, 0.95], dtype=torch.float32)
    ref_comfort = torch.ones(2)
    ref_core = (5.0 * ref_ep + 5.0 * ref_ttc + 2.0 * ref_comfort) / 12.0
    return {
        "gt_pdms": ref_core.clone(),
        "il_pdms": ref_core.clone(),
        "ref_pdms": ref_core.clone(),
        "gt_core": ref_core.clone(),
        "il_core": ref_core.clone(),
        "ref_core": ref_core.clone(),
        "ref_ep": ref_ep,
        "ref_ttc": ref_ttc,
        "ref_comfort": ref_comfort,
        "ref_ddc": torch.tensor([0.97, 0.97], dtype=torch.float32),
        "ref_nc": torch.ones(2),
        "ref_dac": torch.ones(2),
    }


def main() -> None:
    planner = _make_planner()
    components = _components()
    ref = _ref()
    B, G, H = 2, 5, 8
    trajs = torch.zeros(B, G, H, 3)
    trajs[:, :, :, 0] = torch.linspace(0.0, 4.0, H).view(1, 1, H)
    trajs[0, :, -1, 1] = torch.tensor([-0.6, 0.0, 0.7, 0.0, 0.0])

    core_one = planner._compute_pdms_core(
        {
            "ego_progress": torch.tensor([0.8]),
            "time_to_collision_within_bound": torch.tensor([1.0]),
            "history_comfort": torch.tensor([1.0]),
        }
    )
    assert torch.allclose(core_one, torch.tensor([(5 * 0.8 + 5 * 1.0 + 2 * 1.0) / 12.0]))

    adv, group_weight, aux = planner._compute_core_pareto_advantages(
        components["pdms"],
        components,
        trajs,
        ref,
    )
    adv_m = adv.reshape(B, G)
    valid = aux["core_pareto_valid_mask"]
    ep_floor_ok = aux["core_pareto_ep_floor_ok_mask"]

    # DDC guard: ref_ddc=0.97, tolerance=0.01. 0.965 passes, 0.94 fails.
    assert bool(valid[0, 0])
    assert not bool(valid[0, 2])

    # EP floor: ref_ep=0.83, tolerance=0.02. 0.82 passes, 0.79 fails.
    assert bool(ep_floor_ok[0, 0])
    assert not bool(ep_floor_ok[0, 1])

    # TTC tradeoff has a positive penalty for delta_ep=-0.04, delta_ttc=+0.01.
    assert aux["tradeoff_bad_mean"].item() > 0.0

    pareto_components = {
        "ego_progress": torch.tensor([[0.80, 0.82, 0.80]], dtype=torch.float32),
        "time_to_collision_within_bound": torch.tensor([[0.95, 0.95, 0.97]], dtype=torch.float32),
        "history_comfort": torch.ones(1, 3),
    }
    pareto = planner._compute_pareto_front_mask(
        pareto_components,
        torch.ones(1, 3, dtype=torch.bool),
        ("ego_progress", "time_to_collision_within_bound", "history_comfort"),
    )
    assert pareto.tolist() == [[False, True, True]]

    all_invalid_components = {k: v[:1].clone() for k, v in components.items()}
    all_invalid_components["no_at_fault_collisions"].zero_()
    all_invalid_adv, _, _ = planner._compute_core_pareto_advantages(
        all_invalid_components["pdms"],
        all_invalid_components,
        trajs[:1],
        {k: (v[:1] if isinstance(v, torch.Tensor) and v.shape[:1] == (2,) else v) for k, v in ref.items()},
    )
    assert torch.isfinite(all_invalid_adv).all()
    assert (all_invalid_adv <= 1e-6).all()

    # Row 1 is all valid but all EP-floor failures.
    assert group_weight[1].item() == 0.25
    assert (adv_m[1] <= float(planner.core_pareto_positive_slow_fail_cap) + 1e-6).all()
    assert aux["positive_advantage_slow_fail_ratio"].item() == 0.0

    planner.core_pareto_use_phenotype_bucket_grpo = True
    buckets = planner._compute_phenotype_buckets(trajs, components, ref["ref_ep"])
    assert buckets.shape == (B, G)
    assert torch.isfinite(buckets.float()).all()

    assert torch.isfinite(adv).all()
    assert torch.isfinite(group_weight).all()
    print("recogdrive_core_pareto_grpo_smoke_ok")


if __name__ == "__main__":
    main()
