#!/usr/bin/env python
from __future__ import annotations

import tempfile
from pathlib import Path
import sys
import pickle

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.candidate_funnel import ExternalCandidateLoader, build_candidate_pool_for_scene
from navsim.agents.recogdrive.fs_norm import FSNormStats, FSNormTransform
from navsim.agents.recogdrive.offline_rl_buffer import load_elite_record, save_elite_record
from navsim.agents.recogdrive.pareto_support import (
    CandidateRecord,
    build_archive_record,
    pareto_front_mask,
    select_feasible_pareto_support,
)
from navsim.agents.recogdrive.pareto_vector_scorer import ParetoVectorScorer
from navsim.agents.recogdrive.pdas import compute_pdas_metrics
from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner
from navsim.agents.recogdrive.trajectory_feasibility import compute_feasibility_metrics


def _straight(batch: int = 1) -> torch.Tensor:
    x = torch.linspace(1.0, 8.0, 8)
    y = torch.zeros_like(x)
    h = torch.zeros_like(x)
    return torch.stack([x, y, h], dim=-1).unsqueeze(0).repeat(batch, 1, 1)


def test_trajectory_feasibility() -> None:
    straight = _straight()
    metrics = compute_feasibility_metrics(straight, {})
    assert float(metrics.feas_cost.max()) < 1e-5
    reverse = straight.clone()
    reverse[:, -3:, 0] = torch.tensor([6.0, 5.5, 5.0])
    metrics = compute_feasibility_metrics(reverse, {})
    assert float(metrics.tail_reverse_violation.max()) > 0.0
    kink = straight.clone()
    kink[:, 1, 2] = 1.0
    metrics = compute_feasibility_metrics(kink, {})
    assert float(metrics.early_kink_violation.max()) > 0.0


def test_fs_norm_roundtrip() -> None:
    traj = _straight(3)
    transform = FSNormTransform(
        FSNormStats(mean=torch.zeros((8, 3)), std=torch.ones((8, 3)), clip=0.0)
    )
    decoded = transform.decode(transform.encode(traj))
    assert torch.allclose(decoded, traj, atol=1e-5)


def _components(pdms: float, ep: float, ttc: float, ddc: float, nc: float = 1.0, dac: float = 1.0):
    return {
        "pdms": pdms,
        "no_at_fault_collisions": nc,
        "drivable_area_compliance": dac,
        "time_to_collision_within_bound": ttc,
        "ego_progress": ep,
        "history_comfort": 1.0,
        "lane_keeping": 1.0,
        "driving_direction_compliance": ddc,
        "traffic_light_compliance": 1.0,
    }


def test_pareto_support_and_buffer_v3() -> None:
    values = np.array([[1.0, 0.5], [0.8, 0.8], [0.7, 0.4]], dtype=np.float32)
    front = pareto_front_mask(values, np.array([True, True, True]))
    assert front.tolist() == [True, True, False]
    traj = _straight().numpy()[0]
    candidates = [
        CandidateRecord(traj, "gt", "tok", _components(0.8, 0.7, 1.0, 1.0), 0.8, {"feas_cost": 0.0}, 0.8),
        CandidateRecord(traj + np.array([0.5, 0.0, 0.0], dtype=np.float32), "il", "tok", _components(0.82, 0.75, 1.0, 1.0), 0.82, {"feas_cost": 0.0}, 0.82),
        CandidateRecord(traj + np.array([1.0, 0.2, 0.0], dtype=np.float32), "external", "tok", _components(0.85, 0.9, 0.98, 1.0), 0.85, {"feas_cost": 0.01}, 0.85),
    ]
    selected = select_feasible_pareto_support(candidates, candidates[0].components, {"support_top_m": 3})
    record = build_archive_record("tok", candidates, selected, candidates[0].components, {"support_top_m": 3})
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        save_elite_record(root, "tok", record)
        loaded = load_elite_record(root, "tok")
        assert int(loaded["version"]) == 3
        assert "feasibility" in loaded and "support_tags" in loaded


def test_candidate_funnel() -> None:
    traj = _straight().numpy()[0]
    pool = build_candidate_pool_for_scene("tok", traj, traj * 0.95, None, [], {"max_candidates_per_scene": 8})
    assert len(pool) > 0
    assert all(rec.trajectory.shape == (8, 3) for rec in pool)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "submission.pkl"
        with path.open("wb") as f:
            pickle.dump({"predictions": [{"tok": {"poses": traj}}]}, f)
        loaded = ExternalCandidateLoader({"external": str(path)}).load("tok")
        assert len(loaded) == 1
        assert loaded[0].source == "external"
        assert loaded[0].trajectory.shape == (8, 3)


def test_scorer_and_pdas() -> None:
    traj = _straight(4)
    scorer = ParetoVectorScorer(hidden_dim=32)
    out = scorer(torch.zeros(4, 2, 16), torch.zeros(4, 8), torch.zeros(4, 3), traj)
    assert set(out) >= {"pdms", "utility", "feas_cost"}
    assert all(torch.isfinite(value).all() for value in out.values())
    rewards = torch.tensor([[0.1, 0.3, 0.2, 0.5], [0.8, 0.82, 0.1, 0.0]], dtype=torch.float32)
    trajs = traj[:2].unsqueeze(1).repeat(1, 4, 1, 1)
    components = {
        "ego_progress": rewards,
        "time_to_collision_within_bound": torch.ones_like(rewards),
    }
    metrics = compute_pdas_metrics(rewards, components, trajs, {"ref_pdms": torch.tensor([0.2, 0.8]), "ref_ep": torch.tensor([0.2, 0.8])}, {})
    assert metrics.weight.shape == (2,)
    assert torch.isfinite(metrics.weight).all()
    assert not torch.allclose(metrics.weight[0], metrics.weight[1])


def test_feasible_pareto_advantage_runtime() -> None:
    planner = ReCogDriveDiffusionPlanner.__new__(ReCogDriveDiffusionPlanner)
    for key, value in {
        "fp_ep_weight": 0.60,
        "fp_ttc_weight": 0.25,
        "fp_ddc_weight": 0.10,
        "fp_feas_weight": 0.05,
        "fp_ddc_min_absolute": 0.95,
        "fp_ddc_ref_tolerance": 0.01,
        "fp_feas_max": 1.0,
        "fp_comfort_min": 0.95,
        "fp_pareto_front_bonus": 0.15,
        "fp_tradeoff_penalty_weight": 0.2,
        "fp_tradeoff_ttc_rho": 1.0,
        "fp_tradeoff_tolerance": 0.01,
        "fp_regression_negative_advantage": -1.0,
        "fp_invalid_negative_advantage": -1.0,
        "fp_dominated_positive_cap": 0.0,
        "fp_geometry_positive_cap": 0.0,
        "fp_ddc_regression_positive_cap": 0.0,
        "fp_offsupport_positive_cap": 0.0,
        "fp_use_bucketed_advantage": True,
        "fp_inter_bucket_weight": 0.25,
        "fp_inter_bucket_clip": 0.5,
        "fp_use_pdas": True,
        "pdas_eps": 0.05,
        "pdas_alpha": 1.0,
        "pdas_beta": 1.0,
        "pdas_gamma": 0.5,
        "pdas_lambda_bucket": 0.2,
        "pdas_lambda_regression": 0.5,
    }.items():
        setattr(planner, key, value)
    planner.config = type(
        "Cfg",
        (),
        {
            "geo_curvature_weight": 1.0,
            "geo_reverse_weight": 1.0,
            "geo_tail_reverse_weight": 2.0,
            "geo_early_kink_weight": 2.0,
            "geo_jerk_weight": 0.2,
        },
    )()
    rewards = torch.tensor([[0.8, 0.7, 0.0], [0.4, 0.6, 0.5]], dtype=torch.float32)
    components = {
        "pdms": rewards,
        "ego_progress": torch.tensor([[0.8, 0.7, 0.2], [0.4, 0.7, 0.6]]),
        "time_to_collision_within_bound": torch.ones(2, 3),
        "history_comfort": torch.ones(2, 3),
        "no_at_fault_collisions": torch.ones(2, 3),
        "drivable_area_compliance": torch.ones(2, 3),
        "driving_direction_compliance": torch.tensor([[1.0, 0.9, 1.0], [1.0, 1.0, 0.8]]),
    }
    trajs = _straight(2).unsqueeze(1).repeat(1, 3, 1, 1)
    ref = {
        "ref_pdms": torch.tensor([0.7, 0.5]),
        "ref_ep": torch.tensor([0.7, 0.5]),
        "ref_ttc": torch.ones(2),
        "ref_ddc": torch.ones(2),
    }
    adv, group_weight, aux = planner._compute_feasible_pareto_advantages(rewards, components, trajs, ref)
    assert adv.shape == (6,)
    assert group_weight.shape == (2,)
    assert torch.isfinite(adv).all()
    assert torch.isfinite(group_weight).all()
    assert "fp_valid_ratio" in aux and float(aux["fp_ddc_regression_ratio"]) > 0.0


def main() -> None:
    test_trajectory_feasibility()
    test_fs_norm_roundtrip()
    test_pareto_support_and_buffer_v3()
    test_candidate_funnel()
    test_scorer_and_pdas()
    test_feasible_pareto_advantage_runtime()
    print("SG-FPS smoke tests passed")


if __name__ == "__main__":
    main()
