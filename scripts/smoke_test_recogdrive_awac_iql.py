from __future__ import annotations

import lzma
import os
import pickle
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-recogdrive-awac-smoke")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.offline_rl_buffer import (
    REQUIRED_COMPONENT_KEYS,
    load_elite_record,
    save_elite_record,
    token_to_buffer_key,
)
from navsim.agents.recogdrive.recogdrive_diffusion_planner import OfflineRLConfig, ReCogDriveDiffusionPlanner
from scripts.training.validate_recogdrive_stage3_awac_elite_buffer import _validate_record


def _planner_stub() -> ReCogDriveDiffusionPlanner:
    return object.__new__(ReCogDriveDiffusionPlanner)


def _components() -> dict[str, torch.Tensor]:
    values = {
        "pdms": torch.tensor([[0.80, 0.95, 0.90, 0.70, 0.60], [0.60, 0.75, 0.85, 0.55, 0.65]]),
        "no_at_fault_collisions": torch.ones(2, 5),
        "drivable_area_compliance": torch.tensor([[1.0, 0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0, 1.0]]),
        "time_to_collision_within_bound": torch.ones(2, 5),
        "ego_progress": torch.ones(2, 5) * 0.5,
        "history_comfort": torch.ones(2, 5),
        "lane_keeping": torch.ones(2, 5),
        "driving_direction_compliance": torch.tensor([[1.0, 1.0, 0.80, 1.0, 1.0], [1.0, 0.7, 0.8, 0.9, 0.95]]),
        "traffic_light_compliance": torch.ones(2, 5),
    }
    return values


def _record(token: str, valid_mask: np.ndarray, version: int = 2) -> dict:
    k, h = 2, 8
    rewards = np.asarray([0.7, 0.8], dtype=np.float32)
    sources = ["gt", "progress_endpoint"]
    raw_idx = int(np.argmax(rewards))
    if valid_mask.any():
        valid_indices = np.flatnonzero(valid_mask)
        valid_idx = int(valid_indices[int(np.argmax(rewards[valid_indices]))])
    else:
        valid_idx = raw_idx
    components = {key: np.ones((k,), dtype=np.float32) for key in REQUIRED_COMPONENT_KEYS}
    components["pdms"] = rewards.copy()
    payload = {
        "token": token,
        "candidates": np.zeros((k, h, 3), dtype=np.float32),
        "rewards": rewards,
        "components": components,
        "sources": sources,
        "anchor_distance": np.zeros((k,), dtype=np.float32),
        "gt_reward": 0.7,
        "il_reward": 0.7,
        "best_reward": float(rewards[valid_idx]),
        "best_source": sources[valid_idx],
        "version": version,
    }
    if version >= 2:
        payload.update(
            {
                "valid_mask": valid_mask,
                "selection_score": rewards.copy(),
                "best_raw_reward": float(rewards[raw_idx]),
                "best_valid_reward": float(rewards[valid_idx]),
                "best_selected_reward": float(rewards[raw_idx]),
                "best_raw_source": sources[raw_idx],
                "best_valid_source": sources[valid_idx],
                "best_selected_source": sources[raw_idx],
                "has_valid_candidate": bool(valid_mask.any()),
            }
        )
    return payload


def _test_grpo_buffer_dpo_targets_do_not_fallback_il(planner: ReCogDriveDiffusionPlanner) -> None:
    cfg = OfflineRLConfig(grpo_buffer_preference_dpo_include_il=True)
    b, k, h, d = 2, 2, 8, 3
    gt_code = planner._source_code("gt")
    il_code = planner._source_code("il")
    progress_code = planner._source_code("progress_endpoint")
    guidance = {
        "target_trajs": torch.ones(b, 1, h, d),
        "target_rewards": torch.tensor([[0.95], [0.96]], dtype=torch.float32),
        "target_mask": torch.ones(b, 1, dtype=torch.bool),
        "gt_reward": torch.tensor([0.80, 0.81], dtype=torch.float32),
        "il_reward": torch.tensor([0.70, 0.71], dtype=torch.float32),
        "selected_trajs": torch.arange(b * k * h * d, dtype=torch.float32).reshape(b, k, h, d),
        "selected_real_mask": torch.ones(b, k, dtype=torch.bool),
        "selected_source_code": torch.tensor(
            [
                [il_code, progress_code],
                [progress_code, progress_code],
            ],
            dtype=torch.long,
        ),
    }
    action_input = BatchFeature(
        data={
            "action": torch.zeros(b, h, d),
            "his_traj": torch.zeros(b, 4, 3),
            "status_feature": torch.zeros(b, 8),
        }
    )
    out = planner._build_grpo_buffer_preference_dpo_targets(
        torch.zeros(b, 1, 4),
        action_input,
        guidance,
        cfg,
    )
    assert out["target_trajs"].shape == (b, 3, h, d)
    assert torch.equal(out["source_code"][:, 1], torch.tensor([gt_code, gt_code]))
    assert torch.equal(out["source_code"][:, 2], torch.tensor([il_code, il_code]))
    assert bool(out["real_mask"][0, 2])
    assert not bool(out["real_mask"][1, 2])
    assert torch.equal(out["target_trajs"][0, 2], guidance["selected_trajs"][0, 0])


def main() -> None:
    planner = _planner_stub()
    cfg = OfflineRLConfig(strict_reward_submetrics=True, elite_top_m=2, elite_min_candidates=2)
    sources = ["gt", "il", "policy", "progress_endpoint", "lateral_offset"]
    components = _components()
    valid_mask, valid_diag = planner._compute_awac_candidate_valid_mask(components, sources, cfg)
    assert valid_mask.shape == (2, 5)
    assert bool(valid_mask[0, 0])
    assert not bool(valid_mask[0, 1])
    assert not bool(valid_mask[0, 2])
    assert 0.0 < float(valid_diag["valid_candidate_ratio"]) < 1.0

    candidates = torch.zeros(2, 5, 8, 3)
    rewards = components["pdms"]
    anchor_distance = torch.zeros(2, 5)
    selected = planner._select_elite_candidates(
        candidates,
        rewards,
        components,
        sources,
        anchor_distance,
        gt_reward=rewards[:, 0],
        il_reward=rewards[:, 1],
        cfg=cfg,
    )
    assert selected["selected_trajs"].shape[0] == 2
    assert selected["selected_valid_mask"].shape == selected["selected_rewards"].shape
    assert torch.isfinite(selected["selected_selection_score"]).all()
    assert selected["best_reward"].shape == (2,)

    baseline = planner._compute_iql_baseline(
        selected["selected_rewards"],
        selected["gt_reward"],
        selected["il_reward"],
        cfg,
        selected["selected_real_mask"],
    )
    weights, advantages, weight_diag = planner._compute_awac_weights(
        selected["selected_rewards"],
        baseline,
        cfg,
        valid_mask=selected["selected_valid_mask"],
        real_mask=selected["selected_real_mask"],
        gt_reward=selected["gt_reward"],
    )
    assert weights.shape == selected["selected_rewards"].shape
    assert advantages.shape == selected["selected_rewards"].shape
    assert "positive_weight_row_ratio" in weight_diag

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        token = "scene-token"
        save_elite_record(root, token, _record(token, np.asarray([True, False], dtype=np.bool_)))
        loaded = load_elite_record(root, token)
        assert int(loaded["version"]) == 2
        assert np.asarray(loaded["valid_mask"]).tolist() == [True, False]

        row = _validate_record(loaded, strict_v2=True)
        assert row["best_valid_source"] == "gt"
        bad = dict(loaded)
        bad["best_valid_reward"] = bad["best_raw_reward"]
        try:
            _validate_record(bad, strict_v2=True)
            raise AssertionError("validator accepted inconsistent best_valid_reward")
        except ValueError:
            pass

        mismatch_token = "scene-token-mismatch"
        mismatch_path = root / f"{token_to_buffer_key(mismatch_token)}.pkl.xz"
        with lzma.open(mismatch_path, "wb") as f:
            pickle.dump(_record("different-token", np.asarray([True, False], dtype=np.bool_), version=1), f)
        try:
            load_elite_record(root, mismatch_token)
            raise AssertionError("load_elite_record accepted mismatched record token")
        except ValueError:
            pass

        v1_token = "scene-token-v1"
        v1_path = root / f"{token_to_buffer_key(v1_token)}.pkl.xz"
        with lzma.open(v1_path, "wb") as f:
            pickle.dump(_record(v1_token, np.asarray([True, False], dtype=np.bool_), version=1), f)
        loaded_v1 = load_elite_record(root, v1_token)
        assert "valid_mask" not in loaded_v1
        cfg.elite_buffer_path = str(root)
        action_input = BatchFeature(data={"action": torch.zeros(1, 8, 3)})
        loaded_batch = planner._load_awac_buffer_candidates(action_input, [v1_token], {}, cfg)
        assert loaded_batch["selected_valid_mask"].shape == (1, 2)
        assert bool(loaded_batch["selected_valid_mask"][0].any())

    _test_grpo_buffer_dpo_targets_do_not_fallback_il(planner)

    print("recogdrive_awac_iql_smoke_ok")


if __name__ == "__main__":
    main()
