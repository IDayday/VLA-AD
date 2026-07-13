from __future__ import annotations

import json

import pytest
import torch

from navsim.agents.recogdrive.stage2_frontier_sampling import (
    DistributedStage2FrontierSampler,
    load_stage2_frontier_weights,
)


def _write_index(path, tokens: list[str], eligible: set[str]) -> None:
    payload = {
        "version": 1,
        "capacity_semantics": "observed_selected_support_not_scene_intrinsic",
        "no_per_scene_candidate_quota": True,
        "records": [
            {
                "token": token,
                "frontier_eligible": token in eligible,
                "priority": 1.0 if token in eligible else 0.0,
            }
            for token in tokens
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_stage2_frontier_weights_keep_gt_only_scenes_in_uniform_branch(tmp_path) -> None:
    path = tmp_path / "index.json"
    _write_index(path, ["mode", "gt_a", "gt_b"], {"mode"})

    output = load_stage2_frontier_weights(
        path,
        ["mode", "gt_a", "gt_b"],
        uniform_ratio=0.5,
        priority_exponent=0.0,
    )

    assert torch.allclose(
        output.weights,
        torch.tensor([2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0], dtype=torch.float64),
    )
    assert bool((output.weights[~output.eligible_mask] > 0.0).all())
    assert output.diagnostics["stage2_frontier_expected_sampled_scene_ratio"] == pytest.approx(2.0 / 3.0)


def test_stage2_frontier_weights_fail_on_missing_dataset_token(tmp_path) -> None:
    path = tmp_path / "index.json"
    _write_index(path, ["known"], {"known"})

    with pytest.raises(KeyError, match="does not cover"):
        load_stage2_frontier_weights(
            path,
            ["known", "missing"],
            uniform_ratio=0.5,
            priority_exponent=0.5,
        )


def test_distributed_stage2_frontier_sampler_uses_global_mixture(tmp_path) -> None:
    tokens = [f"scene_{index}" for index in range(100)]
    eligible_tokens = set(tokens[:10])
    path = tmp_path / "index.json"
    _write_index(path, tokens, eligible_tokens)
    output = load_stage2_frontier_weights(
        path,
        tokens,
        uniform_ratio=0.5,
        priority_exponent=0.5,
    )
    rank_streams = []
    for rank in range(2):
        sampler = DistributedStage2FrontierSampler(
            len(tokens),
            output.weights,
            eligible_mask=output.eligible_mask,
            num_replicas=2,
            rank=rank,
            seed=13,
            warmup_epochs=0,
            uniform_ratio=0.5,
        )
        sampler.set_epoch(3)
        rank_streams.extend(list(sampler))

    sampled_eligible_ratio = float(output.eligible_mask[rank_streams].float().mean())
    assert sampled_eligible_ratio == pytest.approx(0.55, abs=0.12)
    assert bool((output.weights[~output.eligible_mask] > 0.0).all())
