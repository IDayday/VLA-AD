from __future__ import annotations

import torch

from scripts.generate_last_vla_teacher_trajectory_cache import build_teacher_payload


def test_teacher_cache_proxy_payload_schema_top1_and_candidates():
    candidates = [torch.zeros(8, 3), torch.ones(8, 3) * 0.2]
    payload = build_teacher_payload(
        sample_token="sample_000",
        scene_token="scene_000",
        candidates=candidates,
        candidate_scores=[0.1, 0.4],
        candidate_components=[{"score_mode": "proxy"}, {"score_mode": "proxy"}],
        best_index=1,
        gt_score=0.2,
        gt_components=None,
        score_mode="proxy",
        num_candidates=2,
        save_candidates=True,
    )

    assert payload["teacher_source"] == "best_of_k_proxy"
    assert payload["score_mode"] == "proxy"
    assert torch.allclose(payload["teacher_trajectory"], candidates[1])
    assert torch.isclose(payload["teacher_score"], torch.tensor(0.4))
    assert torch.isclose(payload["gt_score"], torch.tensor(0.2))
    assert torch.isclose(payload["oracle_best_of_k_score"], torch.tensor(0.4))
    assert payload["candidate_scores"].shape == (2,)
    assert payload["candidate_trajectories"].shape == (2, 8, 3)
    assert "pdm_components" not in payload
