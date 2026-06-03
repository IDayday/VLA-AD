from __future__ import annotations

import torch

from scripts.last_vla_v2.pdm_scoring_utils import TrajectoryScorer


def test_pdm_scorer_mock_populates_components_without_proxy():
    calls = []

    def backend(sample, sample_token, trajectory):
        calls.append((sample, sample_token, trajectory.clone()))
        return {
            "PDMS": 0.91,
            "NC": 1.0,
            "DAC": 0.8,
            "TTC": 0.7,
            "comfort": 0.6,
            "EP": 0.5,
            "DDC": 0.4,
        }

    scorer = TrajectoryScorer("pdm", pdm_backend=backend)
    result = scorer.score({"trajectory": torch.zeros(8, 3)}, "sample_001", torch.zeros(8, 3))

    assert len(calls) == 1
    assert result["score_mode"] == "pdm"
    assert result["score"] == 0.91
    assert result["PDMS"] == 0.91
    assert "proxy_score" not in result
