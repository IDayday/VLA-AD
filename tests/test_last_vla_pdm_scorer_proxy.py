from __future__ import annotations

import torch

from scripts.last_vla_v2.pdm_scoring_utils import PDM_COMPONENT_KEYS, TrajectoryScorer


def test_proxy_scorer_works_without_metric_cache_and_nulls_pdm_fields():
    scorer = TrajectoryScorer("proxy")
    sample = {"trajectory": torch.zeros(8, 3)}
    result = scorer.score(sample, "sample_000", torch.ones(8, 3) * 0.1)

    assert result["score_mode"] == "proxy"
    assert result["proxy_score"] == result["score"]
    assert result["trajectory_l1"] is not None
    for key in PDM_COMPONENT_KEYS:
        assert result[key] is None
