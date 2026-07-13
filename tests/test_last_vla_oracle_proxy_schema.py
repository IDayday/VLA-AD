from __future__ import annotations

from scripts.eval_last_vla_best_of_k_oracle import oracle_metrics, synthetic_rows


def test_oracle_proxy_schema_uses_score_fields_not_pdm_claims():
    rows = synthetic_rows(num_samples=2, k=4, seed=7)
    metrics = oracle_metrics(rows, k=4, score_mode="proxy")

    assert metrics["score_mode"] == "proxy"
    assert metrics["proxy_scoring_active"] is True
    assert metrics["pdm_scoring_active"] is False
    assert metrics["deterministic_score"] is not None
    assert metrics["oracle_best_of_K_score"] >= metrics["stochastic_mean_score"]
    assert metrics["deterministic_PDMS"] is None
    assert metrics["oracle_best_of_K_PDMS"] is None
    assert metrics["oracle_delta_vs_A0_baseline"] is None
