import json

from scripts.risk_vla.build_safe_alignment_pairs import build_pairs


def test_safe_alignment_pairs_preserve_split_and_anchor_metadata(tmp_path):
    labels = tmp_path / "labels.jsonl"
    rows = [
        {
            "sample_token": "tok0",
            "split": "train",
            "purpose": "training",
            "candidate_id": 0,
            "strategy_name": "path_intent",
            "pdm_score": 0.7,
            "utility_score": -1.0,
            "positive_anchor": "1",
            "negative_anchor": "0",
            "regresses_nc": True,
            "regresses_ttc": False,
            "repairs_path": True,
            "tail_risk_label": True,
        },
        {
            "sample_token": "tok0",
            "split": "train",
            "purpose": "training",
            "candidate_id": 1,
            "strategy_name": "interaction",
            "pdm_score": 0.8,
            "utility_score": 0.6,
            "positive_anchor": "1",
            "negative_anchor": "0",
            "regresses_nc": False,
            "regresses_ttc": False,
            "repairs_path": True,
            "tail_risk_label": False,
        },
    ]
    with labels.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    summary = build_pairs(labels, tmp_path / "pairs")
    assert summary["num_pairs"] == 1

    pair = json.loads((tmp_path / "pairs" / "safe_alignment_pairs.jsonl").read_text(encoding="utf-8"))
    assert pair["sample_token"] == "tok0"
    assert pair["split"] == "train"
    assert pair["positive_candidate_id"] == "1"
    assert pair["negative_candidate_id"] == "0"
    assert pair["positive_utility_score"] > pair["negative_utility_score"]
    assert pair["pair_type"] == "safe_positive_vs_risky_negative"
