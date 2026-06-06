import csv
import json
from pathlib import Path

import pytest

from scripts.risk_vla.aggregate_strategy_utility_report import aggregate
from scripts.risk_vla.aggregate_strategy_utility_labels import aggregate as aggregate_labels
from scripts.risk_vla.build_safe_alignment_pairs import build_pairs
from scripts.risk_vla.build_strategy_utility_labels import build_labels, CandidateInput, write_outputs
from scripts.risk_vla.check_strategy_utility_labels import check_labels


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_strategy_utility_labels_choose_safe_positive_and_risky_negative(tmp_path):
    base = tmp_path / "base.csv"
    bit = tmp_path / "bit.csv"
    conservative = tmp_path / "conservative.csv"
    _write_csv(base, [{"sample_token": "a", "score": 0.0, "dac": 0, "nc": 1, "ttc": 1, "progress": 0.4, "comfort": 1}])
    _write_csv(bit, [{"sample_token": "a", "score": 0.9, "dac": 1, "nc": 0, "ttc": 1, "progress": 0.9, "comfort": 1}])
    _write_csv(conservative, [{"sample_token": "a", "score": 0.8, "dac": 1, "nc": 1, "ttc": 1, "progress": 0.8, "comfort": 1}])

    rows, summary = build_labels(
        CandidateInput("A0", base, "base"),
        [CandidateInput("BiT", bit, "path_intent"), CandidateInput("Safe", conservative, "interaction")],
        split="train",
        purpose="training",
    )
    assert summary["num_tokens"] == 1
    by_name = {row["candidate_name"]: row for row in rows}
    assert by_name["BiT"]["token"] == "a"
    assert by_name["BiT"]["token_id"] == "a"
    assert by_name["BiT"]["source_method"] == "BiT"
    assert by_name["BiT"]["drivable_area_compliance"] == by_name["BiT"]["dac"]
    assert by_name["BiT"]["no_at_fault_collisions"] == by_name["BiT"]["nc"]
    assert by_name["BiT"]["time_to_collision_within_bound"] == by_name["BiT"]["ttc"]
    assert by_name["BiT"]["ego_progress"] == by_name["BiT"]["progress"]
    assert by_name["BiT"]["score"] == by_name["BiT"]["pdm_score"]
    assert by_name["BiT"]["prog"] == by_name["BiT"]["progress"]
    assert by_name["BiT"]["repairs_path"] is True
    assert by_name["BiT"]["path_repair"] is True
    assert by_name["BiT"]["regresses_nc"] is True
    assert by_name["BiT"]["nc_regression"] is True
    assert by_name["BiT"]["unsafe_nc"] is True
    assert by_name["BiT"]["unsafe_regression"] is True
    assert by_name["BiT"]["delta_score_vs_a0"] == by_name["BiT"]["delta_score"]
    assert by_name["BiT"]["utility_score"] < by_name["Safe"]["utility_score"]
    assert by_name["Safe"]["safe_path_repair"] is True
    assert by_name["Safe"]["no_safe_candidate"] is False
    assert str(by_name["Safe"]["positive_anchor"]) == str(by_name["Safe"]["candidate_id"])
    assert str(by_name["Safe"]["constrained_best_candidate"]) == str(by_name["Safe"]["candidate_id"])
    assert by_name["Safe"]["pair_type"] == "positive"
    assert str(by_name["BiT"]["negative_anchor"]) == str(by_name["BiT"]["candidate_id"])
    assert by_name["BiT"]["pair_type"] == "negative"

    out = tmp_path / "out"
    write_outputs(rows, summary, out)
    check = check_labels(out / "strategy_utility_labels.jsonl")
    assert check["ready_for_training"] is True
    agg = aggregate(out / "strategy_utility_labels.jsonl")
    assert agg["num_rows"] == 2
    agg2 = aggregate_labels(out / "strategy_utility_labels.jsonl")
    assert agg2["transition_counts"]["nc_regression"] == 1
    pair_summary = build_pairs(out / "strategy_utility_labels.jsonl", tmp_path / "pairs")
    assert pair_summary["num_pairs"] == 1


def test_strategy_utility_labels_reject_navtest_for_training(tmp_path):
    base = tmp_path / "base.csv"
    cand = tmp_path / "cand.csv"
    row = {"sample_token": "a", "score": 1, "dac": 1, "nc": 1, "ttc": 1, "progress": 1, "comfort": 1}
    _write_csv(base, [row])
    _write_csv(cand, [row])
    with pytest.raises(RuntimeError, match="Refusing"):
        build_labels(CandidateInput("A0", base, "base"), [CandidateInput("C", cand, "base")], split="navtest", purpose="training")
