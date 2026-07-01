from __future__ import annotations

import json

from navsim.planning.script.run_pdm_score_recogdrive_best_of_n_exact_pool import (
    _candidate_seed,
    _select_best_candidate_row,
)


def test_best_of_n_selects_highest_valid_score():
    rows = [
        {"token": "tok", "valid": True, "rank": 0, "candidate_index": 0, "candidate_seed": 10, "score": 0.2},
        {"token": "tok", "valid": False, "rank": 0, "candidate_index": 1, "candidate_seed": 11},
        {"token": "tok", "valid": True, "rank": 0, "candidate_index": 2, "candidate_seed": 12, "score": 0.9},
        {"token": "tok", "valid": True, "rank": 0, "candidate_index": 3, "candidate_seed": 13, "score": 0.7},
    ]

    selected = _select_best_candidate_row(rows, token="tok", rank=0, index=5, best_of_n=4)

    assert selected["valid"] is True
    assert selected["score"] == 0.9
    assert selected["best_candidate_index"] == 2
    assert selected["candidate_valid_count"] == 3
    assert selected["candidate_score_max"] == 0.9
    assert selected["best_delta_vs_candidate0"] == 0.7
    assert json.loads(selected["candidate_scores_json"]) == [0.2, None, 0.9, 0.7]


def test_best_of_n_tie_breaks_by_lowest_candidate_index():
    rows = [
        {"token": "tok", "valid": True, "rank": 0, "candidate_index": 0, "candidate_seed": 10, "score": 0.8},
        {"token": "tok", "valid": True, "rank": 0, "candidate_index": 1, "candidate_seed": 11, "score": 0.8},
    ]

    selected = _select_best_candidate_row(rows, token="tok", rank=0, index=0, best_of_n=2)

    assert selected["best_candidate_index"] == 0
    assert selected["best_candidate_seed"] == 10


def test_candidate_seed_is_stable_and_candidate_specific():
    first = _candidate_seed(260306049, "abc", 0)
    second = _candidate_seed(260306049, "abc", 0)
    other = _candidate_seed(260306049, "abc", 1)

    assert first == second
    assert first != other
    assert 0 <= first < 2**31 - 1
