from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluation"
    / "aggregate_recogdrive_support_aligned_diversity.py"
)
SPEC = importlib.util.spec_from_file_location("aggregate_support_diversity", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_paired_candidate_win_rate_respects_metric_direction() -> None:
    baseline = pd.DataFrame(
        {
            "token": ["a", "b"],
            "support_f1_auc": [0.5, 0.5],
            "mean_gt_ade_m": [0.5, 0.5],
            "best_gt_ade_m": [0.2, 0.2],
            "reference_mode_count": [1, 2],
        }
    )
    candidate = pd.DataFrame(
        {
            "token": ["a", "b"],
            "support_f1_auc": [0.6, 0.4],
            "mean_gt_ade_m": [0.4, 0.6],
            "best_gt_ade_m": [0.1, 0.3],
            "reference_mode_count": [1, 2],
        }
    )

    comparison = MODULE.paired_comparison(
        baseline,
        candidate,
        np.random.default_rng(0),
        samples=20,
    )
    result = comparison["metrics"]

    assert result["support_f1_auc"]["candidate_win_rate"] == 0.5
    assert result["support_f1_auc"]["higher_is_better"] is True
    assert result["mean_gt_ade_m"]["candidate_win_rate"] == 0.5
    assert result["mean_gt_ade_m"]["higher_is_better"] is False
    assert result["best_gt_ade_m"]["candidate_win_rate"] == 0.5
    assert result["best_gt_ade_m"]["higher_is_better"] is False
    assert comparison["strata"]["mode_1"]["num_aligned_scenes"] == 1
    assert comparison["strata"]["mode_2_3"]["num_aligned_scenes"] == 1
    assert comparison["strata"]["mode_1"]["metrics"]["mean_gt_ade_m"]["candidate_win_rate"] == 1.0
    assert comparison["strata"]["mode_2_3"]["metrics"]["mean_gt_ade_m"]["candidate_win_rate"] == 0.0
