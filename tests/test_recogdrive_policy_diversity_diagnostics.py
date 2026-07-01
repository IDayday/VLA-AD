import math

import numpy as np

from navsim.planning.script.run_recogdrive_policy_diversity_diagnostics import _trajectory_diagnostics


def test_trajectory_diagnostics_uses_min_gt_error_and_mean_pairwise_spread():
    candidates = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 4.0, 0.0]],
        ],
        dtype=np.float32,
    )
    gt = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)

    result = _trajectory_diagnostics(candidates, gt)

    assert result["quality_min_ade"] == 0.0
    assert result["quality_min_fde"] == 0.0
    assert math.isclose(result["diversity_mean_pade"], (1.0 + 2.0 + math.sqrt(20.0) / 2.0) / 3.0)
    assert math.isclose(result["diversity_mean_pfde"], (2.0 + 4.0 + math.sqrt(20.0)) / 3.0)
