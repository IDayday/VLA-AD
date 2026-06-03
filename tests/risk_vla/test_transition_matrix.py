import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def test_transition_matrix_counts_and_rates(tmp_path):
    base_csv = tmp_path / "base.csv"
    method_csv = tmp_path / "method.csv"
    output_csv = tmp_path / "transition.csv"
    output_md = tmp_path / "transition.md"
    pd.DataFrame(
        [
            {"token": "t1", "score": 0.0, "drivable_area_compliance": 0.0, "no_at_fault_collisions": 1.0, "time_to_collision_within_bound": 1.0, "ego_progress": 0.1, "comfort": 0.2},
            {"token": "t2", "score": 0.8, "drivable_area_compliance": 1.0, "no_at_fault_collisions": 1.0, "time_to_collision_within_bound": 1.0, "ego_progress": 0.1, "comfort": 0.1},
            {"token": "t3", "score": 0.8, "drivable_area_compliance": 1.0, "no_at_fault_collisions": 1.0, "time_to_collision_within_bound": 1.0, "ego_progress": 0.5, "comfort": 0.9},
            {"token": "t4", "score": 0.8, "drivable_area_compliance": 1.0, "no_at_fault_collisions": 1.0, "time_to_collision_within_bound": 1.0, "ego_progress": 0.3, "comfort": 0.3},
        ]
    ).to_csv(base_csv, index=False)
    pd.DataFrame(
        [
            {"token": "t1", "score": 0.9, "drivable_area_compliance": 1.0, "no_at_fault_collisions": 1.0, "time_to_collision_within_bound": 1.0, "ego_progress": 0.2, "comfort": 0.3},
            {"token": "t2", "score": 0.7, "drivable_area_compliance": 0.0, "no_at_fault_collisions": 1.0, "time_to_collision_within_bound": 1.0, "ego_progress": 0.5, "comfort": 0.5},
            {"token": "t3", "score": 0.0, "drivable_area_compliance": 1.0, "no_at_fault_collisions": 0.0, "time_to_collision_within_bound": 1.0, "ego_progress": 0.5, "comfort": 0.9},
            {"token": "t4", "score": 0.0, "drivable_area_compliance": 1.0, "no_at_fault_collisions": 1.0, "time_to_collision_within_bound": 0.0, "ego_progress": 0.3, "comfort": 0.3},
        ]
    ).to_csv(method_csv, index=False)

    subprocess.run(
        [
            sys.executable,
            "scripts/risk_vla/build_risk_transition_matrix.py",
            "--base-csv",
            str(base_csv),
            "--method-csv",
            str(method_csv),
            "--method-name",
            "method",
            "--output-csv",
            str(output_csv),
            "--output-md",
            str(output_md),
        ],
        cwd=ROOT,
        check=True,
    )

    row = pd.read_csv(output_csv).iloc[0]
    assert int(row["num_matched_tokens"]) == 4
    assert int(row["num_base_dac0"]) == 1
    assert int(row["num_method_dac0"]) == 1
    assert int(row["num_method_nc0"]) == 1
    assert int(row["num_method_ttc0"]) == 1
    assert int(row["num_path_repair"]) == 1
    assert int(row["num_path_regression"]) == 1
    assert int(row["num_interaction_regression"]) == 2
    assert int(row["num_safe_path_repair"]) == 1
    assert float(row["path_repair_rate"]) == 1.0
    assert "RISK-VLA Transition Matrix" in output_md.read_text()
