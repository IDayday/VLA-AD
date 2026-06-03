import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def test_risk_label_builder_outputs_jsonl_csv_markdown(tmp_path):
    input_csv = tmp_path / "pdm.csv"
    output_jsonl = tmp_path / "labels.jsonl"
    output_csv = tmp_path / "labels.csv"
    output_md = tmp_path / "labels.md"
    pd.DataFrame(
        [
            {
                "token": "a",
                "score": 0.0,
                "no_at_fault_collisions": 1.0,
                "drivable_area_compliance": 0.0,
                "ego_progress": 0.1,
                "time_to_collision_within_bound": 1.0,
                "comfort": 0.9,
                "driving_direction_compliance": 1.0,
            },
            {
                "token": "b",
                "score": 0.5,
                "no_at_fault_collisions": 0.0,
                "drivable_area_compliance": 1.0,
                "ego_progress": 0.9,
                "time_to_collision_within_bound": 1.0,
                "comfort": 0.1,
                "driving_direction_compliance": 1.0,
            },
            {
                "token": "c",
                "score": 1.0,
                "no_at_fault_collisions": 1.0,
                "drivable_area_compliance": 1.0,
                "ego_progress": 0.2,
                "time_to_collision_within_bound": 0.0,
                "comfort": 0.8,
                "driving_direction_compliance": 1.0,
            },
        ]
    ).to_csv(input_csv, index=False)

    subprocess.run(
        [
            sys.executable,
            "scripts/risk_vla/build_risk_labels_from_pdm.py",
            "--input-csv",
            str(input_csv),
            "--output-jsonl",
            str(output_jsonl),
            "--output-csv",
            str(output_csv),
            "--output-md",
            str(output_md),
            "--horizon",
            "4",
            "--schema",
            "both",
        ],
        cwd=ROOT,
        check=True,
    )

    rows = [json.loads(line) for line in output_jsonl.read_text().splitlines()]
    assert len(rows) == 3
    assert len(rows[0]["generic_risk_labels"]) == 4
    assert len(rows[0]["risk_labels"]) == 4
    assert len(rows[0]["risk_labels"][0]) == 6
    assert rows[0]["hard_risk_flags"]["low_score"] is True
    assert rows[0]["hard_risk_flags"]["path_dac"] is True
    assert rows[1]["hard_risk_flags"]["interaction_nc"] is True
    assert rows[2]["hard_risk_flags"]["ttc"] is True
    summary = pd.read_csv(output_csv)
    assert {"label_low_score", "label_path_dac", "label_ttc"}.issubset(summary.columns)
    assert "RISK-VLA Label Build Report" in output_md.read_text()
