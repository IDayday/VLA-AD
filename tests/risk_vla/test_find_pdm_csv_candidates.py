from __future__ import annotations

import pandas as pd

from scripts.risk_vla.find_pdm_csv_candidates import find_candidates


PDM_ROW = {
    "token": "t0",
    "score": 0.5,
    "no_at_fault_collisions": 1.0,
    "drivable_area_compliance": 1.0,
    "ego_progress": 0.7,
    "time_to_collision_within_bound": 1.0,
    "comfort": 1.0,
}


def test_find_pdm_csv_candidates_ranks_valid_csv_and_handles_bad_csv(tmp_path):
    valid = tmp_path / "navval_A0_pdm.csv"
    pd.DataFrame([PDM_ROW, {**PDM_ROW, "token": "t1"}]).to_csv(valid, index=False)
    bad = tmp_path / "bad.csv"
    bad.write_text("not,pdm\n1,2\n", encoding="utf-8")

    df = find_candidates(
        search_roots=[tmp_path],
        max_depth=2,
        include_patterns=[],
        method_hints=["A0", "base", "B3", "bit"],
        split_hints=["navtrain", "navval"],
    )

    assert len(df) == 2
    assert df.iloc[0]["path"].endswith("navval_A0_pdm.csv")
    assert bool(df.iloc[0]["has_required_pdm_columns"])
    assert df.iloc[0]["row_count"] == 2
    assert not bool(df[df["path"].str.endswith("bad.csv")].iloc[0]["has_required_pdm_columns"])
