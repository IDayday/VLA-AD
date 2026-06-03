from __future__ import annotations

import torch

from scripts.risk_vla.discover_round1_inputs import discover_inputs, write_markdown


def _pdm_csv(path):
    path.write_text(
        "token,score,no_at_fault_collisions,drivable_area_compliance,ego_progress,time_to_collision_within_bound,comfort,driving_direction_compliance\n"
        "a,0.9,1,1,0.5,1,1,1\n",
        encoding="utf-8",
    )


def test_discover_round1_inputs_reports_missing(tmp_path):
    report = discover_inputs(
        a0_pdm_csv=None,
        bit_pdm_csv=None,
        chunk_cache_dir=tmp_path / "missing_cache",
        checkpoint_dir=tmp_path / "missing_ckpt",
    )
    assert "a0_pdm_csv" in report["missing_inputs"]
    assert "chunk_cache_dir" in report["missing_inputs"]
    output = tmp_path / "report.md"
    write_markdown(report, output)
    assert "Missing Inputs" in output.read_text(encoding="utf-8")


def test_discover_round1_inputs_validates_synthetic_inputs(tmp_path):
    a0 = tmp_path / "a0.csv"
    bit = tmp_path / "bit.csv"
    _pdm_csv(a0)
    _pdm_csv(bit)
    cache = tmp_path / "cache"
    (cache / "samples").mkdir(parents=True)
    torch.save({"sample_token": "a"}, cache / "samples" / "a.pt")
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    torch.save({"state_dict": {}}, ckpt / "best.ckpt")

    report = discover_inputs(a0_pdm_csv=a0, bit_pdm_csv=bit, chunk_cache_dir=cache, checkpoint_dir=ckpt)

    assert report["missing_inputs"] == []
    assert report["found_inputs"]["a0_pdm_csv"] == str(a0)
    assert report["found_inputs"]["checkpoint_candidates"]
