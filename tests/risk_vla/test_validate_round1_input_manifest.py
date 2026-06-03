from __future__ import annotations

import pandas as pd
import yaml

from scripts.risk_vla.round1_manifest import validate_manifest


PDM_ROW = {
    "token": "t0",
    "score": 0.5,
    "no_at_fault_collisions": 1.0,
    "drivable_area_compliance": 1.0,
    "ego_progress": 0.7,
    "time_to_collision_within_bound": 1.0,
    "comfort": 1.0,
}


def _pdm(path):
    pd.DataFrame([PDM_ROW]).to_csv(path, index=False)


def _manifest(tmp_path, *, train_split="navtrain"):
    a0 = tmp_path / "a0.csv"
    b3 = tmp_path / "b3.csv"
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    for path in (a0, b3, train, val):
        _pdm(path)
    cache = tmp_path / "cache"
    cache.mkdir()
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("ckpt", encoding="utf-8")
    data = {
        "round": "risk_vla_round1",
        "analysis_pdm": {
            "A0_base": {"csv": str(a0), "split": "navtest1024", "use_for_training": False},
            "B3_direct_bit": {"csv": str(b3), "split": "navtest1024", "use_for_training": False},
        },
        "trainval_pdm": {
            "train": {"csv": str(train), "split": train_split, "use_for_training": True},
            "val": {"csv": str(val), "split": "navval", "use_for_training": True},
        },
        "cache": {"source_chunk_cache_dir": str(cache), "overlay_output_dir": str(tmp_path / "overlay")},
        "checkpoints": {"base_or_bit_checkpoint": str(ckpt)},
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_manifest_accepts_analysis_only_navtest(tmp_path):
    report = validate_manifest(_manifest(tmp_path))
    assert report["valid"]
    assert report["found_inputs"]["a0_pdm_csv"].endswith("a0.csv")
    assert report["analysis_pdm"]["A0_base"]["use_for_training"] is False


def test_manifest_rejects_test_split_for_training(tmp_path):
    report = validate_manifest(_manifest(tmp_path, train_split="navtest"))
    assert not report["valid"]
    assert any("cannot be used for training" in error or "training split is blocked" in error for error in report["errors"])
