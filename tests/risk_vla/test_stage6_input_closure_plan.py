from __future__ import annotations

import subprocess
import os

import pandas as pd
import yaml

from scripts.risk_vla.build_round1_execution_plan import build_plan, write_plan
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


def _write_manifest(tmp_path, *, train_split="navtrain"):
    paths = {}
    for name in ("a0", "b3", "train", "val"):
        path = tmp_path / f"{name}.csv"
        pd.DataFrame([PDM_ROW]).to_csv(path, index=False)
        paths[name] = path
    cache = tmp_path / "cache"
    cache.mkdir()
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("ckpt", encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "round": "risk_vla_round1",
                "analysis_pdm": {
                    "A0_base": {"csv": str(paths["a0"]), "split": "navtest1024", "use_for_training": False},
                    "B3_direct_bit": {"csv": str(paths["b3"]), "split": "navtest1024", "use_for_training": False},
                },
                "trainval_pdm": {
                    "train": {"csv": str(paths["train"]), "split": train_split, "use_for_training": True},
                    "val": {"csv": str(paths["val"]), "split": "navval", "use_for_training": True},
                },
                "cache": {"source_chunk_cache_dir": str(cache), "overlay_output_dir": str(tmp_path / "overlay")},
                "checkpoints": {"base_or_bit_checkpoint": str(ckpt)},
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_execution_plan_uses_manifest_paths_and_comments(tmp_path):
    manifest = _write_manifest(tmp_path)
    report = validate_manifest(manifest)
    plan_dir = tmp_path / "plan"
    plan = build_plan(report, plan_dir, max_samples=64, limit_train_batches=5, limit_val_batches=2, input_manifest_yaml=manifest)
    write_plan(plan, plan_dir)

    collect = (plan_dir / "commands" / "00_collect_matched_pdm.sh").read_text(encoding="utf-8")
    labels = (plan_dir / "commands" / "01_build_risk_labels.sh").read_text(encoding="utf-8")
    assert str(tmp_path / "a0.csv") in collect
    assert "analysis-only PDM A0_base" in collect
    assert str(tmp_path / "train.csv") in labels
    assert "EXECUTE=0; would build train/val risk labels" in labels


def test_stage6_closure_script_defaults_to_no_execution(tmp_path):
    work = tmp_path / "work"
    exp = tmp_path / "exp"
    ckpts = tmp_path / "ckpts"
    work.mkdir()
    exp.mkdir()
    ckpts.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "BIT_WORK_ROOT": str(work),
            "BIT_EXP_ROOT": str(exp),
            "CHECKPOINT_ROOT": str(ckpts),
            "EXECUTE": "0",
            "DRY_RUN": "1",
        }
    )
    result = subprocess.run(
        ["bash", "scripts/risk_vla/round1/run_15_close_input_gaps_and_build_plan.sh"],
        cwd="/mnt/project/bit_drive_left_tail/VLA-AD",
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    summary = exp / "round1" / "stage6_input_closure" / "stage6_input_closure_summary.md"
    assert summary.is_file()
    assert "A0/B3 CSVs are still missing" in summary.read_text(encoding="utf-8")
    assert "never starts training/eval" in result.stdout


def test_r0_from_manifest_wrapper_rejects_test_training_split(tmp_path):
    manifest = _write_manifest(tmp_path, train_split="navtest")
    env = os.environ.copy()
    env.update({"MANIFEST_YAML": str(manifest), "BIT_EXP_ROOT": str(tmp_path / "exp"), "EXECUTE": "0"})
    result = subprocess.run(
        ["bash", "scripts/risk_vla/round1/run_16_execute_round1_r0_from_manifest.sh"],
        cwd="/mnt/project/bit_drive_left_tail/VLA-AD",
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "navtest" in result.stdout
