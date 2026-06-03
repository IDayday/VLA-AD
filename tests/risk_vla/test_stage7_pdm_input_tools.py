from __future__ import annotations

import os
import subprocess

import pandas as pd

from scripts.risk_vla.build_round1_artifact_dashboard import build_dashboard
from scripts.risk_vla.build_round1_manifest_from_env import build_manifest
from scripts.risk_vla.build_small_pdm_eval_commands import build_commands


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
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([PDM_ROW]).to_csv(path, index=False)


def test_manifest_from_env_registers_existing_csvs_only(tmp_path):
    a0 = tmp_path / "a0.csv"
    train = tmp_path / "train.csv"
    _pdm(a0)
    _pdm(train)
    args = type(
        "Args",
        (),
        {
            "a0_pdm_csv": str(a0),
            "b3_pdm_csv": str(tmp_path / "missing.csv"),
            "train_pdm_csv": str(train),
            "val_pdm_csv": "",
            "source_chunk_cache_dir": str(tmp_path / "cache"),
            "overlay_output_dir": str(tmp_path / "overlay"),
            "checkpoint": str(tmp_path / "ckpt"),
            "analysis_split": "analysis_only",
        },
    )()
    manifest = build_manifest(args)
    assert "A0_base" in manifest["analysis_pdm"]
    assert "B3_direct_bit" not in manifest["analysis_pdm"]
    assert "train" in manifest["trainval_pdm"]


def test_small_pdm_command_builder_writes_blockers_when_inputs_missing(tmp_path):
    args = type(
        "Args",
        (),
        {
            "output_dir": tmp_path / "pdm_inputs",
            "a0_checkpoint": "",
            "b3_checkpoint": "",
            "cache_path": "",
            "metric_cache_path": "",
            "navsim_log_path": "",
            "sensor_blobs_path": "",
            "vlm_path": "",
            "split": "navval",
            "max_samples": 256,
            "devices": 1,
            "master_port": 29671,
        },
    )()
    summary = build_commands(args)
    assert summary["has_blockers"]
    command = (tmp_path / "pdm_inputs" / "commands" / "generate_a0_pdm.sh").read_text(encoding="utf-8")
    assert "PDM generation blocked" in command
    assert (tmp_path / "pdm_inputs" / "evaluator_path_report.md").is_file()


def test_small_pdm_command_builder_includes_max_scenes_guard_when_ready(tmp_path, monkeypatch):
    ckpt_root = tmp_path / "ckpts"
    vlm = ckpt_root / "recogdrive" / "ReCogDrive-VLM-2B"
    vlm.mkdir(parents=True)
    a0 = tmp_path / "a0.ckpt"
    b3 = tmp_path / "b3.ckpt"
    metric = tmp_path / "metric"
    logs = tmp_path / "logs"
    blobs = tmp_path / "blobs"
    for path in (a0, b3):
        path.write_text("ckpt", encoding="utf-8")
    for path in (metric, logs, blobs):
        path.mkdir()
    monkeypatch.setenv("CHECKPOINT_ROOT", str(ckpt_root))
    args = type(
        "Args",
        (),
        {
            "output_dir": tmp_path / "pdm_inputs",
            "a0_checkpoint": str(a0),
            "b3_checkpoint": str(b3),
            "cache_path": "",
            "metric_cache_path": str(metric),
            "navsim_log_path": str(logs),
            "sensor_blobs_path": str(blobs),
            "vlm_path": "",
            "split": "navval",
            "max_samples": 17,
            "devices": 1,
            "master_port": 29671,
        },
    )()
    summary = build_commands(args)
    assert not summary["has_blockers"]
    script = (tmp_path / "pdm_inputs" / "commands" / "generate_b3_pdm.sh").read_text(encoding="utf-8")
    assert "EXECUTE=0" in script
    assert "train_test_split.scene_filter.max_scenes=${MAX_SAMPLES}" in script
    assert "configs/bit_drive/v3/bit_v3_C1_lateral_terminal.yaml" not in script
    assert "agent.use_bit_drive=true" in script


def test_artifact_dashboard_validates_pdm_columns(tmp_path):
    round_dir = tmp_path / "round1"
    _pdm(round_dir / "pdm_inputs" / "A0_base" / "pdm.csv")
    dashboard = build_dashboard(round_dir)
    assert dashboard["pdm_csvs"]["A0_base"]["has_required_pdm_columns"]
    assert not dashboard["summary"]["all_required_pdm_csvs_ready"]


def test_run18_defaults_to_no_eval(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "BIT_WORK_ROOT": str(tmp_path / "work"),
            "BIT_EXP_ROOT": str(tmp_path / "exp"),
            "CHECKPOINT_ROOT": str(tmp_path / "ckpts"),
            "EXECUTE": "0",
            "DRY_RUN": "1",
        }
    )
    (tmp_path / "work").mkdir()
    (tmp_path / "exp").mkdir()
    (tmp_path / "ckpts").mkdir()
    result = subprocess.run(
        ["bash", "scripts/risk_vla/round1/run_18_materialize_or_register_pdm_inputs.sh"],
        cwd="/mnt/project/bit_drive_left_tail/VLA-AD",
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "EXECUTE=0" in result.stdout
    assert (tmp_path / "exp" / "round1" / "stage7_pdm_inputs" / "stage7_pdm_input_status.md").is_file()
