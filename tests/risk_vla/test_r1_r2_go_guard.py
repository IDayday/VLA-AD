from __future__ import annotations

import subprocess
import os


def test_r1_r2_guard_blocks_predicted_router_on_no_go(tmp_path):
    go = tmp_path / "go.md"
    go.write_text("Decision: `NO_GO_FIX_LABELS`\n", encoding="utf-8")
    env = os.environ.copy()
    env.update({"GO_NO_GO_REPORT": str(go), "EXECUTE": "0"})
    result = subprocess.run(
        ["bash", "scripts/risk_vla/round1/run_17_execute_r1_r2_only_after_go.sh"],
        cwd="/mnt/project/bit_drive_left_tail/VLA-AD",
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "R2 blocked" in result.stdout
    assert "run_14" not in result.stdout


def test_r1_r2_guard_allows_r1_only_without_executing_r2(tmp_path):
    go = tmp_path / "go.md"
    go.write_text("Decision: `GO_R1_ONLY`\n", encoding="utf-8")
    env = os.environ.copy()
    env.update({"GO_NO_GO_REPORT": str(go), "BIT_EXP_ROOT": str(tmp_path / "exp"), "EXECUTE": "0"})
    result = subprocess.run(
        ["bash", "scripts/risk_vla/round1/run_17_execute_r1_r2_only_after_go.sh"],
        cwd="/mnt/project/bit_drive_left_tail/VLA-AD",
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "ORACLE ROUTER IS ANALYSIS-ONLY" in result.stdout
    assert "R2 blocked by GO_R1_ONLY" in result.stdout
    assert "[run_14]" not in result.stdout
