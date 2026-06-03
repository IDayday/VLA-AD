from __future__ import annotations

from scripts.risk_vla.build_round1_execution_plan import build_plan, write_plan


def test_execution_plan_writes_dry_run_commands_only(tmp_path):
    report = {
        "found_inputs": {
            "a0_pdm_csv": "/abs/a0.csv",
            "bit_pdm_csv": "/abs/bit.csv",
            "chunk_cache_dir": "/abs/cache",
            "checkpoint_dir": "/abs/ckpt",
        },
        "missing_inputs": [],
    }
    plan = build_plan(report, tmp_path, max_samples=128, limit_train_batches=10, limit_val_batches=5)
    write_plan(plan, tmp_path)

    plan_md = (tmp_path / "round1_execution_plan.md").read_text(encoding="utf-8")
    assert "RISK-VLA Round 1 Execution Plan" in plan_md
    command_text = (tmp_path / "commands" / "04_train_r0_diagnostic.sh").read_text(encoding="utf-8")
    assert "EXECUTE='0'" in command_text
    assert "MAX_SAMPLES='128'" in command_text
    assert "run_10_execute_r0_diagnostic_small.sh" in command_text
