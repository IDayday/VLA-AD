from __future__ import annotations

from argparse import Namespace
import sys
import yaml

from scripts.risk_vla import build_oracle_router_pilot_eval_command as oracle
from scripts.risk_vla import build_predicted_router_pilot_eval_command as predicted


def test_oracle_eval_command_sets_analysis_only_oracle_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--checkpoint",
            str(tmp_path / "ckpt.pt"),
            "--output-dir",
            str(tmp_path / "oracle"),
        ],
    )
    assert oracle.main() == 0
    output = capsys.readouterr().out
    assert "analysis-only" in output
    cfg = yaml.safe_load((tmp_path / "oracle" / "oracle_router_pilot_config_snapshot.yaml").read_text(encoding="utf-8"))
    assert cfg["risk_vla_use_oracle_router"] is True
    assert cfg["risk_vla_strategy_token_scale"] == 0.25


def test_predicted_eval_command_disables_oracle_flag(tmp_path):
    args = Namespace(
        checkpoint=tmp_path / "ckpt.pt",
        output_dir=tmp_path / "predicted",
        base_config=None,
        split="navval",
        navsim_root=tmp_path / "navsim",
        chunk_cache_dir=None,
        chunk_cache_root=None,
        metric_cache_dir=None,
        max_samples=64,
        precision="fp32",
        risk_loss_weight=0.0,
        extra_override=[],
        execute=False,
        dry_run=True,
    )
    snapshot, _, command = predicted.write_artifacts(args)
    cfg = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
    assert cfg["risk_vla_use_oracle_router"] is False
    assert "--max-samples" in command
    assert "64" in command
