from __future__ import annotations

from argparse import Namespace

import yaml

from scripts.risk_vla.run_risk_head_diagnostic_train import build_train_command, diagnostic_config, write_artifacts


def _args(tmp_path):
    return Namespace(
        cache_path=tmp_path / "cache",
        output_dir=tmp_path / "out",
        checkpoint_path=tmp_path / "ckpt.pt",
        max_epochs=1,
        max_samples=32,
        limit_train_batches=5,
        limit_val_batches=None,
        num_workers=2,
        gpus=None,
        devices=None,
        master_port=29541,
        risk_loss_weight=0.05,
        extra_override=[],
        dry_run=True,
        execute=False,
    )


def test_diagnostic_train_command_disables_strategy_conditioning(tmp_path):
    args = _args(tmp_path)
    cfg = diagnostic_config(args)
    assert cfg["use_risk_vla"] is True
    assert cfg["risk_vla_strategy_token_scale"] == 0.0
    assert cfg["risk_vla_horizon_residual_scale"] == 0.0
    assert cfg["risk_vla_use_oracle_router"] is False
    assert cfg["risk_vla_risk_loss_weight"] > 0.0

    snapshot, command_path, command = write_artifacts(args)
    loaded = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
    assert loaded["risk_vla_strategy_token_scale"] == 0.0
    assert "scripts/train_bit_drive_chunked.py" in command
    assert command_path.is_file()
