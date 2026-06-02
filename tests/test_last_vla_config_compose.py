from __future__ import annotations

from pathlib import Path

import yaml


def test_last_vla_yaml_configs_load_directly():
    config_dir = Path("configs/last_vla_v2")
    for path in (
        config_dir / "last_vla_cot_alignment.yaml",
        config_dir / "last_vla_progressive_bottleneck.yaml",
        config_dir / "last_vla_teacher_traj_sft.yaml",
        config_dir / "last_vla_progressive_bottleneck_eval.yaml",
        config_dir / "last_vla_teacher_traj_sft_eval.yaml",
    ):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["use_last_vla"] is True
        assert data["use_last_rd"] is False
        assert data["policy_kd_loss_weight"] == 0.0


def test_last_vla_hydra_experiment_yaml_loads():
    for name in (
        "last_vla_cot_alignment.yaml",
        "last_vla_progressive_bottleneck.yaml",
        "last_vla_teacher_traj_sft.yaml",
    ):
        path = Path("navsim/planning/script/config/experiment") / name
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["agent"]["use_last_vla"] is True
        assert data["agent"]["use_last_rd"] is False


def test_last_vla_hydra_compose_if_available():
    try:
        from hydra import compose, initialize_config_module
        from hydra.core.global_hydra import GlobalHydra
    except Exception:
        return
    GlobalHydra.instance().clear()
    with initialize_config_module(config_module="navsim.planning.script.config.training", version_base=None):
        cfg = compose(
            config_name="default_training",
            overrides=["train_test_split=trainval", "+experiment=last_vla_cot_alignment"],
        )
    assert cfg.agent.use_last_vla is True
    assert cfg.agent.last_vla_stage == "cot_alignment"
