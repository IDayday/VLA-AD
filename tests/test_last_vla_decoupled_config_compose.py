from __future__ import annotations

from pathlib import Path

import yaml


def _assert_decoupled(agent) -> None:
    assert agent["use_last_vla"] is True
    assert agent["use_last_rd"] is False
    assert agent["use_expert_features"] is False
    assert agent["use_vggt"] is False
    assert agent["last_vla_condition_mode"] == "decoupled_cot_residual"
    assert agent["last_vla_cot_bottleneck_mode"] is False
    assert agent["last_vla_raw_vlm_context_to_dit"] is True
    assert "last_vla_vlm_summary_tokens" not in agent
    assert agent["last_vla_use_risk_head"] is False
    assert agent["num_risk_tokens"] == 0
    assert agent["last_vla_cot_num_tokens"] == 192
    assert agent["num_jepa_tokens"] == 128
    assert agent["num_dynamic_tokens"] == 128
    assert agent["num_geometry_tokens"] == 192
    assert agent["last_vla_geometry_teacher_dim"] == 512
    assert agent["last_vla_geometry_grid_rows"] == 12
    assert agent["last_vla_geometry_grid_cols"] == 16
    assert agent["last_vla_residual_anchor_source"] == "vlm_text_traj"


def test_decoupled_yaml_base_has_formal_values():
    base = yaml.safe_load(Path("configs/last_vla_v2/decoupled_highcap_no_risk/base.yaml").read_text(encoding="utf-8"))
    _assert_decoupled(base)


def test_decoupled_hydra_experiments_have_formal_values():
    exp_dir = Path("navsim/planning/script/config/experiment")
    for name in (
        "last_vla_decoupled_cot_alignment_highcap_no_risk.yaml",
        "last_vla_decoupled_progressive_highcap_no_risk.yaml",
    ):
        payload = yaml.safe_load((exp_dir / name).read_text(encoding="utf-8"))
        _assert_decoupled(payload["agent"])
        assert payload["agent"]["last_vla_use_residual_diffusion"] is False


def test_decoupled_vlm_text_residual_experiment_is_separate_from_current_ab_configs():
    exp_dir = Path("navsim/planning/script/config/experiment")
    payload = yaml.safe_load(
        (exp_dir / "last_vla_decoupled_progressive_highcap_no_risk_vlm_text_residual.yaml").read_text(encoding="utf-8")
    )
    _assert_decoupled(payload["agent"])
    assert payload["agent"]["last_vla_use_residual_diffusion"] is True
    assert payload["agent"]["last_vla_residual_anchor_source"] == "vlm_text_traj"
    assert payload["agent"]["last_vla_require_residual_anchor"] is True
    assert payload["agent"]["last_vla_residual_alpha_start"] == 1.0
    assert payload["agent"]["last_vla_residual_alpha_end"] == 1.0
    assert payload["agent"]["last_vla_residual_alpha_warmup_epochs"] == 0


def test_decoupled_hydra_compose_if_available():
    try:
        from hydra import compose, initialize_config_module
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf
    except Exception:
        return
    for experiment in (
        "last_vla_decoupled_cot_alignment_highcap_no_risk",
        "last_vla_decoupled_progressive_highcap_no_risk",
        "last_vla_decoupled_vlm_lora_cot_alignment_highcap_no_risk",
        "last_vla_decoupled_progressive_highcap_no_risk_vlm_text_residual",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_module(config_module="navsim.planning.script.config.training", version_base=None):
            cfg = compose(config_name="default_training", overrides=["train_test_split=trainval", f"+experiment={experiment}"])
        _assert_decoupled(OmegaConf.to_container(cfg.agent, resolve=True))
