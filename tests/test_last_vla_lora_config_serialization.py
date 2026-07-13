from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch

from scripts.build_recogdrive_hidden_cache_with_lora import _resolve_lora_runtime_config


def _adapter_dir(root: Path) -> Path:
    adapter = root / "vlm_lora"
    adapter.mkdir()
    config = {
        "preset": "attention_mlp",
        "scope": "llm",
        "r": 32,
        "alpha": 64,
        "dropout": 0.05,
        "bias": "none",
        "use_rslora": True,
        "use_dora": False,
        "resolved_target_modules": ["q_proj", "gate_proj"],
        "adapter_config_hash": "abc",
    }
    (adapter / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    (adapter / "lora_metadata.json").write_text(json.dumps(config), encoding="utf-8")
    torch.save({"base_model.model.q_proj.lora_A.default.weight": torch.zeros(1, 1)}, adapter / "adapter_model.bin")
    return adapter


def test_hidden_cache_regeneration_reads_adapter_dir_config(tmp_path: Path):
    adapter = _adapter_dir(tmp_path)
    args = argparse.Namespace(
        vlm_lora_adapter_dir=adapter,
        vlm_lora_adapter=None,
        lora_r=None,
        lora_alpha=None,
        lora_dropout=None,
        lora_target_modules=None,
        lora_use_rslora=None,
        lora_use_dora=None,
        allow_lora_config_override=False,
    )

    config, state_path, source = _resolve_lora_runtime_config(args)

    assert source == "adapter_dir"
    assert state_path == adapter / "adapter_model.bin"
    assert config["r"] == 32
    assert config["alpha"] == 64
    assert config["target_modules"] == ["q_proj", "gate_proj"]


def test_hidden_cache_regeneration_rejects_cli_config_mismatch(tmp_path: Path):
    adapter = _adapter_dir(tmp_path)
    args = argparse.Namespace(
        vlm_lora_adapter_dir=adapter,
        vlm_lora_adapter=None,
        lora_r=16,
        lora_alpha=None,
        lora_dropout=None,
        lora_target_modules=None,
        lora_use_rslora=None,
        lora_use_dora=None,
        allow_lora_config_override=False,
    )

    with pytest.raises(ValueError, match="does not match"):
        _resolve_lora_runtime_config(args)

    args.allow_lora_config_override = True
    config, _, _ = _resolve_lora_runtime_config(args)
    assert config["r"] == 32
