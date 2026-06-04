from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch


def test_lora_adapter_extract_writes_state_and_metadata(tmp_path: Path):
    ckpt = tmp_path / "model.ckpt"
    torch.save(
        {
            "state_dict": {
                "agent.action_head.last_vla_cot.foo": torch.ones(1),
                "agent.backbone.model.base_model.model.layers.0.q_proj.lora_A.default.weight": torch.ones(1, 1),
            }
        },
        ckpt,
    )
    out = tmp_path / "adapters"

    subprocess.run(
        [
            sys.executable,
            "scripts/last_vla_v2/extract_vlm_lora_and_cot_adapters.py",
            "--checkpoint",
            str(ckpt),
            "--output-dir",
            str(out),
            "--base-vlm-path",
            "/tmp/vlm",
            "--preset",
            "attention_mlp",
            "--scope",
            "llm",
            "--r",
            "32",
            "--alpha",
            "64",
            "--dropout",
            "0.05",
        ],
        check=True,
    )

    metadata = json.loads((out / "vlm_lora" / "lora_metadata.json").read_text(encoding="utf-8"))
    assert (out / "last_vla_cot_adapter.pt").is_file()
    assert (out / "vlm_lora_adapter_state.pt").is_file()
    assert (out / "vlm_lora" / "adapter_model.bin").is_file()
    assert metadata["r"] == 32
    assert metadata["alpha"] == 64
    assert metadata["preset"] == "attention_mlp"
    assert metadata["scope"] == "llm"


def test_lora_adapter_extract_rejects_training_config_mismatch(tmp_path: Path):
    ckpt = tmp_path / "latest.ckpt"
    torch.save(
        {
            "state_dict": {
                "agent.action_head.last_vla_cot.foo": torch.ones(1),
                "agent.backbone.model.base_model.model.layers.0.q_proj.lora_A.default.weight": torch.ones(1, 1),
            }
        },
        ckpt,
    )
    (tmp_path / "lora_training_config.json").write_text(
        json.dumps(
            {
                "preset": "all_linear",
                "scope": "llm",
                "r": 64,
                "alpha": 128,
                "dropout": 0.05,
                "bias": "none",
                "use_rslora": True,
                "use_dora": False,
                "target_modules": ["llm.layers.0.q_proj"],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/last_vla_v2/extract_vlm_lora_and_cot_adapters.py",
            "--checkpoint",
            str(ckpt),
            "--output-dir",
            str(tmp_path / "bad"),
            "--preset",
            "attention_mlp",
            "--scope",
            "llm",
            "--r",
            "32",
            "--alpha",
            "64",
        ],
        text=True,
        stderr=subprocess.PIPE,
    )

    assert result.returncode != 0
    assert "do not match" in result.stderr
