from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _env(tmp_path: Path) -> dict[str, str]:
    ckpt = tmp_path / "a0.ckpt"
    cache = tmp_path / "cache"
    vlm = tmp_path / "vlm"
    ckpt.write_bytes(b"dummy")
    vlm.write_text("dummy", encoding="utf-8")
    cache.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "FULL_GEOMETRY_CHUNK_ROOT": str(cache),
            "A0_INIT_CHECKPOINT": str(ckpt),
            "OUT_ROOT": str(tmp_path / "out"),
            "MASTER_PORT": "29990",
            "VLM_PATH": str(vlm),
            "NAVSIM_LOG_PATH": str(tmp_path / "navsim_logs"),
            "SENSOR_BLOBS_PATH": str(tmp_path / "sensor_blobs"),
            "PYTHON_BIN": "python",
            "TORCHRUN_BIN": "/bin/false",
        }
    )
    return env


def test_round2_launchers_dryrun_do_not_launch_training(tmp_path: Path):
    env = _env(tmp_path)
    subprocess.run(["bash", "scripts/last_vla_v2/round2/serverA_frozen_vlm_full_sft.sh"], env=env, check=True)
    env["RUN_TRAIN"] = "0"
    subprocess.run(["bash", "scripts/last_vla_v2/round2/serverB_vlm_lora_full_sft.sh"], env=env, check=True)

    logs = [
        tmp_path / "out" / "serverA_frozen" / "commands.log",
        tmp_path / "out" / "serverB_lora" / "commands.log",
    ]
    for path in logs:
        text = path.read_text(encoding="utf-8")
        assert "agent.last_vla_adapter_checkpoint=" in text
        assert "++agent.last_vla_adapter_checkpoint" not in text
        assert "+experiment=last_vla" in text


def test_vlm_lora_alignment_launcher_dryrun(tmp_path: Path):
    env = _env(tmp_path)
    env["OUTPUT_DIR"] = str(tmp_path / "out" / "vlm_lora_single")
    env["TRAIN_CHUNK_CACHE_ROOT"] = env["FULL_GEOMETRY_CHUNK_ROOT"]
    subprocess.run(["bash", "scripts/last_vla_v2/run_vlm_lora_cot_alignment_8gpu.sh"], env=env, check=True)
    text = (Path(env["OUTPUT_DIR"]) / "commands.log").read_text(encoding="utf-8")
    assert "+experiment=last_vla_vlm_lora_cot_alignment" in text
    assert "agent.cache_hidden_state=false" in text
    assert "agent.expert_cache_dir=" in text
