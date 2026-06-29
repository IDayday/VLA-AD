from argparse import Namespace
from pathlib import Path

import torch
import yaml

from scripts.onevl.audit_onevl_stage2_contract import run_audit, write_markdown


def write_minimal_chunk(root: Path) -> None:
    samples = root / "samples"
    samples.mkdir(parents=True)
    sample_path = samples / "sample.pt"
    payload = {
        "last_hidden_state": torch.randn(6, 4),
        "history_trajectory": torch.zeros(4, 3),
        "high_command_one_hot": torch.tensor([0.0, 1.0, 0.0]),
        "status_feature": torch.tensor([0.0, 0.0, 0.0, 1.0, 3.0, 0.1, 0.2, 0.3]),
        "trajectory": torch.tensor(
            [
                [1.0, 0.0, 0.01],
                [2.0, 0.1, 0.02],
                [3.0, 0.1, 0.03],
                [4.0, 0.1, 0.04],
                [5.0, 0.2, 0.05],
                [6.0, 0.2, 0.06],
                [7.0, 0.2, 0.07],
                [8.0, 0.3, 0.08],
            ],
            dtype=torch.float32,
        ),
        "support_trajectories": torch.stack([torch.zeros(8, 3), torch.ones(8, 3), torch.full((8, 3), 2.0)]),
        "support_mask": torch.tensor([True, True, False]),
        "support_weights": torch.tensor([0.7, 0.3, 0.0]),
        "support_scores": torch.tensor([0.9, 0.8, 0.0]),
        "support_missing_mask": torch.tensor(False),
        "meta": {
            "row_image_count": 2,
            "row_image_first": "old.jpg",
            "row_image_last": "current.jpg",
            "current_image_policy": "first",
            "current_image_selected": "old.jpg",
            "token": "token-1",
            "support_index_token": "token-1",
            "prompt_source": "row",
            "prompt_command": "MOVE FORWARD",
            "scene_command": "MOVE FORWARD",
            "prompt_scene_alignment_pass": True,
            "status_policy": "navsim_command3_velocity2_acceleration3",
            "high_command_policy": "raw_command_first3",
            "raw_command_shape": [3],
            "target_source": "navsim_future_trajectory",
            "valid_hidden_length": 6,
            "input_token_count": 6,
            "hidden_source": "synthetic",
            "hidden_layer": "final",
            "hidden_extraction_mode": "unit_test",
            "model_checkpoint": "unit",
            "processor_checkpoint": "unit",
            "prompt_template_hash": "hash",
        },
    }
    torch.save(payload, sample_path)
    (root / "index.jsonl").write_text('{"path": "samples/sample.pt", "sample_token": "token-1"}\n', encoding="utf-8")


def test_contract_audit_smoke(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"vlm_feature_dim": 4}), encoding="utf-8")
    cache_root = tmp_path / "cache"
    write_minimal_chunk(cache_root)

    args = Namespace(
        config=config_path,
        train_cache_root=cache_root,
        val_cache_root=None,
        nav_cache_root=None,
        chunk_name_pattern="shard_*",
        val_chunk_name_pattern="val6000_chunk_*",
        max_samples_per_split=16,
        output_json=tmp_path / "audit.json",
        output_md=tmp_path / "audit.md",
        support_index_path=None,
        data_jsonl=None,
        nav_json=None,
    )
    payload = run_audit(args)
    write_markdown(args.output_md, payload)

    assert payload["splits"]["train"]["sample_count"] == 1
    assert payload["splits"]["train"]["hidden"]["token_length"]["max"] == 6.0
    assert "trajectory_norm_odo" in payload["splits"]["train"]["trajectory_support"]
    assert args.output_md.is_file()
