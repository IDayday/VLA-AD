from __future__ import annotations

import torch

from navsim.agents.recogdrive.expert_cache import atomic_torch_save, write_index
from scripts.last_vla_v2.two_expert_slot.run_two_expert_vlm_sft import TwoExpertStage1Dataset


def _path_tensor(text: str) -> torch.Tensor:
    return torch.tensor([ord(ch) for ch in text] + [0], dtype=torch.long)


def test_stage1_dataset_loads_replay_fields(tmp_path):
    token = "sample_a"
    base = tmp_path / "base"
    jepa = tmp_path / "jepa"
    vggt = tmp_path / "vggt"
    replay = tmp_path / "replay"
    for root in (base, jepa, vggt, replay):
        (root / "samples").mkdir(parents=True)
    atomic_torch_save(
        {
            "image_path_tensor": _path_tensor("/tmp/image.jpg"),
            "status_feature": torch.zeros(8),
            "high_command_one_hot": torch.tensor([0.0, 1.0, 0.0]),
            "history_trajectory": torch.zeros(4, 3),
            "trajectory": torch.zeros(8, 3),
        },
        base / "samples" / f"{token}.pt",
    )
    atomic_torch_save(
        {
            "jepa_dynamic_teacher_tokens": torch.zeros(3, 12, 1024),
            "jepa_dynamic_teacher_metadata": {"strict_dynamic_teacher": True},
        },
        jepa / "samples" / f"{token}.pt",
    )
    atomic_torch_save(
        {
            "vggt_feature23_tokens": torch.zeros(12, 16),
            "vggt_feature23_metadata": {"strict_geometry_teacher": True, "feature_dim": 16},
        },
        vggt / "samples" / f"{token}.pt",
    )
    atomic_torch_save(
        {
            "prompt": "<image>\nplan",
            "answer_text": "Here is the planning trajectory [PT, " + ", ".join(["(+0.0, +0.0, +0.0)"] * 8) + "].",
            "replay_source": "official_navsim_traj",
            "official_recogdrive_stage1": True,
            "parse_ok": True,
        },
        replay / "samples" / f"{token}.pt",
    )
    for root in (base, jepa, vggt, replay):
        write_index(root, [{"sample_token": token, "path": f"samples/{token}.pt"}])
    dataset = TwoExpertStage1Dataset(
        base,
        {token: jepa / "samples" / f"{token}.pt"},
        {token: vggt / "samples" / f"{token}.pt"},
        replay_index={token: replay / "samples" / f"{token}.pt"},
        require_replay=True,
    )
    item = dataset[0]
    assert item["sample_token"] == token
    assert item["replay_prompt"].startswith("<image>")
    assert item["replay_answer_text"]
    assert item["replay_official_recogdrive_stage1"] is True


def test_stage1_replay_only_dataset_derives_context_from_prompt(tmp_path):
    token = "sample_replay_only"
    jepa = tmp_path / "jepa"
    vggt = tmp_path / "vggt"
    replay = tmp_path / "replay"
    for root in (jepa, vggt, replay):
        (root / "samples").mkdir(parents=True)
    atomic_torch_save(
        {
            "jepa_dynamic_teacher_tokens": torch.zeros(3, 12, 1024),
            "jepa_dynamic_teacher_metadata": {"strict_dynamic_teacher": True},
        },
        jepa / "samples" / f"{token}.pt",
    )
    atomic_torch_save(
        {
            "vggt_feature23_tokens": torch.zeros(12, 16),
            "vggt_feature23_metadata": {"strict_geometry_teacher": True, "feature_dim": 16},
        },
        vggt / "samples" / f"{token}.pt",
    )
    prompt = (
        "<image>\nHistorical motion context (last 4 timesteps):"
        "   - t-3: (0.0, 0.0, 0.0)"
        "   - t-2: (1.0, 0.0, 0.0)"
        "   - t-1: (3.0, 1.0, 0.1)"
        "   - t-0: (6.0, 1.5, 0.1)\n"
        "Active navigation command: [TURN RIGHT]"
    )
    atomic_torch_save(
        {
            "image_path": "/tmp/image.jpg",
            "prompt": prompt,
            "answer_text": "Here is the planning trajectory [PT, " + ", ".join(["(+0.0, +0.0, +0.0)"] * 8) + "].",
            "history_trajectory": None,
            "high_command_one_hot": None,
            "status_feature": None,
            "replay_source": "official_navsim_traj",
            "official_recogdrive_stage1": True,
            "parse_ok": True,
        },
        replay / "samples" / f"{token}.pt",
    )
    for root in (jepa, vggt, replay):
        write_index(root, [{"sample_token": token, "path": f"samples/{token}.pt"}])
    dataset = TwoExpertStage1Dataset(
        tmp_path / "missing_base",
        {token: jepa / "samples" / f"{token}.pt"},
        {token: vggt / "samples" / f"{token}.pt"},
        replay_index={token: replay / "samples" / f"{token}.pt"},
        require_replay=True,
        allow_replay_only_base=True,
    )

    item = dataset[0]

    assert item["history_trajectory"].shape == (4, 3)
    assert torch.allclose(item["history_trajectory"][-1], torch.tensor([6.0, 1.5, 0.1]))
    assert torch.allclose(item["high_command_one_hot"], torch.tensor([0.0, 0.0, 1.0]))
    assert item["status_feature"].shape == (8,)
    assert torch.allclose(item["status_feature"][:3], torch.tensor([0.0, 0.0, 1.0]))
    assert not torch.allclose(item["status_feature"][3:], torch.zeros(5))
    assert item["trajectory"].shape == (8, 3)
