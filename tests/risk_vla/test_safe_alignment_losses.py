import json

import numpy as np
import torch

from navsim.agents.recogdrive.risk_vla.safe_alignment_losses import safe_alignment_total_loss
from scripts.risk_vla.train_safe_alignment_sft import main as train_safe_alignment_main


def test_safe_alignment_losses_mask_and_margin():
    pred = torch.zeros(2, 8, 3)
    positive = torch.ones(2, 8, 3) * 0.1
    negative = torch.ones(2, 8, 3)
    out = safe_alignment_total_loss(pred, positive, negative, safety_mask=torch.tensor([True, False]))
    assert out["safealign_total_loss"].ndim == 0
    assert out["safealign_negative_margin_loss"].item() == 0.0


def test_train_safe_alignment_sft_dry_run(tmp_path, capsys):
    np.savez_compressed(
        tmp_path / "candidate_trajectories.npz",
        sample_tokens=np.array(["a"]),
        trajectories=np.zeros((1, 2, 8, 3), dtype=np.float32),
    )
    pair = {
        "sample_token": "a",
        "positive_candidate_id": "1",
        "negative_candidate_id": "0",
        "negative_regresses_nc": True,
        "negative_regresses_ttc": False,
        "positive_repairs_path": True,
        "tail_risk_label": True,
    }
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps(pair) + "\n", encoding="utf-8")
    assert train_safe_alignment_main(
        [
            "--candidate-npz",
            str(tmp_path / "candidate_trajectories.npz"),
            "--pairs-jsonl",
            str(pairs),
            "--output-dir",
            str(tmp_path / "safealign"),
            "--dry-run",
        ]
    ) == 0
    assert '"pairs": 1' in capsys.readouterr().out
