import torch

from navsim.agents.recogdrive.risk_vla.safe_alignment_losses import safe_alignment_total_loss


def test_safe_alignment_losses_mask_and_margin():
    pred = torch.zeros(2, 8, 3)
    positive = torch.ones(2, 8, 3) * 0.1
    negative = torch.ones(2, 8, 3)
    out = safe_alignment_total_loss(pred, positive, negative, safety_mask=torch.tensor([True, False]))
    assert out["safealign_total_loss"].ndim == 0
    assert out["safealign_negative_margin_loss"].item() == 0.0
