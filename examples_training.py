"""Loss helpers for the receiving trainer; no training is launched here.

The caller chooses parameters, LoRA/full update, optimizer and trajectory loss.
These helpers deliberately make no claim to be a validated distillation recipe.
"""
import torch


def representation_loss(student_post,teacher_post,image_mask,text_mask):
    """Mean FP32 MSE over valid tokens, excluding native padding."""
    valid=(image_mask|text_mask).bool()
    if valid.ndim==1:valid=valid[None]
    target=teacher_post.detach().float()
    if student_post.shape!=target.shape:raise ValueError('Token layouts and interface shapes must match')
    return (student_post.float()-target).square()[valid].mean()


def build_teacher_target(frozen_Base_post,image_mask,text_mask,token,library,decisions):
    """Generate from a separately frozen RL Base, never the changing student."""
    with torch.no_grad():
        return library.selected_target(frozen_Base_post,image_mask,text_mask,token,decisions).detach()


def delta_prediction_loss(predicted_delta,actual_teacher_shift,Base_post,image_mask,text_mask):
    """Optional compact corrector target, [1,384] shared over valid tokens.

    The target is the applied/capped shift, not the uncapped requested shift.
    The all-token loss above also captures tiny token-specific FP32 rounding.
    """
    del Base_post,image_mask,text_mask
    if actual_teacher_shift.shape[-2:]!=(2,384):raise ValueError('Expected applied image/text shifts')
    if not torch.equal(actual_teacher_shift[...,0,:],actual_teacher_shift[...,1,:]):raise ValueError('This helper only covers the current shared384 banks')
    return torch.nn.functional.mse_loss(predicted_delta.float(),actual_teacher_shift[...,0,:].detach().float())
