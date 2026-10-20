"""One improved step per scene; all other steps retain ordinary reference OPD."""
import torch
from .suffix_rollout import NativeAdapter


def objective(student, reference, tokens, inputs, noise, student_version, *,
              selected_target=None, selected_index=None, anchor=None, anchor_weight=0.):
    student_adapter = NativeAdapter(student)
    rollin = student_adapter.sample_rollin(tokens, inputs, noise, student_version)
    return objective_from_rollin(student, reference, rollin, tokens, inputs, student_version,
                                 selected_target=selected_target, selected_index=selected_index,
                                 anchor=anchor, anchor_weight=anchor_weight)


def objective_from_rollin(student, reference, rollin, tokens, inputs, student_version, *,
                          selected_target=None, selected_index=None, anchor=None, anchor_weight=0.):
    with torch.no_grad():
        ref = NativeAdapter(reference)
        context = ref.prepare_conditioning(tokens, inputs)
        targets = torch.stack([ref.transition_stats(rollin.states[:, i], i, context).mean for i in range(5)], 1)
        ordinary_targets = targets.clone()
        if selected_target is not None:
            if selected_index is None or len(selected_target) != len(tokens):
                raise ValueError('Selected targets must align with scenes/steps')
            targets[torch.arange(len(tokens), device=tokens.device), selected_index] = selected_target
        anchor_targets = None
        if anchor is not None and anchor_weight:
            aa = NativeAdapter(anchor)
            ac = aa.prepare_conditioning(tokens, inputs)
            anchor_targets = torch.stack([aa.transition_stats(rollin.states[:, i], i, ac).mean for i in range(5)], 1)
    prediction = NativeAdapter(student).evaluate_rollin(rollin, tokens, inputs, student_version)
    error = (prediction.float()-targets.float()).square()
    loss = error.mean()  # Exactly B*5*8*3, independent of selected/accepted counts.
    anchor_loss = (prediction.float()-anchor_targets.float()).square().mean() if anchor_targets is not None else loss*0.
    return dict(loss=loss+anchor_weight*anchor_loss, opd=loss, anchor=anchor_loss, rollin=rollin,
                predictions=prediction.detach(), targets=targets.detach(),
                ordinary_discrepancy=(prediction.detach().float()-ordinary_targets.float()).square().mean((2,3)),
                per_step=error.detach().mean((2,3)))
