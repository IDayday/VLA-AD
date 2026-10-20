"""Exact ordinary fallback and global loss normalization, including mixed batches."""
from types import SimpleNamespace
import importlib
import torch
import pytest


class Adapter:
    def __init__(self,head):self.head=head
    def prepare_conditioning(self,tokens,inputs):return {}
    def transition_stats(self,state,index,context):return SimpleNamespace(mean=state+index*.1)
    def evaluate_rollin(self,rollin,tokens,inputs,version):return self.head.weight.expand(len(tokens),5,8,3)


def setup(monkeypatch):
    module=importlib.import_module('navsim.agents.recogdrive.process_improvement.objective')
    monkeypatch.setattr(module,'NativeAdapter',Adapter)
    head=SimpleNamespace(weight=torch.nn.Parameter(torch.tensor(.3)))
    return module,head,SimpleNamespace(states=torch.zeros(2,6,8,3)),torch.zeros(2,1,1)


def test_all_rejected_exact_loss_and_gradient_despite_bad_reference_rounding(monkeypatch):
    module,head,rollin,tokens=setup(monkeypatch)
    ordinary=module.objective_from_rollin(head,object(),rollin,tokens,{},0)
    ordinary['loss'].backward();gradient=head.weight.grad.clone();head.weight.grad=None
    fallback=module.objective_from_rollin(head,object(),rollin,tokens,{},0,
        selected_target=torch.full((2,8,3),100.),selected_index=torch.tensor([1,4]),selected_mask=torch.tensor([False,False]))
    fallback['loss'].backward()
    assert torch.equal(ordinary['loss'],fallback['loss']) and torch.equal(ordinary['targets'],fallback['targets'])
    assert torch.equal(gradient,head.weight.grad)


def test_mixed_accepted_targets_keep_all_steps_global_normalizer(monkeypatch):
    module,head,rollin,tokens=setup(monkeypatch)
    result=module.objective_from_rollin(head,object(),rollin,tokens,{},0,
        selected_target=torch.full((2,8,3),2.),selected_index=torch.tensor([1,4]),selected_mask=torch.tensor([True,False]))
    expected=torch.arange(5)[None,:,None,None].expand(2,5,8,3)*.1;expected=expected.clone();expected[0,1]=2.
    assert torch.equal(result['targets'],expected)
    assert torch.equal(result['loss'],(head.weight-expected).square().sum()/(2*5*8*3))


def test_mask_shape_is_not_silently_broadcast(monkeypatch):
    module,head,rollin,tokens=setup(monkeypatch)
    with pytest.raises(ValueError):module.objective_from_rollin(head,object(),rollin,tokens,{},0,
        selected_target=torch.zeros(2,8,3),selected_index=torch.tensor([1,4]),selected_mask=torch.tensor([False]))
