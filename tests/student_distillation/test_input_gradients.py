import json
import pytest
import torch
from torch import nn

from student_distillation.losses import scene_loss, transition_weight


class TinyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(1, 384))
    def forward(self, post, image, text, state):
        valid = image | text
        output = torch.where(valid.unsqueeze(-1), post + self.shift.unsqueeze(1), post)
        return {'post': output, 'actual_rms': self.shift.detach().square().mean().sqrt()}


class RecordingBridge:
    def __init__(self):
        self.calls = []
        self.rollouts = []
    def encode_context(self, *args):
        return None
    def states_from_post(self, post, context, tape, stop_before):
        assert not post.requires_grad and not torch.is_grad_enabled()
        self.rollouts.append(post.clone())
        return torch.full((1, 8, 3), float(post.mean()))
    def mean_from_post(self, actions, step_index, post, context):
        self.calls.append((actions, step_index, post.requires_grad, torch.is_grad_enabled()))
        mean = actions + post.mean().expand_as(actions)
        return mean, torch.zeros_like(actions), mean


def batch():
    return {'base_post': torch.ones(1, 4, 384, requires_grad=True),
            'image_mask': torch.tensor([[False, True, True, False]]),
            'text_mask': torch.tensor([[False, False, False, True]]),
            'teacher_actual_shift': torch.full((1, 384), 0.05, requires_grad=True),
            'state24': torch.zeros(1, 24), 'history12': torch.zeros(1, 12),
            'status8': torch.zeros(1, 8), 'time_index': 2, 'noise_bank': 1,
            'noise_tape': torch.zeros(6, 1, 8, 3)}


CALIBRATION = {'h_mse_scale': 0.01, 'mu_mse_scale': [0.02] * 5}


@pytest.mark.parametrize('experiment', ['H_full', 'H_gain1'])
def test_a_b_never_call_bridge_even_after_curriculum(experiment):
    class ForbiddenBridge:
        def __getattr__(self, name):
            raise AssertionError('A/B invoked bridge')
    adapter = TinyAdapter()
    cfg = {'experiment': experiment, 'enable_transition_loss': False, 'state_source': 'none'}
    loss, stats = scene_loss(batch(), adapter, ForbiddenBridge(), cfg, 53639, CALIBRATION)
    loss.backward()
    assert adapter.shift.grad.norm() > 0
    assert not stats['transition_executed']
    json.dumps(stats)
    with pytest.raises(ValueError, match='A/B'):
        transition_weight(dict(cfg, enable_transition_loss=True), 53639)


@pytest.mark.parametrize('source', ['teacher', 'student'])
def test_same_state_and_time_teacher_detach_and_live_student_gradient(source):
    adapter, bridge, item = TinyAdapter(), RecordingBridge(), batch()
    cfg = {'experiment': 'H_mu_' + source, 'enable_transition_loss': True, 'state_source': source}
    loss, stats = scene_loss(item, adapter, bridge, cfg, 9000, CALIBRATION)
    loss.backward()
    assert adapter.shift.grad.norm() > 0
    assert item['base_post'].grad is None and item['teacher_actual_shift'].grad is None
    teacher, student = bridge.calls
    assert teacher[0] is student[0] and teacher[1] == student[1] == 2
    assert not teacher[0].requires_grad
    assert teacher[2:] == (False, False)
    assert student[2:] == (True, True)
    assert stats['transition_executed']
    json.dumps(stats)
    if source == 'student':
        before = bridge.rollouts[-1]
        with torch.no_grad():
            adapter.shift.add_(0.1)
        scene_loss(item, adapter, bridge, cfg, 9001, CALIBRATION)
        assert not torch.equal(bridge.rollouts[-1], before)


def test_h_warmup_never_generates_states_and_curriculum_boundaries():
    cfg = {'experiment': 'H_mu_student', 'enable_transition_loss': True, 'state_source': 'student'}
    assert transition_weight(cfg, 5363) == 0
    assert transition_weight(cfg, 5364) == 0
    assert transition_weight(cfg, 6705) == 0.25
    assert transition_weight(cfg, 8046) == 0.5
    bridge = RecordingBridge()
    scene_loss(batch(), TinyAdapter(), bridge, cfg, 5364, CALIBRATION)
    assert bridge.calls == bridge.rollouts == []


def test_invalid_scale_rejected_and_zero_target_uses_fixed_h_scale():
    cfg = {'experiment': 'H_gain1', 'enable_transition_loss': False, 'state_source': 'none'}
    with pytest.raises(ValueError, match='h_mse_scale'):
        scene_loss(batch(), TinyAdapter(), None, cfg, 0, {'h_mse_scale': 0})
    adapter, item = TinyAdapter(), batch()
    item['teacher_actual_shift'] = torch.zeros(1, 384)
    with torch.no_grad():
        adapter.shift.fill_(0.1)
    loss, stats = scene_loss(item, adapter, None, cfg, 0, CALIBRATION)
    assert loss.item() == pytest.approx(1, abs=2e-6)
    assert stats['target_is_zero']


def test_c_cache_wiring_keeps_loss_gradient_and_d_refresh_unchanged(monkeypatch):
    import sys
    from types import SimpleNamespace
    import student_distillation.losses as objectives
    saved = {}
    cache_calls = []
    options = {'identity': {'teacher_bundle_sha256': 'fixture'}, 'max_entries': 10, 'receipt': {}}
    def settings(cfg):
        return options if cfg['experiment'] == 'H_mu_teacher' and cfg.get('performance', {}).get('frozen_teacher_query_cache') else None
    monkeypatch.setattr(objectives, 'teacher_query_cache_settings', settings)
    def cached(bridge, post, context, tape, j, **kwargs):
        assert not torch.is_grad_enabled() and not post.requires_grad
        key = (kwargs['token'], kwargs['training_seed'], kwargs['noise_bank'])
        hit = key in saved
        if not hit:
            state = bridge.states_from_post(post, context, tape, stop_before=j).detach()
            teacher_mean = bridge.mean_from_post(state, j, post, context)[0].detach()
            saved[key] = (state, teacher_mean)
        cache_calls.append(hit)
        return *saved[key], hit
    monkeypatch.setitem(sys.modules, 'student_distillation.teacher_query_cache', SimpleNamespace(cached_teacher_query=cached))
    item = batch()
    item.update(token='fixture_scene', metadata={'actual_Base_interface_sha256': 'immutable_fixture'})
    plain = {'experiment': 'H_mu_teacher', 'enable_transition_loss': True, 'state_source': 'teacher', 'training_seed': 20261027}
    enabled = dict(plain, performance={'frozen_teacher_query_cache': True})
    adapter = TinyAdapter()
    for value in (0., .1):
        with torch.no_grad():
            adapter.shift.fill_(value)
        adapter.zero_grad(set_to_none=True)
        reference, _ = scene_loss(item, adapter, RecordingBridge(), plain, 9000, CALIBRATION)
        reference.backward()
        gradient = adapter.shift.grad.clone()
        adapter.zero_grad(set_to_none=True)
        cached_loss, stats = scene_loss(item, adapter, RecordingBridge(), enabled, 9000, CALIBRATION)
        cached_loss.backward()
        assert torch.equal(reference, cached_loss)
        assert torch.equal(gradient, adapter.shift.grad)
        assert stats['teacher_query_cache_hit'] is (value == .1)
    assert cache_calls == [False, True]
    student_cfg = dict(enabled, experiment='H_mu_student', state_source='student')
    bridge = RecordingBridge()
    scene_loss(item, adapter, bridge, student_cfg, 9000, CALIBRATION)
    with torch.no_grad():
        adapter.shift.add_(.1)
    scene_loss(item, adapter, bridge, student_cfg, 9001, CALIBRATION)
    assert len(bridge.rollouts) == 2 and not torch.equal(*bridge.rollouts)
    assert cache_calls == [False, True]
