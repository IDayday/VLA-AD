"""Exercise verbatim frozen native methods with small differentiable modules.

This is an engineering unit test, not real-checkpoint/GPU qualification.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn

from noise_replay import replay_bank
from student_distillation.native_bridge import NativeDDIMBridge
from student_distillation.noise import make_noise_tape


def native_methods():
    path = Path(__file__).parents[2] / 'vendor/upstream/model/recogdrive_diffusion_planner.py'
    module = ast.parse(path.read_text())
    source_class = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'ReCogDriveDiffusionPlanner')
    names = {'get_action', 'p_mean_variance', 'extract', 'make_timesteps', 'denorm_odo'}
    functions = [n for n in source_class.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {'torch': torch, 'BatchFeature': lambda data: data}
    ast_module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + functions,
                            type_ignores=[])
    exec(compile(ast.fix_missing_locations(ast_module), str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}


class ActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 384)
    def forward(self, x, t):
        return self.projection(x) + t[:, None, None].float() * 0.0001


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.01))
    def forward(self, hidden_states, encoder_hidden_states, conditioning_features, timesteps):
        return self.scale * (hidden_states + encoder_hidden_states.mean(1).unsqueeze(1) + conditioning_features.unsqueeze(1))


class Eta(nn.Module):
    def forward(self, x):
        return torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)


class SmallNativeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(sampling_method='ddim', action_horizon=8, action_dim=3)
        self.ddim_steps = 5
        self.register_buffer('ddim_t', torch.tensor([80, 60, 40, 20, 0]))
        self.register_buffer('ddim_alphas', torch.tensor([0.15, 0.35, 0.55, 0.75, 0.99]))
        self.register_buffer('ddim_alphas_prev', torch.tensor([0.35, 0.55, 0.75, 0.99, 1.0]))
        self.register_buffer('ddim_sqrt_one_minus_alphas', (1 - self.ddim_alphas).sqrt())
        self.feature_encoder = nn.Linear(1536, 384)
        self.his_traj_encoder = nn.Linear(12, 384)
        self.ego_status_encoder = nn.Linear(8, 384)
        self.action_encoder = ActionEncoder()
        self.fusion_projector = nn.Linear(1152, 384)
        self.action_decoder = nn.Linear(384, 3)
        self.model = Model()
        self.eta = Eta()
        self.eval_min_sampling_denoising_std = 1e-4
        self.eval_randn_clip_value = 1.0
        self.final_action_clip_value = 1.0


for _name, _method in native_methods().items():
    setattr(SmallNativeHead, _name, _method)


def fixture():
    torch.manual_seed(100)
    torch.set_num_threads(2)
    head = SmallNativeHead()
    bridge = NativeDDIMBridge(head)
    hidden = torch.randn(1, 11, 1536)
    history, status = torch.randn(1, 12), torch.randn(1, 8)
    return head, bridge, hidden, history, status


@pytest.mark.parametrize('teacher_shift', [0.0, 0.03])
def test_verbatim_native_get_action_equals_bridge_and_every_transition(teacher_shift):
    head, bridge, hidden, history, status = fixture()
    post = head.feature_encoder(hidden).detach() + teacher_shift
    tape = make_noise_tape(184, 'cpu')
    captured = []
    original = head.p_mean_variance
    def capture(actions, *args, **kwargs):
        outputs = original(actions, *args, **kwargs)
        captured.append((actions.clone(), tuple(o.clone() for o in outputs)))
        return outputs
    head.p_mean_variance = capture
    hook = head.feature_encoder.register_forward_hook(lambda module, args, output: post)
    try:
        with torch.no_grad(), replay_bank(tape):
            expected = head.get_action(hidden, SimpleNamespace(his_traj=history, status_feature=status))['pred_traj']
    finally:
        hook.remove()
    head.p_mean_variance = original
    context = bridge.encode_context(history, status)
    with torch.no_grad():
        result = bridge.states_from_post(post, context, tape)
    assert torch.equal(expected, result['trajectory'])
    for index, (actions, outputs) in enumerate(captured):
        assert torch.equal(actions, result['states'][index])
        assert torch.equal(outputs[0], result['means'][index])
        assert torch.equal(outputs[1], result['logvars'][index])
        assert torch.equal(bridge.states_from_post(post, context, tape, stop_before=index), actions)
    assert (result['states'][0].abs() > 1).any()
    assert (result['stds'][-1] == 1e-4).all()
    assert result['clipped_noises'][-1].abs().max() <= 1
    assert not torch.equal(result['means'][-1], result['before_final_clip'])


def test_bridge_keeps_input_gradient_without_reprojection():
    head, bridge, hidden, history, status = fixture()
    post = head.feature_encoder(hidden).detach().requires_grad_()
    def forbidden(*args):
        raise AssertionError('post384 projected twice')
    head.feature_encoder.forward = forbidden
    context = bridge.encode_context(history, status)
    mean, _, _ = bridge.mean_from_post(torch.zeros(1, 8, 3), 1, post, context)
    mean.square().mean().backward()
    assert post.grad.norm() > 0
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in head.parameters())
    assert all(not value.requires_grad for value in context)


def test_native_timesteps_require_python_int_and_do_not_remove_padding():
    head, bridge, hidden, history, status = fixture()
    post = head.feature_encoder(hidden).detach()
    context = bridge.encode_context(history, status)
    with pytest.raises(ValueError, match='Python int'):
        bridge.mean_from_post(torch.zeros(1, 8, 3), torch.tensor([1]), post, context)
    mean, _, _ = bridge.mean_from_post(torch.zeros(1, 8, 3), 1, post, context)
    changed_padding = post.clone()
    changed_padding[:, :2] += 20
    other, _, _ = bridge.mean_from_post(torch.zeros(1, 8, 3), 1, changed_padding, context)
    assert not torch.equal(mean, other)


def test_rejects_non_native_std_and_schedule():
    head = SmallNativeHead()
    head.eval_min_sampling_denoising_std = 0.04
    with pytest.raises(ValueError, match='std floor'):
        NativeDDIMBridge(head)
    head = SmallNativeHead()
    head.ddim_t[0] = 99
    with pytest.raises(ValueError, match='timesteps differ'):
        NativeDDIMBridge(head)
