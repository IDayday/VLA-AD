"""CPU model of CUDA graph writable input/output buffers catches storage aliases."""
import torch

from student_distillation.inference_graph import InferenceGraphBridge
from student_distillation.noise import make_noise_tape
from student_distillation.qualification import native_replay
from test_native_bridge_replay import fixture


def install_static_graph(monkeypatch):
    captures = []
    def capture(function, sample_args, **kwargs):
        captures.append(sample_args)
        outputs = tuple(value.clone() for value in function(*sample_args))
        def replay(*inputs):
            for surface, value in zip(sample_args, inputs):
                surface.copy_(value)
            for surface, value in zip(outputs, function(*sample_args)):
                surface.copy_(value)
            return outputs
        return replay
    monkeypatch.setattr(torch.cuda, 'make_graphed_callables', capture)
    return captures


def clone_tree(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {key: clone_tree(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [clone_tree(child) for child in value]
    return value


def assert_tree_equal(value, before):
    if torch.is_tensor(value):
        assert torch.equal(value.view(torch.int32), before.view(torch.int32))
    elif isinstance(value, dict):
        for key in value:
            assert_tree_equal(value[key], before[key])
    elif isinstance(value, (tuple, list)):
        for child, original in zip(value, before):
            assert_tree_equal(child, original)
    else:
        assert value == before


def test_graph_owns_static_inputs_and_preserves_prior_callers_and_retained_chains(monkeypatch):
    captures = install_static_graph(monkeypatch)
    head, native, hidden, history, status = fixture()
    graph = InferenceGraphBridge(native)
    with torch.no_grad():
        post = head.feature_encoder(hidden).detach()
        context = native.encode_context(history, status)
        actions = torch.randn(1, 8, 3)
        tape = make_noise_tape(33, 'cpu')
        inputs = (actions, post, context, tape, hidden, history, status)
        before = clone_tree(inputs)
        means = graph.mean_from_post(actions, 2, post, context)
        means_before = clone_tree(means)
        result = graph.states_from_post(post, context, tape)
        result_before = clone_tree(result)
        other_post = post.clone() + .001
        other_context = tuple(value.clone() + .002 for value in context)
        other_actions = actions.clone() + .003
        other_tape = tape.clone() + .004
        other = (other_actions, other_post, other_context, other_tape)
        other_before = clone_tree(other)
        graph.mean_from_post(other_actions, 2, other_post, other_context)
        graph.states_from_post(other_post, other_context, other_tape)
        assert_tree_equal(inputs, before)
        assert_tree_equal(other, other_before)
        assert_tree_equal(means, means_before)
        assert_tree_equal(result, result_before)
    assert len(captures) == 5
    caller_pointers = {value.data_ptr() for value in (actions, post, *context)}
    assert all(surface.data_ptr() not in caller_pointers for args in captures for surface in args)


def test_qualification_cross_input_replay_proves_original_inputs_and_outputs_immutable(monkeypatch):
    install_static_graph(monkeypatch)
    head, native, hidden, history, status = fixture()
    post = head.feature_encoder(hidden).detach()
    result = native_replay(InferenceGraphBridge(native), post,
                           {'base_post': post, 'history12': history, 'status8': status},
                           make_noise_tape(54, 'cpu'))
    assert result['all_steps_exact']
    assert result['caller_inputs_bitwise_immutable']
    assert result['retained_chain_outputs_bitwise_immutable']
