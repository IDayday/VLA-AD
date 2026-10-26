import pytest
import torch

from student_distillation.adapter import ContinuousD48Adapter


def make_adapter(d0=0.02, cap=4.0):
    basis = torch.eye(384, dtype=torch.float64)[:, :48]
    return ContinuousD48Adapter(basis, d0, torch.zeros(384), torch.ones(384),
                                torch.zeros(24), torch.ones(24), cap)


def inputs():
    generator = torch.Generator().manual_seed(17)
    post = torch.randn(1, 9, 384, generator=generator)
    image = torch.tensor([[False, False, True, True, True, False, False, False, False]])
    text = torch.tensor([[False, False, False, False, False, True, True, True, True]])
    state = torch.randn(1, 24, generator=generator)
    return post, image, text, state


def test_zero_identity_preserves_padding_and_live_first_output_gradient():
    torch.set_num_threads(2)
    adapter = make_adapter()
    post, image, text, state = inputs()
    post[0, 0, 0] = -0.0
    result = adapter(post, image, text, state)
    assert torch.equal(result['post'], post)
    assert torch.equal(result['post'][:, :2].view(torch.int32), post[:, :2].view(torch.int32))
    assert torch.count_nonzero(result['delta']) == 0
    assert torch.equal(result['coefficient_scale'], torch.ones(1, 1))
    assert not result['cap_triggered'].any()
    target = post.clone()
    target[:, image[0] | text[0], :48] += 0.01
    (result['post'] - target).square().mean().backward()
    assert adapter.output_projection.weight.grad.norm() > 0
    assert adapter.input_proj.weight.grad.norm() == 0


def test_second_update_reaches_earlier_layers():
    torch.set_num_threads(2)
    adapter = make_adapter()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    post, image, text, state = inputs()
    for _ in range(2):
        optimizer.zero_grad()
        result = adapter(post, image, text, state)
        loss = (result['delta'][:, :48] - 0.01).square().mean()
        loss.backward()
        optimizer.step()
    assert adapter.input_proj.weight.grad.norm() > 0


def test_cap_uses_actual_native_fp32_rounding():
    adapter = make_adapter(d0=1.03e-4)
    post, image, text, state = inputs()
    post.fill_(2000.0)
    with torch.no_grad():
        adapter.output_projection.bias.fill_(100)
    result = adapter(post, image, text, state)
    valid = image | text
    actual = (result['post'][:, valid[0]].double() - post[:, valid[0]].double()).square().mean().sqrt()
    assert actual == result['actual_rms'][0]
    assert actual <= result['cap_rms'] + result['cap_tolerance']
    assert result['cap_triggered'].all()
    assert result['cap_corrections'] > 0
    assert torch.equal(result['post'][:, ~valid[0]], post[:, ~valid[0]])


def test_memory_ignores_padding_but_native_output_keeps_it():
    adapter = make_adapter()
    with torch.no_grad():
        adapter.output_projection.weight.normal_(std=0.001)
    post, image, text, state = inputs()
    altered = post.clone()
    altered[:, :2] += 10.0
    first = adapter(post, image, text, state)
    second = adapter(altered, image, text, state)
    torch.testing.assert_close(first['delta'], second['delta'], rtol=0, atol=0)
    assert torch.equal(second['post'][:, :2], altered[:, :2])
    assert second['post'].shape == post.shape


def test_rejects_nonorthonormal_basis_and_overlap():
    basis = torch.eye(384, dtype=torch.float64)[:, :48] * 2
    with pytest.raises(ValueError, match='orthonormal'):
        ContinuousD48Adapter(basis, 1, torch.zeros(384), torch.ones(384),
                             torch.zeros(24), torch.ones(24), 4)
    post, image, text, state = inputs()
    with pytest.raises(ValueError, match='overlap'):
        make_adapter()(post, image, image | text, state)
