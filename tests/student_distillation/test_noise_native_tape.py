import hashlib
import torch

from student_distillation.noise import (evaluation_noise_tape, make_noise_tape,
                                         scene_seed, training_schedule, train_noise_seed)


def test_seed_namespaces_and_25_visit_coverage():
    token, seed = 'test_scene', 20261027
    expected = int(hashlib.sha256(f'NRKD_train_v1|{seed}|{token}|2'.encode()).hexdigest()[:8], 16)
    assert train_noise_seed(token, seed, 2) == expected
    pairs = [training_schedule(token, seed, epoch) for epoch in range(4, 29)]
    assert len({(p['time_index'], p['noise_bank']) for p in pairs}) == 25
    assert all(type(v) is int for pair in pairs for v in pair.values())
    assert training_schedule(token, seed, 4) == training_schedule(token, seed, 29)
    assert scene_seed(token, 0) != train_noise_seed(token, seed, 0)


def test_six_native_draws_match_reseeded_rng_and_leave_global_state_unchanged():
    torch.manual_seed(31)
    before = torch.get_rng_state().clone()
    tape = make_noise_tape(9021, 'cpu')
    assert torch.equal(before, torch.get_rng_state())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9021)
        reference = torch.stack([torch.randn(1, 8, 3) for _ in range(6)])
    assert torch.equal(tape, reference)
    assert tape.shape == (6, 1, 8, 3)
    assert (tape[0].abs() > 1).any()  # initial Gaussian remains unclipped


def test_native_draw_shapes_are_six_separate_calls(monkeypatch):
    original = torch.randn
    calls = []
    def recorded(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, 'randn', recorded)
    make_noise_tape(42, 'cpu')
    assert calls == [(1, 8, 3)] * 6


def test_explicit_reference_tape_is_reused_without_rng(monkeypatch):
    tape = make_noise_tape(42, 'cpu')
    monkeypatch.setattr(torch, 'randn', lambda *a, **k: (_ for _ in ()).throw(AssertionError('RNG called')))
    assert evaluation_noise_tape('scene', 5, 'cpu', reference_tape=tape) is tape
