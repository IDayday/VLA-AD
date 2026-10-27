"""Real-GPU engineering checks and strictly gated local-teacher qualification.

A synthetic orthonormal basis can qualify engineering while the new local fit is
running. Such a result NEVER grants CODE_CHECKS_PASS or formal training admission.
Run --ddp-only under torchrun on >=2 GPUs, then pass its receipt to the full CLI.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from .asset_identity import atomic_json, canonical_sha, load_library_bundle, read_json, resolve_ref, sha256_file
from .adapter import ContinuousD48Adapter
from .data import SceneDataset, batch_scene
from .checkpointing import clone_nested, load_checkpoint, read_checkpoint, rng_state, save_checkpoint
from .losses import backward_deterministically, deterministic_backward_context, deterministic_student_forward_context

CHECKS = ('asset_identity', 'zero_identity', 'actual_teacher_injection', 'native_bridge_replay',
          'zero_point_gradient', 'known_realizable_positive_control', 'dit_input_gradient',
          'ddp_tail_objective', 'next_update_recovery')


def code_identity():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted((root / 'student_distillation').glob('*.py'))}


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def exact_tensor(a, b, label):
    if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
        error = float((a.double() - b.double()).abs().max()) if a.shape == b.shape else None
        raise AssertionError(f'{label} differs: shape/dtype/value, maxabs={error}')


def nested_equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(nested_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(nested_equal(x, y) for x, y in zip(a, b))
    return a == b


def nested_differences(a, b, path='root'):
    """Exact, persisted failure diagnostics; never a tolerance-based acceptance."""
    if torch.is_tensor(a):
        if torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype:
            if torch.equal(a.cpu(), b.cpu()):
                return []
            x, y = a.detach().cpu().double(), b.detach().cpu().double()
            delta = x - y
            return [{'path': path, 'shape': list(a.shape), 'dtype': str(a.dtype),
                     'unequal_elements': int((x != y).sum()), 'maxabs': float(delta.abs().max()),
                     'relative_l2': float(delta.norm() / x.norm().clamp_min(1e-300))}]
        return [{'path': path, 'difference': 'tensor shape/dtype/type'}]
    if isinstance(a, np.ndarray):
        return [] if isinstance(b, np.ndarray) and np.array_equal(a, b) else [{'path': path, 'difference': 'numpy'}]
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [{'path': path, 'difference': 'dict keys'}]
        return [d for k in a for d in nested_differences(a[k], b[k], path + '.' + str(k))]
    if isinstance(a, (list, tuple)) and type(a) is type(b) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in nested_differences(x, y, path + f'[{i}]')]
    return [] if a == b else [{'path': path, 'first': str(a), 'second': str(b)}]


def atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as f:
        torch.save(clone_nested(value), f)
        f.flush(); os.fsync(f.fileno())
    temporary.replace(path)
    return sha256_file(path)


class ReferenceNoise:
    def __init__(self, manifest):
        self.path = Path(manifest).resolve()
        spec = read_json(self.path)
        if spec.get('schema') == 'native_explicit_evaluation_noise_v1':
            from .noise import EvaluationNoiseStore
            self.store = EvaluationNoiseStore(self.path, 'train', required_indices=[0, 1, 2, 3, 4])
            self.identity = self.store.identity
            self.bank = None
        else:
            # Historical explicit random tensors are legitimate paired action
            # noise, independent of historical expert parameters or choices.
            s = spec['train']
            bank = resolve_ref({'path': s['bank'], 'sha256': s['bank_sha256']}, self.path.parent)
            mapping = resolve_ref({'path': s['token_map'], 'sha256': s['token_map_sha256']}, self.path.parent)
            tokens = read_json(mapping)
            self.positions = {t if isinstance(t, str) else t['token']: i for i, t in enumerate(tokens)}
            if len(self.positions) != len(tokens):
                raise ValueError('Duplicate reference-noise tokens')
            self.bank = np.load(bank, mmap_mode='r', allow_pickle=False)
            if self.bank.dtype != np.float32 or tuple(self.bank.shape[2:]) != (6, 1, 8, 3) or self.bank.shape[1] < 5:
                raise ValueError('Reference noise has wrong native six-draw layout')
            self.identity = {'manifest_sha256': sha256_file(self.path), 'bank_sha256': s['bank_sha256'],
                             'tokens_sha256': s['token_map_sha256'], 'explicit_reference': True}

    def tape(self, token, index, device):
        if self.bank is None:
            return self.store.tape(token, index, device)
        return torch.from_numpy(np.array(self.bank[self.positions[token], index], copy=True)).to(device)


def _basis(args, out):
    out.mkdir(parents=True, exist_ok=True)
    if args.teacher_bundle:
        bundle = load_library_bundle(args.teacher_bundle)
        return bundle['basis'], bundle['d0'], bundle['cap_in_d0'], bundle['identity'], bundle
    rng = np.random.default_rng(20261027)
    basis, _ = np.linalg.qr(rng.standard_normal((384, 48)))
    basis = np.asarray(basis, np.float64)
    path = out / 'ENGINEERING_ONLY_SYNTHETIC_BASIS.npz'
    np.savez(path, basis=basis, D0=np.float64(.1), cap_in_d0=np.float64(4.))
    identity = {'teacher_origin': 'ENGINEERING_ONLY_SYNTHETIC_NOT_FORMAL_TEACHER',
                'basis_sha256': sha256_file(path), 'd0': .1, 'cap_in_d0': 4., 'seed': 20261027}
    return basis, .1, 4., identity, None


def _adapter(basis, d0, cap, stats, device):
    return ContinuousD48Adapter(basis, d0, stats['token_mean'], stats['token_std'],
                               stats['state_mean'], stats['state_std'], cap_in_d0=cap).to(device).float()


def _ordinary_batch(batch):
    return {k: batch[k] for k in ('base_post', 'image_mask', 'text_mask', 'state24')}


def _predict(adapter, batch):
    return adapter(**_ordinary_batch(batch))


def _loss(adapter, batch, target_shift, scale):
    result = _predict(adapter, batch)
    valid = batch['image_mask'] | batch['text_mask']
    target = torch.where(valid.unsqueeze(-1), batch['base_post'] + target_shift[None, None], batch['base_post'])
    return (result['post'][:, valid[0]] - target[:, valid[0]]).square().mean() / scale


def _optimizer(adapter, updates):
    from .train import make_optimizer
    config = {'optimizer': {'lr': 1e-4, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': .01},
              'scheduler': {'warmup_updates': 16, 'total_updates': updates, 'min_lr': 1e-5}}
    return make_optimizer(adapter, config)


def _next_indices(sampler, count, size):
    result = []
    for _ in range(count):
        if sampler['cursor'] == size:
            sampler['permutation'] = np.random.permutation(size).tolist()
            sampler['cursor'] = 0
            sampler['passes'] += 1
        result.append(sampler['permutation'][sampler['cursor']])
        sampler['cursor'] += 1
    return result


def _control_update(adapter, optimizer, scheduler, batches, targets, scale, sampler, batch_size):
    indices = _next_indices(sampler, batch_size, len(batches))
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.
    for i in indices:
        loss = _loss(adapter, batches[i], targets[i], scale)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite known-realizable control loss')
        backward_deterministically(loss / len(indices))
        loss_sum += float(loss.detach())
    norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    return {'loss': loss_sum / len(indices), 'indices': indices, 'grad_norm': float(norm)}


def native_replay(bridge, post, batch, tape):
    """Compare the REAL original get_action method with post injection, stepwise."""
    from noise_replay import replay_bank
    def tensor_leaves(value, prefix=''):
        if torch.is_tensor(value):
            return {prefix: value}
        if isinstance(value, dict):
            return {name: item for key, child in value.items()
                    for name, item in tensor_leaves(child, prefix + '/' + str(key)).items()}
        if isinstance(value, (list, tuple)):
            return {name: item for key, child in enumerate(value)
                    for name, item in tensor_leaves(child, prefix + '/' + str(key)).items()}
        return {}
    inputs = tensor_leaves({'post': post, 'base_post': batch['base_post'], 'history': batch['history12'],
                            'status': batch['status8'], 'tape': tape})
    before = {key: value.detach().clone() for key, value in inputs.items()}
    head = bridge.head
    captured = []
    original = head.p_mean_variance
    def capture(actions, *args, **kwargs):
        values = original(actions, *args, **kwargs)
        captured.append((actions.detach().clone(), tuple(x.detach().clone() for x in values)))
        return values
    head.p_mean_variance = capture
    hook = head.feature_encoder.register_forward_hook(lambda module, inputs, output: post)
    try:
        dummy_hidden = torch.zeros((1, 1, 1536), dtype=torch.float32, device=post.device)
        with torch.no_grad(), replay_bank(tape):
            native = head.get_action(dummy_hidden, SimpleNamespace(his_traj=batch['history12'], status_feature=batch['status8']))['pred_traj']
    finally:
        hook.remove()
        head.p_mean_variance = original
    with torch.no_grad():
        context = bridge.encode_context(batch['history12'], batch['status8'])
        result = bridge.states_from_post(post, context, tape)
    if len(captured) != 5:
        raise AssertionError('Native get_action did not execute five DDIM transitions')
    exact_tensor(native, result['trajectory'], 'native final trajectory')
    traces = []
    for j, (state, values) in enumerate(captured):
        mean, logvar, _ = values
        std = torch.exp(.5 * logvar).to(post.dtype).clamp(min=bridge.std_floor)
        clipped_noise = tape[j + 1].clamp(-bridge.noise_clip, bridge.noise_clip)
        exact_tensor(state, result['states'][j], f'input state {j}')
        exact_tensor(mean, result['means'][j], f'transition mean {j}')
        exact_tensor(logvar, result['logvars'][j], f'transition logvar {j}')
        exact_tensor(std, result['stds'][j], f'transition std {j}')
        exact_tensor(clipped_noise, result['clipped_noises'][j], f'transition clipped noise {j}')
        next_state = mean + std * clipped_noise
        expected = result['states'][j + 1] if j < 4 else result['before_final_clip']
        exact_tensor(next_state, expected, f'stochastic next state {j}')
        exact_tensor(bridge.states_from_post(post, context, tape, stop_before=j), state, f'stop-before state {j}')
        traces.append({'step': j, 'state_sha256': tensor_sha(state), 'mean_sha256': tensor_sha(mean),
                       'std_sha256': tensor_sha(std), 'noise_sha256': tensor_sha(clipped_noise)})
    if not bool((result['stds'][-1] >= 1e-4).all()):
        raise AssertionError('Last-step get_action minimum noise was removed')
    # Distinct subsequent inputs must not overwrite a graph's first caller or
    # any retained DDIM outputs. Same-input replay alone cannot expose aliases.
    retained = tensor_leaves(result)
    retained_before = {key: value.detach().clone() for key, value in retained.items()}
    context_before = [value.detach().clone() for value in context]
    with torch.no_grad():
        other_post = post.detach().clone() + .001
        other_context = tuple(value.detach().clone() + .002 for value in context)
        other_tape = tape.detach().clone() + .003
        other_inputs = tensor_leaves((other_post, other_context, other_tape))
        other_before = {key: value.clone() for key, value in other_inputs.items()}
        bridge.states_from_post(other_post, other_context, other_tape)
    for key, value in inputs.items():
        exact_tensor(value.view(torch.int32), before[key].view(torch.int32), 'native replay caller input unchanged ' + key)
    for index, value in enumerate(context):
        exact_tensor(value.view(torch.int32), context_before[index].view(torch.int32), 'native replay encoded context unchanged')
    for key, value in retained.items():
        exact_tensor(value.view(torch.int32), retained_before[key].view(torch.int32), 'native replay retained output unchanged ' + key)
    for key, value in other_inputs.items():
        exact_tensor(value.view(torch.int32), other_before[key].view(torch.int32), 'native replay alternate input unchanged ' + key)
    return {'trajectory': native.detach(), 'trace': traces, 'trajectory_sha256': tensor_sha(native),
            'explicit_noise_sha256': tensor_sha(tape), 'all_steps_exact': True,
            'caller_inputs_bitwise_immutable': True, 'retained_chain_outputs_bitwise_immutable': True}


def check_input_gradients(bridge, batch, tape):
    context = bridge.encode_context(batch['history12'], batch['status8'])
    with torch.no_grad():
        states = bridge.states_from_post(batch['base_post'], context, tape)['states']
    checks = []
    original = bridge.head.feature_encoder.forward
    def forbidden(*args, **kwargs):
        raise AssertionError('A post384 condition was incorrectly projected a second time')
    bridge.head.feature_encoder.forward = forbidden
    try:
        for j, state in enumerate(states):
            post = batch['base_post'].detach().clone().requires_grad_()
            with deterministic_student_forward_context():
                mean = bridge.mean_from_post(state.detach(), j, post, context)[0]
            backward_deterministically(mean.square().mean())
            norm = float(post.grad.norm()) if post.grad is not None else 0.
            if not np.isfinite(norm) or norm <= 0:
                raise AssertionError(f'Frozen DiT input gradient is absent at step {j}')
            if any(p.grad is not None or p.requires_grad for p in bridge.head.parameters()):
                raise AssertionError('Frozen action head received parameter gradients')
            checks.append({'step': j, 'post_gradient_norm': norm, 'head_parameter_gradients': False})
    finally:
        bridge.head.feature_encoder.forward = original
    return {'steps': checks, 'same_detached_query_state': True, 'no_second_feature_encoder': True}


def student_forward_native_parity(bridge, batches, noise):
    """Prove the grad-enabled deterministic metadata scope keeps Native means."""
    if len(batches) < 64:
        raise ValueError('Student Native-forward parity requires 64 real scenes')
    cases = []
    for batch in batches[:64]:
        if torch.are_deterministic_algorithms_enabled():
            raise AssertionError('Native baseline forward must retain its original deterministic=False scope')
        token = batch['token']
        context = bridge.encode_context(batch['history12'], batch['status8'])
        with torch.no_grad():
            states = bridge.states_from_post(batch['base_post'], context, noise.tape(token, 0, batch['base_post'].device))['states']
        steps = []
        for j, state in enumerate(states):
            with torch.no_grad():
                native = bridge.mean_from_post(state.detach(), j, batch['base_post'], context)[0]
            student_post = batch['base_post'].detach().clone().requires_grad_()
            with deterministic_student_forward_context():
                student = bridge.mean_from_post(state.detach(), j, student_post, context)[0]
            exact_tensor(native.view(torch.int32), student.detach().view(torch.int32), 'grad-enabled deterministic-scope Native mean byte parity')
            if torch.are_deterministic_algorithms_enabled():
                raise AssertionError('Student metadata scope leaked into subsequent Native forward')
            steps.append({'step': j, 'mean_sha256': tensor_sha(native), 'bitwise_equal': True})
        cases.append({'token': token, 'steps': steps})
    return {'status': 'PASS', 'scenes': len(cases), 'time_indices': [0,1,2,3,4], 'bitwise': True,
            'native_forward_flag': False, 'grad_enabled_student_metadata_flag': True,
            'native_attention_backend_unchanged': True, 'cases': cases}


def positive_control(args, batches, adapter_factory, out, identities):
    from .distributed import Environment
    student = adapter_factory()
    initial = clone_nested(student.state_dict())
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    with torch.no_grad():
        teacher.output_projection.weight.normal_(std=5e-5)
        teacher.output_projection.bias.normal_(std=.005)
        targets = []
        for batch in batches:
            result = _predict(teacher, batch)
            target = result['delta'][0].detach().clone()
            reconstructed = torch.where(result['valid_mask'].unsqueeze(-1), batch['base_post'] + target[None, None], batch['base_post'])
            exact_tensor(reconstructed, result['post'], 'known teacher compact reconstruction')
            targets.append(target)
        scale = float(torch.stack(targets).double().square().mean())
    if scale <= 0 or not np.isfinite(scale):
        raise AssertionError('Known-realizable control has no nonzero target')
    def mean_error(model):
        with torch.no_grad():
            return float(np.mean([float(_loss(model, b, t, scale)) for b, t in zip(batches, targets)]))
    initial_loss = mean_error(student)
    teacher_state_sha = atomic_torch(out / 'KNOWN_REALIZABLE_TEACHER.pt', teacher.state_dict())
    targets_path = out / 'KNOWN_REALIZABLE_TARGET_SHIFTS.npy'
    np.save(targets_path, torch.stack(targets).cpu().numpy(), allow_pickle=False)
    target_shifts_sha = sha256_file(targets_path)
    # Independent two-step probe; its optimizer is discarded before the control.
    probe = adapter_factory(); probe.load_state_dict(clone_nested(initial))
    opt, sched, _ = _optimizer(probe, args.control_updates)
    gradient_steps = []
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        backward_deterministically(_loss(probe, batches[0], targets[0], scale))
        output_norm = float(probe.output_projection.weight.grad.norm())
        preceding_norm = float(probe.input_proj.weight.grad.norm())
        if not np.isfinite(output_norm) or output_norm <= 0 or (step == 1 and preceding_norm <= 0):
            raise AssertionError('Zero-point output/second-step upstream gradient is broken')
        gradient_steps.append({'step': step + 1, 'output_projection_gradient_norm': output_norm,
                               'input_projection_gradient_norm': preceding_norm})
        opt.step(); sched.step()
    del probe, opt, sched
    optimizer, scheduler, groups = _optimizer(student, args.control_updates)
    sampler = {'permutation': np.random.permutation(len(batches)).tolist(), 'cursor': 0, 'passes': 0}
    env = Environment(device=str(batches[0]['base_post'].device))
    logs = []
    for update in range(args.control_updates):
        log = _control_update(student, optimizer, scheduler, batches, targets, scale, sampler, args.control_batch)
        logs.append(dict(update=update + 1, **log))
        if (update + 1) % 16 == 0:
            atomic_json(out / 'LIVE.json', {'status': 'KNOWN_REALIZABLE_CONTROL', 'completed_updates': update + 1,
                        'total_updates': args.control_updates, 'recent_loss': log['loss'], 'formal_training': False})
        if (update + 1) % 64 == 0:
            print(json.dumps({'control_update': update + 1, 'loss': log['loss']}), flush=True)
    final_loss = mean_error(student)
    control = {'status': 'PASS' if final_loss <= initial_loss * .01 else 'FAIL',
               'real_native_scenes': len(batches), 'updates': args.control_updates, 'microbatch': 1,
               'accumulation': args.control_batch, 'initial_normalized_mse': initial_loss,
               'final_normalized_mse': final_loss, 'relative_mse': final_loss / initial_loss,
               'required_relative_mse_max': .01, 'teacher': 'same_adapter_initial_hidden_weights_and_small_nonzero_output_weights',
               'teacher_output_weight_std': 5e-5, 'teacher_output_bias_std': .005,
               'teacher_state_sha256': teacher_state_sha, 'target_shifts_sha256': target_shifts_sha,
               'h_mse_scale': scale, 'logs': logs, 'engineering_only': True}
    atomic_json(out / 'KNOWN_REALIZABLE_CONTROL.json', control)
    if control['status'] != 'PASS':
        raise AssertionError('Known-realizable positive control did not reach predefined relative MSE <=.01')
    progress = {'global_step': args.control_updates, 'epoch': sampler['passes'], 'global_batch_cursor': sampler['cursor']}
    checkpoint = out / 'control_checkpoint.pt'
    save_checkpoint(checkpoint, adapter=student, optimizer=optimizer, scheduler=scheduler,
                    progress=progress, sampler=sampler, identities=identities, config=vars(args),
                    calibration={'h_mse_scale': scale}, parameter_groups=groups, env=env)
    source_snapshot = read_checkpoint(checkpoint, identities)
    source_snapshot_before = clone_nested(source_snapshot)
    independent = []
    for label in ('first', 'second'):
        model = adapter_factory()
        opt, sch, groups = _optimizer(model, args.control_updates)
        state = load_checkpoint(checkpoint, model, opt, sch, env, identities)
        local_sampler = clone_nested(state['sampler'])
        # Exercise every restored RNG stream as well as the next shuffled batch.
        rng_probe = {'python': random.random(), 'numpy': float(np.random.rand()),
                     'torch_cpu': torch.rand(4), 'torch_cuda': torch.rand(4, device=env.device)}
        log = _control_update(model, opt, sch, batches, targets, scale, local_sampler, args.control_batch)
        next_path = out / f'control_next_update_{label}.pt'
        save_checkpoint(next_path, adapter=model, optimizer=opt, scheduler=sch,
                        progress=dict(progress, global_step=args.control_updates + 1, next_update_loss=log['loss'], next_batch_indices=log['indices']), sampler=local_sampler,
                        identities=identities, config=vars(args), calibration={'h_mse_scale': scale},
                        parameter_groups=groups, env=env)
        independent.append({'adapter': clone_nested(model.state_dict()), 'optimizer': clone_nested(opt.state_dict()),
                            'scheduler': clone_nested(sch.state_dict()), 'sampler': clone_nested(local_sampler),
                            'loss': log, 'RNG': rng_state(), 'rng_probe': clone_nested(rng_probe)})
        del model, opt, sch
    source_unchanged = nested_equal(source_snapshot, source_snapshot_before)
    replay_equal = nested_equal(independent[0], independent[1])
    diagnostic = {'independent_replays_exact': replay_equal, 'source_snapshot_unchanged': source_unchanged,
                  'differences': nested_differences(independent[0], independent[1]),
                  'source_snapshot_differences': nested_differences(source_snapshot, source_snapshot_before),
                  'first_next_loss': independent[0]['loss'], 'second_next_loss': independent[1]['loss']}
    atomic_json(out / 'NEXT_UPDATE_EXACT_COMPARISON.json', diagnostic)
    if not replay_equal or not source_unchanged:
        raise AssertionError('Independent next-update exact comparison failed; see NEXT_UPDATE_EXACT_COMPARISON.json')
    recovery = {'status': 'PASS', 'independent_loads': 2, 'compared': list(independent[0]),
                'source_snapshot_unchanged': True, 'source_checkpoint_sha256': sha256_file(checkpoint),
                'next_update_checkpoints': {label: sha256_file(out / f'control_next_update_{label}.pt') for label in ('first','second')}}
    return control, recovery, {'status': 'PASS', 'steps': gradient_steps}


def _active_transition_update(adapter, optimizer, scheduler, bridge, batches, shifts, calibration, args, state_source, phase):
    from .losses import scene_loss
    from .noise import training_schedule, train_noise_seed, make_noise_tape
    cfg = {'experiment': 'H_mu_' + state_source, 'enable_transition_loss': True, 'state_source': state_source,
           'training_seed': args.seed, 'teacher_target': 'gain1'}
    cache_cfg = _qualification_cache_config(args) if state_source == 'teacher' else None
    if cache_cfg is not None:
        cfg.update(cache_cfg)
    optimizer.zero_grad(set_to_none=True)
    results = []
    for batch, shift in zip(batches, shifts):
        token = batch['token']
        schedule = training_schedule(token, args.seed, 6 + phase)
        tape = make_noise_tape(train_noise_seed(token, args.seed, schedule['noise_bank']), batch['base_post'].device)
        current = dict(batch, teacher_actual_shift=shift[None].detach(), noise_tape=tape, **schedule)
        queries = []
        original = bridge.mean_from_post
        def capture(actions, j, post, context):
            queries.append({'actions': actions.detach().clone(), 'j': j, 'post_grad': post.requires_grad,
                            'actions_grad': actions.requires_grad})
            return original(actions, j, post, context)
        if cache_cfg is None:
            bridge.mean_from_post = capture
            try:
                loss, stats = scene_loss(current, adapter, bridge, cfg, args.active_curriculum_update + phase, calibration)
            finally:
                bridge.mean_from_post = original
        else:
            # Do not monkeypatch an admitted bridge: its method fingerprints
            # are deliberately part of cache attachment. Capture only the NEW
            # adapter output, then independently verify the actual cached point
            # and native mean against the loss's normalized functional term.
            observed = {}
            hook = adapter.register_forward_hook(lambda module, inputs, result: observed.update(post=result['post']))
            try:
                loss, stats = scene_loss(current, adapter, bridge, cfg, args.active_curriculum_update + phase, calibration)
            finally:
                hook.remove()
            from .teacher_query_cache import teacher_query_cache_debug_last
            debug = teacher_query_cache_debug_last(bridge, engineering=True)
            j = schedule['time_index']
            state = debug['states'][j].to(batch['base_post'].device).detach()
            cached_mean = debug['means'][j].to(batch['base_post'].device).detach()
            valid = batch['image_mask'] | batch['text_mask']
            teacher_post = torch.where(valid.unsqueeze(-1), batch['base_post'] + shift[None, None], batch['base_post']).detach()
            context = bridge.encode_context(batch['history12'], batch['status8'])
            with torch.no_grad():
                teacher_mean = original(state, j, teacher_post, context)[0]
            exact_tensor(cached_mean, teacher_mean, 'formal C cached teacher mean at the actual query point')
            with deterministic_student_forward_context():
                student_mean = original(state, j, observed['post'], context)[0]
            expected_mse = float((student_mean - teacher_mean).square().mean().detach())
            if expected_mse != stats['mu_mse'] or not observed['post'].requires_grad or state.requires_grad:
                raise AssertionError('Cached C loss changed the actual same-state native student/teacher query')
            queries = [{'actions': state, 'j': j, 'post_grad': False, 'actions_grad': False},
                       {'actions': state, 'j': j, 'post_grad': True, 'actions_grad': False}]
        if not stats['transition_executed'] or not stats['mu_weight'] > 0:
            raise AssertionError('C/D recovery fixture did not execute active functional supervision')
        if len(queries) != (2 if cache_cfg is not None else schedule['time_index'] + 2):
            raise AssertionError('C/D query state was not freshly rolled out to the selected native step')
        teacher_query, student_query = queries[-2:]
        exact_tensor(teacher_query['actions'], student_query['actions'], 'C/D same detached teacher/student query state')
        if teacher_query['j'] != student_query['j'] or teacher_query['j'] != schedule['time_index'] or teacher_query['post_grad'] or not student_query['post_grad'] or student_query['actions_grad']:
            raise AssertionError('C/D teacher/student gradient or same-step protocol is invalid')
        backward_deterministically(loss / len(batches))
        if any(p.grad is not None or p.requires_grad for p in bridge.head.parameters()):
            raise AssertionError('C/D active recovery updated a frozen native head parameter')
        results.append({'token': token, 'loss': float(loss.detach()), 'stats': stats, **schedule,
                        'noise_tape_sha256': tensor_sha(tape), 'same_detached_query_state': True,
                        'student_condition_gradient': True, 'native_head_parameter_gradient': False})
    norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
    optimizer.step(); scheduler.step()
    return {'loss': float(np.mean([r['loss'] for r in results])), 'grad_norm': float(norm), 'scenes': results}


def _qualification_cache_config(args):
    """Use the same explicit common performance protocol as the formal C run."""
    import yaml
    common = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/student_distillation/common.yaml').read_text())
    performance = common.get('performance', {})
    engineering_receipt = getattr(args, 'engineering_teacher_query_cache_receipt', None)
    if engineering_receipt:
        if not args.active_recovery_only_from or args.teacher_bundle:
            raise ValueError('Engineering cache override is only for persisted synthetic-control active-only recovery')
        source = Path(args.active_recovery_only_from).resolve()
        synthetic_basis = source / 'ENGINEERING_ONLY_SYNTHETIC_BASIS.npz'
        synthetic_targets = source / 'KNOWN_REALIZABLE_TARGET_SHIFTS.npy'
        if not synthetic_basis.is_file() or not synthetic_targets.is_file():
            raise ValueError('Persisted engineering basis/known-realizable targets are missing')
        paths = read_json(args.paths)
        paths.update(feature_manifest=str(Path(args.feature_manifest).resolve()),
                     targets_manifest=str(synthetic_targets), teacher_bundle=str(synthetic_basis),
                     teacher_query_cache_receipt=str(Path(engineering_receipt).resolve()),
                     results_root=str(Path(args.out).resolve()))
        return {'performance': dict(performance, frozen_teacher_query_cache=True), 'paths': paths,
                'teacher_target': 'ENGINEERING_ONLY_KNOWN_REALIZABLE_FIXED_TARGET'}
    if not all(getattr(args, key, None) for key in ('teacher_bundle', 'targets_manifest', 'calibration')):
        return None
    if performance.get('frozen_teacher_query_cache') is not True:
        return None
    paths = read_json(args.paths)
    paths.update(feature_manifest=str(Path(args.feature_manifest).resolve()),
                 targets_manifest=str(Path(args.targets_manifest).resolve()),
                 teacher_bundle=str(Path(args.teacher_bundle).resolve()))
    if not paths.get('results_root'):
        raise ValueError('Formal cached C qualification requires the explicit results_root in --paths')
    return {'performance': performance, 'paths': paths}


def active_transition_recovery(args, batches, shifts, adapter_factory, bridge, out, source_checkpoint,
                               calibration, identities):
    """Native C/D input-backward replay with genuinely populated Adam history."""
    from .distributed import Environment
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    env = Environment(device=str(batches[0]['base_post'].device))
    baseline = read_checkpoint(source_checkpoint)
    baseline_before = clone_nested(baseline)
    if len(batches) not in (4, 8) or not any(bool(t.any()) for t in shifts):
        raise ValueError('Active C/D recovery needs 4 or 8 actual native scenes with nonzero fixed targets')
    result = {}
    for state_source in ('teacher', 'student'):
        adapter = adapter_factory()
        optimizer, scheduler, groups = _optimizer(adapter, args.control_updates)
        loaded = load_checkpoint(source_checkpoint, adapter, optimizer, scheduler, env)
        before_steps = [int(s['step']) for s in optimizer.state.values() if 'step' in s]
        if not before_steps or min(before_steps) < 2 or not any(bool(s.get('exp_avg', torch.zeros(1)).any()) for s in optimizer.state.values()):
            raise AssertionError('Native functional recovery probe requires already-updated Adam state')
        preparation = [_active_transition_update(adapter, optimizer, scheduler, bridge, batches, shifts, calibration,
                                                args, state_source, phase) for phase in (0, 1)]
        prepared = out / f'active_{state_source}_checkpoint.pt'
        source_step = int(loaded['progress']['global_step'])
        progress = dict(loaded['progress'], global_step=source_step + 2, active_fixture_phase=2,
                        curriculum_fixture_update=args.active_curriculum_update + 2)
        sampler = {'control_source': clone_nested(loaded['sampler']), 'active_fixture_phase': 2,
                   'case_tokens': [b['token'] for b in batches], 'training_noise_namespace': 'NRKD_train_v1'}
        save_checkpoint(prepared, adapter=adapter, optimizer=optimizer, scheduler=scheduler, progress=progress,
                        sampler=sampler, identities=identities, config=dict(vars(args), state_source=state_source,
                        fixture_only=True), calibration=calibration, parameter_groups=groups, env=env)
        independent = []
        for label in ('first', 'second'):
            if state_source == 'teacher' and _qualification_cache_config(args) is not None:
                # Checkpoint caches are excluded because they are pure memoized
                # functions. Compare both restores cold; the separate mandatory
                # admission proves arbitrary cold/warm/evicted query equivalence.
                from .teacher_query_cache import clear_teacher_query_cache
                clear_teacher_query_cache(bridge)
            model = adapter_factory(); opt, sch, names = _optimizer(model, args.control_updates)
            state = load_checkpoint(prepared, model, opt, sch, env, identities)
            phase = int(state['sampler']['active_fixture_phase'])
            rng_probe = {'python': random.random(), 'numpy': float(np.random.rand()), 'CPU': torch.rand(4),
                         'CUDA': torch.rand(4, device=env.device)}
            value = _active_transition_update(model, opt, sch, bridge, batches, shifts, calibration, args, state_source, phase)
            next_path = out / f'active_{state_source}_next_{label}.pt'
            next_progress = dict(progress, global_step=source_step + 3, active_fixture_phase=phase + 1,
                                 next_update_loss=value['loss'])
            next_sampler = dict(state['sampler'], active_fixture_phase=phase + 1)
            save_checkpoint(next_path, adapter=model, optimizer=opt, scheduler=sch, progress=next_progress,
                            sampler=next_sampler, identities=identities, config=dict(vars(args), state_source=state_source,
                            fixture_only=True), calibration=calibration, parameter_groups=names, env=env)
            independent.append({'parameters': clone_nested(model.state_dict()), 'Adam': clone_nested(opt.state_dict()),
                                'scheduler': clone_nested(sch.state_dict()), 'RNG': rng_state(),
                                'rng_probe': clone_nested(rng_probe), 'loss_and_same_state_queries': value,
                                'sampler': next_sampler})
            del model, opt, sch
        differences = nested_differences(independent[0], independent[1])
        value = {'status': 'PASS' if not differences else 'FAIL', 'trained_Adam_state': True,
                 'transition_loss_active': True, 'independent_loads': 2, 'real_native_scenes': len(batches),
                 'optimizer_steps_before_probe': min(before_steps), 'preparation_functional_updates': 2,
                 'curriculum_fixture_update': args.active_curriculum_update, 'fixture_is_not_formal_training': True,
                 'same_detached_query_state': True, 'head_forward_backend_unchanged': True, 'deterministic_algorithms_scope': 'grad_enabled_student_query_and_backward',
                 'native_get_action_teacher_state_forward_deterministic_algorithms': False,
                 'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                 'head_parameters_frozen': True, 'source_snapshot_unchanged': nested_equal(baseline, baseline_before),
                 'differences': differences, 'preparation': preparation,
                 'first_next_update': independent[0]['loss_and_same_state_queries'],
                 'second_next_update': independent[1]['loss_and_same_state_queries'],
                 'source_checkpoint_sha256': sha256_file(source_checkpoint),
                 'prepared_checkpoint_sha256': sha256_file(prepared),
                 'next_checkpoint_sha256': {label: sha256_file(out / f'active_{state_source}_next_{label}.pt') for label in ('first','second')}}
        value['qualified_teacher_query_cache_enabled'] = state_source == 'teacher' and _qualification_cache_config(args) is not None
        if value['qualified_teacher_query_cache_enabled']:
            value['cache_recovery_state_policy'] = 'cold before each independent restore; separate admission proves all cold/warm/reverse/eviction values and RNG exact'
        atomic_json(out / f'ACTIVE_{state_source.upper()}_EXACT_COMPARISON.json', value)
        result[state_source] = value
        del adapter, optimizer, scheduler
    atomic_json(out / 'ACTIVE_TRANSITION_RECOVERY.json', result)
    if any(v['status'] != 'PASS' or not v['source_snapshot_unchanged'] for v in result.values()):
        raise AssertionError('Active C/D native-head next-update exact replay failed; see ACTIVE_TRANSITION_RECOVERY.json')
    return result


def run_active_only(args):
    """Resume an engineering control's real Adam256; do not rerun its 256 updates."""
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    source = Path(args.active_recovery_only_from).resolve()
    if not torch.cuda.is_available() or not str(args.device).startswith('cuda'):
        raise ValueError('Active native-head recovery needs an actually dispatched CUDA GPU')
    torch.cuda.set_device(torch.device(args.device))
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    start_code = code_identity()
    identities = _identities(args, read_json(args.paths))
    checkpoint = source / 'control_checkpoint.pt'
    previous = read_checkpoint(checkpoint)
    for key in ('feature_manifest_sha256', 'input_statistics_sha256', 'checkpoint_sha256'):
        if previous['identities'][key] != identities[key]:
            raise ValueError('Active diagnostic changed its real input/checkpoint identity: ' + key)
    basis, d0, cap, basis_identity, _ = _basis(args, out)
    stats = read_json(args.input_statistics)
    dataset = SceneDataset(args.feature_manifest, 'train', None)
    cases = read_json(source / 'SOURCE.json')['cases'][:64]
    batches = [batch_scene(dataset[dataset.features.by_token[c['token']]], args.device) for c in cases]
    targets = np.load(source / 'KNOWN_REALIZABLE_TARGET_SHIFTS.npy', allow_pickle=False)[:args.active_scenes]
    shifts = [torch.from_numpy(np.array(t, copy=True)).to(args.device) for t in targets]
    factory = lambda: _adapter(basis, d0, cap, stats, args.device)
    from .train import bridge_factory
    paths = read_json(args.paths); paths['results_root'] = str(out / 'runtime')
    paths['feature_manifest'] = args.feature_manifest
    bridge = bridge_factory(paths, args.device)
    if not args.native_reference_cache_index:
        raise ValueError('Active recovery startup test requires --native-reference-cache-index')
    reference_index = Path(args.native_reference_cache_index).resolve()
    references = {r['token']: r for r in map(json.loads, reference_index.read_text().splitlines())}
    explicit_noise = ReferenceNoise(args.reference_noise_manifest)
    parity = []
    for batch in batches[:2]:
        ref = references[batch['token']]; file = reference_index.parent / ref['npz']
        if sha256_file(file) != ref['sha256']:
            raise ValueError('Corrupt saved Native Base trajectory source')
        with np.load(file, allow_pickle=False) as z:
            for j in range(5):
                replay = native_replay(bridge, batch['base_post'], batch, explicit_noise.tape(batch['token'], j, args.device))
                saved = torch.from_numpy(np.array(z['Base_trajectories'][j], copy=True)).to(args.device).unsqueeze(0)
                exact_tensor(replay['trajectory'].view(torch.int32), saved.view(torch.int32), 'active fixture saved Native forward parity')
                parity.append({'token': batch['token'], 'noise_index': j, 'trajectory_sha256': replay['trajectory_sha256'], 'saved_native_trajectory_exact': True})
    atomic_json(out / 'NATIVE_STARTUP_FORWARD_PARITY.json', {'status': 'PASS', 'cases': parity, 'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG')})
    student_parity = student_forward_native_parity(bridge, batches, explicit_noise)
    atomic_json(out / 'STUDENT_FORWARD_NATIVE_PARITY.json', student_parity)
    batches = batches[:args.active_scenes]
    calibration = {'h_mse_scale': previous['calibration']['h_mse_scale'], 'mu_mse_scale': [1e-6] * 5,
                   'engineering_only_mu_floor': True}
    checks = active_transition_recovery(args, batches, shifts, factory, bridge, out, checkpoint, calibration, identities)
    if code_identity() != start_code:
        raise RuntimeError('Source changed during active C/D diagnostic')
    result = {'schema': 'nrkd_active_transition_recovery_v1', 'status': 'ENGINEERING_ACTIVE_TRANSITION_PASS',
              'formal_asset_qualification': False, 'identities': identities, 'code_sha256': start_code,
              'basis_identity': basis_identity, 'active_transition_recovery': checks,
              'student_forward_native_parity': student_parity,
              'source_control_checkpoint_sha256': sha256_file(checkpoint)}
    cache_cfg = _qualification_cache_config(args)
    if cache_cfg is not None:
        from .losses import teacher_query_cache_settings
        configured = dict(cache_cfg, experiment='H_mu_teacher', state_source='teacher',
                          teacher_target=cache_cfg.get('teacher_target', 'gain1'))
        result['performance_admissions'] = {'teacher_query_cache': teacher_query_cache_settings(configured, refresh=True)['admission']}
    atomic_json(out / 'QUALIFICATION.json', result)
    return result


def _record(out, name, value):
    path = out / (name.upper() + '.json')
    atomic_json(path, value)
    return {'status': 'PASS', 'artifact': {'path': str(path), 'sha256': sha256_file(path)}}


def _identities(args, paths):
    result = {}
    for key, value in [('teacher_bundle', args.teacher_bundle), ('feature_manifest', args.feature_manifest),
                       ('targets_manifest', args.targets_manifest), ('calibration', args.calibration)]:
        result[key + '_sha256'] = sha256_file(value) if value else None
    result['checkpoint_sha256'] = sha256_file(paths['checkpoint'])
    result['input_statistics_sha256'] = sha256_file(args.input_statistics)
    result['reference_noise_manifest_sha256'] = sha256_file(args.reference_noise_manifest)
    return result


def run_qualification(args):
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    initial_code_identity = code_identity()
    if not str(args.device).startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('This CLI requires an actually dispatched CUDA GPU; CPU tests cannot qualify it')
    torch.cuda.set_device(torch.device(args.device))
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    paths = read_json(args.paths)
    paths['results_root'] = str(out / 'runtime')
    paths['feature_manifest'] = args.feature_manifest
    basis, d0, cap, basis_identity, bundle = _basis(args, out)
    stats = read_json(args.input_statistics)
    identities = _identities(args, paths)
    dataset = SceneDataset(args.feature_manifest, 'train', None)
    indices = sorted(range(len(dataset)), key=lambda i: hashlib.sha256(('NRKD_qualification_v1|' + dataset.metadata[i]['token']).encode()).hexdigest())[:max(64, args.control_scenes)]
    items = [dataset[i] for i in indices]
    batches = [batch_scene(item, args.device) for item in items]
    adapter_factory = lambda: _adapter(basis, d0, cap, stats, args.device)
    adapter = adapter_factory().eval()
    checks = {name: {'status': 'PENDING'} for name in CHECKS}
    source = {'GPU': torch.cuda.get_device_name(args.device), 'device': str(args.device),
              'torch': torch.__version__, 'torch_git': torch.version.git_version, 'cudnn': torch.backends.cudnn.version(),
              'tf32': False, 'autocast': False, 'native_post_dtype': 'float32',
              'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
              'native_get_action_teacher_state_forward_deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
              'grad_enabled_student_query_deterministic_metadata': True,
              'deterministic_algorithms_during_backward': True,
              'cases': [dict(item['metadata']) for item in items], 'basis_identity': basis_identity,
              'hostname': __import__('socket').gethostname(), 'pid': os.getpid(),
              'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'real_frozen_head_checkpoint_sha256': identities['checkpoint_sha256']}
    # Hash the actual checkpoint and every required VLM file, not only a stale receipt.
    import check_assets
    asset_result = check_assets.check(args.paths, full=True)
    baseline = read_json(Path(__file__).resolve().parents[1] / 'configs/repair_final_baseline.json')
    identities['vlm_sha256'] = canonical_sha(baseline['vlm_sha256'])
    checks['asset_identity'] = _record(out, 'asset_identity', dict(asset_result, checkpoint_sha256=identities['checkpoint_sha256'],
                      vlm_files_sha256=baseline['vlm_sha256'], basis_identity=basis_identity))
    with torch.no_grad():
        zero_cases = []
        for item, batch in zip(items, batches):
            result = _predict(adapter, batch)
            exact_tensor(result['post'].view(torch.int32), batch['base_post'].view(torch.int32), 'zero-init native Base identity')
            if bool(result['delta'].any()):
                raise AssertionError('Zero-init adapter emitted nonzero shift')
            zero_cases.append({'token': item['metadata']['token'], 'post_sha256': tensor_sha(result['post']), 'native_padding_exact': True})
        capped = adapter_factory().eval()
        capped.output_projection.bias[0] = cap * 8.
        cap_result = _predict(capped, batches[0])
        valid = cap_result['valid_mask']
        exact_tensor(cap_result['post'][:, ~valid[0]].view(torch.int32), batches[0]['base_post'][:, ~valid[0]].view(torch.int32), 'capped native padding')
        if not bool(cap_result['cap_triggered'].all()) or bool((cap_result['actual_rms'] > cap_result['cap_rms'] + cap_result['cap_tolerance']).any()):
            raise AssertionError('Actual post-rounding FP32 cap failed')
        zero_value = {'status': 'PASS', 'cases': zero_cases, 'cap_probe_actual_rms': cap_result['actual_rms'].cpu().tolist(),
                      'cap_rms': float(cap_result['cap_rms']), 'cap_tolerance': float(cap_result['cap_tolerance'])}
    checks['zero_identity'] = _record(out, 'zero_identity', zero_value)
    del capped
    from .train import bridge_factory
    bridge = bridge_factory(paths, args.device)
    noise = ReferenceNoise(args.reference_noise_manifest)
    source['explicit_reference_noise'] = noise.identity
    atomic_json(out / 'SOURCE.json', source)
    reference_cache = None
    reference_index_path = args.native_reference_cache_index or paths.get('native_reference_cache_index')
    if reference_index_path:
        rp = Path(reference_index_path).resolve()
        reference_cache = (rp.parent, {r['token']: r for r in map(json.loads, rp.read_text().splitlines())})
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') and reference_cache is None:
        raise ValueError('A changed cuBLAS startup environment requires --native-reference-cache-index for saved Native forward byte parity')
    replay_cases = []
    for item, batch in list(zip(items, batches))[:args.parity_scenes]:
        for j in range(5):
            tape = noise.tape(item['metadata']['token'], j, args.device)
            value = native_replay(bridge, batch['base_post'], batch, tape)
            saved_parity = None
            if reference_cache is not None:
                root, records = reference_cache
                reference = records[item['metadata']['token']]
                source_file = root / reference['npz']
                if sha256_file(source_file) != reference['sha256'] or reference['sha256'] != item['metadata']['source_npz_sha256']:
                    raise ValueError('Saved Native forward reference belongs to different Base features')
                with np.load(source_file, allow_pickle=False) as z:
                    saved = torch.from_numpy(np.array(z['Base_trajectories'][j], copy=True)).to(args.device).unsqueeze(0)
                    exact_tensor(value['trajectory'].view(torch.int32), saved.view(torch.int32), 'saved Native Base forward trajectory under startup environment')
                saved_parity = {'status': 'PASS', 'source_npz_sha256': reference['sha256'], 'all_trajectory_bits_exact': True}
            with torch.no_grad():
                student = _predict(adapter, batch)['post']
                student_chain = bridge.states_from_post(student, bridge.encode_context(batch['history12'], batch['status8']), tape)
                exact_tensor(student_chain['trajectory'], value['trajectory'], 'zero-init Base/student replay')
            replay_cases.append({'token': item['metadata']['token'], 'noise_index': j, 'saved_native_forward_parity': saved_parity,
                                 **{k: v for k, v in value.items() if k != 'trajectory'}})
    checks['native_bridge_replay'] = _record(out, 'native_bridge_replay', {'status': 'PASS', 'cases': replay_cases, 'real_frozen_head': True})
    student_parity = student_forward_native_parity(bridge, batches, noise)
    gradient_value = check_input_gradients(bridge, batches[0], noise.tape(items[0]['metadata']['token'], 0, args.device))
    gradient_value['student_forward_native_parity'] = student_parity
    checks['dit_input_gradient'] = _record(out, 'dit_input_gradient', gradient_value)
    control, recovery, zero_grad = positive_control(args, batches[:args.control_scenes], adapter_factory, out, identities)
    checks['known_realizable_positive_control'] = _record(out, 'known_realizable_positive_control', control)
    checks['next_update_recovery'] = _record(out, 'next_update_recovery', recovery)
    checks['zero_point_gradient'] = _record(out, 'zero_point_gradient', zero_grad)
    active_batches = batches[:args.active_scenes]
    active_shifts = [torch.from_numpy(np.array(t, copy=True)).to(args.device) for t in np.load(out / 'KNOWN_REALIZABLE_TARGET_SHIFTS.npy', allow_pickle=False)[:args.active_scenes]]
    active_calibration = {'h_mse_scale': control['h_mse_scale'], 'mu_mse_scale': [1e-6] * 5, 'engineering_only_mu_floor': True}
    if bundle is not None:
        from .build_targets import read_local_selection, compact_actual_shift
        from .evaluate import _score, _scoring
        selection = read_local_selection(bundle, 'train')
        frozen_targets = None
        if args.targets_manifest:
            frozen_targets = SceneDataset({'feature_manifest': args.feature_manifest, 'targets_manifest': args.targets_manifest}, 'train', 'full')
            target_manifest = read_json(args.targets_manifest)
            if target_manifest['teacher_identity']['bundle_sha256'] != bundle['identity']['bundle_sha256']:
                raise ValueError('Compact targets belong to another local teacher bundle')
        if args.calibration:
            calibration = read_json(args.calibration)
            if calibration.get('status') != 'COMPLETE' or calibration.get('population') != 'train' or calibration.get('teacher_origin') != 'local_refit':
                raise ValueError('Formal qualification requires completed Train-only local-teacher calibration')
            if calibration.get('targets_manifest_sha256') != identities['targets_manifest_sha256']:
                raise ValueError('Calibration was computed from a different compact-target identity')
        lookup = {token: i for i, token in enumerate(selection['tokens'])}
        teacher_cases = []
        teacher_indices = [i for i in sorted(range(len(dataset)), key=lambda i: hashlib.sha256(('NRKD_teacher_injection_v1|' + dataset.metadata[i]['token']).encode()).hexdigest()) if int(selection['choice'][lookup[dataset.metadata[i]['token']]]) > 0 and bool(selection['gain1_keep'][lookup[dataset.metadata[i]['token']]])][:max(args.parity_scenes, args.active_scenes)]
        if len(teacher_indices) < args.parity_scenes:
            raise ValueError('Not enough non-Base selected local targets for actual teacher injection')
        active_batches = []; active_shifts = []
        if args.calibration:
            active_calibration = read_json(args.calibration)
        for teacher_index in teacher_indices:
            item = dataset[teacher_index]; batch = batch_scene(item, args.device)
            token = item['metadata']['token']; choice = int(selection['choice'][lookup[token]])
            kwargs = {'raw_aux24': item['state24'].numpy()} if getattr(bundle['library'], 'requires_aux24', False) else {}
            target = bundle['library'].targets(batch['base_post'], batch['image_mask'], batch['text_mask'], [choice], **kwargs)
            shift, actual_rms, cap_rms = compact_actual_shift(batch['base_post'], batch['image_mask'], batch['text_mask'], target)
            if not np.any(shift):
                raise ValueError('Nonzero teacher injection qualification selected a literal zero shift')
            if 'selected_actual_role_shifts' in selection and not np.array_equal(shift, selection['selected_actual_role_shifts'][lookup[token], 0]):
                raise ValueError('Qualified runtime shift differs from the actually scored local teacher')
            if frozen_targets is not None:
                target_item = frozen_targets[frozen_targets.features.by_token[token]]
                if not np.array_equal(shift, target_item['teacher_actual_shift'].numpy()):
                    raise ValueError('Frozen compact target differs from native teacher injection')
            active_batches.append(batch); active_shifts.append(torch.from_numpy(shift.copy()).to(args.device))
            if len(active_batches) > args.parity_scenes:
                continue
            noises = []
            for j in range(5):
                tape = noise.tape(token, j, args.device)
                native = native_replay(bridge, target['interfaces'], batch, tape)
                base = native_replay(bridge, batch['base_post'], batch, tape)
                base_score = _score(base['trajectory'][0].cpu().numpy(), item['metadata'], _scoring())
                teacher_score = _score(native['trajectory'][0].cpu().numpy(), item['metadata'], _scoring())
                noises.append({'noise_index': j, 'teacher_trajectory_sha256': native['trajectory_sha256'],
                               'base_trajectory_sha256': base['trajectory_sha256'], 'base_score': base_score, 'teacher_score': teacher_score})
            base_points = float(np.mean([r['base_score']['score'] for r in noises]) * 100.)
            teacher_points = float(np.mean([r['teacher_score']['score'] for r in noises]) * 100.)
            saved = selection['q_points'][lookup[token]]
            if not np.isclose(base_points, saved[0], atol=1e-7, rtol=0) or not np.isclose(teacher_points, saved[choice], atol=1e-7, rtol=0):
                raise ValueError('Replayed native Base/teacher PDMS differs from the new local selection scores')
            teacher_cases.append({'token': token, 'choice': choice, 'base_mean_points': base_points, 'teacher_mean_points': teacher_points, 'actual_shift_sha256': hashlib.sha256(shift.tobytes()).hexdigest(),
                                  'actual_rms': actual_rms, 'cap_rms': cap_rms, 'compact_reconstruction_exact': True, 'noise': noises})
        if frozen_targets is not None:
            checks['actual_teacher_injection'] = _record(out, 'actual_teacher_injection', {'status': 'PASS', 'new_local_teacher': True, 'cases': teacher_cases})
        else:
            _record(out, 'actual_teacher_injection_engineering_only', {'status': 'ENGINEERING_ONLY_MISSING_FROZEN_COMPACT_TARGETS', 'new_local_teacher': True, 'cases': teacher_cases})
    active_result = active_transition_recovery(args, active_batches[:args.active_scenes], active_shifts[:args.active_scenes], adapter_factory, bridge, out / 'active_transition_recovery', out / 'control_checkpoint.pt', active_calibration, identities)
    recovery['active_transition_recovery'] = active_result
    checks['next_update_recovery'] = _record(out, 'next_update_recovery', recovery)
    execution = None
    if args.ddp_receipt:
        ddp = read_json(args.ddp_receipt)
        execution = validate_ddp_execution(ddp, getattr(args, 'training_world_size', None))
        for key in ('teacher_bundle_sha256', 'feature_manifest_sha256', 'calibration_sha256'):
            if ddp['identities'].get(key) != identities.get(key):
                raise ValueError(f'DDP receipt identity mismatch: {key}')
        if ddp.get('code_sha256') != code_identity():
            raise ValueError('DDP receipt code changed since its actual check')
        checks['ddp_tail_objective'] = {'status': 'PASS', 'artifact': {'path': str(Path(args.ddp_receipt).resolve()), 'sha256': sha256_file(args.ddp_receipt)}}
    if code_identity() != initial_code_identity:
        raise RuntimeError('Student implementation changed during qualification; rerun after the implementation is frozen')
    all_pass = all(c['status'] == 'PASS' for c in checks.values())
    formal = bool(bundle is not None and args.targets_manifest and args.calibration and all_pass)
    result = {'schema': 'nrkd_code_checks_v1', 'status': 'CODE_CHECKS_PASS' if formal else ('ENGINEERING_CHECKS_PASS_SYNTHETIC_BASIS' if bundle is None else 'PENDING_FORMAL_QUALIFICATION'),
              'formal_asset_qualification': formal, 'identities': identities, 'code_sha256': code_identity(),
              'execution': execution,
              'source': source, 'checks': checks, 'pending_checks': [k for k, v in checks.items() if v['status'] != 'PASS']}
    cache_cfg = _qualification_cache_config(args)
    if cache_cfg is not None:
        from .losses import teacher_query_cache_settings
        configured = dict(cache_cfg, experiment='H_mu_teacher', state_source='teacher', teacher_target='gain1')
        options = teacher_query_cache_settings(configured, refresh=True)
        if active_result['teacher'].get('qualified_teacher_query_cache_enabled') is not True:
            raise ValueError('Formal C qualification did not exercise its admitted teacher query cache')
        result['performance_admissions'] = {'teacher_query_cache': options['admission']}
    atomic_json(out / 'QUALIFICATION.json', result)
    return result


def validate_ddp_execution(receipt, expected_world=None):
    """Bind a real tail-objective proof to its requested rank/accumulation layout."""
    from .train import formal_execution
    world = receipt.get('world_size')
    execution = formal_execution({'runtime': {'world_size': world},
                                  'training': {'gradient_accumulation_steps': 64 // world if type(world) is int and world else None}})
    if expected_world is not None and world != expected_world:
        raise ValueError('DDP qualification world_size differs from --training-world-size')
    required = {'schema': 'nrkd_ddp_tail_check_v1', 'status': 'PASS', 'real_scene_count': 53,
                'global_slots': 64, 'zero_weight_padded_slots': 11, 'native_microbatch': 1,
                'gradient_accumulation_steps': execution['gradient_accumulation_steps'], 'DDP_adapter_only': True}
    if any(receipt.get(key) != expected for key, expected in required.items()):
        raise ValueError('Real DDP tail proof must use microbatch1, accumulation64/world and 53 real of 64 slots')
    return execution


def run_ddp_tail(args):
    from . import distributed as dd
    from .losses import scene_loss
    env = dd.initialize()
    if env.world_size not in (2, 4) or not str(env.device).startswith('cuda'):
        raise ValueError('--ddp-only requires torchrun on two or four actual CUDA GPUs')
    if getattr(args, 'training_world_size', None) not in (None, env.world_size):
        raise ValueError('Actual torchrun WORLD_SIZE differs from --training-world-size')
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    basis, d0, cap, basis_identity, _ = _basis(args, out if env.rank == 0 else out / f'rank_{env.rank}')
    stats = read_json(args.input_statistics)
    dataset = SceneDataset(args.feature_manifest, 'train', None)
    indices = sorted(range(len(dataset)), key=lambda i: hashlib.sha256(('NRKD_qualification_v1|' + dataset.metadata[i]['token']).encode()).hexdigest())[:64]
    batches = [batch_scene(dataset[i], env.device) for i in indices]
    adapter = _adapter(basis, d0, cap, stats, env.device)
    with torch.no_grad():
        adapter.output_projection.weight.normal_(std=5e-5)
        adapter.output_projection.bias.normal_(std=.005)
    initial = clone_nested(adapter.state_dict())
    target = torch.as_tensor(np.asarray(basis[:, 0]) * (d0 * np.sqrt(384.) * .03), device=env.device, dtype=torch.float32)
    for b in batches:
        b['teacher_actual_shift'] = target[None].clone()
    calibration = {'h_mse_scale': float(target.double().square().mean())}
    cfg = {'enable_transition_loss': False, 'state_source': 'none'}
    adapter = dd.wrap_adapter(adapter, env)
    slots, real_count = dd.global_batch_slots(list(range(53)), 0, env.rank, env.world_size)
    local_loss = 0.
    for micro, (i, real) in enumerate(slots):
        with dd.synchronization(adapter, micro == len(slots) - 1):
            loss = scene_loss(batches[i], adapter, None, cfg, 0, calibration)[0]
            backward_deterministically(loss * (env.world_size / real_count if real else 0.))
        if real:
            local_loss += float(loss.detach())
    global_loss = dd.sum_tensor([local_loss], env)[0] / real_count
    distributed_gradients = {n: p.grad.detach().cpu().clone() if p.grad is not None else None for n, p in dd.unwrap(adapter).named_parameters()}
    value = None
    if env.rank == 0:
        reference = _adapter(basis, d0, cap, stats, env.device)
        reference.load_state_dict(initial)
        serial_loss = 0.
        for b in batches[:53]:
            loss = scene_loss(b, reference, None, cfg, 0, calibration)[0]
            backward_deterministically(loss / 53.); serial_loss += float(loss.detach()) / 53.
        comparisons = []
        for name, p in reference.named_parameters():
            actual = distributed_gradients[name]
            if p.grad is None:
                if actual is not None:
                    raise AssertionError(f'DDP gradient unexpectedly present: {name}')
                continue
            expected = p.grad.detach().cpu()
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-4, msg=lambda msg: name + ': ' + msg)
            comparisons.append({'parameter': name, 'maxabs': float((actual - expected).abs().max())})
        if not np.isclose(serial_loss, global_loss, rtol=1e-6, atol=1e-8):
            raise AssertionError('DDP 53-real-scene loss denominator differs from serial53')
        identities = _identities(args, read_json(args.paths))
        value = {'schema': 'nrkd_ddp_tail_check_v1', 'status': 'PASS', 'real_scene_count': 53,
                 'global_slots': 64, 'zero_weight_padded_slots': 11, 'world_size': env.world_size,
                 'gradient_accumulation_steps': len(slots),
                 'identities': identities, 'basis_identity': basis_identity, 'code_sha256': code_identity(),
                 'serial_loss': serial_loss, 'ddp_loss': global_loss, 'gradient_atol': 2e-6, 'gradient_rtol': 1e-4,
                 'gradient_comparisons': comparisons, 'real_native_cases': [dataset.metadata[i]['token'] for i in indices[:53]],
                 'native_microbatch': 1, 'DDP_adapter_only': True}
        value['execution'] = validate_ddp_execution(value, getattr(args, 'training_world_size', None))
        atomic_json(out / 'DDP_TAIL_53.json', value)
    dd.barrier()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return value


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths', required=True)
    p.add_argument('--feature-manifest', required=True)
    p.add_argument('--input-statistics', required=True)
    p.add_argument('--reference-noise-manifest', required=True)
    p.add_argument('--native-reference-cache-index', help='Audited Native Base INDEX with saved S0–S4 trajectories, used to prove unchanged head forward')
    p.add_argument('--teacher-bundle')
    p.add_argument('--targets-manifest')
    p.add_argument('--calibration')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--control-scenes', type=int, choices=[32, 64], default=64)
    p.add_argument('--control-updates', type=int, default=256)
    p.add_argument('--control-batch', type=int, default=8)
    p.add_argument('--parity-scenes', type=int, default=2)
    p.add_argument('--seed', type=int, default=20261027)
    p.add_argument('--active-scenes', type=int, choices=[4, 8], default=8)
    p.add_argument('--active-curriculum-update', type=int, choices=[6000, 9000], default=9000)
    p.add_argument('--active-recovery-only-from', help='Successful engineering control directory with persisted Adam/teacher targets')
    p.add_argument('--engineering-teacher-query-cache-receipt', help='Qualified cache admission for active-only synthetic control; never formal asset admission')
    p.add_argument('--ddp-only', action='store_true')
    p.add_argument('--ddp-receipt')
    p.add_argument('--training-world-size', type=int, choices=(2, 4),
                   help='Bind DDP tail and full qualification to the formal run rank count')
    p.add_argument('--out', required=True)
    args = p.parse_args(argv)
    if args.engineering_teacher_query_cache_receipt and (not args.active_recovery_only_from or args.teacher_bundle):
        p.error('Engineering cache override is only allowed for synthetic persisted-control active-only recovery')
    if not args.ddp_only and int(os.environ.get('WORLD_SIZE', '1')) != 1:
        p.error('Use --ddp-only for torchrun; the full real-head qualification is one GPU process')
    if not 200 <= args.control_updates <= 400 or not 1 <= args.control_batch <= 64 or args.parity_scenes < 2:
        p.error('Qualification requires 200–400 optimizer updates, >=2 parity scenes, and valid control batch')
    start = time.time()
    try:
        result = (run_ddp_tail(args) if args.ddp_only else (run_active_only(args) if args.active_recovery_only_from else run_qualification(args)))
        if result is not None:
            result['elapsed_seconds'] = time.time() - start
            atomic_json(Path(args.out) / ('DDP_TAIL_53.json' if args.ddp_only else 'QUALIFICATION.json'), result)
            print(json.dumps({'status': result['status'], 'out': args.out}), flush=True)
    except BaseException:
        import traceback
        atomic_json(Path(args.out) / f'FAILURE_rank{os.environ.get("RANK", "0")}.json',
                    {'status': 'ENGINEERING_FAILURE', 'traceback': traceback.format_exc(), 'elapsed_seconds': time.time() - start})
        raise


if __name__ == '__main__':
    main()
