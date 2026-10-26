"""New local D_C8 runtime; unchanged original thirteen-family numerical kernels.

This module reads only newly fitted artifacts and ordinary inputs. Original
source is imported for math, never its historical expert weights or choices.
"""
from __future__ import annotations
import hashlib
import importlib
import json
from pathlib import Path
import sys
import numpy as np


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def resolve(ref, root):
    path = Path(ref['path'] if isinstance(ref, dict) else ref)
    path = path if path.is_absolute() else Path(root) / path
    if isinstance(ref, dict) and sha(path) != ref['sha256']:
        raise ValueError(f'New local artifact SHA mismatch: {path}')
    return path.resolve()


def original_modules(source_root):
    source = Path(source_root).resolve()
    if not (source / 'ol_l1_operators_v1.py').exists():
        source = source / '1024-representation_distillation_diagnosis_batch5/stage2/offline_library/code'
    sys.path[:0] = [str(source), str(source.parents[1] / 'code'), str(source.parents[2] / 'code')]
    ops = importlib.import_module('ol_l1_operators_v1')
    geo = importlib.import_module('ol_l1_geometry_v1')
    return ops, geo


def coefficient(payload, v, aux, fraction, basis, center, view, networks=None, key=None):
    """Exact one-row original FP64 coordinate path; FP32 MLP then FP64 decode."""
    ops = payload['_ops']
    kind, method = payload['kind'], payload['method']
    z = (np.asarray(v, np.float64) - center) @ basis
    if payload.get('Base_fallback_required') or kind == 'base':
        return np.zeros(48, np.float64)
    if kind == 'analytic':
        return ops.predict_original(z, payload['model'])
    if kind in ('paired', 'paired_single'):
        if aux is None:
            raise ValueError(f'{method} requires ordinary raw_aux24; no zero/teacher fallback')
        model = payload['model']
        if kind == 'paired':
            return ops.predict_paired(v, aux, model, method, basis, view, np.array([fraction], np.float64))[0]
        raw = ops.predict_ridge(np.c_[v[None], aux[None]], model['learned'])
        rank, d0 = model['rank'], model['D0']
        if method == 'paired_RRR':
            return raw[0, :rank] * (d0 * np.sqrt(384.))
        return ops.paired.da_decode(raw[:, :rank], raw[:, -1], basis, view, np.array([fraction], np.float64), d0)[0][0]
    if kind != 'mlp':
        raise ValueError(f'Unsupported newly fitted kind {kind}')
    import torch
    saved = payload['snapshot']
    if networks is None:
        networks = {}
    key = key if key is not None else id(payload)
    if key not in networks:
        with torch.random.fork_rng(devices=[]):
            model = ops.old.make_mlp(48, saved['seed'], 'cpu')
        model.load_state_dict(saved['model'], strict=True)
        networks[key] = model.eval().requires_grad_(False)
    with torch.no_grad():
        x = torch.from_numpy(((z - saved['mean']) / saved['scale'])[None].astype(np.float32))
        return networks[key](x).numpy()[0].astype(np.float64) * saved['D0']


class LocalLibrary:
    requires_aux24 = True

    @classmethod
    def from_bundle(cls, path):
        path = Path(path).resolve()
        bundle = json.loads(path.read_text())
        if bundle.get('teacher_origin') != 'local_refit' or bundle.get('status') != 'FROZEN_LOCAL_TEACHER':
            raise ValueError('Only the newly frozen local teacher is accepted')
        config_path = resolve(bundle['library'], path.parent)
        config = json.loads(config_path.read_text())
        root = resolve(bundle.get('artifact_root', str(config_path.parent)), path.parent)
        ops, geo = original_modules(bundle['source_code_root'])
        for ref in bundle['original_math_sources']:
            resolve(ref, path.parent)
        import torch
        self = cls()
        self.config, self.bundle, self.root = config, bundle, root
        self.name = config['id']
        self.ops, self.geo = ops, geo
        with np.load(resolve(config['space'], root), allow_pickle=False) as z:
            self.basis, self.center, self.D0 = np.array(z['basis']), np.array(z['center']), float(z['D0'])
        self.parameters, self.payloads, self.networks = [], [], {}
        for slot in config['experts']:
            with np.load(resolve(slot['parameters'], root), allow_pickle=False) as z:
                self.parameters.append({key: np.array(z[key]) for key in z.files})
            payload = torch.load(resolve(slot['runtime_state'], root), map_location='cpu', weights_only=False)
            payload['_ops'] = ops
            self.payloads.append(payload)
        return self

    def requests(self, role_means, counts, choices=None, raw_aux24=None):
        roles, counts = np.asarray(role_means), np.asarray(counts)
        if roles.dtype != np.float64 or roles.shape != (2, 384) or counts.dtype != np.int64 or counts.shape != (2,) or (counts <= 0).any():
            raise ValueError('Native pooled input must be FP64[2,384] and positive int64[2]')
        if not np.isfinite(roles).all():
            raise ValueError('Nonfinite ordinary pooled input')
        aux = None if raw_aux24 is None else np.asarray(raw_aux24, np.float64).reshape(-1)
        if aux is not None and (aux.shape != (24,) or not np.isfinite(aux).all()):
            raise ValueError('raw_aux24 must contain the ordinary finite state24')
        choices = list(range(1, 9)) if choices is None else list(choices)
        if not choices or any(not isinstance(value, (int, np.integer)) or not 0 <= value <= 8 for value in choices):
            raise ValueError('Choice is an offline Base/slot index in 0..8')
        v = np.einsum('ng,ngd->nd', counts[None] / counts.sum(), roles[None])[0]
        fraction = float(counts[0] / counts.sum())
        raw, caps = [], []
        for choice in choices:
            if choice == 0:
                raw.append(np.zeros((2, 384), np.float64)); caps.append(0.); continue
            slot, payload = self.config['experts'][choice - 1], self.payloads[choice - 1]
            q = coefficient(payload, v, aux, fraction, self.basis, self.center,
                            {'view': 'valid_shared384'}, self.networks, choice - 1)
            decoded = q[None] @ self.basis.T
            physical = (np.stack([decoded, decoded], axis=-2) * float(slot['alpha']))[0]
            raw.append(physical * np.float64(self.config['raw_shift_factor']))
            caps.append(np.float64(slot['cap_D0']) * self.D0 * np.float64(self.config['cap_factor']))
        raw = np.asarray(raw, np.float64)
        return {'choices': np.asarray(choices, np.int64), 'requested_shifts_float64': raw,
                'requested_shifts_float32': raw.astype(np.float32), 'cap_RMS_float64': np.asarray(caps, np.float64)}

    def targets(self, base_interface, image, text, choices, raw_aux24=None):
        import torch
        from interface_teacher.bank import pool_base
        from interface_teacher.native_cap import apply
        base = base_interface.detach()
        if base.ndim == 2:
            base = base[None]
        image = torch.as_tensor(image, dtype=torch.bool, device=base.device).reshape(1, -1)
        text = torch.as_tensor(text, dtype=torch.bool, device=base.device).reshape(1, -1)
        roles, counts = pool_base(base[0].cpu().numpy(), image[0].cpu().numpy(), text[0].cpu().numpy())
        request = self.requests(roles, counts, choices, raw_aux24=raw_aux24)
        with torch.no_grad(), torch.autocast(base.device.type, enabled=False):
            result = apply(base, image, text, torch.from_numpy(request['requested_shifts_float32']).to(base.device),
                           torch.from_numpy(request['cap_RMS_float64']).to(base.device))
        return dict(**result, **request)

    def selected_target(self, base_interface, image, text, token, decisions, raw_aux24=None):
        return self.targets(base_interface, image, text, [decisions.choice(token)], raw_aux24=raw_aux24)['interfaces']
