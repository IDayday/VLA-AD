"""Strict identities for the independently refitted local Round1 teacher."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256_file(path, block_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    import os
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)


def read_json(path):
    return json.loads(Path(path).read_text())


def resolve_ref(ref, root, *, verify=True):
    if isinstance(ref, (str, Path)):
        p = Path(ref)
        p = p if p.is_absolute() else Path(root) / p
        return p.resolve()
    if not isinstance(ref, dict) or 'path' not in ref:
        raise ValueError('An artifact reference needs path and sha256')
    p = resolve_ref(ref['path'], root)
    if not p.is_file():
        raise FileNotFoundError(p)
    if verify:
        expected = ref.get('sha256')
        if not expected or sha256_file(p) != expected:
            raise ValueError(f'Artifact SHA mismatch: {p}')
    return p


def _npz(ref, root):
    with np.load(resolve_ref(ref, root), allow_pickle=False) as z:
        return {k: np.array(z[k]) for k in z.files}


def load_library_bundle(path):
    """Return local basis/D0/library/identity; never fall back to historical PPU assets.

    BUNDLE.json: status=FROZEN_LOCAL_TEACHER, teacher_origin=local_refit,
    fit_population=train, source_checkpoint_sha256, library={path,sha256},
    optional artifact_root; library space/operator refs resolve under artifact_root
    (or library JSON parent). selections[population]={path,sha256,units:points}.
    """
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f'Local teacher is not ready: {p}')
    bundle = read_json(p)
    if bundle.get('status') != 'FROZEN_LOCAL_TEACHER' or bundle.get('teacher_origin') != 'local_refit':
        raise ValueError('Round1 requires the new FROZEN_LOCAL_TEACHER/local_refit bundle')
    if bundle.get('fit_population') != 'train':
        raise ValueError('Teacher fitting must use Train only')
    checkpoint_sha = bundle.get('source_checkpoint_sha256', '')
    if len(checkpoint_sha) != 64:
        raise ValueError('Missing source checkpoint SHA256')
    library_path = resolve_ref(bundle['library'], p.parent)
    config = read_json(library_path)
    root = resolve_ref(bundle.get('artifact_root', str(library_path.parent)), p.parent)
    space = _npz(config['space'], root)
    basis = np.asarray(space['basis'])
    center = np.asarray(space['center'])
    d0 = float(space['D0'])
    if basis.shape != (384, 48) or basis.dtype != np.float64 or not np.isfinite(basis).all():
        raise ValueError('Local D48 must be finite FP64[384,48]')
    orthogonality_error = float(np.max(np.abs(basis.T @ basis - np.eye(48))))
    if orthogonality_error > 1e-8:
        raise ValueError(f'Local basis is not orthonormal: {orthogonality_error}')
    if center.shape != (384,) or not np.isfinite(center).all() or not np.isfinite(d0) or d0 <= 0:
        raise ValueError('Invalid local center/D0')
    if config.get('K') != 8 or len(config.get('experts', [])) != 8:
        raise ValueError('Round1 requires a newly refitted K8 library')
    parameters = [_npz(s['parameters'], root) for s in config['experts']]
    caps = [float(s['cap_D0']) * float(config['cap_factor']) for s in config['experts']]
    if not np.isfinite(caps).all() or min(caps) < 0 or not 0 < max(caps) <= 4.0:
        raise ValueError('New local dose menu requires nonnegative per-slot caps and 0<shared maximum<=4 D0')
    for expert, cap in zip(config['experts'], caps):
        if cap == 0 and not ((expert.get('Base') is True or str(expert.get('method', '')).lower() == 'base') and float(expert['alpha']) == 0):
            raise ValueError('A zero teacher slot cap is allowed only for explicit literal Base with alpha=0')
    # Preserve every frozen expert's actual native cap. The student uses ONE
    # fixed maximum bound for every scene, independent of source/choice.
    student_cap_in_d0 = max(caps)
    runtime_ref = bundle.get('runtime_predictor')
    runtime_sha = None
    if runtime_ref is not None:
        import importlib.util
        import sys
        runtime_path = resolve_ref(runtime_ref, p.parent)
        runtime_sha = sha256_file(runtime_path)
        module_name = 'nrkd_local_teacher_' + runtime_sha
        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            spec = importlib.util.spec_from_file_location(module_name, runtime_path)
            if spec is None or spec.loader is None:
                raise ValueError('Cannot import new local teacher runtime')
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        cls = getattr(module, runtime_ref.get('class', 'LocalLibrary'))
        library = (cls.from_bundle(str(p)) if hasattr(cls, 'from_bundle') else
                   cls(bundle=bundle, config=config, space=space, parameters=parameters, artifact_root=str(root)))
        if not np.array_equal(np.asarray(library.basis), basis) or not np.array_equal(np.asarray(library.center), center) or float(library.D0) != d0:
            raise ValueError('Local runtime basis/center/D0 differs from frozen artifacts')
        if library.config != config or not callable(getattr(library, 'targets', None)):
            raise ValueError('Local runtime must expose the frozen config and native targets API')
    else:
        if any(s['method'] not in ('centroid_D0', 'local', 'mlp') for s in config['experts']):
            raise ValueError('This newly selected operator requires a checksum-verified runtime_predictor')
        from interface_teacher.bank import Library
        # Reuse exact immutable math only for supported operator families, with
        # NEW local arrays. Never mutate bank.ROOT or any historical artifact.
        library = Library.__new__(Library)
        library.name = config.get('id', config.get('name', 'local_D_C8_double'))
        library.config = config
        library.basis, library.center, library.D0 = basis, center, d0
        library.parameters, library.networks = parameters, {}
    if not hasattr(library, 'name'):
        library.name = config.get('id', config.get('name', 'local_D_C8_double'))
    identity = {'bundle_path': str(p), 'bundle_sha256': sha256_file(p),
                'library_sha256': sha256_file(library_path), 'basis_sha256': config['space']['sha256'],
                'source_checkpoint_sha256': checkpoint_sha, 'teacher_origin': 'local_refit', 'runtime_predictor_sha256': runtime_sha,
                'orthogonality_maxabs': orthogonality_error, 'per_slot_cap_in_d0': caps,
                'student_cap_in_d0': student_cap_in_d0,
                'teacher_identity': bundle.get('teacher_identity', bundle.get('name', library.name))}
    return {'basis_FP64': basis, 'basis': basis, 'D0': d0, 'd0': d0,
            'cap_in_d0': student_cap_in_d0, 'student_cap_in_d0': student_cap_in_d0,
            'per_slot_cap_in_d0': caps, 'library': library, 'identity': identity,
            'bundle': bundle, 'bundle_path': str(p), 'artifact_root': str(root)}


def population_rows(population, frozen_repo=None):
    import csv
    import gzip
    if population not in ('train', 'validation', 'test'):
        raise ValueError(population)
    repo = Path(frozen_repo or Path(__file__).resolve().parents[1])
    with gzip.open(repo / 'manifests' / f'{population}.csv.gz', 'rt') as f:
        rows = list(csv.DictReader(f))
    tokens = [r['token'] for r in rows]
    if len(set(tokens)) != len(tokens):
        raise ValueError(f'Duplicate tokens in {population}')
    return rows


def assert_population_isolation(frozen_repo=None):
    rows = {p: population_rows(p, frozen_repo) for p in ('train', 'validation', 'test')}
    sets = {p: {r['token'] for r in rs} for p, rs in rows.items()}
    for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        if sets[a] & sets[b]:
            raise ValueError(f'Token leakage: {a}/{b}')
    train_logs = {r['session'] for r in rows['train']}
    val_logs = {r['session'] for r in rows['validation']}
    if train_logs & val_logs:
        raise ValueError('Train/Validation session leakage')
    return {'counts': {p: len(rs) for p, rs in rows.items()},
            'train_validation_session_disjoint': True,
            'population_token_sha256': {p: canonical_sha(sorted(ts)) for p, ts in sets.items()}}
