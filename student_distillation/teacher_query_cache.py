"""C-only RAM prefix memoization of the unchanged frozen-teacher DDIM query.

No filesystem access, RNG draws, autograd queries, parameter writes or student
queries occur here. Formal admission is a preloaded, verified qualification dict;
engineering=True is an explicit unqualified transport probe only.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import marshal
import threading
from types import CodeType
import torch

SCHEMA = 'nrkd_teacher_query_cache_qualification_v1'
QUALIFIED = 'QUALIFIED_FROZEN_TEACHER_QUERY_PREFIX_CACHE'
MAX_ENTRIES = 500000
IDENTITY_FIELDS = ('feature_manifest_sha256', 'targets_manifest_sha256', 'teacher_bundle_sha256', 'teacher_target')
_ATTACH_LOCK = threading.RLock()


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _normal_code(code):
    # marshal serializes intern/reference details that can change after first
    # execution. Fingerprint executable fields and constants directly instead.
    def constant(value):
        if isinstance(value, CodeType):
            return {'code': _normal_code(value)}
        if isinstance(value, bytes):
            return {'bytes': value.hex()}
        if isinstance(value, tuple):
            return {'tuple': [constant(x) for x in value]}
        if isinstance(value, frozenset):
            return {'frozenset': sorted((constant(x) for x in value), key=repr)}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return {'type': type(value).__name__, 'repr': repr(value)}
    return {'bytecode': code.co_code.hex(), 'argcount': code.co_argcount,
            'posonlyargcount': code.co_posonlyargcount, 'kwonlyargcount': code.co_kwonlyargcount,
            'flags': code.co_flags, 'names': list(code.co_names), 'varnames': list(code.co_varnames),
            'freevars': list(code.co_freevars), 'cellvars': list(code.co_cellvars),
            'constants': [constant(x) for x in code.co_consts]}


def _function_code(value):
    value = getattr(value, '__func__', value)
    code = getattr(value, '__code__', None)
    if code is None:
        raise ValueError('Qualified cache requires ordinary inspectable native Python bridge methods')
    return _canonical(_normal_code(code))


def cache_runtime_identity():
    """Path-independent in-memory code fingerprint, with no source-file I/O."""
    functions = [v for k, v in globals().items() if callable(v) and getattr(v, '__module__', None) == __name__ and hasattr(v, '__code__')]
    functions += [v for k, v in _PrefixCache.__dict__.items() if callable(v) and hasattr(v, '__code__')]
    return _canonical({v.__qualname__: _function_code(v) for v in functions})


def native_runtime_identity(bridge):
    methods = {'bridge.mean_from_post': bridge.mean_from_post,
               'bridge.states_from_post': bridge.states_from_post,
               'bridge.encode_context': bridge.encode_context,
               'head.p_mean_variance': bridge.head.p_mean_variance,
               'head.get_action': bridge.head.get_action,
               'head.denorm_odo': bridge.head.denorm_odo}
    return _canonical({name: _function_code(value) for name, value in methods.items()})


def _is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_cache_receipt(receipt):
    """Validate a PRELOADED RAM dict; caller verifies listed source files once.

    File SHA checks belong to the existing formal preflight, outside this pure
    RAM cache. At attachment the actual in-memory native method code is checked.
    """
    if not isinstance(receipt, dict):
        raise ValueError('Preload the cache qualification JSON outside the pure-RAM cache')
    if receipt.get('schema') != SCHEMA or receipt.get('status') != QUALIFIED:
        raise ValueError('Frozen teacher query cache has no completed transport qualification')
    if receipt.get('policy') != 'C_ONLY_FROZEN_TEACHER_PREFIX':
        raise ValueError('Only C frozen-teacher prefix queries may be cached')
    if not _is_sha(receipt.get('cache_module_sha256')):
        raise ValueError('Cache module source SHA is missing')
    sources = receipt.get('native_source_sha256')
    if not isinstance(sources, dict) or not sources or any(not _is_sha(v) for v in sources.values()):
        raise ValueError('Native source SHA map is missing')
    if receipt.get('cache_runtime_sha256') != cache_runtime_identity() or not _is_sha(receipt.get('native_runtime_sha256')):
        raise ValueError('Cache implementation differs from its actual qualification')
    proof = receipt.get('proof', {})
    if (proof.get('real_scenes', 0) < 64 or proof.get('time_indices') != [0,1,2,3,4] or
        proof.get('training_seeds') != [20261027,20261028] or proof.get('noise_banks') != [0,1,2,3,4]):
        raise ValueError('Cache needs 64 real cases, both seeds, all banks and all five steps')
    for name in ('cold_bytewise', 'warm_bytewise', 'reverse_order_bytewise', 'eviction_bytewise',
                 'last_step_floor_and_unclipped_initial_exact', 'rng_unchanged', 'head_unchanged',
                 'returned_values_do_not_alias_cache', 'no_gradient_or_optimizer_updates'):
        if proof.get(name) is not True:
            raise ValueError('Cache transport qualification is incomplete: ' + name)
    return {'schema': SCHEMA, 'policy': receipt['policy'],
            'cache_module_sha256': receipt['cache_module_sha256'],
            'native_source_sha256': dict(sources), 'cache_runtime_sha256': receipt['cache_runtime_sha256'],
            'native_runtime_sha256': receipt['native_runtime_sha256'],
            'feature_manifest_sha256': receipt.get('feature_manifest_sha256'),
            'proof_sha256': _canonical(proof)}


@dataclass(slots=True)
class _Entry:
    states: torch.Tensor  # CPU FP32 [6,1,8,3]: a0..a4 plus the next a5.
    means: torch.Tensor   # CPU FP32 [5,1,8,3].
    computed: int = 0


class _PrefixCache:
    def __init__(self, bridge, capacity):
        self.lock = threading.RLock()
        self.entries = OrderedDict()
        self.capacity = capacity
        self.head = bridge.head
        self.parameters = [(name, p, p._version) for name, p in self.head.named_parameters()]
        self.runtime = native_runtime_identity(bridge)
        self.method_objects = (getattr(bridge.mean_from_post, '__func__', bridge.mean_from_post),
                               getattr(bridge.head.p_mean_variance, '__func__', bridge.head.p_mean_variance))
        self.admission = None
        self.admission_receipt = None
        self.last_key = None
        self.stats = {'queries': 0, 'hits': 0, 'misses': 0, 'prefix_extensions': 0,
                      'native_mean_queries': 0, 'evictions': 0}

    def check_head(self, bridge):
        if bridge.head is not self.head or self.head.training:
            raise ValueError('Frozen teacher head identity/eval mode changed')
        if self.method_objects != (getattr(bridge.mean_from_post, '__func__', bridge.mean_from_post),
                                   getattr(bridge.head.p_mean_variance, '__func__', bridge.head.p_mean_variance)):
            raise ValueError('Native teacher query implementation changed after cache attachment')
        current_parameters = list(self.head.named_parameters())
        if len(current_parameters) != len(self.parameters):
            raise ValueError('Frozen teacher parameter inventory changed')
        for (name, parameter, version), (current_name, current_parameter) in zip(self.parameters, current_parameters):
            if current_name != name or current_parameter is not parameter or parameter._version != version or parameter.requires_grad or parameter.grad is not None or parameter.dtype != torch.float32:
                raise ValueError('Frozen FP32 teacher parameters/gradients changed: ' + name)
        if float(bridge.std_floor) != 1e-4 or float(bridge.noise_clip) != 1.0 or tuple(bridge.timesteps) != (80,60,40,20,0):
            raise ValueError('Native teacher DDIM protocol changed')

    def trim(self):
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
            self.stats['evictions'] += 1

    def query(self, bridge, h_t, context, tape, j, key):
        self.stats['queries'] += 1
        self.check_head(bridge)
        entry = self.entries.get(key)
        hit = entry is not None and entry.computed > j
        if hit:
            self.stats['hits'] += 1
        else:
            self.stats['misses'] += 1
            if entry is None:
                if not bool(torch.isfinite(h_t).all()):
                    raise ValueError('Nonfinite frozen teacher post')
                entry = _Entry(torch.empty((6,1,8,3), dtype=torch.float32, device='cpu'),
                               torch.empty((5,1,8,3), dtype=torch.float32, device='cpu'))
                entry.states[0].copy_(tape[0].detach().cpu())
                self.entries[key] = entry
            else:
                self.stats['prefix_extensions'] += 1
            current = entry.states[entry.computed].to(h_t.device).clone()
            for step in range(entry.computed, j + 1):
                # These are the verbatim NativeDDIMBridge transition equations.
                # Initial Gaussian is unclipped; all five update noises retain
                # the 1e-4 floor and +/-1 clip, including the last transition.
                mean, logvar, _ = bridge.mean_from_post(current, step, h_t, context)
                std = torch.exp(.5 * logvar).to(h_t.dtype).clamp(min=bridge.std_floor)
                noise = tape[step + 1].clamp(-bridge.noise_clip, bridge.noise_clip)
                next_state = mean + std * noise
                if mean.dtype != torch.float32 or mean.shape != (1,8,3) or mean.requires_grad:
                    raise ValueError('Frozen native teacher mean must be detached FP32[1,8,3]')
                if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(next_state).all()):
                    raise ValueError('Nonfinite native teacher prefix')
                entry.means[step].copy_(mean.detach().cpu())
                entry.states[step + 1].copy_(next_state.detach().cpu())
                entry.computed = step + 1
                self.stats['native_mean_queries'] += 1
                current = next_state
        self.entries.move_to_end(key)
        self.last_key = key
        self.trim()
        # CPU fixtures and GPU callers both receive independent storage.
        return (entry.states[j].to(h_t.device).clone(), entry.means[j].to(h_t.device).clone(), bool(hit))


def _inputs(bridge, h_t, context, tape, j, token, training_seed, noise_bank, identity):
    if torch.is_grad_enabled() or h_t.requires_grad:
        raise ValueError('Teacher cache is only for the frozen no-grad C branch')
    if torch.are_deterministic_algorithms_enabled():
        raise ValueError('Teacher/state forward must retain the untouched native metadata scope')
    if h_t.dtype != torch.float32 or h_t.ndim != 3 or h_t.shape[0] != 1 or h_t.shape[2] != 384:
        raise ValueError('Teacher cache requires full native FP32[1,L,384]')
    if type(j) is not int or not 0 <= j < 5 or type(training_seed) is not int or training_seed < 0 or type(noise_bank) is not int or not 0 <= noise_bank < 5:
        raise ValueError('Use integer native step/bank0..4 and a nonnegative training seed')
    if not isinstance(token, str) or not token or not isinstance(identity, dict):
        raise ValueError('Frozen teacher cache needs a token and immutable identity mapping')
    if identity.get('state_source') not in (None, 'teacher') or identity.get('experiment') in ('H_mu_student','H_full','H_gain1'):
        raise ValueError('D and H-only experiments may never use the teacher cache')
    if any(k not in identity or identity[k] in (None, '') for k in IDENTITY_FIELDS):
        raise ValueError('Missing frozen feature/target/bundle/target-version identity')
    if tape.shape != (6,1,8,3) or tape.dtype != torch.float32 or tape.device != h_t.device or tape.requires_grad or not bool(torch.isfinite(tape).all()):
        raise ValueError('Teacher cache requires the exact detached native six-draw FP32 tape')
    if not isinstance(context, (tuple,list)) or len(context) != 2:
        raise ValueError('Use the original encoded history/status context')
    for x in context:
        if x.dtype != torch.float32 or x.device != h_t.device or x.requires_grad or not bool(torch.isfinite(x).all()):
            raise ValueError('Teacher context must be frozen finite FP32 on the native device')
    tape_sha = hashlib.sha256(tape.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    context_sha = hashlib.sha256(b''.join(x.detach().cpu().contiguous().numpy().tobytes() for x in context)).hexdigest()
    return (token, training_seed, noise_bank, _canonical(identity), tape_sha, context_sha)


def cached_teacher_query(bridge, h_t, context, tape, j, *, token, training_seed, noise_bank,
                         identity, max_entries=MAX_ENTRIES, receipt=None, engineering=False):
    """Return (a_j, teacher_mean, hit), without querying/changing a student.

    The identity must describe genuinely frozen per-token H0 and selected target;
    a mutable student post must never be supplied under that identity. Exact tape
    and ordinary context bytes are additionally included in every cache key.
    """
    if type(max_entries) is not int or not 1 <= max_entries <= MAX_ENTRIES:
        raise ValueError('RAM teacher cache capacity must be in 1..500000')
    key = _inputs(bridge, h_t, context, tape, j, token, training_seed, noise_bank, identity)
    with _ATTACH_LOCK:
        cache = getattr(bridge, '_nrkd_teacher_query_prefix_cache', None)
        if cache is None:
            cache = _PrefixCache(bridge, max_entries)
            bridge._nrkd_teacher_query_prefix_cache = cache
    with cache.lock:
        if not engineering:
            admission = validate_cache_receipt(receipt) if cache.admission is None or receipt is not cache.admission_receipt else cache.admission
            if receipt is None or receipt.get('schema') != SCHEMA or receipt.get('status') != QUALIFIED or receipt.get('cache_runtime_sha256') != admission['cache_runtime_sha256']:
                raise ValueError('Qualified cache receipt changed or disappeared')
            if admission['native_runtime_sha256'] != cache.runtime:
                raise ValueError('Actual native bridge/head code differs from cache qualification')
            if admission.get('feature_manifest_sha256') != identity['feature_manifest_sha256']:
                raise ValueError('Cache qualification used different frozen Base features')
            if training_seed not in (20261027,20261028):
                raise ValueError('Formal teacher cache is qualified only for the two Round1 training seeds')
            cache.admission = admission
            cache.admission_receipt = receipt
        cache.capacity = max_entries
        return cache.query(bridge, h_t, context, tape, j, key)


def clear_teacher_query_cache(bridge):
    """Discard RAM memoization only; mathematical queries remain unchanged."""
    cache = getattr(bridge, '_nrkd_teacher_query_prefix_cache', None)
    if cache is not None:
        with cache.lock:
            cache.entries.clear(); cache.last_key = None


def teacher_query_cache_statistics(bridge):
    cache = getattr(bridge, '_nrkd_teacher_query_prefix_cache', None)
    if cache is None:
        return {'entries': 0, 'capacity': 0, 'cpu_tensor_bytes': 0}
    with cache.lock:
        return dict(cache.stats, entries=len(cache.entries), capacity=cache.capacity,
                    cpu_tensor_bytes=len(cache.entries) * (6 + 5) * 1 * 8 * 3 * 4)


def teacher_query_cache_debug_last(bridge, *, engineering=False):
    if not engineering:
        raise ValueError('Raw cache snapshots are an engineering diagnostic only')
    cache = getattr(bridge, '_nrkd_teacher_query_prefix_cache', None)
    if cache is None or cache.last_key not in cache.entries:
        raise ValueError('No retained teacher prefix entry')
    with cache.lock:
        entry = cache.entries[cache.last_key]
        return {'computed': entry.computed, 'states': entry.states[:entry.computed + 1].clone(),
                'means': entry.means[:entry.computed].clone()}
