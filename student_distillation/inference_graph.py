"""Optional launch-overhead optimization for the unchanged native head.

Only no-grad queries use CUDA graphs. Autograd queries always use the original
NativeDDIMBridge. The native sampler and all floating-point head operations stay
unchanged. Cache-growth checks for the fixed eight action positions and the
frozen eta=1 scalar are evaluated before capture, then their original methods
are restored immediately. No vendor file or parameter is changed.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import MethodType
import torch

from .native_bridge import NativeDDIMBridge


@contextmanager
def _capture_metadata(head):
    """Move two input-independent host synchronizations outside capture."""
    restored = []
    original_eta = head.eta
    try:
        for module in head.modules():
            if type(module).__name__ != 'RotaryEmbedding':
                continue
            if module.max_seq_len_cached < 8:
                raise ValueError('Warm the native eight-action RoPE cache first')
            old_forward = module.forward

            def forward(self, x, position_ids):
                # The native Attention builds arange(N_q); N_q is fixed at 8.
                if position_ids.shape != (1, 8):
                    raise ValueError('CUDA graph specialization requires native eight-action positions')
                gather = position_ids.unsqueeze(1).unsqueeze(3).expand(-1, -1, -1, self.dim)
                cos = self.cos_cached.gather(2, gather)
                sin = self.sin_cached.gather(2, gather)
                return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

            module.forward = MethodType(forward, module)
            restored.append((module, old_forward))
        parameter = next(head.parameters())
        value = original_eta(torch.zeros((1, 8, 3), device=parameter.device, dtype=torch.float32)).detach().clone()
        if not torch.equal(value, torch.ones_like(value)):
            raise ValueError('Only the actual frozen stochastic eta=1 head is supported')

        class CachedEta(torch.nn.Module):
            def __init__(self, tensor):
                super().__init__()
                self.register_buffer('value', tensor)

            def forward(self, x):
                if x.shape != (1, 8, 3):
                    raise ValueError('Native action batch shape changed')
                return self.value

        head.eta = CachedEta(value)
        # This allocation predates graph capture: the graph owner must retain
        # it after the original eta module is restored.
        yield (value, head.eta)
    finally:
        head.eta = original_eta
        for module, method in reversed(restored):
            module.forward = method


class InferenceGraphBridge(NativeDDIMBridge):
    """Opt-in bridge with per-layout/per-step graph caches and exact first-use gate."""
    def __init__(self, native):
        self.__dict__.update(native.__dict__)
        self.eager = native
        self.graphs = {}
        self.graph_statistics = {'captured': 0, 'inference_queries': 0, 'autograd_queries': 0}

    def mean_from_post(self, actions, step_index, post, context):
        if torch.is_grad_enabled():
            self.graph_statistics['autograd_queries'] += 1
            return self.eager.mean_from_post(actions, step_index, post, context)
        self._post(post)
        step_index = self._step(step_index)
        history, status = context
        if actions.shape != (1, 8, 3) or actions.dtype != torch.float32 or actions.device != post.device:
            raise ValueError('Native actions must be FP32 [1,8,3] on the post device')
        if not bool(torch.isfinite(actions).all()):
            raise ValueError('Non-finite native actions')
        if history.device != post.device or status.device != post.device:
            raise ValueError('Native context is on the wrong device')
        inputs = (actions.detach(), post.detach(), history.detach(), status.detach())
        key = (step_index, tuple(post.shape), tuple(history.shape), tuple(status.shape), str(post.device))
        if key not in self.graphs:
            expected = self.eager.mean_from_post(actions, step_index, post, context)
            t = self.head.make_timesteps(1, self.timesteps[step_index], actions.device)
            index = self.head.make_timesteps(1, step_index, actions.device)

            def raw(x, h, his, state):
                return self.head.p_mean_variance(x, t, index, h, his, state, deterministic=False)

            # make_graphed_callables reuses its sample_args as writable static
            # input storage on every replay. detach() alone aliases a caller's
            # Base post, saved state or context; later branches would overwrite
            # those inputs. Every static surface must own independent storage.
            capture_inputs = tuple(value.detach().clone() for value in inputs)
            with _capture_metadata(self.head) as constants:
                graph = torch.cuda.make_graphed_callables(raw, capture_inputs, num_warmup_iters=3,
                                                        allow_unused_input=True)
                actual = tuple(x.clone() for x in graph(*inputs))
            if not all(torch.equal(a, b) for a, b in zip(expected, actual)):
                raise RuntimeError('CUDA graph first-use forward differs from the untouched native head')
            self.graphs[key] = {'call': graph, 'constants': (t, index, *constants)}
            self.graph_statistics['captured'] += 1
        self.graph_statistics['inference_queries'] += 1
        # Replay outputs have static storage: independent clones preserve every
        # caller's saved states/means when another branch reuses this graph.
        return tuple(x.clone() for x in self.graphs[key]['call'](*inputs))


def qualified_inference_bridge(native, receipt, *, feature_manifest):
    """Require actual two-host, full-chain evidence before production opt-in."""
    from .asset_identity import read_json, resolve_ref, sha256_file
    path = Path(receipt).resolve()
    value = read_json(path)
    if value.get('schema') != 'native_inference_graph_qualification_v1' or value.get('status') != 'QUALIFIED_NATIVE_INFERENCE_ONLY':
        raise ValueError('Missing completed inference-graph qualification')
    module_sha = sha256_file(__file__)
    native_sha = sha256_file(Path(__file__).with_name('native_bridge.py'))
    features_sha = sha256_file(feature_manifest)
    expected_checkpoint = '23fb949b10387efeff87e093703d5faf41b61b9e5ead16d4076661e3c7c08a20'
    if value.get('checkpoint_sha256') != expected_checkpoint:
        raise ValueError('Graph qualification belongs to another frozen RL Base')
    hosts = set()
    for reference in value.get('host_qualifications', []):
        proof = read_json(resolve_ref(reference, path.parent))
        if (proof.get('status') != 'PASS_REAL_NATIVE_INFERENCE_GRAPH_PARITY' or
                proof.get('graph_source_sha256') != module_sha or
                proof.get('native_source_sha256') != native_sha or
                proof.get('feature_manifest_sha256') != features_sha or
                proof.get('noise_indices') != [0, 1, 2, 3, 4] or
                len(proof.get('cases', [])) < 48 or
                proof.get('caller_inputs_preserved_bytewise') is not True or
                proof.get('retained_states_preserved_bytewise') is not True or
                proof.get('frozen_parameters_and_metadata_restored') is not True or
                proof.get('autograd_uses_untouched_native_function') is not True):
            raise ValueError('Actual graph parity evidence is incomplete or implementation changed')
        if any(case.get('variants') != 2 or case.get('five_noise_all_states_and_outputs_bitwise') is not True
               for case in proof['cases']):
            raise ValueError('Graph proof did not compare both full native chains')
        hosts.add(proof['hostname'])
    if len(hosts) < 2:
        raise ValueError('Graph production admission requires two distinct qualified hosts')
    return InferenceGraphBridge(native)
