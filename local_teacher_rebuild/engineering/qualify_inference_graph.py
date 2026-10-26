"""Real-input native graph parity qualification, independent of formal training."""
import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    for name in ('repo', 'paths', 'features', 'noise', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--per-population', type=int, default=16)
    args = parser.parse_args()
    sys.path.insert(0, args.repo)
    from student_distillation.asset_identity import atomic_json, sha256_file
    from student_distillation.data import SceneDataset, batch_scene
    from student_distillation.inference_graph import InferenceGraphBridge
    from student_distillation.noise import EvaluationNoiseStore
    from student_distillation.train import bridge_factory

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paths = json.loads(pathlib.Path(args.paths).read_text())
    paths['results_root'] = str(out / 'runtime')
    native = bridge_factory(paths, 'cuda:0')
    fast = InferenceGraphBridge(native)
    parameter_versions = {k: (id(v), v._version) for k, v in native.head.named_parameters()}
    original_methods = [(module, module.forward) for module in native.head.modules()
                        if type(module).__name__ == 'RotaryEmbedding']
    original_eta = native.head.eta
    report = {'status': 'RUNNING', 'formal_training_qualification': False,
              'cases': [], 'populations': ['train', 'validation', 'test'],
              'noise_indices': list(range(5)), 'graph_source_sha256': sha256_file(pathlib.Path(args.repo) / 'student_distillation/inference_graph.py'),
              'native_source_sha256': sha256_file(pathlib.Path(args.repo) / 'student_distillation/native_bridge.py'),
              'feature_manifest_sha256': sha256_file(args.features),
              'noise_manifest_sha256': sha256_file(args.noise),
              'torch': torch.__version__, 'cudnn': torch.backends.cudnn.version(),
              'hostname': __import__('socket').gethostname(), 'native_forward_backend_unchanged': True}
    started = time.monotonic()
    eager_seconds = graph_seconds = 0.

    def exact(a, b, label):
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)):
            raise AssertionError(f'{label}: maxabs={float((a.double()-b.double()).abs().max())}')

    for population in report['populations']:
        dataset = SceneDataset(args.features, population, None)
        noise = EvaluationNoiseStore(args.noise, population, required_indices=list(range(5)))
        selected = sorted(range(len(dataset)), key=lambda i: hashlib.sha256(
            ('NRKD_native_graph_parity_v1|' + dataset.metadata[i]['token']).encode()).digest())[:args.per_population]
        for position in selected:
            item = dataset[position]; batch = batch_scene(item, 'cuda:0')
            valid = batch['image_mask'] | batch['text_mask']
            context = native.encode_context(batch['history12'], batch['status8'])
            # This bounded synthetic shared repair tests changed inputs; it is
            # not a fitted teacher and never becomes a training target.
            shift = torch.sin(torch.arange(384, dtype=torch.float32, device='cuda:0')) * .025
            inputs = [batch['base_post'], torch.where(valid.unsqueeze(-1), batch['base_post'] + shift[None, None], batch['base_post'])]
            for variant, post in enumerate(inputs):
                for j in range(5):
                    tape = noise.tape(item['metadata']['token'], j, 'cuda:0')
                    caller_inputs = {'post': post, 'history': context[0], 'status': context[1], 'noise': tape}
                    caller_snapshots = {key: value.clone() for key, value in caller_inputs.items()}
                    with torch.no_grad():
                        torch.cuda.synchronize(); began = time.monotonic()
                        expected = native.states_from_post(post, context, tape)
                        torch.cuda.synchronize(); eager_seconds += time.monotonic() - began
                        for key, value in caller_inputs.items():
                            exact(value, caller_snapshots[key], 'untouched native caller ' + key)
                        expected_saved = {key: [value.clone() for value in expected[key]]
                                          for key in ('states', 'means', 'logvars', 'stds', 'clipped_noises')}
                        began = time.monotonic()
                        actual = fast.states_from_post(post, context, tape)
                        torch.cuda.synchronize(); graph_seconds += time.monotonic() - began
                    for key in ('trajectory', 'normalized_actions', 'before_final_clip'):
                        exact(expected[key], actual[key], key)
                    for key in ('states', 'means', 'logvars', 'stds', 'clipped_noises'):
                        for a, b in zip(expected[key], actual[key]):
                            exact(a, b, key)
                    # Results retained from a prior query must not alias graph storage.
                    saved = actual['trajectory'].clone()
                    retained_states = [value.clone() for value in actual['states']]
                    with torch.no_grad():
                        fast.mean_from_post(tape[0] * .25, 0, post + .001, context)
                    exact(actual['trajectory'], saved, 'retained output storage')
                    for key, value in caller_inputs.items():
                        exact(value, caller_snapshots[key], 'graph replay mutated caller ' + key)
                    for current, snapshot in zip(actual['states'], retained_states):
                        exact(current, snapshot, 'retained graph state storage')
                    for key, snapshots in expected_saved.items():
                        for current, snapshot in zip(expected[key], snapshots):
                            exact(current, snapshot, 'independent native reference mutated ' + key)
            if native.head.eta is not original_eta or any(module.forward != method for module, method in original_methods):
                raise AssertionError('Native host metadata methods were not restored')
            if parameter_versions != {k: (id(v), v._version) for k, v in native.head.named_parameters()}:
                raise AssertionError('Frozen native parameter identities/versions changed')
            report['cases'].append({'population': population, 'token': item['metadata']['token'],
                                    'length': len(valid[0]), 'valid': int(valid.sum()),
                                    'variants': 2, 'five_noise_all_states_and_outputs_bitwise': True})
            atomic_json(out / 'LIVE.json', {'completed_scenes': len(report['cases']), 'total_scenes': 3 * args.per_population,
                                          'seconds': time.monotonic() - started})
    post = batch['base_post'].detach().clone().requires_grad_()
    before_count = fast.graph_statistics['autograd_queries']
    mean = fast.mean_from_post(tape[0], 2, post, context)[0]
    mean.square().mean().backward()
    if post.grad is None or not bool(torch.isfinite(post.grad).all()) or float(post.grad.norm()) <= 0:
        raise AssertionError('Original native autograd fallback lost input gradients')
    if fast.graph_statistics['autograd_queries'] != before_count + 1:
        raise AssertionError('An autograd query used the inference graph')
    report.update(status='PASS_REAL_NATIVE_INFERENCE_GRAPH_PARITY',
                  seconds=time.monotonic() - started, eager_seconds=eager_seconds, graph_seconds=graph_seconds,
                  speedup=eager_seconds / graph_seconds, statistics=fast.graph_statistics,
                  autograd_uses_untouched_native_function=True, actual_input_gradient_norm=float(post.grad.norm()),
                  frozen_parameters_and_metadata_restored=True,
                  caller_inputs_preserved_bytewise=True, retained_states_preserved_bytewise=True)
    atomic_json(out / 'COMPLETE.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}), flush=True)


if __name__ == '__main__':
    main()
