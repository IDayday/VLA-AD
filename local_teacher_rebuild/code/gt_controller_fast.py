"""Transport-only GT controller acceleration; original science code is unchanged.

This private process rebinds only read/sha/atomic I/O plus their descriptor and
source-function-loader globals. Sixteen bounded threads prewarm completed current
stage receipts, RESULT JSON and referenced array hashes. GPU workers are untouched.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import importlib.util
import io
import json
import marshal
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from types import FunctionType
import uuid

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import gt_numeric as numeric


def stat_key(path):
    s = Path(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def function_identity(function):
    return {'function': function.__name__, 'source_file': function.__code__.co_filename,
            'function_code_sha256': hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()}


class VerifiedIO:
    """Metadata-keyed byte/SHA cache, with no mutable JSON returned to callers."""
    def __init__(self, json_cache_mib=1024):
        self._entries = {}
        self._locks = {}
        self._guard = threading.RLock()
        self._json_order = OrderedDict()
        self._json_bytes = 0
        self._json_limit = int(json_cache_mib) * 1024 * 1024
        self.counters = {'physical_json_reads': 0, 'physical_sha_reads': 0,
                         'json_hits': 0, 'sha_hits': 0, 'invalidations': 0,
                         'verified_bytes_read': 0, 'json_evictions': 0, 'fresh_coarse_timestamp_bypasses': 0}

    @staticmethod
    def _path(path):
        return str(Path(path).resolve())

    def _lock(self, path):
        with self._guard:
            return self._locks.setdefault(path, threading.RLock())

    def _invalidate_locked(self, path):
        old = self._entries.pop(path, None)
        if old is not None and 'json' in old:
            self._json_bytes -= old['bytes']
            self._json_order.pop(path, None)
        if old is not None:
            self.counters['invalidations'] += 1

    def invalidate(self, path):
        path = self._path(path)
        with self._lock(path), self._guard:
            self._invalidate_locked(path)

    def _cached(self, path, key, want_json):
        with self._guard:
            entry = self._entries.get(path)
            # This NFS has whole-second timestamps. Until a coarse timestamp
            # is safely old, same-size in-place changes can be invisible to
            # stat. Re-read bytes during that ambiguity window.
            coarse_fresh = (key[3] % 1_000_000_000 == 0 and key[4] % 1_000_000_000 == 0
                            and time.time_ns() - max(key[3], key[4]) < 2_000_000_000)
            if entry is not None and coarse_fresh:
                self._invalidate_locked(path)
                self.counters['fresh_coarse_timestamp_bypasses'] += 1
                entry = None
            if entry is not None and entry['stat'] != key:
                self._invalidate_locked(path)
                entry = None
            if entry is not None and (not want_json or 'json' in entry):
                self.counters['json_hits' if want_json else 'sha_hits'] += 1
                if 'json' in entry:
                    self._json_order.move_to_end(path)
                return dict(entry)
            return None

    def _load(self, path, want_json):
        path = self._path(path)
        with self._lock(path):
            key = stat_key(path)
            hit = self._cached(path, key, want_json)
            if hit is not None:
                return hit
            for _ in range(4):
                before = stat_key(path)
                digest = hashlib.sha256()
                chunks = [] if want_json else None
                with open(path, 'rb') as f:
                    opened = os.fstat(f.fileno())
                    opened_key = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
                    for block in iter(lambda: f.read(8 << 20), b''):
                        digest.update(block)
                        if want_json:
                            chunks.append(block)
                    closed = os.fstat(f.fileno())
                    closed_key = (closed.st_dev, closed.st_ino, closed.st_size, closed.st_mtime_ns, closed.st_ctime_ns)
                after = stat_key(path)
                if before == opened_key == closed_key == after:
                    break
            else:
                raise RuntimeError(f'Input changed repeatedly during verified read: {path}')
            entry = {'stat': after, 'sha256': digest.hexdigest(), 'bytes': after[2]}
            if want_json:
                entry['json'] = json.loads(b''.join(chunks))
            with self._guard:
                old = self._entries.get(path)
                if old is not None and old['stat'] == after and old['sha256'] != entry['sha256']:
                    raise ValueError(f'Bytes changed without an identity change: {path}')
                if old is not None:
                    self._invalidate_locked(path)
                self._entries[path] = entry
                # Hold the decoded value independently of later LRU eviction,
                # including an individual JSON larger than the cache limit.
                returned = dict(entry)
                self.counters['physical_json_reads' if want_json else 'physical_sha_reads'] += 1
                self.counters['verified_bytes_read'] += after[2]
                if want_json:
                    self._json_order[path] = None
                    self._json_bytes += entry['bytes']
                    while self._json_bytes > self._json_limit and self._json_order:
                        victim, _ = self._json_order.popitem(last=False)
                        cached = self._entries[victim]
                        self._json_bytes -= cached['bytes']
                        # Preserve verified hashes after evicting decoded JSON.
                        cached.pop('json', None)
                        self.counters['json_evictions'] += 1
            return returned

    def read(self, path):
        # Every read has independent nested objects. Cached records cannot be
        # mutated by the unchanged generator or cumulative selection code.
        entry = self._load(path, True)
        return copy.deepcopy(entry['json'])

    def sha(self, path):
        return self._load(path, False)['sha256']

    def atomic(self, path, value):
        numeric.atomic(path, value)
        self.invalidate(path)

    def statistics(self):
        with self._guard:
            return dict(self.counters, cached_files=len(self._entries),
                        cached_json_files=len(self._json_order), cached_json_source_bytes=self._json_bytes,
                        json_cache_limit_source_bytes=self._json_limit,
                        stat_identity=['device', 'inode', 'size', 'mtime_ns', 'ctime_ns'],
                        coarse_timestamp_safety_seconds=2)


class PrivateController:
    SCIENCE = ('configuration_tasks', 'initialize', 'aggregate_receipts', 'summarize', 'generate_next', 'advance')

    def __init__(self, original_path, source_root, *, threads=16, json_cache_mib=1024,
                 cached=True, fixture_stamp=None):
        if not 1 <= threads <= 16:
            raise ValueError('I/O prewarming is bounded to 1..16 threads')
        self.original_path = Path(original_path).resolve()
        self.source_root = Path(source_root).resolve()
        self.threads = threads
        self.io = VerifiedIO(json_cache_mib)
        name = '_private_original_gt_controller_' + uuid.uuid4().hex
        specification = importlib.util.spec_from_file_location(name, self.original_path)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        self.original_module = module
        ns = dict(module.__dict__)
        numeric_ns = dict(numeric.__dict__)
        if cached:
            for scope in (ns, numeric_ns):
                scope.update(read=self.io.read, sha=self.io.sha, atomic=self.io.atomic)
        # The descriptor and function loader have their own module globals. Give
        # them private I/O globals too; do not mutate gt_numeric or the original.
        for name in ('descriptor', 'bind_function', 'arrays_at'):
            function = getattr(numeric, name)
            ns[name] = FunctionType(function.__code__, numeric_ns, function.__name__,
                                    function.__defaults__, function.__closure__)
        if fixture_stamp is not None:
            ns['stamp'] = lambda: fixture_stamp
        for name, function in module.__dict__.items():
            if isinstance(function, FunctionType) and function.__module__ == module.__name__:
                ns[name] = FunctionType(function.__code__, ns, function.__name__,
                                        function.__defaults__, function.__closure__)
        self.namespace = ns
        self.cached = cached
        sources = [self.original_path, Path(numeric.__file__).resolve(), Path(__file__).resolve(),
                   self.source_root / numeric.G5 / 'g5_runtime.py',
                   self.source_root / numeric.G5 / 'g5_next_stage.py']
        self.protected_sources = {str(p): numeric.sha(p) for p in sources}
        self.identity = {'schema': 'gt_controller_private_verified_io_v1',
                         'original_controller': {'path': str(self.original_path), 'sha256': numeric.sha(self.original_path)},
                         'original_numeric': {'path': str(Path(numeric.__file__).resolve()), 'sha256': numeric.sha(numeric.__file__)},
                         'wrapper_transport': {'path': str(Path(__file__).resolve()), 'sha256': numeric.sha(__file__)},
                         'original_science_functions': {n: function_identity(getattr(module, n)) for n in self.SCIENCE},
                         'rebound_science_functions': {n: function_identity(ns[n]) for n in self.SCIENCE},
                         'descriptor': function_identity(numeric.descriptor),
                         'source_function_loader': function_identity(numeric.bind_function),
                         'protected_source_sha256': self.protected_sources, 'io_threads': threads,
                         'rebindings': ['read', 'sha', 'atomic', 'descriptor private read/sha', 'bind_function private sha'],
                         'science_code_unchanged': all(ns[n].__code__ is getattr(module, n).__code__ for n in self.SCIENCE),
                         'cached_JSON_always_deepcopy': True, 'GPU_workers_modified': False,
                         'fixture_fixed_timestamp': fixture_stamp}
        if module.read is not numeric.read or module.sha is not numeric.sha:
            raise AssertionError('Original module globals were accidentally modified')

    def call(self, name, *args, **kwargs):
        return self.namespace[name](*args, **kwargs)

    def assert_sources_unchanged(self):
        for path, expected in self.protected_sources.items():
            if numeric.sha(path) != expected:
                raise ValueError(f'Controller/source function bytes changed after binding: {path}')

    def prewarm(self, root, stage):
        root = Path(root).resolve()
        began = time.monotonic()
        tasks = self.io.read(root / 'runs' / stage / 'TASKS.json')
        jobs = [(scene, position, config) for scene in tasks for position, config in enumerate(scene['configs'])]
        def warm(job):
            scene, position, config = job
            receipt = root / 'runs' / stage / 'configuration_receipts' / f"{scene['id']}__c{position:04d}.json"
            if not receipt.exists():
                return False
            value = self.io.read(receipt)
            if value['status'] != 'COMPLETE' or value['config_sha256'] != numeric.digest(config):
                raise ValueError('Configuration receipt identity mismatch')
            result_path = root / value['result_path']
            result = self.io.read(result_path)
            if self.io.sha(result_path) != value['result_sha256']:
                raise ValueError('Configuration RESULT SHA mismatch')
            array = result['arrays']
            if self.io.sha(root / array['path']) != array['sha256']:
                raise ValueError('GT descriptor array SHA mismatch')
            return True
        with ThreadPoolExecutor(max_workers=self.threads) as pool:
            warmed = sum(pool.map(warm, jobs))
        return {'stage': stage, 'configuration_receipts_verified': warmed,
                'configuration_tasks': len(jobs), 'seconds': time.monotonic() - began,
                'cache': self.io.statistics()}


def _fixture(root, source_root):
    import numpy as np
    source = source_root / numeric.G5 / 'g5_runtime.py'
    configurations, _ = numeric.bind_function(source, 'configurations', {'Path': Path})
    rows = [{'global_index': i, 'token': f'{i:016x}', 'session': 'fixture_log',
             'subset': 'exploration', 'legacy_best': None, 'legacy_configurations': {}}
            for i in (11, 22, 33, 44)]
    numeric.atomic(root / 'manifests/SCENES.json', rows)
    tasks = [{'id': f"scene_{r['global_index']:06d}", 'kind': 'baseline', 'index': r['global_index'],
              'configs': configurations()} for r in rows]
    numeric.atomic(root / 'runs/baseline/TASKS.json', tasks)
    numeric.atomic(root / 'runs/QUEUE.json', {'stages': ['baseline'], 'active': True})
    def populate(stage, tasks):
        for scene in tasks:
            for position, config in enumerate(scene['configs']):
                block = config['block']
                maxima = {11: (.8, .8), 22: (.96, .75), 33: (.92, .98), 44: (.6, .95)}
                score = maxima[scene['index']][block == 'interface_post']
                if stage != 'baseline' and position == 0:
                    score += .01
                folder = root / 'analysis' / stage / scene['id'] / config['id']
                folder.mkdir(parents=True, exist_ok=True)
                array_path = folder / 'CANDIDATES.npz'
                np.savez(array_path, values=np.full((2, 4), scene['index'] + position, np.float32))
                records = [{'arm': 'GT_reference', 'score': {'score': .8 if scene['index'] == 44 else .99}},
                           {'arm': 'baseline', 'iteration': 0, 'surrogate_loss': 2., 'score': {'score': .1}},
                           {'arm': 'optimized', 'iteration': 1, 'surrogate_loss': 1. + position / 100., 'score': {'score': score}}]
                result = {'status': 'COMPLETE', 'config': config, 'block': block, 'records': records,
                          'selected_index': 2, 'transfer': [{'seed_index': 1, 'score': {'score': .12}}],
                          'arrays': {'path': str(array_path.relative_to(root)), 'sha256': numeric.sha(array_path)}}
                result_path = folder / 'RESULT.json'
                numeric.atomic(result_path, result)
                receipt = {'status': 'COMPLETE', 'config_sha256': numeric.digest(config),
                           'result_path': str(result_path.relative_to(root)), 'result_sha256': numeric.sha(result_path)}
                numeric.atomic(root / 'runs' / stage / 'configuration_receipts' / f"{scene['id']}__c{position:04d}.json", receipt)
    populate('baseline', tasks)
    return tasks, populate


def _json_tree(root, relative):
    folder = Path(root) / relative
    return {str(p.relative_to(root)): numeric.read(p) for p in sorted(folder.rglob('*.json'))}


def selftest(args):
    from contextlib import redirect_stdout
    source_root = Path(args.source_root).resolve()
    folder = Path(args.selftest_dir or HERE.parent / 'engineering' / ('gt_fast_selftest_' + str(os.getpid()) + '_' + uuid.uuid4().hex[:8])).resolve()
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Self-test directory must be new or empty')
    root = folder / 'private_fixture'; root.mkdir(parents=True)
    original = PrivateController(args.original_controller, source_root, threads=args.io_threads, cached=False, fixture_stamp='SELFTEST_FIXED_TIME')
    fast = PrivateController(args.original_controller, source_root, threads=args.io_threads, fixture_stamp='SELFTEST_FIXED_TIME')
    tasks, populate = _fixture(root, source_root)
    original.call('configuration_tasks', root, 'baseline')
    original_progress = original.call('aggregate_receipts', root, 'baseline')
    original_receipts = _json_tree(root, 'runs/baseline/receipts')
    shutil.rmtree(root / 'runs/baseline/receipts')
    warm = fast.prewarm(root, 'baseline')
    fast_progress = fast.call('aggregate_receipts', root, 'baseline')
    assert original_progress == fast_progress
    assert original_receipts == _json_tree(root, 'runs/baseline/receipts')
    reference_summary = original.call('summarize', root, 'baseline')
    reference_outputs = {str(p.relative_to(root)): numeric.read(p) for p in
                         [root / 'analysis/summary_baseline/SUMMARY.json', root / 'analysis/GT_exception_interim.json', root / 'runs/baseline/ACCEPTED.json']}
    for name in reference_outputs:
        (root / name).unlink()
    accelerated_summary = fast.call('summarize', root, 'baseline')
    assert reference_summary == accelerated_summary
    assert reference_outputs == {name: numeric.read(root / name) for name in reference_outputs}
    generation = []
    for stage in numeric.STAGES[1:] + ['export_post']:
        queue_path = root / 'runs/QUEUE.json'
        numeric.atomic(queue_path, {'stages': ['baseline'], 'active': True})
        with redirect_stdout(io.StringIO()):
            original.call('generate_next', root, source_root, stage, 'baseline')
        expected = _json_tree(root, 'runs/' + stage)
        expected_queue = numeric.read(queue_path)
        shutil.rmtree(root / 'runs' / stage)
        numeric.atomic(queue_path, {'stages': ['baseline'], 'active': True})
        with redirect_stdout(io.StringIO()):
            fast.call('generate_next', root, source_root, stage, 'baseline')
        assert expected == _json_tree(root, 'runs/' + stage), stage
        assert expected_queue == numeric.read(queue_path), stage
        generation.append({'stage': stage, 'JSON_files_exact': len(expected),
                           'configuration_tasks': expected.get('runs/' + stage + '/CONFIGURATION_QUEUE.json', {}).get('configuration_tasks')})
    # Exercise cumulative strict-improvement/tie behavior, not just baseline.
    stage = 'strength_A'
    populate(stage, numeric.read(root / 'runs' / stage / 'TASKS.json'))
    original.call('aggregate_receipts', root, stage)
    expected_summary = original.call('summarize', root, stage)
    (root / 'analysis/summary_strength_A/SUMMARY.json').unlink()
    (root / 'runs/strength_A/ACCEPTED.json').unlink()
    fast.prewarm(root, stage)
    assert expected_summary == fast.call('summarize', root, stage)
    # Deepcopy independence and invalidation after atomic replacement/in-place
    # source changes (including restoring mtime) are explicit contract tests.
    scratch = root / 'scratch.json'; numeric.atomic(scratch, {'v': [1]})
    first = fast.io.read(scratch); first['v'].append(99)
    assert fast.io.read(scratch) == {'v': [1]}
    before = fast.io.sha(scratch)
    numeric.atomic(scratch, {'v': [2]})
    assert fast.io.read(scratch) == {'v': [2]} and fast.io.sha(scratch) != before
    old_stat = scratch.stat(); payload = scratch.read_bytes().replace(b'2', b'3')
    scratch.write_bytes(payload); os.utime(scratch, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    assert fast.io.read(scratch) == {'v': [3]}
    large = root / 'large_private_cache_test.json'
    numeric.atomic(large, {'value': 'x' * (2 * 1024 * 1024)})
    small_cache = VerifiedIO(json_cache_mib=1)
    assert small_cache.read(large) == small_cache.read(large) == numeric.read(large)
    protected = root / 'private_source.py'; protected.write_text('value = 1\n')
    fast.protected_sources[str(protected)] = fast.io.sha(protected)
    protected.write_text('value = 2\n')
    try:
        fast.assert_sources_unchanged()
    except ValueError:
        source_change_rejected = True
    else:
        raise AssertionError('Bound source mutation was not rejected')
    fast.protected_sources.pop(str(protected))
    result_path = next((root / 'analysis/baseline').glob('*/*/RESULT.json'))
    result = numeric.read(result_path); array = root / result['arrays']['path']; good = array.read_bytes()
    fast.io.sha(array); array.write_bytes(good + b'CORRUPTION')
    for check in [lambda: fast.prewarm(root, 'baseline'), lambda: original.call('descriptor', root, result_path)]:
        try:
            check()
        except ValueError:
            pass
        else:
            raise AssertionError('Corrupted array file was accepted')
    array.write_bytes(good)
    fast.prewarm(root, 'baseline')
    fast.assert_sources_unchanged()
    report = {'status': 'PASS_TRANSPORT_SELFTEST_ONLY_NO_GPU_OR_SCIENTIFIC_RESULTS',
              'private_fixture': str(root), 'original_vs_fast_aggregate_exact': True,
              'original_vs_fast_summary_exact': True, 'cumulative_strict_selection_exact': True,
              'generated_tasks_JSON_semantics_exact': generation, 'cache_JSON_deepcopy_verified': True,
              'atomic_replace_invalidates': True, 'inplace_change_with_restored_mtime_invalidates': True,
              'coarse_NFS_timestamp_freshness_bypass_verified': True, 'oversized_JSON_eviction_safe': True,
              'bound_source_change_rejected': source_change_rejected, 'array_corruption_rejected': True,
              'identity': fast.identity, 'prewarm': warm, 'cache': fast.io.statistics()}
    numeric.atomic(folder / 'SELFTEST.json', report)
    print(json.dumps({'status': report['status'], 'report': str(folder / 'SELFTEST.json'),
                      'generated_stage_comparisons': len(generation)}, sort_keys=True), flush=True)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', nargs='?', choices=['init','advance','status','watch','prewarm'], default='watch')
    p.add_argument('--root')
    p.add_argument('--source-root', default=str(HERE.parent / 'source_snapshot/ReCogDrive'))
    p.add_argument('--original-controller', default=str(HERE / 'gt_opt_controller.py'))
    p.add_argument('--train-manifest'); p.add_argument('--paths')
    p.add_argument('--metric-registry', default='/mnt/project/recogdrive-first42-a800-20261022/LOCAL_METRIC_REGISTRY.json')
    p.add_argument('--poll-seconds', type=float, default=30)
    p.add_argument('--deadline-unix', type=float)
    p.add_argument('--io-threads', type=int, default=16)
    p.add_argument('--json-cache-mib', type=int, default=1024)
    p.add_argument('--selftest', action='store_true'); p.add_argument('--selftest-dir')
    p.add_argument('--report', help='Read-only prewarm report path, outside live GT outputs')
    args = p.parse_args(argv)
    if not 1 <= args.io_threads <= 16 or args.json_cache_mib < 1 or not 0 < args.poll_seconds <= 60:
        p.error('Require 1..16 I/O threads, positive JSON cache size and poll interval <=60s')
    if args.selftest:
        return selftest(args)
    if not args.root:
        p.error('--root is required outside --selftest')
    root, source = Path(args.root).resolve(), Path(args.source_root).resolve()
    controller = PrivateController(args.original_controller, source, threads=args.io_threads, json_cache_mib=args.json_cache_mib)
    controller.assert_sources_unchanged()
    if args.command == 'prewarm':
        queue = controller.io.read(root / 'runs/QUEUE.json')
        result = controller.prewarm(root, queue['stages'][-1])
        if args.report:
            numeric.atomic(args.report, {'identity': controller.identity, **result})
        print(json.dumps(result, sort_keys=True), flush=True)
        return result
    # Same original single-controller flock. This wrapper never replaces,
    # signals, stops or restarts an existing owner or any GPU worker.
    owner = controller.call('lock_root', root)
    numeric.atomic(root / 'runs/FAST_CONTROLLER_IO_IDENTITY.json', controller.identity)
    try:
        if args.command == 'init':
            if not args.train_manifest or not args.paths:
                p.error('init requires --train-manifest and --paths')
            controller.call('initialize', args, root)
            controller.prewarm(root, 'baseline')
            result = controller.call('aggregate_receipts', root, 'baseline')
        elif args.command == 'status':
            queue = controller.io.read(root / 'runs/QUEUE.json')
            stages = []
            for stage in queue['stages']:
                controller.prewarm(root, stage)
                stages.append(controller.call('aggregate_receipts', root, stage))
            result = {'active': queue['active'], 'stages': stages}
        else:
            while True:
                controller.assert_sources_unchanged()
                queue = controller.io.read(root / 'runs/QUEUE.json')
                if args.command == 'watch' and (not queue['active'] or (root / 'STOP_NEW_CLAIMS.json').exists() or
                   (args.deadline_unix and time.time() >= args.deadline_unix)):
                    result = controller.io.read(root / 'CONTROLLER_STATUS.json') if (root / 'CONTROLLER_STATUS.json').exists() else {'status':'WAITING'}
                    break
                warm = controller.prewarm(root, queue['stages'][-1])
                result = controller.call('advance', root, source)
                numeric.atomic(root / 'CONTROLLER_STATUS.json', {'time': numeric.stamp(), **result})
                numeric.atomic(root / 'runs/FAST_CONTROLLER_IO_STATS.json', {'time': numeric.stamp(), 'prewarm': warm,
                               'cache': controller.io.statistics(), 'wrapper_sha256': controller.identity['wrapper_transport']['sha256']})
                print(json.dumps(result, sort_keys=True), flush=True)
                if args.command != 'watch' or result.get('status') == 'GT_POST_COMPLETE':
                    break
                time.sleep(args.poll_seconds)
        print(json.dumps(result, sort_keys=True), flush=True)
        return result
    finally:
        owner.close()


if __name__ == '__main__':
    main()
