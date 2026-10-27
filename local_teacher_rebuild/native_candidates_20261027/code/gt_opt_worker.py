"""Persistent one-head worker with durable per-configuration claims/receipts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import socket
import time
import traceback

from gt_numeric import (LocalGTNumeric, ReferenceNoise, atomic, digest, read,
                        sha, stamp)


def process_start_ticks():
    return int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])


def claim_ready(root, base_root, rows, worker_id, max_attempts):
    queue = read(root / 'runs/QUEUE.json')
    if not queue['active']:
        return None, None
    stage = queue['stages'][-1]
    candidates = list((root / 'runs' / stage / 'configuration_tasks').glob('*.json'))
    pending = root / 'runs' / stage / 'REUSE_PENDING.json'
    if pending.exists() and not (pending.parent / 'REUSE_IMPORT_COMPLETE.json').exists():
        blocked = set(read(pending)['blocked_old_intersection_indices'])
        candidates = [p for p in candidates if int(p.stem.split('__')[0].replace('scene_', '')) not in blocked]
    # Stable worker-specific rotation avoids every GPU contending on the same
    # first configuration, while all tasks remain eligible on every worker.
    candidates.sort(key=lambda path: digest([worker_id, path.name]))
    for path in candidates:
        receipt = root / 'runs' / stage / 'configuration_receipts' / path.name
        if receipt.exists():
            continue
        task = read(path)
        row = rows[task['index']]
        if not (base_root / 'train/cases' / row['token'] / 'COMPLETE.json').exists():
            continue
        failures = root / 'runs' / stage / 'failures' / task['id']
        if len(list(failures.glob('*.json'))) >= max_attempts:
            continue
        claim = root / 'runs' / stage / 'claims' / path.name
        claim.parent.mkdir(parents=True, exist_ok=True)
        stream = claim.open('a+')
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close()
            continue
        if receipt.exists():
            stream.close()
            continue
        if len(list(failures.glob('*.json'))) >= max_attempts:
            stream.close()
            continue
        stream.seek(0)
        stream.truncate()
        json.dump({'worker': worker_id, 'pid': os.getpid(), 'process_start_ticks': process_start_ticks(),
                   'hostname': socket.gethostname(), 'time': stamp(), 'task_sha256': sha(path)}, stream)
        stream.flush()
        os.fsync(stream.fileno())
        return task, stream
    return None, None


def score_pool_setup(frozen_repo, paths, root):
    """Bootstrap already happened in LocalGTNumeric; scoring uses same modules."""
    from repair_score import score_group
    return score_group


def run(args):
    root, base_root = Path(args.root).resolve(), Path(args.base_root).resolve()
    worker_id = args.worker_id or f'{socket.gethostname()}-gpu{os.environ.get("CUDA_VISIBLE_DEVICES", "0")}-pid{os.getpid()}'
    status = root / 'workers' / worker_id / 'STATUS.json'
    rows = {int(row['global_index']): row for row in read(root / 'manifests/SCENES.json')}
    protocol = read(root / 'PROTOCOL.json')
    if len(rows) != protocol['candidate_count'] or any(row['population'] != 'train' or row['subset'] != 'exploration' for row in rows.values()):
        raise ValueError('GT worker must preserve the complete verified local candidate manifest')
    if sha(protocol['local_candidates']['path']) != protocol['local_candidates']['sha256']:
        raise ValueError('Local candidate manifest changed')
    atomic(status, {'status': 'VALIDATING_REFERENCE_NOISE', 'time': stamp(), 'pid': os.getpid(),
                    'process_start_ticks': process_start_ticks(), 'worker_id': worker_id})
    noise = ReferenceNoise(args.noise_protocol, args.noise_index, args.noise_bank)
    engine = LocalGTNumeric(root, args.source_root, args.frozen_repo, args.paths, base_root, noise)
    atomic(root / 'workers' / worker_id / 'NUMERICS_BINDINGS.json', engine.identity)
    completed = failed = 0
    began = time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as pool:
        while True:
            if args.deadline_unix and time.time() >= args.deadline_unix:
                break
            if (root / 'STOP_NEW_CLAIMS.json').exists():
                break
            task, owned = claim_ready(root, base_root, rows, worker_id, args.max_attempts)
            if task is None:
                queue = read(root / 'runs/QUEUE.json')
                atomic(status, {'status': 'WAITING_READY_CONFIGURATION' if queue['active'] else 'SEARCH_COMPLETE',
                                'time': stamp(), 'completed': completed, 'failed': failed, 'worker_id': worker_id,
                                'pid': os.getpid(), 'process_start_ticks': process_start_ticks()})
                if args.once or not queue['active']:
                    break
                time.sleep(args.poll_seconds)
                continue
            stage, config = task['kind'], task['config']
            destination = (root / 'exports/post' / rows[task['index']]['token'] if stage == 'export_post' else
                           root / 'analysis' / stage / task['scene_id'] / config['id'])
            started = time.monotonic()
            atomic(status, {'status': 'OPTIMIZING', 'time': stamp(), 'task': task, 'completed': completed,
                            'failed': failed, 'worker_id': worker_id, 'pid': os.getpid(),
                            'process_start_ticks': process_start_ticks()})
            try:
                result_path = destination / ('COMPLETE.json' if stage == 'export_post' else 'RESULT.json')
                if result_path.exists():
                    result = read(result_path)
                    if stage == 'export_post':
                        valid = result['status'] == 'PASS_LOCAL_FROZEN_BASE_GT_POST_PAIR' and result['selection'] == config
                    else:
                        valid = result['status'] == 'COMPLETE' and result['config'] == config
                    if not valid:
                        raise ValueError('Existing config result does not match its original task')
                    array = root / result['arrays']['path']
                    if sha(array) != result['arrays']['sha256']:
                        raise ValueError('Existing config arrays SHA mismatch')
                else:
                    if stage == 'export_post':
                        result = engine.export_selected(rows[task['index']], config, pool, destination)
                    else:
                        result = engine.run_config(rows[task['index']], config, pool, destination)
                receipt = {'status': 'COMPLETE', 'time': stamp(), 'task_id': task['id'],
                           'task_sha256': digest(task), 'config_sha256': digest(config),
                           'result_path': str(result_path.relative_to(root)), 'result_sha256': sha(result_path),
                           'worker': worker_id, 'seconds': time.monotonic() - started,
                           'numerics_binding_sha256': digest(engine.identity),
                           'selected_on': 'actual S0 PDMS; transfer S1/S2 never selection'}
                atomic(root / 'runs' / stage / 'configuration_receipts' / (task['id'] + '.json'), receipt)
                completed += 1
            except Exception:
                failed += 1
                folder = root / 'runs' / stage / 'failures' / task['id']
                folder.mkdir(parents=True, exist_ok=True)
                attempt = len(list(folder.glob('*.json'))) + 1
                atomic(folder / f'attempt_{attempt:03d}.json',
                       {'status': 'FAILED_CONFIGURATION', 'time': stamp(), 'task': task, 'worker': worker_id,
                        'traceback': traceback.format_exc(), 'seconds': time.monotonic() - started,
                        'retry_boundary': 'same original configuration; no Adam state carried from a failed attempt'})
                print(traceback.format_exc(), flush=True)
                import torch
                torch.cuda.empty_cache()
                if args.fail_fast:
                    raise
            finally:
                owned.close()
            if args.once or (args.max_tasks and completed + failed >= args.max_tasks):
                break
    atomic(status, {'status': 'EXITED', 'time': stamp(), 'worker_id': worker_id, 'completed': completed,
                    'failed': failed, 'seconds': time.monotonic() - began, 'pid': os.getpid(),
                    'process_start_ticks': process_start_ticks()})
    return {'worker': worker_id, 'completed': completed, 'failed': failed}


def main():
    parser = argparse.ArgumentParser()
    for name in ['root', 'source-root', 'frozen-repo', 'paths', 'base-root', 'noise-protocol', 'noise-index']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--noise-bank')
    parser.add_argument('--worker-id')
    parser.add_argument('--max-tasks', type=int, default=0)
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--poll-seconds', type=float, default=5)
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--fail-fast', action='store_true')
    args = parser.parse_args()
    if args.max_attempts < 1 or not 0 < args.poll_seconds <= 60:
        parser.error('require max-attempts>=1 and poll-seconds in (0,60]')
    print(json.dumps(run(args), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
