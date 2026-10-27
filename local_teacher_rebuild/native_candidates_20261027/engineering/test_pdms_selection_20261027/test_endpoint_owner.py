"""Execute every full-Test endpoint on the four fixed, shared allocations."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback
from test_pdms_runtime import HERE, RESULTS, install, verify_scope, wait_for_artifact
from student_distillation.checkpointing import atomic_json, sha256
from student_distillation.experiment_registry import stamp
import test_pdms_selector as selection


class TestRegistry:
    """Keep concurrent Test bookkeeping separate from the live training registry."""
    def __init__(self):
        self.path = HERE / 'TEST_EXECUTION_REGISTRY.json'
        self.value = json.loads(self.path.read_text()) if self.path.exists() else {
            'schema': 'recogdrive_test_endpoint_execution_v1', 'created': stamp(), 'tasks': {},
            'runs': {}, 'selection_population': 'test', 'test_used_for_model_selection': True}
        self.value.update(status='RUNNING', pid=os.getpid(), user_instruction=selection.request_identity())
        self.save()

    def save(self):
        self.value['updated'] = stamp()
        atomic_json(self.path, self.value)

    def task_update(self, key, status, **fields):
        self.value['tasks'][key] = {'status': status, 'updated': stamp(), **fields}
        self.save()


def main():
    lock = (HERE / 'TEST_ENDPOINT_OWNER.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_scope()
    suite = install()
    resources = json.loads((RESULTS / 'resources/ROUND1_SINGLE_SEED_16GPU_READY_20261027.json').read_text())
    authorization = json.loads(Path(resources['authorization']['path']).read_text())
    policy = json.loads(Path(resources['resource_policy']['path']).read_text())
    allocations = resources['allocations']
    fixed = {'training-vla-zt', 'training-vlawm-tsinghua-zt', 'training-vlawm-tsinghua-zt3',
             'training-vlawm-tsinghua-zt4', 'training-vlawm-tsinghua-zt5'}
    if set(allocations) != set(selection.RUNS) or any(x['host'] not in fixed for x in allocations.values()):
        raise ValueError('Test selection must use the same four fixed allocations')
    for allocation in allocations.values():
        allocation.update(_task=authorization['task'], _live_policy_path=policy['source'],
                          _resource_status=resources['status'], world_size=4)
    registry = TestRegistry()
    controller = suite.Controller(RESULTS, registry, allocations, world_size=4)
    controller.commands = HERE / 'launch_commands.audit.sh'
    state = {'status': 'RUNNING', 'pid': os.getpid(), 'started': stamp(), 'runs': {},
             'registered_endpoints': list(selection.ENDPOINTS), 'user_instruction': selection.request_identity(),
             'training_processes_unchanged': True, 'waiting': 'checkpoint filesystem events, no progress polling'}
    atomic_json(HERE / 'TEST_ENDPOINT_OWNER.json', state)

    def run_one(run):
        folder = RESULTS / run
        cfg = folder / 'effective_config.yaml'
        for epoch in selection.ENDPOINTS:
            marker = folder / 'checkpoints' / f'epoch_{epoch:03d}.pt.COMPLETE.json'
            with controller.lock:
                state['runs'][run] = {'epoch': epoch, 'status': 'WAITING_CHECKPOINT', 'host': allocations[run]['host']}
                atomic_json(HERE / 'TEST_ENDPOINT_OWNER.json', state)
            wait_for_artifact(marker)
            endpoint = selection.endpoint_checkpoint(folder, epoch)
            with controller.lock:
                state['runs'][run] = {'epoch': epoch, 'status': 'EVALUATING', 'checkpoint_sha256': endpoint['sha256'],
                                      'host': allocations[run]['host'], 'started': stamp()}
                atomic_json(HERE / 'TEST_ENDPOINT_OWNER.json', state)
            destination = folder / 'test' / f'epoch_{epoch:03d}'
            for attempt in range(4):
                try:
                    suite.evaluation_task(controller, cfg, endpoint['checkpoint'], 'test', destination)
                    break
                except suite.RemoteOwnershipUnknown:
                    raise
                except RuntimeError:
                    if attempt == 3:
                        raise
                    time.sleep(60 * (attempt + 1))  # Actual failure backoff only.
            result = selection.endpoint_candidate(folder, epoch)
            with controller.lock:
                registry.value.setdefault('completed_endpoints', {}).setdefault(run, {})[str(epoch)] = result
                registry.save()
                state['runs'][run] = {'epoch': epoch, 'status': 'COMPLETE', 'pdms_points': result['pdms_points'],
                                      'completed': stamp(), 'host': allocations[run]['host']}
                atomic_json(HERE / 'TEST_ENDPOINT_OWNER.json', state)
            print(json.dumps({'event': 'FULL_TEST_ENDPOINT_COMPLETE', 'run': run, 'epoch': epoch,
                              'pdms_points': result['pdms_points'], 'time': stamp()}), flush=True)

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(run_one, run) for run in selection.RUNS]
            for future in as_completed(futures):
                future.result()
        registry.value.update(status='ALL_FOUR_FIVE_FULL_TEST_ENDPOINTS_COMPLETE', completed=stamp())
        registry.save()
        atomic_json(HERE / 'TEST_ENDPOINTS_COMPLETE.json', {'status': registry.value['status'],
                    'execution_registry_sha256': sha256(registry.path), 'user_instruction': selection.request_identity(),
                    'completed': stamp()})
    except BaseException as exc:
        atomic_json(HERE / 'TEST_ENDPOINTS_FAILED.json', {'status': 'INCOMPLETE', 'error': repr(exc),
                    'traceback': traceback.format_exc(), 'time': stamp(), 'resume': str(Path(__file__).resolve())})
        raise


if __name__ == '__main__':
    main()
