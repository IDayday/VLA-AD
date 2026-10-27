"""Complete all selected A Test/loss evaluations and the actual correlation report."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from correlation_runtime import HERE, RESULTS, RUN, EPOCHS, EXTRA_TEST_EPOCHS, install, checkpoint_identity
from student_distillation.checkpointing import atomic_json, sha256
from student_distillation.experiment_registry import stamp


class Registry:
    def __init__(self):
        self.path = HERE / 'EXECUTION_REGISTRY.json'
        self.value = json.loads(self.path.read_text()) if self.path.exists() else {
            'schema': 'nrkd_loss_PDMS_execution_v1', 'tasks': {}, 'runs': {}, 'created': stamp()}
        self.value.update(status='RUNNING', pid=os.getpid(), started=stamp())
        self.save()
    def save(self):
        atomic_json(self.path, self.value)
    def task_update(self, key, status, **fields):
        self.value['tasks'][key] = {'status': status, 'updated': stamp(), **fields}
        self.save()


def main():
    lease = (HERE / 'OWNER.lock').open('a+')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    suite = install()
    resources = json.loads((RESULTS / 'resources/ROUND1_SINGLE_SEED_16GPU_READY_20261027.json').read_text())
    authorization = json.loads(Path(resources['authorization']['path']).read_text())
    policy = json.loads(Path(resources['resource_policy']['path']).read_text())
    registry, lock = Registry(), threading.RLock()
    # Additional Test work is balanced across the other three existing fixed
    # allocations. A's formal Test8/16 owner stays the sole producer for those.
    pins = {1: 'H_gain1_seed20261027', 4: 'H_mu_teacher_seed20261027',
            12: 'H_mu_student_seed20261027', 8: 'H_full_seed20261027', 16: 'H_full_seed20261027'}
    controllers = {}
    for epoch, resource_run in pins.items():
        allocation = dict(resources['allocations'][resource_run])
        allocation.update(_task=authorization['task'], _live_policy_path=policy['source'],
                          _resource_status=resources['status'], world_size=4)
        controller = suite.Controller(RESULTS, registry, {RUN.name: allocation}, world_size=4)
        controller.lock, controller.commands = lock, HERE / 'launch_commands.audit.sh'
        controllers[epoch] = controller
    atomic_json(HERE / 'ACTUAL_OWNER.json', {'status': 'RUNNING', 'pid': os.getpid(), 'started': stamp(),
                'epochs': list(EPOCHS), 'extra_Test_epochs': list(EXTRA_TEST_EPOCHS),
                'resource_pins': {str(k): resources['allocations'][v]['host'] for k,v in pins.items()},
                'original_four_training_and_primary_Test_owners_unchanged': True})

    def evaluate(epoch, test):
        endpoint = checkpoint_identity(epoch)
        folder = RUN / 'diagnostics/loss_pdms_correlation' / f'epoch_{epoch:03d}'
        destination = folder / ('test' if test else 'train_panel_representation')
        cfg = RUN / 'effective_config.yaml'
        for attempt in range(4):
            try:
                if test:
                    suite.evaluation_task(controllers[epoch], cfg, endpoint['checkpoint'], 'test', destination)
                else:
                    suite.evaluation_task(controllers[epoch], cfg, endpoint['checkpoint'], 'train', destination,
                                          target='full', panel=True, representation_only=True)
                break
            except suite.RemoteOwnershipUnknown:
                raise
            except RuntimeError:
                if attempt == 3:
                    raise
                time.sleep(60 * (attempt + 1))
        print(json.dumps({'event': 'CORRELATION_EVALUATION_COMPLETE', 'epoch': epoch,
                          'kind': 'Test' if test else 'fixed_Train8192_H', 'time': stamp()}), flush=True)

    def extra_point(epoch):
        evaluate(epoch, True)
        evaluate(epoch, False)

    def reused_points():
        # These cheap, noise-free H runs can overlap the ongoing full Test.
        for epoch in (8, 16):
            evaluate(epoch, False)

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(extra_point, epoch) for epoch in EXTRA_TEST_EPOCHS]
            futures.append(pool.submit(reused_points))
            for future in as_completed(futures):
                future.result()
        from test_pdms_runtime import wait_for_artifact
        for epoch in (8, 16):
            wait_for_artifact(RUN / 'test' / f'epoch_{epoch:03d}' / 'COMPLETE.json')
        command = [sys.executable, '-B', str(HERE / 'correlation_analysis.py'),
                   '--manifest', str(HERE / 'INPUT_MANIFEST.json'), '--out', str(HERE / 'analysis')]
        analysis_freeze = json.loads((HERE / 'ANALYSIS_CODE_FREEZE.json').read_text())
        for name, checksum in analysis_freeze['files'].items():
            if sha256(HERE / name) != checksum:
                raise ValueError('Correlation analysis source changed after its freeze: ' + name)
        subprocess.run(command, check=True, cwd=HERE)
        registry.value.update(status='ALL_EVALUATIONS_AND_CORRELATION_REPORT_COMPLETE', completed=stamp())
        registry.save()
        measured = [x for x in registry.value['tasks'].values() if x.get('wall_seconds') is not None]
        costs = {'child_wall_seconds_sum': sum(x['wall_seconds'] for x in measured),
                 'allocated_process_GPU_hours': sum(x['wall_seconds'] * len(x.get('devices', [])) for x in measured) / 3600,
                 'completed_measured_tasks': len(measured),
                 'allocation_hours_are_not_unique_physical_GPU_hours': True,
                 'reused_formal_Test8_and16_cost_is_recorded_in_main_Test_registry_not_double_counted': True}
        atomic_json(HERE / 'COSTS.json', costs)
        atomic_json(HERE / 'COMPLETE.json', {'status': registry.value['status'], 'completed': stamp(),
                    'analysis_directory': str(HERE / 'analysis'), 'execution_registry_sha256': sha256(registry.path),
                    'actual_costs': costs, 'primary_training_and_selection_unchanged': True})
    except BaseException as exc:
        atomic_json(HERE / 'FAILED.json', {'status': 'INCOMPLETE', 'error': repr(exc), 'time': stamp(),
                    'traceback': traceback.format_exc(), 'resume_command': [sys.executable, '-B', str(Path(__file__).resolve())]})
        raise


if __name__ == '__main__':
    main()
