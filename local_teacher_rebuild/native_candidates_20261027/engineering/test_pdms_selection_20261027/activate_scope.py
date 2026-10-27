"""Replace only the CPU suite owner; preserve all live training processes."""
import fcntl
import ctypes
import json
import os
from pathlib import Path
import select
import signal
import sys
from test_pdms_runtime import HERE, ROOT, OLD_SCOPE, RESULTS, verify_scope
from student_distillation.checkpointing import atomic_json, sha256
from student_distillation.experiment_registry import stamp


def main():
    verify_scope()
    with (HERE / 'ACTIVATE.lock').open('a+') as activation_lock:
        fcntl.flock(activation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (HERE / 'LAUNCH.json').exists():
            raise ValueError('Test selection owners already launched; inspect their leases')
        original = json.loads((OLD_SCOPE / 'ACTIVE_OWNER_IDENTITY.json').read_text())['processes']
        if len(original) != 1:
            raise ValueError('Expected the single known CPU owner')
        owner = original[0]
        pid = owner['pid']
        actual = Path(f'/proc/{pid}')
        ticks = actual.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19]
        argv = [x.decode() for x in actual.joinpath('cmdline').read_bytes().split(b'\0') if x]
        if ticks != str(owner['start_ticks']) or argv != owner['argv']:
            raise ValueError('Known old CPU owner identity changed')
        retained = json.loads((OLD_SCOPE / 'OWNER_SPEC.json').read_text())['original_retained_training_specs']
        proofs = {}
        for run, entry in retained.items():
            spec = Path(entry['spec']['path'])
            status = json.loads(spec.with_name('STATUS.json').read_text())
            if (status['status'] != 'RUNNING' or status['child_pid'] != entry['child_pid'] or
                    status['child_start_ticks'] != entry['child_start_ticks'] or
                    sha256(spec) != entry['spec']['sha256']):
                raise ValueError('A retained live training identity changed: ' + run)
            proofs[run] = dict(child_pid=status['child_pid'], child_start_ticks=status['child_start_ticks'],
                               spec_sha256=sha256(spec), host=json.loads(spec.read_text())['host'])
        libc = ctypes.CDLL(None, use_errno=True)
        fd = libc.syscall(434, pid, 0)
        if fd < 0:
            raise OSError(ctypes.get_errno(), 'pidfd_open')
        def send_signal(value):
            if libc.syscall(424, fd, value, 0, 0) < 0:
                raise OSError(ctypes.get_errno(), 'pidfd_send_signal')
        try:
            send_signal(signal.SIGSTOP)
            atomic_json(HERE / 'PRIOR_CPU_OWNER_HELD.json', {'time': stamp(), 'owner': owner,
                        'retained_training_identities': proofs, 'all_GPU_children_untouched': True})
            registry_path = RESULTS / 'EXPERIMENT_REGISTRY.json'
            prior = registry_path.read_bytes()
            (HERE / 'PRIOR_TRAINING_REGISTRY.json').write_bytes(prior)
            registry = json.loads(prior)
            registry['latest_selection_override'] = {'user_instruction': {'path': str(HERE / 'REQUEST.json'),
                'sha256': sha256(HERE / 'REQUEST.json')}, 'selection_population': 'test',
                'test_used_for_model_selection': True, 'independent_held_out_test': False,
                'validation_used_for_model_selection': False,
                'test_execution_registry': str(HERE / 'TEST_EXECUTION_REGISTRY.json'),
                'frozen_training_checkpoint_identities_preserved': True}
            atomic_json(registry_path, registry)
            send_signal(signal.SIGKILL)
            if not select.select([fd], [], [], 15)[0]:
                raise RuntimeError('Old CPU owner has not actually exited')
        finally:
            os.close(fd)
        for path in (RESULTS / 'SUITE_CONTROLLER.lock', RESULTS / 'preparation_owner/OWNER.lock'):
            with path.open('a+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(HERE / 'PRIOR_CPU_OWNER_EXITED.json', {'time': stamp(), 'owner': owner,
                    'actual_exit_observed_by_pidfd': True, 'signal': 'SIGKILL_CPU_OWNER_ONLY',
                    'CPU_owner_leases_free': True, 'retained_training_identities': proofs,
                    'GPU_children_restarted_or_signalled': []})
        sys.path.insert(0, str(ROOT / 'engineering/permanent_fullflow'))
        from permanent_dispatch import install, grant, PYTHON
        grant()
        install()
        import dispatch
        directory = ROOT / 'dispatch/test_pdms_selection_owners_20261027'
        jobs = [('full_test_endpoint_owner', '-', [PYTHON, '-B', str(HERE / 'test_endpoint_owner.py')]),
                ('full_suite_test_pdms_owner', '-', [PYTHON, '-B', str(HERE / 'test_suite_owner.py')])]
        launch = dispatch.launch('training-vla-zt', directory, jobs)
        atomic_json(HERE / 'LAUNCH.json', {'status': 'TEST_PDMS_OWNERS_LAUNCHED', 'time': stamp(),
                    'launch': launch, 'dispatch_directory': str(directory), 'jobs': jobs,
                    'source_freeze_sha256': sha256(HERE / 'SCOPE_FREEZE.json'),
                    'retained_training_identities': proofs,
                    'source_HEAD_config_optimizer_rng_and_GPU_children_unchanged': True,
                    'complete_status_requires_actual_Test_rows_and_final_full_report': True})
        print(json.dumps(launch), flush=True)


if __name__ == '__main__':
    main()
