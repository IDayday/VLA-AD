"""Private CPU integration check of owner admission/recovery, never GPU evidence."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from gt_numeric import atomic, sha
from round1_prepare_and_run import PreparationOwner


def main():
    parent = ROOT / 'selftests/preparation_owner'
    folder = parent / ('run_' + str(time.time_ns()))
    code = folder / 'code'; code.mkdir(parents=True, exist_ok=True)
    modules = code / 'student_distillation'; modules.mkdir(exist_ok=True)
    source = modules / 'fixture.py'; source.write_text('VALUE = 1\n')
    if not (code / '.git').exists():
        subprocess.run(['git', 'init', '-q', str(code)], check=True)
        subprocess.run(['git', 'add', '.'], cwd=code, check=True)
        subprocess.run(['git', '-c', 'user.name=Owner engineering check', '-c',
                        'user.email=owner-test@localhost', 'commit', '-qm', 'Private CPU fixture'], cwd=code, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code, text=True).strip()
    freeze = folder / 'CODE_FREEZE.json'
    atomic(freeze, {'status': 'IMPLEMENTATION_FROZEN', 'code_files': {'student_distillation/fixture.py': sha(source)},
                    'git_commit': commit, 'engineering_fixture_only': True})
    args = SimpleNamespace(results=folder / 'results', root=ROOT, code=code, python=sys.executable,
                           code_freeze=freeze, poll_seconds=.01, engineering_retries=1)
    owner = PreparationOwner(args)
    owner.code_gate()
    bound=folder/'BOUND_PRODUCTION.py';bound.write_text('BOUND = 1\n')
    original_fitter=folder/'ORIGINAL_FITTER.py';original_fitter.write_text('ORIGINAL = 1\n')
    production=folder/'PRODUCTION_SOURCE.json'
    atomic(production,{'files':[{'active_source':str(bound),'sha256':sha(bound)}],
        'external_original_fitter':{'path':str(original_fitter),'sha256':sha(original_fitter)}})
    frozen=json.loads(freeze.read_text());frozen['production_source_snapshot']={'path':str(production),'sha256':sha(production)}
    atomic(freeze,frozen);owner.code_gate()
    bound.write_text('BOUND = 2\n')
    try:owner.code_gate()
    except ValueError:pass
    else:raise AssertionError('Changed external production source passed its reviewed manifest')
    bound.write_text('BOUND = 1\n')
    original_fitter.write_text('ORIGINAL = 2\n')
    try:owner.code_gate()
    except ValueError:pass
    else:raise AssertionError('Changed original fitter source passed its approved SHA')
    original_fitter.write_text('ORIGINAL = 1\n');owner.code_gate()
    source.write_text('VALUE = 2\n')
    rejected = False
    try:
        owner.code_gate()
    except ValueError:
        rejected = True
    if not rejected:
        raise AssertionError('Changed source passed the implementation freeze')
    source.write_text('VALUE = 1\n')
    bad = folder / 'BAD_READY.json'; atomic(bad, {'status': 'PENDING'})
    try:
        owner.wait_for_file(bad, 'PRIVATE_CPU_CHECK', lambda x: x['status'] == 'COMPLETE')
    except ValueError:
        pass
    else:
        raise AssertionError('Incomplete readiness marker passed')
    # Simulated dispatch transport, but real CPU child exits/artifacts. The
    # first attempt fails; the second executes the exact same original command.
    output = folder / 'ACTUAL_CPU_CHILD.json'
    module_key = 'student_distillation.run_suite'
    original_module = sys.modules.get(module_key)
    fake = ModuleType(module_key)
    calls = []
    allocation_method = owner.gpu_allocation
    owner.gpu_allocation = lambda *args: {'engineering_fixture_only': True}

    class PrivateOwnershipUnknown(RuntimeError):
        pass

    class PrivateCPUController:
        def __init__(self, out, registry, allocations):
            self.registry = registry

        def execute(self, key, command, allocation_run=None, success_check=None):
            if success_check():
                return 0
            calls.append(list(command))
            rc = 7 if len(calls) == 1 else subprocess.run(command, cwd=code).returncode
            self.registry.task_update(key, 'FAILED' if rc else 'COMPLETE', actual_exit=rc)
            if rc:
                raise RuntimeError('Actual fixture exit7')
            if not success_check():
                raise RuntimeError('Actual child lacks required output')
            return 0

    fake.Controller = PrivateCPUController
    fake.RemoteOwnershipUnknown = PrivateOwnershipUnknown
    sys.modules[module_key] = fake
    command = ['-c', 'from pathlib import Path; Path(' + repr(str(output)) + ').write_text(' +
               repr(json.dumps({'status': 'ACTUAL_CPU_CHILD_COMPLETED'})) + ')']
    try:
        owner.gpu_job('PRIVATE_CPU_RECOVERY', 'NO_REAL_HOST', ['NO_REAL_GPU'], command, output)
        if len(calls) != 2 or json.loads(output.read_text())['status'] != 'ACTUAL_CPU_CHILD_COMPLETED':
            raise AssertionError('Bounded recovery did not require actual successful output')
        owner.gpu_job('PRIVATE_CPU_RECOVERY', 'NO_REAL_HOST', ['NO_REAL_GPU'], command, output)
        if len(calls) != 2:
            raise AssertionError('Restart duplicated an already registered completed job')
    finally:
        owner.gpu_allocation = allocation_method
        if original_module is None:
            sys.modules.pop(module_key, None)
        else:
            sys.modules[module_key] = original_module
    # Exercise the real detached stable launcher as a CPU-only suite transport.
    # The child fails once, then resumes successfully into a new attempt.
    first_flag = folder / 'SUITE_FIRST_FAILED.txt'
    suite_output = folder / 'SUITE_ACTUAL_CPU_OUTPUT.txt'
    program = ('from pathlib import Path\nimport sys\np=Path(' + repr(str(first_flag)) + ')\n'
               'if not p.exists():\n p.write_text("failure retained")\n sys.exit(9)\n'
               'Path(' + repr(str(suite_output)) + ').write_text("actual resumed child completed")\n')
    child_script = folder / 'ACTUAL_SUITE_CHILD.py'
    child_script.write_text(program)
    suite_command = [sys.executable, str(child_script)]
    status = owner.execute_full_suite(suite_command)
    attempts = sorted((owner.folder / 'full_suite_dispatch').glob('attempt_*'))
    if len(attempts) != 2 or not suite_output.exists() or json.loads(status.read_text())['returncode'] != 0:
        raise AssertionError('Suite terminal failure did not create a successful actual fresh attempt')
    if owner.execute_full_suite(suite_command) != status or len(list((owner.folder / 'full_suite_dispatch').glob('attempt_*'))) != 2:
        raise AssertionError('A recovered successful suite was duplicated')
    atomic(folder / 'COMPLETE.json', {'status': 'PASS_PRIVATE_CPU_PREPARATION_OWNER',
           'source_change_rejected': True, 'incomplete_ready_rejected': True,
           'active_production_and_original_fitter_source_changes_rejected': True,
           'actual_failed_then_successful_CPU_child': True, 'completed_job_not_duplicated': True,
           'real_GPU_or_SSH_calls': 0, 'formal_qualification': False,
           'actual_stable_launcher_suite_failure_then_fresh_resume': True})
    atomic(parent / 'LATEST.json', {'path': str(folder / 'COMPLETE.json'), 'sha256': sha(folder / 'COMPLETE.json')})
    print((folder / 'COMPLETE.json').read_text())


if __name__ == '__main__':
    main()
