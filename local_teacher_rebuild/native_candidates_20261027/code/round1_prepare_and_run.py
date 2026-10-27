"""Root-owned execution chain: NEW local teacher -> targets -> all eight runs.

This is an actual blocking owner with resumable durable remote children. It may
not mark completion until run_suite has produced REPORT_COMPLETE. The independent
teacher owner owns independent Validation oracle scoring/targets on allowed permanent GPUs.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
import threading
import traceback

from gt_numeric import atomic, read, sha, stamp

ROOT = Path(__file__).resolve().parents[1]
CODE = Path('/mnt/project/recogdrive-no-routing-round1-20261026')
RESULTS = Path('/mnt/project/recogdrive-no-routing-round1-results-20261026')
LEGACY = Path('/mnt/project/recogdrive-teacher2-a800-20261026')
PYTHON = '/mnt/project/recogdrive-distillation-first42-a800-20261022/.venv/bin/python'


def reference(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


class PreparationOwner:
    def __init__(self, args):
        self.args = args
        self.folder = args.results / 'preparation_owner'
        self.folder.mkdir(parents=True, exist_ok=True)
        self.lock = (self.folder / 'OWNER.lock').open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.state_path = self.folder / 'REGISTRY.json'
        self.state_guard = threading.RLock()
        self.state = read(self.state_path) if self.state_path.exists() else {'schema': 'nrkd_preparation_owner_v1', 'history': []}
        ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]
        self.update('WAITING_LOCAL_TEACHER', pid=os.getpid(), hostname=socket.gethostname(),
                    process_start_ticks=ticks, teacher_root=str(args.root.resolve()),
                    implementation_root=str(args.code.resolve()))

    def update(self, status=None, **fields):
        with self.state_guard:
            if status and status != self.state.get('status'):
                self.state['history'].append({'time': stamp(), 'status': status})
                self.state['status'] = status
            self.state.update(fields, updated_at=stamp())
            atomic(self.state_path, self.state)

    def wait_for_file(self, path, label, predicate=None):
        path = Path(path)
        while True:
            if path.exists():
                value = read(path)
                if predicate is None or predicate(value):
                    return value
                raise ValueError(f'{label} exists but does not prove the required complete state: {path}')
            self.update('WAITING_' + label, pending_artifact=str(path))
            time.sleep(self.args.poll_seconds)

    def code_gate(self):
        frozen = self.wait_for_file(self.args.code_freeze, 'IMPLEMENTATION_FREEZE',
                                    lambda x: x.get('status') == 'IMPLEMENTATION_FROZEN')
        actual = {str(p.relative_to(self.args.code)): sha(p)
                  for p in sorted((self.args.code / 'student_distillation').glob('*.py'))}
        if not actual or frozen.get('code_files') != actual:
            raise ValueError('Current implementation differs from the independently accepted code freeze')
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.args.code, text=True).strip()
        if commit != frozen.get('git_commit'):
            raise ValueError('Implementation branch commit changed after acceptance')
        production=frozen.get('production_source_snapshot')
        if production:
            if sha(production['path'])!=production['sha256']:
                raise ValueError('Reviewed production source manifest changed')
            snapshot=read(production['path'])
            for row in snapshot['files']:
                if sha(row['active_source'])!=row['sha256']:
                    raise ValueError('Reviewed active production source changed: '+row['active_source'])
            original=snapshot['external_original_fitter']
            if sha(original['path'])!=original['sha256']:
                raise ValueError('Approved original fitter source changed')
        return frozen

    def inventory_command(self, host, query):
        command = ['nvidia-smi', '--query-gpu=' + query, '--format=csv,noheader,nounits']
        if host == 'training-vla-zt' and socket.gethostname() == 'training-vla-zt-worker-0':
            return subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
        return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=6', host,
                              'ZT_AUTO_PROXY=0 ' + shlex.join(command)],
                              capture_output=True, text=True, check=True, timeout=30)

    def gpu_allocation(self, name, host, uuids):
        """Actually admit one preparation child; this is not all32 training readiness."""
        policy_path = Path('/mnt/project/server_dispatch_policy.json')
        policy = read(policy_path)
        grant = policy['task_authorizations']['recogdrive_no_routing_round1_20261026']
        if grant.get('status') != 'ACTIVE_TASK_BOUNDED_GRANT' or host not in grant.get('temporary_hosts', []):
            raise ValueError('Preparation child no longer has an active host grant')
        probe = self.inventory_command(host, 'index,uuid,memory.free')
        inventory = {x[1].strip(): {'index': int(x[0]), 'free_MB': int(x[2])}
                     for line in probe.stdout.splitlines() if len(x := line.split(',')) == 3}
        for uuid in uuids:
            gpu = inventory[uuid]
            if (gpu['index'] not in grant['authorized_gpu_indices'][host] or
                gpu['index'] in grant.get('excluded_gpu_indices', {}).get(host, []) or gpu['free_MB'] < 12000):
                raise ValueError('Preparation UUID/live grant/headroom check failed; preserve other jobs')
        evidence = self.folder / 'admissions' / name / (str(time.time_ns()) + '.json')
        atomic(evidence, {'scope': 'ONE_PREPARATION_CHILD_NOT_EIGHT_FORMAL_TRAINING_RUNS',
               'host': host, 'actual_devices': {u: inventory[u] for u in uuids},
               'observed_at': stamp(), 'live_policy': reference(policy_path)})
        allocation = {'host': host, 'python': self.args.python, 'workdir': str(self.args.code),
                'gpu_uuids': uuids, '_task': 'recogdrive_no_routing_round1_20261026',
                '_live_policy_path': str(policy_path), '_resource_status': 'READY_FOR_FORMAL_DISPATCH',
                'preparation_only_headroom_receipt': reference(evidence)}
        if host == 'training-vla-zt':
            allocation['ssh_target'] = 'local'
        return allocation

    def gpu_output_complete(self, name, path):
        if not Path(path).is_file():
            return False
        if name.startswith('targets_') or name in ('train_calibration','formal_ddp','formal_single'):
            from preparation_identity import validate_current_output
            try:
                current=getattr(self,'current_paths',None) or read(self.paths_file)
                return validate_current_output(name,path,paths=current,results=self.args.results,
                    code=self.args.code,world_size=getattr(self,'training_world_size',None))
            except (ValueError,KeyError,TypeError,OSError,json.JSONDecodeError):
                return False
        value = read(path)
        return value.get('status') == 'ACTUAL_CPU_CHILD_COMPLETED' and name == 'PRIVATE_CPU_RECOVERY'

    def retire_incompatible_output(self,name,path,ownership_error):
        """Preserve incompatible artifacts only after proving no child is live."""
        path=Path(path)
        if not path.exists() or self.gpu_output_complete(name,path):return
        family=self.folder/'remote_execution/jobs'/name;family.mkdir(parents=True,exist_ok=True)
        with (family/'FAMILY_LEASE.lock').open('a+') as lease:
            try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return  # Controller reattaches the live owner.
            pointer=family/'ACTIVE_ATTEMPT.json'
            if pointer.exists():
                spec=Path(read(pointer)['spec']);state_path=spec.parent/'STATUS.json'
                state=read(state_path) if state_path.exists() else {}
                terminal=(state.get('status')=='EXITED' and isinstance(state.get('returncode'),int) and state['returncode']>=0)
                no_child=(state.get('status') in ('STARTUP_FAILED','SUPERSEDED_BEFORE_START') and state.get('child_started') is False)
                if not (terminal or no_child):
                    raise ownership_error('Incompatible preparation output has a live/unknown recorded child; preserve it')
            source=path.parent if name.startswith('targets_') or name in ('formal_ddp','formal_single') else path
            destination=self.folder/'incompatible_outputs'/name/str(time.time_ns())
            destination.parent.mkdir(parents=True,exist_ok=True)
            source.rename(destination)
            atomic(destination.parent/(destination.name+'.json'),{'status':'PRESERVED_INCOMPATIBLE_OUTPUT',
                   'name':name,'original_path':str(source),'preserved_path':str(destination),'time':stamp(),
                   'no_live_or_unknown_child_moved':True,'reason':'Current teacher/assets/code completion identity differs'})

    def gpu_job(self, name, host, uuids, arguments, complete_path):
        """Use the same lease/versioned-attempt implementation as the full suite."""
        self.code_gate()
        sys.path.insert(0, str(self.args.code))
        from student_distillation.run_suite import Controller, RemoteOwnershipUnknown
        allocation = self.gpu_allocation(name, host, uuids)
        owner = self

        class PreparationRegistry:
            @property
            def value(self):
                return owner.state

            def task_update(self, key, status, **fields):
                with owner.state_guard:
                    owner.state.setdefault('tasks', {})[key] = dict(fields, status=status, updated_at=stamp())
                    owner.update()

            def run_update(self, *args, **kwargs):
                raise RuntimeError('Preparation child may not mutate formal training run state')

        execution_root = self.folder / 'remote_execution'; execution_root.mkdir(exist_ok=True)
        controller = Controller(execution_root, PreparationRegistry(), {name: allocation})
        command = [self.args.python, '-B', *map(str, arguments)]
        for attempt in range(self.args.engineering_retries + 1):
            try:
                self.retire_incompatible_output(name,complete_path,RemoteOwnershipUnknown)
                controller.execute(name, command, allocation_run=name,
                                   success_check=lambda: self.gpu_output_complete(name, complete_path))
                atomic(self.folder / f'{name}.COMPLETE.json', {'status': 'COMPLETE',
                       'output': reference(complete_path), 'durable_job_family': str(execution_root / 'jobs' / name)})
                return
            except RemoteOwnershipUnknown:
                # Never use a retry to duplicate surviving ranks or unknown work.
                raise
            except RuntimeError:
                if attempt == self.args.engineering_retries:
                    raise
                self.update('RECOVERING_PREPARATION_CHILD', recovering_child=name, local_retry=attempt + 1)

    def paths(self):
        values = read(LEGACY / 'paths.local.json')
        values.update(frozen_repo=str(self.args.code), paths_json=str(LEGACY / 'paths.local.json'),
                      teacher_bundle=str(self.args.root / 'frozen_teacher/BUNDLE.json'),
                      feature_manifest=str(self.args.results / 'features/MANIFEST.ALL_POPULATIONS.json'),
                      targets_manifest=str(self.args.results / 'targets/train/MANIFEST.json'),
                      validation_targets_manifest=str(self.args.results / 'targets/validation/MANIFEST.json'),
                      test_targets_manifest=str(self.args.results / 'targets/test/MANIFEST.json'),
                      calibration=str(self.args.results / 'calibration/CALIBRATION.json'),
                      qualification_receipt=str(self.args.results / 'qualification/formal_single/QUALIFICATION.json'),
                      evaluation_noise_manifest=str(self.args.results / 'reference_noise/EVALUATION_NOISE_MANIFEST.json'),
                      inference_graph_receipt=str(self.args.results / 'qualification/NATIVE_INFERENCE_GRAPH_ADMISSION.json'),
                      bridge_factory='student_distillation.train:bridge_factory', results_root=str(self.args.results))
        folder = self.args.results / 'protocol'; folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'PATHS.json'
        if path.exists() and read(path) != values:
            raise ValueError('Actual registered paths changed')
        atomic(path, values)
        config = folder / 'PATHS_CONFIG.json'
        atomic(config, {'paths': values})
        return values, path, config

    def targets(self, population, uuid):
        manifest=self.args.results/'targets'/population/'MANIFEST.json'
        if manifest.exists() and read(manifest)['populations'][population]['identity'].get('execution_device_type')=='cpu_verified_native_cache':
            from cached_target_publication import validate_publication
            validate_publication(manifest,self.args.results)
            if not self.gpu_output_complete('targets_'+population,manifest):
                raise ValueError('Published cached actual target identity failed')
            atomic(self.folder/f'targets_{population}.COMPLETE.json',{'status':'COMPLETE',
                   'output':reference(manifest),'construction':'cpu_verified_native_cache',
                   'publication':reference(manifest.parent/'ADOPTION.json')})
            self.update('TARGETS_ACCEPTED',accepted_population=population)
            return
        arguments = ['-m', 'student_distillation.build_targets', '--paths', self.paths_file,
                     '--population', population, '--target-versions', 'full', 'gain1', '--device', 'cuda:0',
                     '--out', self.args.results / 'targets']
        if population != 'train':
            arguments.append('--diagnostic-only')
        self.gpu_job('targets_' + population, self.preparation_host, [uuid], arguments,
                     self.args.results / 'targets' / population / 'MANIFEST.json')

    def ready_resources(self):
        planned = read(self.args.resources)
        policy_path = Path('/mnt/project/server_dispatch_policy.json')
        live = read(policy_path)['task_authorizations']['recogdrive_no_routing_round1_20261026']
        if live['status'] != 'ACTIVE_TASK_BOUNDED_GRANT':
            raise ValueError('The explicit task authorization is no longer active')
        inventory = {}
        for allocation in planned['allocations'].values():
            host = allocation['host']
            if host not in live['temporary_hosts']:
                raise ValueError('A planned host is no longer authorized')
            if host not in inventory:
                result = self.inventory_command(host, 'index,uuid,memory.free')
                inventory[host] = {row[1].strip(): {'index': int(row[0]), 'free_MB': int(row[2])}
                                   for line in result.stdout.splitlines() if (row := line.split(',')) and len(row) == 3}
            for uuid in allocation['gpu_uuids']:
                gpu = inventory[host][uuid]
                if (gpu['index'] not in live['authorized_gpu_indices'][host] or
                    gpu['index'] in live.get('excluded_gpu_indices', {}).get(host, []) or gpu['free_MB'] < 12000):
                    raise ValueError('Formal GPU authorization/headroom gate failed; preserve unrelated jobs')
        planned['status'] = 'READY_FOR_FORMAL_DISPATCH'
        planned['actual_headroom'] = inventory
        planned['headroom_observed_at'] = stamp()
        planned['live_policy_observation'] = reference(policy_path)
        target = self.args.results / 'resources/ROUND1_PERMANENT_READY.json'
        if target.exists():
            # Keep the scientific allocation and its original observation immutable.
            old = read(target)
            if old['allocations'] != planned['allocations']:
                raise ValueError('Formal resource allocation changed during recovery')
        else:
            atomic(target, planned)
        return target

    @staticmethod
    def read_launcher_status(path):
        # The external stable launcher writes status with truncate+write rather
        # than atomic replace. A transient empty/partial file is not an exit.
        try:
            return read(path)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def execute_full_suite(self, command):
        """Reattach live attempts; retain failures and create fresh resume attempts."""
        base = self.folder / 'full_suite_dispatch'; base.mkdir(parents=True, exist_ok=True)
        intent = base / 'COMMAND.json'
        if intent.exists() and read(intent) != command:
            raise ValueError('Registered full-suite scientific command changed')
        if not intent.exists():
            atomic(intent, command)
        prior = sorted(base.glob('attempt_[0-9][0-9][0-9][0-9]'))
        def terminal(path):
            if not path.exists():
                return False
            value = self.read_launcher_status(path)
            return value.get('returncode') is not None or value.get('child_exit') == 'NOT_LAUNCHED'
        for directory in prior[:-1]:
            state = directory / 'status/full_round1_suite.json'
            if not terminal(state):
                raise RuntimeError('An earlier suite attempt has no proven terminal state; inspect its lease before recovery')
        pending = prior[-1] if prior else None
        if pending is not None:
            state = pending / 'status/full_round1_suite.json'
            if state.exists() and self.read_launcher_status(state).get('returncode') == 0:
                return state
            if terminal(state):
                pending = None
        number = int(prior[-1].name.split('_')[-1]) + 1 if prior else 0
        failures_this_invocation = 0
        while failures_this_invocation <= self.args.engineering_retries:
            self.code_gate()
            launch = pending if pending is not None else base / f'attempt_{number:04d}'
            if pending is None:
                number += 1
            pending = None
            launch.mkdir(parents=True, exist_ok=True)
            process = None
            wrapper_exit = launch / 'WRAPPER_EXIT.json'
            if not (launch / 'launcher.pid').exists():
                complete_command = ['env', 'ZT_AUTO_PROXY=0', 'OPENBLAS_CORETYPE=Sandybridge',
                                    'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1',
                                    'PYTHONPATH=' + str(self.args.code), 'CUBLAS_WORKSPACE_CONFIG=:4096:8', *command]
                (launch / 'jobs.tsv').write_text('full_round1_suite\t-\t' + shlex.join(complete_command) + '\n')
                with (launch / 'launcher.log').open('ab', buffering=0) as stream:
                    supervisor = ('import json,os,subprocess,sys\n'
                                  'from pathlib import Path\n'
                                  'rc=subprocess.call(sys.argv[2:])\n'
                                  'p=Path(sys.argv[1]); t=p.with_suffix(".tmp")\n'
                                  't.write_text(json.dumps({"actual_launcher_returncode":rc,"pid":os.getpid()}))\n'
                                  'os.replace(t,p)\n'
                                  'sys.exit(rc)\n')
                    process = subprocess.Popen([self.args.python, '-B', '-c', supervisor, str(wrapper_exit),
                        self.args.python, '/mnt/project/skill/stable-gpu-job-launch/scripts/stable_gpu_launcher.py',
                        '--jobs-tsv', str(launch / 'jobs.tsv'), '--out-root', str(launch), '--poll-seconds', '5'],
                        cwd=self.args.code, stdin=subprocess.DEVNULL, stdout=stream,
                        stderr=subprocess.STDOUT, start_new_session=True)
                (launch / 'launcher.pid').write_text(str(process.pid) + '\n')
                stat = Path(f'/proc/{process.pid}/stat').read_text().split(') ', 1)[1].split()
                atomic(launch / 'SUPERVISOR.json', {'pid': process.pid, 'start_ticks': stat[19]})
            status = launch / 'status/full_round1_suite.json'
            while not status.exists() or self.read_launcher_status(status).get('returncode') is None:
                if wrapper_exit.exists():
                    # A wrapper failure is not a proven child exit. Retry only
                    # when the stable launcher never registered any child.
                    if not status.exists() and not list((launch / 'pids').glob('*.pid')):
                        atomic(status, {'state': 'startup_failed', 'returncode': None,
                               'child_exit': 'NOT_LAUNCHED', 'wrapper_exit': reference(wrapper_exit)})
                        break
                    raise RuntimeError('Suite wrapper ended without a proven child exit; preserve UNKNOWN and inspect the recorded child lease before recovery')
                if process is not None and process.poll() is not None:
                    raise RuntimeError('Suite supervisor ended without its exit receipt; child exit is UNKNOWN')
                identity = launch / 'SUPERVISOR.json'
                if identity.exists():
                    saved = read(identity)
                    proc = Path(f'/proc/{saved["pid"]}/stat')
                    fields = proc.read_text().split(') ', 1)[1].split() if proc.exists() else []
                    if not fields or fields[0] == 'Z' or fields[19] != saved['start_ticks']:
                        raise RuntimeError('Recorded suite supervisor is gone without a terminal receipt; child exit is UNKNOWN')
                self.update('EXECUTING_ALL_EIGHT_FULL_RUNS', actual_suite_state=str(status))
                time.sleep(self.args.poll_seconds)
            if self.read_launcher_status(status).get('returncode') == 0:
                return status
            failures_this_invocation += 1
            self.update('RESUMING_AFTER_SUITE_ENGINEERING_FAILURE', failed_attempt=reference(status),
                        failures_this_invocation=failures_this_invocation)
        raise RuntimeError('Suite recovery budget exhausted in this owner invocation; inspect failures and restart this owner to resume in a fresh attempt')

    def run(self):
        teacher = self.wait_for_file(self.args.root / 'teacher_suite/FULL_TRAIN_TEST_TEACHER_READY.json',
                                    'LOCAL_TRAIN_TEST_TEACHER', lambda x: x.get('status') == 'COMPLETE')
        for reference_ in (teacher['teacher_bundle'], teacher['teacher_report']):
            if sha(reference_['path']) != reference_['sha256']:
                raise ValueError('Locally generated teacher readiness artifact changed')
        self.code_gate()
        values, self.paths_file, config_file = self.paths()
        self.current_paths=values
        planned = read(self.args.resources)
        allocation = planned['allocations']['H_mu_teacher_seed20261027']
        self.training_world_size = int(planned.get('world_size', len(allocation['gpu_uuids'])))
        if self.training_world_size not in (2, 4):
            raise ValueError('Permanent resource world size must be two or four')
        preparation_plan=read(self.args.preparation_resources) if self.args.preparation_resources else None
        preparation=preparation_plan['allocation'] if preparation_plan else allocation
        self.preparation_host = preparation['host']
        preparation_uuids=preparation['gpu_uuids']
        if len(preparation_uuids)!=self.training_world_size or len(set(preparation_uuids))!=self.training_world_size:
            raise ValueError('Preparation resource plan must contain the chosen world size distinct UUIDs')
        uuids = preparation_uuids[:2]
        self.update('PREPARING_TRAIN_TEST_TARGETS',preparation_host=self.preparation_host,
                    preparation_gpu_uuids=preparation_uuids)
        with ThreadPoolExecutor(max_workers=2) as workers:
            futures = [workers.submit(self.targets, population, uuid)
                       for population, uuid in zip(('train', 'test'), uuids)]
            for future in futures:
                future.result()
        self.update('CALIBRATING_TRAIN_ONLY')
        self.gpu_job('train_calibration', self.preparation_host, [uuids[0]],
                     ['-m', 'student_distillation.calibration', '--config', config_file, '--train-only',
                      '--device', 'cuda:0', '--out', self.args.results / 'calibration'], values['calibration'])
        common = ['--paths', self.paths_file, '--feature-manifest', values['feature_manifest'],
                  '--input-statistics', self.args.results / 'calibration/BASE_INPUT_STATISTICS.json',
                  '--reference-noise-manifest', values['evaluation_noise_manifest'],
                  '--native-reference-cache-index', LEGACY / 'cache/train/INDEX.jsonl',
                  '--teacher-bundle', values['teacher_bundle'], '--targets-manifest', values['targets_manifest'],
                  '--calibration', values['calibration']]
        ddp_out = self.args.results / 'qualification/formal_ddp'
        self.update('QUALIFYING_NEW_TEACHER_WORLD_SIZE',training_world_size=self.training_world_size)
        self.gpu_job('formal_ddp', self.preparation_host, preparation_uuids,
                     ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=' + str(self.training_world_size), '--module',
                      'student_distillation.qualification', '--ddp-only', '--training-world-size', str(self.training_world_size), *common, '--out', ddp_out], ddp_out / 'DDP_TAIL_53.json')
        self.gpu_job('formal_single', self.preparation_host, [uuids[1]],
                     ['-m', 'student_distillation.qualification', *common, '--device', 'cuda:0', '--training-world-size', str(self.training_world_size),
                      '--ddp-receipt', ddp_out / 'DDP_TAIL_53.json', '--control-scenes', '64',
                      '--control-updates', '256', '--out', self.args.results / 'qualification/formal_single'],
                     values['qualification_receipt'])
        qualification = read(values['qualification_receipt'])
        if qualification.get('status') != 'CODE_CHECKS_PASS' or qualification.get('formal_asset_qualification') is not True:
            raise ValueError('Actual formal qualification did not complete all mandatory checks')
        resources = self.ready_resources()
        command = [self.args.python, '-B', '-m', 'student_distillation.run_suite', '--config-dir',
                   str(self.args.code / 'configs/student_distillation'), '--paths-config', str(config_file),
                   '--resources', str(resources), '--world-size', str(self.training_world_size), '--engineering-retries', '3',
                   '--complete-through', 'report', '--out', str(self.args.results)]
        self.code_gate()
        self.update('EXECUTING_ALL_EIGHT_FULL_RUNS', command=command)
        self.execute_full_suite(command)
        registry = read(self.args.results / 'EXPERIMENT_REGISTRY.json')
        if registry.get('status') != 'REPORT_COMPLETE' or not (self.args.results / 'ROUND1_REPORT.md').is_file():
            raise RuntimeError('Suite returned without all final results/report')
        # Actor walls overlap on shared GPUs. Report their interval union as a
        # distinct physical allocation cost before freezing the final report SHA.
        subprocess.run([self.args.python, '-B', str(self.args.root / 'engineering/permanent_fullflow/finalize_round1_physical_costs.py'),
                        '--root', str(self.args.root), '--prior-root', '/mnt/project/VLA-AD/experiments/recogdrive_dc8_refit_20261026', '--results', str(self.args.results), '--round1-report', str(self.args.results / 'ROUND1_REPORT.md')],
                       cwd=self.args.code, check=True)
        self.update('REPORT_COMPLETE', final_report=reference(self.args.results / 'ROUND1_REPORT.md'))
        atomic(self.folder / 'COMPLETE.json', self.state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--code', type=Path, default=CODE)
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--python', default=PYTHON)
    parser.add_argument('--resources', type=Path, default=RESULTS / 'resources/ROUND1_PERMANENT_32GPU_ALLOCATION.json')
    parser.add_argument('--preparation-resources',type=Path,help='Preparation-only host/UUID plan; formal student allocation stays unchanged')
    parser.add_argument('--code-freeze', type=Path, default=RESULTS / 'protocol/CODE_FREEZE.json')
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--engineering-retries', type=int, default=2)
    args = parser.parse_args()
    if not 0 < args.poll_seconds <= 60:
        parser.error('poll-seconds must be in (0,60]')
    try:
        PreparationOwner(args).run()
    except BaseException:
        atomic(args.results / 'preparation_owner/INCOMPLETE.json', {'status': 'INCOMPLETE', 'time': stamp(),
               'traceback': traceback.format_exc(), 'resume_argv': [sys.executable, __file__, *sys.argv[1:]]})
        raise


if __name__ == '__main__':
    main()
