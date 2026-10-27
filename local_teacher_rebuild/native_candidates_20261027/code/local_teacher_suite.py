"""Durable actual-execution owner for the single local D_C8 teacher pipeline.

Root launches this owner. It waits for all GT actors to exit, then exclusively
reuses the registered UUID allocation for pilot, calibration and full scoring.
Every phase waits for real receipts and actor exit; submission is never success.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback
from gt_numeric import atomic, digest, read, sha, stamp
from fit_library import ref, collect_request, emit_validation_diagnostics, emit_teacher_report

ROOT = Path(__file__).resolve().parents[1]
ROUND1 = Path('/mnt/project/recogdrive-no-routing-round1-20261026')
RESULTS = Path('/mnt/project/recogdrive-no-routing-round1-results-20261026')
OLD = Path('/mnt/project/recogdrive-first42-a800-20261022')
LEGACY = Path('/mnt/project/recogdrive-teacher2-a800-20261026')
PYTHON = '/mnt/project/recogdrive-distillation-first42-a800-20261022/.venv/bin/python'


def actor_summary(jobs):
    result = {'total': len(jobs), 'done': 0, 'failed': 0, 'pending_or_running': 0, 'entries': []}
    for job in jobs:
        path = Path(job['status'])
        if not path.exists():
            result['pending_or_running'] += 1
            continue
        state = None
        for attempt in range(5):
            try:
                state = read(path)
                break
            except (json.JSONDecodeError, OSError):
                time.sleep(.1)
        if state is None:
            result['pending_or_running'] += 1
            continue
        terminal = state.get('state') in ('done', 'failed') and state.get('returncode') is not None
        if terminal:
            result['done' if state['returncode'] == 0 else 'failed'] += 1
        else:
            result['pending_or_running'] += 1
        result['entries'].append({'status': str(path), **state})
    result['all_exited'] = result['pending_or_running'] == 0 and result['done'] + result['failed'] == len(jobs)
    result['all_exit_zero'] = result['all_exited'] and result['failed'] == 0
    return result


def queue_progress(folder):
    folder = Path(folder)
    queue = read(folder / 'QUEUE.json')
    completed = sum((folder / 'receipts' / (identity + '.json')).exists() for identity in queue['tasks'])
    return {'tasks': len(queue['tasks']), 'completed': completed, 'pending': len(queue['tasks']) - completed,
            'failed_attempts': len(list((folder / 'failures').glob('*/*.json')))}


def required_GT_jobs(path, fallback):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError('The complete GT actor exit requirement registry is required')
    value = read(path)
    records = value if isinstance(value, list) else value.get('job_manifests', value.get('manifests', value.get('requirements')))
    if records is None:
        raise ValueError('GT_EXIT_REQUIREMENTS must contain job manifest references')
    jobs = {}
    for reference in records:
        filename = Path(reference['path'] if isinstance(reference, dict) else reference)
        if isinstance(reference, dict) and reference.get('sha256') and sha(filename) != reference['sha256']:
            raise ValueError('An immutable GT actor job manifest changed')
        for job in read(filename):
            jobs[job['status']] = job
    return list(jobs.values()), ref(path)


def process_start_ticks(pid):
    try:
        return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
    except FileNotFoundError:
        return None


def execute_cpu_task(filename):
    """Detached wrapper persists the real child exit even if the owner restarts."""
    task = read(filename)
    lock = Path(task['lock']).open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start = time.monotonic()
    state = {'status': 'RUNNING', 'host': socket.gethostname(), 'pid': os.getpid(),
             'start_ticks': process_start_ticks(os.getpid()), 'command': task['command'], 'started': stamp()}
    atomic(task['state'], state)
    process = subprocess.Popen(task['command'], cwd=task['cwd'])
    state.update(child_pid=process.pid, child_start_ticks=process_start_ticks(process.pid))
    atomic(task['state'], state)
    returncode = process.wait()
    if returncode == 0 and task.get('complete_path') and not Path(task['complete_path']).exists():
        returncode = 1
    state.update(status='COMPLETE' if returncode == 0 else 'FAILED', returncode=returncode,
                 ended=stamp(), wall_seconds=time.monotonic() - start)
    if returncode == 0 and task.get('complete_path'):
        state['artifact'] = ref(task['complete_path'])
    atomic(task['state'], state)
    if returncode:
        raise SystemExit(returncode)


class Owner:
    def __init__(self, args):
        self.args = args
        self.folder = args.root / 'teacher_suite'
        self.folder.mkdir(parents=True, exist_ok=True)
        self.lock = (self.folder / 'OWNER.lock').open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.registry_path = self.folder / 'REGISTRY.json'
        self.registry = read(self.registry_path) if self.registry_path.exists() else {'schema': 'nrkd_local_teacher_suite_v1', 'stages': {}, 'history': []}
        self.allocation = read(args.allocation)
        if len(self.allocation) < 32 or len({job['uuid'] for job in self.allocation}) != len(self.allocation):
            raise ValueError('Use a verified allocation with at least 32 unique physical GPU UUIDs')
        if any('tsinghua-zt6' in job['host'] and int(job['gpu']) == 0 for job in self.allocation):
            raise ValueError('The allocation includes explicitly excluded tsinghua-zt6 GPU0')
        self.validation_allocation = [job for job in self.allocation
            if ((job['host'] in ('training-vlawm-tsinghua-zt6', 'training-vlawm-zt') and int(job['gpu']) in (5, 6, 7)) or
                (job['host'] == 'training-vlawm-tsinghua-zt3' and int(job['gpu']) in range(8)))]
        if len(self.validation_allocation) != len(self.allocation) - 32:
            raise ValueError('The registered validation reserve must leave exactly 32 formal training GPUs')
        self.registry.update(owner_pid=os.getpid(), owner_host=socket.gethostname(),
                             owner_start_ticks=process_start_ticks(os.getpid()), allocation=ref(args.allocation))
        self.write('WAITING_GT')

    def write(self, status=None, **fields):
        if status and status != self.registry.get('status'):
            self.registry.setdefault('history', []).append({'time': stamp(), 'status': status})
            self.registry['status'] = status
        self.registry.update(fields, time=stamp())
        atomic(self.registry_path, self.registry)

    def wait_tick(self, status, **fields):
        self.write(status, **fields)
        print(json.dumps({'status': status, **fields}, allow_nan=False), flush=True)
        time.sleep(self.args.poll_seconds)

    def wait_for_gt(self):
        while True:
            required, requirements = required_GT_jobs(self.args.gt_exit_requirements, self.allocation)
            gt = actor_summary(required)
            export_path = self.args.root / 'native_gt/GT_POST_COMPLETE.json'
            count = len(list((self.args.root / 'native_gt/exports/post').glob('*/COMPLETE.json')))
            if gt['all_exited'] and not gt['all_exit_zero']:
                raise RuntimeError('GT workers exited with a failure; no teacher GPU stage may overlap or silently replace them')
            if export_path.exists() and count == 1850 and gt['all_exit_zero']:
                complete = read(export_path)
                if complete.get('train_failures') != 1850 or complete.get('actual_exports') != 1850:
                    raise ValueError('GT export completion marker does not prove all1850')
                atomic(self.folder / 'GT_EXIT_ACCEPTANCE.json', {'status': 'PASS', 'GT_export': ref(export_path),
                       'workers': gt, 'requirements': requirements,
                       'all1850_native_pairs_present': True, 'no_GT_teacher_GPU_overlap': True})
                return
            self.wait_tick('WAITING_GT', GT_exports=count, required=1850, GT_actors_done=gt['done'],
                           GT_actors_running=gt['pending_or_running'], GT_actors_failed=gt['failed'])

    def cpu(self, name, arguments, complete_path=None):
        entry = self.registry['stages'].get(name, {})
        if entry.get('status') == 'COMPLETE' and (complete_path is None or Path(complete_path).exists()):
            if entry.get('artifact') and sha(entry['artifact']['path']) != entry['artifact']['sha256']:
                raise ValueError(f'Completed CPU artifact changed: {name}')
            return
        command = [self.args.python, '-B', str(self.args.root / 'code/fit_library.py'), *arguments, '--root', str(self.args.root)]
        if name in ('fit_inner', 'refit'):
            command += ['--fit-workers', str(self.args.fit_workers)]
        for attempt in range(self.args.engineering_retries + 1):
            try:
                self.run_process(name, command, complete_path)
                return
            except RuntimeError:
                if attempt == self.args.engineering_retries:
                    raise

    def run_process(self, name, command, complete_path=None):
        log = self.folder / 'logs' / (name + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        task_path = self.folder / 'CPU_TASKS' / (name + '.json')
        state_path = self.folder / 'CPU_TASKS' / (name + '.STATE.json')
        task_path.parent.mkdir(parents=True, exist_ok=True)
        task = {'command': command, 'cwd': str(self.args.round1_repo), 'state': str(state_path),
                'lock': str(task_path.with_suffix('.lock')), 'complete_path': str(complete_path) if complete_path else None}
        if task_path.exists() and read(task_path) != task:
            raise ValueError('A durable CPU phase command changed after registration')
        atomic(task_path, task)
        state = read(state_path) if state_path.exists() else None
        if state and state.get('status') == 'FAILED':
            failures = list(task_path.parent.glob(name + '.FAILED_ATTEMPT_*.json'))
            if len(failures) >= self.args.engineering_retries:
                raise RuntimeError(f'CPU phase {name} exhausted bounded engineering retries; inspect preserved failures')
            atomic(task_path.with_name(name + f'.FAILED_ATTEMPT_{time.time_ns()}.json'), state)
            state_path.unlink()
            state = None
        if state and state.get('status') == 'RUNNING':
            if state.get('host') != socket.gethostname():
                raise RuntimeError('A CPU phase is owned by another host; resume its owner there')
            alive = process_start_ticks(state['pid']) == state['start_ticks']
            child_alive = state.get('child_pid') and process_start_ticks(state['child_pid']) == state['child_start_ticks']
            if not alive and child_alive:
                raise RuntimeError('CPU phase wrapper lost while its science child is live; preserve it and inspect before resuming')
            if not alive:
                atomic(task_path.with_name(name + f'.UNKNOWN_EXIT_{time.time_ns()}.json'), state)
                state = None
        spawned = None
        if not state:
            with log.open('a') as output:
                spawned = subprocess.Popen([self.args.python, '-B', __file__, '--execute-cpu-command', str(task_path)],
                                           cwd=self.args.round1_repo, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            self.registry['stages'][name] = {'status': 'RUNNING', 'command': command, 'log': str(log), 'started': stamp(),
                                             'wrapper_pid': spawned.pid, 'state': str(state_path)}
            self.write(name)
        while not state_path.exists() or read(state_path).get('status') == 'RUNNING':
            if spawned is not None and spawned.poll() is not None:
                raise RuntimeError(f'CPU phase {name} wrapper exited without a terminal durable receipt')
            if state_path.exists():
                current = read(state_path)
                if current.get('status') == 'RUNNING' and process_start_ticks(current['pid']) != current['start_ticks']:
                    raise RuntimeError(f'CPU phase {name} wrapper was lost; inspect the persisted child state before recovery')
            self.wait_tick(name, log=str(log), durable_child_state=str(state_path))
        state = read(state_path)
        entry = self.registry['stages'].setdefault(name, {})
        entry.update(state, log=str(log))
        if state.get('artifact') and sha(state['artifact']['path']) != state['artifact']['sha256']:
            raise ValueError(f'Terminal CPU artifact changed: {name}')
        if state.get('returncode') or (complete_path is not None and not Path(complete_path).exists()):
            entry['status'] = 'FAILED'; self.write('ENGINEERING_FAILURE')
            raise RuntimeError(f'{name} did not finish: inspect {log}')
        entry['status'] = 'COMPLETE'
        if complete_path is not None:
            entry['artifact'] = ref(complete_path)
        self.write()

    def verify_base_qualification(self):
        folder = self.args.root / 'requests/base_qualification'
        request, scores, base = collect_request(folder)
        if len(request['indices']) != 2 or request['noise_indices'] != [0, 1, 2, 3, 4]:
            raise ValueError('Base qualification must contain the complete two-scene five-noise check')
        lineage = None
        feature_ref = request['feature_manifest']
        if sha(feature_ref['path']) != feature_ref['sha256']:
            lineage_path = folder / 'FEATURE_MANIFEST_LINEAGE.json'
            lineage = read(lineage_path)
            archived = lineage['archived_original_manifest']
            if (lineage['original_request_sha256'] != sha(folder / 'REQUEST.json') or
                    archived['sha256'] != feature_ref['sha256'] or sha(archived['path']) != archived['sha256'] or
                    read(archived['path'])['populations']['train'] != read(self.args.features)['populations']['train']):
                raise ValueError('Base qualification manifest relocation lineage is not exact')
            lineage = ref(lineage_path)
        queue = read(folder / 'QUEUE.json')
        for identity in queue['tasks']:
            result = read(read(folder / 'receipts' / (identity + '.json'))['result']['path'])
            if result.get('baseline_validation') != 'all_Base_trajectory_and_seven_metric_values_exact_to_independent_local_cache':
                raise ValueError('Base qualification did not prove all native trajectories and PDM fields')
        atomic(self.folder / 'BASE_QUALIFICATION_ACCEPTED.json', {'status': 'PASS', 'evidence': ref(folder / 'COMPLETE.json'),
                                                               'manifest_relocation_lineage': lineage,
                                                               'noise_indices': [0, 1, 2, 3, 4], 'scenes': 2})

    def live_headroom(self, allocation=None):
        required, requirements = required_GT_jobs(self.args.gt_exit_requirements, self.allocation)
        if not actor_summary(required)['all_exit_zero']:
            raise RuntimeError('A GT overlap actor is still live or failed; no teacher actor may be admitted')
        by_host = defaultdict(list)
        selected = self.allocation if allocation is None else allocation
        grant = read('/mnt/project/server_dispatch_policy.json')['task_authorizations']['recogdrive_no_routing_round1_20261026']
        if grant.get('status') != 'ACTIVE_TASK_BOUNDED_GRANT':
            raise RuntimeError('The current Round1 task grant is no longer active')
        for job in selected:
            if (job['host'] not in grant['temporary_hosts'] or int(job['gpu']) not in grant['authorized_gpu_indices'][job['host']] or
                    int(job['gpu']) in grant.get('excluded_gpu_indices', {}).get(job['host'], [])):
                raise RuntimeError('The registered physical GPU is outside the current Round1 grant')
            by_host[job['host']].append(job['uuid'])
        proof = {}
        for host, uuids in by_host.items():
            command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
                       'nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits']
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            if result.returncode:
                raise RuntimeError(f'Cannot verify actual resource headroom on {host}: {result.stderr[-500:]}')
            values = {line.split(',')[0].strip(): int(line.split(',')[1].strip()) for line in result.stdout.splitlines() if ',' in line}
            if any(uuid not in values or values[uuid] < self.args.minimum_free_mb for uuid in uuids):
                raise RuntimeError(f'Registered GPU UUID/headroom changed on {host}; preserve other work and resume after capacity is available')
            proof[host] = {uuid: values[uuid] for uuid in uuids}
        return proof

    def launch_attempt(self, name, queue_folder, attempt, population):
        stage = self.args.root / 'dispatch/local_teacher' / name / f'attempt_{attempt:02d}'
        jobs_path = stage / 'JOBS.json'
        sys.path.insert(0, str(LEGACY / 'local'))
        import dispatch
        dispatch.TASK = 'recogdrive_no_routing_round1_20261026'
        launch = dispatch.launch
        by_host = defaultdict(list)
        jobs = read(jobs_path) if jobs_path.exists() else []
        allocation = self.validation_allocation if population == 'validation' else self.allocation
        if jobs:
            for job in jobs:
                by_host[job['host']].append((job['name'], job['uuid'], job['arguments']))
            for host, host_jobs in by_host.items():
                directory = stage / host
                if not (directory / 'launcher.pid').exists():
                    self.live_headroom(allocation)
                    launch(host, directory, host_jobs)
            return jobs
        proof = self.live_headroom(allocation)
        cohort = 'navtest' if population == 'test' else 'navtrain'
        allocations = [allocated for allocated in allocation for _ in range(self.args.actors_per_gpu)]
        for rank, allocated in enumerate(allocations):
            host = allocated['host']; worker = f'{name}_attempt{attempt:02d}_rank{rank:03d}'
            arguments = [self.args.python, '-B', str(self.args.root / 'code/teacher_eval_worker.py'),
                         '--root', str(self.args.root), '--queue', str(queue_folder / 'QUEUE.json'),
                         '--round1-repo', str(self.args.round1_repo), '--paths', str(self.args.paths),
                         '--noise-protocol', str(self.args.noise_protocol),
                         '--noise-index', str(self.args.noise_root / 'manifests' / f'{cohort}_scenes.json'),
                         '--noise-bank', str(self.args.noise_root / 'noise' / f'{cohort}_noise.npy'),
                         '--worker-id', worker, '--max-attempts', str(self.args.engineering_retries + 1), '--fail-fast']
            if self.args.deadline_unix:
                arguments += ['--deadline-unix', str(self.args.deadline_unix)]
            if self.args.inference_graph_receipt:
                arguments += ['--inference-graph-receipt', str(self.args.inference_graph_receipt)]
            by_host[host].append((worker, allocated['uuid'], arguments))
            jobs.append({'name': worker, 'host': host, 'gpu': allocated['gpu'], 'uuid': allocated['uuid'],
                         'arguments': arguments,
                         'status': str(stage / host / 'status' / (worker + '.json')),
                         'worker_receipt': str(queue_folder / 'workers' / worker / 'COMPLETE.json')})
        stage.mkdir(parents=True, exist_ok=True)
        atomic(stage / 'HEADROOM.json', {'time': stamp(), 'free_MB_by_UUID': proof})
        # Save the complete dispatch intent before launching any host; restart
        # reuses per-host launcher.pid and never creates duplicate actors.
        atomic(jobs_path, jobs)
        for host, host_jobs in by_host.items():
            directory = stage / host
            if not (directory / 'launcher.pid').exists():
                launch(host, directory, host_jobs)
        return jobs

    def scoring(self, name, population):
        folder = self.args.root / 'requests' / name
        if (folder / 'COMPLETE.json').exists():
            complete = read(folder / 'COMPLETE.json')
            if sha(complete['matrix']['path']) != complete['matrix']['sha256']:
                raise ValueError('A completed teacher score matrix is corrupt')
            launches = sorted((self.args.root / 'dispatch/local_teacher' / name).glob('attempt_*/JOBS.json'))
            if not launches:
                raise ValueError('A score matrix exists without durable native actor exit evidence')
            jobs = read(launches[-1])
            while not actor_summary(jobs)['all_exited']:
                self.wait_tick(f'WAITING_{name.upper()}_ACTOR_EXIT', actors=actor_summary(jobs))
            if not actor_summary(jobs)['all_exit_zero']:
                raise RuntimeError('Completed scores have a failed actor exit; retain evidence and resolve engineering failure')
            return
        for attempt in range(self.args.engineering_retries + 1):
            jobs = self.launch_attempt(name, folder, attempt, population)
            while True:
                actors = actor_summary(jobs)
                progress = queue_progress(folder)
                if actors['all_exited']:
                    atomic(self.folder / f'{name}_attempt{attempt:02d}_EXIT.json', actors)
                    if progress['pending'] == 0 and actors['all_exit_zero']:
                        collect_request(folder)
                        self.registry['stages'][name] = {'status': 'COMPLETE', 'attempt': attempt,
                                                        'actor_exit_evidence': ref(self.folder / f'{name}_attempt{attempt:02d}_EXIT.json'),
                                                        'score_matrix': ref(folder / 'COMPLETE.json')}
                        self.write()
                        return
                    break
                self.wait_tick(f'SCORING_{name.upper()}', queue_progress=progress,
                               actors_done=actors['done'], actors_failed=actors['failed'], actors_running=actors['pending_or_running'])
            if self.args.deadline_unix and time.time() >= self.args.deadline_unix:
                raise RuntimeError('Allocated task deadline reached; complete receipts preserved and no new actors admitted')
        raise RuntimeError(f'Native {name} scoring remains incomplete after bounded same-protocol recovery attempts')

    def targets(self, population):
        if population != 'validation':
            raise ValueError('Train/Test target generation belongs exclusively to the root preparation owner')
        output = self.args.results / 'targets' / population
        path = output / 'MANIFEST.json'
        if path.exists():
            value = read(path)
            if population in value['populations']:
                spec = value['populations'][population]
                expected = 85813 if population == 'train' else 17475 if population == 'validation' else 12146
                if spec['count'] != expected or set(spec['versions']) != {'full', 'gain1'}:
                    raise ValueError('Existing target population is incomplete')
                for artifact in [spec['index'], *[version['shifts'] for version in spec['versions'].values()]]:
                    if sha(artifact['path']) != artifact['sha256']:
                        raise ValueError('Existing target bytes are corrupt')
                launches = sorted((self.args.root / 'dispatch/local_teacher/validation_targets').glob('attempt_*/JOBS.json'))
                if not launches:
                    raise ValueError('Validation target output lacks a registered real GPU export exit receipt')
                while not actor_summary(read(launches[-1]))['all_exited']:
                    self.wait_tick('WAITING_VALIDATION_TARGET_ACTOR_EXIT', actors=actor_summary(read(launches[-1])))
                if not actor_summary(read(launches[-1]))['all_exit_zero']:
                    raise RuntimeError('Validation target GPU actor failed after writing output; preserve and recover its exit evidence')
                return path
        paths = read(self.args.paths)
        paths.update(frozen_repo=str(self.args.round1_repo), feature_manifest=str(self.args.features),
                     teacher_bundle=str(self.args.root / 'frozen_teacher/BUNDLE.json'))
        config = self.folder / f'TARGET_PATHS_{population}.json'
        atomic(config, paths)
        command = ['env','PYTHONPATH='+str(self.args.round1_repo),self.args.python, '-B', '-m', 'student_distillation.build_targets', '--paths', str(config),
                   '--population', population, '--target-versions', 'full', 'gain1', '--device', 'cuda', '--out', str(output)]
        if population != 'train':
            command += ['--diagnostic-only']
        sys.path.insert(0, str(LEGACY / 'local'))
        import dispatch
        dispatch.TASK = 'recogdrive_no_routing_round1_20261026'
        launch = dispatch.launch
        allocated = self.validation_allocation[0]
        for attempt in range(self.args.engineering_retries + 1):
            directory = self.args.root / 'dispatch/local_teacher/validation_targets' / f'attempt_{attempt:02d}' / allocated['host']
            job_name = f'validation_targets_attempt{attempt:02d}'
            jobs_path = directory.parent / 'JOBS.json'
            jobs = [{'name': job_name, 'host': allocated['host'], 'gpu': allocated['gpu'], 'uuid': allocated['uuid'],
                     'status': str(directory / 'status' / (job_name + '.json')), 'arguments': command}]
            if not jobs_path.exists():
                self.live_headroom([allocated])
                atomic(jobs_path, jobs)
            if not (directory / 'launcher.pid').exists():
                launch(allocated['host'], directory, [(job_name, allocated['uuid'], command)])
            while not actor_summary(jobs)['all_exited']:
                self.wait_tick('VALIDATION_GPU_TARGETS', actors=actor_summary(jobs))
            if actor_summary(jobs)['all_exit_zero'] and path.exists():
                self.registry['stages']['build_targets_validation'] = {'status': 'COMPLETE', 'command': command,
                    'actor_exit': actor_summary(jobs), 'artifact': ref(path)}
                self.write()
                return path
        raise RuntimeError('Validation GPU target export did not finish after bounded recovery')

    def run(self):
        self.wait_for_gt()
        self.verify_base_qualification()
        self.cpu('fit_inner', ['fit-inner'], self.args.root / 'fitting/INNER/BANK.json')
        self.cpu('pack_pilot', ['pack-pilot'], self.args.root / 'requests/pilot/QUEUE.json')
        self.scoring('pilot', 'train')
        self.cpu('collect_pilot', ['collect-pilot'], self.args.root / 'fitting/COARSE_SELECTION.json')
        self.cpu('pack_cal', ['pack-cal'], self.args.root / 'requests/cal/QUEUE.json')
        self.scoring('cal', 'train')
        self.cpu('collect_cal', ['collect-cal'], self.args.root / 'fitting/FROZEN_INNER_SELECTION.json')
        self.cpu('refit', ['refit'], self.args.root / 'fitting/ALLTRAIN/BANK.json')
        self.cpu('freeze', ['freeze'], self.args.root / 'frozen_teacher/BUNDLE.json')
        for population in ('train', 'test', 'validation'):
            self.cpu(f'queue_{population}', ['queue-population', '--population', population, '--features', str(self.args.features)],
                     self.args.root / 'requests' / population / 'QUEUE.json')
            self.scoring(population, population)
            self.cpu(f'collect_{population}', ['collect-population', '--population', population],
                     self.args.root / 'frozen_teacher' / f'{population.upper()}_EVALUATION.json')
            if population == 'test':
                emit_teacher_report(self.args)
                report_snapshot = self.args.root / 'frozen_teacher/LOCAL_TEACHER_TRAIN_TEST_REPORT.md'
                if not report_snapshot.exists():
                    shutil.copyfile(self.args.root / 'frozen_teacher/LOCAL_TEACHER_REPORT.md', report_snapshot)
                atomic(self.folder / 'FULL_TRAIN_TEST_TEACHER_READY.json', {'status': 'COMPLETE',
                       'teacher_bundle': ref(self.args.root / 'frozen_teacher/BUNDLE.json'),
                       'teacher_report': ref(report_snapshot), 'Train_Test_targets_exclusively_owned_by_root_preparation': True,
                       'reserved_validation_GPU_UUIDs': [job['uuid'] for job in self.validation_allocation]})
        val_targets = self.targets('validation')
        emit_teacher_report(self.args)
        self.args.targets_manifest, self.args.diagnostics_out = val_targets, self.args.results / 'VAL_TEACHER_DIAGNOSTICS.json'
        emit_validation_diagnostics(self.args)
        costs = self.costs()
        self.write('LOCAL_TEACHER_COMPLETE', validation_targets=ref(val_targets),
                   validation_teacher_diagnostics=ref(self.args.diagnostics_out))
        atomic(self.folder / 'COMPLETE.json', {'status': 'LOCAL_TEACHER_COMPLETE',
               'all_registered_GPU_actors_exited': True, 'final_attempt_each_required_stage_exit_zero': True,
               'failed_actor_attempts_retained': costs['failed_actor_attempts'],
               'costs': ref(self.folder / 'COSTS.json'), 'teacher_bundle': ref(self.args.root / 'frozen_teacher/BUNDLE.json'),
               'Train_Test_targets_exclusively_owned_by_root_preparation': True, 'validation_targets': ref(val_targets),
               'validation_teacher_diagnostics': ref(self.args.diagnostics_out)})

    def costs(self):
        import importlib.util
        path = self.args.root / 'engineering/report_teacher_physical_costs.py'
        spec = importlib.util.spec_from_file_location('nrkd_teacher_physical_costs', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        manifests = sorted((self.args.root / 'dispatch/local_teacher').glob('*/attempt_*/JOBS.json'))
        measured = module.compute(manifests)
        value = {'status': 'COMPLETE_MEASURED_ACTOR_AND_PHYSICAL_GPU_COST', **measured,
                 'includes_CPU_scoring_and_IO_wait_within_each_GPU_actor': True,
                 'CPU_phases': {name: {'wall_seconds': entry.get('wall_seconds'), 'command': entry.get('command')}
                                for name, entry in self.registry['stages'].items() if entry.get('child_pid')},
                 'GT_search_cost_tracked_separately': True,
                 'physical_cost_function_source': ref(path)}
        atomic(self.folder / 'COSTS.json', value)
        return value


def selftest(root):
    """CPU-only admission/exit accounting and detached child receipt checks."""
    folder = Path(root) / 'selftests/teacher_owner'
    folder.mkdir(parents=True, exist_ok=True)
    a, b = folder / 'actor_a.json', folder / 'actor_b.json'
    jobs = [{'status': str(a)}, {'status': str(b)}]
    atomic(a, {'state': 'done', 'returncode': 0})
    atomic(b, {'state': 'running'})
    if actor_summary(jobs)['all_exited']:
        raise AssertionError('A running actor was accepted as exited')
    atomic(b, {'state': 'failed', 'returncode': 1})
    if actor_summary(jobs)['all_exit_zero']:
        raise AssertionError('A failed actor was accepted as successful')
    atomic(b, {'state': 'done', 'returncode': 0})
    if not actor_summary(jobs)['all_exit_zero']:
        raise AssertionError('Actual zero exits were not accepted')
    manifest = folder / 'JOBS.json'; atomic(manifest, jobs)
    requirements = folder / 'GT_EXIT_REQUIREMENTS.json'
    atomic(requirements, {'job_manifests': [ref(manifest)]})
    required, _ = required_GT_jobs(requirements, [])
    if len(required) != 2 or not actor_summary(required)['all_exit_zero']:
        raise AssertionError('Expanded GT actor requirements were ignored')
    marker = folder / 'CPU_OUTPUT.txt'
    task_path = folder / 'CPU_TASK.json'
    task = {'command': [sys.executable, '-c', 'from pathlib import Path\nPath(' + repr(str(marker)) + ').write_text("actual CPU child completed\\n")'],
            'cwd': str(folder), 'state': str(folder / 'CPU_TASK.STATE.json'), 'lock': str(folder / 'CPU_TASK.lock'),
            'complete_path': str(marker)}
    atomic(task_path, task)
    subprocess.run([sys.executable, '-B', __file__, '--execute-cpu-command', str(task_path)], check=True)
    state = read(task['state'])
    if state['status'] != 'COMPLETE' or state['returncode'] != 0 or state['artifact'] != ref(marker):
        raise AssertionError('Durable CPU wrapper did not persist the actual child success')
    value = {'status': 'PASS_CPU_DURABLE_TEACHER_OWNER', 'real_GPU_or_ssh_calls': 0,
             'actual_child_exit_and_artifact_SHA_checked': True, 'failed_or_running_actor_rejected': True,
             'all_expanded_GT_manifest_exit_requirements_checked': True}
    atomic(folder / 'COMPLETE.json', value)
    return value


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--execute-cpu-command':
        execute_cpu_task(sys.argv[2])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--round1-repo', type=Path, default=ROUND1)
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--python', default=PYTHON)
    parser.add_argument('--paths', type=Path, default=LEGACY / 'paths.local.json')
    parser.add_argument('--features', type=Path, default=RESULTS / 'features/MANIFEST.ALL_POPULATIONS.json')
    parser.add_argument('--allocation', type=Path, default=ROOT / 'dispatch/gt_full/JOBS.json')
    parser.add_argument('--gt-exit-requirements', type=Path, default=ROOT / 'dispatch/GT_EXIT_REQUIREMENTS.json')
    parser.add_argument('--noise-protocol', type=Path, default=OLD / 'reference/noise/PROTOCOL.json')
    parser.add_argument('--noise-root', type=Path, default=OLD / 'reference')
    parser.add_argument('--poll-seconds', type=float, default=30.)
    parser.add_argument('--engineering-retries', type=int, default=2)
    parser.add_argument('--minimum-free-mb', type=int, default=20000)
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--fit-workers', type=int, choices=range(1, 9), default=8)
    parser.add_argument('--inference-graph-receipt', type=Path, default=RESULTS / 'qualification/NATIVE_INFERENCE_GRAPH_ADMISSION.json')
    parser.add_argument('--actors-per-gpu', type=int, choices=(1, 2), default=2)
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        print(json.dumps(selftest(args.root)), flush=True)
        return
    args.targets_manifest = None; args.diagnostics_out = None
    try:
        Owner(args).run()
    except BaseException:
        atomic(args.root / 'teacher_suite/INCOMPLETE.json', {'status': 'INCOMPLETE', 'time': stamp(),
               'traceback': traceback.format_exc(), 'resume_argv': [sys.executable, __file__, *sys.argv[1:]],
               'complete_flags_not_written_for_unfinished_tasks': True})
        raise


if __name__ == '__main__':
    main()
