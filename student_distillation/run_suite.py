"""Execute and resume the full eight-run plan through the locked Test and report.

This controller executes child processes and waits for their real completion.
It never treats a launched/background job as a completed scientific task.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import fcntl
import os
from pathlib import Path
import shlex
import subprocess
import socket
import sys
import time
import threading
import yaml
from .checkpointing import atomic_json, sha256
from .experiment_registry import EXPERIMENTS, SEEDS, Registry, identities_for, load_config, merge, stamp, validate_config


class RemoteOwnershipUnknown(RuntimeError):
    """Do not duplicate an unproven remote exit or orphaned GPU process."""


def inventory():
    result = {"time": stamp(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        value = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=15, check=True)
        result["gpus"] = value.stdout.strip().splitlines()
        value = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=15, check=True)
        result["compute_processes"] = value.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError) as error:
        result["gpu_inventory_error"] = str(error)
    try:
        import psutil
        result.update(cpu_percent=psutil.cpu_percent(), memory_available_bytes=psutil.virtual_memory().available,
                      io_bytes=psutil.disk_io_counters()._asdict())
    except ImportError:
        pass
    return result


def child_command(module, arguments, world_size):
    prefix = [sys.executable, "-B"]
    if world_size > 1:
        prefix += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world_size}", "--module", module]
    else:
        prefix += ["-m", module]
    return prefix + [str(value) for value in arguments]


def _start_ticks(pid):
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


def _lock_held(path):
    with Path(path).open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def _read_json(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else {}


def _admit_spec(spec):
    """Validate the actual host, source and live grant before starting a child."""
    requested = set(spec["gpu_uuids"])
    probe = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True, timeout=20)
    devices = {parts[1].strip(): (int(parts[0]), int(parts[2])) for parts in
               (line.split(",") for line in probe.stdout.splitlines() if line.strip())}
    if not requested <= set(devices):
        raise ValueError("Allocated GPU UUIDs are not present on the selected host")
    live = _read_json(spec["live_policy_path"])
    grant = live.get("task_authorizations", {}).get(spec["task"])
    if not grant or grant.get("status") != "ACTIVE_TASK_BOUNDED_GRANT" or grant.get("owned_resources_released") is True:
        raise ValueError("The live task-specific resource grant is no longer active")
    allowed = set(grant.get("authorized_gpu_indices", {}).get(spec["host"], []))
    excluded = set(grant.get("excluded_gpu_indices", {}).get(spec["host"], []))
    if any(devices[uuid][0] not in allowed or devices[uuid][0] in excluded for uuid in requested):
        raise ValueError("The current live policy forbids an allocated GPU index")
    if any(devices[uuid][1] < spec["min_free_memory_mib"] for uuid in requested):
        raise ValueError("An allocated GPU lacks measured headroom; preserve its other occupants")
    for relative, checksum in spec["code_files"].items():
        if sha256(Path(spec["workdir"]) / relative) != checksum:
            raise ValueError(f"Remote source identity differs: {relative}")
    return {"live_policy_sha256_at_admission": sha256(spec["live_policy_path"]),
            "gpu_index_and_free_mib_at_admission": {uuid: devices[uuid] for uuid in requested}}


def execute_spec(path):
    """Run one immutable attempt; old/delayed wrappers cannot start a new child."""
    import signal
    import traceback
    path = Path(path).resolve()
    spec = _read_json(path)
    folder = path.parent
    family = Path(spec.get("family", folder)).resolve()
    family_lease = (family / "FAMILY_LEASE.lock").open("a+")
    lease = (folder / "LEASE.lock").open("a+")
    acquired = []
    try:
        for handle in (family_lease, lease):
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired.append(handle)
            except BlockingIOError:
                return 75
        active = _read_json(family / "ACTIVE_ATTEMPT.json")
        if active and (active.get("spec") != str(path) or active.get("spec_sha256") != sha256(path)):
            # A transport-delayed wrapper is harmless after a newer pointer was
            # committed under the same family lease. It must never launch.
            if not (folder / "STATUS.json").exists():
                atomic_json(folder / "STATUS.json", {"status": "SUPERSEDED_BEFORE_START", "child_started": False,
                            "spec_sha256": sha256(path), "end": stamp()})
            return 75
        status = folder / "STATUS.json"
        previous = _read_json(status)
        if previous:
            if previous.get("spec_sha256") != sha256(path):
                raise ValueError("Durable job spec changed")
            if previous.get("status") == "EXITED":
                return int(previous["returncode"])
            # Each attempt starts its child at most once. A missing owner after
            # a recorded start is an unknown exit, never permission to repeat.
            if previous.get("child_started") or previous.get("status") in ("STARTING", "RUNNING", "EXIT_UNKNOWN"):
                return 76
            return 1
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        header = {"spec_sha256": sha256(path), "host": socket.gethostname(), "pid": os.getpid(),
                  "process_start_ticks": _start_ticks(os.getpid()), "gpu_uuids": spec["gpu_uuids"],
                  "command": spec["command"], "start": stamp(), "child_started": False}
        atomic_json(family / "FAMILY_OWNER.json", {"attempt": str(folder), **header})
        atomic_json(status, {"status": "ADMITTING", **header})
        started = time.monotonic()
        process = None
        try:
            header.update(_admit_spec(spec))
            environment = os.environ.copy()
            environment.update(spec["environment"])
            atomic_json(status, {"status": "STARTING", **header})
            with Path(spec["log"]).open("a") as log:
                process = subprocess.Popen(spec["command"], cwd=spec["workdir"], env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True,
                                           pass_fds=(family_lease.fileno(), lease.fileno()))
                header.update(child_started=True, child_pid=process.pid)
                try:
                    header["child_start_ticks"] = _start_ticks(process.pid)
                except FileNotFoundError:
                    pass  # An immediately completed child still has a real wait status.
                atomic_json(status, {"status": "RUNNING", **header})
                last_resource = 0.
                while process.poll() is None:
                    now = time.monotonic()
                    if now - last_resource >= 30:
                        metrics = {"time": stamp(), "elapsed_seconds": now - started,
                                   "allocated_gpu_uuids": spec["gpu_uuids"], **inventory()}
                        rows = [line.split(",") for line in metrics.get("gpus", [])]
                        utilization = [float(row[-1].strip().rstrip(" %")) for row in rows
                                       if len(row) >= 6 and row[1].strip() in spec["gpu_uuids"]]
                        gpu_util = sum(utilization) / len(utilization) if utilization else None
                        metrics["mean_allocated_gpu_utilization_percent"] = gpu_util
                        metrics["bottleneck_observation"] = ("GPU busy" if gpu_util is not None and gpu_util >= 75 else
                            "CPU busy while GPU underused" if gpu_util is not None and gpu_util < 25 and metrics.get("cpu_percent", 0) >= 85 else
                            "GPU underused; inspect CPU/IO counters" if gpu_util is not None and gpu_util < 25 else "mixed or unmeasured")
                        atomic_json(folder / "RESOURCE_STATE.json", metrics)
                        with (folder / "runtime.jsonl").open("a") as handle:
                            handle.write(json.dumps(metrics) + "\n")
                        last_resource = now
                    time.sleep(2)
                result = process.returncode
            atomic_json(status, {"status": "EXITED", **header, "returncode": result,
                                 "wall_seconds": time.monotonic() - started, "end": stamp()})
            return result
        except Exception as error:
            # A source/policy/import/Popen failure before any child is a known
            # startup failure. A wrapper failure after launch is not a child exit.
            atomic_json(status, {"status": "STARTUP_FAILED" if process is None else "EXIT_UNKNOWN", **header,
                        "returncode": 1 if process is None else None, "error": repr(error),
                        "traceback": traceback.format_exc(), "wall_seconds": time.monotonic() - started, "end": stamp()})
            return 1 if process is None else 76
    finally:
        for handle in reversed(acquired):
            fcntl.flock(handle, fcntl.LOCK_UN)
        lease.close()
        family_lease.close()


def load_resources(path, out):
    value = json.loads(Path(path).read_text())
    allocations = value.get("allocations", value.get("runs"))
    expected = {f"{experiment}_seed{seed}" for experiment in EXPERIMENTS for seed in SEEDS}
    if not isinstance(allocations, dict) or set(allocations) != expected:
        raise ValueError("Resources must explicitly allocate all eight registered runs")
    used = set()
    for run_id, allocation in allocations.items():
        uuids = allocation.get("gpu_uuids", [])
        if len(uuids) != 4 or len(set(uuids)) != 4 or any(not str(uuid).startswith("GPU-") for uuid in uuids):
            raise ValueError(f"{run_id} requires four explicit GPU UUIDs")
        if used & set(uuids):
            raise ValueError("Independent runs may not overlap an allocated GPU")
        used.update(uuids)
        if not allocation.get("host") or not Path(allocation.get("python", "")).is_absolute() or not Path(allocation.get("workdir", "")).is_absolute():
            raise ValueError("Each run allocation requires host, absolute python and shared workdir")
    if used & set(value.get("protected_gpu_uuids", [])):
        raise ValueError("A run allocation includes a protected GPU")
    for key in ("authorization", "resource_policy"):
        ref = value.get(key)
        if not ref or not Path(ref.get("path", "")).is_file() or sha256(ref["path"]) != ref.get("sha256"):
            raise ValueError(f"Resources require a verified root-owned {key} artifact")
    authorization = json.loads(Path(value["authorization"]["path"]).read_text())
    policy = json.loads(Path(value["resource_policy"]["path"]).read_text())
    for allocation in allocations.values():
        allocation["_task"] = authorization["task"]
        allocation["_live_policy_path"] = policy["source"]
        allocation["_resource_status"] = value.get("status")
    atomic_json(Path(out) / "RESOURCE_ALLOCATION.json", {"source": str(Path(path).resolve()),
                                                        "sha256": sha256(path), "allocation": value})
    return allocations


class Controller:
    def __init__(self, out, registry, allocations):
        self.out, self.registry = Path(out), registry
        self.allocations, self.world_size = allocations, 4
        self.lock = threading.RLock()
        self.commands = self.out / "launch_commands.audit.sh"
        if not self.commands.exists():
            self.commands.write_text("#!/usr/bin/env bash\nset -euo pipefail\n")

    def execute(self, key, command, run_id=None, allocation_run=None, success_check=None):
        allocation_run = allocation_run or run_id or sorted(self.allocations)[0]
        allocation = self.allocations[allocation_run]
        if allocation.get("_resource_status") != "READY_FOR_FORMAL_DISPATCH":
            raise ValueError("The root resource/headroom allocation is still planned, not READY_FOR_FORMAL_DISPATCH")
        selected = allocation["gpu_uuids"]
        environment = {"CUDA_VISIBLE_DEVICES": ",".join(selected), "PYTHONDONTWRITEBYTECODE": "1",
                       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OPENBLAS_CORETYPE": "Sandybridge", "ZT_AUTO_PROXY": "0"}
        command = [str(allocation["python"]), *map(str, command[1:])]
        logs = self.out / "controller_logs"
        logs.mkdir(exist_ok=True)
        family = (self.out / "jobs" / key).resolve()
        family.mkdir(parents=True, exist_ok=True)
        code_root = Path(__file__).resolve().parents[1]
        code_files = {str(path.relative_to(code_root)): sha256(path) for path in sorted((code_root / "student_distillation").glob("*.py"))}
        scientific = {"command": command, "environment": environment, "gpu_uuids": selected,
                      "workdir": allocation["workdir"], "host": allocation["host"], "code_files": code_files,
                      "live_policy_path": allocation["_live_policy_path"], "task": allocation["_task"],
                      "min_free_memory_mib": int(allocation.get("min_free_memory_mib", 12000))}
        family_lease = (family / "FAMILY_LEASE.lock").open("a+")
        held = False
        try:
            try:
                fcntl.flock(family_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = True
            except BlockingIOError:
                pass
            pointer = _read_json(family / "ACTIVE_ATTEMPT.json")
            if not held:
                if not pointer.get("spec"):
                    raise RemoteOwnershipUnknown(f"Held family lease lacks an owner record: {key}")
                job = Path(pointer["spec"]).parent
                spec_path = job / "SPEC.json"
                old = _read_json(spec_path)
                if any(old.get(name) != value for name, value in scientific.items()):
                    raise RemoteOwnershipUnknown(f"Live job differs from requested command/source/allocation: {key}")
                active = True
            else:
                active = False
                previous_job = Path(pointer["spec"]).parent if pointer else None
                previous = _read_json(previous_job / "STATUS.json") if previous_job else {}
                if previous.get("status") == "EXITED" and previous.get("returncode") == 0 and success_check is not None and success_check():
                    return float(previous.get("wall_seconds", 0.))
                if previous.get("status") in ("STARTING", "RUNNING", "EXIT_UNKNOWN") or (
                        previous.get("status") == "EXITED" and previous.get("returncode", 0) < 0):
                    raise RemoteOwnershipUnknown(f"Unheld lease after a recorded child start or signalled exit: {key}; reconcile PID/start ticks and surviving ranks before retry")
                # An unacknowledged transport may still have a delayed wrapper.
                # Moving this pointer while holding the family lease invalidates
                # that wrapper before it can admit any child.
                if previous_job and not previous:
                    old_spec = previous_job / "SPEC.json"
                    atomic_json(previous_job / "STATUS.json", {"status": "LAUNCH_UNCONFIRMED", "child_started": False,
                                "actual_process_exit": "UNKNOWN", "spec_sha256": sha256(old_spec), "end": stamp()})
                attempts = sorted(family.glob("attempt_[0-9][0-9][0-9][0-9][0-9]"))
                number = max((int(path.name.rsplit("_", 1)[1]) for path in attempts), default=-1) + 1
                job = family / f"attempt_{number:05d}"
                job.mkdir()
                spec_path = job / "SPEC.json"
                spec = {**scientific, "family": str(family), "attempt": number, "log": str(job / "child.log")}
                atomic_json(spec_path, spec)
                atomic_json(family / "ACTIVE_ATTEMPT.json", {"spec": str(spec_path), "spec_sha256": sha256(spec_path), "attempt": number})
                history = _read_json(family / "ATTEMPT_HISTORY.json") or {"schema": "nrkd_job_attempt_history_v1", "key": key, "attempts": []}
                history["attempts"].append({"attempt": number, "spec": str(spec_path), "spec_sha256": sha256(spec_path),
                                            "status": str(job / "STATUS.json"), "log": str(job / "child.log"), "created": stamp()})
                atomic_json(family / "ATTEMPT_HISTORY.json", history)
        finally:
            if held:
                fcntl.flock(family_lease, fcntl.LOCK_UN)
            family_lease.close()
        spec = _read_json(spec_path)
        log, status_path = Path(spec["log"]), job / "STATUS.json"
        attempt_key = key + "__launch" + job.name.rsplit("_", 1)[-1]
        with self.lock:
            if not active:
                with self.commands.open("a") as handle:
                    handle.write("# host " + allocation["host"] + " attempt " + job.name + "\nCUDA_VISIBLE_DEVICES=" +
                                 shlex.quote(environment["CUDA_VISIBLE_DEVICES"]) + " " + shlex.join(command) + "\n")
            existing = self.registry.value.get("tasks", {}).get(attempt_key, {})
            self.registry.task_update(attempt_key, "RUNNING", command=command, devices=selected, host=allocation["host"],
                                      log=str(log), spec=str(spec_path), logical_key=key, start=existing.get("start", stamp()))
            if run_id:
                self.registry.run_update(run_id, "TRAINING", command=command, world_size=4, devices=selected, host=allocation["host"])
        start = time.monotonic()
        launcher = None
        wrapper_log = job / "wrapper.log"
        if not active:
            wrapper = [allocation["python"], "-B", "-m", "student_distillation.run_suite", "--execute-spec", str(spec_path)]
            host = allocation.get("ssh_target", allocation["host"])
            try:
                if host in ("local", "localhost", socket.gethostname()):
                    with wrapper_log.open("a") as output:
                        launcher = subprocess.Popen(wrapper, cwd=allocation["workdir"], stdout=output, stderr=subprocess.STDOUT,
                                                    stdin=subprocess.DEVNULL, start_new_session=True, env={**os.environ, **environment})
                else:
                    # Startup BLAS/proxy policy must also cover imports before the
                    # durable wrapper creates its child environment.
                    prefix = ["env", *[name + "=" + value for name, value in environment.items()]]
                    script = "cd " + shlex.quote(allocation["workdir"]) + "\nnohup " + shlex.join(prefix + wrapper) + " >" + shlex.quote(str(wrapper_log)) + " 2>&1 </dev/null &\n"
                    launcher = subprocess.Popen(["ssh", "-o", "BatchMode=yes", host, shlex.join(["bash", "-lc", script])],
                                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError as error:
                self._unconfirmed_launch(key, attempt_key, job, spec_path, repr(error))
                raise RuntimeError(f"Wrapper could not be started for {key}; preserved {job}") from error
        last_heartbeat = 0.
        unheld_observations = 0
        while True:
            value = _read_json(status_path)
            state = value.get("status")
            if state in ("EXITED", "STARTUP_FAILED"):
                returncode, elapsed = value["returncode"], value.get("wall_seconds", 0.)
                break
            if state in ("EXIT_UNKNOWN",):
                raise RemoteOwnershipUnknown(f"Wrapper lost a known child for {key}; preserve {status_path} and reconcile its process")
            if state in ("SUPERSEDED_BEFORE_START", "LAUNCH_UNCONFIRMED", "TRANSPORT_FAILED"):
                raise RuntimeError(f"Attempt did not start its child for {key}; preserved {status_path}; resume uses a new attempt")
            launch_code = launcher.poll() if launcher is not None else None
            if not value and ((launch_code not in (None, 0)) or time.monotonic() - start > 180):
                if self._unconfirmed_launch(key, attempt_key, job, spec_path, f"launcher_returncode={launch_code}"):
                    raise RuntimeError(f"Wrapper transport/startup was not acknowledged for {key}; actual process exit UNKNOWN; a resumed attempt supersedes any delayed wrapper")
            now = time.monotonic()
            if now - last_heartbeat >= 30.:
                heartbeat = dict(task=attempt_key, elapsed_seconds=now - start, host=allocation["host"], remote=value)
                if (job / "RESOURCE_STATE.json").exists():
                    heartbeat["resources"] = _read_json(job / "RESOURCE_STATE.json")
                if run_id:
                    state_path = self.out / run_id / "TRAIN_STATE.json"
                    if state_path.exists():
                        progress = _read_json(state_path)
                        with self.lock:
                            self.registry.run_update(run_id, global_step=progress["global_step"], epoch=progress["epoch"],
                                                     real_scenes_seen=progress["real_scenes_seen"], eta_seconds=progress["eta_seconds"])
                        heartbeat["training"] = progress
                with self.lock:
                    with (logs / "runtime.jsonl").open("a") as handle:
                        handle.write(json.dumps(heartbeat, allow_nan=False) + "\n")
                print(json.dumps({"task": attempt_key, "elapsed_seconds": now - start,
                                  "global_step": heartbeat.get("training", {}).get("global_step")}), flush=True)
                if state in ("ADMITTING", "STARTING", "RUNNING"):
                    unheld_observations = unheld_observations + 1 if not _lock_held(family / "FAMILY_LEASE.lock") else 0
                    if unheld_observations >= 2:
                        with self.lock:
                            self.registry.task_update(attempt_key, "REMOTE_EXIT_UNKNOWN", remote=value, host=allocation["host"])
                        raise RemoteOwnershipUnknown(f"Two observations found no owner for {key}; actual exit UNKNOWN; reconcile before retry")
                last_heartbeat = now
            time.sleep(2.)
        with self.lock:
            self.registry.task_update(attempt_key, "COMPLETE" if returncode == 0 else "FAILED", command=command, devices=selected,
                                      host=allocation["host"], log=str(log), spec=str(spec_path), logical_key=key,
                                      returncode=returncode, wall_seconds=elapsed, start=value.get("start"), end=value.get("end", stamp()))
            if returncode and run_id:
                self.registry.run_update(run_id, "ENGINEERING_FAILURE", failed_task=attempt_key, log=str(log))
        if returncode:
            if returncode < 0:
                raise RemoteOwnershipUnknown(f"Child {key} was signalled ({returncode}); reconcile surviving ranks before retry")
            raise RuntimeError(f"Task {key} failed with exit code {returncode}; inspect {log}; a resumed call creates a fresh attempt")
        if success_check is not None and not success_check():
            raise RuntimeError(f"Task {key} exited zero without valid prescribed artifacts; resume repairs them in a fresh attempt")
        return elapsed

    def _unconfirmed_launch(self, key, attempt_key, job, spec_path, error):
        family = job.parent
        with (family / "FAMILY_LEASE.lock").open("a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            if (job / "STATUS.json").exists():
                return False
            atomic_json(job / "STATUS.json", {"status": "TRANSPORT_FAILED", "spec_sha256": sha256(spec_path),
                        "child_started": False, "actual_process_exit": "UNKNOWN", "error": error, "end": stamp()})
        with self.lock:
            self.registry.task_update(attempt_key, "TRANSPORT_FAILED", logical_key=key, error=error,
                                      actual_process_exit="UNKNOWN", child_started=False, spec=str(spec_path), end=stamp())
        return True


def _verify_evaluation(path, expected=None):
    from .evaluate import verify_evaluation
    return verify_evaluation(path, expected, required=False) is not None


def evaluation_task(controller, cfg_path, checkpoint, population, destination,
                    noise=(0, 1, 2, 3, 4), target=None, panel=False,
                    representation_only=False, mode="student", functional_only=False, subset_size=None):
    cfg = load_config(cfg_path)
    count = 8192 if panel else subset_size or cfg["data"][population + "_scenes"]
    expected = {"population": population, "scene_count": count, "checkpoint_sha256": sha256(checkpoint),
                "noise_indices": list(noise), "target_version": target, "mode": mode,
                "artifact_kind": "representation" if representation_only else "functional" if functional_only else "behavior",
                "full_population": not bool(panel or subset_size),
                "feature_manifest_sha256": sha256(cfg["paths"]["feature_manifest"]),
                "teacher_bundle_sha256": sha256(cfg["paths"]["teacher_bundle"]),
                "calibration_sha256": sha256(cfg["paths"]["calibration"]),
                "targets_manifest_sha256": sha256(cfg["paths"]["targets_manifest"]) if target else None}
    if _verify_evaluation(destination, expected):
        return
    arguments = ["--config", cfg_path, "--checkpoint", checkpoint, "--population", population,
                 "--out", destination, "--noise-indices", *noise]
    if target:
        arguments += ["--target-version", target]
    if panel:
        arguments.append("--train-panel")
    if representation_only:
        arguments.append("--representation-only")
    if functional_only:
        arguments.append("--functional-only")
    if subset_size:
        arguments += ["--subset-size", subset_size]
    if mode != "student":
        arguments += ["--mode", mode]
    key = str(Path(destination).relative_to(controller.out)).replace("/", "__")
    run_id = f"{cfg['experiment']}_seed{cfg['training_seed']}"
    controller.execute(key, child_command("student_distillation.evaluate", arguments, controller.world_size),
                       allocation_run=run_id, success_check=lambda: _verify_evaluation(destination, expected))
    if not _verify_evaluation(destination, expected):
        raise ValueError("Evaluation process exited without its prescribed complete actual artifacts")


def _same_checkpoint_result(source, destination, expected_sha):
    """Reuse actual evaluation only for the identical saved checkpoint bytes."""
    source, destination = Path(source), Path(destination)
    if not _verify_evaluation(source):
        return False
    summary = json.loads((source / "SUMMARY.json").read_text())
    if summary["checkpoint_sha256"] != expected_sha:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.symlink_to(source.resolve(), target_is_directory=True)
    elif destination.resolve() != source.resolve():
        return False
    return True


def verify_validation_teacher(out, expected_teacher_sha=None):
    path = Path(out) / "VAL_TEACHER_DIAGNOSTICS.json"
    value = json.loads(path.read_text())
    if (value.get("schema"), value.get("status"), value.get("population"), value.get("scenes"), value.get("noise_indices"), value.get("versions")) != (
            "nrkd_validation_teacher_diagnostics_v1", "COMPLETE", "validation", 17475, [0, 1, 2, 3, 4], ["full", "gain1"]):
        raise ValueError("Full validation teacher diagnostics have no substantive complete evidence")
    if expected_teacher_sha and value.get("teacher_bundle_sha256") != expected_teacher_sha:
        raise ValueError("Validation diagnostic teacher differs from the frozen training teacher")
    if not value.get("artifacts"):
        raise ValueError("Validation teacher diagnostics require actual artifact references")
    for ref in value["artifacts"].values():
        artifact = Path(ref["path"])
        artifact = artifact if artifact.is_absolute() else path.parent / artifact
        if not artifact.is_file() or sha256(artifact) != ref["sha256"]:
            raise ValueError("Validation teacher diagnostic artifact SHA mismatch")
    return value


def validation_teacher_registry(common):
    override = common.get("paths", {}).get("validation_teacher_registry")
    return Path(override).resolve() if override else (
        Path(common["paths"]["teacher_bundle"]).resolve().parent.parent / "teacher_suite" / "REGISTRY.json")


def wait_for_validation_teacher(controller, common, poll_seconds=30.):
    """Wait for the independent owner; absence/failure remains recoverable work."""
    registry, out = controller.registry, controller.out
    teacher_sha = sha256(common["paths"]["teacher_bundle"])
    marker = out / "VAL_TEACHER_DIAGNOSTICS.json"
    owner_path = validation_teacher_registry(common)
    unavailable = 0
    while not marker.exists():
        owner = _read_json(owner_path)
        lock_path = owner_path.parent / "OWNER.lock"
        lease_held = lock_path.is_file() and _lock_held(lock_path)
        owner_live = None
        if owner.get("owner_host") in ("local", "localhost", socket.gethostname()):
            try:
                owner_live = _start_ticks(int(owner["owner_pid"])) == owner["owner_start_ticks"]
            except (OSError, ValueError, KeyError):
                owner_live = False
        registered = owner.get("schema") == "nrkd_local_teacher_suite_v1"
        healthy_owner = registered and lease_held and owner_live is not False
        unavailable = 0 if healthy_owner else unavailable + 1
        pending = {"status": "WAITING_VALIDATION_TEACHER", "owner_registry": str(owner_path),
                   "owner_registry_sha256": sha256(owner_path) if owner_path.is_file() else None,
                   "owner_status": owner.get("status"), "owner_pid": owner.get("owner_pid"),
                   "owner_host": owner.get("owner_host"), "owner_start_ticks": owner.get("owner_start_ticks"),
                   "owner_lease_held": lease_held, "local_pid_identity_live": owner_live,
                   "teacher_bundle_sha256": teacher_sha, "checked": stamp(), "recoverable": True}
        if not healthy_owner:
            pending["reason"] = "Independent Validation teacher owner is absent, failed, or lacks its registered live lease"
        with controller.lock:
            registry.value["status"] = "WAITING_VALIDATION_TEACHER"
            registry.value["validation_teacher_diagnostics"] = pending
            registry.save()
        atomic_json(out / "VALIDATION_TEACHER_WAIT_STATE.json", pending)
        if unavailable >= 2:
            raise RuntimeError("Validation teacher evidence remains pending and its independent owner is unavailable; "
                               f"resume that registered owner and the suite: {owner_path}")
        # A live owner may still be fitting/scoring earlier stages. Its scientific
        # completion has no arbitrary wall-time timeout and cannot be skipped.
        time.sleep(poll_seconds)
    result = verify_validation_teacher(out, teacher_sha)
    with controller.lock:
        registry.value["validation_teacher_diagnostics"] = result
        registry.value["status"] = "VALIDATION_TEACHER_COMPLETE"
        registry.save()
    atomic_json(out / "VALIDATION_TEACHER_WAIT_STATE.json", {"status": "COMPLETE", "marker": str(marker),
                "marker_sha256": sha256(marker), "owner_registry": str(owner_path), "checked": stamp()})
    return result


def _verify_training_complete(folder):
    try:
        from .select_checkpoint import select
        select(folder)  # Full budget, sealed checkpoints and all five full-Val endpoints.
        return True
    except (OSError, ValueError, KeyError, RuntimeError):
        return False


def parallel_tasks(items, function):
    failures = []
    with ThreadPoolExecutor(max_workers=8) as workers:
        futures = {workers.submit(function, item): item for item in items}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append((str(futures[future]), repr(error)))
    if failures:
        raise RuntimeError("Incomplete independent tasks (other allocated runs were preserved): " + repr(failures))


def run(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    reproduction = [sys.executable, "-B", "-m", "student_distillation.run_suite", *sys.argv[1:]]
    (out / "reproduce_commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd " +
        shlex.quote(str(Path(__file__).resolve().parents[1])) + "\n" + shlex.join(reproduction) + "\n")
    if tuple(args.experiments) != EXPERIMENTS or tuple(args.training_seeds) != SEEDS:
        raise ValueError("The formal suite must include all A/B/C/D arms and both registered training seeds")
    if args.epochs != 40 or args.global_batch_size != 64:
        raise ValueError("The formal suite cannot shorten the training budget")
    if not args.resources or args.world_size not in (None, 4):
        raise ValueError("The eight-run scheduler requires the explicit per-run four-GPU host/UUID resource manifest")
    owner = (out / "SUITE_CONTROLLER.lock").open("a+")
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    allocations = load_resources(args.resources, out)
    world = 4
    registry = Registry(out)
    atomic_json(out / "RESOURCE_INVENTORY.json", inventory())
    controller = Controller(out, registry, allocations)
    configs = {}
    common_path = args.config_dir / "common.yaml"
    common = load_config(common_path)
    if args.paths_config:
        overlay = load_config(args.paths_config)
        common = merge(common, overlay if "paths" in overlay else {"paths": overlay})
    common.setdefault("paths", {})["results_root"] = str(out)
    registry.value["independent_validation_teacher"] = {
        "status": "REGISTERED_INDEPENDENT_OWNER", "registry": str(validation_teacher_registry(common)),
        "required_marker": str(out / "VAL_TEACHER_DIAGNOSTICS.json")}
    registry.save()
    from .train import verify_formal_config, verify_qualification
    for experiment in args.experiments:
        for seed in args.training_seeds:
            run_id = f"{experiment}_seed{seed}"
            cfg = merge(common, load_config(args.config_dir / f"{experiment}.yaml"))
            cfg["training_seed"] = seed
            verify_formal_config(cfg)
            qualification = verify_qualification(cfg)
            folder = out / run_id
            folder.mkdir(exist_ok=True)
            config_path = folder / "effective_config.yaml"
            if config_path.exists() and load_config(config_path) != cfg:
                raise ValueError(f"Effective configuration changed for existing run {run_id}")
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            configs[run_id] = (cfg, config_path)
            registry.run_update(run_id, config=str(config_path), config_sha256=sha256(config_path))
    atomic_json(out / "CODE_CHECKS_PASS.json", {"status": "CODE_CHECKS_PASS", "source": qualification})
    registry.value["status"] = "CODE_CHECKS_PASS"
    registry.value["asset_status"] = "ASSET_READY"
    registry.value["target_status"] = "TARGETS_READY"
    registry.value["asset_qualification"] = qualification
    registry.save()
    # The root-owned independent validation-teacher controller may run on its
    # reserved six GPUs while these eight disjoint four-GPU allocations train.
    # Its evidence is mandatory before final diagnostics/report, never a Train label.
    def train_one(run_id):
        cfg, config_path = configs[run_id]
        folder = out / run_id
        if not _verify_training_complete(folder):
            retries = args.engineering_retries
            for attempt in range(retries + 1):
                try:
                    controller.execute(f"train__{run_id}__attempt{attempt:02d}", child_command("student_distillation.train",
                                       ["--config", config_path, "--out", folder, "--resume", args.resume], world), run_id,
                                       success_check=lambda: _verify_training_complete(folder))
                    break
                except RemoteOwnershipUnknown:
                    raise
                except RuntimeError:
                    if attempt == retries:
                        raise
        complete = json.loads((folder / "TRAIN_COMPLETE.json").read_text())
        if complete["global_step"] != 53640 or complete["real_scenes_seen"] != 3432520:
            raise ValueError(f"Incomplete formal budget for {run_id}")
        with controller.lock:
            registry.run_update(run_id, "TRAIN_COMPLETE", global_step=53640, epoch=40)
        for epoch in (8, 16, 24, 32, 40):
            evaluation_task(controller, config_path, folder / "checkpoints" / f"epoch_{epoch:03d}.pt",
                            "validation", folder / "validation" / f"epoch_{epoch:03d}")
        with controller.lock:
            registry.run_update(run_id, "FULL_VAL_COMPLETE")
    parallel_tasks(configs, train_one)
    from .select_checkpoint import lock_suite
    lock = lock_suite(out)
    for run_id in configs:
        registry.run_update(run_id, "SELECTION_LOCKED")
    # No Test process can be launched above this all-eight-run selection lock.
    def test_one(run_id):
        cfg, config_path = configs[run_id]
        folder = out / run_id
        selection = lock["runs"][run_id]
        for label, entry in (("last", selection["last"]), ("best", selection["best"])):
            destination = folder / "test" / label
            if label == "best" and selection["best"]["sha256"] == selection["last"]["sha256"]:
                if _same_checkpoint_result(folder / "test/last", destination, entry["sha256"]):
                    continue
            evaluation_task(controller, config_path, entry["checkpoint"], "test", destination)
        evaluation_task(controller, config_path, selection["best"]["checkpoint"], "validation",
                        folder / "diagnostics/validation_new_noise", noise=(5, 6, 7, 8, 9))
        with controller.lock:
            registry.run_update(run_id, "TEST_COMPLETE")
    parallel_tasks(configs, test_one)
    val_marker = out / "VAL_TEACHER_DIAGNOSTICS.json"
    if not val_marker.exists():
        command = common.get("suite", {}).get("validation_teacher_command")
        if command:
            registry.value["validation_teacher_diagnostics"] = {"status": "RUNNING", "command": command}
            registry.save()
            def validation_evidence_ready():
                if not val_marker.exists():
                    return False
                verify_validation_teacher(out, sha256(common["paths"]["teacher_bundle"]))
                return True
            controller.execute("validation_teacher_diagnostics", command, success_check=validation_evidence_ready)
    wait_for_validation_teacher(controller, common)
    # Teacher panel inference is checkpoint independent. Execute each frozen
    # target version once, then reuse that exact verified panel for all arms.
    teacher_sources = {}
    for run_id, (cfg, config_path) in configs.items():
        teacher_sources.setdefault(cfg["teacher_target"], run_id)
    def teacher_panel(target):
        run_id = teacher_sources[target]
        cfg, config_path = configs[run_id]
        evaluation_task(controller, config_path, lock["runs"][run_id]["last"]["checkpoint"], "train",
                        out / "train_panel_teachers" / target, target=target, panel=True, mode="teacher")
    parallel_tasks(teacher_sources, teacher_panel)
    def diagnostics_one(run_id):
        cfg, config_path = configs[run_id]
        folder = out / run_id
        selection = lock["runs"][run_id]
        # A separate validation-target manifest never participates in Train input
        # or normalization. Only diagnostic evaluator configurations reference it.
        diagnostic_cfg_path = config_path
        val_targets = cfg["paths"].get("validation_targets_manifest")
        if not val_targets:
            raise ValueError("Full validation representation/function diagnostics require the separate frozen diagnostic target manifest")
        diagnostic_cfg = merge(cfg, {"paths": {"targets_manifest": val_targets}})
        diagnostic_cfg_path = folder / "diagnostic_effective_config.yaml"
        diagnostic_cfg_path.write_text(yaml.safe_dump(diagnostic_cfg, sort_keys=False))
        for label, entry in (("last", selection["last"]), ("best", selection["best"])):
            for population in ("train", "validation"):
                destination = folder / "diagnostics" / f"{population}_representation_{label}"
                evaluation_task(controller, diagnostic_cfg_path if population == "validation" else config_path,
                                entry["checkpoint"], population, destination, target=cfg["teacher_target"],
                                representation_only=True)
                evaluation_task(controller, diagnostic_cfg_path if population == "validation" else config_path,
                                entry["checkpoint"], population,
                                folder / "diagnostics" / f"{population}_functional_{label}",
                                target=cfg["teacher_target"], functional_only=True, subset_size=2048, noise=(0,))
            panel_destination = folder / "diagnostics" / f"train_panel_{label}"
            evaluation_task(controller, config_path, entry["checkpoint"], "train", panel_destination,
                            target=cfg["teacher_target"], panel=True)
        # Panel teacher shares the same target version and selected scene/noise cohort.
        teacher_destination = out / "train_panel_teachers" / cfg["teacher_target"]
        for label in ("last", "best"):
            student_summary = json.loads((folder / "diagnostics" / f"train_panel_{label}" / "SUMMARY.json").read_text())
            teacher_summary = json.loads((teacher_destination / "SUMMARY.json").read_text())
            student_protocol = json.loads((folder / "diagnostics" / f"train_panel_{label}" / "PROTOCOL.json").read_text())
            teacher_protocol = json.loads((teacher_destination / "PROTOCOL.json").read_text())
            if student_summary["scenes"] != 8192 or student_summary["noise_indices"] != teacher_summary["noise_indices"] or student_protocol["tokens_sha256"] != teacher_protocol["tokens_sha256"]:
                raise ValueError("Gain capture requires matching panel and noise population")
            student = student_summary["behavior"]
            teacher = teacher_summary["behavior"]
            denominator = teacher["pdms_points"] - teacher["base_pdms_points"]
            atomic_json(folder / "diagnostics" / f"capture_{label}.json",
                        {"target_version": cfg["teacher_target"], "population": "fixed_train_panel_8192", "noise_indices": [0, 1, 2, 3, 4],
                         "student_pdms_points": student["pdms_points"], "teacher_pdms_points": teacher["pdms_points"],
                         "base_pdms_points": teacher["base_pdms_points"],
                         "tokens_sha256": student_protocol["tokens_sha256"],
                         "student_summary_sha256": sha256(folder / "diagnostics" / f"train_panel_{label}" / "SUMMARY.json"),
                         "teacher_summary_sha256": sha256(teacher_destination / "SUMMARY.json"),
                         "gain_capture": (student["pdms_points"] - teacher["base_pdms_points"]) / denominator if denominator else None})
    parallel_tasks(configs, diagnostics_one)
    from .aggregate import report
    result = report(out, require_complete=True)
    for run_id in configs:
        registry.run_update(run_id, "REPORT_COMPLETE")
        atomic_json(out / run_id / "COMPLETE.json", {"status": "REPORT_COMPLETE", "selection_lock_sha256": lock["lock_sha256"],
                                                     "required_evaluations_complete": True})
    registry.value["status"] = "REPORT_COMPLETE"
    registry.save()
    atomic_json(out / "COMPLETE.json", {"status": "REPORT_COMPLETE", **result})
    return result


def main():
    if "--execute-spec" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--execute-spec", type=Path, required=True)
        args = parser.parse_args()
        raise SystemExit(execute_spec(args.execute_spec))
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=Path("configs/student_distillation"))
    parser.add_argument("--paths-config", type=Path)
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS))
    parser.add_argument("--training-seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--complete-through", choices=("report",), default="report")
    parser.add_argument("--resources", type=Path, required=True, help="Verified all-eight-run host/four-GPU UUID allocation")
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--engineering-retries", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as error:
        args.out.mkdir(parents=True, exist_ok=True)
        atomic_json(args.out / "INCOMPLETE_OR_FAILED_TASKS.json",
                    {"status": "INCOMPLETE", "error": repr(error), "resume_command": shlex.join([sys.executable, "-m", "student_distillation.run_suite", *sys.argv[1:]])})
        raise


if __name__ == "__main__":
    main()
