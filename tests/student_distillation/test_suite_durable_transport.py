"""Real CPU children exercise the scheduler transport, without GPU admission."""
import fcntl
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from student_distillation import run_suite as suite
from student_distillation.checkpointing import atomic_json, sha256
from student_distillation.experiment_registry import Registry


def fixture(tmp_path, monkeypatch):
    registry = Registry(tmp_path)
    allocation = {"host": socket.gethostname(), "ssh_target": "local", "python": sys.executable,
                  "workdir": str(tmp_path), "gpu_uuids": ["GPU-CPU-TEST"], "_task": "CPU_TRANSPORT_TEST",
                  "_live_policy_path": str(tmp_path / "unused_policy"), "_resource_status": "READY_FOR_FORMAL_DISPATCH"}
    controller = suite.Controller(tmp_path, registry, {"cpu": allocation})
    original_popen = subprocess.Popen
    original_sleep = time.sleep
    launches = []
    monkeypatch.setattr(suite, "_admit_spec", lambda spec: {})
    monkeypatch.setattr(suite, "inventory", lambda: {"cpu_only_test": True})
    monkeypatch.setattr(suite.time, "sleep", lambda seconds: original_sleep(min(seconds, .01)))

    def popen(command, **kwargs):
        if "--execute-spec" in command:
            launches.append(command)
            code = suite.execute_spec(Path(command[-1]))
            return SimpleNamespace(poll=lambda: code)
        return original_popen(command, **kwargs)
    monkeypatch.setattr(suite.subprocess, "Popen", popen)
    return controller, launches, popen, original_sleep


def attempts(tmp_path, key="job"):
    return sorted((tmp_path / "jobs" / key).glob("attempt_?????"))


def test_transport_failure_resume_versions_attempt_and_preserves_unknown_exit(tmp_path, monkeypatch):
    controller, launches, popen, _ = fixture(tmp_path, monkeypatch)
    marker = tmp_path / "real_cpu_artifact"
    command = [sys.executable, "-c", "from pathlib import Path; Path('real_cpu_artifact').write_text('complete')"]
    monkeypatch.setattr(suite.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(poll=lambda: 255))
    with pytest.raises(RuntimeError, match="transport/startup"):
        controller.execute("job", command, success_check=marker.is_file)
    failed = attempts(tmp_path)[0]
    status_bytes, spec_bytes = (failed / "STATUS.json").read_bytes(), (failed / "SPEC.json").read_bytes()
    status = json.loads(status_bytes)
    assert status["status"] == "TRANSPORT_FAILED" and status["actual_process_exit"] == "UNKNOWN"
    assert status["child_started"] is False and not marker.exists()
    monkeypatch.setattr(suite.subprocess, "Popen", popen)
    controller.execute("job", command, success_check=marker.is_file)
    assert marker.read_text() == "complete" and len(attempts(tmp_path)) == 2
    assert (failed / "STATUS.json").read_bytes() == status_bytes
    assert (failed / "SPEC.json").read_bytes() == spec_bytes
    history = json.loads((failed.parent / "ATTEMPT_HISTORY.json").read_text())
    assert [row["attempt"] for row in history["attempts"]] == [0, 1]
    assert len(controller.registry.value["tasks"]) == 2
    # The old transport-delayed wrapper is now superseded and cannot launch.
    assert suite.execute_spec(failed / "SPEC.json") == 75
    assert (failed / "STATUS.json").read_bytes() == status_bytes
    controller.execute("job", command, success_check=marker.is_file)
    assert len(attempts(tmp_path)) == 2 and len(launches) == 1


def test_startup_guard_failure_is_terminal_before_child_and_retry_is_fresh(tmp_path, monkeypatch):
    controller, launches, popen, _ = fixture(tmp_path, monkeypatch)
    command = [sys.executable, "-c", "print('real CPU success')"]
    def reject(spec):
        raise ValueError("live resource grant withdrawn")
    monkeypatch.setattr(suite, "_admit_spec", reject)
    with pytest.raises(RuntimeError, match="exit code 1"):
        controller.execute("job", command)
    old = attempts(tmp_path)[0]
    status_bytes = (old / "STATUS.json").read_bytes()
    value = json.loads(status_bytes)
    assert value["status"] == "STARTUP_FAILED" and value["child_started"] is False
    assert "grant withdrawn" in value["traceback"]
    monkeypatch.setattr(suite, "_admit_spec", lambda spec: {})
    controller.execute("job", command)
    assert len(attempts(tmp_path)) == 2 and len(launches) == 2
    assert (old / "STATUS.json").read_bytes() == status_bytes


def test_positive_failure_and_missing_success_artifact_cannot_poison_resume(tmp_path, monkeypatch):
    controller, launches, _, _ = fixture(tmp_path, monkeypatch)
    command = [sys.executable, "-c", "from pathlib import Path; import sys; "
               "sys.exit(3) if not Path('allow').exists() else None; "
               "Path('artifact').write_text('ok') if Path('write').exists() else None"]
    gate = (tmp_path / "artifact").is_file
    with pytest.raises(RuntimeError, match="exit code 3"):
        controller.execute("job", command, success_check=gate)
    first = (attempts(tmp_path)[0] / "STATUS.json").read_bytes()
    (tmp_path / "allow").touch()
    with pytest.raises(RuntimeError, match="zero without valid"):
        controller.execute("job", command, success_check=gate)
    assert len(attempts(tmp_path)) == 2
    (tmp_path / "write").touch()
    controller.execute("job", command, success_check=gate)
    assert len(attempts(tmp_path)) == 3 and len(launches) == 3
    assert (attempts(tmp_path)[0] / "STATUS.json").read_bytes() == first
    spec = json.loads((attempts(tmp_path)[-1] / "SPEC.json").read_text())
    assert spec["environment"]["OPENBLAS_CORETYPE"] == "Sandybridge"
    assert spec["environment"]["ZT_AUTO_PROXY"] == "0"


def test_live_family_lease_reattaches_and_unknown_started_child_never_duplicates(tmp_path, monkeypatch):
    controller, launches, _, original_sleep = fixture(tmp_path, monkeypatch)
    command = [sys.executable, "-c", "pass"]
    controller.execute("job", command)
    folder = attempts(tmp_path)[0]
    previous = json.loads((folder / "STATUS.json").read_text())
    atomic_json(folder / "STATUS.json", {**previous, "status": "RUNNING", "child_started": True})
    lease = (folder.parent / "FAMILY_LEASE.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def finish():
        original_sleep(.03)
        atomic_json(folder / "STATUS.json", previous)
        fcntl.flock(lease, fcntl.LOCK_UN)
        lease.close()
    thread = threading.Thread(target=finish)
    thread.start()
    controller.execute("job", command, success_check=lambda: True)
    thread.join()
    assert len(attempts(tmp_path)) == 1 and len(launches) == 1
    for state in ({"status": "RUNNING", "child_started": True}, {"status": "EXITED", "returncode": -9}):
        atomic_json(folder / "STATUS.json", {**previous, **state})
        with pytest.raises(suite.RemoteOwnershipUnknown, match="reconcile"):
            controller.execute("job", command)
    assert len(attempts(tmp_path)) == 1 and len(launches) == 1


def validation_fixture(tmp_path, monkeypatch):
    controller, _, _, _ = fixture(tmp_path, monkeypatch)
    teacher = tmp_path / "frozen_teacher/BUNDLE.json"
    atomic_json(teacher, {"test_teacher": True})
    common = {"paths": {"teacher_bundle": str(teacher)}}
    owner_path = suite.validation_teacher_registry(common)
    atomic_json(owner_path, {"schema": "nrkd_local_teacher_suite_v1", "status": "SCORING_VALIDATION",
                "owner_host": socket.gethostname(), "owner_pid": __import__('os').getpid(),
                "owner_start_ticks": suite._start_ticks(__import__('os').getpid())})
    return controller, common, owner_path, teacher


def validation_marker(tmp_path, teacher):
    artifact = tmp_path / "actual_validation_evidence.json"
    atomic_json(artifact, {"strict_test_fixture": True})
    atomic_json(tmp_path / "VAL_TEACHER_DIAGNOSTICS.json", {
        "schema": "nrkd_validation_teacher_diagnostics_v1", "status": "COMPLETE", "population": "validation",
        "scenes": 17475, "noise_indices": [0, 1, 2, 3, 4], "versions": ["full", "gain1"],
        "teacher_bundle_sha256": sha256(teacher), "artifacts": {"actual": {"path": str(artifact), "sha256": sha256(artifact)}}})


def test_validation_wait_tracks_live_independent_owner_then_verifies_evidence(tmp_path, monkeypatch):
    controller, common, owner_path, teacher = validation_fixture(tmp_path, monkeypatch)
    lease = (owner_path.parent / "OWNER.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    observed = []
    def complete_after_poll(seconds):
        observed.append(json.loads((tmp_path / "VALIDATION_TEACHER_WAIT_STATE.json").read_text()))
        validation_marker(tmp_path, teacher)
    monkeypatch.setattr(suite.time, "sleep", complete_after_poll)
    try:
        result = suite.wait_for_validation_teacher(controller, common, poll_seconds=.001)
    finally:
        lease.close()
    assert result["status"] == "COMPLETE" and len(observed) == 1
    assert observed[0]["owner_lease_held"] is True
    assert observed[0]["local_pid_identity_live"] is True
    assert controller.registry.value["status"] == "VALIDATION_TEACHER_COMPLETE"


def test_validation_absent_owner_is_recoverable_pending_not_complete(tmp_path, monkeypatch):
    controller, common, owner_path, teacher = validation_fixture(tmp_path, monkeypatch)
    atomic_json(owner_path, {**json.loads(owner_path.read_text()), "status": "ENGINEERING_FAILURE"})
    with pytest.raises(RuntimeError, match="independent owner is unavailable"):
        suite.wait_for_validation_teacher(controller, common, poll_seconds=.001)
    value = json.loads((tmp_path / "VALIDATION_TEACHER_WAIT_STATE.json").read_text())
    assert value["recoverable"] is True and value["owner_status"] == "ENGINEERING_FAILURE"
    assert controller.registry.value["status"] == "WAITING_VALIDATION_TEACHER"
    assert not (tmp_path / "COMPLETE.json").exists()
    validation_marker(tmp_path, teacher)
    assert suite.wait_for_validation_teacher(controller, common)["status"] == "COMPLETE"
    (tmp_path / "actual_validation_evidence.json").write_text('changed')
    with pytest.raises(ValueError, match="SHA mismatch"):
        suite.wait_for_validation_teacher(controller, common)
