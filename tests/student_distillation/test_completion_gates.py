import json
from pathlib import Path
import numpy as np
import pytest
import torch

from student_distillation.aggregate import report, summarize_records
from student_distillation.checkpointing import atomic_json, digest, sha256
from student_distillation.evaluate import verify_evaluation
from student_distillation.experiment_registry import Registry, EXPERIMENTS, SEEDS
from student_distillation.losses import deterministic_backward_context, deterministic_student_forward_context
from student_distillation.noise import EvaluationNoiseStore
from student_distillation.train import QUALIFICATION_CHECKS, verify_qualification


def native_score():
    return {"score": 1., "no_at_fault_collisions": 1., "drivable_area_compliance": 1.,
            "ego_progress": 1., "time_to_collision_within_bound": 1., "comfort": 1.,
            "driving_direction_compliance": 1.}


def evaluation_fixture(folder, kind="behavior"):
    folder.mkdir(parents=True, exist_ok=True)
    protocol = {"population": "validation", "noise_indices": [0], "checkpoint_sha256": "a" * 64,
                "target_version": None if kind == "behavior" else "gain1", "mode": "student",
                "scene_count": 2, "tokens_sha256": digest(["A", "B"]), "artifact_kind": kind,
                "full_population": False}
    checksum = digest(protocol)
    rows = []
    for token in ("A", "B"):
        row = {"token": token, "log_id": token, "protocol_sha256": checksum, "noise": []}
        if kind == "behavior":
            row["noise"] = [{"noise_index": 0, "score": native_score(), "base_score": native_score(),
                             "trajectory": np.zeros((8, 3)).tolist(), "base_trajectory": np.zeros((8, 3)).tolist()}]
        elif kind == "functional":
            row["function"] = [{"noise_index": 0, **{source: [{"step_index": j, "mse": .1, "normalized_mse": .2}
                                      for j in range(5)] for source in ("teacher", "student")}}]
        rows.append(row)
    atomic_json(folder / "PROTOCOL.json", dict(protocol, protocol_sha256=checksum))
    (folder / "scene_scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = summarize_records(rows)
    summary.update({key: protocol[key] for key in ("checkpoint_sha256", "population", "noise_indices", "mode", "artifact_kind", "target_version", "full_population")})
    summary.update(scenes=2, protocol_sha256=checksum)
    atomic_json(folder / "SUMMARY.json", summary)
    reseal(folder, checksum)
    return protocol, rows


def reseal(folder, checksum):
    atomic_json(folder / "COMPLETE.json", {"status": "COMPLETE", "protocol_sha256": checksum,
        "artifacts": {name: sha256(folder / name) for name in ("PROTOCOL.json", "SUMMARY.json", "scene_scores.jsonl")}})


def test_completed_evaluation_checks_requested_identity_and_actual_records(tmp_path):
    protocol, rows = evaluation_fixture(tmp_path)
    assert verify_evaluation(tmp_path, {"checkpoint_sha256": "a" * 64})["scenes"] == 2
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        verify_evaluation(tmp_path, {"checkpoint_sha256": "b" * 64})
    rows[0]["noise"] = []
    (tmp_path / "scene_scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    reseal(tmp_path, digest(protocol))
    with pytest.raises(ValueError, match="behavior noise"):
        verify_evaluation(tmp_path)


def test_all_five_function_steps_are_required_at_both_state_sources(tmp_path):
    protocol, rows = evaluation_fixture(tmp_path, "functional")
    assert verify_evaluation(tmp_path)["functional"]["scenes"] == 2
    rows[1]["function"][0]["student"].pop()
    (tmp_path / "scene_scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    reseal(tmp_path, digest(protocol))
    with pytest.raises(ValueError, match="all five"):
        verify_evaluation(tmp_path)


def test_bare_summary_flags_cannot_produce_complete_report(tmp_path):
    registry = Registry(tmp_path)
    assert registry.value["status"] == "WAITING_ASSET_QUALIFICATION"
    atomic_json(tmp_path / "CHECKPOINT_SELECTION_LOCK.json", {})
    atomic_json(tmp_path / "VAL_TEACHER_DIAGNOSTICS.json", {"status": "COMPLETE"})
    for run_id in registry.value["runs"]:
        folder = tmp_path / run_id
        atomic_json(folder / "checkpoints/validation_best.json", {"best": {"epoch": 8}, "last": {}})
        for label in ("last", "best"):
            for relative in (f"test/{label}", f"diagnostics/train_representation_{label}",
                             f"diagnostics/validation_representation_{label}", f"diagnostics/train_panel_{label}"):
                atomic_json(folder / relative / "SUMMARY.json", {})
        atomic_json(folder / "diagnostics/validation_new_noise/COMPLETE.json", {})
    with pytest.raises(RuntimeError, match="incomplete"):
        report(tmp_path, require_complete=True)
    assert not (tmp_path / "ROUND1_REPORT.md").exists()
    assert json.loads((tmp_path / "INCOMPLETE_OR_FAILED_TASKS.json").read_text())["complete"] is False


def test_explicit_noise_requires_coverage_and_sha_no_seed_fallback(tmp_path):
    bank = np.random.default_rng(12).normal(size=(2, 10, 6, 1, 8, 3)).astype(np.float32)
    np.save(tmp_path / "noise.npy", bank)
    atomic_json(tmp_path / "tokens.json", [{"token": "A"}, {"token": "B"}])
    spec = {"path": str(tmp_path / "noise.npy"), "sha256": sha256(tmp_path / "noise.npy"),
            "dtype": "float32", "shape": list(bank.shape), "tokens_path": str(tmp_path / "tokens.json"),
            "tokens_sha256": sha256(tmp_path / "tokens.json"), "noise_indices": list(range(10)),
            "source": "synthetic test of explicit transport"}
    atomic_json(tmp_path / "manifest.json", {"schema": "native_explicit_evaluation_noise_v1", "populations": {"validation": spec}})
    store = EvaluationNoiseStore(tmp_path / "manifest.json", "validation", [5, 6, 7, 8, 9], ["A", "B"])
    assert torch.equal(store.tape("A", 9, "cpu"), torch.from_numpy(bank[0, 9]))
    with pytest.raises(ValueError, match="fallback"):
        store.tape("missing", 0, "cpu")
    spec["sha256"] = "0" * 64
    atomic_json(tmp_path / "manifest.json", {"schema": "native_explicit_evaluation_noise_v1", "populations": {"validation": spec}})
    with pytest.raises(ValueError, match="SHA"):
        EvaluationNoiseStore(tmp_path / "manifest.json", "validation", [0])


def test_qualification_rejects_h_only_recovery_and_synthetic_status(tmp_path):
    keys = ("teacher_bundle", "feature_manifest", "targets_manifest", "calibration", "checkpoint")
    for key in keys:
        (tmp_path / key).write_text(key)
    paths = {key: str(tmp_path / key) for key in keys}
    paths["qualification_receipt"] = str(tmp_path / "qualification.json")
    identities = {"code_files": {}, "files": {key: {"sha256": sha256(paths[key])} for key in keys[:-1]}}
    checks = {}
    for check in QUALIFICATION_CHECKS:
        artifact = tmp_path / (check + ".json")
        value = {"status": "PASS"}
        if check == "dit_input_gradient":
            value["student_forward_native_parity"] = {"status": "PASS", "scenes": 64, "time_indices": list(range(5)), "bitwise": True}
        atomic_json(artifact, value)
        checks[check] = {"status": "PASS", "artifact": {"path": str(artifact), "sha256": sha256(artifact)}}
    receipt = {"schema": "nrkd_code_checks_v1", "status": "CODE_CHECKS_PASS", "formal_asset_qualification": True,
               "code_sha256": {}, "identities": {key + "_sha256": sha256(paths[key]) for key in keys}, "checks": checks}
    atomic_json(paths["qualification_receipt"], receipt)
    with pytest.raises(ValueError, match="active teacher-state"):
        verify_qualification({"paths": paths}, identities)
    receipt["status"] = "ENGINEERING_CHECKS_PASS_SYNTHETIC_BASIS"
    atomic_json(paths["qualification_receipt"], receipt)
    with pytest.raises(ValueError, match="Synthetic"):
        verify_qualification({"paths": paths}, identities)


def test_deterministic_metadata_scopes_restore_flags_even_after_failure():
    enabled, warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(False, warn_only=True)
        with deterministic_student_forward_context():
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        with pytest.raises(RuntimeError, match="fixture"):
            with deterministic_backward_context():
                raise RuntimeError("fixture")
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn)
