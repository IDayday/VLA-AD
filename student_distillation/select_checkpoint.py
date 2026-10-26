"""Pre-registered full-Validation checkpoint selection and all-run Test lock."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .checkpointing import atomic_json, digest, sha256, read_checkpoint
from .evaluate import verify_evaluation

ENDPOINTS = (8, 16, 24, 32, 40)


def select(run_dir):
    run_dir = Path(run_dir)
    if not (run_dir / "TRAIN_COMPLETE.json").exists():
        raise ValueError("All 40 training epochs must finish before checkpoint selection")
    completed = json.loads((run_dir / "TRAIN_COMPLETE.json").read_text())
    if completed.get("status") != "TRAIN_COMPLETE" or completed.get("global_step") != 53640 or completed.get("real_scenes_seen") != 3432520 or completed.get("epoch_index") != 40:
        raise ValueError("Checkpoint selection requires the exact full training budget")
    identities = json.loads((run_dir / "ASSET_IDENTITIES.json").read_text())
    if completed.get("qualification") != identities.get("qualification"):
        raise ValueError("Formal training completion is missing its qualified code/assets identity")
    candidates = []
    for epoch in ENDPOINTS:
        folder = run_dir / "validation" / f"epoch_{epoch:03d}"
        checkpoint = run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
        checksum = sha256(checkpoint)
        state = read_checkpoint(checkpoint, identities)
        if state["progress"]["global_step"] != epoch * 1341 or state["progress"]["real_scenes_seen"] != epoch * 85813 or state["progress"]["epoch_index"] != epoch:
            raise ValueError("Checkpoint endpoint has an incomplete actual training exposure")
        summary = verify_evaluation(folder, {"population": "validation", "scene_count": 17475,
            "noise_indices": [0, 1, 2, 3, 4], "full_population": True,
            "checkpoint_sha256": checksum, "artifact_kind": "behavior", "mode": "student"})
        candidates.append({"epoch": epoch, "pdms_points": summary["behavior"]["pdms_points"],
                           "checkpoint": str(checkpoint.resolve()), "sha256": checksum,
                           "validation_summary_sha256": sha256(folder / "SUMMARY.json"),
                           "validation_complete_sha256": sha256(folder / "COMPLETE.json"),
                           "checkpoint_complete_sha256": sha256(str(checkpoint) + ".COMPLETE.json")})
    best = sorted(candidates, key=lambda value: (-value["pdms_points"], value["epoch"]))[0]
    result = {"rule": "full_scene_mean_of_five_noise_pdms_max_exact_ties_earlier_epoch",
              "candidates": candidates, "best": best, "last": candidates[-1], "base_is_not_a_candidate": True}
    result["protocol_sha256"] = identities["protocol_sha256"]
    result["teacher_bundle_sha256"] = identities["files"]["teacher_bundle"]["sha256"]
    result["train_complete_sha256"] = sha256(run_dir / "TRAIN_COMPLETE.json")
    result["qualification"] = identities["qualification"]
    path = run_dir / "checkpoints/validation_best.json"
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError("A selected checkpoint changed after selection")
    atomic_json(path, result)
    return result


def lock_suite(out):
    out = Path(out)
    registry = json.loads((out / "EXPERIMENT_REGISTRY.json").read_text())
    from .experiment_registry import EXPERIMENTS, SEEDS
    expected_runs = {f"{experiment}_seed{seed}" for experiment in EXPERIMENTS for seed in SEEDS}
    if set(registry["runs"]) != expected_runs:
        raise ValueError("The Test lock requires all eight runs")
    selections = {}
    for run in sorted(registry["runs"]):
        selections[run] = select(out / run)
    value = {"schema": 1, "test_may_start": True, "all_eight_train_and_full_validation_complete": True,
             "runs": selections, "selection_protocol_sha256": digest({"endpoints": ENDPOINTS,
               "noise_indices": [0, 1, 2, 3, 4], "population": 17475, "tie_break": "earlier_epoch"})}
    value["lock_sha256"] = digest(value)
    path = out / "CHECKPOINT_SELECTION_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError("Cannot replace an existing Test selection lock")
    atomic_json(path, value)
    return value


def verify_selection_lock(out, *, full=False):
    out = Path(out)
    value = json.loads((out / "CHECKPOINT_SELECTION_LOCK.json").read_text())
    checksum = value.get("lock_sha256")
    if digest({key: item for key, item in value.items() if key != "lock_sha256"}) != checksum:
        raise ValueError("All-run Test selection lock digest changed")
    from .experiment_registry import EXPERIMENTS, SEEDS
    if set(value.get("runs", {})) != {f"{experiment}_seed{seed}" for experiment in EXPERIMENTS for seed in SEEDS} or value.get("test_may_start") is not True or value.get("all_eight_train_and_full_validation_complete") is not True:
        raise ValueError("Test requires the complete registered eight-run selection lock")
    for run_id, entry in value["runs"].items():
        folder = out / run_id
        if sha256(folder / "TRAIN_COMPLETE.json") != entry.get("train_complete_sha256"):
            raise ValueError("Training completion changed after the all-run lock")
        if full:
            if select(folder) != entry:
                raise ValueError("Full-Validation selection changed after the lock")
        else:
            for candidate in entry["candidates"]:
                marker = folder / "validation" / f"epoch_{candidate['epoch']:03d}" / "COMPLETE.json"
                if sha256(marker) != candidate["validation_complete_sha256"]:
                    raise ValueError("Full-Validation completion evidence changed after the lock")
            for label in ("last", "best"):
                candidate = entry[label]
                if sha256(candidate["checkpoint"]) != candidate["sha256"] or sha256(candidate["checkpoint"] + ".COMPLETE.json") != candidate["checkpoint_complete_sha256"]:
                    raise ValueError("Locked checkpoint bytes or completion evidence changed")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    parser.add_argument("--suite", type=Path)
    args = parser.parse_args()
    if bool(args.run) == bool(args.suite):
        parser.error("Specify exactly one of --run or --suite")
    select(args.run) if args.run else lock_suite(args.suite)


if __name__ == "__main__":
    main()
