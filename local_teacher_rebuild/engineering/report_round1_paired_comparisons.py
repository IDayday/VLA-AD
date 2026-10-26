"""Post-completion paired cross-arm log bootstrap; never edits frozen core code.

All models share the same saved log draws for a population. A draw retains all
scenes in each sampled log, so unequal log sizes do not change the prescribed
equal-scene objective. S0--S4 are averaged within a scene, not treated as seeds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

import numpy as np

EXPERIMENTS = ("H_full", "H_gain1", "H_mu_teacher", "H_mu_student")
SEEDS = (20261027, 20261028)
CONTRASTS = (("B-A", "H_full", "H_gain1"),
             ("C-B", "H_gain1", "H_mu_teacher"),
             ("D-C", "H_mu_teacher", "H_mu_student"))
NOISE = [0, 1, 2, 3, 4]
BEGIN = "<!-- ROUND1_PAIRED_LOG_COMPARISONS_BEGIN -->"
END = "<!-- ROUND1_PAIRED_LOG_COMPARISONS_END -->"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def cohort(records, expected_count=None, bank_sha=None):
    """Join by token and retain per-token log/noise identities for pairing."""
    by_token = {}
    for record in records:
        token = record["token"]
        if token in by_token:
            raise ValueError("Duplicate scene token in a paired cohort")
        noises = record.get("noise", [])
        if [entry["noise_index"] for entry in noises] != NOISE:
            raise ValueError("Paired cohorts require the same complete S0--S4 repetitions")
        if bank_sha is not None and any(entry.get("noise_bank_sha256") != bank_sha for entry in noises):
            raise ValueError("Scene/noise record differs from its frozen explicit bank identity")
        scores = np.asarray([entry["score"]["score"] for entry in noises], dtype=np.float64)
        if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1 + 1e-8):
            raise ValueError("Invalid native normalized PDM scores")
        by_token[token] = {"log_id": str(record["log_id"]),
                           "noise_identity": [(entry["noise_index"], entry.get("noise_bank_sha256")) for entry in noises],
                           "mean_points": float(scores.mean() * 100.0)}
    if not by_token or (expected_count is not None and len(by_token) != expected_count):
        raise ValueError("Paired population does not have its prescribed full scene count")
    return by_token


def require_same_cohort(left, right):
    if set(left) != set(right):
        raise ValueError("Paired models have different scene-token cohorts")
    for token in left:
        if left[token]["log_id"] != right[token]["log_id"]:
            raise ValueError("Paired models disagree on a scene's log identity")
        if left[token]["noise_identity"] != right[token]["noise_identity"]:
            raise ValueError("Paired models have different scene/noise identities")


def make_draws(reference, replicates=2000, seed=20261027):
    logs = sorted({record["log_id"] for record in reference.values()})
    if len(logs) < 2:
        raise ValueError("Log bootstrap requires at least two actual log clusters")
    random = np.random.default_rng(seed)
    return logs, random.integers(0, len(logs), size=(replicates, len(logs)), dtype=np.int64)


def paired_bootstrap(left, right, ordered_logs, draws):
    """Compute right-left with the same cluster draws and equal scene weights."""
    require_same_cohort(left, right)
    if sorted({row["log_id"] for row in left.values()}) != list(ordered_logs):
        raise ValueError("Saved bootstrap draws belong to different log clusters")
    positions = {log: index for index, log in enumerate(ordered_logs)}
    totals = np.zeros((len(ordered_logs), 3), dtype=np.float64)
    for token in sorted(left):
        position = positions[left[token]["log_id"]]
        totals[position] += [left[token]["mean_points"], right[token]["mean_points"], 1.]
    draws = np.asarray(draws)
    if draws.ndim != 2 or draws.shape[1] != len(ordered_logs) or draws.dtype.kind not in "iu" or draws.min() < 0 or draws.max() >= len(ordered_logs):
        raise ValueError("Malformed shared log bootstrap draws")
    sampled = totals[draws].sum(axis=1)
    bootstrap_left = sampled[:, 0] / sampled[:, 2]
    bootstrap_right = sampled[:, 1] / sampled[:, 2]
    paired_delta = bootstrap_right - bootstrap_left
    population = totals.sum(axis=0)
    return {"scenes": len(left), "logs": len(ordered_logs),
            "left_pdms_points": float(population[0] / population[2]),
            "right_pdms_points": float(population[1] / population[2]),
            "right_minus_left_points": float((population[1] - population[0]) / population[2]),
            "paired_log_bootstrap_95ci_points": np.quantile(paired_delta, [.025, .975]).tolist(),
            "left_log_bootstrap_95ci_points": np.quantile(bootstrap_left, [.025, .975]).tolist(),
            "right_log_bootstrap_95ci_points": np.quantile(bootstrap_right, [.025, .975]).tolist(),
            "replicates": len(draws), "resampling_unit": "log", "scene_objective_retained": True,
            "both_models_share_identical_draws": True}, paired_delta


def self_test(out=None):
    def records(delta_by_token):
        return [{"token": f"S{i}", "log_id": "large" if i < 9 else "small",
                 "noise": [{"noise_index": j, "noise_bank_sha256": "f" * 64,
                            "score": {"score": .5 + delta_by_token(i)}} for j in NOISE]} for i in range(10)]
    left_records = records(lambda i: 0.)
    right_records = records(lambda i: .1 if i < 9 else -.5)
    left, right = cohort(left_records, 10), cohort(right_records, 10)
    logs, draws = make_draws(left)
    result, samples = paired_bootstrap(left, right, logs, draws)
    assert abs(result["right_minus_left_points"] - 4.) < 1e-12
    assert not np.isclose(result["right_minus_left_points"], -20.)  # wrong equal-log estimator
    assert np.allclose(result["paired_log_bootstrap_95ci_points"], [-50., 10.])
    reverse, reversed_samples = paired_bootstrap(right, left, logs, draws)
    assert np.allclose(samples, -reversed_samples, atol=1e-12, rtol=0)
    identical, _ = paired_bootstrap(left, left, logs, draws)
    assert identical["paired_log_bootstrap_95ci_points"] == [0., 0.]
    failures = 0
    for modification in ("missing_scene", "wrong_log", "wrong_noise"):
        changed = json.loads(json.dumps(right_records))
        if modification == "missing_scene":
            changed.pop()
        elif modification == "wrong_log":
            changed[0]["log_id"] = "different"
        else:
            changed[0]["noise"][0]["noise_bank_sha256"] = "e" * 64
        try:
            paired_bootstrap(left, cohort(changed), logs, draws)
        except ValueError:
            failures += 1
    assert failures == 3
    value = {"status": "PASS_CPU_PAIRED_BOOTSTRAP_SELFTEST", "engineering_only": True,
             "known_scene_weighted_delta_points": result["right_minus_left_points"],
             "known_95ci_points": result["paired_log_bootstrap_95ci_points"],
             "equal_log_weighting_would_be_wrong_points": -20., "shared_draw_reverse_contrast_checked": True,
             "identical_models_zero_interval_checked": True, "cohort_log_noise_mismatches_rejected": failures,
             "replicates": 2000, "source_sha256": sha(__file__)}
    if out:
        atomic(out, value)
    return value


def code_root_for(args):
    if args.code_root:
        return args.code_root.resolve()
    review = args.out / "qualification/CODE_FREEZE.json"
    if not review.is_file():
        raise ValueError("Supply --code-root or retain the reviewed implementation freeze receipt")
    return Path(read(review)["code_root"]).resolve()


def append_report(path, text):
    before = path.read_text()
    if before.count(BEGIN) != before.count(END) or before.count(BEGIN) > 1:
        raise ValueError("Existing paired report markers are incomplete or ambiguous")
    if BEGIN in before:
        start, rest = before.split(BEGIN, 1)
        _, end = rest.split(END, 1)
        after = start.rstrip() + "\n\n" + text + end
    else:
        after = before.rstrip() + "\n\n" + text + "\n"
    temporary = path.with_name(path.name + ".paired." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(after)
    temporary.replace(path)


def run(args):
    args.out = args.out.resolve()
    report = (args.round1_report or args.out / "ROUND1_REPORT.md").resolve()
    if not report.is_file():
        raise ValueError("Actual completed ROUND1_REPORT.md must exist before this postprocessor")
    if read(args.out / "COMPLETE.json").get("status") != "REPORT_COMPLETE":
        raise ValueError("Cross-arm final reporting requires actual REPORT_COMPLETE")
    registry = read(args.out / "EXPERIMENT_REGISTRY.json")
    expected_runs = {f"{experiment}_seed{seed}" for experiment in EXPERIMENTS for seed in SEEDS}
    if registry.get("status") != "REPORT_COMPLETE" or set(registry.get("runs", {})) != expected_runs:
        raise ValueError("All eight registered runs must actually be complete")
    code_root = code_root_for(args)
    freeze_path = args.out / "protocol/CODE_FREEZE.json"
    if not freeze_path.is_file():
        freeze_path = args.out / "qualification/CODE_FREEZE.json"
    reviewed = read(freeze_path)
    if reviewed.get("status") != "IMPLEMENTATION_FROZEN":
        raise ValueError("An authoritative implementation freeze is required")
    core_before = {str(path.relative_to(code_root)): sha(path) for path in sorted((code_root / "student_distillation").glob("*.py"))}
    if core_before != reviewed["code_files"]:
        raise ValueError("Frozen implementation differs from the reviewed code map")
    sys.path.insert(0, str(code_root))
    from student_distillation.select_checkpoint import verify_selection_lock
    from student_distillation.evaluate import verify_evaluation
    selection = verify_selection_lock(args.out, full=True)
    for run_id in sorted(expected_runs):
        marker = read(args.out / run_id / "COMPLETE.json")
        if marker.get("status") != "REPORT_COMPLETE" or marker.get("selection_lock_sha256") != selection["lock_sha256"]:
            raise ValueError("A run is missing its completed report/locked checkpoint evidence")
    destination = args.out / "paired_comparisons"
    destination.mkdir(exist_ok=True)
    populations = ["test"] + (["validation"] if args.include_validation else [])
    results, sources, draw_proofs = [], {}, {}
    for population in populations:
        count = 12146 if population == "test" else 17475
        prepared, reference, common_protocol = {}, None, None
        for label in ("best", "last"):
            for run_id in sorted(expected_runs):
                locked = selection["runs"][run_id][label]
                folder = (args.out / run_id / "test" / label if population == "test" else
                          args.out / run_id / "validation" / f"epoch_{locked['epoch']:03d}")
                summary = verify_evaluation(folder, {"population": population, "scene_count": count,
                    "noise_indices": NOISE, "mode": "student", "artifact_kind": "behavior",
                    "full_population": True, "checkpoint_sha256": locked["sha256"]})
                protocol = read(folder / "PROTOCOL.json")
                common = {key: protocol[key] for key in ("population", "noise_indices", "feature_manifest_sha256",
                          "teacher_bundle_sha256", "calibration_sha256", "noise_identity", "scoring", "tokens_sha256")}
                if common_protocol is None:
                    common_protocol = common
                elif common != common_protocol:
                    raise ValueError("Cross-arm common population/feature/teacher/noise/scoring identities differ")
                with (folder / "scene_scores.jsonl").open() as stream:
                    data = cohort((json.loads(line) for line in stream if line.strip()), count,
                                  protocol["noise_identity"]["sha256"])
                if reference is None:
                    reference = data
                else:
                    require_same_cohort(reference, data)
                prepared[(run_id, label)] = data
                sources[f"{population}/{run_id}/{label}"] = {
                    "checkpoint_sha256": locked["sha256"], "epoch": locked["epoch"],
                    "scene_scores": str(folder / "scene_scores.jsonl"), "scene_scores_sha256": sha(folder / "scene_scores.jsonl"),
                    "complete_sha256": sha(folder / "COMPLETE.json"), "summary_sha256": sha(folder / "SUMMARY.json"),
                    "verified_pdms_points": summary["behavior"]["pdms_points"]}
        logs, draws = make_draws(reference, 2000, args.bootstrap_seed)
        draw_path = destination / f"shared_log_draws_{population}.npy"
        draw_temp = draw_path.with_name(draw_path.name + "." + uuid.uuid4().hex + ".tmp")
        with draw_temp.open("wb") as stream:
            np.save(stream, draws, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        draw_temp.replace(draw_path)
        draw_proofs[population] = {"path": str(draw_path), "sha256": sha(draw_path), "ordered_logs": logs,
                                  "ordered_logs_sha256": canonical(logs), "seed": args.bootstrap_seed,
                                  "shape": list(draws.shape), "same_draws_for_all_arms_labels_and_training_seeds": True}
        for label in ("best", "last"):
            for seed in SEEDS:
                for contrast, left_arm, right_arm in CONTRASTS:
                    left_run, right_run = f"{left_arm}_seed{seed}", f"{right_arm}_seed{seed}"
                    result, _ = paired_bootstrap(prepared[(left_run, label)], prepared[(right_run, label)], logs, draws)
                    left_entry, right_entry = selection["runs"][left_run][label], selection["runs"][right_run][label]
                    results.append({"population": population, "checkpoint": label, "training_seed": seed,
                                    "contrast": contrast, "left_run": left_run, "right_run": right_run,
                                    "left_epoch": left_entry["epoch"], "right_epoch": right_entry["epoch"],
                                    "left_checkpoint_sha256": left_entry["sha256"], "right_checkpoint_sha256": right_entry["sha256"],
                                    "shared_log_draws_sha256": draw_proofs[population]["sha256"], **result})
    if len(results) != 12 * len(populations):
        raise RuntimeError("A required cross-arm/seed/checkpoint comparison is missing")
    core_after = {str(path.relative_to(code_root)): sha(path) for path in sorted((code_root / "student_distillation").glob("*.py"))}
    if core_before != core_after:
        raise RuntimeError("Frozen core implementation changed during paired reporting")
    payload = {"schema": "nrkd_paired_log_comparisons_v1", "status": "COMPLETE", "results": results,
               "selection_lock_sha256": selection["lock_sha256"],
               "selection_lock_file_sha256": sha(args.out / "CHECKPOINT_SELECTION_LOCK.json"),
               "source_sha256": sha(__file__), "frozen_code_files": core_before,
               "implementation_freeze": {"path": str(freeze_path), "sha256": sha(freeze_path)},
               "populations": populations, "training_seeds": list(SEEDS), "noise_indices": NOISE,
               "training_seeds_are_not_noise_repetitions": True, "bootstrap_draws": draw_proofs,
               "sources": sources, "confidence_level": .95, "interval_kind": "pointwise paired log-cluster bootstrap percentile",
               "multiple_comparisons_adjusted": False, "postselection_validation_intervals_exploratory": args.include_validation}
    atomic(destination / "PAIRED_COMPARISONS.json", payload)
    fields = ["population", "checkpoint", "training_seed", "contrast", "left_epoch", "right_epoch", "scenes", "logs",
              "left_pdms_points", "right_pdms_points", "right_minus_left_points", "ci_low_points", "ci_high_points", "replicates"]
    csv_path = destination / "PAIRED_COMPARISONS.csv"
    csv_temp = csv_path.with_name(csv_path.name + "." + uuid.uuid4().hex + ".tmp")
    with csv_temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for result in results:
            row = {key: result[key] for key in fields if key in result}
            row["ci_low_points"], row["ci_high_points"] = result["paired_log_bootstrap_95ci_points"]
            writer.writerow(row)
    csv_temp.replace(csv_path)
    lines = [BEGIN, "## Paired cross-arm comparisons", "",
             "Each row compares two models within one training seed on identical complete scenes and S0–S4. The five noise repetitions are averaged within each scene. Intervals resample whole logs 2,000 times with identical saved draws for both models, and divide resampled scene totals by resampled scene counts; logs do not receive equal weight regardless of size.", "",
             "These are pointwise 95% bootstrap percentile intervals, without adjustment across contrasts. The two training seeds are listed separately; S0–S4 are not training seeds. Positive differences favor the right-hand arm.", "",
             "| Population | Checkpoint | Training seed | Contrast | Difference (PDMS points) | Paired log 95% CI |",
             "|---|---|---:|---|---:|---|"]
    for result in results:
        low, high = result["paired_log_bootstrap_95ci_points"]
        lines.append(f"| {result['population']} | {result['checkpoint']} | {result['training_seed']} | {result['contrast']} | {result['right_minus_left_points']:+.6f} | [{low:+.6f}, {high:+.6f}] |")
    if args.include_validation:
        lines += ["", "Validation has already informed checkpoint selection. Its best-checkpoint intervals are descriptive post-selection diagnostics, not fresh blind confirmation."]
    lines += ["", "Exact source artifacts, checkpoint SHAs, log lists and shared bootstrap draws are retained in `paired_comparisons/PAIRED_COMPARISONS.json` and the companion CSV/NPY files.", END]
    append_report(report, "\n".join(lines))
    atomic(destination / "COMPLETE.json", {"status": "COMPLETE", "comparisons": len(results),
        "artifacts": {name: sha(destination / name) for name in
                      ("PAIRED_COMPARISONS.json", "PAIRED_COMPARISONS.csv", *[f"shared_log_draws_{p}.npy" for p in populations])},
        "round1_report": str(report), "round1_report_sha256": sha(report), "frozen_core_unchanged": True})
    return {"status": "COMPLETE", "comparisons": len(results), "report": str(report),
            "summary": str(destination / "PAIRED_COMPARISONS.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--round1-report", type=Path)
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--include-validation", action="store_true")
    parser.add_argument("--bootstrap-seed", type=int, default=20261027)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-out", type=Path)
    args = parser.parse_args()
    if args.self_test:
        result = self_test(args.self_test_out)
    else:
        if args.out is None:
            parser.error("--out is required after the full suite has completed")
        result = run(args)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
