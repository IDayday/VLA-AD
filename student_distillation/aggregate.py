"""Paired scientific summaries, log bootstrap, complete-run tables and figures."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from .checkpointing import atomic_json


def grouped_bootstrap(records, replicates=2000, seed=20261027):
    """Resample logs with replacement; scene weighting is retained within logs."""
    by_log = defaultdict(lambda: [0., 0., 0.])
    for record in records:
        if not record.get("noise"):
            continue
        student = np.mean([value["score"]["score"] for value in record["noise"]]) * 100
        base = np.mean([value["base_score"]["score"] for value in record["noise"]]) * 100
        values = by_log[record["log_id"]]
        values[0] += student
        values[1] += base
        values[2] += 1
    if not by_log:
        return None
    values = np.asarray([by_log[key] for key in sorted(by_log)], dtype=np.float64)
    random = np.random.default_rng(seed)
    sampled = random.integers(len(values), size=(replicates, len(values)))
    sums = values[sampled].sum(axis=1)
    student = sums[:, 0] / sums[:, 2]
    base = sums[:, 1] / sums[:, 2]
    return {"unit": "log", "replicates": replicates, "seed": seed,
            "student_pdms_points_95ci": np.quantile(student, [.025, .975]).tolist(),
            "base_pdms_points_95ci": np.quantile(base, [.025, .975]).tolist(),
            "paired_delta_points_95ci": np.quantile(student - base, [.025, .975]).tolist()}


def _group_summary(rows):
    if not rows:
        return {"scenes": 0}
    delta = np.asarray([row["delta"] for row in rows])
    return {"scenes": len(rows), "pdms_points": float(np.mean([row["student"] for row in rows])),
            "base_pdms_points": float(np.mean([row["base"] for row in rows])),
            "delta_points": float(delta.mean()), "improved": int((delta > 0).sum()),
            "tied": int((delta == 0).sum()), "degraded": int((delta < 0).sum()),
            "positive_gain_sum_points": float(delta[delta > 0].sum()),
            "negative_loss_sum_points": float(-delta[delta < 0].sum())}


def summarize_records(records, bootstrap_replicates=2000):
    scenes = []
    components = defaultdict(list)
    base_components = defaultdict(list)
    events = {"new_nc_penalty": 0, "new_dac_penalty": 0, "new_ttc_penalty": 0,
              "new_zero_score": 0, "paired_base_gt90_student_zero": 0, "scene_noise_pairs": 0}
    paired_severe = []
    groups = {"base_bucket": defaultdict(list), "source_choice": defaultdict(list),
              "target_keep_repair": defaultdict(list), "log": defaultdict(list)}
    representations = []
    functional = []
    for record in records:
        if "representation" in record:
            representations.append(record["representation"])
        if record.get("function"):
            functional.append(record["function"])
        noise = record.get("noise", [])
        if not noise:
            continue
        student = float(np.mean([value["score"]["score"] for value in noise]) * 100)
        base = float(np.mean([value["base_score"]["score"] for value in noise]) * 100)
        row = {"token": record["token"], "log_id": record["log_id"], "student": student,
               "base": base, "delta": student - base}
        scenes.append(row)
        bucket = "lt70" if base < 70 else "70_to_90" if base <= 90 else "gt90"
        groups["base_bucket"][bucket].append(row)
        groups["source_choice"][str(record.get("original_choice", "unknown"))].append(row)
        zero = record.get("target_is_zero")
        groups["target_keep_repair"]["unknown" if zero is None else "keep" if zero else "repair"].append(row)
        groups["log"][str(record["log_id"])].append(row)
        keys = set(noise[0]["score"])
        for key in keys:
            components[key].append(float(np.mean([value["score"][key] for value in noise])) * 100)
            base_components[key].append(float(np.mean([value["base_score"][key] for value in noise])) * 100)
        for value in noise:
            score, reference = value["score"], value["base_score"]
            events["scene_noise_pairs"] += 1
            events["new_nc_penalty"] += int(score["no_at_fault_collisions"] < reference["no_at_fault_collisions"])
            events["new_dac_penalty"] += int(score["drivable_area_compliance"] < reference["drivable_area_compliance"])
            events["new_ttc_penalty"] += int(score["time_to_collision_within_bound"] < reference["time_to_collision_within_bound"])
            events["new_zero_score"] += int(score["score"] == 0 and reference["score"] > 0)
            severe = reference["score"] > .9 and score["score"] == 0
            events["paired_base_gt90_student_zero"] += int(severe)
            if severe:
                paired_severe.append({"token": record["token"], "log_id": record["log_id"],
                                      "noise_index": value["noise_index"], "base_pdms_points": reference["score"] * 100})
    result = {"behavior": _group_summary(scenes), "components_points": {key: float(np.mean(value)) for key, value in components.items()},
              "base_components_points": {key: float(np.mean(value)) for key, value in base_components.items()},
              "events": events, "paired_severe_events": paired_severe,
              "groups": {key: {name: _group_summary(rows) for name, rows in values.items()} for key, values in groups.items()},
              "most_harmed": sorted(scenes, key=lambda row: (row["delta"], row["token"]))[:100],
              "most_helped": sorted(scenes, key=lambda row: (-row["delta"], row["token"]))[:100],
              "bootstrap": grouped_bootstrap(records, bootstrap_replicates)}
    if representations:
        def representation_group(values):
            if not values:
                return {"scenes": 0}
            numerical = ("h_mse", "h_normalized", "target_rms", "predicted_rms", "direction_cosine", "amplitude_ratio")
            output = {"scenes": len(values)}
            for key in numerical:
                valid = [row[key] for row in values if row.get(key) is not None]
                output[key] = float(np.mean(valid)) if valid else None
            output["cap_trigger_fraction"] = float(np.mean([row["cap_triggered"] for row in values]))
            return output
        result["representation"] = {"all": representation_group(representations),
                                    "keep": representation_group([row for row in representations if row["target_is_zero"]]),
                                    "repair": representation_group([row for row in representations if not row["target_is_zero"]]),
                                    "sources": {str(source): representation_group([row for row in representations if row["original_choice"] == source])
                                                for source in {row["original_choice"] for row in representations}}}
    if functional:
        result["functional"] = {"scenes": len(functional), "time_indices": list(range(5)),
                                "state_sources": ["teacher", "student"],
                                "noise_indices": [entry["noise_index"] for entry in functional[0]]}
        for source in ("teacher", "student"):
            result["functional"][source] = []
            for j in range(5):
                values = [noise[source][j] for scene in functional for noise in scene]
                result["functional"][source].append({"step_index": j,
                    "mse": float(np.mean([value["mse"] for value in values])),
                    "normalized_mse": float(np.mean([value["normalized_mse"] for value in values])),
                    "scene_noise_observations": len(values)})
    return result


def read_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _summary(path):
    return json.loads(path.read_text()) if path.exists() else None


def report(out, require_complete=True):
    out = Path(out)
    registry = json.loads((out / "EXPERIMENT_REGISTRY.json").read_text())
    required = []
    from .evaluate import verify_evaluation
    from .select_checkpoint import verify_selection_lock
    from .run_suite import verify_validation_teacher
    from .experiment_registry import EXPERIMENTS, SEEDS
    expected_runs = {f"{experiment}_seed{seed}" for experiment in EXPERIMENTS for seed in SEEDS}
    if set(registry.get("runs", {})) != expected_runs:
        required.append("The registry must contain exactly all eight registered run IDs")
    try:
        locked = verify_selection_lock(out, full=True)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        locked = {"runs": {}}
        required.append("Actual all-eight training/full-Validation selection lock: " + str(error))
    def verified(path, expected, label):
        try:
            return verify_evaluation(path, expected)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            required.append(label + ": " + str(error))
            return None
    rows = []
    teacher_identities = set()
    for run_id, run in registry["runs"].items():
        folder = out / run_id
        selection = _summary(folder / "checkpoints/validation_best.json")
        if not selection:
            required.append(f"{run_id}: checkpoint selection")
            continue
        cfg_path = folder / "effective_config.yaml"
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
        teacher_target = cfg.get("teacher_target")
        identities = _summary(folder / "ASSET_IDENTITIES.json") or {}
        if not cfg or not identities.get("qualification"):
            required.append(f"{run_id}: code/asset engineering qualification")
        if identities.get("files", {}).get("teacher_bundle", {}).get("sha256"):
            teacher_identities.add(identities["files"]["teacher_bundle"]["sha256"])
        try:
            from .train import verify_qualification
            if verify_qualification(cfg) != identities.get("qualification"):
                raise ValueError("Qualified source/assets identity changed")
        except (OSError, ValueError, KeyError) as error:
            required.append(f"{run_id}: actual code/asset qualification: {error}")
        for label in ("last", "best"):
            paths = {"test": folder / "test" / label / "SUMMARY.json",
                     "train_representation": folder / "diagnostics" / f"train_representation_{label}" / "SUMMARY.json",
                     "train_panel": folder / "diagnostics" / f"train_panel_{label}" / "SUMMARY.json",
                     "validation_representation": folder / "diagnostics" / f"validation_representation_{label}" / "SUMMARY.json"}
            entry = selection.get(label, {})
            checksum = entry.get("sha256")
            expectations = {
                "test": {"population": "test", "scene_count": 12146, "noise_indices": [0, 1, 2, 3, 4], "artifact_kind": "behavior", "mode": "student", "full_population": True, "target_version": None},
                "train_representation": {"population": "train", "scene_count": 85813, "noise_indices": [0, 1, 2, 3, 4], "artifact_kind": "representation", "mode": "student", "full_population": True, "target_version": teacher_target},
                "validation_representation": {"population": "validation", "scene_count": 17475, "noise_indices": [0, 1, 2, 3, 4], "artifact_kind": "representation", "mode": "student", "full_population": True, "target_version": teacher_target},
                "train_panel": {"population": "train", "scene_count": 8192, "noise_indices": [0, 1, 2, 3, 4], "artifact_kind": "behavior", "mode": "student", "full_population": False, "target_version": teacher_target}}
            summaries = {key: verified(path.parent, dict(expectations[key], checkpoint_sha256=checksum),
                                       f"{run_id}: {label} {key}") for key, path in paths.items()}
            function_summaries = {}
            for population in ("train", "validation"):
                function_summaries[population] = verified(folder / "diagnostics" / f"{population}_functional_{label}",
                         {"population": population, "scene_count": 2048, "noise_indices": [0],
                          "artifact_kind": "functional", "mode": "student", "full_population": False,
                          "target_version": teacher_target, "checkpoint_sha256": checksum},
                         f"{run_id}: {label} {population} all-five functional diagnostics")
            teacher_summary = verified(out / "train_panel_teachers" / str(teacher_target),
                                      {"population": "train", "scene_count": 8192, "noise_indices": [0, 1, 2, 3, 4],
                                       "artifact_kind": "behavior", "mode": "teacher", "target_version": teacher_target},
                                      f"{run_id}: matched frozen teacher panel")
            capture = _summary(folder / "diagnostics" / f"capture_{label}.json")
            if not capture or capture.get("target_version") != teacher_target or capture.get("noise_indices") != [0, 1, 2, 3, 4]:
                required.append(f"{run_id}: {label} matched-population gain capture")
            elif summaries["train_panel"] and teacher_summary:
                from .checkpointing import sha256
                teacher_folder = out / "train_panel_teachers" / str(teacher_target)
                student_folder = paths["train_panel"].parent
                student_protocol = _summary(student_folder / "PROTOCOL.json")
                teacher_protocol = _summary(teacher_folder / "PROTOCOL.json")
                if capture.get("student_summary_sha256") != sha256(student_folder / "SUMMARY.json") or capture.get("teacher_summary_sha256") != sha256(teacher_folder / "SUMMARY.json") or capture.get("tokens_sha256") != student_protocol.get("tokens_sha256") or student_protocol.get("tokens_sha256") != teacher_protocol.get("tokens_sha256"):
                    required.append(f"{run_id}: {label} gain capture cohort/artifact identity")
            epoch = 40 if label == "last" else selection["best"]["epoch"]
            validation = _summary(folder / "validation" / f"epoch_{epoch:03d}" / "SUMMARY.json")
            test = summaries["test"] or {}
            behavior = test.get("behavior", {})
            train_repr = (summaries["train_representation"] or {}).get("representation", {}).get("all", {})
            val_repr = (summaries["validation_representation"] or {}).get("representation", {}).get("all", {})
            train_complete = _summary(folder / "TRAIN_COMPLETE.json") or {}
            row = {"run": run_id, "experiment": run["experiment"], "training_seed": run["training_seed"],
                   "checkpoint": label, "epoch": epoch, "checkpoint_sha256": test.get("checkpoint_sha256"),
                   "train_h_normalized": train_repr.get("h_normalized"), "val_h_normalized": val_repr.get("h_normalized"),
                   "validation_pdms_points": (validation or {}).get("behavior", {}).get("pdms_points"),
                   "test_pdms_points": behavior.get("pdms_points"), "test_delta_points": behavior.get("delta_points"),
                   "new_zero_events": test.get("events", {}).get("new_zero_score"),
                   "paired_base_gt90_student_zero": test.get("events", {}).get("paired_base_gt90_student_zero"),
                   "train_wall_seconds": train_complete.get("wall_seconds_total", train_complete.get("wall_seconds_this_process")),
                   "test_wall_seconds": test.get("wall_seconds_this_process"),
                   "world_size": train_complete.get("world_size"),
                   "global_batch_size": train_complete.get("global_batch_size"),
                   "microbatch_per_device": train_complete.get("microbatch_per_device"),
                   "gradient_accumulation_steps": train_complete.get("gradient_accumulation_steps"),
                   "status": run["status"]}
            row.update({f"test_{key}_points": value for key, value in test.get("components_points", {}).items()})
            representation = (summaries["train_representation"] or {}).get("representation", {})
            row.update(train_keep_predicted_rms=representation.get("keep", {}).get("predicted_rms"),
                       train_keep_h_normalized=representation.get("keep", {}).get("h_normalized"),
                       train_repair_h_normalized=representation.get("repair", {}).get("h_normalized"),
                       train_repair_direction_cosine=representation.get("repair", {}).get("direction_cosine"),
                       train_repair_amplitude_ratio=representation.get("repair", {}).get("amplitude_ratio"),
                       train_panel_gain_capture=(capture or {}).get("gain_capture"),
                       train_panel_teacher_pdms_points=(capture or {}).get("teacher_pdms_points"),
                       train_panel_student_pdms_points=(capture or {}).get("student_pdms_points"))
            for population, function_summary in function_summaries.items():
                for source in ("teacher", "student"):
                    values = (function_summary or {}).get("functional", {}).get(source, [])
                    row[f"{population}_functional_{source}_mean_normalized_mse"] = float(np.mean([value["normalized_mse"] for value in values])) if len(values) == 5 else None
            tasks = [value for key, value in registry.get("tasks", {}).items()
                     if key.startswith("train__" + run_id + "__") or key.startswith(run_id + "__")]
            measured = [value for value in tasks if value.get("wall_seconds") is not None]
            row["allocated_gpu_hours_measured"] = sum(value["wall_seconds"] * len(value.get("devices", [])) for value in measured) / 3600
            row["tasks_with_unknown_wall_cost"] = len(tasks) - len(measured)
            training_tasks = [value for key, value in registry.get("tasks", {}).items() if key.startswith("train__" + run_id + "__")]
            row["train_wall_seconds_total_measured"] = sum(value["wall_seconds"] for value in training_tasks if value.get("wall_seconds") is not None)
            row["train_attempts_with_unknown_wall_cost"] = sum(value.get("wall_seconds") is None for value in training_tasks)
            rows.append(row)
        verified(folder / "diagnostics/validation_new_noise",
                 {"population": "validation", "scene_count": 17475, "noise_indices": [5, 6, 7, 8, 9],
                  "artifact_kind": "behavior", "mode": "student", "full_population": True,
                  "target_version": None, "checkpoint_sha256": selection.get("best", {}).get("sha256")},
                 f"{run_id}: validation S5-S9")
    try:
        if len(teacher_identities) != 1:
            raise ValueError("All eight runs must share exactly one frozen local teacher")
        verify_validation_teacher(out, next(iter(teacher_identities)))
    except (OSError, ValueError, KeyError) as error:
        required.append("Full validation frozen teacher diagnostics: " + str(error))
    atomic_json(out / "INCOMPLETE_OR_FAILED_TASKS.json", {"incomplete": required, "complete": not required})
    if required and require_complete:
        raise RuntimeError("Round1 report is incomplete: " + "; ".join(required))
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "ROUND1_RESULTS.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    def fmt(value):
        return "pending" if value is None else f"{value:.5f}" if isinstance(value, float) else str(value)
    table = ["| Run | Checkpoint | Epoch | Train H | Val H | Val PDMS | Test PDMS | Test delta | New zero | Paired >90 to zero |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        table.append("| " + " | ".join(fmt(row.get(key)) for key in ("run", "checkpoint", "epoch", "train_h_normalized", "val_h_normalized", "validation_pdms_points", "test_pdms_points", "test_delta_points", "new_zero_events", "paired_base_gt90_student_zero")) + " |")
    (out / "ROUND1_RESULTS.md").write_text("\n".join(table) + "\n")
    grouped = defaultdict(list)
    for row in rows:
        if row["test_pdms_points"] is not None:
            grouped[(row["experiment"], row["checkpoint"])].append(row["test_pdms_points"])
    seed_summary = {"|".join(key): {"n_training_seeds": len(values), "mean_test_pdms_points": float(np.mean(values)),
                                   "sample_std_test_pdms_points": float(np.std(values, ddof=1)) if len(values) > 1 else None}
                    for key, values in grouped.items()}
    atomic_json(out / "TRAINING_SEED_SUMMARY.json", seed_summary)
    text = ["# Round1 continuous no-routing distillation", "",
            "Status: " + ("REPORT_COMPLETE" if not required else "INCOMPLETE"), "",
            "All student checkpoints begin from the original RL Base and share the same independently refit, frozen local D_C8 teacher. Historical PPU oracle scores are not student results.", "",
            "Each scene receives equal weight after averaging its five action-noise repetitions. S0–S4 are sampling repetitions, while 20261027 and 20261028 are the two independent training seeds. Intervals in each SUMMARY.json resample entire logs 2,000 times with paired Base/student draws.", "",
            *table, "", "## Scientific comparisons", ""]
    best_by_exp = {experiment: [row for row in rows if row["experiment"] == experiment and row["checkpoint"] == "best"] for experiment in ("H_full", "H_gain1", "H_mu_teacher", "H_mu_student")}
    for left, right, question in (("H_full", "H_gain1", "Target sparsification (B−A)"), ("H_gain1", "H_mu_teacher", "Teacher-state functional supervision (C−B)"), ("H_mu_teacher", "H_mu_student", "Student-state supervision (D−C)")):
        differences = []
        for seed in (20261027, 20261028):
            a = next((row for row in best_by_exp[left] if row["training_seed"] == seed), None)
            b = next((row for row in best_by_exp[right] if row["training_seed"] == seed), None)
            if a and b and a["test_pdms_points"] is not None and b["test_pdms_points"] is not None:
                differences.append(f"seed {seed}: {b['test_pdms_points'] - a['test_pdms_points']:+.5f} Test points")
        text += [f"{question}: " + ("; ".join(differences) if differences else "pending") + ".", ""]
    text += ["## Representation, preservation, behavior and limitations", "",
             "The per-run summaries report full-Train representation error, held-out-log validation representation error, direction and amplitude, zero-target leakage, and source groups. Train behavior is restricted to the fixed 8,192-scene log × Base-score panel; it is not full-Train PDMS. Each behavior summary separates positive gains from negative losses, all native PDM components, per-log changes and paired severe events.", "",
             "The additional S5–S9 validation run occurs after the selection lock and never changes the selected checkpoint. Teacher validation choices are post-hoc diagnostic upper bounds and never enter training or calibration. Two training seeds provide limited information about training randomness. A failure of this adapter does not establish that all representation distillation is impossible.", "",
             "Full Train/Validation error differences below are descriptive log-held-out gaps; no significance claim is inferred from their ratio. Functional diagnostics use the same fixed 2,048-scene Train and Validation cohorts for every arm, explicit S0, all five native DDIM steps, and both teacher and current-student state distributions. They are diagnostics, not additional training exposures.", ""]
    for row in rows:
        if row["checkpoint"] != "best":
            continue
        text += [f"{row['run']}: Train/Validation normalized H={fmt(row['train_h_normalized'])}/{fmt(row['val_h_normalized'])}; zero-target predicted RMS={fmt(row['train_keep_predicted_rms'])}; repaired-target direction cosine={fmt(row['train_repair_direction_cosine'])}, amplitude ratio={fmt(row['train_repair_amplitude_ratio'])}. Matched Train-panel student/teacher PDMS={fmt(row['train_panel_student_pdms_points'])}/{fmt(row['train_panel_teacher_pdms_points'])}, gain capture={fmt(row['train_panel_gain_capture'])}. Validation normalized functional MSE on teacher/student states={fmt(row.get('validation_functional_teacher_mean_normalized_mse'))}/{fmt(row.get('validation_functional_student_mean_normalized_mse'))}.", ""]
    text += ["Proxy error and PDMS are reported separately: a reduction in H or functional error is not called a behavior improvement without the complete paired PDMS result. Positive gains, normal-scene damage, severe paired regressions and per-log differences remain in every behavior SUMMARY.json, including negative experiments. Oracle values use only the matched target version and cohort; full-Train oracle scores are never used as panel denominators. There is no extra keep loss, deployment route, scorer or choice-table input. Upstream LoRA or larger input representations remain possible future work rather than replacements for this completed fixed adapter budget.", "",
             "## Measured cost and provenance", "",
             "ROUND1_RESULTS.csv records measured training/evaluation wall time and world size. Runtime logs and the registry preserve launches, restarts and errors. Allocated GPU hours use measured child-process wall time times the allocated GPU count, including CPU and IO waits; they do not claim measured CUDA-active time. Missing costs remain unknown rather than estimated as measurements.", ""]
    if required:
        text += ["## Remaining work", "", *[f"- {item}" for item in required], ""]
    (out / "ROUND1_REPORT.md").write_text("\n".join(text))
    make_figures(out, rows)
    return {"rows": len(rows), "incomplete": required}


def make_figures(out, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for run in sorted({row["run"] for row in rows}):
        path = out / run / "logs/train_metrics.jsonl"
        if not path.exists():
            continue
        records = read_records(path)
        # Epoch aggregation keeps full scientific histories readable.
        by_epoch = defaultdict(list)
        for value in records:
            by_epoch[value["epoch"]].append(value)
        epochs = sorted(by_epoch)
        axes[0].plot(epochs, [np.average([value["h"] for value in by_epoch[epoch]], weights=[value["real_scenes"] for value in by_epoch[epoch]]) for epoch in epochs], label=run)
        axes[1].plot(epochs, [np.average([value["mu"] for value in by_epoch[epoch]], weights=[value["real_scenes"] for value in by_epoch[epoch]]) for epoch in epochs], label=run)
    axes[0].set(xlabel="Epoch", ylabel="Normalized H loss")
    axes[1].set(xlabel="Epoch", ylabel="Normalized transition-mean loss")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(figures / "learning_curves.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(6, 4))
    for row in rows:
        if row["train_h_normalized"] is not None and row["test_pdms_points"] is not None:
            axis.scatter(row["train_h_normalized"], row["test_pdms_points"], label=f"{row['run']} {row['checkpoint']}")
    axis.set(xlabel="Full Train normalized H error", ylabel="Full Test PDMS points")
    axis.legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(figures / "representation_vs_pdms.png", dpi=160)
    plt.close(fig)
    for filename, xkey, ykey in (("repair_gain_vs_preservation_loss", "positive_gain_sum_points", "negative_loss_sum_points"),):
        fig, axis = plt.subplots(figsize=(6, 4))
        for row in rows:
            summary = _summary(out / row["run"] / "test" / row["checkpoint"] / "SUMMARY.json") or {}
            behavior = summary.get("behavior", {})
            if xkey in behavior:
                axis.scatter(behavior[xkey], behavior[ykey], label=f"{row['run']} {row['checkpoint']}")
        axis.set(xlabel="Sum of scene gains (points)", ylabel="Sum of scene losses (points)")
        fig.tight_layout()
        fig.savefig(figures / f"{filename}.png", dpi=160)
        plt.close(fig)
    fig, axis = plt.subplots(figsize=(12, 4))
    for row in rows:
        if row["checkpoint"] != "best":
            continue
        summary = _summary(out / row["run"] / "test/best/SUMMARY.json") or {}
        logs = summary.get("groups", {}).get("log", {})
        axis.plot(range(len(logs)), [logs[key]["delta_points"] for key in sorted(logs)], label=row["run"])
    axis.set(xlabel="Sorted Test log", ylabel="Paired PDMS delta points")
    axis.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(figures / "per_log_deltas.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for row in rows:
        if row["checkpoint"] != "best":
            continue
        summary = _summary(out / row["run"] / "diagnostics/validation_functional_best/SUMMARY.json") or {}
        for axis, source in zip(axes, ("teacher", "student")):
            values = summary.get("functional", {}).get(source, [])
            if len(values) == 5:
                axis.plot(range(5), [value["normalized_mse"] for value in values], marker="o", label=row["run"])
            axis.set(xlabel="Native DDIM step index", ylabel="Normalized transition-mean MSE", title=source + " states")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(figures / "per_step_function_error.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    report(args.out, not args.allow_incomplete)


if __name__ == "__main__":
    main()
