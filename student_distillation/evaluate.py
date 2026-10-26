"""Resumable native DDIM replay and real PDM scoring, with paired Base records."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .checkpointing import atomic_json, digest, read_checkpoint, sha256
from . import distributed

SCORE_KEYS = ("score", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
              "time_to_collision_within_bound", "comfort", "driving_direction_compliance")


def fixed_log_subset(metadata, count, namespace="NRKD_monitor_v1"):
    """Deterministic proportional log coverage; no outcome-dependent selection."""
    logs = defaultdict(list)
    for index, meta in enumerate(metadata):
        logs[str(meta["log_id"])].append(index)
    return stratified_indices(logs, metadata, min(count, len(metadata)), namespace)


def stratified_indices(strata, metadata, count, namespace):
    total = sum(map(len, strata.values()))
    allocation = {key: count * len(indices) // total for key, indices in strata.items()}
    remaining = count - sum(allocation.values())
    order = sorted(strata, key=lambda key: (-(count * len(strata[key]) % total), key))
    for key in order[:remaining]:
        allocation[key] += 1
    result = []
    for key in sorted(strata):
        ordered = sorted(strata[key], key=lambda index: hashlib.sha256(f"{namespace}|{metadata[index]['token']}".encode()).hexdigest())
        result.extend(ordered[:allocation[key]])
    return sorted(result)


def train_panel(metadata, count=8192):
    from .calibration import build_train_behavior_panel
    normalized = []
    for meta in metadata:
        row = dict(meta)
        row.setdefault("base_mean_points", row.get("base_mean_pdms_points"))
        if row["base_mean_points"] is None:
            raise ValueError("Train panel requires frozen five-noise Base mean in points")
        normalized.append(row)
    return build_train_behavior_panel(normalized, count)["indices"]


def metadata_rows(dataset):
    # Dataset providers can expose a cheap metadata sequence to avoid loading H0.
    value = getattr(dataset, "metadata", None)
    if value is not None and not callable(value):
        return list(value)
    return [dataset[index]["metadata"] for index in range(len(dataset))]


def _scoring():
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "configs/repair_final_baseline.json").read_text())["scoring"]


def _score(trajectory, meta, scoring):
    from repair_score import score_group
    path = meta.get("metric_cache", meta.get("metric_path"))
    if not path:
        raise ValueError(f"Metric cache is missing for {meta['token']}")
    checksum = meta.get("local_metric_sha256", meta.get("metric_cache_sha256", meta.get("metric_sha256")))
    if not checksum:
        raise ValueError("Evaluation requires an explicitly verified local metric-cache SHA")
    return score_group({"metric_cache": path, "metric_sha256": checksum, "scoring": scoring,
                        "records": [{"arm": "student", "seed_index": 0,
                                     "trajectory": np.asarray(trajectory, dtype=np.float32).tolist()}]})[0]["score"]


def verify_evaluation(path, expected=None, *, required=True):
    """Validate real artifacts and their intended scientific identity, not flags."""
    path = Path(path)
    marker = path / "COMPLETE.json"
    if not marker.exists():
        if required:
            raise ValueError(f"Missing completed evaluation: {path}")
        return None
    complete = json.loads(marker.read_text())
    names = ("scene_scores.jsonl", "SUMMARY.json", "PROTOCOL.json")
    if complete.get("status") != "COMPLETE" or set(complete.get("artifacts", {})) != set(names):
        raise ValueError(f"Invalid evaluation completion manifest: {path}")
    for name in names:
        if sha256(path / name) != complete["artifacts"][name]:
            raise ValueError(f"Evaluation artifact SHA mismatch: {path/name}")
    protocol = json.loads((path / "PROTOCOL.json").read_text())
    protocol_sha = protocol.pop("protocol_sha256")
    if digest(protocol) != protocol_sha or complete.get("protocol_sha256") != protocol_sha:
        raise ValueError("Evaluation protocol digest mismatch")
    for key, value in (expected or {}).items():
        if protocol.get(key) != value:
            raise ValueError(f"Evaluation identity differs for {key}: {path}")
    summary = json.loads((path / "SUMMARY.json").read_text())
    if summary.get("protocol_sha256") != protocol_sha or summary.get("scenes") != protocol.get("scene_count") or not summary.get("scenes"):
        raise ValueError("Evaluation summary coverage/protocol mismatch")
    for key in ("checkpoint_sha256", "population", "noise_indices", "mode", "artifact_kind", "target_version", "full_population"):
        if summary.get(key) != protocol.get(key):
            raise ValueError(f"Evaluation summary identity mismatch for {key}")
    kind = protocol.get("artifact_kind")
    if kind not in ("behavior", "representation", "functional"):
        raise ValueError("Unknown evaluation artifact kind")
    tokens = []
    with (path / "scene_scores.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            tokens.append(row["token"])
            if row.get("protocol_sha256") != protocol_sha:
                raise ValueError("Scene record belongs to a different evaluation protocol")
            if kind == "behavior":
                if [item["noise_index"] for item in row.get("noise", [])] != protocol["noise_indices"]:
                    raise ValueError("Scene is missing required behavior noise scores")
                for item in row["noise"]:
                    for field in ("trajectory", "base_trajectory"):
                        poses = np.asarray(item.get(field))
                        if poses.shape != (8, 3) or not np.isfinite(poses).all():
                            raise ValueError("Behavior evidence must retain actual paired native trajectories")
                    for field in ("score", "base_score"):
                        if set(item[field]) != set(SCORE_KEYS) or any(not math.isfinite(v) or not 0 <= v <= 1 + 1e-8 for v in item[field].values()):
                            raise ValueError("Invalid or incomplete native PDM components")
            elif kind == "representation":
                representation = row.get("representation", {})
                if not all(key in representation and math.isfinite(representation[key]) for key in ("h_mse", "h_normalized", "predicted_rms", "target_rms")):
                    raise ValueError("Scene has no finite teacher representation diagnostics")
            else:
                records = row.get("function", [])
                if [record["noise_index"] for record in records] != protocol["noise_indices"]:
                    raise ValueError("Scene is missing required functional noise records")
                for record in records:
                    for source in ("teacher", "student"):
                        values = record.get(source, [])
                        if [item["step_index"] for item in values] != list(range(5)) or any(not math.isfinite(item["normalized_mse"]) for item in values):
                            raise ValueError("Functional diagnostic must cover all five steps at both state sources")
    if len(tokens) != protocol["scene_count"] or len(set(tokens)) != len(tokens) or digest(tokens) != protocol["tokens_sha256"]:
        raise ValueError("Actual scene artifact does not cover the prescribed cohort")
    if kind == "representation" and summary.get("representation", {}).get("all", {}).get("scenes") != len(tokens):
        raise ValueError("Representation summary does not cover all scenes")
    if kind == "functional" and summary.get("functional", {}).get("scenes") != len(tokens):
        raise ValueError("Functional summary does not cover all scenes")
    if kind == "behavior" and summary.get("behavior", {}).get("scenes") != len(tokens):
        raise ValueError("Behavior summary does not cover all scenes")
    return summary


def functional_record(bridge, batch, result, context, tape, calibration, noise_index):
    valid = batch["image_mask"] | batch["text_mask"]
    teacher = torch.where(valid.unsqueeze(-1), batch["base_post"] + batch["teacher_actual_shift"].unsqueeze(1), batch["base_post"])
    student = result["post"]
    output = {"noise_index": noise_index}
    for source, post in (("teacher", teacher), ("student", student)):
        chain = bridge.states_from_post(post, context, tape)
        values = []
        for j, state in enumerate(chain["states"]):
            if source == "teacher":
                teacher_mean = chain["means"][j]
                student_mean = bridge.mean_from_post(state, j, student, context)[0]
            else:
                student_mean = chain["means"][j]
                teacher_mean = bridge.mean_from_post(state, j, teacher, context)[0]
            mse = float((student_mean.double() - teacher_mean.double()).square().mean())
            values.append({"step_index": j, "mse": mse, "normalized_mse": mse / calibration["mu_mse_scale"][j]})
        output[source] = values
    return output


def representation_record(item, result, calibration):
    valid = item["image_mask"] | item["text_mask"]
    delta = result["delta"][0].detach().cpu().float()
    target = item["teacher_actual_shift"].detach().cpu().float()
    post = result["post"][0].detach().cpu().float()
    base = item["base_post"].detach().cpu().float()
    actual = (post[valid.cpu()] - base[valid.cpu()])
    # Native FP32 addition can make shifts differ slightly across token values.
    teacher_post = torch.where(valid.cpu().unsqueeze(-1), base + target.unsqueeze(0), base)
    error = (post[valid.cpu()] - teacher_post[valid.cpu()]).square().mean().item()
    prediction_norm, target_norm = float(delta.norm()), float(target.norm())
    cosine = float(torch.dot(delta, target) / (delta.norm() * target.norm())) if prediction_norm > 0 and target_norm > 0 else None
    zero = bool(item["metadata"].get("target_is_zero", target_norm == 0.))
    return {"token": item["metadata"]["token"], "log_id": item["metadata"]["log_id"],
            "h_mse": error, "h_normalized": error / calibration["h_mse_scale"],
            "target_is_zero": zero, "target_rms": target.square().mean().sqrt().item(),
            "predicted_rms": actual.square().mean().sqrt().item(), "direction_cosine": cosine,
            "amplitude_ratio": prediction_norm / target_norm if target_norm else None,
            "original_choice": item["metadata"].get("original_choice"),
            "cap_triggered": bool(result.get("cap_triggered", False))}


def evaluate_model(adapter, cfg, bridge, population, out, noise_indices, env=None,
                   checkpoint=None, subset_size=None, panel=False, representation_only=False,
                   target_version=None, mode="student", functional_only=False):
    from .data import SceneDataset, batch_scene
    from .noise import EvaluationNoiseStore
    from .train import dataset_manifest
    env = env or distributed.initialize()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dataset = SceneDataset(dataset_manifest(cfg), population, target_version)
    metadata = metadata_rows(dataset)
    if representation_only and functional_only:
        raise ValueError("Choose one diagnostic kind")
    if (representation_only or functional_only) and target_version not in ("full", "gain1"):
        raise ValueError("Teacher diagnostics require the registered explicit target version")
    expected_count = cfg["data"].get(f"{population}_scenes")
    if expected_count is not None and len(dataset) != expected_count:
        raise ValueError(f"Incomplete {population} population: {len(dataset)} != {expected_count}")
    indices = list(range(len(dataset)))
    if panel:
        indices = train_panel(metadata)
    elif subset_size:
        indices = fixed_log_subset(metadata, subset_size)
    selected_tokens = [metadata[index]["token"] for index in indices]
    noise_store = None
    if not representation_only:
        manifest = cfg["paths"].get("evaluation_noise_manifest")
        if not manifest:
            raise ValueError("Formal evaluation requires the frozen explicit noise manifest; seed fallback is forbidden")
        noise_store = EvaluationNoiseStore(manifest, population, noise_indices, selected_tokens)
    artifact_kind = "representation" if representation_only else "functional" if functional_only else "behavior"
    protocol = {"population": population, "noise_indices": list(noise_indices),
                "checkpoint_sha256": sha256(checkpoint) if checkpoint else "in_memory",
                "feature_manifest_sha256": sha256(cfg["paths"]["feature_manifest"]),
                "teacher_bundle_sha256": sha256(cfg["paths"]["teacher_bundle"]),
                "target_version": target_version, "representation_only": representation_only,
                "artifact_kind": artifact_kind,
                "mode": mode, "scene_count": len(indices), "tokens_sha256": digest(selected_tokens),
                "scoring": _scoring(), "paired_scene_noise_base": artifact_kind == "behavior",
                "targets_manifest_sha256": sha256(cfg["paths"]["targets_manifest"]) if target_version else None,
                "calibration_sha256": sha256(cfg["paths"]["calibration"]),
                "noise_identity": noise_store.identity if noise_store else None,
                "full_population": not bool(subset_size or panel)}
    protocol_sha = digest(protocol)
    if population == "test":
        lock_path = Path(cfg["paths"]["results_root"]) / "CHECKPOINT_SELECTION_LOCK.json"
        if not lock_path.exists():
            raise ValueError("Test requires the all-eight-run checkpoint selection lock")
        from .select_checkpoint import verify_selection_lock
        lock = verify_selection_lock(cfg["paths"]["results_root"])
        allowed = {entry[label]["sha256"] for entry in lock["runs"].values() for label in ("last", "best")}
        if not lock.get("test_may_start") or not lock.get("all_eight_train_and_full_validation_complete") or len(lock["runs"]) != 8 or protocol["checkpoint_sha256"] not in allowed:
            raise ValueError("Test checkpoint is not included in the complete eight-run selection lock")
    marker = out / "COMPLETE.json"
    if marker.exists():
        return verify_evaluation(out, protocol)
    if env.rank == 0:
        atomic_json(out / "PROTOCOL.json", dict(protocol, protocol_sha256=protocol_sha))
        if panel or subset_size:
            atomic_json(out / "PANEL.json", {"tokens": selected_tokens, "sha256": digest(selected_tokens), "scenes": len(indices)})
    calibration = json.loads(Path(cfg["paths"]["calibration"]).read_text())
    adapter.eval()
    rank_file = out / f"rank_{env.rank:04d}.jsonl"
    done = {}
    if rank_file.exists():
        for line in rank_file.read_text().splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                # A preemption may interrupt the final append. Rewrite only valid records.
                break
            if value.get("protocol_sha256") != protocol_sha:
                raise ValueError("Partial evaluation protocol mismatch")
            done[value["token"]] = value
        with rank_file.open("w") as handle:
            for value in done.values():
                handle.write(json.dumps(value) + "\n")
    baseline_root = Path(cfg["paths"].get("results_root", out.parents[1])) / "paired_base_cache" / population
    scoring = _scoring()
    base_identity = digest({"features": protocol["feature_manifest_sha256"], "scoring": scoring,
                            "noise_identity": protocol["noise_identity"], "checkpoint_sha256": sha256(cfg["paths"]["checkpoint"]),
                            "cache_schema": "paired_native_base_v2"})
    baseline_root = baseline_root / base_identity
    start = time.monotonic()
    local_completed = 0
    for index in indices[env.rank::env.world_size]:
        item = dataset[index]
        meta = item["metadata"]
        token = meta["token"]
        if token in done:
            continue
        batch = batch_scene(item, env.device)
        with torch.no_grad(), torch.autocast("cuda" if "cuda" in env.device else "cpu", enabled=False):
            result = adapter(batch["base_post"], batch["image_mask"], batch["text_mask"], batch["state24"])
            record = {"token": token, "log_id": str(meta["log_id"]), "protocol_sha256": protocol_sha,
                      "original_choice": meta.get("original_choice"), "target_is_zero": meta.get("target_is_zero"), "noise": []}
            if (representation_only or functional_only) and not meta.get("target_available", False):
                raise ValueError("Every diagnostic scene must have a frozen teacher target")
            if target_version is not None and meta.get("target_available", False):
                record["representation"] = representation_record(item, result, calibration)
            if not representation_only:
                if bridge is None:
                    raise ValueError("Behavior evaluation requires the verified native bridge")
                context = bridge.encode_context(batch["history12"], batch["status8"])
                post = result["post"]
                if mode == "base":
                    post = batch["base_post"]
                elif mode == "teacher":
                    if not meta.get("target_available", False):
                        raise ValueError("Teacher behavior diagnostics require frozen offline targets")
                    valid = batch["image_mask"] | batch["text_mask"]
                    post = batch["base_post"] + valid.unsqueeze(-1) * batch["teacher_actual_shift"].unsqueeze(1)
                for noise_index in noise_indices:
                    tape = noise_store.tape(token, noise_index, env.device)
                    if functional_only:
                        record.setdefault("function", []).append(functional_record(bridge, batch, result, context, tape, calibration, noise_index))
                        continue
                    trajectory = bridge.states_from_post(post, context, tape)["trajectory"][0].cpu().numpy()
                    score = _score(trajectory, meta, scoring)
                    baseline_file = baseline_root / token / f"S{noise_index}.json"
                    if baseline_file.exists():
                        baseline = json.loads(baseline_file.read_text())
                        if baseline.get("identity") != base_identity or baseline.get("token") != token or baseline.get("noise_index") != noise_index:
                            raise ValueError("Cached Base scene/noise/protocol identity mismatch")
                        if digest({key: value for key, value in baseline.items() if key != "payload_sha256"}) != baseline.get("payload_sha256"):
                            raise ValueError("Cached Base score/trajectory payload SHA mismatch")
                    else:
                        trajectory_base = bridge.states_from_post(batch["base_post"], context, tape)["trajectory"][0].cpu().numpy()
                        baseline = {"score": _score(trajectory_base, meta, scoring), "token": token,
                                    "noise_index": noise_index, "identity": base_identity,
                                    "trajectory": trajectory_base.tolist(),
                                    "trajectory_sha256": hashlib.sha256(np.asarray(trajectory_base, dtype=np.float32).tobytes()).hexdigest()}
                        baseline["payload_sha256"] = digest(baseline)
                        atomic_json(baseline_file, baseline)
                    record["noise"].append({"noise_index": noise_index, "score": score, "base_score": baseline["score"],
                                            "trajectory": trajectory.tolist(), "base_trajectory": baseline["trajectory"],
                                            "noise_bank_sha256": noise_store.identity["sha256"]})
        with rank_file.open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
        local_completed += 1
        if local_completed % 50 == 0:
            atomic_json(out / f"STATE_rank{env.rank:04d}.json", {"completed_this_process": local_completed,
                        "total_this_rank": len(indices[env.rank::env.world_size]), "elapsed_seconds": time.monotonic() - start})
    distributed.barrier()
    summary = None
    if env.rank == 0:
        combined = {}
        for path in out.glob("rank_*.jsonl"):
            for line in path.read_text().splitlines():
                value = json.loads(line)
                if value["protocol_sha256"] == protocol_sha:
                    if value["token"] in combined and combined[value["token"]] != value:
                        raise ValueError("Conflicting duplicated scene evaluation")
                    combined[value["token"]] = value
        if set(combined) != set(selected_tokens):
            raise ValueError("Evaluation did not cover every prescribed scene")
        records = [combined[token] for token in selected_tokens]
        scene_path = out / "scene_scores.jsonl"
        temporary = scene_path.with_suffix(".tmp")
        with temporary.open("w") as handle:
            for value in records:
                if artifact_kind == "behavior" and [entry["noise_index"] for entry in value["noise"]] != list(noise_indices):
                    raise ValueError("A scene is missing an evaluation noise")
                handle.write(json.dumps(value, allow_nan=False) + "\n")
        temporary.replace(scene_path)
        from .aggregate import summarize_records
        summary = summarize_records(records, bootstrap_replicates=cfg["evaluation"]["bootstrap_replicates"])
        summary.update(protocol_sha256=protocol_sha, scenes=len(records), noise_indices=list(noise_indices),
                       wall_seconds_this_process=time.monotonic() - start,
                       checkpoint_sha256=protocol["checkpoint_sha256"], population=population,
                       full_population=not bool(subset_size or panel), mode=mode,
                       artifact_kind=artifact_kind, target_version=target_version)
        atomic_json(out / "SUMMARY.json", summary)
        atomic_json(marker, {"status": "COMPLETE", "protocol_sha256": protocol_sha,
                             "artifacts": {name: sha256(out / name) for name in ("scene_scores.jsonl", "SUMMARY.json", "PROTOCOL.json")}})
    distributed.barrier()
    if env.rank == 0:
        verify_evaluation(out, protocol)
    distributed.barrier()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--population", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--noise-indices", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-version", choices=("full", "gain1"))
    parser.add_argument("--representation-only", action="store_true")
    parser.add_argument("--functional-only", action="store_true")
    parser.add_argument("--subset-size", type=int)
    parser.add_argument("--train-panel", action="store_true")
    parser.add_argument("--mode", choices=("student", "base", "teacher"), default="student")
    args = parser.parse_args()
    from .experiment_registry import load_config
    from .train import make_adapter, make_bridge
    cfg = load_config(args.config)
    env = distributed.initialize()
    adapter, _, _ = make_adapter(cfg, env.device)
    checkpoint = read_checkpoint(args.checkpoint)
    adapter.load_state_dict(checkpoint["adapter"], strict=True)
    for key in ("teacher_bundle", "feature_manifest", "calibration"):
        expected = checkpoint["identities"].get("files", {}).get(key, {}).get("sha256")
        if expected and sha256(cfg["paths"][key]) != expected:
            raise ValueError(f"Evaluation {key} identity differs from training")
    bridge = None if args.representation_only else make_bridge(cfg, env.device, args.out)
    evaluate_model(adapter, cfg, bridge, args.population, args.out, args.noise_indices, env,
                   checkpoint=args.checkpoint, panel=args.train_panel, representation_only=args.representation_only,
                   target_version=args.target_version, mode=args.mode, functional_only=args.functional_only,
                   subset_size=args.subset_size)


if __name__ == "__main__":
    main()
