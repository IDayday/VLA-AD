"""Stream verified inference artifacts through the existing NAVSIM v1 evaluators."""
import argparse
import concurrent.futures
import csv
import dataclasses
import hashlib
import json
import lzma
import pickle
import time
import traceback
from pathlib import Path

import numpy as np

from eval_support import run_paths

OUT = run_paths()["output"]
FIELDS = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound",
          "comfort", "driving_direction_compliance", "score"]
STATE = {}

def read(path):
    return json.loads(Path(path).read_text())

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)

def inputs(phase):
    identity = read(OUT / "identity.json")
    if phase == "nav":
        rows = [{**x, "metric_cache_path": x["cache_path"]} for x in read(identity["nav_metric"])]
    else:
        rows = read(identity["diag_manifest"])["scenes"]
    assert len(rows) == len({row["token"] for row in rows})
    return identity, digest(OUT / "identity.json"), rows

def initialize(phase, identity_hash, root, archive=None):
    STATE.update(phase=phase, identity_hash=identity_hash, root=root, archive=archive)
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    sampling = TrajectorySampling(num_poses=40, interval_length=.1)
    STATE.update(sampling=sampling, simulator=PDMSimulator(sampling), scorer=PDMScorer(sampling, PDMScorerConfig()))

def nav_scores(cache, trajectories):
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    results = []
    for poses in trajectories:
        try:
            proposal = Trajectory(np.asarray(poses, dtype=np.float64), TrajectorySampling(num_poses=8, interval_length=.5))
            score = dataclasses.asdict(pdm_score(cache, proposal, STATE["sampling"], STATE["simulator"], STATE["scorer"]))
            if set(score) != set(FIELDS) or not all(np.isfinite(list(score.values()))):
                raise ValueError("Invalid official metric result")
            results.append(dict(status="ok", **{metric: float(score[metric]) for metric in FIELDS}))
        except Exception as err:
            results.append(dict(status="failed", error=repr(err)))
    return results

def pairwise_groups(trajectories):
    xy = trajectories[..., :2].astype(np.float64)
    first, second = np.triu_indices(xy.shape[-3], k=1)
    return np.linalg.norm(xy[..., first, :, :] - xy[..., second, :, :], axis=-1).mean(axis=(-1, -2))

def process(row):
    token = row["token"]
    root = STATE["root"]
    path = root / "predictions" / f"{token}.npz"
    declared_hash = path.with_suffix(".sha256").read_text().strip()
    assert digest(path) == declared_hash
    record = dict(token=token, log=row["log"], identity_hash=STATE["identity_hash"], proposal_sha256=declared_hash)
    try:
        with np.load(path, allow_pickle=False) as data:
            assert str(data["token"]) == token and str(data["identity_hash"]) == STATE["identity_hash"]
            with lzma.open(row["metric_cache_path"], "rb") as handle:
                cache = pickle.load(handle)
            if STATE["phase"] == "nav":
                trajectories = data["trajectories"]
                assert trajectories.shape == (2, 8, 3)
                record["models"] = nav_scores(cache, trajectories)
            else:
                rollouts = data["native_grpo"]
                assert rollouts.shape in {(2, 4, 16, 8, 3), (1, 4, 16, 8, 3)}
                from navsim.evaluate.pdm_score_batch import pdm_score_batch_same_cache
                flat = rollouts.astype(np.float64).reshape(-1, 8, 3)
                result = pdm_score_batch_same_cache(cache, flat, STATE["sampling"], STATE["simulator"],
                                                    STATE["scorer"], use_exact_array_conversion=True)
                scores = np.asarray([r.score for r in result]).reshape(rollouts.shape[:-2])
                assert np.isfinite(scores).all() and ((scores >= -1e-8) & (scores <= 1 + 1e-8)).all()
                record["models"] = [
                    dict(status="ok", inference_pairwise_ADE=eval_ade.tolist(), grpo_pairwise_ADE=native_ade.tolist(),
                         grpo_PDMS_mean=score.mean(axis=-1).tolist(), grpo_PDMS_max=score.max(axis=-1).tolist(),
                         grpo_PDMS_min=score.min(axis=-1).tolist(), grpo_PDMS_samples=score.tolist())
                    for eval_ade, native_ade, score in zip(pairwise_groups(data["eval"]), pairwise_groups(rollouts), scores)
                ]
                if len(record["models"]) == 1:
                    archive = STATE["archive"]
                    assert archive is not None
                    record["models"].append(archive["records"][token])
                    record["historical_PSI_sha256"] = archive["sha256"]

    except (OSError, EOFError):
        raise
    except Exception:
        record["models"] = [dict(status="failed", error=traceback.format_exc()) for _ in range(2)]
    atomic_json(root / "scores" / f"{token}.json", record)
    return [m["status"] for m in record["models"]]

def diagnostic_parity(row, root):
    token = row["token"]
    with np.load(root / "predictions" / f"{token}.npz", allow_pickle=False) as data:
        trajectories = data["native_grpo"].reshape(-1, 8, 3)[::8]
    from navsim.evaluate.pdm_score_batch import pdm_score_batch_same_cache
    with lzma.open(row["metric_cache_path"], "rb") as handle:
        cache = pickle.load(handle)
    batch = pdm_score_batch_same_cache(cache, trajectories.astype(np.float64), STATE["sampling"],
                                       STATE["simulator"], STATE["scorer"], use_exact_array_conversion=True)
    scalar = nav_scores(cache, trajectories)
    max_error = max(abs(getattr(b, field) - s[field]) for b, s in zip(batch, scalar) for field in FIELDS)
    assert max_error <= 1e-8
    return dict(token=token, max_abs_error=max_error, trajectories=len(batch), fields=FIELDS)

def paired_interval(diff, logs):
    different_logs, codes = np.unique(logs, return_inverse=True)
    totals = np.bincount(codes, weights=diff)
    counts = np.bincount(codes)
    rng = np.random.default_rng(20261122)
    weights = rng.multinomial(len(different_logs), np.full(len(different_logs), 1 / len(different_logs)), size=3000)
    draws = (weights @ totals) / (weights @ counts)
    return [float(q) for q in np.quantile(draws, [.025, .975])]

def aggregate(identity, identity_hash, rows, root, phase):
    scores = []
    for row in rows:
        token = row["token"]
        value = read(root / "scores" / f"{token}.json")
        path = root / "predictions" / f"{token}.npz"
        assert value["token"] == token and value["log"] == row["log"] and value["identity_hash"] == identity_hash
        assert value["proposal_sha256"] == path.with_suffix(".sha256").read_text().strip() == digest(path)
        scores.append(value)
    models = [m["name"] for m in identity["models"]]
    failures = {model: [row["token"] for row in scores if row["models"][i]["status"] != "ok"]
                for i, model in enumerate(models)}
    measures = FIELDS if phase == "nav" else ["inference_pairwise_ADE", "grpo_pairwise_ADE",
                                                       "grpo_PDMS_mean", "grpo_PDMS_max", "grpo_PDMS_min"]
    paired = [row for row in scores if all(model["status"] == "ok" for model in row["models"])]
    if not paired:
        raise RuntimeError("All scores failed")
    values = np.asarray([[[float(np.mean(model[key])) for key in measures] for model in row["models"]] for row in paired])
    scale = np.array([1.0 if "ADE" in m else 100.0 for m in measures])
    logs = [row["log"] for row in paired]
    differences = (values[:, 0] - values[:, 1]) * scale
    result = dict(phase=phase, expected_scenes=len(rows), evaluated_scenes=len(scores), paired_valid_scenes=len(paired),
                  valid_scenes={key: len(rows) - len(v) for key, v in failures.items()}, failures=failures,
                  results={model: dict(zip(measures, (values[:, i].mean(axis=0) * scale).tolist()))
                           for i, model in enumerate(models)},
                  delta_current_minus_PSI=dict(zip(measures, differences.mean(axis=0).tolist())),
                  delta_ci95_log_cluster={key: paired_interval(differences[:, i], logs) for i, key in enumerate(measures)},
                  identity_sha256=identity_hash)
    if phase == "nav":
        result["zero_PDMS"] = {
            model: sum(row["models"][i].get("score") == 0 for row in paired) for i, model in enumerate(models)
        }
    else:
        result.update(groups_per_scene=identity["diagnostic_groups"], group_size=identity["group_size"])
        reused = [row["historical_PSI_sha256"] for row in paired if "historical_PSI_sha256" in row]
        if reused:
            assert len(set(reused)) == 1
            result.update(PSI_archived_scene_count=len(reused), PSI_archive_sha256=reused[0])

    with (root / "paired_scenes.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["token", "log", *[name + "." + measure for name in models for measure in measures]])
        for row, scene_values in zip(paired, values * scale):
            writer.writerow([row["token"], row["log"], *scene_values.reshape(-1)])
    atomic_json(root / "summary.json", result)
    if len(paired) == len(rows):
        atomic_json(root / "COMPLETE.json", result)
    print(json.dumps(result), flush=True)

def run(args):
    phase = args.phase
    identity, ident, rows = inputs(phase)
    root = OUT / (phase + ("_smoke" if args.smoke else ""))
    (root / "scores").mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    if args.smoke:
        initialize(phase, ident, root)
        existing = [row for row in rows if (root / "predictions" / f"{row['token']}.sha256").is_file()]
        if not existing:
            raise RuntimeError("No smoke predictions")
        for row in existing:
            print(process(row), flush=True)
        if phase == "diag":
            atomic_json(root / "parity.json", [diagnostic_parity(row, root) for row in existing])
        return
    remaining = {row["token"]: row for row in rows if not (root / "scores" / f"{row['token']}.json").exists()}
    done = len(rows) - len(remaining)
    archive = read(args.historical_psi) if args.historical_psi else None
    if archive:
        assert phase == "diag"
        assert archive["reference_checkpoint_sha256"] == identity["models"][1]["sha256"]
        assert digest(Path(archive["path"])) == archive["sha256"]
        assert set(archive["records"]) == {row["token"] for row in rows}
    last = start
    active = {}
    failures = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                                               initargs=(phase, ident, root, archive)) as pool:
        while remaining or active:
            available = args.workers * 3 - len(active)
            if available > 0:
                for token in list(remaining):
                    if (root / "predictions" / f"{token}.sha256").is_file():
                        row = remaining.pop(token)
                        active[pool.submit(process, row)] = row
                        available -= 1
                        if not available:
                            break
            for f in concurrent.futures.wait(active, timeout=2, return_when=concurrent.futures.FIRST_COMPLETED).done:
                row = active.pop(f)
                token = row["token"]
                try:
                    statuses = f.result()
                except (OSError, EOFError):
                    failures[token] = failures.get(token, 0) + 1
                    if failures[token] > 3:
                        raise
                    remaining[token] = row
                    continue
                done += 1
                if statuses != ["ok", "ok"]:
                    print("SCORE_FAILURE", token, statuses, flush=True)
            now = time.monotonic()
            if now - last >= 15:
                print("SCORE_PROGRESS", phase, done, len(rows), "seconds", round(now-start, 1), flush=True)
                last = now
            if now - start > args.timeout:
                raise TimeoutError(f"Scoring completed {done}/{len(rows)} scenes by the deadline")
            if not active and remaining:
                time.sleep(2)
    aggregate(identity, ident, rows, root, phase)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["nav", "diag"])
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--historical-psi", type=Path)
    run(parser.parse_args())
