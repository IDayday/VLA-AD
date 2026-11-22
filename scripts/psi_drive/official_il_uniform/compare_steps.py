"""Paired step trends on the full Navtest and historical PSI diagnostic populations."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

NAV_FIELDS = ["score", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
              "time_to_collision_within_bound", "comfort", "driving_direction_compliance"]
DIAG_FIELDS = ["inference_pairwise_ADE", "grpo_pairwise_ADE", "grpo_PDMS_mean", "grpo_PDMS_max", "grpo_PDMS_min"]
PSI = "psi_sft_epoch011"

def read_outputs(root, phase, expected):
    identity = json.loads((root / "identity.json").read_text())
    current = identity["models"][0]["name"]
    result = json.loads((root / phase / "COMPLETE.json").read_text())
    assert result["paired_valid_scenes"] == result["expected_scenes"] == expected
    assert not any(result["failures"].values())
    with (root / phase / "paired_scenes.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len({r["token"] for r in rows}) == expected
    return current, identity, result, {r["token"]: r for r in rows}

def bootstrap_difference(contrast, counts, groups):
    totals = np.bincount(groups, weights=contrast)
    n = len(counts)
    random = np.random.default_rng(20261122)
    weights = random.multinomial(n, np.full(n, 1/n), size=3000)
    distribution = (weights @ totals) / (weights @ counts)
    return [float(x) for x in np.quantile(distribution, [.025, .975])]

def compare(roots, phase, measures, expected):
    data = {step: read_outputs(root, phase, expected) for step, root in roots.items()}
    first = data["37960"]
    scenes = first[3]
    assert set(scenes) == set(data["50000"][3]) == set(data["60000"][3])
    for current, identity, _, scene_rows in data.values():
        assert identity["models"][1]["sha256"] == first[1]["models"][1]["sha256"]
        assert identity["stage1"] == first[1]["stage1"]
        for key in ("nav_manifest_sha256", "nav_metric_sha256", "diag_manifest_sha256",
                    "nav_seed", "diagnostic_seed", "diagnostic_groups", "group_size", "planner_sha256"):
            assert identity[key] == first[1][key]
        assert all(scene_rows[t]["log"] == scenes[t]["log"] for t in scenes)
    _, groups = np.unique([s["log"] for s in scenes.values()], return_inverse=True)
    counts = np.bincount(groups)
    metrics = {}
    for measure in measures:
        psi = np.array([float(scenes[token][f"{PSI}.{measure}"]) for token in scenes])
        values = {}
        for step, (current, _, _, rows) in data.items():
            reference = np.array([float(rows[token][f"{PSI}.{measure}"]) for token in scenes])
            assert np.allclose(reference, psi, atol=1e-6, rtol=0.)
            values[step] = np.array([float(rows[token][f"{current}.{measure}"]) for token in scenes])
        contrasts = {}
        for a, b in [("50000","37960"), ("60000","50000"), ("60000","37960")]:
            diff = values[a] - values[b]
            contrasts[a + "_minus_" + b] = dict(delta=float(diff.mean()),
                                                ci95_log_cluster=bootstrap_difference(diff, counts, groups))
        metrics[measure] = dict(PSI=float(psi.mean()),
                                steps={step: float(vals.mean()) for step, vals in values.items()}, contrasts=contrasts)
    return dict(paired_scenes=expected, log_count=len(counts), metrics=metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for step in (37960, 50000, 60000):
        parser.add_argument(f"--step-{step}", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    roots = {str(step): getattr(args, f"step_{step}") for step in (37960, 50000, 60000)}
    summary = dict(nav=compare(roots, "nav", NAV_FIELDS, 12146),
                   diag=compare(roots, "diag", DIAG_FIELDS, 1000))
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
