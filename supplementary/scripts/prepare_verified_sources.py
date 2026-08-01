#!/usr/bin/env python3
"""Adapt the audited NAVSIM exports into explicit canonical source tables.

Only column renaming, row stacking, and deterministic state labels are
performed.  The script never changes a metric and deliberately uses the
historical scalar-GRPO versus Pareto-v2 pair for the paper's 658-scene study.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import (
    ANALYSIS_SEED,
    atomic_write_json,
    command_display,
    environment_versions,
    git_revision,
    repository_root,
    sha256_file,
    supplementary_root,
)


DEFAULT_INPUTS = {
    "v1_final": "outputs/ampt_pdms914505_navtest_submission_20260730/full_navtest_pdms.csv",
    "v1_anchor": "outputs/pdms914487_numeric_anchor_reverse_old_residual_conflict_only_top20_graft0025_full_navtest_seed20260726_20260727/epoch_000_step_252/candidate/local_sharded_aggregated.csv",
    "v2_final": "outputs/fair_ec_pdms914153_navtest_20260727/raw_path_temporal_cap2_final_score/navsim_v2_navtest_epdms.csv",
    "hard658": "outputs/navtest_hard_stable_zero658_recogdrive_stage3_vs_pdms914505_20260727/per_token_comparison.csv",
}


def parse_args() -> argparse.Namespace:
    supp = supplementary_root()
    parser = argparse.ArgumentParser(
        description="Create evidence-preserving adapters for the audited v1/v2 and 658-scene exports."
    )
    for key, default in DEFAULT_INPUTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=Path, default=repository_root() / default)
    parser.add_argument("--output-dir", type=Path, default=supp / "derived/source_adapters")
    parser.add_argument("--manifest", type=Path, default=supp / "derived/source_adapters_manifest.json")
    parser.add_argument("--seed", type=int, default=ANALYSIS_SEED, help="Analysis provenance seed; metrics are not resampled here.")
    return parser.parse_args()


def common(frame: pd.DataFrame, *, benchmark: str, method: str, stage: str, variant: str,
           seed: int | None, round_index: int | None, raw_path: Path) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["scene_id"] = frame["token"].astype(str)
    out["split"] = "navtest"
    out["benchmark"] = benchmark
    out["method"] = method
    out["stage"] = stage
    out["variant"] = variant
    out["seed"] = seed
    out["round"] = round_index
    out["sample_index"] = 0
    out["teacher_source"] = pd.NA
    out["teacher_status"] = pd.NA
    out["raw_source_file"] = raw_path.relative_to(repository_root()).as_posix()
    return out


def adapt_v1(path: Path, *, variant: str, round_index: int | None) -> pd.DataFrame:
    raw = pd.read_csv(path)
    # The sharded aggregator appends one summary row whose token is literally
    # ``average``.  It is evidence for the mean, not a scene.
    raw = raw.loc[~raw["token"].astype(str).str.contains("average|score", case=False, na=False)].copy()
    required = {"token", "score", "no_at_fault_collisions", "drivable_area_compliance",
                "ego_progress", "time_to_collision_within_bound", "comfort",
                "driving_direction_compliance"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")
    out = common(raw, benchmark="NAVSIM v1", method="AMPT", stage="APR",
                 variant=variant, seed=20260726, round_index=round_index, raw_path=path)
    rename = {
        "score": "aggregate_score", "no_at_fault_collisions": "NC",
        "drivable_area_compliance": "DAC", "driving_direction_compliance": "DDC",
        "ego_progress": "EP", "time_to_collision_within_bound": "TTC", "comfort": "C",
    }
    for source, target in rename.items():
        out[target] = pd.to_numeric(raw[source], errors="raise")
    out["feasible"] = out["NC"].ge(1.0 - 1e-12) & out["DAC"].ge(1.0 - 1e-12)
    out["zero_score"] = out["aggregate_score"].abs().le(1e-12)
    return out


def adapt_v2(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    out = common(raw, benchmark="NAVSIM v2", method="AMPT", stage="APR",
                 variant="paper_final", seed=None, round_index=3, raw_path=path)
    rename = {
        "score": "aggregate_score", "no_at_fault_collisions": "NC",
        "drivable_area_compliance": "DAC", "driving_direction_compliance": "DDC",
        "traffic_light_compliance": "TLC", "ego_progress": "EP",
        "time_to_collision_within_bound": "TTC", "lane_keeping": "LK",
        "history_comfort": "HC", "two_frame_extended_comfort": "EC",
    }
    missing = set(rename) - set(raw)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")
    for source, target in rename.items():
        out[target] = pd.to_numeric(raw[source], errors="raise")
    out["feasible"] = (
        out["NC"].ge(1.0 - 1e-12) & out["DAC"].ge(1.0 - 1e-12)
        & out["DDC"].ge(1.0 - 1e-12) & out["TLC"].ge(1.0 - 1e-12)
    )
    out["zero_score"] = out["aggregate_score"].abs().le(1e-12)
    return out


def adapt_hard658(path: Path) -> pd.DataFrame:
    """Use the paper-aligned historical pair; the later 91.45 columns are intentionally excluded."""
    raw = pd.read_csv(path)
    frames: list[pd.DataFrame] = []
    specifications = [
        ("scalar GRPO", "scalar_grpo", "orig"),
        ("AMPT", "paper_recovery_checkpoint", "old_v2"),
    ]
    for method, variant, prefix in specifications:
        out = common(raw, benchmark="NAVSIM v1", method=method, stage="FF-PGRPO",
                     variant=variant, seed=None, round_index=None, raw_path=path)
        out["aggregate_score"] = pd.to_numeric(raw[f"{prefix}_PDMS"], errors="raise")
        for raw_suffix, target in (("NC", "NC"), ("DAC", "DAC"), ("DDC", "DDC"),
                                   ("EP", "EP"), ("TTC", "TTC"), ("Comfort", "C")):
            out[target] = pd.to_numeric(raw[f"{prefix}_{raw_suffix}"], errors="raise")
        out["feasible"] = out["aggregate_score"].gt(1e-12)
        out["zero_score"] = ~out["feasible"]
        out["initial_failure"] = True
        out["recovered"] = out["feasible"]
        # Every token belongs to the fixed initial-failure set.  ``new_failure``
        # is therefore false with respect to that initial policy; regressions
        # relative to scalar GRPO are retained separately below.
        out["new_failure"] = False
        out["baseline_aggregate_score"] = pd.to_numeric(raw["orig_PDMS"], errors="raise")
        out["transition_from_scalar"] = "baseline"
        if prefix == "old_v2":
            before = pd.to_numeric(raw["orig_PDMS"], errors="raise")
            after = out["aggregate_score"]
            out["transition_from_scalar"] = "positive_to_positive"
            out.loc[before.le(1e-12) & after.le(1e-12), "transition_from_scalar"] = "zero_to_zero"
            out.loc[before.le(1e-12) & after.gt(1e-12), "transition_from_scalar"] = "zero_to_positive"
            out.loc[before.gt(1e-12) & after.le(1e-12), "transition_from_scalar"] = "positive_to_zero"
        out["root_cause_group"] = raw["root_cause_group"].astype(str)
        frames.append(out)
    stacked = pd.concat(frames, ignore_index=True)
    counts = stacked.loc[stacked["variant"].eq("paper_recovery_checkpoint"), "transition_from_scalar"].value_counts()
    expected = {"zero_to_positive": 93, "positive_to_zero": 20,
                "zero_to_zero": 198, "positive_to_positive": 347}
    observed = {key: int(counts.get(key, 0)) for key in expected}
    if observed != expected:
        raise ValueError(f"658-scene transition mismatch: observed={observed}, expected={expected}")
    return stacked


def main() -> int:
    args = parse_args()
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    paths = {key: Path(getattr(args, key)).resolve() for key in DEFAULT_INPUTS}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{key}: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapters = {
        "v1_final.csv": adapt_v1(paths["v1_final"], variant="paper_final", round_index=3),
        "v1_anchor.csv": adapt_v1(paths["v1_anchor"], variant="paired_anchor", round_index=None),
        "v2_final.csv": adapt_v2(paths["v2_final"]),
        "hard658.csv": adapt_hard658(paths["hard658"]),
    }
    outputs = []
    for name, frame in adapters.items():
        target = args.output_dir / name
        frame.to_csv(target, index=False)
        outputs.append({"path": target.relative_to(repository_root()).as_posix(), "rows": len(frame), "sha256": sha256_file(target)})
    manifest = {
        "schema_version": 1,
        "analysis_seed": args.seed,
        "command": command_display(),
        "git": git_revision(repository_root()),
        "environment": environment_versions(),
        "policy": "metrics are copied without imputation; hard658 excludes the later 91.45 checkpoint",
        "inputs": [{"id": key, "path": path.relative_to(repository_root()).as_posix(), "sha256": sha256_file(path)} for key, path in paths.items()],
        "outputs": outputs,
    }
    atomic_write_json(args.manifest, manifest)
    print(json.dumps({"status": "complete", "rows": {key: len(value) for key, value in adapters.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
