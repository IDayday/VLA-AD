#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


METRICS = (
    "lfp_transition_floor_endpoint_std_x_m",
    "lfp_transition_floor_endpoint_std_y_m",
    "lfp_transition_floor_endpoint_std_heading_rad",
    "lfp_group_pairwise_ade_m_mean",
    "lfp_group_pairwise_ade_m_p50",
    "lfp_group_endpoint_std_x_m",
    "lfp_group_endpoint_std_y_m",
    "lfp_group_endpoint_std_heading_rad",
    "lfp_group_scalar_span_mean",
    "lfp_scalar_mean",
    "lfp_delta_scalar_mean",
    "lfp_ep_mean",
    "lfp_ttc_mean",
    "lfp_quality_mean",
    "lfp_nc_mean",
    "lfp_dac_mean",
    "lfp_ddc_mean",
    "lfp_feasible_ratio",
    "lfp_progress_ok_ratio",
    "lfp_ttc_ok_ratio",
    "lfp_quality_ok_ratio",
    "lfp_reference_pareto_ok_ratio",
    "lfp_reference_dominated_ratio",
    "lfp_quality_reference_regression_ratio",
    "lfp_pareto_front_ratio",
    "lfp_positive_advantage_ratio",
    "lfp_negative_advantage_ratio",
    "lfp_zero_advantage_ratio",
    "lfp_positive_advantage_below_ref_scalar_ratio",
    "lfp_positive_advantage_below_ref_quality_ratio",
    "lfp_positive_advantage_reference_dominated_ratio",
    "lfp_positive_advantage_scalar_delta_mean",
    "lfp_positive_advantage_quality_delta_mean",
    "lfp_credit_active_group_ratio",
    "lfp_global_centered_std",
    "lfp_frontier_energy_mean",
    "lfp_frontier_base_energy_mean",
    "lfp_pareto_tradeoff_intensity_mean",
    "lfp_pareto_tradeoff_intensity_p50",
    "lfp_frontier_tradeoff_multiplier_mean",
    "lfp_v1_ep_ttc_ddc_tradeoff_intensity_mean",
    "lfp_v1_ep_ttc_ddc_pareto_front_ratio",
    "lfp_v1_ep_scalar_ddc_tradeoff_intensity_mean",
    "lfp_v1_ep_scalar_ddc_pareto_front_ratio",
    "lfp_policy_loss",
    "lfp_exact_kl",
    "lfp_policy_gradient_norm",
    "lfp_planning_adapter_gradient_norm",
    "lfp_policy_gradient_finite_ratio",
    "lfp_planning_adapter_gradient_finite_ratio",
)


def read_profile(profile_dir: Path) -> Dict[str, float]:
    event_files = sorted(profile_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file under {profile_dir}.")
    accumulator = EventAccumulator(str(event_files[-1]), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = set(accumulator.Tags().get("scalars", ()))
    row: Dict[str, float] = {}
    for metric in METRICS:
        candidates = (f"train/{metric}_step", f"train/{metric}_epoch", f"train/{metric}")
        tag = next((candidate for candidate in candidates if candidate in tags), None)
        if tag is None:
            continue
        values = accumulator.Scalars(tag)
        if values:
            row[metric] = float(values[-1].value)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    root = args.matrix_root.resolve()
    profiles_path = root / "profiles.csv"
    profiles = (
        [item.strip() for item in profiles_path.read_text().strip().split(",") if item.strip()]
        if profiles_path.is_file()
        else sorted(path.name for path in root.iterdir() if path.is_dir() and path.name != "logs")
    )
    summary = {profile: read_profile(root / profile) for profile in profiles}
    output_path = args.output_json or root / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    columns = ["profile", *METRICS]
    lines = ["\t".join(columns)]
    for profile in profiles:
        row = summary[profile]
        lines.append(
            "\t".join(
                [profile, *("" if metric not in row else f"{row[metric]:.9g}" for metric in METRICS)]
            )
        )
    (root / "summary.tsv").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
