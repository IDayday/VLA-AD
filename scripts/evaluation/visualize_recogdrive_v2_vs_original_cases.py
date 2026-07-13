#!/usr/bin/env python3
"""Visualize V2 Pareto-GRPO trajectories against original ReCogDrive stage3."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("NUPLAN_MAPS_ROOT", "/mnt/navsim/maps")
os.environ.setdefault("OPENSCENE_DATA_ROOT", "/mnt/navsim")
os.environ.setdefault("NAVSIM_DISABLE_TQDM", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image

from navsim.common.dataclasses import SceneFilter, SensorConfig, Trajectory
from navsim.common.dataloader import SceneLoader
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
from navsim.visualization.camera import add_camera_ax, add_trajectory_to_camera_ax
from navsim.visualization.plots import configure_ax, configure_bev_ax


CATEGORY_DIRS = {
    "safety_improved_by_v2": "01_safety_improved",
    "bad_trajectory_repaired_by_v2": "02_bad_trajectory_repaired",
    "progress_gain_without_safety_drop_by_v2": "03_progress_gain_without_safety_drop",
}

BEV_TRAJ_CONFIGS: Dict[str, Dict[str, Any]] = {
    "gt": {
        "line_color": "#2ca02c",
        "line_color_alpha": 0.95,
        "line_width": 2.6,
        "line_style": "-",
        "marker": "o",
        "marker_size": 4,
        "marker_edge_color": "white",
        "zorder": 7,
    },
    "v2": {
        "line_color": "#d62728",
        "line_color_alpha": 0.95,
        "line_width": 3.0,
        "line_style": "-",
        "marker": "o",
        "marker_size": 4,
        "marker_edge_color": "white",
        "zorder": 8,
    },
    "original": {
        "line_color": "#1f77b4",
        "line_color_alpha": 0.90,
        "line_width": 2.6,
        "line_style": "--",
        "marker": "s",
        "marker_size": 3.6,
        "marker_edge_color": "white",
        "zorder": 6,
    },
}

CAMERA_TRAJ_CONFIGS: Dict[str, Dict[str, Any]] = {
    name: {
        **config,
        "marker": None,
        "arrow_color": config["line_color"],
        "arrow_edge_color": config["line_color"],
        "arrow_alpha": config["line_color_alpha"],
        "arrow_line_width": config["line_width"],
    }
    for name, config in BEV_TRAJ_CONFIGS.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-cases", type=Path, required=True)
    parser.add_argument("--v2-predictions", type=Path, required=True)
    parser.add_argument("--original-predictions", type=Path, required=True)
    parser.add_argument(
        "--v2-results",
        type=Path,
        default=None,
        help="Optional pdm_results_full.csv for the exact V2 predictions being plotted. "
        "Defaults to the prediction file sibling.",
    )
    parser.add_argument(
        "--original-results",
        type=Path,
        default=None,
        help="Optional pdm_results_full.csv for the exact original predictions being plotted. "
        "Defaults to the prediction file sibling.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=Path("/mnt/navsim/test_navsim_logs/test"))
    parser.add_argument("--sensor-blobs-path", type=Path, default=Path("/mnt/navsim/test_sensor_blobs/test"))
    parser.add_argument("--maps-root", type=Path, default=Path("/mnt/navsim/maps"))
    parser.add_argument("--num-history-frames", type=int, default=4)
    parser.add_argument("--num-future-frames", type=int, default=10)
    parser.add_argument("--frame-interval", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--mismatch-warn-threshold", type=float, default=0.1)
    return parser.parse_args()


def load_predictions(path: Path) -> Dict[str, Dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected prediction list in {path}, got {type(records).__name__}")
    output: Dict[str, Dict[str, Any]] = {}
    for record in records:
        token = str(record["token"])
        output[token] = record
    return output


def default_results_path(prediction_path: Path) -> Path:
    return prediction_path.parent / "pdm_results_full.csv"


def load_pdm_results(path: Optional[Path], prediction_path: Path) -> Dict[str, Dict[str, Any]]:
    result_path = path or default_results_path(prediction_path)
    if not result_path.is_file():
        return {}
    df = pd.read_csv(result_path)
    if "token" not in df.columns:
        raise ValueError(f"Missing token column in {result_path}")
    return {str(row.token): row._asdict() for row in df.itertuples(index=False)}


def prediction_score(record: Dict[str, Any], result: Dict[str, Any]) -> float:
    if result and result.get("score", "") != "":
        return float(result["score"])
    pdm = record.get("pdm", {})
    if "score" not in pdm:
        raise KeyError(f"Prediction record for token {record.get('token')} has no pdm score")
    return float(pdm["score"])


def prediction_metric(record: Dict[str, Any], result: Dict[str, Any], key: str) -> float:
    if result and result.get(key, "") != "":
        return float(result[key])
    pdm = record.get("pdm", {})
    if key not in pdm:
        raise KeyError(f"Prediction record for token {record.get('token')} has no pdm metric {key}")
    return float(pdm[key])


def build_plot_row(
    row: pd.Series,
    token: str,
    v2_record: Dict[str, Any],
    original_record: Dict[str, Any],
    v2_result: Dict[str, Any],
    original_result: Dict[str, Any],
    mismatch_warn_threshold: float,
) -> pd.Series:
    data = row.to_dict()
    data["selection_v2_PDMS"] = float(data.get("v2_PDMS", np.nan))
    data["selection_orig_PDMS"] = float(data.get("orig_PDMS", np.nan))
    data["selection_delta_PDMS"] = float(data.get("v2_minus_orig_PDMS", np.nan))

    data["plotted_v2_PDMS"] = prediction_score(v2_record, v2_result)
    data["plotted_orig_PDMS"] = prediction_score(original_record, original_result)
    data["plotted_delta_PDMS"] = data["plotted_v2_PDMS"] - data["plotted_orig_PDMS"]

    metric_map = {
        "EP": "ego_progress",
        "NC": "no_at_fault_collisions",
        "DAC": "drivable_area_compliance",
        "TTC": "time_to_collision_within_bound",
        "Comfort": "comfort",
        "DDC": "driving_direction_compliance",
    }
    for short, key in metric_map.items():
        data[f"plotted_v2_{short}"] = prediction_metric(v2_record, v2_result, key)
        data[f"plotted_orig_{short}"] = prediction_metric(original_record, original_result, key)

    data["selection_vs_plotted_orig_abs_diff"] = abs(data["selection_orig_PDMS"] - data["plotted_orig_PDMS"])
    data["selection_vs_plotted_v2_abs_diff"] = abs(data["selection_v2_PDMS"] - data["plotted_v2_PDMS"])
    data["selection_plot_mismatch"] = bool(
        data["selection_vs_plotted_orig_abs_diff"] > mismatch_warn_threshold
        or data["selection_vs_plotted_v2_abs_diff"] > mismatch_warn_threshold
    )
    data["v2_prediction_log_name"] = str(v2_record.get("log_name", ""))
    data["original_prediction_log_name"] = str(original_record.get("log_name", ""))
    data["v2_prediction_scene_token"] = str(v2_record.get("scene_token", ""))
    data["original_prediction_scene_token"] = str(original_record.get("scene_token", ""))
    data["token"] = token
    return pd.Series(data)


def assert_prediction_alignment(
    token: str,
    selected_log_name: str,
    v2_record: Dict[str, Any],
    original_record: Dict[str, Any],
) -> None:
    for name, record in (("v2", v2_record), ("original", original_record)):
        record_token = str(record.get("token", ""))
        if record_token != token:
            raise ValueError(f"{name} prediction token mismatch: expected {token}, got {record_token}")
        record_log_name = str(record.get("log_name", ""))
        if selected_log_name and record_log_name and record_log_name != selected_log_name:
            raise ValueError(
                f"{name} prediction log_name mismatch for token={token}: "
                f"selected={selected_log_name}, prediction={record_log_name}"
            )
    if str(v2_record.get("scene_token", "")) != str(original_record.get("scene_token", "")):
        raise ValueError(
            f"V2/original scene_token mismatch for token={token}: "
            f"{v2_record.get('scene_token')} vs {original_record.get('scene_token')}"
        )


def assert_gt_alignment(token: str, v2_record: Dict[str, Any], original_record: Dict[str, Any]) -> float:
    if "gt_traj" not in v2_record or "gt_traj" not in original_record:
        return float("nan")
    v2_gt = np.asarray(v2_record["gt_traj"], dtype=np.float32)
    original_gt = np.asarray(original_record["gt_traj"], dtype=np.float32)
    if v2_gt.shape != original_gt.shape:
        raise ValueError(f"GT trajectory shape mismatch for token={token}: {v2_gt.shape} vs {original_gt.shape}")
    max_abs_diff = float(np.max(np.abs(v2_gt - original_gt))) if v2_gt.size else 0.0
    if max_abs_diff > 1e-4:
        raise ValueError(f"GT trajectory mismatch for token={token}: max_abs_diff={max_abs_diff:.6f}")
    return max_abs_diff


def trajectory_from_record(record: Dict[str, Any], key: str) -> Trajectory:
    poses = np.asarray(record[key], dtype=np.float32)
    return Trajectory(poses=poses)


def metric_text(row: pd.Series) -> str:
    return (
        f"Selection/full-navtest row: V2 {row.selection_v2_PDMS:.3f} vs Orig {row.selection_orig_PDMS:.3f} "
        f"(d={row.selection_delta_PDMS:+.3f})\n"
        f"Plotted pred_traj re-score: V2 {row.plotted_v2_PDMS:.3f} vs Orig {row.plotted_orig_PDMS:.3f} "
        f"(d={row.plotted_delta_PDMS:+.3f}); "
        f"EP {row.plotted_v2_EP:.3f}/{row.plotted_orig_EP:.3f}; "
        f"NC {row.plotted_v2_NC:.1f}/{row.plotted_orig_NC:.1f}; "
        f"DAC {row.plotted_v2_DAC:.1f}/{row.plotted_orig_DAC:.1f}; "
        f"TTC {row.plotted_v2_TTC:.1f}/{row.plotted_orig_TTC:.1f}; "
        f"DDC {row.plotted_v2_DDC:.1f}/{row.plotted_orig_DDC:.1f}"
    )


def add_legend(ax: plt.Axes) -> None:
    handles = [
        Line2D([0], [0], color=BEV_TRAJ_CONFIGS["v2"]["line_color"], lw=3, label="V2 Pareto-GRPO"),
        Line2D([0], [0], color=BEV_TRAJ_CONFIGS["original"]["line_color"], lw=2.6, ls="--", label="Original stage3"),
        Line2D([0], [0], color=BEV_TRAJ_CONFIGS["gt"]["line_color"], lw=2.6, label="GT"),
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.92, fontsize=8)


def save_bev(
    scene: Any,
    row: pd.Series,
    gt_traj: Trajectory,
    v2_traj: Trajectory,
    original_traj: Trajectory,
    output_path: Path,
    dpi: int,
) -> None:
    frame_idx = scene.scene_metadata.num_history_frames - 1
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 7.6))
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    add_trajectory_to_bev_ax(ax, gt_traj, BEV_TRAJ_CONFIGS["gt"])
    add_trajectory_to_bev_ax(ax, original_traj, BEV_TRAJ_CONFIGS["original"])
    add_trajectory_to_bev_ax(ax, v2_traj, BEV_TRAJ_CONFIGS["v2"])
    configure_bev_ax(ax)
    configure_ax(ax)
    add_legend(ax)
    ax.set_title(f"{row.candidate_category} | {row.token}", fontsize=9)
    ax.text(
        0.01,
        0.01,
        metric_text(row),
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.90},
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_camera_bev(
    scene: Any,
    row: pd.Series,
    gt_traj: Trajectory,
    v2_traj: Trajectory,
    original_traj: Trajectory,
    output_path: Path,
    dpi: int,
) -> None:
    frame_idx = scene.scene_metadata.num_history_frames - 1
    frame = scene.frames[frame_idx]
    fig = plt.figure(figsize=(13.8, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.03)

    camera_ax = fig.add_subplot(gs[0])
    add_camera_ax(camera_ax, frame.cameras.cam_f0)
    for name, trajectory in (("gt", gt_traj), ("original", original_traj), ("v2", v2_traj)):
        add_trajectory_to_camera_ax(camera_ax, frame.cameras.cam_f0, trajectory, CAMERA_TRAJ_CONFIGS[name])
    camera_ax.axis("off")
    camera_ax.set_title("cam_f0 trajectory overlay", fontsize=9)

    bev_ax = fig.add_subplot(gs[1])
    add_configured_bev_on_ax(bev_ax, scene.map_api, frame)
    add_trajectory_to_bev_ax(bev_ax, gt_traj, BEV_TRAJ_CONFIGS["gt"])
    add_trajectory_to_bev_ax(bev_ax, original_traj, BEV_TRAJ_CONFIGS["original"])
    add_trajectory_to_bev_ax(bev_ax, v2_traj, BEV_TRAJ_CONFIGS["v2"])
    configure_bev_ax(bev_ax)
    configure_ax(bev_ax)
    add_legend(bev_ax)
    bev_ax.set_title(f"{row.token}", fontsize=9)
    fig.suptitle(metric_text(row), fontsize=8, y=0.985)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def image_is_nonblank(path: Path) -> bool:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image)
    return bool(arr.std() > 1.0 and arr.shape[0] > 200 and arr.shape[1] > 200)


def build_scene_loader(rows: pd.DataFrame, args: argparse.Namespace) -> SceneLoader:
    os.environ["NUPLAN_MAPS_ROOT"] = str(args.maps_root)
    tokens: List[str] = rows["token"].astype(str).tolist()
    log_names: List[str] = sorted(rows["log_name"].astype(str).unique().tolist())
    scene_filter = SceneFilter(
        num_history_frames=args.num_history_frames,
        num_future_frames=args.num_future_frames,
        frame_interval=args.frame_interval,
        log_names=log_names,
        tokens=tokens,
    )
    return SceneLoader(
        data_path=args.data_path,
        sensor_blobs_path=args.sensor_blobs_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_all_sensors(include=[args.num_history_frames - 1]),
        load_image_path=False,
    )


def main() -> None:
    args = parse_args()
    rows = pd.read_csv(args.selected_cases)
    v2_predictions = load_predictions(args.v2_predictions)
    original_predictions = load_predictions(args.original_predictions)
    v2_results = load_pdm_results(args.v2_results, args.v2_predictions)
    original_results = load_pdm_results(args.original_results, args.original_predictions)
    scene_loader = build_scene_loader(rows, args)

    manifest_rows: List[Dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        selected_row = pd.Series(row._asdict())
        token = str(row.token)
        if token not in v2_predictions or token not in original_predictions:
            raise KeyError(f"Missing predictions for token {token}")
        if token not in scene_loader.tokens:
            raise KeyError(f"Scene token {token} was not loaded from {args.data_path}")

        v2_record = v2_predictions[token]
        original_record = original_predictions[token]
        assert_prediction_alignment(token, str(row.log_name), v2_record, original_record)
        gt_max_abs_diff = assert_gt_alignment(token, v2_record, original_record)
        row_series = build_plot_row(
            selected_row,
            token,
            v2_record,
            original_record,
            v2_results.get(token, {}),
            original_results.get(token, {}),
            args.mismatch_warn_threshold,
        )

        scene = scene_loader.get_scene_from_token(token)
        v2_traj = trajectory_from_record(v2_record, "pred_traj")
        original_traj = trajectory_from_record(original_record, "pred_traj")
        gt_traj = trajectory_from_record(v2_record, "gt_traj")

        category_dir = CATEGORY_DIRS.get(str(row.candidate_category), str(row.candidate_category))
        stem = f"{int(row.case_rank):02d}_{token}"
        bev_path = args.output_dir / category_dir / f"{stem}_bev.png"
        camera_path = args.output_dir / category_dir / f"{stem}_camera_bev.png"

        save_bev(scene, row_series, gt_traj, v2_traj, original_traj, bev_path, args.dpi)
        save_camera_bev(scene, row_series, gt_traj, v2_traj, original_traj, camera_path, args.dpi)

        manifest_rows.append(
            {
                "candidate_category": row.candidate_category,
                "case_rank": row.case_rank,
                "token": token,
                "log_name": row.log_name,
                "bev_path": str(bev_path),
                "camera_bev_path": str(camera_path),
                "bev_nonblank": image_is_nonblank(bev_path),
                "camera_bev_nonblank": image_is_nonblank(camera_path),
                "selection_v2_PDMS": row_series.selection_v2_PDMS,
                "selection_orig_PDMS": row_series.selection_orig_PDMS,
                "selection_delta_PDMS": row_series.selection_delta_PDMS,
                "plotted_v2_PDMS": row_series.plotted_v2_PDMS,
                "plotted_orig_PDMS": row_series.plotted_orig_PDMS,
                "plotted_delta_PDMS": row_series.plotted_delta_PDMS,
                "selection_vs_plotted_orig_abs_diff": row_series.selection_vs_plotted_orig_abs_diff,
                "selection_vs_plotted_v2_abs_diff": row_series.selection_vs_plotted_v2_abs_diff,
                "selection_plot_mismatch": row_series.selection_plot_mismatch,
                "gt_max_abs_diff": gt_max_abs_diff,
                "v2_prediction_path": str(args.v2_predictions),
                "original_prediction_path": str(args.original_predictions),
                "v2_results_path": str(args.v2_results or default_results_path(args.v2_predictions)),
                "original_results_path": str(args.original_results or default_results_path(args.original_predictions)),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_dir / "manifest.csv", index=False)
    provenance = {
        "selected_cases": str(args.selected_cases),
        "v2_predictions": str(args.v2_predictions),
        "original_predictions": str(args.original_predictions),
        "v2_results": str(args.v2_results or default_results_path(args.v2_predictions)),
        "original_results": str(args.original_results or default_results_path(args.original_predictions)),
        "data_path": str(args.data_path),
        "sensor_blobs_path": str(args.sensor_blobs_path),
        "maps_root": str(args.maps_root),
        "note": (
            "Plots show raw NAVSIM ego-local pred_traj overlays plus GT. "
            "Plotted scores are re-scored from the exact prediction files above. "
            "Selection/full-navtest scores are retained only as case-selection metadata."
        ),
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    mismatch_count = int(manifest["selection_plot_mismatch"].sum()) if not manifest.empty else 0
    print(f"Wrote {len(manifest)} cases to {args.output_dir}")
    print(
        manifest[
            [
                "candidate_category",
                "token",
                "bev_nonblank",
                "camera_bev_nonblank",
                "selection_orig_PDMS",
                "plotted_orig_PDMS",
                "selection_plot_mismatch",
            ]
        ].to_string(index=False)
    )
    if mismatch_count:
        print(
            f"WARNING: {mismatch_count}/{len(manifest)} cases have selection/full-navtest scores "
            f"mismatching the plotted prediction re-score by more than {args.mismatch_warn_threshold}."
        )


if __name__ == "__main__":
    main()
