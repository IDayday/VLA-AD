#!/usr/bin/env python3
"""Visualize V2 Pareto-GRPO trajectories against original ReCogDrive stage3."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=Path("/mnt/navsim/test_navsim_logs/test"))
    parser.add_argument("--sensor-blobs-path", type=Path, default=Path("/mnt/navsim/test_sensor_blobs/test"))
    parser.add_argument("--maps-root", type=Path, default=Path("/mnt/navsim/maps"))
    parser.add_argument("--num-history-frames", type=int, default=4)
    parser.add_argument("--num-future-frames", type=int, default=10)
    parser.add_argument("--frame-interval", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=170)
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


def trajectory_from_record(record: Dict[str, Any], key: str) -> Trajectory:
    poses = np.asarray(record[key], dtype=np.float32)
    return Trajectory(poses=poses)


def metric_text(row: pd.Series) -> str:
    return (
        f"Full navtest: V2 {row.v2_PDMS:.3f} vs Orig {row.orig_PDMS:.3f} "
        f"(d={row.v2_minus_orig_PDMS:+.3f})\n"
        f"Online traj: V2 {row.v2_online_PDMS:.3f} vs Orig {row.orig_online_PDMS:.3f} "
        f"(d={row.online_delta_PDMS:+.3f}); "
        f"EP {row.v2_online_EP:.3f}/{row.orig_online_EP:.3f}; "
        f"NC {row.v2_online_NC:.1f}/{row.orig_online_NC:.1f}; "
        f"DAC {row.v2_online_DAC:.1f}/{row.orig_online_DAC:.1f}; "
        f"TTC {row.v2_online_TTC:.1f}/{row.orig_online_TTC:.1f}; "
        f"DDC {row.v2_online_DDC:.1f}/{row.orig_online_DDC:.1f}"
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
    scene_loader = build_scene_loader(rows, args)

    manifest_rows: List[Dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        token = str(row.token)
        if token not in v2_predictions or token not in original_predictions:
            raise KeyError(f"Missing predictions for token {token}")
        if token not in scene_loader.tokens:
            raise KeyError(f"Scene token {token} was not loaded from {args.data_path}")

        scene = scene_loader.get_scene_from_token(token)
        v2_traj = trajectory_from_record(v2_predictions[token], "pred_traj")
        original_traj = trajectory_from_record(original_predictions[token], "pred_traj")
        gt_traj = trajectory_from_record(v2_predictions[token], "gt_traj")

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
                "v2_online_PDMS": row.v2_online_PDMS,
                "orig_online_PDMS": row.orig_online_PDMS,
                "online_delta_PDMS": row.online_delta_PDMS,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_dir / "manifest.csv", index=False)
    print(f"Wrote {len(manifest)} cases to {args.output_dir}")
    print(manifest[["candidate_category", "token", "bev_nonblank", "camera_bev_nonblank"]].to_string(index=False))


if __name__ == "__main__":
    main()
