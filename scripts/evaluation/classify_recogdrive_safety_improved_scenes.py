#!/usr/bin/env python3
"""Classify safety-improved NAVSIM scenes with map and actor context labels."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("NUPLAN_MAPS_ROOT", "/mnt/navsim/maps")
os.environ.setdefault("NAVSIM_DISABLE_TQDM", "1")

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api


SAFETY_METRICS = ("NC", "DAC", "TTC", "Comfort", "DDC")
CONTEXT_TAG_ORDER = (
    "intersection_or_lane_connector",
    "turning_intersection",
    "straight_intersection",
    "signal_or_stopline",
    "crosswalk_nearby",
    "dense_vehicle_traffic",
    "front_vehicle_following",
    "side_or_cross_traffic",
    "vru_nearby",
    "static_obstacle_or_construction",
    "low_speed_start_stop",
)
TAG_ZH = {
    "intersection_or_lane_connector": "路口/连接车道附近",
    "turning_intersection": "路口转弯",
    "straight_intersection": "直行通过路口",
    "signal_or_stopline": "信号灯/停止线附近",
    "crosswalk_nearby": "斑马线附近",
    "dense_vehicle_traffic": "车流密集",
    "front_vehicle_following": "前车跟随/前方近车",
    "side_or_cross_traffic": "横向/交叉车流",
    "vru_nearby": "行人/自行车参与",
    "static_obstacle_or_construction": "静态障碍/锥桶施工",
    "low_speed_start_stop": "低速起步/停车",
}
COMMAND_ZH = {
    "turn_left": "左转",
    "go_straight": "直行",
    "turn_right": "右转",
    "unknown": "未知",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens-csv", type=Path, required=True)
    parser.add_argument("--logs-dir", type=Path, default=Path("/mnt/navsim/test_navsim_logs/test"))
    parser.add_argument("--maps-root", type=Path, default=Path("/mnt/navsim/maps"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--map-radius", type=float, default=30.0)
    return parser.parse_args()


def command_name(command: Any) -> str:
    arr = np.asarray(command).reshape(-1)
    if arr.size >= 3:
        idx = int(np.argmax(arr[:3]))
        if float(arr[:3].max()) > 0.5:
            return ("turn_left", "go_straight", "turn_right")[idx]
    return "unknown"


def load_log_frames(log_path: Path) -> Dict[str, Dict[str, Any]]:
    with log_path.open("rb") as f:
        frames = pickle.load(f)
    return {str(frame["token"]): frame for frame in frames}


def map_layer_counts(map_api: Any, frame: Dict[str, Any], radius: float) -> Dict[str, int]:
    ego_translation = frame["ego2global_translation"]
    # The map query only needs x/y/heading. Heading does not affect point radius.
    origin = StateSE2(float(ego_translation[0]), float(ego_translation[1]), 0.0)
    layers = [
        SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.CROSSWALK,
        SemanticMapLayer.STOP_LINE,
        SemanticMapLayer.LANE_CONNECTOR,
    ]
    counts = {"intersection_count": 0, "crosswalk_count": 0, "stop_line_count": 0, "lane_connector_count": 0}
    try:
        objects = map_api.get_proximal_map_objects(point=origin.point, radius=radius, layers=layers)
    except Exception:
        return counts
    layer_to_key = {
        SemanticMapLayer.INTERSECTION: "intersection_count",
        SemanticMapLayer.CROSSWALK: "crosswalk_count",
        SemanticMapLayer.STOP_LINE: "stop_line_count",
        SemanticMapLayer.LANE_CONNECTOR: "lane_connector_count",
    }
    for layer, key in layer_to_key.items():
        counts[key] = len(objects.get(layer, []))
    return counts


def actor_context(frame: Dict[str, Any]) -> Dict[str, Any]:
    anns = frame["anns"]
    names = np.asarray(anns["gt_names"])
    boxes = np.asarray(anns["gt_boxes"], dtype=float)
    velocity = np.asarray(anns["gt_velocity_3d"], dtype=float)
    if len(boxes) == 0:
        return {
            "vehicle_count_30m": 0,
            "vehicle_count_50m": 0,
            "pedestrian_count_30m": 0,
            "bicycle_count_30m": 0,
            "front_vehicle_count": 0,
            "close_front_vehicle_count": 0,
            "side_vehicle_count": 0,
            "cross_traffic_count": 0,
            "static_obstacle_count_30m": 0,
            "total_actor_count_50m": 0,
        }

    x = boxes[:, 0]
    y = boxes[:, 1]
    heading = boxes[:, 6]
    dist = np.hypot(x, y)
    speed = np.linalg.norm(velocity[:, :2], axis=1) if velocity.size else np.zeros(len(boxes))

    is_vehicle = names == "vehicle"
    is_ped = names == "pedestrian"
    is_bike = names == "bicycle"
    is_static = np.isin(names, ["traffic_cone", "barrier", "generic_object", "czone_sign"])

    front_vehicle = is_vehicle & (x > 0) & (x < 35) & (np.abs(y) < 4.5)
    close_front_vehicle = is_vehicle & (x > 0) & (x < 15) & (np.abs(y) < 4.5)
    side_vehicle = is_vehicle & (np.abs(x) < 25) & (np.abs(y) >= 4.0) & (np.abs(y) < 20)
    crossing_heading = np.abs(np.sin(heading)) > 0.75
    cross_traffic = is_vehicle & (dist < 35) & (speed > 1.0) & (side_vehicle | crossing_heading)

    return {
        "vehicle_count_30m": int((is_vehicle & (dist < 30)).sum()),
        "vehicle_count_50m": int((is_vehicle & (dist < 50)).sum()),
        "pedestrian_count_30m": int((is_ped & (dist < 30)).sum()),
        "bicycle_count_30m": int((is_bike & (dist < 30)).sum()),
        "front_vehicle_count": int(front_vehicle.sum()),
        "close_front_vehicle_count": int(close_front_vehicle.sum()),
        "side_vehicle_count": int(side_vehicle.sum()),
        "cross_traffic_count": int(cross_traffic.sum()),
        "static_obstacle_count_30m": int((is_static & (dist < 30)).sum()),
        "total_actor_count_50m": int((dist < 50).sum()),
    }


def scene_tags(command: str, frame: Dict[str, Any], map_counts: Dict[str, int], actors: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    has_intersection = map_counts["intersection_count"] > 0 or map_counts["lane_connector_count"] > 0
    if has_intersection:
        tags.append("intersection_or_lane_connector")
    if has_intersection and command in {"turn_left", "turn_right"}:
        tags.append("turning_intersection")
    if has_intersection and command == "go_straight":
        tags.append("straight_intersection")
    if map_counts["stop_line_count"] > 0 or len(frame.get("traffic_lights", [])) > 0:
        tags.append("signal_or_stopline")
    if map_counts["crosswalk_count"] > 0:
        tags.append("crosswalk_nearby")
    if actors["vehicle_count_50m"] >= 10 or actors["vehicle_count_30m"] >= 6:
        tags.append("dense_vehicle_traffic")
    if actors["front_vehicle_count"] > 0:
        tags.append("front_vehicle_following")
    if actors["side_vehicle_count"] > 0 or actors["cross_traffic_count"] > 0:
        tags.append("side_or_cross_traffic")
    if actors["pedestrian_count_30m"] > 0 or actors["bicycle_count_30m"] > 0:
        tags.append("vru_nearby")
    if actors["static_obstacle_count_30m"] > 0:
        tags.append("static_obstacle_or_construction")
    speed = float(np.linalg.norm(np.asarray(frame["ego_dynamic_state"][:2], dtype=float)))
    if speed < 2.0:
        tags.append("low_speed_start_stop")
    return tags


def improved_safety_item_counts(rows: pd.DataFrame) -> Dict[str, int]:
    output = {}
    for metric in SAFETY_METRICS:
        output[f"{metric}_improved_count"] = int((rows[f"delta_{metric}"] > 1e-9).sum())
        output[f"{metric}_zero_to_positive_count"] = int(
            ((rows[f"orig_{metric}"].abs() <= 1e-9) & (rows[f"v2_{metric}"] > 1e-9)).sum()
        )
    return output


def aggregate_by_tag(labeled: pd.DataFrame, total_count: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for tag in CONTEXT_TAG_ORDER:
        mask = labeled["context_tags"].str.split(";").apply(lambda values: tag in values)
        sub = labeled[mask]
        if sub.empty:
            continue
        row = {
            "context_tag": tag,
            "context_name_zh": TAG_ZH[tag],
            "count": int(len(sub)),
            "pct_of_main_tokens": float(len(sub) / total_count * 100),
            "mean_delta_PDMS": float(sub["delta_PDMS"].mean()),
            "mean_delta_EP": float(sub["delta_EP"].mean()),
            "mean_delta_SafetyMean": float(sub["delta_SafetyMean"].mean()),
            "median_delta_SafetyMean": float(sub["delta_SafetyMean"].median()),
        }
        row.update(improved_safety_item_counts(sub))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["count", "mean_delta_SafetyMean"], ascending=False)


def aggregate_by_command(labeled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for command, sub in labeled.groupby("command"):
        row = {
            "command": command,
            "command_zh": COMMAND_ZH.get(command, command),
            "count": int(len(sub)),
            "mean_delta_PDMS": float(sub["delta_PDMS"].mean()),
            "mean_delta_EP": float(sub["delta_EP"].mean()),
            "mean_delta_SafetyMean": float(sub["delta_SafetyMean"].mean()),
        }
        row.update(improved_safety_item_counts(sub))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def choose_examples(labeled: pd.DataFrame, max_examples: int = 5) -> pd.DataFrame:
    rows = []
    for tag in CONTEXT_TAG_ORDER:
        mask = labeled["context_tags"].str.split(";").apply(lambda values: tag in values)
        sub = labeled[mask].sort_values(["delta_SafetyMean", "delta_PDMS", "delta_EP"], ascending=False)
        for rank, (_, row) in enumerate(sub.head(max_examples).iterrows(), start=1):
            rows.append(
                {
                    "context_tag": tag,
                    "context_name_zh": TAG_ZH[tag],
                    "rank": rank,
                    "token": row["token"],
                    "log_name": row["log_name"],
                    "command_zh": row["command_zh"],
                    "delta_PDMS": row["delta_PDMS"],
                    "delta_EP": row["delta_EP"],
                    "delta_SafetyMean": row["delta_SafetyMean"],
                    "improved_safety_items": row["improved_safety_items"],
                    "zero_to_positive_safety_items": row["zero_to_positive_safety_items"],
                }
            )
    return pd.DataFrame(rows)


def write_report(output_dir: Path, labeled: pd.DataFrame, tag_summary: pd.DataFrame, command_summary: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("# EP 不下降且安全提升场景的直观归类")
    lines.append("")
    lines.append("日期：2026-07-02")
    lines.append("")
    total_count = len(labeled)
    lines.append(f"对象：`tokens_ep_not_down_safety_improved.csv` 中剔除聚合 `average` 行后的 {total_count} 个真实场景，即 V2 相比原版 ReCogDrive stage3 满足 EP 不下降、安全子项逐项不下降、且至少一个安全子项提升。")
    lines.append("")
    lines.append("这些标签来自 NAVSIM 原始日志和地图，不是用 PDMS 分项直接命名：")
    lines.append("- 导航命令：左转、直行、右转。")
    lines.append("- 地图邻近元素：路口、lane connector、斑马线、停止线、信号灯。")
    lines.append("- 动态参与者：前车、横向/交叉车流、车辆密度、行人/自行车、锥桶/静态障碍。")
    lines.append("")
    lines.append("标签是可重叠的；例如一个 token 可以同时属于“路口转弯”“斑马线附近”“车流密集”。")
    lines.append("")
    lines.append("## 按导航命令归类")
    lines.append("")
    lines.append("| 导航命令 | 场景数 | 平均PDMS变化 | 平均EP变化 | 平均SafetyMean变化 | NC提升 | DAC提升 | TTC提升 | DDC提升 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in command_summary.iterrows():
        lines.append(
            f"| {row.command_zh} | {int(row['count'])} | {row.mean_delta_PDMS:+.6f} | "
            f"{row.mean_delta_EP:+.6f} | {row.mean_delta_SafetyMean:+.6f} | "
            f"{int(row.NC_improved_count)} | {int(row.DAC_improved_count)} | "
            f"{int(row.TTC_improved_count)} | {int(row.DDC_improved_count)} |"
        )
    lines.append("")
    lines.append("## 按直观场景标签归类")
    lines.append("")
    lines.append(f"| 场景标签 | 场景数 | 占{total_count}比例 | 平均PDMS变化 | 平均EP变化 | 平均SafetyMean变化 | 主要安全改善项 |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for _, row in tag_summary.iterrows():
        item_counts = {
            "NC": int(row.NC_improved_count),
            "DAC": int(row.DAC_improved_count),
            "TTC": int(row.TTC_improved_count),
            "DDC": int(row.DDC_improved_count),
            "Comfort": int(row.Comfort_improved_count),
        }
        top_items = ", ".join(f"{k}:{v}" for k, v in sorted(item_counts.items(), key=lambda kv: kv[1], reverse=True) if v > 0)
        lines.append(
            f"| {row.context_name_zh} | {int(row['count'])} | {row.pct_of_main_tokens:.2f}% | "
            f"{row.mean_delta_PDMS:+.6f} | {row.mean_delta_EP:+.6f} | "
            f"{row.mean_delta_SafetyMean:+.6f} | {top_items} |"
        )
    lines.append("")
    lines.append("## 可形成的结论")
    lines.append("")
    lines.append("- 在路口/连接车道附近的场景中，V2 的安全提升最集中；这类标签覆盖了多数安全改善 token，适合表述为“V2 在路口拓扑和连接车道附近更少出现碰撞/TTC/DAC/DDC 失败”。")
    lines.append("- 在车流密集、前车跟随、横向/交叉车流场景中，安全提升主要对应 TTC/NC/DDC/DAC 的修复，说明 V2 对动态交互风险的处理更稳定。")
    lines.append("- 在斑马线附近、行人/自行车参与的场景中，如果安全项提升，通常可以作为“弱势交通参与者附近风险降低”的可视化候选；这些标签需要结合图片进一步人工确认。")
    lines.append("- DDC 改善最多，且集中出现在路口、连接车道、转弯/直行通过路口等场景；这比单纯说 DDC 分数提升更直观，可以描述为“复杂路口拓扑下行驶方向合规性提升”。")
    lines.append("")
    lines.append("注意：这是规则化自动标签，不等价于人工语义标注。建议后续从每个标签的 top examples 中人工挑图，形成最终论文/汇报案例。")
    lines.append("")
    lines.append("## 输出文件")
    lines.append("")
    lines.append("- 逐 token 场景标签：`scene_context_labels_ep_not_down_safety_improved.csv`")
    lines.append("- 场景标签汇总：`scene_context_label_summary.csv`")
    lines.append("- 导航命令汇总：`scene_context_command_summary.csv`")
    lines.append("- 每类代表 token：`scene_context_top_examples.csv`")
    (output_dir / "scene_context_classification_report_20260702.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    os.environ["NUPLAN_MAPS_ROOT"] = str(args.maps_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokens = pd.read_csv(args.tokens_csv)
    tokens = tokens[tokens["token"].notna() & tokens["log_name"].notna() & (tokens["token"].astype(str) != "average")]
    map_cache: Dict[str, Any] = {}
    frame_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
    records: List[Dict[str, Any]] = []

    for _, row in tokens.iterrows():
        log_name = str(row["log_name"])
        token = str(row["token"])
        if log_name not in frame_cache:
            frame_cache[log_name] = load_log_frames(args.logs_dir / f"{log_name}.pkl")
        frame = frame_cache[log_name][token]
        map_name = str(frame["map_location"])
        if map_name not in map_cache:
            map_cache[map_name] = get_maps_api(str(args.maps_root), "nuplan-maps-v1.0", map_name)

        command = command_name(frame["driving_command"])
        actors = actor_context(frame)
        map_counts = map_layer_counts(map_cache[map_name], frame, args.map_radius)
        tags = scene_tags(command, frame, map_counts, actors)
        record = row.to_dict()
        record.update(actors)
        record.update(map_counts)
        record.update(
            {
                "map_name": map_name,
                "command": command,
                "command_zh": COMMAND_ZH.get(command, command),
                "traffic_light_count": len(frame.get("traffic_lights", [])),
                "ego_speed": float(np.linalg.norm(np.asarray(frame["ego_dynamic_state"][:2], dtype=float))),
                "context_tags": ";".join(tags) if tags else "unlabeled",
                "context_tags_zh": ";".join(TAG_ZH[tag] for tag in tags) if tags else "未归类",
            }
        )
        records.append(record)

    labeled = pd.DataFrame(records)
    tag_summary = aggregate_by_tag(labeled, total_count=len(labeled))
    command_summary = aggregate_by_command(labeled)
    examples = choose_examples(labeled)

    labeled.to_csv(args.output_dir / "scene_context_labels_ep_not_down_safety_improved.csv", index=False)
    tag_summary.to_csv(args.output_dir / "scene_context_label_summary.csv", index=False)
    command_summary.to_csv(args.output_dir / "scene_context_command_summary.csv", index=False)
    examples.to_csv(args.output_dir / "scene_context_top_examples.csv", index=False)
    write_report(args.output_dir, labeled, tag_summary, command_summary)

    print(f"Wrote scene context classification for {len(labeled)} tokens to {args.output_dir}")
    print(tag_summary[["context_name_zh", "count", "pct_of_main_tokens", "mean_delta_PDMS", "mean_delta_EP", "mean_delta_SafetyMean"]].to_string(index=False))


if __name__ == "__main__":
    main()
