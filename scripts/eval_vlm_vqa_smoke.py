#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.utils.internvl_preprocess import load_image  # noqa: E402
from navsim.common.dataloader import SceneLoader  # noqa: E402
from navsim.common.dataclasses import SceneFilter, SensorConfig  # noqa: E402
from navsim.common.enums import BoundingBoxIndex  # noqa: E402


SYSTEM_MESSAGE = (
    "You are an autonomous-driving vision-language assistant. "
    "Answer questions about the driving scene using the provided camera image. "
    "Be concise and do not output a trajectory unless the question explicitly asks for one."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small NAVSIM VQA smoke test for VLM/LoRA scene understanding.")
    parser.add_argument("--model-path", type=Path, default=Path("/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B"))
    parser.add_argument("--lora-adapter-dir", type=Path, default=None)
    parser.add_argument("--navsim-log-path", type=Path, default=Path("/mnt/navsim/test_navsim_logs/test"))
    parser.add_argument("--sensor-blobs-path", type=Path, default=Path("/mnt/navsim/test_sensor_blobs/test"))
    parser.add_argument("--scene-filter-yaml", type=Path, default=Path("navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int, default=400)
    parser.add_argument("--num-scene-samples", type=int, default=24)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def configure_env() -> None:
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    os.environ.setdefault("NUPLAN_MAPS_ROOT", "/mnt/navsim/maps")
    os.environ.setdefault("OPENSCENE_DATA_ROOT", "/mnt/navsim")


def torch_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def load_scene_filter(path: Path, max_scenes: int) -> SceneFilter:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    data["max_scenes"] = max_scenes
    return SceneFilter(**data)


def load_model(args: argparse.Namespace, device: torch.device) -> Tuple[Any, Any]:
    dtype = torch_dtype(args.precision) if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
    )
    model = model.eval().to(device)
    if args.lora_adapter_dir is not None:
        from peft import LoraConfig, get_peft_model

        from navsim.agents.recogdrive.vlm_lora_utils import (
            load_lora_adapter_metadata,
            peft_lora_config_kwargs_supported,
        )

        lora_config = load_lora_adapter_metadata(args.lora_adapter_dir)
        supported_kwargs = peft_lora_config_kwargs_supported()
        lora_kwargs: Dict[str, Any] = {
            "r": int(lora_config["r"]),
            "lora_alpha": int(lora_config["alpha"]),
            "lora_dropout": float(lora_config["dropout"]),
            "target_modules": lora_config["target_modules"],
            "bias": str(lora_config["bias"]),
        }
        if "use_rslora" in supported_kwargs:
            lora_kwargs["use_rslora"] = bool(lora_config.get("use_rslora", False))
        if "use_dora" in supported_kwargs:
            lora_kwargs["use_dora"] = bool(lora_config.get("use_dora", False))
        model = get_peft_model(model, LoraConfig(**lora_kwargs))
        state_path = args.lora_adapter_dir / "adapter_model.bin"
        state = torch.load(state_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model_state = model.state_dict()
        filtered: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            mapped_key = key[len("agent.") :] if isinstance(key, str) and key.startswith("agent.") else key
            if isinstance(mapped_key, str) and mapped_key.startswith("backbone.model."):
                mapped_key = mapped_key[len("backbone.model.") :]
            elif isinstance(mapped_key, str) and mapped_key.startswith("backbone."):
                mapped_key = mapped_key[len("backbone.") :]
            if (
                isinstance(mapped_key, str)
                and "lora_" in mapped_key
                and isinstance(value, torch.Tensor)
                and mapped_key in model_state
                and tuple(model_state[mapped_key].shape) == tuple(value.shape)
            ):
                filtered[mapped_key] = value
        if not filtered:
            raise RuntimeError(f"No compatible VLM LoRA tensors found in {state_path}")
        model.load_state_dict(filtered, strict=False)
        print(f"Loaded {len(filtered)} VLM LoRA tensors from {state_path}.", flush=True)
        model = model.merge_and_unload().eval().to(device)
    if hasattr(model, "system_message"):
        model.system_message = SYSTEM_MESSAGE
    return model, tokenizer


def front_objects(frame: Dict[str, Any], max_distance: float = 40.0, fov_deg: float = 65.0) -> List[Dict[str, Any]]:
    anns = frame["anns"]
    objects: List[Dict[str, Any]] = []
    for box, name, vel in zip(anns["gt_boxes"], anns["gt_names"], anns["gt_velocity_3d"]):
        label = str(name).lower()
        if label not in {"vehicle", "pedestrian", "bicycle"}:
            continue
        x = float(box[BoundingBoxIndex.X])
        y = float(box[BoundingBoxIndex.Y])
        if x <= 0:
            continue
        distance = math.hypot(x, y)
        angle = abs(math.degrees(math.atan2(y, x)))
        if distance <= max_distance and angle <= fov_deg / 2:
            objects.append(
                {
                    "label": label,
                    "x": x,
                    "y": y,
                    "distance": distance,
                    "speed": math.hypot(float(vel[0]), float(vel[1])),
                }
            )
    objects.sort(key=lambda item: item["distance"])
    return objects


def yes_no_from_answer(text: str) -> Optional[str]:
    lowered = text.strip().lower()
    if re.search(r"\b(yes|yeah|yep)\b", lowered):
        return "yes"
    if re.search(r"\b(no|not|none|cannot see|don't see|do not see)\b", lowered):
        return "no"
    return None


def nearest_label_from_answer(text: str) -> str:
    lowered = text.strip().lower()
    hits = []
    for label in ("pedestrian", "bicycle", "vehicle", "none"):
        if label in lowered:
            hits.append(label)
    if "car" in lowered or "truck" in lowered or "bus" in lowered or "van" in lowered:
        hits.append("vehicle")
    return hits[0] if hits else ""


def command_from_answer(text: str) -> str:
    lowered = text.strip().lower()
    if "left" in lowered:
        return "turn left"
    if "right" in lowered:
        return "turn right"
    if "straight" in lowered or "forward" in lowered:
        return "go straight"
    return ""


def score_answer(task: str, response: str, expected: str) -> bool:
    if task in {"vehicle_yesno", "vru_yesno"}:
        return yes_no_from_answer(response) == expected
    if task == "nearest_object":
        return nearest_label_from_answer(response) == expected
    if task == "command":
        return command_from_answer(response) == expected
    return False


def build_vqa_items(loader: SceneLoader, num_scene_samples: int, seed: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    buckets: Dict[str, List[str]] = defaultdict(list)
    token_meta: Dict[str, Dict[str, Any]] = {}
    commands = ["turn left", "go straight", "turn right"]

    for token in loader.tokens:
        frame = loader.scene_frames_dicts[token][loader._scene_filter.num_history_frames - 1]
        objects = front_objects(frame)
        counts = Counter(obj["label"] for obj in objects)
        command_vec = frame["driving_command"]
        command_idx = int(np.argmax(command_vec)) if len(command_vec) else 1
        command = commands[command_idx] if 0 <= command_idx < len(commands) else "unknown"
        nearest = objects[0]["label"] if objects else "none"
        meta = {
            "objects": objects,
            "counts": dict(counts),
            "has_vehicle": counts["vehicle"] > 0,
            "has_vru": counts["pedestrian"] > 0 or counts["bicycle"] > 0,
            "nearest": nearest,
            "command": command,
        }
        token_meta[token] = meta
        buckets["vru_yes" if meta["has_vru"] else "vru_no"].append(token)
        buckets["vehicle_yes" if meta["has_vehicle"] else "vehicle_no"].append(token)
        buckets[f"nearest_{nearest}"].append(token)

    selected: List[str] = []
    for bucket in ("vru_yes", "vru_no", "vehicle_yes", "vehicle_no", "nearest_vehicle", "nearest_pedestrian", "nearest_bicycle", "nearest_none"):
        tokens = buckets.get(bucket, [])
        if tokens:
            take = min(4, len(tokens))
            chosen = list(rng.choice(tokens, size=take, replace=False))
            selected.extend(chosen)
    if len(selected) < num_scene_samples:
        remaining = [token for token in loader.tokens if token not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[: num_scene_samples - len(selected)])
    selected = list(dict.fromkeys(selected))[:num_scene_samples]

    items: List[Dict[str, Any]] = []
    for token in selected:
        agent_input = loader.get_agent_input_from_token(token)
        image_path = str(agent_input.cameras[-1].cam_f0.image)
        meta = token_meta[token]
        items.extend(
            [
                {
                    "id": f"{token}_vehicle_yesno",
                    "token": token,
                    "task": "vehicle_yesno",
                    "image_path": image_path,
                    "question": "<image>\nAre there any vehicles visible ahead of the ego vehicle? Answer yes or no.",
                    "expected": "yes" if meta["has_vehicle"] else "no",
                    "metadata": meta,
                },
                {
                    "id": f"{token}_vru_yesno",
                    "token": token,
                    "task": "vru_yesno",
                    "image_path": image_path,
                    "question": "<image>\nAre there any vulnerable road users visible ahead, such as pedestrians or bicycles? Answer yes or no.",
                    "expected": "yes" if meta["has_vru"] else "no",
                    "metadata": meta,
                },
                {
                    "id": f"{token}_nearest_object",
                    "token": token,
                    "task": "nearest_object",
                    "image_path": image_path,
                    "question": "<image>\nWhat is the closest relevant object ahead? Answer exactly one of: vehicle, pedestrian, bicycle, none.",
                    "expected": meta["nearest"],
                    "metadata": meta,
                },
                {
                    "id": f"{token}_command",
                    "token": token,
                    "task": "command",
                    "image_path": image_path,
                    "question": (
                        "<image>\nThe navigation command for this scene is one of: turn left, go straight, turn right. "
                        f"The command signal is {meta['command']}. Restate the command exactly."
                    ),
                    "expected": meta["command"],
                    "metadata": meta,
                },
            ]
        )
    return items


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    task_metrics = {}
    for task, task_rows in sorted(by_task.items()):
        task_metrics[task] = {
            "num_samples": len(task_rows),
            "accuracy": sum(int(row["correct"]) for row in task_rows) / len(task_rows) if task_rows else None,
        }
    trajectory_like = sum("[PT" in row["response"] or "trajectory" in row["response"].lower() for row in rows)
    return {
        "num_samples": len(rows),
        "accuracy": sum(int(row["correct"]) for row in rows) / len(rows) if rows else None,
        "task_metrics": task_metrics,
        "trajectory_like_rate": trajectory_like / len(rows) if rows else None,
    }


def main() -> int:
    args = parse_args()
    configure_env()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene_filter = load_scene_filter(args.scene_filter_yaml, args.max_scenes)
    loader = SceneLoader(
        sensor_blobs_path=args.sensor_blobs_path,
        data_path=args.navsim_log_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_all_sensors(include=[0, 1, 2, 3]),
        load_image_path=True,
    )
    items = build_vqa_items(loader, args.num_scene_samples, args.seed)
    (args.output_dir / "vqa_items.json").write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args, device)
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
    }
    generation_config = {key: value for key, value in generation_config.items() if value is not None}

    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        pixel_values = load_image(item["image_path"], max_num=12).to(device=device, dtype=torch_dtype(args.precision))
        with torch.no_grad():
            response = model.chat(
                tokenizer,
                pixel_values,
                item["question"],
                generation_config=dict(generation_config),
                num_patches_list=[pixel_values.shape[0]],
            )
        correct = score_answer(item["task"], response, item["expected"])
        row = {
            **{key: item[key] for key in ("id", "token", "task", "image_path", "question", "expected")},
            "response": response,
            "correct": correct,
        }
        rows.append(row)
        metrics = summarize(rows)
        progress = {"partial": True, "done": idx, "total": len(items), **metrics}
        (args.output_dir / "progress.json").write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"done": idx, "total": len(items), "task": item["task"], "expected": item["expected"], "response": response, "correct": correct}, ensure_ascii=True), flush=True)

    metrics = {"partial": False, "model_path": str(args.model_path), "lora_adapter_dir": str(args.lora_adapter_dir) if args.lora_adapter_dir else None, **summarize(rows)}
    (args.output_dir / "predictions.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
