#!/usr/bin/env python3
"""Evaluate a Stage1 ReCogDrive VLM on the clip-held-out B2D trajectory set."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERNVL_ROOT = REPO_ROOT / "internvl_chat"
for path in (REPO_ROOT, INTERNVL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from internvl.train.dataset import (  # noqa: E402
    IGNORE_TOKEN_ID,
    build_transform,
    dynamic_preprocess,
    preprocess_internvl2_5,
)

IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
POINT_PATTERN = re.compile(
    rf"\(\s*({FLOAT_PATTERN})\s*,\s*({FLOAT_PATTERN})\s*,\s*({FLOAT_PATTERN})\s*\)"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--annotation",
        type=Path,
        default=REPO_ROOT / "outputs/bench2drive_recogdrive_stage1_sft_data_v1/val.jsonl",
    )
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/data/Bench2Drive-Base"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("nll", "generate"), default="nll")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-dynamic-patch", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--controls",
        nargs="*",
        choices=("shuffled_image", "shuffled_command", "shuffled_answer"),
        default=[],
        help="Optional teacher-forced counterfactuals; valid only in nll mode.",
    )
    parser.add_argument("--baseline-report", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--use-flash-attn", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def stable_digest(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def select_rows(rows: Sequence[Dict[str, Any]], max_samples: Optional[int], seed: int) -> List[Dict[str, Any]]:
    """Select deterministically while covering clips round-robin."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("clip", row.get("id", "unknown")))].append(row)
    for clip, items in groups.items():
        items.sort(key=lambda row: stable_digest(seed, str(row.get("id", ""))))
    if max_samples is None or max_samples >= len(rows):
        return [row for clip in sorted(groups) for row in groups[clip]]
    if max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    selected: List[Dict[str, Any]] = []
    depth = 0
    clips = sorted(groups, key=lambda clip: stable_digest(seed, clip))
    while len(selected) < max_samples:
        added = False
        for clip in clips:
            if depth < len(groups[clip]):
                selected.append(groups[clip][depth])
                added = True
                if len(selected) == max_samples:
                    break
        if not added:
            break
        depth += 1
    return selected


def init_distributed() -> Tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        if world_size != 1:
            raise RuntimeError("Distributed Stage1 evaluation requires CUDA")
        return rank, world_size, local_rank, torch.device("cpu")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def conversation_parts(row: Dict[str, Any]) -> Tuple[str, str, str]:
    system = next((str(turn["value"]) for turn in row["conversations"] if turn.get("from") == "system"), "")
    question = next(str(turn["value"]) for turn in row["conversations"] if turn.get("from") == "human")
    answer = next(str(turn["value"]) for turn in row["conversations"] if turn.get("from") == "gpt")
    return system, question, answer


def shifted_command(question: str) -> str:
    replacements = {
        "[TURN LEFT]": "[TURN RIGHT]",
        "[TURN RIGHT]": "[GO STRAIGHT]",
        "[GO STRAIGHT]": "[TURN LEFT]",
    }
    for source, target in replacements.items():
        if source in question:
            return question.replace(source, target, 1)
    return question.replace("[UNKNOWN]", "[TURN LEFT]", 1)


def counterfactual_pairs(rows: Sequence[Dict[str, Any]], seed: int) -> Dict[str, Dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: stable_digest(seed + 1, str(row.get("id", ""))))
    result: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(ordered):
        candidate_index = (index + max(1, len(ordered) // 2)) % len(ordered)
        for offset in range(len(ordered)):
            candidate = ordered[(candidate_index + offset) % len(ordered)]
            if candidate.get("clip") != row.get("clip"):
                result[str(row["id"])] = candidate
                break
        else:
            result[str(row["id"])] = ordered[candidate_index]
    return result


def image_tensors(
    image_path: Path,
    *,
    max_dynamic_patch: int,
    transform: Any,
) -> torch.Tensor:
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    image = Image.open(image_path).convert("RGB")
    tiles = dynamic_preprocess(
        image,
        min_num=1,
        max_num=max_dynamic_patch,
        image_size=448,
        use_thumbnail=True,
    )
    return torch.stack([transform(tile) for tile in tiles])


def prepare_teacher_forced(
    row: Dict[str, Any],
    *,
    image_row: Dict[str, Any],
    answer_row: Dict[str, Any],
    shift_command: bool,
    tokenizer: Any,
    num_image_token: int,
    data_root: Path,
    max_dynamic_patch: int,
    transform: Any,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    system, question, _ = conversation_parts(row)
    _, _, answer = conversation_parts(answer_row)
    if shift_command:
        question = shifted_command(question)
    conversations = [
        {"from": "system", "value": system},
        {"from": "human", "value": question},
        {"from": "gpt", "value": answer},
    ]
    pixels = image_tensors(
        data_root / str(image_row["image"]),
        max_dynamic_patch=max_dynamic_patch,
        transform=transform,
    )
    processed = preprocess_internvl2_5(
        "internvl2_5",
        [copy.deepcopy(conversations)],
        tokenizer,
        [num_image_token * int(pixels.shape[0])],
        group_by_length=True,
        ds_name="Bench2Drive_Stage1_Eval",
    )
    return pixels, processed


@torch.inference_mode()
def sample_nll(
    model: Any,
    tokenizer: Any,
    row: Dict[str, Any],
    *,
    image_row: Dict[str, Any],
    answer_row: Dict[str, Any],
    shift_command: bool,
    data_root: Path,
    max_dynamic_patch: int,
    transform: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[float, int, int]:
    pixels, processed = prepare_teacher_forced(
        row,
        image_row=image_row,
        answer_row=answer_row,
        shift_command=shift_command,
        tokenizer=tokenizer,
        num_image_token=int(model.num_image_token),
        data_root=data_root,
        max_dynamic_patch=max_dynamic_patch,
        transform=transform,
    )
    input_ids = processed["input_ids"].to(device)
    attention_mask = processed["attention_mask"].to(device)
    labels = processed["labels"].to(device)
    token_count = int((labels != IGNORE_TOKEN_ID).sum().item())
    image_flags = torch.ones(pixels.shape[0], dtype=torch.long, device=device)
    outputs = model(
        pixel_values=pixels.to(device=device, dtype=dtype),
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        image_flags=image_flags,
        return_dict=True,
    )
    return float(outputs.loss.float().item()), token_count, int(pixels.shape[0])


def parse_trajectory(text: str) -> Optional[np.ndarray]:
    marker = text.find("[PT")
    candidate = text[marker:] if marker >= 0 else text
    points = POINT_PATTERN.findall(candidate)
    if len(points) != 8:
        return None
    trajectory = np.asarray([[float(value) for value in point] for point in points], dtype=np.float64)
    return trajectory if np.isfinite(trajectory).all() else None


def command_name(question: str) -> str:
    match = re.search(r"Active navigation command: \[([^\]]+)\]", question)
    return match.group(1).lower().replace(" ", "_") if match else "unknown"


@torch.inference_mode()
def generate_row(
    model: Any,
    tokenizer: Any,
    row: Dict[str, Any],
    *,
    data_root: Path,
    max_dynamic_patch: int,
    max_new_tokens: int,
    transform: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    system, question, answer = conversation_parts(row)
    pixels = image_tensors(
        data_root / str(row["image"]),
        max_dynamic_patch=max_dynamic_patch,
        transform=transform,
    ).to(device=device, dtype=dtype)
    model.system_message = system
    response = model.chat(
        tokenizer=tokenizer,
        pixel_values=pixels,
        question=question,
        generation_config={
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
        },
        num_patches_list=[int(pixels.shape[0])],
        history=None,
        return_history=False,
    )
    target = parse_trajectory(answer)
    prediction = parse_trajectory(response)
    result: Dict[str, Any] = {
        "id": str(row["id"]),
        "clip": str(row.get("clip", "")),
        "command": command_name(question),
        "parse_success": prediction is not None,
        "response": response,
        "num_image_patches": int(pixels.shape[0]),
    }
    if prediction is not None and target is not None:
        difference = prediction - target
        heading = np.arctan2(np.sin(difference[:, 2]), np.cos(difference[:, 2]))
        xy = np.linalg.norm(difference[:, :2], axis=-1)
        result.update(
            {
                "xy_ade": float(xy.mean()),
                "xy_fde": float(xy[-1]),
                "heading_mae": float(np.abs(heading).mean()),
                "raw_l1": float(np.abs(np.column_stack([difference[:, :2], heading])).mean()),
            }
        )
    return result


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else math.nan


def summarize_nll(rows: Sequence[Dict[str, Any]], controls: Sequence[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"examples": len(rows)}
    names = ["real", *controls]
    for name in names:
        key = f"{name}_nll"
        tokens_key = f"{name}_tokens"
        weighted = sum(float(row[key]) * int(row[tokens_key]) for row in rows)
        tokens = sum(int(row[tokens_key]) for row in rows)
        nll = weighted / tokens
        summary[f"{name}_token_nll"] = nll
        summary[f"{name}_perplexity"] = math.exp(min(20.0, nll))
        summary[f"{name}_sample_nll"] = safe_mean(float(row[key]) for row in rows)
        summary[f"{name}_tokens"] = tokens
        if name != "real":
            summary[f"{name}_minus_real_nll"] = nll - float(summary["real_token_nll"])
    return summary


def metric_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    parsed = [row for row in rows if row["parse_success"]]
    summary: Dict[str, Any] = {
        "examples": len(rows),
        "parsed": len(parsed),
        "parse_success_rate": len(parsed) / len(rows) if rows else math.nan,
    }
    for metric in ("xy_ade", "xy_fde", "heading_mae", "raw_l1"):
        summary[metric] = safe_mean(float(row[metric]) for row in parsed if metric in row)
    by_command: Dict[str, Any] = {}
    for command in sorted({str(row["command"]) for row in rows}):
        subset = [row for row in rows if row["command"] == command]
        by_command[command] = metric_summary_without_groups(subset)
    summary["by_command"] = by_command
    return summary


def metric_summary_without_groups(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    parsed = [row for row in rows if row["parse_success"]]
    result = {
        "examples": len(rows),
        "parse_success_rate": len(parsed) / len(rows) if rows else math.nan,
    }
    for metric in ("xy_ade", "xy_fde", "heading_mae", "raw_l1"):
        result[metric] = safe_mean(float(row[metric]) for row in parsed if metric in row)
    return result


def comparison(current: Dict[str, Any], baseline: Dict[str, Any], mode: str) -> Dict[str, Any]:
    if mode == "nll":
        key = "real_token_nll"
        base = float(baseline["summary"][key])
        value = float(current["summary"][key])
        return {
            "baseline_real_token_nll": base,
            "current_real_token_nll": value,
            "absolute_nll_change": value - base,
            "relative_nll_reduction": (base - value) / base,
        }
    result: Dict[str, Any] = {}
    for key in ("parse_success_rate", "xy_ade", "xy_fde", "heading_mae", "raw_l1"):
        base = float(baseline["summary"][key])
        value = float(current["summary"][key])
        result[f"baseline_{key}"] = base
        result[f"current_{key}"] = value
        result[f"relative_{key}_change"] = (value - base) / base if base else math.nan
    return result


def main() -> int:
    args = parse_args()
    if args.mode != "nll" and args.controls:
        raise ValueError("--controls are supported only with --mode nll")
    rank, world_size, _, device = init_distributed()
    dtype = dtype_from_name(args.dtype)
    rows = select_rows(load_rows(args.annotation), args.max_samples, args.seed)
    row_index = {str(row["id"]): index for index, row in enumerate(rows)}
    paired = counterfactual_pairs(rows, args.seed)
    local_rows = rows[rank::world_size]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    tokenizer.model_max_length = args.max_length
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=args.use_flash_attn,
        local_files_only=True,
    ).to(device).eval()
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    transform = build_transform(is_train=False, input_size=448, normalize_type="imagenet")

    started = time.time()
    local_results: List[Dict[str, Any]] = []
    for local_index, row in enumerate(local_rows):
        if args.mode == "nll":
            real_nll, real_tokens, patches = sample_nll(
                model,
                tokenizer,
                row,
                image_row=row,
                answer_row=row,
                shift_command=False,
                data_root=args.data_root,
                max_dynamic_patch=args.max_dynamic_patch,
                transform=transform,
                device=device,
                dtype=dtype,
            )
            result: Dict[str, Any] = {
                "id": str(row["id"]),
                "clip": str(row.get("clip", "")),
                "command": command_name(conversation_parts(row)[1]),
                "real_nll": real_nll,
                "real_tokens": real_tokens,
                "num_image_patches": patches,
            }
            other = paired[str(row["id"])]
            for control in args.controls:
                control_nll, control_tokens, _ = sample_nll(
                    model,
                    tokenizer,
                    row,
                    image_row=other if control == "shuffled_image" else row,
                    answer_row=other if control == "shuffled_answer" else row,
                    shift_command=control == "shuffled_command",
                    data_root=args.data_root,
                    max_dynamic_patch=args.max_dynamic_patch,
                    transform=transform,
                    device=device,
                    dtype=dtype,
                )
                result[f"{control}_nll"] = control_nll
                result[f"{control}_tokens"] = control_tokens
            local_results.append(result)
        else:
            local_results.append(
                generate_row(
                    model,
                    tokenizer,
                    row,
                    data_root=args.data_root,
                    max_dynamic_patch=args.max_dynamic_patch,
                    max_new_tokens=args.max_new_tokens,
                    transform=transform,
                    device=device,
                    dtype=dtype,
                )
            )
        if args.log_every and (local_index + 1) % args.log_every == 0:
            print(f"rank={rank} completed={local_index + 1}/{len(local_rows)}", flush=True)

    gathered: List[Any] = [None for _ in range(world_size)]
    if world_size > 1:
        dist.all_gather_object(gathered, local_results)
    else:
        gathered = [local_results]
    if rank == 0:
        results = [item for shard in gathered for item in shard]
        results.sort(key=lambda item: row_index[item["id"]])
        report: Dict[str, Any] = {
            "model_path": str(args.model_path.resolve()),
            "annotation": str(args.annotation.resolve()),
            "data_root": str(args.data_root.resolve()),
            "mode": args.mode,
            "world_size": world_size,
            "seed": args.seed,
            "selected_examples": len(rows),
            "controls": list(args.controls),
            "elapsed_seconds": time.time() - started,
            "summary": summarize_nll(results, args.controls) if args.mode == "nll" else metric_summary(results),
            "rows": results,
        }
        if args.baseline_report:
            baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
            report["comparison_to_baseline"] = comparison(report, baseline, args.mode)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report["summary"], indent=2, sort_keys=True), flush=True)
        if "comparison_to_baseline" in report:
            print(json.dumps(report["comparison_to_baseline"], indent=2, sort_keys=True), flush=True)
        print(f"wrote {args.output}", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
