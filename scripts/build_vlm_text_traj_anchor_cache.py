#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import atomic_torch_save, iter_index  # noqa: E402
from navsim.agents.recogdrive.utils.internvl_preprocess import load_image  # noqa: E402
from navsim.common.dataloader import SceneLoader  # noqa: E402
from navsim.common.dataclasses import SensorConfig  # noqa: E402
from scripts.eval_vlm_direct_text_pdm import (  # noqa: E402
    build_prompt,
    configure_env,
    load_model,
    load_scene_filter,
    parse_pt_trajectory,
    torch_dtype,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed 2B-base direct-text trajectory anchor cache for Last-VLA residual diffusion.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--lora-adapter-dir", type=Path, default=None)
    parser.add_argument("--navsim-log-path", type=Path, required=True)
    parser.add_argument("--sensor-blobs-path", type=Path, required=True)
    parser.add_argument("--scene-filter-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-chunk-root", type=Path, default=None, help="Optional chunk cache root used only to restrict tokens.")
    parser.add_argument("--chunk-name-pattern", default="train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*,navtest_full_chunk_*")
    parser.add_argument("--tokens-file", type=Path, default=None, help="One token per line, or JSONL records with sample_token.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--cam-type", choices=("single", "multi_view", "cont"), default="single")
    parser.add_argument("--prompt-type", choices=("base", "vel_and_acc"), default="base")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--allow-tolerant-parse", action="store_true")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def chunk_dirs(root: Path, pattern: str) -> List[Path]:
    if (root / "index.jsonl").is_file():
        return [root]
    dirs: List[Path] = []
    seen = set()
    for item in str(pattern).split(","):
        item = item.strip()
        if not item:
            continue
        for path in sorted(root.glob(item)):
            if path.is_dir() and (path / "index.jsonl").is_file() and path not in seen:
                dirs.append(path)
                seen.add(path)
    if not dirs:
        raise FileNotFoundError(f"No chunk directories matching {pattern!r} under {root}")
    return dirs


def tokens_from_chunk_root(root: Path, pattern: str) -> Set[str]:
    tokens: Set[str] = set()
    for chunk_dir in chunk_dirs(root, pattern):
        for record in iter_index(chunk_dir):
            token = record.get("sample_token")
            if token is not None:
                tokens.add(str(token))
    return tokens


def tokens_from_file(path: Path) -> Set[str]:
    tokens: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                record = json.loads(line)
                token = record.get("sample_token") or record.get("token")
                if token is not None:
                    tokens.add(str(token))
            else:
                tokens.add(line)
    return tokens


def iter_shard(tokens: List[str], num_shards: int, shard_index: int, max_samples: Optional[int]) -> Iterable[str]:
    selected = [token for idx, token in enumerate(tokens) if idx % num_shards == shard_index]
    if max_samples is not None:
        selected = selected[:max_samples]
    yield from selected


def norm_odo(trajectory: torch.Tensor) -> torch.Tensor:
    x = 2 * (trajectory[..., 0:1] + 1.57) / 66.74 - 1
    y = 2 * (trajectory[..., 1:2] + 19.68) / 42 - 1
    heading = 2 * (trajectory[..., 2:3] + 1.67) / 3.53 - 1
    return torch.cat([x, y, heading], dim=-1)


def write_index(output_dir: Path, records: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = output_dir / ".index.jsonl.tmp"
    with tmp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    tmp_path.replace(output_dir / "index.jsonl")


def progress_payload(args: argparse.Namespace, total: int, written: int, parse_ok: int, failed: int) -> Dict[str, Any]:
    return {
        "overlay_type": "vlm_text_anchor",
        "anchor_type": "vlm_text_traj",
        "anchor_source": "2b_base_direct_text_pt",
        "model_path": str(args.model_path),
        "prompt_type": args.prompt_type,
        "cam_type": args.cam_type,
        "precision": args.precision,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "total_requested": total,
        "num_written": written,
        "num_parse_ok": parse_ok,
        "num_failed": failed,
        "parse_success_rate": parse_ok / written if written else None,
    }


def main() -> int:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards).")
    configure_env()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "samples").mkdir(parents=True, exist_ok=True)

    scene_filter = load_scene_filter(args.scene_filter_yaml)
    scene_loader = SceneLoader(
        sensor_blobs_path=args.sensor_blobs_path,
        data_path=args.navsim_log_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_all_sensors(include=[0, 1, 2, 3]),
        load_image_path=True,
    )
    token_set = set(str(token) for token in scene_loader.tokens)
    if args.base_chunk_root is not None:
        token_set &= tokens_from_chunk_root(args.base_chunk_root, args.chunk_name_pattern)
    if args.tokens_file is not None:
        token_set &= tokens_from_file(args.tokens_file)
    tokens = list(iter_shard(sorted(token_set), args.num_shards, args.shard_index, args.max_samples))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args, device)
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
    }
    generation_config = {key: value for key, value in generation_config.items() if value is not None}

    records: List[Dict[str, Any]] = []
    written = parse_ok_count = failed = 0
    for idx, token in enumerate(tokens):
        sample_path = args.output_dir / "samples" / f"{token}.pt"
        try:
            agent_input = scene_loader.get_agent_input_from_token(token)
            question, image_paths = build_prompt(agent_input, args.cam_type, args.prompt_type)
            pixel_values = [load_image(path, max_num=12) for path in image_paths]
            num_patches_list = [item.shape[0] for item in pixel_values]
            pixel_values_tensor = torch.cat(pixel_values, dim=0).to(device=device, dtype=torch_dtype(args.precision))
            with torch.no_grad():
                response = model.chat(
                    tokenizer,
                    pixel_values_tensor,
                    question,
                    generation_config=dict(generation_config),
                    num_patches_list=num_patches_list,
                )
            pred_np, parse_ok, parse_mode = parse_pt_trajectory(response, args.allow_tolerant_parse)
            traj = torch.from_numpy(pred_np).float()
            payload = {
                "sample_token": token,
                "scene_token": token,
                "vlm_text_trajectory": traj,
                "vlm_text_trajectory_norm": norm_odo(traj),
                "vlm_text_parse_ok": torch.tensor(float(parse_ok), dtype=torch.float32),
                "vlm_text_anchor_source": "2b_base_direct_text_pt",
                "vlm_text_parse_mode": parse_mode,
                "vlm_text_response": response,
                "vlm_text_model_path": str(args.model_path),
                "vlm_text_prompt_type": args.prompt_type,
                "vlm_text_cam_type": args.cam_type,
            }
            atomic_torch_save(payload, sample_path)
            parse_ok_count += int(parse_ok)
            written += 1
            records.append(
                {
                    "sample_token": token,
                    "scene_token": token,
                    "path": f"samples/{token}.pt",
                    "parse_ok": bool(parse_ok),
                    "parse_mode": parse_mode,
                }
            )
        except Exception as exc:
            failed += 1
            records.append(
                {
                    "sample_token": token,
                    "scene_token": token,
                    "path": "",
                    "parse_ok": False,
                    "parse_mode": "exception",
                    "error": repr(exc),
                }
            )
        if args.save_every > 0 and (idx + 1) % args.save_every == 0:
            write_index(args.output_dir, records)
        if args.progress_every > 0 and (idx + 1) % args.progress_every == 0:
            write_json_atomic(args.output_dir / "progress.json", progress_payload(args, len(tokens), written, parse_ok_count, failed))

    write_index(args.output_dir, records)
    metadata = progress_payload(args, len(tokens), written, parse_ok_count, failed)
    metadata["generation_config"] = generation_config
    metadata["allow_tolerant_parse"] = bool(args.allow_tolerant_parse)
    metadata["partial"] = False
    write_json_atomic(args.output_dir / "metadata.json", metadata)
    write_json_atomic(args.output_dir / "progress.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
