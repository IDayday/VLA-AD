#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import atomic_torch_save, iter_index, load_sample, write_json  # noqa: E402


PRESERVE_KEYS = (
    "history_trajectory",
    "high_command_one_hot",
    "status_feature",
    "trajectory",
    "image_path_tensor",
    "sample_token",
    "scene_token",
    "log_name",
    "jepa_context_tokens",
    "jepa_target_tokens",
    "vggt_context_tokens",
    "vggt_target_tokens",
    "vggt_geometry_tokens",
    "vggt_geometry_target_tokens",
    "vggt_depth_tokens",
    "vggt_pointmap_tokens",
    "vggt_camera_tokens",
    "vggt_geometry_mode_code",
    "teacher_trajectory",
    "teacher_trajectory_norm",
    "teacher_score",
    "gt_score",
    "oracle_best_of_k_score",
    "candidate_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate ReCogDrive hidden-state chunk cache with a VLM LoRA adapter.")
    parser.add_argument("--base-chunk-root", type=Path, required=True)
    parser.add_argument("--output-chunk-root", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--vlm-type", default="internvl")
    parser.add_argument("--vlm-lora-adapter", type=Path, required=True)
    parser.add_argument("--chunk-name-pattern", default="train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--synthetic-smoke", action="store_true")
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
        raise FileNotFoundError(f"No chunk dirs matching {pattern!r} under {root}")
    return dirs


def resolve_sample_path(chunk_dir: Path, record: Dict[str, Any]) -> Path:
    raw = Path(record["path"])
    if raw.is_file():
        return raw
    return raw if raw.is_absolute() else chunk_dir / raw


def iter_samples(root: Path, pattern: str, max_samples: Optional[int]) -> Iterable[Tuple[Path, Path, Dict[str, Any]]]:
    count = 0
    for chunk_dir in chunk_dirs(root, pattern):
        for record in iter_index(chunk_dir):
            yield chunk_dir, resolve_sample_path(chunk_dir, record), record
            count += 1
            if max_samples is not None and count >= max_samples:
                return


def decode_path_tensor(path_tensor: torch.Tensor) -> str:
    chars = []
    for item in path_tensor.detach().cpu().view(-1):
        value = int(item.item())
        if value == 0:
            break
        chars.append(chr(value))
    return "".join(chars)


def synthetic_hidden(sample: Dict[str, Any]) -> torch.Tensor:
    source = sample.get("last_hidden_state")
    if isinstance(source, torch.Tensor) and source.ndim == 2:
        return (source.float() * 0.0 + 0.123).contiguous()
    return torch.full((10, 1536), 0.123, dtype=torch.float32)


def normalize_hidden_state(hidden: torch.Tensor) -> torch.Tensor:
    hidden = hidden.detach().float().cpu()
    if hidden.ndim == 3 and hidden.shape[0] == 1:
        hidden = hidden.squeeze(0)
    if hidden.ndim != 2 or hidden.shape[-1] != 1536:
        raise ValueError(f"Regenerated VLM hidden state must have shape [N, 1536], got {tuple(hidden.shape)}.")
    return hidden.contiguous()


def load_backbone(args: argparse.Namespace):
    from navsim.agents.recogdrive.recogdrive_backbone import RecogDriveBackbone

    backbone = RecogDriveBackbone(model_type=args.vlm_type, checkpoint_path=str(args.vlm_path), device=args.device)
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("Regenerating hidden cache with LoRA requires peft. Install it with `pip install peft`.") from exc

    target_modules = [item.strip() for item in str(args.lora_target_modules).split(",") if item.strip()]
    if not target_modules:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_cfg = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        target_modules=target_modules,
        bias="none",
    )
    peft_vlm = get_peft_model(backbone.model, lora_cfg)
    for attr in ("img_context_token_id", "system_message"):
        if hasattr(backbone.model, attr) and not hasattr(peft_vlm, attr):
            setattr(peft_vlm, attr, getattr(backbone.model, attr))

    try:
        payload = torch.load(args.vlm_lora_adapter, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(args.vlm_lora_adapter, map_location="cpu")
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"LoRA adapter {args.vlm_lora_adapter} must contain a state_dict or raw state dict.")

    model_state = peft_vlm.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for key, value in state.items():
        mapped_key = key[len("agent."):] if isinstance(key, str) and key.startswith("agent.") else key
        if isinstance(mapped_key, str) and mapped_key.startswith("backbone.model."):
            mapped_key = mapped_key[len("backbone.model."):]
        elif isinstance(mapped_key, str) and mapped_key.startswith("backbone."):
            mapped_key = mapped_key[len("backbone."):]
        if (
            isinstance(mapped_key, str)
            and "lora_" in mapped_key
            and isinstance(value, torch.Tensor)
            and mapped_key in model_state
            and tuple(model_state[mapped_key].shape) == tuple(value.shape)
        ):
            filtered[mapped_key] = value
        else:
            skipped.append(str(key))
    if not filtered:
        raise RuntimeError(f"No compatible VLM LoRA weights found in {args.vlm_lora_adapter}.")
    peft_vlm.load_state_dict(filtered, strict=False)
    print(f"Loaded {len(filtered)} VLM LoRA tensors from {args.vlm_lora_adapter}.")
    if skipped:
        print(f"Skipped {len(skipped)} incompatible/non-LoRA adapter tensors.")
    backbone._patch_internvl_visual_feature_dtype(backbone.model)
    backbone.model = peft_vlm
    backbone.eval()
    return backbone


def compute_hidden(backbone: Any, sample: Dict[str, Any], *, device: str) -> torch.Tensor:
    from navsim.agents.recogdrive.recogdrive_agent import format_number
    from navsim.agents.recogdrive.utils.internvl_preprocess import load_image

    if not isinstance(sample.get("image_path_tensor"), torch.Tensor):
        raise KeyError("LoRA hidden regeneration requires image_path_tensor in the base chunk cache.")
    image_path = decode_path_tensor(sample["image_path_tensor"])
    pixel_values = load_image(image_path, max_num=12).to(device)
    history = sample["history_trajectory"].float()
    command = sample["high_command_one_hot"].float()
    command_names = ["turn left", "go straight", "turn right"]
    command_str = command_names[int(torch.argmax(command).item())]
    history_str = " ".join(
        f"   - t-{3-i}: ({format_number(history[i, 0].item())}, {format_number(history[i, 1].item())}, {format_number(history[i, 2].item())})"
        for i in range(history.shape[0])
    )
    prompt = (
        "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
        f"1. Visual perception from front camera view\n2. Historical motion context (last 4 timesteps):{history_str}\n"
        f"3. Active navigation command: [{command_str.upper()}]\nOutput requirements:\n- Predict 8 future trajectory points\n"
        "- Each point format: (x:float, y:float, heading:float)\n- Use [PT, ...] to encapsulate the trajectory\n"
        "- Maintain numerical precision to 2 decimal places"
    )
    with torch.no_grad():
        outputs = backbone(pixel_values, [prompt], [pixel_values.shape[0]])
    return normalize_hidden_state(outputs.hidden_states[-1])


def regenerate(args: argparse.Namespace) -> Dict[str, Any]:
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards).")
    backbone = None if args.synthetic_smoke else load_backbone(args)
    samples_dir = args.output_chunk_root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    hidden_lengths: List[int] = []
    written = skipped = 0
    for idx, (_, sample_path, record) in enumerate(iter_samples(args.base_chunk_root, args.chunk_name_pattern, args.max_samples)):
        if idx % int(args.num_shards) != int(args.shard_index):
            skipped += 1
            continue
        sample = load_sample(sample_path)
        token = str(sample.get("sample_token") or record.get("sample_token") or sample_path.stem)
        scene_token = str(sample.get("scene_token") or record.get("scene_token") or "")
        log_name = sample.get("log_name") or record.get("log_name")
        log_name = str(log_name) if log_name is not None and str(log_name) else ""
        preserved = {key: sample[key] for key in PRESERVE_KEYS if key in sample}
        if log_name:
            preserved["log_name"] = log_name
        hidden = synthetic_hidden(sample) if args.synthetic_smoke else compute_hidden(backbone, sample, device=args.device)
        hidden = normalize_hidden_state(hidden)
        preserved["last_hidden_state"] = hidden
        hidden_lengths.append(int(hidden.shape[0]))
        out_path = samples_dir / f"{token}.pt"
        atomic_torch_save(preserved, out_path)
        row = {"sample_token": token, "scene_token": scene_token, "path": str(out_path.relative_to(args.output_chunk_root))}
        if log_name:
            row["log_name"] = log_name
        rows.append(row)
        written += 1

    with (args.output_chunk_root / "index.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    metadata = {
        "version": "recogdrive_hidden_cache_with_lora_v1",
        "source_base_cache": str(args.base_chunk_root),
        "vlm_lora_adapter": str(args.vlm_lora_adapter),
        "cache_hidden_state_regenerated": True,
        "hidden_cache_source": "vlm_lora_regenerated",
        "source_lora_adapter": str(args.vlm_lora_adapter),
        "hidden_dtype": "fp32",
        "num_samples": int(written),
        "num_skipped_by_shard": int(skipped),
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "synthetic_smoke": bool(args.synthetic_smoke),
        "token_length": {
            "min": min(hidden_lengths) if hidden_lengths else None,
            "max": max(hidden_lengths) if hidden_lengths else None,
            "mean": float(sum(hidden_lengths) / len(hidden_lengths)) if hidden_lengths else None,
        },
    }
    write_json(args.output_chunk_root / "metadata.json", metadata)
    return metadata


def main() -> int:
    args = parse_args()
    metadata = regenerate(args)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
