#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import atomic_torch_save, iter_index, load_sample, write_json  # noqa: E402
from navsim.agents.recogdrive.vlm_lora_utils import (  # noqa: E402
    load_lora_adapter_metadata,
    peft_lora_config_kwargs_supported,
    stable_config_hash,
)


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
    "jepa_tokenizer_metadata",
    "jepa_num_tokens",
    "vggt_context_tokens",
    "vggt_target_tokens",
    "vggt_geometry_tokens",
    "vggt_geometry_target_tokens",
    "vggt_depth_tokens",
    "vggt_pointmap_tokens",
    "vggt_camera_tokens",
    "vggt_geometry_mode_code",
    "vggt_geometry_mode",
    "vggt_geometry_source",
    "vggt_geometry_tokenizer_metadata",
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
    parser.add_argument("--vlm-lora-adapter", type=Path, default=None, help="Legacy state-only LoRA adapter .pt.")
    parser.add_argument("--vlm-lora-adapter-dir", type=Path, default=None, help="Preferred adapter directory containing adapter_config.json/lora_metadata.json.")
    parser.add_argument("--chunk-name-pattern", default="train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--lora-target-modules", default=None)
    parser.add_argument("--lora-use-rslora", action="store_true", default=None)
    parser.add_argument("--lora-use-dora", action="store_true", default=None)
    parser.add_argument("--allow-lora-config-override", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--cache-variant", default="lora_hidden_regeneration")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing sample .pt files in the output shard.")
    parser.add_argument(
        "--reuse-existing-root",
        type=Path,
        action="append",
        default=[],
        help="Existing cache/shard root to hardlink samples from by sample_token before regenerating.",
    )
    parser.add_argument("--progress-every", type=int, default=0, help="Print progress every N written or reused samples.")
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def _split_targets(value: Any) -> List[str] | str:
    if value is None:
        return []
    if isinstance(value, str):
        if value.strip() == "all-linear":
            return "all-linear"
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value)]


def _adapter_state_path(adapter_dir: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin", "vlm_lora_adapter_state.pt"):
        candidate = adapter_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No adapter model file found under {adapter_dir}")


def _load_lora_state(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return dict(load_file(str(path), device="cpu"))
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"LoRA adapter {path} must contain a state_dict or raw state dict.")
    return state


def _resolve_lora_runtime_config(args: argparse.Namespace) -> tuple[Dict[str, Any], Optional[Path], str]:
    if args.vlm_lora_adapter_dir is not None:
        metadata = load_lora_adapter_metadata(args.vlm_lora_adapter_dir)
        state_path = _adapter_state_path(args.vlm_lora_adapter_dir)
        config = {
            "r": int(metadata.get("r", metadata.get("rank", 0))),
            "alpha": int(metadata.get("alpha", metadata.get("lora_alpha", 0))),
            "dropout": float(metadata.get("dropout", metadata.get("lora_dropout", 0.0))),
            "target_modules": metadata.get("resolved_target_modules", metadata.get("target_modules", [])),
            "bias": metadata.get("bias", "none"),
            "use_rslora": bool(metadata.get("use_rslora", False)),
            "use_dora": bool(metadata.get("use_dora", False)),
            "adapter_config_hash": metadata.get("adapter_config_hash"),
            "metadata": metadata,
        }
        explicit = {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": _split_targets(args.lora_target_modules) if args.lora_target_modules is not None else None,
        }
        mismatches = []
        for key, value in explicit.items():
            if value is None:
                continue
            if key == "target_modules":
                if value != _split_targets(config[key]):
                    mismatches.append(f"{key}: CLI {value!r} != adapter {config[key]!r}")
            elif value != config[key]:
                mismatches.append(f"{key}: CLI {value!r} != adapter {config[key]!r}")
        if mismatches and not args.allow_lora_config_override:
            raise ValueError(
                "Explicit LoRA CLI config does not match adapter directory metadata:\n"
                + "\n".join(f"  - {item}" for item in mismatches)
            )
        return config, state_path, "adapter_dir"
    if args.vlm_lora_adapter is None:
        raise ValueError("Provide --vlm-lora-adapter-dir or legacy --vlm-lora-adapter.")
    config = {
        "r": int(args.lora_r or 16),
        "alpha": int(args.lora_alpha or 32),
        "dropout": float(args.lora_dropout or 0.0),
        "target_modules": _split_targets(args.lora_target_modules or "q_proj,k_proj,v_proj,o_proj"),
        "bias": "none",
        "use_rslora": bool(args.lora_use_rslora or False),
        "use_dora": bool(args.lora_use_dora or False),
        "adapter_config_hash": None,
        "metadata": {},
    }
    return config, args.vlm_lora_adapter, "legacy_state"


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

    lora_config, adapter_state_path, _ = _resolve_lora_runtime_config(args)
    supported_kwargs = peft_lora_config_kwargs_supported()
    if lora_config["use_rslora"] and "use_rslora" not in supported_kwargs:
        raise RuntimeError("Adapter requires use_rslora, but installed PEFT does not support it.")
    if lora_config["use_dora"] and "use_dora" not in supported_kwargs:
        raise RuntimeError("Adapter requires use_dora, but installed PEFT does not support it.")
    lora_kwargs: Dict[str, Any] = {
        "r": int(lora_config["r"]),
        "lora_alpha": int(lora_config["alpha"]),
        "lora_dropout": float(lora_config["dropout"]),
        "target_modules": lora_config["target_modules"],
        "bias": str(lora_config["bias"]),
    }
    if "use_rslora" in supported_kwargs:
        lora_kwargs["use_rslora"] = bool(lora_config["use_rslora"])
    if "use_dora" in supported_kwargs:
        lora_kwargs["use_dora"] = bool(lora_config["use_dora"])
    lora_cfg = LoraConfig(**lora_kwargs)
    peft_vlm = get_peft_model(backbone.model, lora_cfg)
    for attr in ("img_context_token_id", "system_message"):
        if hasattr(backbone.model, attr) and not hasattr(peft_vlm, attr):
            setattr(peft_vlm, attr, getattr(backbone.model, attr))

    state = _load_lora_state(adapter_state_path)

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
        raise RuntimeError(f"No compatible VLM LoRA weights found in {adapter_state_path}.")
    peft_vlm.load_state_dict(filtered, strict=False)
    print(f"Loaded {len(filtered)} VLM LoRA tensors from {adapter_state_path}.")
    if skipped:
        print(f"Skipped {len(skipped)} incompatible/non-LoRA adapter tensors.")
    backbone._patch_internvl_visual_feature_dtype(backbone.model)
    backbone.model = peft_vlm
    backbone.eval()
    return backbone


def build_prompt(sample: Dict[str, Any]) -> str:
    from navsim.agents.recogdrive.recogdrive_agent import format_number

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
    return prompt


def compute_hidden_batch(backbone: Any, batch: List[Dict[str, Any]], *, device: str) -> List[torch.Tensor]:
    if not batch:
        return []
    pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0).to(device)
    prompts = [str(item["prompt"]) for item in batch]
    num_patches_list = [int(item["pixel_values"].shape[0]) for item in batch]
    with torch.no_grad():
        outputs = backbone(pixel_values, prompts, num_patches_list)
    hidden = outputs.hidden_states[-1]
    if hidden.ndim == 2:
        if len(batch) != 1:
            raise ValueError(f"Batched VLM returned 2D hidden state for batch_size={len(batch)}.")
        return [normalize_hidden_state(hidden)]
    if hidden.ndim != 3 or hidden.shape[0] != len(batch):
        raise ValueError(f"Batched VLM hidden state shape {tuple(hidden.shape)} is incompatible with batch_size={len(batch)}.")
    return [normalize_hidden_state(hidden[idx]) for idx in range(len(batch))]


def compute_hidden(backbone: Any, sample: Dict[str, Any], *, device: str) -> torch.Tensor:
    from navsim.agents.recogdrive.utils.internvl_preprocess import load_image

    if not isinstance(sample.get("image_path_tensor"), torch.Tensor):
        raise KeyError("LoRA hidden regeneration requires image_path_tensor in the base chunk cache.")
    image_path = decode_path_tensor(sample["image_path_tensor"])
    item = {
        "pixel_values": load_image(image_path, max_num=12),
        "prompt": build_prompt(sample),
    }
    return compute_hidden_batch(backbone, [item], device=device)[0]


def build_reuse_index(roots: List[Path]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        sample_dirs: List[Path]
        if (root / "samples").is_dir():
            sample_dirs = [root / "samples"]
        else:
            sample_dirs = sorted(path for path in root.glob("*/samples") if path.is_dir())
        for samples_dir in sample_dirs:
            for path in samples_dir.glob("*.pt"):
                mapping.setdefault(path.stem, path)
    return mapping


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def regenerate(args: argparse.Namespace) -> Dict[str, Any]:
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards).")
    lora_config, adapter_state_path, adapter_source = _resolve_lora_runtime_config(args)
    backbone = None if args.synthetic_smoke else load_backbone(args)
    samples_dir = args.output_chunk_root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    hidden_lengths: List[int] = []
    batch: List[Dict[str, Any]] = []
    reuse_index = build_reuse_index(list(args.reuse_existing_root))
    written = skipped = reused = reused_external = regenerated = 0

    def progress() -> None:
        if args.progress_every > 0 and written > 0 and written % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "progress_written": written,
                        "regenerated": regenerated,
                        "reused_existing": reused,
                        "reused_external": reused_external,
                        "skipped_by_shard": skipped,
                        "shard_index": int(args.shard_index),
                        "num_shards": int(args.num_shards),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    def flush_batch() -> None:
        nonlocal regenerated, written
        if not batch:
            return
        if args.synthetic_smoke:
            hidden_states = [normalize_hidden_state(synthetic_hidden(item["sample"])) for item in batch]
        else:
            hidden_states = compute_hidden_batch(backbone, batch, device=args.device)
        for item, hidden in zip(batch, hidden_states):
            preserved = dict(item["preserved"])
            preserved["last_hidden_state"] = hidden
            hidden_lengths.append(int(hidden.shape[0]))
            atomic_torch_save(preserved, item["out_path"])
            rows.append(item["row"])
            regenerated += 1
            written += 1
            progress()
        batch.clear()

    batch_size = max(1, int(args.batch_size))
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
        out_path = samples_dir / f"{token}.pt"
        row = {"sample_token": token, "scene_token": scene_token, "path": str(out_path.relative_to(args.output_chunk_root))}
        if log_name:
            row["log_name"] = log_name
        if args.skip_existing and out_path.is_file():
            rows.append(row)
            reused += 1
            written += 1
            progress()
            continue
        reuse_path = reuse_index.get(token)
        if reuse_path is not None and reuse_path.is_file():
            link_or_copy(reuse_path, out_path)
            rows.append(row)
            reused_external += 1
            written += 1
            progress()
            continue
        item = {
            "sample": sample,
            "preserved": preserved,
            "out_path": out_path,
            "row": row,
        }
        if not args.synthetic_smoke:
            from navsim.agents.recogdrive.utils.internvl_preprocess import load_image

            if not isinstance(sample.get("image_path_tensor"), torch.Tensor):
                raise KeyError("LoRA hidden regeneration requires image_path_tensor in the base chunk cache.")
            image_path = decode_path_tensor(sample["image_path_tensor"])
            item["pixel_values"] = load_image(image_path, max_num=12)
            item["prompt"] = build_prompt(sample)
        batch.append(item)
        if len(batch) >= batch_size:
            flush_batch()
    flush_batch()

    with (args.output_chunk_root / "index.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    metadata = {
        "version": "recogdrive_hidden_cache_with_lora_v1",
        "cache_variant": str(args.cache_variant),
        "source_base_cache": str(args.base_chunk_root),
        "vlm_lora_adapter": str(adapter_state_path) if adapter_state_path is not None else None,
        "vlm_lora_adapter_dir": str(args.vlm_lora_adapter_dir) if args.vlm_lora_adapter_dir is not None else None,
        "vlm_lora_adapter_source": adapter_source,
        "vlm_lora_config": {key: value for key, value in lora_config.items() if key != "metadata"},
        "vlm_lora_adapter_config_hash": lora_config.get("adapter_config_hash") or stable_config_hash(
            {key: value for key, value in lora_config.items() if key not in {"metadata", "adapter_config_hash"}}
        ),
        "cache_hidden_state_regenerated": True,
        "hidden_cache_source": "vlm_lora_regenerated",
        "source_lora_adapter": str(adapter_state_path) if adapter_state_path is not None else None,
        "hidden_dtype": "fp32",
        "num_samples": int(written),
        "num_regenerated": int(regenerated),
        "num_reused_existing": int(reused),
        "num_reused_external": int(reused_external),
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
