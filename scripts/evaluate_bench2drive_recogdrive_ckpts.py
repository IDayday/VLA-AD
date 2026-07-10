#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from transformers.feature_extraction_utils import BatchFeature  # noqa: E402
except ModuleNotFoundError:
    from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs  # noqa: E402

    _install_dependency_stubs()
    from transformers.feature_extraction_utils import BatchFeature  # noqa: E402

from navsim.agents.recogdrive.expert_cache import iter_index, load_sample, validate_sample_payload  # noqa: E402
from scripts.train_recogdrive_expert_chunked import build_planner, shape_safe_load  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ReCogDrive checkpoints on cached Bench2Drive open-loop samples.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint-name", action="append", default=None)
    parser.add_argument("--chunk-cache-root", type=Path, default=None)
    parser.add_argument("--chunk-cache-dir", type=Path, default=None)
    parser.add_argument("--chunk-name-pattern", default="shard_*")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--selection", choices=("even", "first"), default="even")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--init-action-mode", choices=("token_noise", "zeros", "random"), default="token_noise")
    parser.add_argument("--init-action-seed", type=int, default=20260709)
    parser.add_argument("--loss-seed", type=int, default=20260709)
    parser.add_argument("--skip-loss", action="store_true")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dtype_from_precision(precision: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]


def resolve_index_path(chunk_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_file():
        return path
    if not path.is_absolute():
        candidate = chunk_dir / path
        if candidate.is_file():
            return candidate
    return path


def chunk_dirs(args: argparse.Namespace) -> List[Path]:
    if args.chunk_cache_dir is not None:
        return [args.chunk_cache_dir]
    if args.chunk_cache_root is None:
        raise ValueError("Set --chunk-cache-dir or --chunk-cache-root")
    chunks = sorted(path for path in args.chunk_cache_root.glob(args.chunk_name_pattern) if path.is_dir())
    if not chunks:
        raise FileNotFoundError(f"No {args.chunk_name_pattern!r} directories found under {args.chunk_cache_root}")
    return chunks


def load_cache_metadata(chunks: Sequence[Path]) -> Dict[str, Any]:
    items = []
    for chunk in chunks:
        path = chunk / "metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        items.append(
            {
                "chunk": str(chunk),
                "num_records": int(metadata.get("num_records", 0)),
                "num_selected_clips": int(metadata.get("num_selected_clips", 0)),
                "num_skipped_windows": int(metadata.get("num_skipped_windows", 0)),
                "contains_vlm_hidden": bool(metadata.get("contains_vlm_hidden", False)),
                "contains_jepa": bool(metadata.get("contains_jepa", False)),
                "contains_vggt": bool(metadata.get("contains_vggt", False)),
                "hidden_source": metadata.get("hidden_source"),
                "is_dummy": bool(metadata.get("is_dummy", False)),
            }
        )
    return {
        "chunks": items,
        "num_chunks": len(items),
        "num_records": sum(item["num_records"] for item in items),
        "num_selected_clips": sum(item["num_selected_clips"] for item in items),
        "num_skipped_windows": sum(item["num_skipped_windows"] for item in items),
        "all_vlm_hidden": all(item["contains_vlm_hidden"] for item in items),
        "any_dummy": any(item["is_dummy"] for item in items),
    }


class Bench2DriveChunkDataset(Dataset):
    def __init__(self, chunks: Sequence[Path], *, cfg: Dict[str, Any]) -> None:
        self.require_jepa = bool(cfg.get("use_expert_features", False) and cfg.get("use_jepa", True))
        self.require_vggt = bool(cfg.get("use_expert_features", False) and cfg.get("use_vggt", True))
        self.vlm_feature_dim = int(cfg.get("vlm_feature_dim", 1536))
        self.records: List[Tuple[Path, Dict[str, Any]]] = []
        for chunk in chunks:
            for record in iter_index(chunk):
                self.records.append((chunk, record))
        if not self.records:
            raise RuntimeError("No records found in Bench2Drive chunk cache.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        chunk, record = self.records[index]
        path = resolve_index_path(chunk, record["path"])
        sample = load_sample(path)
        validate_sample_payload(
            sample,
            require_jepa=self.require_jepa,
            require_vggt=self.require_vggt,
            require_targets=False,
            vlm_feature_dim=self.vlm_feature_dim,
        )
        for key in ("last_hidden_state", "history_trajectory", "high_command_one_hot", "status_feature", "trajectory"):
            if key not in sample:
                raise KeyError(f"{path} missing required key {key!r}.")
        return {
            **sample,
            "_path": str(path),
            "_chunk": str(chunk),
            "_sample_token": str(sample.get("sample_token", record.get("sample_token", path.stem))),
            "_scene_token": str(sample.get("scene_token", path.stem)),
        }


def evenly_spaced_indices(total: int, count: int) -> List[int]:
    if count >= total:
        return list(range(total))
    if count <= 0:
        raise ValueError("--max-samples must be positive when set.")
    if count == 1:
        return [0]
    return sorted({round(i * (total - 1) / (count - 1)) for i in range(count)})


def collate(samples: List[Dict[str, Any]]) -> Tuple[torch.Tensor, BatchFeature, Dict[str, Any]]:
    last_hidden_state = pad_sequence([sample["last_hidden_state"].float() for sample in samples], batch_first=True, padding_value=0.0)
    his = torch.stack([sample["history_trajectory"].float().view(-1) for sample in samples], dim=0)
    high_command = torch.stack([sample["high_command_one_hot"].float() for sample in samples], dim=0)
    status = torch.stack([sample["status_feature"].float() for sample in samples], dim=0)
    action = torch.stack([sample["trajectory"].float() for sample in samples], dim=0)
    data: Dict[str, torch.Tensor] = {
        "his_traj": his,
        "history_trajectory": torch.stack([sample["history_trajectory"].float() for sample in samples], dim=0),
        "high_command_one_hot": high_command,
        "status_feature": status,
        "action": action,
    }
    for key in (
        "jepa_context_tokens",
        "vggt_context_tokens",
        "vggt_geometry_tokens",
        "vggt_depth_tokens",
        "vggt_pointmap_tokens",
        "vggt_camera_tokens",
    ):
        if key in samples[0]:
            data[key] = torch.stack([sample[key].float() for sample in samples], dim=0)
    meta = {
        "paths": [sample["_path"] for sample in samples],
        "chunks": [sample["_chunk"] for sample in samples],
        "sample_tokens": [sample["_sample_token"] for sample in samples],
        "scene_tokens": [sample["_scene_token"] for sample in samples],
    }
    return last_hidden_state, BatchFeature(data=data), meta


def eval_action_input(action_input: BatchFeature) -> BatchFeature:
    safe_keys = (
        "his_traj",
        "history_trajectory",
        "high_command_one_hot",
        "status_feature",
        "jepa_context_tokens",
        "vggt_context_tokens",
        "vggt_geometry_tokens",
        "vggt_depth_tokens",
        "vggt_pointmap_tokens",
        "vggt_camera_tokens",
    )
    return BatchFeature(data={key: action_input[key] for key in safe_keys if key in action_input})


def tensor_to_device(value: Any, *, device: torch.device, dtype: torch.dtype) -> Any:
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bool:
            return value.to(device=device)
        return value.to(device=device, dtype=dtype)
    return value


def batch_to_device(batch: BatchFeature, *, device: torch.device, dtype: torch.dtype) -> BatchFeature:
    return BatchFeature(data={key: tensor_to_device(value, device=device, dtype=dtype) for key, value in batch.items()})


def token_seed(token: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def make_init_actions(
    sample_tokens: Sequence[str],
    *,
    mode: str,
    seed: int,
    horizon: int,
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    shape = (len(sample_tokens), horizon, action_dim)
    if mode == "random":
        return None
    if mode == "zeros":
        return torch.zeros(shape, device=device, dtype=dtype)
    rows = []
    for token in sample_tokens:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(token_seed(token, seed))
        rows.append(torch.randn((horizon, action_dim), generator=generator, dtype=torch.float32))
    return torch.stack(rows, dim=0).to(device=device, dtype=dtype)


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.point_count = 0
        self.sums: Dict[str, float] = {}
        self.max_values: Dict[str, float] = {}
        self.by_step_sum: Optional[torch.Tensor] = None
        self.by_dim_sum: Optional[torch.Tensor] = None
        self.loss_count = 0

    def add_scalar(self, name: str, value: torch.Tensor, weight: int) -> None:
        self.sums[name] = self.sums.get(name, 0.0) + float(value.detach().float().item()) * weight

    def add_max(self, name: str, value: torch.Tensor) -> None:
        current = float(value.detach().float().item())
        self.max_values[name] = max(self.max_values.get(name, current), current)

    def update_prediction(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        batch = int(pred.shape[0])
        self.count += batch
        self.point_count += int(pred.shape[0] * pred.shape[1])
        diff = pred - target
        abs_diff = diff.abs()
        xy_l2 = torch.linalg.vector_norm(diff[..., :2], dim=-1)
        norm_pred = self._norm_odo(pred)
        norm_target = self._norm_odo(target)
        self.add_scalar("raw_l1", abs_diff.mean(), batch)
        self.add_scalar("xy_ade", xy_l2.mean(), batch)
        self.add_scalar("xy_fde", xy_l2[:, -1].mean(), batch)
        self.add_scalar("heading_abs", abs_diff[..., 2].mean(), batch)
        self.add_scalar("norm_l1", (norm_pred - norm_target).abs().mean(), batch)
        self.add_max("pred_max_abs", pred.abs().amax())
        self.add_max("target_max_abs", target.abs().amax())
        by_step = xy_l2.sum(dim=0).detach().cpu()
        by_dim = abs_diff.sum(dim=(0, 1)).detach().cpu()
        self.by_step_sum = by_step if self.by_step_sum is None else self.by_step_sum + by_step
        self.by_dim_sum = by_dim if self.by_dim_sum is None else self.by_dim_sum + by_dim

    def update_loss(self, outputs: BatchFeature, batch_size: int) -> None:
        self.loss_count += batch_size
        for key in ("loss", "diffusion_loss", "jepa_alignment_loss", "vggt_alignment_loss"):
            if key in outputs and isinstance(outputs[key], torch.Tensor):
                self.sums[f"teacher_forced_{key}"] = self.sums.get(f"teacher_forced_{key}", 0.0) + float(outputs[key].detach().float().item()) * batch_size

    @staticmethod
    def _norm_odo(trajectory: torch.Tensor) -> torch.Tensor:
        x = 2 * (trajectory[..., 0:1] + 1.57) / 66.74 - 1
        y = 2 * (trajectory[..., 1:2] + 19.68) / 42 - 1
        heading = 2 * (trajectory[..., 2:3] + 1.67) / 3.53 - 1
        return torch.cat([x, y, heading], dim=-1)

    def to_dict(self) -> Dict[str, Any]:
        if self.count <= 0:
            raise RuntimeError("No samples were evaluated.")
        result = {key: value / self.count for key, value in sorted(self.sums.items()) if not key.startswith("teacher_forced_")}
        result.update(self.max_values)
        for key, value in sorted(self.sums.items()):
            if key.startswith("teacher_forced_"):
                result[key] = value / max(self.loss_count, 1)
        if self.by_step_sum is not None:
            result["xy_l2_by_step"] = (self.by_step_sum / self.count).tolist()
        if self.by_dim_sum is not None:
            denom = max(self.point_count, 1)
            result["raw_l1_by_dim"] = (self.by_dim_sum / denom).tolist()
        result["num_samples"] = self.count
        result["loss_num_samples"] = self.loss_count
        return result


def checkpoint_payload_metadata(path: Path) -> Dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        return {}
    metrics = obj.get("metrics") if isinstance(obj.get("metrics"), dict) else {}
    return {
        "global_step": obj.get("global_step"),
        "metrics": metrics,
    }


def make_planner_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        vlm_feature_dim=None,
        precision=args.precision,
        jepa_align_weight=None,
        vggt_align_weight=None,
        diffusion_loss_weight=None,
        vlm_adapter_type=None,
        vlm_adapter_dropout=None,
        use_action_aware_aux=None,
        action_aware_aux_weight=None,
        action_aware_aux_space=None,
        expert_gate_init=None,
    )


def checkpoint_names(paths: Sequence[Path], names: Optional[Sequence[str]]) -> List[str]:
    if names is not None:
        if len(names) != len(paths):
            raise ValueError("--checkpoint-name must be supplied once per --checkpoint when used.")
        return [str(name) for name in names]
    result = []
    seen: Dict[str, int] = {}
    for path in paths:
        name = path.stem if path.is_file() else path.name
        seen[name] = seen.get(name, 0) + 1
        result.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return result


def evaluate_checkpoint(
    *,
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    checkpoint: Path,
    name: str,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    planner = build_planner(cfg, make_planner_args(args)).to(device)
    if dtype != torch.float32:
        planner = planner.to(dtype=dtype)
    load_report = shape_safe_load(planner, checkpoint, strict_original=False)
    planner.eval()

    stats = RunningStats()
    start = time.monotonic()
    horizon = int(getattr(planner.config, "action_horizon", 8))
    action_dim = int(getattr(planner.config, "action_dim", 3))
    with torch.no_grad():
        for batch_idx, (vl_features, action_input, meta) in enumerate(loader):
            vl_features = vl_features.to(device=device, dtype=dtype)
            action_input = batch_to_device(action_input, device=device, dtype=dtype)
            if not args.skip_loss:
                torch.manual_seed(args.loss_seed + batch_idx)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.loss_seed + batch_idx)
                outputs = planner(vl_features, action_input)
                stats.update_loss(outputs, int(action_input["action"].shape[0]))
            pred_input = eval_action_input(action_input)
            init_actions = make_init_actions(
                meta["sample_tokens"],
                mode=args.init_action_mode,
                seed=args.init_action_seed,
                horizon=horizon,
                action_dim=action_dim,
                device=device,
                dtype=dtype,
            )
            pred = planner.get_action(
                vl_features,
                pred_input,
                init_actions=init_actions,
                deterministic=bool(args.deterministic),
            )["pred_traj"].detach().float()
            target = action_input["action"].detach().float()
            if not torch.isfinite(pred).all():
                raise RuntimeError(f"Non-finite prediction while evaluating {checkpoint}")
            stats.update_prediction(pred, target)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - start
    metrics = stats.to_dict()
    metrics.update(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_name": name,
            "elapsed_seconds": elapsed,
            "samples_per_second": metrics["num_samples"] / max(elapsed, 1e-9),
            "load_report": load_report,
            "checkpoint_payload": checkpoint_payload_metadata(checkpoint) if checkpoint.is_file() else {},
        }
    )
    return metrics


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    chunks = chunk_dirs(args)
    cache_summary = load_cache_metadata(chunks)
    if cache_summary["any_dummy"]:
        raise RuntimeError("Refusing to evaluate dummy Bench2Drive cache.")
    if not cache_summary["all_vlm_hidden"]:
        raise RuntimeError("Bench2Drive evaluation requires cached VLM hidden states.")

    dataset = Bench2DriveChunkDataset(chunks, cfg=cfg)
    selected_indices: Optional[List[int]] = None
    if args.max_samples is not None and args.max_samples < len(dataset):
        selected_indices = (
            evenly_spaced_indices(len(dataset), args.max_samples)
            if args.selection == "even"
            else list(range(args.max_samples))
        )
        eval_dataset: Dataset = Subset(dataset, selected_indices)
    else:
        eval_dataset = dataset
    loader_kwargs: Dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": collate,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(eval_dataset, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_precision(args.precision) if device.type == "cuda" else torch.float32
    args.output_dir.mkdir(parents=True, exist_ok=True)

    names = checkpoint_names(args.checkpoint, args.checkpoint_name)
    results = []
    total_start = time.monotonic()
    for checkpoint, name in zip(args.checkpoint, names):
        metrics = evaluate_checkpoint(
            args=args,
            cfg=cfg,
            checkpoint=checkpoint,
            name=name,
            loader=loader,
            device=device,
            dtype=dtype,
        )
        results.append(metrics)
        (args.output_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({key: metrics[key] for key in ("checkpoint_name", "num_samples", "raw_l1", "xy_ade", "xy_fde", "samples_per_second")}, sort_keys=True), flush=True)

    summary = {
        "config": str(args.config),
        "precision": args.precision,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "batch_size": args.batch_size,
        "init_action_mode": args.init_action_mode,
        "init_action_seed": args.init_action_seed,
        "deterministic": bool(args.deterministic),
        "skip_loss": bool(args.skip_loss),
        "selection": args.selection,
        "selected_indices": selected_indices[:20] if selected_indices is not None else None,
        "num_dataset_samples": len(dataset),
        "num_eval_samples": len(eval_dataset),
        "cache_summary": cache_summary,
        "results": results,
        "elapsed_seconds": time.monotonic() - total_start,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_lines = [
        "# Bench2Drive ReCogDrive Checkpoint Evaluation",
        "",
        f"Config: `{args.config}`",
        f"Samples: {len(eval_dataset)} / {len(dataset)}",
        f"Init actions: `{args.init_action_mode}` seed={args.init_action_seed}",
        f"Cache records: {cache_summary['num_records']} across {cache_summary['num_chunks']} chunks",
        "",
        "| checkpoint | samples | raw_l1 | norm_l1 | xy_ade | xy_fde | teacher_forced_loss | samples/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        report_lines.append(
            "| {checkpoint_name} | {num_samples} | {raw_l1:.6f} | {norm_l1:.6f} | "
            "{xy_ade:.6f} | {xy_fde:.6f} | {loss:.6f} | {sps:.2f} |".format(
                checkpoint_name=item["checkpoint_name"],
                num_samples=item["num_samples"],
                raw_l1=item["raw_l1"],
                norm_l1=item["norm_l1"],
                xy_ade=item["xy_ade"],
                xy_fde=item["xy_fde"],
                loss=item.get("teacher_forced_loss", float("nan")),
                sps=item["samples_per_second"],
            )
        )
    report_lines.extend(
        [
            "",
            "This is an open-loop Bench2Drive cache evaluation. It does not compute closed-loop Bench2Drive or NAVSIM PDM scores.",
            "Evaluation inputs exclude future trajectory labels and train-only teacher target tokens; labels are used only for metrics/loss.",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(args.output_dir / "summary.json"), "report": str(args.output_dir / "report.md")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
