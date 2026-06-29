#!/usr/bin/env python3
"""Audit OneVL AR Answer -> ReCogDrive-style DiT Stage2 cache contracts."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from navsim.agents.recogdrive.expert_cache import iter_index, load_sample, validate_sample_payload
    from navsim.agents.recogdrive.recogdrive_diffusion_planner import (
        ReCogDriveDiffusionPlanner,
        ReCogDriveDiffusionPlannerConfig,
    )
except ModuleNotFoundError:
    from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

    _install_dependency_stubs()
    from navsim.agents.recogdrive.expert_cache import iter_index, load_sample, validate_sample_payload
    from navsim.agents.recogdrive.recogdrive_diffusion_planner import (
        ReCogDriveDiffusionPlanner,
        ReCogDriveDiffusionPlannerConfig,
    )


PROVENANCE_KEYS = (
    "hidden_source",
    "hidden_layer",
    "hidden_extraction_mode",
    "model_checkpoint",
    "processor_checkpoint",
    "prompt_template_hash",
    "input_token_count",
    "valid_hidden_length",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/onevl_ar_answer_stage2_small.yaml"))
    parser.add_argument("--train-cache-root", type=Path, default=None)
    parser.add_argument("--val-cache-root", type=Path, default=None)
    parser.add_argument("--nav-cache-root", type=Path, default=None)
    parser.add_argument("--chunk-name-pattern", default="shard_*")
    parser.add_argument("--val-chunk-name-pattern", default="val6000_chunk_*")
    parser.add_argument("--max-samples-per-split", type=int, default=1024)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--support-index-path", type=Path, default=None)
    parser.add_argument("--data-jsonl", type=Path, default=None)
    parser.add_argument("--nav-json", type=Path, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def tiny_planner(vlm_feature_dim: int) -> ReCogDriveDiffusionPlanner:
    cfg = ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg={
            "num_heads": 1,
            "head_dim": 8,
            "num_layers": 1,
            "output_dim": 16,
            "dropout": 0.0,
            "attention_bias": True,
            "norm_eps": 1e-5,
            "interleave_attention": True,
        },
        input_embedding_dim=8,
        hidden_size=16,
        action_dim=3,
        action_horizon=8,
        sampling_method="ddim",
        num_inference_steps=1,
        vlm_feature_dim=vlm_feature_dim,
    )
    return ReCogDriveDiffusionPlanner(cfg).eval()


def chunk_dirs(root: Path | None, pattern: str) -> list[Path]:
    if root is None:
        return []
    if not root.exists():
        return []
    if (root / "index.jsonl").is_file() or list(root.glob("*.pt")) or (root / "samples").is_dir():
        return [root]
    return sorted(path for path in root.glob(pattern) if path.is_dir())


def resolve_record_path(chunk_dir: Path, record: dict[str, Any]) -> Path:
    path = Path(record["path"])
    if path.is_absolute():
        return path
    return chunk_dir / path


def finite_values(tensor: torch.Tensor) -> torch.Tensor:
    values = tensor.detach().float().reshape(-1)
    return values[torch.isfinite(values)]


def scalar_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "count": int(tensor.numel()),
        "min": float(tensor.min().item()),
        "mean": float(tensor.mean().item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "p99": float(torch.quantile(tensor, 0.99).item()),
        "max": float(tensor.max().item()),
    }


def dim_stats(tensor: torch.Tensor | None) -> dict[str, Any]:
    if tensor is None or tensor.numel() == 0:
        return {"count": 0}
    values = tensor.detach().float().reshape(-1, tensor.shape[-1])
    finite_mask = torch.isfinite(values).all(dim=1)
    values = values[finite_mask]
    if values.numel() == 0:
        return {"count": 0}
    return {
        "count": int(values.shape[0]),
        "min": [float(x) for x in values.min(dim=0).values.tolist()],
        "max": [float(x) for x in values.max(dim=0).values.tolist()],
        "mean": [float(x) for x in values.mean(dim=0).tolist()],
        "std": [float(x) for x in values.std(dim=0, unbiased=False).tolist()],
    }


def ratio_from_bools(values: list[bool]) -> float | None:
    if not values:
        return None
    return float(sum(1 for value in values if value) / len(values))


def tensor_or_none(sample: dict[str, Any], key: str) -> torch.Tensor | None:
    value = sample.get(key)
    return value.detach().cpu() if isinstance(value, torch.Tensor) else None


def load_support_tokens(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    obj = torch.load(path, map_location="cpu")
    token_to_row = obj.get("token_to_row", {}) if isinstance(obj, dict) else {}
    return set(map(str, token_to_row.keys())) if isinstance(token_to_row, dict) else set()


class SplitAccumulator:
    def __init__(self, name: str, planner: ReCogDriveDiffusionPlanner, vlm_feature_dim: int, support_tokens: set[str]):
        self.name = name
        self.planner = planner
        self.vlm_feature_dim = vlm_feature_dim
        self.support_tokens = support_tokens
        self.sample_count = 0
        self.warnings: list[str] = []
        self.hidden_lengths: list[float] = []
        self.valid_hidden_lengths: list[float] = []
        self.input_token_counts: list[float] = []
        self.hidden_dims: collections.Counter[str] = collections.Counter()
        self.hidden_finite_rates: list[float] = []
        self.hidden_norms: list[float] = []
        self.hidden_norm_max: list[float] = []
        self.pad_norms: list[float] = []
        self.valid_norms: list[float] = []
        self.row_image_counts: list[float] = []
        self.first_eq_last: list[bool] = []
        self.first_last_different: list[bool] = []
        self.selected_policy: collections.Counter[str] = collections.Counter()
        self.support_index_hit: list[bool] = []
        self.prompt_source: collections.Counter[str] = collections.Counter()
        self.prompt_alignment_pass: list[bool] = []
        self.prompt_command_mismatch: list[bool] = []
        self.status_policy: collections.Counter[str] = collections.Counter()
        self.high_command_policy: collections.Counter[str] = collections.Counter()
        self.raw_command_shape: collections.Counter[str] = collections.Counter()
        self.high_command_distribution: collections.Counter[str] = collections.Counter()
        self.fallback_command_count = 0
        self.target_source: collections.Counter[str] = collections.Counter()
        self.trajectory_values: list[torch.Tensor] = []
        self.trajectory_norm_values: list[torch.Tensor] = []
        self.trajectory_norm_out_ratios: list[float] = []
        self.roundtrip_errors: list[float] = []
        self.support_values: list[torch.Tensor] = []
        self.support_norm_values: list[torch.Tensor] = []
        self.support_norm_out_ratios: list[float] = []
        self.support_valid_counts: list[float] = []
        self.support_weight_sums: list[float] = []
        self.support_scores: list[float] = []
        self.support_missing: list[bool] = []
        self.missing_provenance: collections.Counter[str] = collections.Counter()
        self.examples: list[dict[str, Any]] = []

    def add(self, sample: dict[str, Any], record: dict[str, Any], path: Path) -> None:
        meta = sample.get("meta") if isinstance(sample.get("meta"), dict) else {}
        self.sample_count += 1
        try:
            validate_sample_payload(
                sample,
                require_jepa=False,
                require_vggt=False,
                require_targets=False,
                vlm_feature_dim=self.vlm_feature_dim,
            )
        except Exception as exc:  # noqa: BLE001
            self.warnings.append(f"{path}: validate_sample_payload failed: {exc}")

        hidden = tensor_or_none(sample, "last_hidden_state")
        if hidden is not None and hidden.ndim == 2:
            self.hidden_lengths.append(float(hidden.shape[0]))
            self.hidden_dims[str(int(hidden.shape[1]))] += 1
            finite = torch.isfinite(hidden.float())
            self.hidden_finite_rates.append(float(finite.float().mean().item()))
            norms = hidden.float().norm(dim=-1)
            self.hidden_norms.append(float(norms.mean().item()))
            self.hidden_norm_max.append(float(norms.max().item()))
            valid_len = int(meta.get("valid_hidden_length") or hidden.shape[0])
            valid_len = max(0, min(valid_len, hidden.shape[0]))
            self.valid_hidden_lengths.append(float(valid_len))
            padding_side = str(meta.get("hidden_padding_side", "left"))
            if valid_len < hidden.shape[0]:
                pad = hidden[: hidden.shape[0] - valid_len] if padding_side == "left" else hidden[valid_len:]
                valid = hidden[hidden.shape[0] - valid_len :] if padding_side == "left" else hidden[:valid_len]
                if pad.numel():
                    self.pad_norms.append(float(pad.float().norm(dim=-1).mean().item()))
                if valid.numel():
                    self.valid_norms.append(float(valid.float().norm(dim=-1).mean().item()))
            else:
                self.valid_norms.append(float(norms.mean().item()))
        if "input_token_count" in meta:
            self.input_token_counts.append(float(meta["input_token_count"]))
        for key in PROVENANCE_KEYS:
            if key not in meta:
                self.missing_provenance[key] += 1

        image_count = int(meta.get("row_image_count", 0) or 0)
        if image_count:
            self.row_image_counts.append(float(image_count))
        first = str(meta.get("row_image_first", ""))
        last = str(meta.get("row_image_last", ""))
        if first or last:
            self.first_eq_last.append(bool(first and first == last))
            self.first_last_different.append(bool(first and last and first != last))
        self.selected_policy[str(meta.get("current_image_policy", "unknown"))] += 1
        token = str(meta.get("support_index_token") or meta.get("token") or record.get("sample_token", ""))
        if self.support_tokens:
            self.support_index_hit.append(token in self.support_tokens)

        self.prompt_source[str(meta.get("prompt_source", meta.get("prompt_source_mode", "unknown")))] += 1
        if "prompt_scene_alignment_pass" in meta:
            self.prompt_alignment_pass.append(bool(meta.get("prompt_scene_alignment_pass")))
        prompt_command = str(meta.get("prompt_command", ""))
        scene_command = str(meta.get("scene_command", ""))
        if prompt_command or scene_command:
            self.prompt_command_mismatch.append(bool(prompt_command and scene_command and prompt_command != scene_command))

        self.status_policy[str(meta.get("status_policy", "unknown"))] += 1
        self.high_command_policy[str(meta.get("high_command_policy", "unknown"))] += 1
        self.raw_command_shape[str(meta.get("raw_command_shape", "unknown"))] += 1
        command = tensor_or_none(sample, "high_command_one_hot")
        if command is not None:
            self.high_command_distribution[str([round(float(x), 3) for x in command.flatten().tolist()])] += 1
        if "fallback" in str(meta.get("high_command_policy", "")):
            self.fallback_command_count += 1
        self.target_source[str(meta.get("target_source", "unknown"))] += 1

        trajectory = tensor_or_none(sample, "trajectory")
        if trajectory is not None:
            trajectory = trajectory.float()
            self.trajectory_values.append(trajectory)
            normed = self.planner.norm_odo(trajectory)
            self.trajectory_norm_values.append(normed)
            self.trajectory_norm_out_ratios.append(float(((normed < -1.0) | (normed > 1.0)).float().mean().item()))
            roundtrip = self.planner.denorm_odo(normed)
            self.roundtrip_errors.append(float((roundtrip - trajectory).abs().max().item()))
        support = tensor_or_none(sample, "support_trajectories")
        support_mask = tensor_or_none(sample, "support_mask")
        support_weights = tensor_or_none(sample, "support_weights")
        support_scores = tensor_or_none(sample, "support_scores")
        if support is not None:
            mask = support_mask.bool() if support_mask is not None else torch.ones(support.shape[0], dtype=torch.bool)
            valid_support = support.float()[mask]
            self.support_valid_counts.append(float(mask.sum().item()))
            if valid_support.numel():
                self.support_values.append(valid_support)
                normed_support = self.planner.norm_odo(valid_support)
                self.support_norm_values.append(normed_support)
                self.support_norm_out_ratios.append(
                    float(((normed_support < -1.0) | (normed_support > 1.0)).float().mean().item())
                )
            if support_weights is not None:
                self.support_weight_sums.append(float((support_weights.float() * mask.float()).sum().item()))
            if support_scores is not None and support_scores.numel():
                self.support_scores.extend(float(x) for x in support_scores.float()[mask].flatten().tolist())
        missing_mask = tensor_or_none(sample, "support_missing_mask")
        if missing_mask is not None:
            self.support_missing.append(bool(missing_mask.bool().flatten()[0].item()))

        if len(self.examples) < 5 and (self.warnings or self.first_last_different[-1:] == [True]):
            self.examples.append({
                "path": str(path),
                "token": token,
                "current_image_policy": meta.get("current_image_policy"),
                "row_image_first": meta.get("row_image_first"),
                "row_image_last": meta.get("row_image_last"),
                "prompt_scene_alignment_error": meta.get("prompt_scene_alignment_error"),
            })

    def summary(self) -> dict[str, Any]:
        trajectory = torch.cat(self.trajectory_values, dim=0) if self.trajectory_values else None
        trajectory_norm = torch.cat(self.trajectory_norm_values, dim=0) if self.trajectory_norm_values else None
        support = torch.cat(self.support_values, dim=0) if self.support_values else None
        support_norm = torch.cat(self.support_norm_values, dim=0) if self.support_norm_values else None
        warnings = list(self.warnings)
        if self.missing_provenance:
            warnings.append(f"missing provenance fields: {dict(self.missing_provenance)}")
        if trajectory is not None:
            raw_abs_max = float(trajectory.abs().max().item())
            if raw_abs_max <= 1.05:
                warnings.append("trajectory values are mostly inside [-1,1]; possible double-normalization risk")
        if support is not None:
            support_abs_max = float(support.abs().max().item())
            if support_abs_max <= 1.05:
                warnings.append("support_trajectories are mostly inside [-1,1]; possible double-normalization risk")
        if self.trajectory_norm_out_ratios and sum(self.trajectory_norm_out_ratios) / len(self.trajectory_norm_out_ratios) > 0.05:
            warnings.append("norm_odo(trajectory) has >5% values outside [-1,1]; fixed ReCogDrive range may under-cover targets")
        if self.support_norm_out_ratios and sum(self.support_norm_out_ratios) / len(self.support_norm_out_ratios) > 0.05:
            warnings.append("norm_odo(support_trajectories) has >5% values outside [-1,1]; fixed ReCogDrive range may under-cover support")

        return {
            "sample_count": self.sample_count,
            "image_current_frame": {
                "row_image_count": scalar_stats(self.row_image_counts),
                "first_eq_last_ratio": ratio_from_bools(self.first_eq_last),
                "first_last_different_ratio": ratio_from_bools(self.first_last_different),
                "selected_policy": dict(self.selected_policy),
                "support_index_hit_rate": ratio_from_bools(self.support_index_hit),
            },
            "prompt_planner_alignment": {
                "prompt_source": dict(self.prompt_source),
                "prompt_scene_alignment_pass_rate": ratio_from_bools(self.prompt_alignment_pass),
                "prompt_command_mismatch_rate": ratio_from_bools(self.prompt_command_mismatch),
            },
            "hidden": {
                "token_length": scalar_stats(self.hidden_lengths),
                "valid_hidden_length": scalar_stats(self.valid_hidden_lengths),
                "input_token_count": scalar_stats(self.input_token_counts),
                "hidden_dims": dict(self.hidden_dims),
                "finite_rate": scalar_stats(self.hidden_finite_rates),
                "hidden_norm_mean": scalar_stats(self.hidden_norms),
                "hidden_norm_max": scalar_stats(self.hidden_norm_max),
                "pad_norm_mean": scalar_stats(self.pad_norms),
                "valid_norm_mean": scalar_stats(self.valid_norms),
            },
            "status_command": {
                "status_policy": dict(self.status_policy),
                "high_command_policy": dict(self.high_command_policy),
                "raw_command_shape": dict(self.raw_command_shape),
                "high_command_one_hot_distribution": dict(self.high_command_distribution),
                "fallback_command_ratio": self.fallback_command_count / self.sample_count if self.sample_count else None,
            },
            "trajectory_support": {
                "target_source": dict(self.target_source),
                "trajectory_raw": dim_stats(trajectory),
                "trajectory_norm_odo": dim_stats(trajectory_norm),
                "trajectory_norm_out_of_range_ratio": scalar_stats(self.trajectory_norm_out_ratios),
                "roundtrip_max_error": scalar_stats(self.roundtrip_errors),
                "support_raw": dim_stats(support),
                "support_norm_odo": dim_stats(support_norm),
                "support_norm_out_of_range_ratio": scalar_stats(self.support_norm_out_ratios),
                "support_valid_count": scalar_stats(self.support_valid_counts),
                "support_weight_sum": scalar_stats(self.support_weight_sums),
                "support_scores": scalar_stats(self.support_scores),
                "support_missing_ratio": ratio_from_bools(self.support_missing),
            },
            "warnings": warnings,
            "examples": self.examples,
        }


def audit_split(
    name: str,
    dirs: list[Path],
    *,
    planner: ReCogDriveDiffusionPlanner,
    vlm_feature_dim: int,
    max_samples: int,
    support_tokens: set[str],
) -> dict[str, Any]:
    acc = SplitAccumulator(name, planner, vlm_feature_dim, support_tokens)
    for chunk_dir in dirs:
        for record in iter_index(chunk_dir):
            if max_samples >= 0 and acc.sample_count >= max_samples:
                return acc.summary()
            path = resolve_record_path(chunk_dir, record)
            sample = load_sample(path)
            acc.add(sample, record, path)
    return acc.summary()


def build_recommendations(splits: dict[str, Any]) -> list[str]:
    recs: list[str] = []
    for split_name, summary in splits.items():
        if not summary.get("sample_count"):
            continue
        image = summary["image_current_frame"]
        hidden = summary["hidden"]
        prompt = summary["prompt_planner_alignment"]
        traj = summary["trajectory_support"]
        if (image.get("first_last_different_ratio") or 0.0) > 0.0 and "first" in image.get("selected_policy", {}):
            recs.append(f"{split_name}: P0 check first-vs-last current image before full training.")
        if prompt.get("prompt_scene_alignment_pass_rate") not in (None, 1.0):
            recs.append(f"{split_name}: P1 row prompt and SceneLoader tensors are not strictly aligned.")
        dims = hidden.get("hidden_dims", {})
        if len(dims) != 1:
            recs.append(f"{split_name}: P3 hidden dim distribution is inconsistent: {dims}.")
        if traj["trajectory_norm_out_of_range_ratio"].get("mean", 0.0) > 0.05:
            recs.append(f"{split_name}: P4 trajectory normalization range under-covers this cache.")
        if traj["support_norm_out_of_range_ratio"].get("mean", 0.0) > 0.05:
            recs.append(f"{split_name}: P4 support normalization range under-covers this cache.")
    if not recs:
        recs.append("No hard contract failure found in sampled cache; continue with targeted first/last and adapter ablations.")
    return recs


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# OneVL Stage2 Contract Audit",
        "",
        f"Config: `{payload['config']}`",
        f"Max samples per split: {payload['max_samples_per_split']}",
        "",
        "## Recommendations",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["recommendations"])
    for split_name, summary in payload["splits"].items():
        lines.extend([
            "",
            f"## {split_name}",
            "",
            f"- Samples: {summary['sample_count']}",
            f"- Image policy: `{summary['image_current_frame']['selected_policy']}`",
            f"- First/last different ratio: `{summary['image_current_frame']['first_last_different_ratio']}`",
            f"- Prompt alignment pass rate: `{summary['prompt_planner_alignment']['prompt_scene_alignment_pass_rate']}`",
            f"- Hidden lengths: `{summary['hidden']['token_length']}`",
            f"- Hidden dims: `{summary['hidden']['hidden_dims']}`",
            f"- Trajectory raw stats: `{summary['trajectory_support']['trajectory_raw']}`",
            f"- Trajectory norm out-of-range: `{summary['trajectory_support']['trajectory_norm_out_of_range_ratio']}`",
            f"- Support valid count: `{summary['trajectory_support']['support_valid_count']}`",
            f"- Support weight sum: `{summary['trajectory_support']['support_weight_sum']}`",
        ])
        if summary["warnings"]:
            lines.extend(["", "### Warnings", ""])
            lines.extend(f"- {warning}" for warning in summary["warnings"])
        if summary["examples"]:
            lines.extend(["", "### Examples", ""])
            lines.extend(f"- `{json.dumps(example, ensure_ascii=False, sort_keys=True)}`" for example in summary["examples"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    vlm_feature_dim = int(cfg.get("vlm_feature_dim", 2560))
    planner = tiny_planner(vlm_feature_dim)
    support_tokens = load_support_tokens(args.support_index_path)
    splits = {
        "train": audit_split(
            "train",
            chunk_dirs(args.train_cache_root, args.chunk_name_pattern),
            planner=planner,
            vlm_feature_dim=vlm_feature_dim,
            max_samples=args.max_samples_per_split,
            support_tokens=support_tokens,
        ),
        "val6000": audit_split(
            "val6000",
            chunk_dirs(args.val_cache_root, args.val_chunk_name_pattern),
            planner=planner,
            vlm_feature_dim=vlm_feature_dim,
            max_samples=args.max_samples_per_split,
            support_tokens=support_tokens,
        ),
        "navtest": audit_split(
            "navtest",
            chunk_dirs(args.nav_cache_root, args.chunk_name_pattern),
            planner=planner,
            vlm_feature_dim=vlm_feature_dim,
            max_samples=args.max_samples_per_split,
            support_tokens=support_tokens,
        ),
    }
    payload = {
        "config": str(args.config),
        "max_samples_per_split": int(args.max_samples_per_split),
        "cache_roots": {
            "train": str(args.train_cache_root) if args.train_cache_root else "",
            "val6000": str(args.val_cache_root) if args.val_cache_root else "",
            "navtest": str(args.nav_cache_root) if args.nav_cache_root else "",
        },
        "support_index_path": str(args.support_index_path) if args.support_index_path else "",
        "support_index_token_count": len(support_tokens),
        "splits": splits,
    }
    payload["recommendations"] = build_recommendations(splits)
    return payload


def main() -> None:
    args = parse_args()
    payload = run_audit(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.output_md, payload)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
