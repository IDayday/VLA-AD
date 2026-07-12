#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import lzma
import pickle
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
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
from navsim.agents.recogdrive.recogdrive_diffusion_planner import (  # noqa: E402
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)
from navsim.common.dataclasses import Trajectory  # noqa: E402
from navsim.common.dataloader import MetricCacheLoader  # noqa: E402
from navsim.evaluate.pdm_score import pdm_score  # noqa: E402
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (  # noqa: E402
    PDMScorer,
    PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator  # noqa: E402
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # noqa: E402


_WARNED_TRAIN_ONLY_TARGET_KEYS = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ReCogDrive expert checkpoint on a chunk/disk feature source.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/navsim"))
    parser.add_argument("--split", default="navtest")
    parser.add_argument("--feature-source", choices=("chunk", "disk", "online", "chunk_or_online"), default="chunk")
    parser.add_argument("--chunk-cache-dir", type=Path, default=None)
    parser.add_argument("--chunk-cache-root", type=Path, default=None)
    parser.add_argument("--chunk-name-pattern", default="navtest_full_chunk_*")
    parser.add_argument("--vlm-text-anchor-cache-root", type=Path, default=None)
    parser.add_argument("--metric-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--sample-token-file",
        type=Path,
        default=None,
        help="Optional newline token file restricting evaluation to a deterministic subset.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="fp32")
    parser.add_argument("--trajectory-output-key", choices=("pred_traj", "pred_coarse_traj"), default="pred_traj")
    parser.add_argument("--online-vlm-path", type=Path, default=None)
    parser.add_argument("--online-vlm-type", default="internvl")
    parser.add_argument("--online-vlm-lora-adapter-dir", type=Path, default=None)
    parser.add_argument(
        "--two-expert-corruption-mode",
        choices=("normal", "zero_h_dyn", "zero_h_geo", "zero_all_experts", "raw_vlm_only", "dyn_only", "geo_only"),
        default="normal",
    )
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument(
        "--initial-noise-seed",
        type=int,
        default=None,
        help=(
            "Optional base seed for a sample-token-stable initial diffusion noise. "
            "Use this for checkpoint comparisons that must be invariant to shard count."
        ),
    )
    return parser.parse_args()


def resolve_index_path(chunk_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_file():
        return path
    if not path.is_absolute():
        candidate = chunk_dir / path
        if candidate.is_file():
            return candidate
    return path


def dtype_from_precision(precision: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]


def stable_sample_seed(base_seed: int, sample_token: str) -> int:
    payload = f"{int(base_seed)}:{sample_token}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def make_initial_noise(
    planner: ReCogDriveDiffusionPlanner,
    vl_features: torch.Tensor,
    sample_token: str,
    base_seed: Optional[int],
) -> Optional[torch.Tensor]:
    if base_seed is None:
        return None
    generator = torch.Generator(device=vl_features.device)
    generator.manual_seed(stable_sample_seed(base_seed, sample_token))
    return torch.randn(
        (
            int(vl_features.shape[0]),
            int(planner.config.action_horizon),
            int(planner.config.action_dim),
        ),
        device=vl_features.device,
        dtype=vl_features.dtype,
        generator=generator,
    )


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def build_planner(cfg_dict: Dict[str, Any]) -> ReCogDriveDiffusionPlanner:
    cfg = ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg={
            "num_heads": 8,
            "head_dim": 48,
            "num_layers": int(cfg_dict.get("num_dit_layers", 16)),
            "output_dim": 512,
            "dropout": 0.0,
            "attention_bias": True,
            "norm_eps": 1e-5,
            "interleave_attention": True,
        },
        action_dim=int(cfg_dict.get("action_dim", 3)),
        action_horizon=int(cfg_dict.get("action_horizon", 8)),
        input_embedding_dim=int(cfg_dict.get("planner_dim", 384)),
        planner_dim=int(cfg_dict.get("planner_dim", 384)),
        hidden_size=1024,
        sampling_method=str(cfg_dict.get("sampling_method", "ddim")),
        num_inference_steps=int(cfg_dict.get("num_inference_steps", 5)),
    )
    for key, value in cfg_dict.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.vlm_size = "small"
    cfg.use_expert_features = as_bool(cfg_dict.get("use_expert_features"), True)
    if as_bool(cfg_dict.get("use_alignment_loss"), True) is False:
        cfg.expert_alignment_weight = 0.0
        cfg.jepa_alignment_weight = 0.0
        cfg.vggt_alignment_weight = 0.0
    cfg.allow_future_targets_in_inference = False
    return ReCogDriveDiffusionPlanner(cfg)


def normalize_key(key: str) -> str:
    for prefix in ("agent.action_head.", "action_head."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def checkpoint_candidates(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    suffixes = (".ckpt", ".pth", ".pt", ".safetensors", ".bin")
    files: List[Path] = []
    for suffix in suffixes:
        files.extend(p for p in path.rglob(f"*{suffix}") if p.is_file())
    def score(p: Path) -> tuple[int, int, str]:
        name = p.name.lower()
        value = (1000 if "il" in name else 0) + (500 if "model" in name else 0) + (100 if p.suffix == ".safetensors" else 0)
        return value, p.stat().st_size, str(p)
    return sorted(set(files), key=score, reverse=True)


def load_state_file(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return dict(load_file(str(path), device="cpu"))
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint {path} did not contain a state dict.")
    return {key: value for key, value in state.items() if isinstance(value, torch.Tensor)}


def load_checkpoint(planner: ReCogDriveDiffusionPlanner, path: Path) -> None:
    candidates = checkpoint_candidates(path)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found under {path}")
    model_state = planner.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    for file in candidates:
        state = load_state_file(file)
        for raw_key, value in state.items():
            key = normalize_key(raw_key)
            if key in model_state and tuple(value.shape) == tuple(model_state[key].shape):
                filtered.setdefault(key, value)
    planner.load_state_dict(filtered, strict=False)
    print(f"Loaded {len(filtered)} checkpoint keys from {path}")
    if path.is_dir():
        print("Checkpoint candidates:")
        for idx, candidate in enumerate(candidates):
            marker = " <= selected first" if idx == 0 else ""
            print(f"  - {candidate} ({candidate.stat().st_size} bytes){marker}")


def chunk_dirs(args: argparse.Namespace) -> List[Path]:
    if args.chunk_cache_dir is not None:
        return [args.chunk_cache_dir]
    if args.chunk_cache_root is None:
        raise ValueError("chunk/disk evaluation requires --chunk-cache-dir or --chunk-cache-root")
    patterns = [item.strip() for item in str(args.chunk_name_pattern).split(",") if item.strip()]
    dirs: List[Path] = []
    seen = set()
    for pattern in patterns:
        for path in sorted(args.chunk_cache_root.glob(pattern)):
            if path.is_dir() and path not in seen:
                dirs.append(path)
                seen.add(path)
    if not dirs:
        raise FileNotFoundError(f"No {args.chunk_name_pattern!r} directories found under {args.chunk_cache_root}")
    return dirs


def sample_paths(chunks: List[Path], max_samples: Optional[int]) -> List[Tuple[Path, Path, Dict[str, Any]]]:
    samples: List[Tuple[Path, Path, Dict[str, Any]]] = []
    for chunk_dir in chunks:
        for record in iter_index(chunk_dir):
            samples.append((chunk_dir, resolve_index_path(chunk_dir, record["path"]), record))
            if max_samples is not None and len(samples) >= max_samples:
                return samples
    return samples


def load_sample_token_filter(path: Optional[Path]) -> Optional[set[str]]:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(path)
    tokens: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            token = raw.strip()
            if token and not token.startswith("#"):
                tokens.add(token)
    if not tokens:
        raise RuntimeError(f"No sample tokens found in {path}")
    return tokens


def load_vlm_text_anchor_index(root: Optional[Path]) -> Dict[str, Path]:
    if root is None:
        return {}
    if not root.is_dir():
        raise FileNotFoundError(f"VLM text anchor cache root does not exist: {root}")
    dirs = [root] if (root / "index.jsonl").is_file() else [
        child for child in sorted(root.iterdir()) if child.is_dir() and (child / "index.jsonl").is_file()
    ]
    if not dirs:
        raise FileNotFoundError(f"No VLM text anchor index.jsonl found under {root}")
    index: Dict[str, Path] = {}
    for chunk_dir in dirs:
        for record in iter_index(chunk_dir):
            path = Path(record.get("path", ""))
            if not path:
                continue
            token = str(record.get("sample_token") or path.stem)
            index[token] = path if path.is_absolute() else chunk_dir / path
    return index


def load_eval_sample(path: Path) -> Dict[str, Any]:
    """Load a standard .pt chunk sample or an original ReCogDrive raw token cache dir."""
    if path.is_dir() and (path / "internvl_feature.gz").is_file():
        with gzip.open(path / "internvl_feature.gz", "rb") as f:
            sample = pickle.load(f)
        if not isinstance(sample, dict):
            raise TypeError(f"Raw ReCogDrive feature cache {path} must contain a dict, got {type(sample).__name__}.")
        target_path = path / "trajectory_target.gz"
        if target_path.is_file():
            with gzip.open(target_path, "rb") as f:
                target = pickle.load(f)
            if isinstance(target, dict):
                sample = {**sample, **target}
        sample.setdefault("sample_token", path.name)
        return sample
    return load_sample(path)


def build_pdm_tools() -> Tuple[TrajectorySampling, PDMSimulator, PDMScorer]:
    proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator = PDMSimulator(proposal_sampling=proposal_sampling)
    scorer = PDMScorer(
        proposal_sampling=proposal_sampling,
        config=PDMScorerConfig(
            progress_weight=5.0,
            ttc_weight=5.0,
            comfortable_weight=2.0,
            driving_direction_weight=0.0,
            driving_direction_horizon=1.0,
            driving_direction_compliance_threshold=2.0,
            driving_direction_violation_threshold=6.0,
            stopped_speed_threshold=5e-3,
            progress_distance_threshold=5.0,
        ),
    )
    return proposal_sampling, simulator, scorer


class ScannedMetricCacheLoader:
    def __init__(self, cache_path: Path) -> None:
        self.metric_cache_paths = {
            path.parent.name: path
            for path in cache_path.rglob("metric_cache.pkl")
            if path.is_file()
        }
        if not self.metric_cache_paths:
            raise FileNotFoundError(f"No metric_cache.pkl files found under {cache_path}")

    def get_from_token(self, token: str):
        with lzma.open(self.metric_cache_paths[token], "rb") as f:
            return pickle.load(f)


def build_metric_cache_loader(cache_path: Path):
    scanned_loader = None
    try:
        metadata_loader = MetricCacheLoader(cache_path)
    except Exception as exc:
        warnings.warn(
            f"Could not load metric cache metadata from {cache_path}: {exc!r}. "
            "Falling back to recursive metric_cache.pkl scan.",
            RuntimeWarning,
        )
        return ScannedMetricCacheLoader(cache_path)
    try:
        scanned_loader = ScannedMetricCacheLoader(cache_path)
    except Exception:
        return metadata_loader
    metadata_count = len(getattr(metadata_loader, "metric_cache_paths", {}) or {})
    scanned_count = len(scanned_loader.metric_cache_paths)
    if scanned_count > metadata_count:
        warnings.warn(
            f"Metric cache scan found {scanned_count} files, metadata lists {metadata_count}; "
            "using recursive scan.",
            RuntimeWarning,
        )
        return scanned_loader
    return metadata_loader


def mean_or_none(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def write_pdm_csv(path: Path, rows: List[Dict[str, Any]], averages: Dict[str, Optional[float]]) -> None:
    fields = [
        "sample_token",
        "scene_token",
        "chunk",
        "valid",
        "score",
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "ego_progress",
        "time_to_collision_within_bound",
        "comfort",
        "driving_direction_compliance",
        "trajectory_l1",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
        average_row = {key: averages.get(key) for key in fields}
        average_row.update({"sample_token": "average", "scene_token": "average", "chunk": "average", "valid": all(row.get("valid") for row in rows if "valid" in row)})
        writer.writerow(average_row)


def make_batch(
    sample: Dict[str, Any],
    planner: ReCogDriveDiffusionPlanner,
    device: torch.device,
    dtype: torch.dtype,
    *,
    anchor_index: Optional[Dict[str, Path]] = None,
    two_expert_corruption_mode: str = "normal",
) -> Tuple[torch.Tensor, BatchFeature]:
    global _WARNED_TRAIN_ONLY_TARGET_KEYS
    use_last_rd = bool(getattr(planner.config, "use_last_rd", False))
    use_last_vla = bool(getattr(planner.config, "use_last_vla", False))
    use_two_expert_slots = bool(getattr(planner.config, "use_two_expert_slots", False))
    require_jepa = bool((planner.config.use_expert_features or use_last_rd or use_last_vla) and planner.config.use_jepa)
    require_vggt = bool((planner.config.use_expert_features or use_last_rd or use_last_vla) and planner.config.use_vggt)
    strict_last_vla_full_geometry = bool(
        use_last_vla
        and getattr(planner.config, "last_vla_require_full_geometry", False)
        and not getattr(planner.config, "last_vla_allow_patch_geometry_fallback", False)
    )
    validate_sample_payload(
        sample,
        require_jepa=require_jepa,
        require_vggt=require_vggt,
        require_targets=False,
        use_last_rd=use_last_rd,
        use_last_vla=use_last_vla,
        expected_jepa_tokens=int(getattr(planner.config, "num_jepa_tokens", 12)),
        expected_vggt_tokens=int(getattr(planner.config, "num_vggt_tokens", 12)),
        expected_geometry_tokens=int(getattr(planner.config, "num_geometry_tokens", 12)),
        geometry_teacher_dim=int(getattr(planner.config, "last_vla_geometry_teacher_dim", 512)),
        strict_last_vla_full_geometry=strict_last_vla_full_geometry,
    )
    if "last_hidden_state" not in sample:
        raise KeyError("Evaluation sample is missing last_hidden_state. Build a VLM-hidden chunk first.")
    vl_features = sample["last_hidden_state"].float().unsqueeze(0).to(device=device, dtype=dtype)
    his = sample["history_trajectory"].float().view(1, -1).to(device=device, dtype=dtype)
    status = sample["status_feature"].float().unsqueeze(0).to(device=device, dtype=dtype)
    data = {"his_traj": his, "status_feature": status}
    if use_two_expert_slots:
        if "two_expert_h_dyn" not in sample or "two_expert_h_geo" not in sample:
            raise KeyError("two_expert_slot evaluation requires two_expert_h_dyn and two_expert_h_geo in the hidden cache.")
        h_dyn = sample["two_expert_h_dyn"]
        h_geo = sample["two_expert_h_geo"]
        if not isinstance(h_dyn, torch.Tensor) or tuple(h_dyn.shape[:2]) != (3, 12):
            raise ValueError(f"two_expert_h_dyn must have shape [3,12,D], got {tuple(h_dyn.shape) if isinstance(h_dyn, torch.Tensor) else type(h_dyn).__name__}.")
        if not isinstance(h_geo, torch.Tensor) or h_geo.ndim != 2 or h_geo.shape[0] != 12:
            raise ValueError(f"two_expert_h_geo must have shape [12,D], got {tuple(h_geo.shape) if isinstance(h_geo, torch.Tensor) else type(h_geo).__name__}.")
        data["two_expert_h_dyn"] = h_dyn.float().unsqueeze(0).to(device=device, dtype=dtype)
        data["two_expert_h_geo"] = h_geo.float().unsqueeze(0).to(device=device, dtype=dtype)
        data["two_expert_corruption_mode"] = str(two_expert_corruption_mode)
    if require_jepa:
        data["jepa_context_tokens"] = sample["jepa_context_tokens"].float().unsqueeze(0).to(device=device, dtype=dtype)
    if require_vggt:
        if not strict_last_vla_full_geometry:
            data["vggt_context_tokens"] = sample["vggt_context_tokens"].float().unsqueeze(0).to(device=device, dtype=dtype)

    residual_anchor_source = str(getattr(planner.config, "last_vla_residual_anchor_source", "vlm_text_traj"))
    requires_vlm_text_anchor = bool(
        getattr(planner.config, "last_vla_use_residual_diffusion", False)
        and residual_anchor_source == "vlm_text_traj"
        and getattr(planner.config, "last_vla_require_residual_anchor", True)
    )
    has_vlm_text_anchor = isinstance(sample.get("vlm_text_trajectory_norm"), torch.Tensor) or isinstance(sample.get("vlm_text_trajectory"), torch.Tensor)
    if requires_vlm_text_anchor and anchor_index and not has_vlm_text_anchor:
        token = str(sample.get("sample_token") or sample.get("scene_token") or "")
        anchor_path = anchor_index.get(token)
        if anchor_path is not None:
            anchor = load_sample(anchor_path)
            sample = {**sample, **{key: anchor[key] for key in ("vlm_text_trajectory_norm", "vlm_text_trajectory", "vlm_text_parse_ok") if key in anchor}}
            has_vlm_text_anchor = isinstance(sample.get("vlm_text_trajectory_norm"), torch.Tensor) or isinstance(sample.get("vlm_text_trajectory"), torch.Tensor)
    if requires_vlm_text_anchor and not has_vlm_text_anchor:
        raise KeyError(
            "Residual-anchor evaluation requires navtest cache or --vlm-text-anchor-cache-root "
            "with vlm_text_trajectory_norm or vlm_text_trajectory."
        )

    if use_last_vla:
        for key in ("vggt_geometry_tokens", "vggt_depth_tokens", "vggt_pointmap_tokens", "vggt_camera_tokens"):
            if key in sample and isinstance(sample[key], torch.Tensor):
                data[key] = sample[key].float().unsqueeze(0).to(device=device, dtype=dtype)
        mode_code = sample.get("vggt_geometry_mode_code")
        if isinstance(mode_code, torch.Tensor):
            data["vggt_geometry_mode_code"] = mode_code.view(1).to(device=device)
        elif "vggt_geometry_mode" in sample:
            raw_mode = sample.get("vggt_geometry_mode")
            if isinstance(raw_mode, bytes):
                raw_mode = raw_mode.decode("utf-8", errors="replace")
            mode_map = {"missing": -1, "no_geometry": 0, "patch_fallback": 1, "full_geometry": 2}
            data["vggt_geometry_mode_code"] = torch.tensor([mode_map.get(str(raw_mode), -1)], device=device, dtype=torch.long)
    for key in ("vlm_text_trajectory_norm", "vlm_text_trajectory", "vlm_text_parse_ok"):
        if key in sample and isinstance(sample[key], torch.Tensor):
            data[key] = sample[key].float().unsqueeze(0).to(device=device, dtype=dtype)
    target_keys = [key for key in ("jepa_target_tokens", "vggt_target_tokens", "vggt_geometry_target_tokens") if key in sample]
    if target_keys and not _WARNED_TRAIN_ONLY_TARGET_KEYS:
        warnings.warn(f"Evaluation sample contains train-only target keys {target_keys}; they are not passed to get_action.", RuntimeWarning)
        _WARNED_TRAIN_ONLY_TARGET_KEYS = True
    return vl_features, BatchFeature(data=data)


def build_online_vlm(args: argparse.Namespace, device: torch.device):
    if args.online_vlm_path is None:
        return None, None
    from scripts.build_recogdrive_hidden_cache_with_lora import compute_hidden, load_backbone

    runtime_args = SimpleNamespace(
        vlm_path=args.online_vlm_path,
        vlm_type=args.online_vlm_type,
        device="cuda" if device.type == "cuda" else "cpu",
        vlm_lora_adapter=None,
        vlm_lora_adapter_dir=args.online_vlm_lora_adapter_dir,
        lora_r=None,
        lora_alpha=None,
        lora_dropout=None,
        lora_target_modules=None,
        lora_use_rslora=None,
        lora_use_dora=None,
        allow_lora_config_override=False,
    )
    return load_backbone(runtime_args), compute_hidden


def main() -> int:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards).")
    if args.feature_source in {"chunk", "disk", "chunk_or_online"}:
        chunks = chunk_dirs(args)
    else:
        chunks = []
    if args.feature_source == "online":
        raise NotImplementedError("Online eval feature extraction is reserved for a later integration; use chunk features first.")
    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_precision(args.precision) if device.type == "cuda" else torch.float32
    planner = build_planner(cfg).to(device)
    if dtype != torch.float32:
        planner = planner.to(dtype=dtype)
    load_checkpoint(planner, args.checkpoint)
    planner.eval()
    online_backbone, online_compute_hidden = build_online_vlm(args, device)
    anchor_index = load_vlm_text_anchor_index(args.vlm_text_anchor_cache_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = []
    pdm_rows: List[Dict[str, Any]] = []
    l1_values = []
    pdm_metric_values: Dict[str, List[float]] = {
        "score": [],
        "no_at_fault_collisions": [],
        "drivable_area_compliance": [],
        "ego_progress": [],
        "time_to_collision_within_bound": [],
        "comfort": [],
        "driving_direction_compliance": [],
    }
    metric_cache_loader = build_metric_cache_loader(args.metric_cache_dir) if args.metric_cache_dir is not None else None
    pdm_tools = build_pdm_tools() if metric_cache_loader is not None else None
    missing_metric_cache = 0
    failed_pdm = 0
    sample_token_filter = load_sample_token_filter(args.sample_token_file)
    paths = sample_paths(chunks, None if sample_token_filter is not None else args.max_samples)
    if sample_token_filter is not None:
        paths = [
            item
            for item in paths
            if str(item[2].get("sample_token") or item[1].stem) in sample_token_filter
        ]
        if args.max_samples is not None:
            paths = paths[:args.max_samples]
    if args.num_shards > 1:
        paths = [item for idx, item in enumerate(paths) if idx % args.num_shards == args.shard_index]
    for idx, (chunk_dir, path, index_record) in enumerate(paths):
        sample = load_eval_sample(path)
        sample_token = str(sample.get("sample_token", index_record.get("sample_token", path.stem)))
        if online_backbone is not None and online_compute_hidden is not None:
            sample = dict(sample)
            sample["last_hidden_state"] = online_compute_hidden(
                online_backbone,
                sample,
                device="cuda" if device.type == "cuda" else "cpu",
            )
        vl_features, action_input = make_batch(
            sample,
            planner,
            device,
            dtype,
            anchor_index=anchor_index,
            two_expert_corruption_mode=args.two_expert_corruption_mode,
        )
        init_actions = make_initial_noise(planner, vl_features, sample_token, args.initial_noise_seed)
        with torch.no_grad():
            output = planner.get_action(
                vl_features,
                action_input,
                init_actions=init_actions,
                deterministic=args.deterministic,
            )
        if args.trajectory_output_key not in output:
            available_keys = ", ".join(sorted(output.keys()))
            raise KeyError(f"Missing requested trajectory output {args.trajectory_output_key!r}; available keys: {available_keys}")
        pred = output[args.trajectory_output_key].detach().float().cpu().squeeze(0)
        if not torch.isfinite(pred).all():
            raise RuntimeError(f"Non-finite prediction for {path}")
        record = {
            "path": str(path),
            "chunk": str(chunk_dir),
            "scene_token": str(sample.get("scene_token", path.stem)),
            "sample_token": sample_token,
            "trajectory_output_key": args.trajectory_output_key,
            "pred_traj": pred.tolist(),
            "pdm_valid": None,
        }
        pdm_row: Dict[str, Any] = {
            "sample_token": record["sample_token"],
            "scene_token": record["scene_token"],
            "chunk": chunk_dir.name,
            "valid": None,
        }
        if "trajectory" in sample:
            target = sample["trajectory"].float()
            l1 = torch.nn.functional.l1_loss(pred, target).item()
            record["trajectory_l1"] = l1
            pdm_row["trajectory_l1"] = l1
            l1_values.append(l1)
        if metric_cache_loader is not None and pdm_tools is not None:
            token = record["sample_token"]
            if token not in metric_cache_loader.metric_cache_paths:
                missing_metric_cache += 1
                record["pdm_valid"] = False
                record["pdm_error"] = "missing_metric_cache"
                pdm_row.update({"valid": False, "error": "missing_metric_cache"})
            else:
                try:
                    metric_cache = metric_cache_loader.get_from_token(token)
                    model_trajectory = Trajectory(poses=pred.numpy())
                    future_sampling, simulator, scorer = pdm_tools
                    result = pdm_score(
                        metric_cache=metric_cache,
                        model_trajectory=model_trajectory,
                        future_sampling=future_sampling,
                        simulator=simulator,
                        scorer=scorer,
                    )
                    result_dict = asdict(result)
                    record["pdm_valid"] = True
                    record["pdm"] = result_dict
                    pdm_row.update({"valid": True, **result_dict})
                    for key, value in result_dict.items():
                        pdm_metric_values[key].append(float(value))
                except Exception as exc:
                    failed_pdm += 1
                    record["pdm_valid"] = False
                    record["pdm_error"] = repr(exc)
                    pdm_row.update({"valid": False, "error": repr(exc)})
            pdm_rows.append(pdm_row)
        predictions.append(record)
    pdm_averages = {key: mean_or_none(values) for key, values in pdm_metric_values.items()}
    metrics = {
        "split": args.split,
        "feature_source": args.feature_source,
        "checkpoint": str(args.checkpoint),
        "precision": args.precision,
        "deterministic": bool(args.deterministic),
        "initial_noise_seed": args.initial_noise_seed,
        "trajectory_output_key": args.trajectory_output_key,
        "two_expert_corruption_mode": args.two_expert_corruption_mode,
        "online_vlm_path": str(args.online_vlm_path) if args.online_vlm_path is not None else None,
        "online_vlm_lora_adapter_dir": str(args.online_vlm_lora_adapter_dir) if args.online_vlm_lora_adapter_dir is not None else None,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "chunk_cache_dirs": [str(chunk) for chunk in chunks],
        "metric_cache_dir": str(args.metric_cache_dir) if args.metric_cache_dir is not None else None,
        "num_samples": len(predictions),
        "target_teacher_tokens_disabled_in_eval": True,
        "trajectory_l1": sum(l1_values) / len(l1_values) if l1_values else None,
        "num_pdm_valid": len(pdm_metric_values["score"]),
        "num_pdm_missing_metric_cache": missing_metric_cache,
        "num_pdm_failed": failed_pdm,
        "pdm_score": pdm_averages["score"],
        "PDMS": pdm_averages["score"],
        "NC": pdm_averages["no_at_fault_collisions"],
        "DAC": pdm_averages["drivable_area_compliance"],
        "TTC": pdm_averages["time_to_collision_within_bound"],
        "comfort": pdm_averages["comfort"],
        "EP": pdm_averages["ego_progress"],
        "DDC": pdm_averages["driving_direction_compliance"],
    }
    if args.metric_cache_dir is not None and metrics["num_pdm_valid"] == 0:
        raise RuntimeError(
            f"PDM was requested with --metric-cache-dir={args.metric_cache_dir}, "
            "but zero valid PDM rows were produced."
        )
    (args.output_dir / "predictions.json").write_text(json.dumps(predictions, indent=2) + "\n")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    if pdm_rows:
        write_pdm_csv(args.output_dir / "pdm_results.csv", pdm_rows, pdm_averages)
    report_lines = [
        "# ReCogDrive Evaluation Report",
        "",
        f"Config: `{args.config}`",
        f"Checkpoint: `{args.checkpoint}`",
        f"Split: `{args.split}`",
        f"Samples: {len(predictions)}",
        f"Trajectory output key: `{args.trajectory_output_key}`",
        f"Initial noise seed: {args.initial_noise_seed}",
        f"Target teacher tokens disabled in eval: {metrics['target_teacher_tokens_disabled_in_eval']}",
        f"Trajectory L1: {metrics.get('trajectory_l1')}",
        f"PDM valid samples: {metrics['num_pdm_valid']}",
        f"PDM score: {metrics.get('pdm_score')}",
        f"NC: {metrics.get('NC')}",
        f"DAC: {metrics.get('DAC')}",
        f"TTC: {metrics.get('TTC')}",
        f"Comfort: {metrics.get('comfort')}",
        f"EP: {metrics.get('EP')}",
        f"DDC: {metrics.get('DDC')}",
        "",
        "PDM metrics are computed with NAVSIM metric cache when --metric-cache-dir is supplied. "
        "Without metric cache this report only contains same-pipeline trajectory L1.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
