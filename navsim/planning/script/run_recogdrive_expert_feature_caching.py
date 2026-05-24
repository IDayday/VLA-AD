from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[3]
SCENE_FILTER_DIR = REPO_ROOT / "navsim/planning/script/config/common/train_test_split/scene_filter"
EXPERT_CACHE_FILE_NAME = "expert_features.pt"


@dataclass(frozen=True)
class ExpertSampleRecord:
    """Minimal per-sample information needed to build external expert caches."""

    token: str
    log_name: str
    current_image_paths: List[str]
    future_image_paths: List[str]


@dataclass
class ExpertBatch:
    tokens: List[str]
    log_names: List[str]
    current_image_paths: List[List[str]]
    future_image_paths: List[List[str]]


class ExpertSampleDataset(Dataset):
    def __init__(self, records: List[ExpertSampleRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ExpertSampleRecord:
        return self.records[index]


class ExpertFeatureBackend(nn.Module):
    """Backend interface for frozen JEPA/VGGT feature extraction."""

    jepa_dim: int
    vggt_dim: int

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype

    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def encode_current(self, batch: ExpertBatch) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @torch.no_grad()
    def encode_future_targets(self, batch: ExpertBatch) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class DummyExpertFeatureBackend(ExpertFeatureBackend):
    """
    Deterministic placeholder backend.

    It produces dense pseudo-token features from stable hashes of sample tokens,
    image paths, and model identifiers. The rest of the script is structured so a
    real JEPA/VGGT backend can replace this class without changing cache layout.
    """

    def __init__(
        self,
        jepa_model_path: str,
        vggt_model_path: str,
        device: torch.device,
        dtype: torch.dtype,
        jepa_dim: int = 768,
        vggt_dim: int = 2048,
        dense_jepa_tokens: int = 64,
        dense_vggt_tokens: int = 128,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.jepa_model_path = jepa_model_path
        self.vggt_model_path = vggt_model_path
        self.jepa_dim = jepa_dim
        self.vggt_dim = vggt_dim
        self.dense_jepa_tokens = dense_jepa_tokens
        self.dense_vggt_tokens = dense_vggt_tokens

    @staticmethod
    def _seed(parts: Iterable[str]) -> int:
        digest = hashlib.sha256("||".join(parts).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**63 - 1)

    def _make_dense(
        self,
        token: str,
        image_paths: List[str],
        model_path: str,
        stream: str,
        target: bool,
        dense_tokens: int,
        dim: int,
    ) -> torch.Tensor:
        seed = self._seed([stream, "target" if target else "current", model_path, token, *image_paths])
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        dense = torch.randn(dense_tokens, dim, generator=generator, dtype=torch.float32)
        return dense.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def _encode(self, batch: ExpertBatch, target: bool) -> Dict[str, torch.Tensor]:
        path_batches = batch.future_image_paths if target else batch.current_image_paths
        jepa = [
            self._make_dense(
                token=token,
                image_paths=image_paths,
                model_path=self.jepa_model_path,
                stream="jepa",
                target=target,
                dense_tokens=self.dense_jepa_tokens,
                dim=self.jepa_dim,
            )
            for token, image_paths in zip(batch.tokens, path_batches)
        ]
        vggt = [
            self._make_dense(
                token=token,
                image_paths=image_paths,
                model_path=self.vggt_model_path,
                stream="vggt",
                target=target,
                dense_tokens=self.dense_vggt_tokens,
                dim=self.vggt_dim,
            )
            for token, image_paths in zip(batch.tokens, path_batches)
        ]
        return {
            "jepa": torch.stack(jepa, dim=0),
            "vggt": torch.stack(vggt, dim=0),
        }

    @torch.no_grad()
    def encode_current(self, batch: ExpertBatch) -> Dict[str, torch.Tensor]:
        return self._encode(batch, target=False)

    @torch.no_grad()
    def encode_future_targets(self, batch: ExpertBatch) -> Dict[str, torch.Tensor]:
        return self._encode(batch, target=True)


class RealExpertFeatureBackend(ExpertFeatureBackend):
    """Placeholder for real JEPA/VGGT adapters."""

    def __init__(
        self,
        jepa_model_path: str,
        vggt_model_path: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        raise NotImplementedError(
            "Real JEPA/VGGT extraction is not wired yet. Use --teacher-backend dummy, "
            "or implement model loading and image preprocessing in RealExpertFeatureBackend."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute per-sample JEPA/VGGT expert token caches for ReCogDrive."
    )
    parser.add_argument("--split", required=True, help="NAVSIM data split, e.g. trainval or test.")
    parser.add_argument(
        "--scene-filter",
        required=True,
        help="SceneFilter YAML name under navsim config, or an explicit YAML path.",
    )
    parser.add_argument("--output-cache-dir", required=True, type=Path)
    parser.add_argument("--jepa-checkpoint-or-model-path", required=True)
    parser.add_argument("--vggt-checkpoint-or-model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-jepa-tokens", type=int, default=4)
    parser.add_argument("--num-vggt-tokens", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--compute-future-targets", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--navsim-log-path", type=Path, default=None)
    parser.add_argument("--sensor-blobs-path", type=Path, default=None)
    parser.add_argument("--camera", default="cam_f0", help="Camera key to feed teachers, default: cam_f0.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing expert cache files.")
    parser.add_argument("--teacher-backend", choices=("dummy", "real"), default="dummy")
    parser.add_argument("--dummy-jepa-dim", type=int, default=768)
    parser.add_argument("--dummy-vggt-dim", type=int, default=2048)
    parser.add_argument("--dummy-dense-jepa-tokens", type=int, default=64)
    parser.add_argument("--dummy-dense-vggt-tokens", type=int, default=128)
    return parser.parse_args()


def dtype_from_precision(precision: str) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[precision]


def resolve_default_data_path(split: str, leaf: str, arg_name: str) -> Path:
    data_root = os.environ.get("OPENSCENE_DATA_ROOT")
    if not data_root:
        raise ValueError(
            f"{arg_name} was not set and OPENSCENE_DATA_ROOT is undefined."
        )
    return Path(data_root) / leaf / split


def resolve_scene_filter_path(scene_filter: str) -> Path:
    path = Path(scene_filter)
    if path.is_file():
        return path

    if path.suffix != ".yaml":
        path = path.with_suffix(".yaml")
    candidate = SCENE_FILTER_DIR / path.name
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"Could not resolve scene filter '{scene_filter}'. Tried explicit path and {candidate}."
    )


def load_scene_filter(scene_filter: str, max_samples: Optional[int]):
    from navsim.common.dataclasses import SceneFilter

    path = resolve_scene_filter_path(scene_filter)
    with path.open("r") as fp:
        raw_cfg = yaml.safe_load(fp) or {}
    raw_cfg.pop("_target_", None)
    raw_cfg.pop("_convert_", None)
    if max_samples is not None:
        raw_cfg["max_scenes"] = max_samples
    return SceneFilter(**raw_cfg), path


def frame_camera_path(frame: Dict, sensor_blobs_path: Path, camera_key: str) -> str:
    camera_key = camera_key.lower()
    for raw_key, camera_dict in frame["cams"].items():
        if raw_key.lower() == camera_key:
            return str(sensor_blobs_path / camera_dict["data_path"])
    raise KeyError(f"Camera '{camera_key}' not found in frame cameras: {list(frame['cams'].keys())}")


def build_sample_records(
    scene_filter_name_or_path: str,
    navsim_log_path: Path,
    sensor_blobs_path: Path,
    camera_key: str,
    max_samples: Optional[int],
) -> tuple[List[ExpertSampleRecord], Path, Dict]:
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SensorConfig

    scene_filter, scene_filter_path = load_scene_filter(scene_filter_name_or_path, max_samples)
    scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=navsim_log_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_all_sensors(include=True),
        load_image_path=True,
    )

    records: List[ExpertSampleRecord] = []
    num_history = scene_filter.num_history_frames
    for token in scene_loader.tokens:
        frame_list = scene_loader.scene_frames_dicts[token]
        current_frames = frame_list[:num_history]
        future_frames = frame_list[num_history:]
        current_paths = [frame_camera_path(frame, sensor_blobs_path, camera_key) for frame in current_frames]
        future_paths = [frame_camera_path(frame, sensor_blobs_path, camera_key) for frame in future_frames]
        records.append(
            ExpertSampleRecord(
                token=token,
                log_name=frame_list[num_history - 1]["log_name"],
                current_image_paths=current_paths,
                future_image_paths=future_paths,
            )
        )

    return records, scene_filter_path, asdict(scene_filter)


def collate_records(records: List[ExpertSampleRecord]) -> ExpertBatch:
    return ExpertBatch(
        tokens=[record.token for record in records],
        log_names=[record.log_name for record in records],
        current_image_paths=[record.current_image_paths for record in records],
        future_image_paths=[record.future_image_paths for record in records],
    )


def pool_dense_tokens(dense_tokens: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """
    Pools dense features [B, N, D] to fixed expert tokens [B, K, D].

    ReCogDriveFeatureBuilder consumes per-sample tensors as [K, D].
    """
    if dense_tokens.ndim != 3:
        raise ValueError(f"Expected dense tokens [B, N, D], got {tuple(dense_tokens.shape)}.")
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}.")
    pooled = F.adaptive_avg_pool1d(dense_tokens.transpose(1, 2), num_tokens)
    return pooled.transpose(1, 2).contiguous()


def build_backend(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> ExpertFeatureBackend:
    if args.teacher_backend == "dummy":
        backend = DummyExpertFeatureBackend(
            jepa_model_path=args.jepa_checkpoint_or_model_path,
            vggt_model_path=args.vggt_checkpoint_or_model_path,
            device=device,
            dtype=dtype,
            jepa_dim=args.dummy_jepa_dim,
            vggt_dim=args.dummy_vggt_dim,
            dense_jepa_tokens=args.dummy_dense_jepa_tokens,
            dense_vggt_tokens=args.dummy_dense_vggt_tokens,
        )
    else:
        backend = RealExpertFeatureBackend(
            jepa_model_path=args.jepa_checkpoint_or_model_path,
            vggt_model_path=args.vggt_checkpoint_or_model_path,
            device=device,
            dtype=dtype,
        )
    backend.freeze()
    return backend


def cache_file_path(output_cache_dir: Path, log_name: str, token: str) -> Path:
    return output_cache_dir / log_name / token / EXPERT_CACHE_FILE_NAME


def save_cache_file(path: Path, payload: Dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def write_metadata(path: Path, metadata: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        json.dump(metadata, fp, indent=2, sort_keys=True)
        fp.write("\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")

    navsim_log_path = args.navsim_log_path or resolve_default_data_path(
        args.split, "navsim_logs", "--navsim-log-path"
    )
    sensor_blobs_path = args.sensor_blobs_path or resolve_default_data_path(
        args.split, "sensor_blobs", "--sensor-blobs-path"
    )
    device = torch.device(args.device)
    dtype = dtype_from_precision(args.precision)

    records, scene_filter_path, scene_filter_cfg = build_sample_records(
        scene_filter_name_or_path=args.scene_filter,
        navsim_log_path=navsim_log_path,
        sensor_blobs_path=sensor_blobs_path,
        camera_key=args.camera,
        max_samples=args.max_samples,
    )
    if not records:
        raise RuntimeError("No NAVSIM samples matched the requested split/scene_filter.")
    logger.info("Loaded %d samples for expert feature caching.", len(records))

    if args.compute_future_targets:
        missing_future = [record.token for record in records if not record.future_image_paths]
        if missing_future:
            raise ValueError(
                "compute_future_targets=True requires future frames for every sample. "
                f"First missing token: {missing_future[0]}"
            )

    backend = build_backend(args, device=device, dtype=dtype)
    dataloader = DataLoader(
        ExpertSampleDataset(records),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_records,
    )

    num_written = 0
    num_skipped = 0
    with torch.no_grad():
        for batch in dataloader:
            existing = [
                cache_file_path(args.output_cache_dir, log_name, token).is_file()
                for log_name, token in zip(batch.log_names, batch.tokens)
            ]
            if all(existing) and not args.overwrite:
                num_skipped += len(existing)
                continue

            current_dense = backend.encode_current(batch)
            jepa_tokens = pool_dense_tokens(current_dense["jepa"], args.num_jepa_tokens)
            vggt_tokens = pool_dense_tokens(current_dense["vggt"], args.num_vggt_tokens)

            target_tokens: Dict[str, torch.Tensor] = {}
            if args.compute_future_targets:
                future_dense = backend.encode_future_targets(batch)
                target_tokens["jepa_target_tokens"] = pool_dense_tokens(
                    future_dense["jepa"], args.num_jepa_tokens
                )
                target_tokens["vggt_target_tokens"] = pool_dense_tokens(
                    future_dense["vggt"], args.num_vggt_tokens
                )

            for index, (token, log_name) in enumerate(zip(batch.tokens, batch.log_names)):
                path = cache_file_path(args.output_cache_dir, log_name, token)
                if path.is_file() and not args.overwrite:
                    num_skipped += 1
                    continue

                payload = {
                    "jepa_tokens": jepa_tokens[index].detach().cpu(),
                    "vggt_tokens": vggt_tokens[index].detach().cpu(),
                }
                for key, value in target_tokens.items():
                    payload[key] = value[index].detach().cpu()
                save_cache_file(path, payload)
                num_written += 1

    metadata = {
        "backend": args.teacher_backend,
        "split": args.split,
        "scene_filter": str(scene_filter_path),
        "scene_filter_config": scene_filter_cfg,
        "navsim_log_path": str(navsim_log_path),
        "sensor_blobs_path": str(sensor_blobs_path),
        "camera": args.camera,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "jepa_checkpoint_or_model_path": args.jepa_checkpoint_or_model_path,
        "vggt_checkpoint_or_model_path": args.vggt_checkpoint_or_model_path,
        "jepa_dim": backend.jepa_dim,
        "vggt_dim": backend.vggt_dim,
        "num_jepa_tokens": args.num_jepa_tokens,
        "num_vggt_tokens": args.num_vggt_tokens,
        "precision": args.precision,
        "compute_future_targets": args.compute_future_targets,
        "num_samples": len(records),
        "num_written": num_written,
        "num_skipped": num_skipped,
        "cache_file_name": EXPERT_CACHE_FILE_NAME,
        "cache_layout": "output_cache_dir/log_name/token/expert_features.pt",
    }
    write_metadata(args.output_cache_dir / "metadata.json", metadata)
    logger.info(
        "Finished expert feature caching: wrote=%d skipped=%d output=%s",
        num_written,
        num_skipped,
        args.output_cache_dir,
    )


if __name__ == "__main__":
    main()
