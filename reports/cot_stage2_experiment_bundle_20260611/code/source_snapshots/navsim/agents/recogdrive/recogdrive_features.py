from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple
import json
import os
import warnings
import torch

from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from .expert_backends import (
    DummyExpertBackend,
    EXPERT_ALL_KEYS,
    EXPERT_TARGET_KEYS,
    build_dummy_expert_backend,
    normalize_expert_feature_source,
)
from .expert_cache import iter_index

if TYPE_CHECKING:
    from navsim.common.dataclasses import AgentInput, Scene
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

try:
    from safetensors.torch import load_file as load_safetensors_file
except ImportError:  # safetensors is optional in this repo.
    load_safetensors_file = None

EXPERT_FEATURE_KEYS: Tuple[str, ...] = EXPERT_ALL_KEYS
EXPERT_TARGET_FEATURE_KEYS: Tuple[str, ...] = (
    *EXPERT_TARGET_KEYS,
    "teacher_trajectory",
    "teacher_trajectory_norm",
    "teacher_score",
    "gt_score",
    "oracle_best_of_k_score",
)
DUMMY_EXPERT_CACHE_WARNING = "Dummy cache for computation smoke tests only. Do not use for real training."

def format_number(n, decimal_places=2):
    return f"{n:+.{decimal_places}f}" if abs(round(n, decimal_places)) > 1e-2 else "0.0"

def normalize_high_command_one_hot(command: torch.Tensor) -> torch.Tensor:
    """Return the official three-way NAVSIM command: left, straight, right."""
    command = command.detach().clone().float().view(-1)
    if command.numel() == 3:
        return command
    if command.numel() == 4 and float(command[-1].abs().item()) < 1e-6:
        return command[:3]
    raise ValueError(
        "Unsupported NAVSIM driving_command shape/value for ReCogDrive. "
        f"Expected [3] left/straight/right, or legacy [4] with unused fourth slot zero; got {command.tolist()}."
    )


def stack_optional_expert_features(features: Dict[str, torch.Tensor], features_list: List[Dict[str, torch.Tensor]]) -> None:
    """Stacks optional expert features into an existing collated feature dict."""
    for key in EXPERT_FEATURE_KEYS:
        present = [key in sample_features for sample_features in features_list]
        if not any(present):
            continue
        if not all(present):
            raise KeyError(
                f"Expert feature '{key}' is present for only part of the batch. "
                "Regenerate caches or disable use_expert_features."
        )
        features[key] = torch.stack([sample_features[key] for sample_features in features_list], dim=0).cpu()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def load_expert_cache_metadata(expert_cache_dir: Optional[str | Path]) -> Optional[Dict[str, Any]]:
    """Loads expert-cache metadata.json when present."""
    if not expert_cache_dir:
        return None

    metadata_path = Path(expert_cache_dir) / "metadata.json"
    if not metadata_path.is_file():
        return None

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if not isinstance(metadata, dict):
        raise TypeError(f"Expert cache metadata {metadata_path} must contain a JSON object.")
    return metadata


def is_dummy_expert_cache(expert_cache_dir: Optional[str | Path]) -> bool:
    """Returns True when expert_cache_dir/metadata.json declares is_dummy=true."""
    metadata = load_expert_cache_metadata(expert_cache_dir)
    return bool(metadata and metadata.get("is_dummy") is True)


def assert_real_expert_cache_for_training(
    expert_cache_dir: Optional[str | Path],
    *,
    use_expert_features: bool,
    allow_dummy_expert_cache: bool = False,
    expert_feature_source: str = "none",
) -> None:
    """Fails training early if a dummy expert cache is used without an explicit override."""
    if not use_expert_features:
        return

    env_allows_dummy = _as_bool(os.environ.get("ALLOW_DUMMY_EXPERT_CACHE", False))
    dummy_allowed = _as_bool(allow_dummy_expert_cache) or env_allows_dummy
    if expert_feature_source == "dummy":
        if dummy_allowed:
            warnings.warn(
                f"Using expert_feature_source='dummy' for training. {DUMMY_EXPERT_CACHE_WARNING}",
                RuntimeWarning,
            )
            return
        raise RuntimeError(
            "Refusing to train with expert_feature_source='dummy'. "
            f"{DUMMY_EXPERT_CACHE_WARNING} Set allow_dummy_expert_cache=true "
            "or ALLOW_DUMMY_EXPERT_CACHE=true only for computation smoke tests."
        )

    if not expert_cache_dir or not is_dummy_expert_cache(expert_cache_dir):
        return

    if dummy_allowed:
        warnings.warn(
            f"Using dummy expert cache for training: {expert_cache_dir}. {DUMMY_EXPERT_CACHE_WARNING}",
            RuntimeWarning,
        )
        return

    raise RuntimeError(
        f"Refusing to train with dummy expert cache: {expert_cache_dir}. "
        f"{DUMMY_EXPERT_CACHE_WARNING} Set allow_dummy_expert_cache=true "
        "or ALLOW_DUMMY_EXPERT_CACHE=true only for computation smoke tests."
    )


def warn_if_dummy_expert_cache(
    expert_cache_dir: Optional[str | Path],
    *,
    use_expert_features: bool,
    context: str = "evaluation",
) -> None:
    """Warns loudly when evaluation/inference is pointed at a dummy expert cache."""
    if use_expert_features and expert_cache_dir and is_dummy_expert_cache(expert_cache_dir):
        warnings.warn(
            f"{context} is using dummy expert cache: {expert_cache_dir}. "
            f"{DUMMY_EXPERT_CACHE_WARNING}",
            RuntimeWarning,
        )


def load_expert_cache_sample(
    expert_cache_dir: str | Path,
    sample_token: str,
    *,
    log_name: str = "",
    num_jepa_tokens: int = 12,
    num_vggt_tokens: int = 12,
    allow_expert_target_features: bool = True,
) -> Dict[str, torch.Tensor]:
    """Loads one cached expert sample by token, including flat dummy-cache files.

    The loader uses the same path candidates and validation as ReCogDriveFeatureBuilder.
    For the dummy cache generated by scripts/create_dummy_expert_cache.py, call:

        load_expert_cache_sample(cache_dir, "sample_000000")
    """
    builder = ReCogDriveFeatureBuilder(
        cache_hidden_state=False,
        use_expert_features=True,
        expert_feature_source="chunk",
        expert_cache_dir=str(expert_cache_dir),
        num_jepa_tokens=num_jepa_tokens,
        num_vggt_tokens=num_vggt_tokens,
        allow_expert_target_features=allow_expert_target_features,
    )
    return builder._load_expert_features(log_name, sample_token)


class ReCogDriveFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self,
                 cache_hidden_state: bool = True,
                 model_type: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 device: str = "cuda",
                 cache_mode: bool = False,
                 use_expert_features: bool = False,
                 expert_feature_source: str = "none",
                 expert_cache_dir: Optional[str] = None,
                 num_jepa_tokens: int = 12,
                 num_vggt_tokens: int = 12,
                 num_geometry_tokens: Optional[int] = None,
                 allow_expert_target_features: bool = False,
                 use_jepa: bool = True,
                 use_vggt: bool = True,
                 jepa_dim: int = 1024,
                 vggt_dim: int = 2048,
                 geometry_teacher_dim: Optional[int] = None,
                 dummy_expert_seed: int = 0, ):
        """
        Initializes the feature builder.

        Args:
            cache_hidden_state (bool): If True, operates in online mode, initializes the backbone,
                                       and computes the hidden state. If False, operates in offline
                                       mode, does not initialize the backbone, and returns
                                       pre-computable tensors, including a tensorized representation
                                       of the image file path.
            model_type (str, optional): The type of model to load ('internvl' or 'qwen'). Required if cache_hidden_state is True.
            checkpoint_path (str, optional): Path to the model checkpoint. Required if cache_hidden_state is True.
            device (str): The device to load the model onto.
        """
        super().__init__()
        self.cache_hidden_state = cache_hidden_state
        self.backbone = None
        self.cache_mode = cache_mode
        self.device = torch.device(device)
        self.use_expert_features = use_expert_features
        self.expert_feature_source = normalize_expert_feature_source(
            expert_feature_source,
            use_expert_features=use_expert_features,
            expert_cache_dir=expert_cache_dir,
        )
        self.expert_cache_dir = Path(expert_cache_dir) if expert_cache_dir else None
        self._expert_cache_index_by_log_token: Optional[Dict[Tuple[str, str], Path]] = None
        self._expert_cache_index_by_token: Optional[Dict[str, Path]] = None
        self.num_jepa_tokens = num_jepa_tokens
        self.num_vggt_tokens = num_vggt_tokens
        self.num_geometry_tokens = int(num_geometry_tokens) if num_geometry_tokens is not None else int(num_vggt_tokens)
        self.allow_expert_target_features = allow_expert_target_features
        self.use_jepa = use_jepa
        self.use_vggt = use_vggt
        self.jepa_dim = jepa_dim
        self.vggt_dim = vggt_dim
        self.geometry_teacher_dim = int(geometry_teacher_dim) if geometry_teacher_dim is not None else int(vggt_dim)
        self.dummy_expert_seed = dummy_expert_seed
        self._dummy_expert_backend: Optional[DummyExpertBackend] = None

        if self.use_expert_features and self.expert_feature_source in {"chunk", "disk"} and self.expert_cache_dir is None:
            raise ValueError("use_expert_features=True with chunk/disk features requires expert_cache_dir to be set.")
        if self.use_expert_features and self.expert_feature_source == "online":
            raise NotImplementedError("expert_feature_source='online' is reserved for future JEPA/VGGT teacher integration.")
        if self.use_expert_features and self.expert_feature_source == "dummy":
            self._dummy_expert_backend = build_dummy_expert_backend(
                num_jepa_tokens=self.num_jepa_tokens,
                num_vggt_tokens=self.num_vggt_tokens,
                jepa_dim=self.jepa_dim,
                vggt_dim=self.vggt_dim,
                use_jepa=self.use_jepa,
                use_vggt=self.use_vggt,
                seed=self.dummy_expert_seed,
            )

        if self.cache_hidden_state and self.cache_mode:
            if not model_type or not checkpoint_path:
                raise ValueError("In online mode (cache_hidden_state=True), `model_type` and `checkpoint_path` must be provided.")
            from .recogdrive_backbone import RecogDriveBackbone

            self.backbone = RecogDriveBackbone(
                model_type=model_type,
                checkpoint_path=checkpoint_path,
                device=device
            )

    def get_unique_name(self) -> str:
        return "internvl_feature"

    def _expert_cache_candidates(self, log_name: str, token: str) -> List[Path]:
        """
        Returns candidate cache files using the NAVSIM cache convention.

        Canonical layout:
            expert_cache_dir / log_name / token / expert_features.pt

        This mirrors the existing ReCogDrive feature-cache layout:
            cache_path / log_name / token / internvl_feature.gz

        Chunk-cache roots are resolved through their index.jsonl files by
        _expert_cache_index_candidates.

        Safetensors files are used only when safetensors is importable; .pt files
        are always supported.
        """
        assert self.expert_cache_dir is not None

        base_dir = self.expert_cache_dir / log_name / token
        supported_suffixes = [".pt"]
        if load_safetensors_file is not None:
            supported_suffixes.insert(0, ".safetensors")

        candidates: List[Path] = []
        for suffix in supported_suffixes:
            candidates.extend([
                base_dir / f"expert_features{suffix}",
                base_dir / f"expert_feature{suffix}",
                base_dir / f"features{suffix}",
                self.expert_cache_dir / log_name / f"{token}{suffix}",
                self.expert_cache_dir / f"{token}{suffix}",
                self.expert_cache_dir / "samples" / f"{token}{suffix}",
                self.expert_cache_dir / log_name / f"{token}{suffix}",
                self.expert_cache_dir / log_name / token / f"expert_features{suffix}",
            ])
        return candidates

    @staticmethod
    def _expert_cache_chunk_dirs(cache_root: Path) -> List[Path]:
        if (cache_root / "index.jsonl").is_file():
            return [cache_root]
        if not cache_root.is_dir():
            return []
        return sorted(
            child for child in cache_root.iterdir()
            if child.is_dir() and (child / "index.jsonl").is_file()
        )

    def _ensure_expert_cache_index(self) -> None:
        if self._expert_cache_index_by_token is not None and self._expert_cache_index_by_log_token is not None:
            return

        by_token: Dict[str, Path] = {}
        by_log_token: Dict[Tuple[str, str], Path] = {}
        if self.expert_cache_dir is not None:
            for chunk_dir in self._expert_cache_chunk_dirs(self.expert_cache_dir):
                for record in iter_index(chunk_dir):
                    raw_path = record.get("path")
                    if not raw_path:
                        continue
                    sample_path = Path(raw_path)
                    token = str(record.get("sample_token") or sample_path.stem)
                    if not token:
                        continue
                    by_token.setdefault(token, sample_path)
                    record_log_name = record.get("log_name")
                    if record_log_name is not None:
                        by_log_token.setdefault((str(record_log_name), token), sample_path)

        self._expert_cache_index_by_token = by_token
        self._expert_cache_index_by_log_token = by_log_token

    def _expert_cache_index_candidates(self, log_name: str, token: str) -> List[Path]:
        if self.expert_feature_source not in {"chunk", "disk"}:
            return []
        self._ensure_expert_cache_index()
        assert self._expert_cache_index_by_token is not None
        assert self._expert_cache_index_by_log_token is not None

        candidates: List[Path] = []
        log_token_path = self._expert_cache_index_by_log_token.get((str(log_name), str(token)))
        if log_token_path is not None:
            candidates.append(log_token_path)
        token_path = self._expert_cache_index_by_token.get(str(token))
        if token_path is not None and token_path not in candidates:
            candidates.append(token_path)
        return candidates

    def _resolve_expert_cache_path(self, log_name: str, token: str) -> Path:
        candidates = self._expert_cache_candidates(log_name, token)
        for path in candidates:
            if path.is_file():
                return path

        index_candidates = self._expert_cache_index_candidates(log_name, token)
        for path in index_candidates:
            if path.is_file():
                return path

        unsupported_safetensors: List[Path] = []
        if load_safetensors_file is None and self.expert_cache_dir is not None:
            unsupported_safetensors = [
                path.with_suffix(".safetensors")
                for path in candidates
                if path.suffix == ".pt" and path.with_suffix(".safetensors").is_file()
            ]

        message = (
            "Expert feature cache not found for "
            f"log_name='{log_name}', token='{token}'. Tried: "
            + ", ".join(str(path) for path in [*candidates, *index_candidates])
        )
        if unsupported_safetensors:
            message += (
                ". Found safetensors cache files but safetensors is not installed: "
                + ", ".join(str(path) for path in unsupported_safetensors)
            )
        raise FileNotFoundError(message)

    def _cache_identity_from_agent_input(self, agent_input: AgentInput) -> Tuple[str, str]:
        log_name = getattr(agent_input, "log_name", None)
        token = getattr(agent_input, "token", None)
        if log_name and token:
            return str(log_name), str(token)

        image_path = getattr(agent_input.cameras[-1].cam_f0, "image", None)
        raise ValueError(
            "use_expert_features=True requires AgentInput.log_name and AgentInput.token. "
            "These are populated by SceneLoader/Scene.get_agent_input. "
            f"Could not resolve cache identity from image path: {image_path!r}"
        )

    @staticmethod
    def _load_expert_cache_file(path: Path) -> Dict[str, Any]:
        if path.suffix == ".safetensors":
            if load_safetensors_file is None:
                raise ImportError(
                    f"Cannot load safetensors expert cache at {path}: safetensors is not installed."
                )
            return dict(load_safetensors_file(str(path)))

        if path.suffix == ".pt":
            try:
                data = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                data = torch.load(path, map_location="cpu")
            if not isinstance(data, dict):
                raise TypeError(f"Expert cache {path} must contain a dict, got {type(data).__name__}.")
            return data

        raise ValueError(f"Unsupported expert cache file extension for {path}.")

    def _validate_expert_tensor(self, key: str, tensor: torch.Tensor, path: Path) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Expert cache {path} key '{key}' must be a torch.Tensor, got {type(tensor).__name__}."
            )
        if key == "vggt_geometry_mode_code":
            if tensor.ndim > 1:
                raise ValueError(
                    f"Expert cache {path} key '{key}' must be scalar or [1], got {tuple(tensor.shape)}."
                )
            return tensor.detach().cpu().long()
        if key in {"teacher_score", "gt_score", "oracle_best_of_k_score", "candidate_count"}:
            if tensor.ndim > 1:
                raise ValueError(
                    f"Expert cache {path} key '{key}' must be scalar or [1], got {tuple(tensor.shape)}."
                )
            return tensor.detach().cpu().float()
        if key == "vlm_text_parse_ok":
            if tensor.ndim > 1:
                raise ValueError(
                    f"Expert cache {path} key '{key}' must be scalar or [1], got {tuple(tensor.shape)}."
                )
            return tensor.detach().cpu().float()
        if key in {"teacher_trajectory", "teacher_trajectory_norm", "vlm_text_trajectory", "vlm_text_trajectory_norm"}:
            if tuple(tensor.shape) != (8, 3):
                raise ValueError(
                    f"Expert cache {path} key '{key}' must have shape [8, 3], got {tuple(tensor.shape)}."
                )
            return tensor.detach().cpu().float()
        if tensor.ndim != 2:
            raise ValueError(
                f"Expert cache {path} key '{key}' must have shape [K, D], got {tuple(tensor.shape)}."
            )

        expected_tokens: Optional[int]
        normalized_key = {"jepa_tokens": "jepa_context_tokens", "vggt_tokens": "vggt_context_tokens"}.get(key, key)
        expected_dim: Optional[int]
        if normalized_key.startswith("jepa_"):
            expected_tokens = self.num_jepa_tokens
            expected_dim = self.jepa_dim
        elif normalized_key in {
            "vggt_geometry_tokens",
            "vggt_geometry_target_tokens",
            "vggt_depth_tokens",
            "vggt_pointmap_tokens",
            "vggt_camera_tokens",
            "vggt_depth_target_tokens",
            "vggt_pointmap_target_tokens",
        }:
            expected_tokens = self.num_geometry_tokens
            expected_dim = self.geometry_teacher_dim
        elif normalized_key.startswith("vggt_"):
            expected_tokens = self.num_vggt_tokens
            expected_dim = self.vggt_dim
        else:
            expected_tokens = None
            expected_dim = None

        if expected_tokens is not None and tensor.shape[0] != expected_tokens:
            raise ValueError(
                f"Expert cache {path} key '{key}' has K={tensor.shape[0]}, "
                f"expected K={expected_tokens}. Full shape: {tuple(tensor.shape)}."
            )
        if expected_dim is not None and tensor.shape[-1] != expected_dim:
            raise ValueError(
                f"Expert cache {path} key '{key}' has D={tensor.shape[-1]}, "
                f"expected D={expected_dim}. Full shape: {tuple(tensor.shape)}."
            )

        return tensor.detach().cpu().float()

    def _load_expert_features(self, log_name: str, token: str) -> Dict[str, torch.Tensor]:
        if not self.use_expert_features:
            return {}
        if self.expert_feature_source == "none":
            return {}
        if self.expert_feature_source == "dummy":
            assert self._dummy_expert_backend is not None
            return self._dummy_expert_backend.generate_single(
                sample_key=f"{log_name}/{token}",
                include_targets=self.allow_expert_target_features,
            )
        if self.expert_feature_source == "online":
            raise NotImplementedError("expert_feature_source='online' is not wired yet.")

        path = self._resolve_expert_cache_path(log_name, token)
        data = self._load_expert_cache_file(path)

        alias_map = {"jepa_tokens": "jepa_context_tokens", "vggt_tokens": "vggt_context_tokens"}
        expert_features: Dict[str, torch.Tensor] = {}
        for key in EXPERT_FEATURE_KEYS:
            if key in data:
                output_key = alias_map.get(key, key)
                if output_key.startswith("jepa_") and not self.use_jepa:
                    continue
                if output_key in {"vggt_context_tokens", "vggt_target_tokens"} and not self.use_vggt:
                    continue
                if output_key in EXPERT_TARGET_FEATURE_KEYS and not self.allow_expert_target_features:
                    warnings.warn(
                        f"Expert cache {path} contains train-only '{output_key}', but "
                        "allow_expert_target_features=False. Dropping it to prevent future-frame leakage.",
                        RuntimeWarning,
                    )
                    continue
                expert_features.setdefault(output_key, self._validate_expert_tensor(output_key, data[key], path))

        if not expert_features:
            raise KeyError(
                f"Expert cache {path} did not contain any supported expert keys: {EXPERT_FEATURE_KEYS}."
            )

        return expert_features

    def _add_expert_features(self, features: Dict[str, torch.Tensor], agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        if not self.use_expert_features:
            return features

        log_name, token = self._cache_identity_from_agent_input(agent_input)
        features.update(self._load_expert_features(log_name, token))
        return features

    def add_expert_features_from_token_path(
        self,
        features: Dict[str, torch.Tensor],
        token_path: Path,
    ) -> Dict[str, torch.Tensor]:
        """Adds external expert features for an existing cache_path/log_name/token cache item."""
        if not self.use_expert_features:
            return features

        token_path = Path(token_path)
        log_name = token_path.parent.name
        token = token_path.name
        for key, value in self._load_expert_features(log_name, token).items():
            features.setdefault(key, value)
        return features

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:

        ego_statuses = agent_input.ego_statuses
        cameras = agent_input.cameras

        history_trajectory = torch.tensor(
            [[float(e.ego_pose[0]), float(e.ego_pose[1]), float(e.ego_pose[2])] for e in ego_statuses[:4]],
            dtype=torch.float32
        )
        raw_driving_command = torch.tensor(ego_statuses[-1].driving_command, dtype=torch.float32)
        high_command_one_hot = normalize_high_command_one_hot(raw_driving_command)
        status_feature = torch.cat([
            raw_driving_command.clone(),
            torch.tensor(ego_statuses[-1].ego_velocity, dtype=torch.float32),
            torch.tensor(ego_statuses[-1].ego_acceleration, dtype=torch.float32)
        ], dim=-1)


        if not self.cache_hidden_state:
            image_path = str(cameras[-1].cam_f0.image)

            path_as_ordinals = [ord(char) for char in image_path]

            path_tensor = torch.tensor(path_as_ordinals, dtype=torch.long)

            features = {
                "history_trajectory": history_trajectory.cpu(),
                "high_command_one_hot": high_command_one_hot.cpu(),
                "status_feature": status_feature.cpu(),
                "image_path_tensor": path_tensor.cpu(),
            }
            return self._add_expert_features(features, agent_input)
        else:
            if self.backbone is None:
                raise RuntimeError("FeatureBuilder is in online mode, but the backbone was not initialized.")
            from .utils.internvl_preprocess import load_image

            pixel_values = load_image(str(cameras[-1].cam_f0.image),max_num=12).unsqueeze(0)

            pixel_values_squeezed = pixel_values.squeeze(1)
            num_patches_list = [pv.shape[0] for pv in pixel_values_squeezed]
            pixel_values_cat = torch.cat(list(pixel_values_squeezed), dim=0)

            navigation_commands = ['turn left', 'go straight', 'turn right']
            command_str = next((navigation_commands[i] for i, v in enumerate(high_command_one_hot) if v == 1), "unknown")
            history_str = " ".join([f'   - t-{3-i}: ({format_number(history_trajectory[i, 0].item())}, {format_number(history_trajectory[i, 1].item())}, {format_number(history_trajectory[i, 2].item())})' for i in range(4)])

            prompt = f"<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n1. Visual perception from front camera view\n2. Historical motion context (last 4 timesteps):{history_str}\n3. Active navigation command: [{command_str.upper()}]"
            output_requirements = "\nOutput requirements:\n- Predict 8 future trajectory points\n- Each point format: (x:float, y:float, heading:float)\n- Use [PT, ...] to encapsulate the trajectory\n- Maintain numerical precision to 2 decimal places"
            questions = [f"{prompt}{output_requirements}"]

            outputs = self.backbone(pixel_values_cat.to(self.device), questions, num_patches_list=num_patches_list)
            last_hidden_state = outputs.hidden_states[-1]

            features = {
                "history_trajectory": history_trajectory.cpu(),
                "high_command_one_hot": high_command_one_hot.cpu(),
                "last_hidden_state": last_hidden_state.squeeze(0).float().cpu(),
                "status_feature": status_feature.cpu(),
            }
            return self._add_expert_features(features, agent_input)


class TrajectoryTargetBuilder(AbstractTargetBuilder):
    def __init__(self, trajectory_sampling: TrajectorySampling):
        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        return "trajectory_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        future_trajectory = scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses)
        return {"trajectory": torch.tensor(future_trajectory.poses)}
