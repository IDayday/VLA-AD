from typing import Any, List, Dict, Optional, Union
import os
from pathlib import Path
import warnings
import torch
from torch.optim import Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from omegaconf import DictConfig, OmegaConf
from transformers.feature_extraction_utils import BatchFeature
import math

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from .utils.internvl_preprocess import load_image
from .utils.lr_scheduler import WarmupCosLR
from .utils.utils import format_number, build_from_configs
from .expert_backends import (
    DUMMY_EXPERT_WARNING,
    DummyExpertBackend,
    build_dummy_expert_backend,
    normalize_expert_feature_source,
)
from .recogdrive_features import EXPERT_FEATURE_KEYS, ReCogDriveFeatureBuilder ,TrajectoryTargetBuilder
from .recogdrive_backbone import RecogDriveBackbone
from .recogdrive_diffusion_planner import (
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)


class ReCogDriveAgent(AbstractAgent):
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        vlm_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        cam_type: Optional[str] = 'single', 
        vlm_type: Optional[str] = 'internvl', 
        dit_type: Optional[str] = 'small', 
        sampling_method: Optional[str] = 'ddim', 
        cache_mode: bool = False, 
        cache_hidden_state: bool = True, 
        lr: float = 1e-4,
        grpo: bool = False,
        metric_cache_path: Optional[str] = '', 
        reference_policy_checkpoint: Optional[str] = '', 
        vlm_size: Optional[str] = 'small', 
        train_backbone: bool = False,
        use_expert_features: bool = False,
        expert_feature_source: str = "none",
        expert_cache_dir: Optional[str] = None,
        num_jepa_tokens: int = 4,
        num_vggt_tokens: int = 4,
        allow_expert_target_features: bool = False,
        allow_random_init: bool = True,
        allow_dummy_expert_cache: bool = False,
        use_jepa: bool = True,
        use_vggt: bool = True,
        jepa_dim: int = 0,
        vggt_dim: int = 0,
        expert_dropout: float = 0.0,
        expert_fusion_mode: str = "concat_context",
        use_expert_type_embedding: bool = True,
        use_expert_gates: bool = True,
        expert_alignment_weight: float = 0.0,
        jepa_alignment_weight: float = 0.0,
        vggt_alignment_weight: float = 0.0,
        alignment_loss_type: str = "mse",
    ):
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        self.vlm_path = vlm_path
        self.checkpoint_path = checkpoint_path
        self.vlm_type = vlm_type
        self.dit_type = dit_type
        self.cache_mode = cache_mode
        self.cache_hidden_state = cache_hidden_state
        self._lr = lr
        self.grpo = grpo
        self.backbone = None
        self.metric_cache_path = metric_cache_path
        self.reference_policy_checkpoint = reference_policy_checkpoint
        self.vlm_size = vlm_size
        self.train_backbone = train_backbone
        self.use_expert_features = use_expert_features
        self.expert_feature_source = normalize_expert_feature_source(
            expert_feature_source,
            use_expert_features=use_expert_features,
            expert_cache_dir=expert_cache_dir,
        )
        self.expert_cache_dir = expert_cache_dir
        self.num_jepa_tokens = num_jepa_tokens
        self.num_vggt_tokens = num_vggt_tokens
        self.allow_expert_target_features = allow_expert_target_features
        self.allow_random_init = allow_random_init
        self.allow_dummy_expert_cache = allow_dummy_expert_cache
        self.use_jepa = use_jepa
        self.use_vggt = use_vggt
        self.jepa_dim = jepa_dim
        self.vggt_dim = vggt_dim
        self.expert_dropout = expert_dropout
        self.expert_fusion_mode = expert_fusion_mode
        self.use_expert_type_embedding = use_expert_type_embedding
        self.use_expert_gates = use_expert_gates
        self.expert_alignment_weight = expert_alignment_weight
        self.jepa_alignment_weight = jepa_alignment_weight
        self.vggt_alignment_weight = vggt_alignment_weight
        self.alignment_loss_type = alignment_loss_type
        self._warned_random_init = False
        self._warned_dummy_features = False
        self._dummy_expert_backend: Optional[DummyExpertBackend] = None

        if self.use_expert_features and self.expert_feature_source == "real":
            raise NotImplementedError("expert_feature_source='real' is reserved for future JEPA/VGGT teacher integration.")
        if self.use_expert_features and self.expert_feature_source == "dummy":
            self._dummy_expert_backend = build_dummy_expert_backend(
                num_jepa_tokens=self.num_jepa_tokens,
                num_vggt_tokens=self.num_vggt_tokens,
                jepa_dim=self.jepa_dim,
                vggt_dim=self.vggt_dim,
                use_jepa=self.use_jepa,
                use_vggt=self.use_vggt,
            )

        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        self.device = device
        if not self.cache_hidden_state and not self.cache_mode:
            print("Agent running in 'no-cache' mode. Initializing internal backbone.")
            if not self.vlm_path or not self.vlm_type:
                raise ValueError("In 'no-cache' mode, vlm_path and vlm_type are required.")
            self.backbone = RecogDriveBackbone(
                model_type=self.vlm_type,
                checkpoint_path=self.vlm_path,
                device=str(device)
            )

            if not self.train_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad = False
            else:
                for p in self.backbone.parameters():
                    p.requires_grad = True

        if self.dit_type == "large":
            cfg = make_recogdrive_config(self.dit_type, action_dim=3, action_horizon=8, grpo=self.grpo, input_embedding_dim=1536,sampling_method=sampling_method)
        elif self.dit_type == "small":
            cfg = make_recogdrive_config(self.dit_type, action_dim=3, action_horizon=8, grpo=self.grpo, input_embedding_dim=384,sampling_method=sampling_method)

        cfg.vlm_size = self.vlm_size
        cfg.use_expert_features = self.use_expert_features
        cfg.expert_feature_source = self.expert_feature_source
        cfg.allow_random_init = self.allow_random_init
        cfg.allow_dummy_expert_cache = self.allow_dummy_expert_cache
        cfg.use_jepa = self.use_jepa
        cfg.use_vggt = self.use_vggt
        cfg.jepa_dim = self.jepa_dim
        cfg.vggt_dim = self.vggt_dim
        cfg.expert_dropout = self.expert_dropout
        cfg.expert_fusion_mode = self.expert_fusion_mode
        cfg.use_expert_type_embedding = self.use_expert_type_embedding
        cfg.use_expert_gates = self.use_expert_gates
        cfg.expert_alignment_weight = self.expert_alignment_weight
        cfg.jepa_alignment_weight = self.jepa_alignment_weight
        cfg.vggt_alignment_weight = self.vggt_alignment_weight
        cfg.alignment_loss_type = self.alignment_loss_type

        if self.grpo:
            cfg.grpo_cfg.metric_cache_path = self.metric_cache_path
            cfg.grpo_cfg.reference_policy_checkpoint = self.reference_policy_checkpoint
            
        self.action_head = ReCogDriveDiffusionPlanner(cfg).to(device)
        self.num_inference_samples = 1
        self.inference_selection_mode = "median"

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        if self.checkpoint_path:
            self._safe_load_checkpoint(self.checkpoint_path)
            return

        if self.allow_random_init and not self._warned_random_init:
            warnings.warn(
                "ReCogDriveAgent is using random initialization because checkpoint_path is empty. "
                "Random initialization is for computation-flow validation only and must not be "
                "interpreted as driving performance.",
                RuntimeWarning,
            )
            self._warned_random_init = True

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_all_sensors(include=[0, 1, 2, 3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [TrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [ReCogDriveFeatureBuilder(
            cache_hidden_state=self.cache_hidden_state,
            model_type=self.vlm_type,
            checkpoint_path=self.vlm_path,
            device=str(self.device),
            cache_mode=self.cache_mode,
            use_expert_features=self.use_expert_features,
            expert_feature_source=self.expert_feature_source,
            expert_cache_dir=self.expert_cache_dir,
            num_jepa_tokens=self.num_jepa_tokens,
            num_vggt_tokens=self.num_vggt_tokens,
            allow_expert_target_features=self.allow_expert_target_features and self.training,
            use_jepa=self.use_jepa,
            use_vggt=self.use_vggt,
            jepa_dim=self.jepa_dim,
            vggt_dim=self.vggt_dim,
        )]

    def forward(self, features: Dict[str, torch.Tensor], targets=None, tokens_list=None) -> Dict[str, torch.Tensor]:
        action_device = next(self.action_head.parameters()).device
        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                features[key] = tensor.to(action_device)

        model_dtype = next(self.action_head.parameters()).dtype
        for key in EXPERT_FEATURE_KEYS:
            if key in features and isinstance(features[key], torch.Tensor):
                features[key] = features[key].to(model_dtype)
        self._add_dummy_expert_features_if_needed(features, action_device, model_dtype)

        history_trajectory = features["history_trajectory"].to(action_device)
        high_command_one_hot = features["high_command_one_hot"].to(action_device)
        
        if history_trajectory.ndim == 2:
            history_trajectory = history_trajectory.unsqueeze(0)
        if high_command_one_hot.ndim == 1:
            high_command_one_hot = high_command_one_hot.unsqueeze(0)

        if self.cache_hidden_state:
            last_hidden_state = features["last_hidden_state"].to(action_device)
        else:
            if self.backbone is None:
                raise RuntimeError("Agent is in 'no-cache' mode, but backbone is not initialized.")
            image_path_tensor = features["image_path_tensor"]
            if image_path_tensor.ndim == 1:
                image_path_tensor = image_path_tensor.unsqueeze(0)
            image_paths = self._decode_paths_from_tensor(image_path_tensor)
            
            pixel_values_list = [load_image(path) for path in image_paths]
            
            num_patches_list = [p.shape[0] for p in pixel_values_list]
            pixel_values_cat = torch.cat(pixel_values_list, dim=0).to(action_device)
            

            navigation_commands = ['turn left', 'go straight', 'turn right']
            command_indices = torch.argmax(high_command_one_hot, dim=-1)
            command_str_list = [navigation_commands[idx.item()] for idx in command_indices]

            questions = []
            batch_size = high_command_one_hot.shape[0]
            for i in range(batch_size):
                history_trajectory_sample = history_trajectory[i]
                command_str_sample = command_str_list[i]

                history_str = ' '.join([
                    f'   - t-{3-j}: ({format_number(history_trajectory_sample[j, 0].item())}, '
                    f'{format_number(history_trajectory_sample[j, 1].item())}, '
                    f'{format_number(history_trajectory_sample[j, 2].item())})'
                    for j in range(history_trajectory_sample.shape[0])
                ])
                
                prompt = (
                    "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
                    "1. Visual perception from front camera view\n"
                    f"2. Historical motion context (last 4 timesteps):{history_str}\n"
                    f"3. Active navigation command: [{command_str_sample.upper()}]"
                )
                output_requirements = (
                    "\nOutput requirements:\n- Predict 8 future trajectory points\n"
                    "- Each point format: (x:float, y:float, heading:float)\n"
                    "- Use [PT, ...] to encapsulate the trajectory\n"
                    "- Maintain numerical precision to 2 decimal places"
                )
                questions.append(f"{prompt}{output_requirements}")

            outputs = self.backbone(pixel_values_cat, questions, num_patches_list=num_patches_list)
            last_hidden_state = outputs.hidden_states[-1]

        status_feature = features["status_feature"].to(action_device)
        if status_feature.ndim == 1:
            status_feature = status_feature.unsqueeze(0)
        if last_hidden_state.ndim == 2:
            last_hidden_state = last_hidden_state.unsqueeze(0)

        last_hidden_state = last_hidden_state.to(model_dtype)
        history_trajectory_reshaped = history_trajectory.view(history_trajectory.size(0), -1)
        input_state = torch.cat([status_feature, history_trajectory_reshaped], dim=1)
        action_input_data = {
            "state": input_state.to(model_dtype),
            "his_traj": history_trajectory_reshaped.to(model_dtype),
            "status_feature": status_feature.to(model_dtype),
        }
        for key in EXPERT_FEATURE_KEYS:
            if key in features and isinstance(features[key], torch.Tensor):
                action_input_data[key] = features[key].to(model_dtype)

        if self.training and not self.grpo:
            action_inputs = BatchFeature(
                data={**action_input_data, "action": targets["trajectory"].to(device=action_device, dtype=model_dtype)}
            )
            return self.action_head(last_hidden_state, action_inputs)
        elif self.training and self.grpo:
            action_inputs = BatchFeature(
                data={**action_input_data, "action": targets["trajectory"].to(device=action_device, dtype=model_dtype)}
            )
            return self.action_head.forward_grpo(last_hidden_state, action_inputs, tokens_list)
        else: 
            action_inputs = BatchFeature(action_input_data)
            return self.action_head.get_action(last_hidden_state.to(model_dtype), action_inputs)

    @staticmethod
    def _is_expert_parameter_key(key: str) -> bool:
        expert_markers = (
            "jepa_projector",
            "vggt_projector",
            "expert_type_embedding",
            "jepa_gate",
            "vggt_gate",
        )
        return key.startswith("action_head.") and any(marker in key for marker in expert_markers)

    @staticmethod
    def _checkpoint_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
        if isinstance(checkpoint, dict):
            return checkpoint
        raise TypeError(f"Checkpoint must be a dict or contain a 'state_dict' dict, got {type(checkpoint).__name__}.")

    def _safe_load_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            if self.allow_random_init:
                warnings.warn(
                    f"Checkpoint not found at {path}. Continuing with random initialization because "
                    "allow_random_init=True. This must not be interpreted as driving performance.",
                    RuntimeWarning,
                )
                return
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        state_dict = self._checkpoint_state_dict(checkpoint)
        model_dict = self.state_dict()

        filtered_state: Dict[str, torch.Tensor] = {}
        unexpected_keys: List[str] = []
        skipped_expert_shape: List[str] = []
        shape_mismatches: List[str] = []

        for key, value in state_dict.items():
            mapped_key = key[len("agent."):] if key.startswith("agent.") else key
            if mapped_key not in model_dict:
                unexpected_keys.append(mapped_key)
                continue
            if not isinstance(value, torch.Tensor):
                unexpected_keys.append(mapped_key)
                continue
            expected = model_dict[mapped_key]
            if expected.shape != value.shape:
                message = f"{mapped_key}: checkpoint {tuple(value.shape)} vs model {tuple(expected.shape)}"
                if self._is_expert_parameter_key(mapped_key):
                    skipped_expert_shape.append(message)
                    continue
                shape_mismatches.append(message)
                continue
            filtered_state[mapped_key] = value

        if shape_mismatches:
            raise RuntimeError(
                "Checkpoint shape mismatch for existing baseline parameters:\n"
                + "\n".join(f"  - {item}" for item in shape_mismatches)
            )

        incompatible = self.load_state_dict(filtered_state, strict=False)
        missing_keys = list(incompatible.missing_keys)
        missing_expert = [key for key in missing_keys if self._is_expert_parameter_key(key)]
        missing_other = [key for key in missing_keys if not self._is_expert_parameter_key(key)]

        print(f"Loaded checkpoint from {path} with strict=False.")
        print(f"  loaded keys: {len(filtered_state)}")
        print(f"  missing keys: {len(missing_keys)}")
        if missing_expert:
            print("  expected missing expert keys:")
            for key in missing_expert:
                print(f"    - {key}")
        if missing_other:
            print("  missing non-expert keys:")
            for key in missing_other:
                print(f"    - {key}")
        if unexpected_keys or incompatible.unexpected_keys:
            print("  unexpected keys:")
            for key in [*unexpected_keys, *incompatible.unexpected_keys]:
                print(f"    - {key}")
        if skipped_expert_shape:
            print("  skipped expert shape mismatches:")
            for item in skipped_expert_shape:
                print(f"    - {item}")

    def _add_dummy_expert_features_if_needed(
        self,
        features: Dict[str, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if (
            not self.use_expert_features
            or self.expert_feature_source != "dummy"
            or self._dummy_expert_backend is None
        ):
            return

        required_keys = []
        if self.use_jepa:
            required_keys.append("jepa_tokens")
        if self.use_vggt:
            required_keys.append("vggt_tokens")
        if all(key in features for key in required_keys):
            return

        batch_size = features["history_trajectory"].shape[0] if features["history_trajectory"].ndim >= 3 else 1
        include_targets = self.training and self.allow_expert_target_features
        dummy_features = self._dummy_expert_backend.generate(
            batch_size,
            device=device,
            dtype=dtype,
            include_targets=include_targets,
        )
        for key, value in dummy_features.items():
            features.setdefault(key, value)

        if not self._warned_dummy_features:
            warnings.warn(
                f"{DUMMY_EXPERT_WARNING} No benchmark metrics should be reported in dummy mode.",
                RuntimeWarning,
            )
            self._warned_dummy_features = True

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        self.eval()

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["pred_traj"].float().cpu().squeeze(0)

        return Trajectory(poses)

    def compute_trajectory_vis(self, agent_input: AgentInput) -> Trajectory:
        self.eval()

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["pred_traj"].float().cpu().squeeze(0)
        return Trajectory(poses)


    def compute_loss(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.training and self.grpo:
            return predictions
        elif self.training:
            return predictions.loss
        else:
            return torch.nn.functional.l1_loss(predictions["pred_traj"], targets["trajectory"])

    def get_optimizers(self) -> Union[Optimizer, Dict[str, LRScheduler]]:
        optimizer_cfg = DictConfig(dict(type="AdamW", lr=self._lr, weight_decay=1e-4, betas=(0.9, 0.95)))

        params = list(self.action_head.parameters())
        if self.backbone is not None and self.train_backbone:
            params += list(self.backbone.parameters())

        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        
        if self.grpo:
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=0.0, epochs=10, warmup_epochs=0)
        else:
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=1e-6, epochs=200, warmup_epochs=3)
            
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    @staticmethod
    def _decode_paths_from_tensor(path_tensor: torch.Tensor) -> List[str]:
        """
        Decodes a batch of path tensors back into a list of file path strings.
        
        Args:
            path_tensor (torch.Tensor): A 2D tensor of shape 
                (batch_size, max_path_length) from the collate_fn.
        
        Returns:
            List[str]: A list of decoded file path strings.
        """
        decoded_paths = []
        for single_path_tensor in path_tensor:
            chars = []
            for code in single_path_tensor:
                code_item = code.item()
                if code_item == 0: 
                    break
                chars.append(chr(code_item))
            decoded_paths.append("".join(chars))
        return decoded_paths

def make_recogdrive_config(
    size: str,
    *,
    action_dim: int,
    action_horizon: int,
    input_embedding_dim: int,
    sampling_method: str = 'ddim',
    num_inference_steps: int = 5,
    grpo: bool = False,
    model_dtype: str = "float16",
) -> ReCogDriveDiffusionPlannerConfig:
    """
    A factory function to create a ReCogDriveDiffusionPlannerConfig object.

    This function simplifies configuration by using a size preset ("small",
    "large", "large_new") to define the core DiT architecture, while allowing
    other important planner settings to be specified.

    Args:
        size (str): The size preset for the DiT backbone.
        action_dim (int): The dimension of the action space.
        action_horizon (int): The number of future action steps to predict.
        input_embedding_dim (int): Dimension of the input embeddings to the DiT.
        sampling_method (str): The core training and sampling methodology.
        num_inference_steps (int): Number of steps for inference sampling.
        grpo (bool): If True, enables GRPO-specific logic.
        model_dtype (str): The data type for model computations.

    Returns:
        ReCogDriveDiffusionPlannerConfig: An instantiated and configured planner config object.
    """
    size = size.lower()
    if size == "small":
        diffusion_model_cfg = {"num_heads": 8, "head_dim": 48, "num_layers": 16,"output_dim":512}
    elif size == "large":
        diffusion_model_cfg = {"num_heads": 32, "head_dim": 48, "num_layers": 16,"output_dim":1536}
    else:
        raise ValueError(f"Unknown model size: {size!r}")

    common_params: Dict[str, any] = {
        "dropout": 0.0,
        "attention_bias": True,
        "norm_eps": 1e-5,
        "interleave_attention": True,
    }
    diffusion_model_cfg.update(common_params)

    config = ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg=diffusion_model_cfg,
        action_dim=action_dim,
        action_horizon=action_horizon,
        input_embedding_dim=input_embedding_dim,
        sampling_method=sampling_method,
        num_inference_steps=num_inference_steps,
        grpo=grpo,
        model_dtype=model_dtype,
    )
    
    return config
