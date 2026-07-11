from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from .bench2drive_contract import BENCH2DRIVE_SYSTEM_MESSAGE
from .utils.conversation import get_conv_template

IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'

NAVSIM_SYSTEM_MESSAGE = """
You are a vehicle trajectory prediction model for autonomous driving. Your task is to predict the ego vehicle's 4-second trajectory based on the following inputs: multi-view images from 8 cameras, ego vehicle states (position), and discrete navigation commands. The input provides a 2-second history, and your output should ensure a safe trajectory for the next 4 seconds. Your predictions must adhere to the following metrics:
1. **No at-fault Collisions (NC)**: Avoid collisions with other objects/vehicles.
2. **Drivable Area Compliance (DAC)**: Stay within the drivable area.
3. **Time to Collision (TTC)**: Maintain a safe distance from other vehicles.
4. **Ego Progress (EP)**: Ensure the ego vehicle moves forward without being stuck.
5. **Comfort (C)**: Avoid sharp turns and sudden decelerations.
6. **Driving Direction Compliance (DDC)**: Align with the intended driving direction.
For evaluation, use the **PDM Score**, which combines these metrics: **PDM Score** = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12.
Your predictions will be evaluated through a non-reactive 4-second simulation with an LQR controller and background actors following their recorded trajectories. The better your predictions, the higher your score.
""".strip()

RECOGDRIVE_SYSTEM_MESSAGES = {
    "navsim": NAVSIM_SYSTEM_MESSAGE,
    "bench2drive": BENCH2DRIVE_SYSTEM_MESSAGE,
}

# Backward-compatible alias for callers that imported the original module global.
system_message = NAVSIM_SYSTEM_MESSAGE


def resolve_recogdrive_system_message(profile: str = "navsim") -> str:
    """Resolve a named prompt profile without silently falling back."""
    normalized = str(profile).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "navsim": "navsim",
        "b2d": "bench2drive",
        "bench2drive": "bench2drive",
    }
    try:
        return RECOGDRIVE_SYSTEM_MESSAGES[aliases[normalized]]
    except KeyError as exc:
        supported = ", ".join(sorted(RECOGDRIVE_SYSTEM_MESSAGES))
        raise ValueError(f"Unsupported ReCogDrive system prompt profile {profile!r}; choose one of: {supported}") from exc


def _navigation_command_name(high_command_one_hot: Sequence[float] | torch.Tensor) -> str:
    command = torch.as_tensor(high_command_one_hot, dtype=torch.float32).reshape(-1)
    if command.numel() != 3:
        raise ValueError(f"Expected a 3-class navigation command, got shape {tuple(command.shape)}")
    if not torch.isfinite(command).all() or float(command.max()) <= 0.0:
        return "unknown"
    return ("turn left", "go straight", "turn right")[int(command.argmax().item())]


def _format_prompt_number(value: float, decimal_places: int = 2) -> str:
    rounded = round(float(value), decimal_places)
    if abs(rounded) <= 10 ** (-decimal_places):
        return "0.0"
    return f"{rounded:+.{decimal_places}f}"


def build_recogdrive_planning_question(
    history_trajectory: Sequence[Sequence[float]] | torch.Tensor,
    high_command_one_hot: Sequence[float] | torch.Tensor,
) -> str:
    """Build the single canonical question used by B2D SFT, caching, and serving."""
    history = torch.as_tensor(history_trajectory, dtype=torch.float32)
    if tuple(history.shape) != (4, 3):
        raise ValueError(f"Expected history trajectory shape (4, 3), got {tuple(history.shape)}")
    if not torch.isfinite(history).all():
        raise ValueError("History trajectory contains non-finite values")
    history_lines = "\n".join(
        f"   - t-{3 - index}: "
        f"({_format_prompt_number(row[0])}, {_format_prompt_number(row[1])}, {_format_prompt_number(row[2])})"
        for index, row in enumerate(history.tolist())
    )
    command = _navigation_command_name(high_command_one_hot)
    return (
        "<image>\n"
        "As an autonomous driving system, predict the vehicle's trajectory based on:\n"
        "1. Visual perception from front camera view\n"
        f"2. Historical motion context (last 4 timesteps):\n{history_lines}\n"
        f"3. Active navigation command: [{command.upper()}]\n"
        "Output requirements:\n"
        "- Predict 8 future trajectory points\n"
        "- Each point format: (x:float, y:float, heading:float)\n"
        "- Use [PT, ...] to encapsulate the trajectory\n"
        "- Maintain numerical precision to 2 decimal places"
    )


def format_recogdrive_trajectory_answer(trajectory: Sequence[Sequence[float]] | torch.Tensor) -> str:
    """Format an 8-pose target exactly as requested by the canonical question."""
    poses = torch.as_tensor(trajectory, dtype=torch.float32)
    if tuple(poses.shape) != (8, 3):
        raise ValueError(f"Expected future trajectory shape (8, 3), got {tuple(poses.shape)}")
    if not torch.isfinite(poses).all():
        raise ValueError("Future trajectory contains non-finite values")

    def value_text(value: float) -> str:
        rounded = round(float(value), 2)
        return "0.00" if abs(rounded) < 0.005 else f"{rounded:.2f}"

    points = ", ".join(
        f"({value_text(x)},{value_text(y)},{value_text(heading)})"
        for x, y, heading in poses.tolist()
    )
    return f"Here is the planning trajectory [PT, {points}]."


class RecogDriveBackbone(nn.Module):
    """
    A simplified vision-language model backbone with direct loading logic
    for different model architectures (InternVL, Qwen-VL).
    """
    def __init__(self,
                 model_type: str,
                 checkpoint_path: str,
                 device: str = "cuda",
                 system_prompt_profile: str = "navsim"):
        """
        Initializes and loads the specified model and its preprocessor/tokenizer.

        Args:
            model_type (str): The type of model to load. Supported: 'internvl', 'qwen'.
            checkpoint_path (str): The path to the model checkpoint.
            device (str): The device to load the model onto ('cuda', 'cpu').
        """
        super().__init__()

        self.model = None
        self.tokenizer = None  
        self.model_type = model_type.lower()
        self.device = device
        self.system_prompt_profile = system_prompt_profile
        self.system_message = resolve_recogdrive_system_message(system_prompt_profile)

        print(f"Initializing backbone of type: '{self.model_type}' from path: '{checkpoint_path}'")

        if self.model_type == 'internvl':
            # --- Load InternVL Model and Tokenizer ---
            self.model = AutoModel.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_flash_attn=True,
                device_map=self.device
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
                use_fast=False
            )
            # Load model-specific configuration
            self._configure_internvl()
            self.num_image_token = 256

        elif self.model_type == 'qwen':
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True
            )
            self.tokenizer = AutoProcessor.from_pretrained(
                checkpoint_path,
                trust_remote_code=True
            )
            
        else:
            raise ValueError(f"Unsupported model_type: '{self.model_type}'. Please choose 'internvl' or 'qwen'.")


        print(f"Backbone '{self.model_type}' loaded successfully on device '{self.device}'.")

    def _configure_internvl(self):
        """Applies specific configurations required for the InternVL model."""
        self.model.system_message = self.system_message
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self.img_context_token_id
        print("InternVL model configured.")
    
    @staticmethod
    def _patch_groups(
        questions: Sequence[str],
        num_patches_list: Sequence[int] | Sequence[Sequence[int]],
    ) -> List[List[int]]:
        """Normalize legacy one-image counts and formal multi-image counts."""

        if len(num_patches_list) != len(questions):
            raise ValueError(
                "num_patches_list must contain one entry per question; "
                f"got {len(num_patches_list)} entries for {len(questions)} questions"
            )
        groups: List[List[int]] = []
        for question, raw_counts in zip(questions, num_patches_list):
            if isinstance(raw_counts, int):
                counts = [int(raw_counts)]
            else:
                counts = [int(value) for value in raw_counts]
            if not counts or any(value <= 0 for value in counts):
                raise ValueError(f"Patch counts must be positive, got {counts}")
            image_slots = question.count("<image>")
            if image_slots == 0:
                if len(counts) != 1:
                    raise ValueError("A prompt without <image> can only receive one image")
            elif image_slots != len(counts):
                raise ValueError(
                    f"Prompt has {image_slots} <image> slots but received {len(counts)} patch groups"
                )
            groups.append(counts)
        return groups

    def forward(
        self,
        pixel_values: torch.Tensor,
        questions: List[str],
        num_patches_list: Sequence[int] | Sequence[Sequence[int]],
    ):
        if not self.model:
            raise RuntimeError("Backbone model has not been initialized. Call initialize() on the agent first.")
        
        model_dtype = next(self.model.parameters()).dtype

        patch_groups = self._patch_groups(questions, num_patches_list)
        expected_patches = sum(sum(group) for group in patch_groups)
        if int(pixel_values.size(0)) != expected_patches:
            raise ValueError(
                f"pixel_values contains {pixel_values.size(0)} patches, expected {expected_patches}"
            )

        queries = []
        for idx, patch_counts in enumerate(patch_groups):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            
            template = get_conv_template("internvl2_5")
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            for num_patches in patch_counts:
                image_tokens = (
                    IMG_START_TOKEN
                    + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                    + IMG_END_TOKEN
                )
                query = query.replace('<image>', image_tokens, 1)
            if '<image>' in query:
                raise ValueError("Not all image placeholders were expanded")
            queries.append(query)
        self.tokenizer.padding_side = 'left'
        model_inputs = self.tokenizer(
            queries,
            return_tensors='pt',
            padding=True,
            truncation=False,
        )
        if int(model_inputs['input_ids'].shape[-1]) > 12288:
            raise ValueError(
                "ReCogDrive prompt exceeds the Stage1 max sequence length of 12288 tokens: "
                f"{model_inputs['input_ids'].shape[-1]}"
            )
        device = torch.device(self.device)
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        
        num_patches = pixel_values.size(0)
        image_flags = torch.ones(num_patches, dtype=torch.long, device=device)

        # Cache builders use one prompt at a time and therefore have no padding.
        # Retaining the mask also lets callers remove padding when batching is
        # introduced later without changing the public model output type.
        self._last_attention_mask = attention_mask.detach()


        return self.model(
                pixel_values=pixel_values.to(model_dtype),
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_flags=image_flags.squeeze(-1),
                output_hidden_states=True,
                return_dict=True,
        )

    def active_hidden_state(self, outputs, batch_index: int = 0) -> torch.Tensor:
        """Return final-layer tokens selected by the prompt attention mask."""

        if not hasattr(self, "_last_attention_mask"):
            raise RuntimeError("No attention mask is available; call the backbone first")
        hidden = outputs.hidden_states[-1][batch_index]
        mask = self._last_attention_mask[batch_index].to(device=hidden.device, dtype=torch.bool)
        if hidden.shape[0] != mask.shape[0]:
            raise RuntimeError(
                f"Hidden/token mask mismatch: hidden={hidden.shape[0]}, mask={mask.shape[0]}"
            )
        return hidden[mask]

    
