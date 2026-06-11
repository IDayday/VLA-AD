import inspect
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from .last_vla_latent_slots import LastVLALatentSlotConfig, LastVLASoftLatentSlots, StructuredCausalMaskBuilder
from .utils.conversation import get_conv_template

IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'

system_message = """
You are a vehicle trajectory prediction model for autonomous driving. Your task is to predict the ego vehicle's 4-second trajectory based on the following inputs: multi-view images from 8 cameras, ego vehicle states (position), and discrete navigation commands. The input provides a 2-second history, and your output should ensure a safe trajectory for the next 4 seconds. Your predictions must adhere to the following metrics:
1. **No at-fault Collisions (NC)**: Avoid collisions with other objects/vehicles.
2. **Drivable Area Compliance (DAC)**: Stay within the drivable area.
3. **Time to Collision (TTC)**: Maintain a safe distance from other vehicles.
4. **Ego Progress (EP)**: Ensure the ego vehicle moves forward without being stuck.
5. **Comfort (C)**: Avoid sharp turns and sudden decelerations.
6. **Driving Direction Compliance (DDC)**: Align with the intended driving direction.
For evaluation, use the **PDM Score**, which combines these metrics: **PDM Score** = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12.
Your predictions will be evaluated through a non-reactive 4-second simulation with an LQR controller and background actors following their recorded trajectories. The better your predictions, the higher your score.
"""

class RecogDriveBackbone(nn.Module):
    """
    A simplified vision-language model backbone with direct loading logic
    for different model architectures (InternVL, Qwen-VL).
    """
    def __init__(self,
                 model_type: str,
                 checkpoint_path: str,
                 device: str = "cuda"):
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
        self.last_vla_soft_slots: Optional[LastVLASoftLatentSlots] = None

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

    def configure_last_vla_latent_slots(self, config: LastVLALatentSlotConfig) -> LastVLASoftLatentSlots:
        self.last_vla_soft_slots = LastVLASoftLatentSlots(config)
        return self.last_vla_soft_slots

    def _configure_internvl(self):
        """Applies specific configurations required for the InternVL model."""
        self.model.system_message = system_message
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self.img_context_token_id
        self._patch_internvl_visual_feature_dtype(self.model)
        print("InternVL model configured.")

    @staticmethod
    def _infer_model_compute_dtype(model: nn.Module) -> torch.dtype:
        """Prefer the base model dtype over FP32 LoRA adapter parameters."""
        modules_to_scan = []
        if hasattr(model, "get_base_model"):
            try:
                modules_to_scan.append(model.get_base_model())
            except Exception:
                pass
        modules_to_scan.append(model)

        fallback_dtype = None
        for module in modules_to_scan:
            for name, parameter in module.named_parameters():
                if fallback_dtype is None:
                    fallback_dtype = parameter.dtype
                if "lora_" not in name:
                    return parameter.dtype
        if fallback_dtype is not None:
            return fallback_dtype
        raise ValueError("Cannot infer compute dtype from a model without parameters.")

    @staticmethod
    def _infer_language_embedding_dtype(model: nn.Module) -> Optional[torch.dtype]:
        language_model = getattr(model, "language_model", None)
        if language_model is None and hasattr(model, "get_base_model"):
            try:
                language_model = getattr(model.get_base_model(), "language_model", None)
            except Exception:
                language_model = None
        if language_model is None or not hasattr(language_model, "get_input_embeddings"):
            return None
        try:
            embedding = language_model.get_input_embeddings()
        except Exception:
            return None
        weight = getattr(embedding, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.is_floating_point():
            return weight.dtype
        return None

    @staticmethod
    def _patch_internvl_visual_feature_dtype(model: nn.Module) -> None:
        if getattr(model, "_recogdrive_cast_visual_feature_dtype", False):
            return
        if not hasattr(model, "extract_feature") or not hasattr(model, "language_model"):
            return
        original_extract_feature = model.extract_feature

        def extract_feature_with_language_dtype(pixel_values, *args, **kwargs):
            features = original_extract_feature(pixel_values, *args, **kwargs)
            target_dtype = RecogDriveBackbone._infer_language_embedding_dtype(model)
            if (
                target_dtype is not None
                and isinstance(features, torch.Tensor)
                and features.is_floating_point()
                and features.dtype != target_dtype
            ):
                features = features.to(dtype=target_dtype)
            return features

        model.extract_feature = extract_feature_with_language_dtype
        model._recogdrive_cast_visual_feature_dtype = True

    def _build_internvl_queries(self, pixel_values: Optional[torch.Tensor], questions: List[str], num_patches_list: List[int]) -> List[str]:
        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question

            template = get_conv_template("internvl2_5")
            template.system_message = system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)
        return queries

    @staticmethod
    def _language_embedding_layer(model: nn.Module) -> nn.Module:
        language_model = getattr(model, "language_model", None)
        if language_model is None and hasattr(model, "get_base_model"):
            try:
                language_model = getattr(model.get_base_model(), "language_model", None)
            except Exception:
                language_model = None
        if language_model is None or not hasattr(language_model, "get_input_embeddings"):
            raise NotImplementedError(
                "Strict LaST-VLA soft slots require access to language_model.get_input_embeddings()."
            )
        embeddings = language_model.get_input_embeddings()
        if embeddings is None:
            raise NotImplementedError("language_model.get_input_embeddings() returned None.")
        return embeddings

    @staticmethod
    def _model_accepts_inputs_embeds(model: nn.Module) -> bool:
        try:
            signature = inspect.signature(model.forward)
        except (TypeError, ValueError):
            return False
        if "inputs_embeds" in signature.parameters:
            return True
        return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())

    @staticmethod
    def _gather_masked_tokens(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim != 2:
            raise ValueError("mask must have shape [B, N].")
        lengths = mask.long().sum(dim=1)
        max_len = int(lengths.max().item()) if lengths.numel() else 0
        if max_len == 0:
            return hidden.new_zeros(hidden.shape[0], 0, hidden.shape[-1])
        out = hidden.new_zeros(hidden.shape[0], max_len, hidden.shape[-1])
        for index in range(hidden.shape[0]):
            selected = hidden[index, mask[index]]
            out[index, : selected.shape[0]] = selected
        return out

    def forward(self, pixel_values: torch.Tensor, questions: List[str], num_patches_list: List[int]):
        if not self.model:
            raise RuntimeError("Backbone model has not been initialized. Call initialize() on the agent first.")

        model_dtype = self._infer_model_compute_dtype(self.model)

        queries = self._build_internvl_queries(pixel_values, questions, num_patches_list)
        self.tokenizer.padding_side = 'left'
        model_inputs = self.tokenizer(queries, return_tensors='pt', padding='max_length', max_length=2800)
        device = torch.device(self.device)
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        
        num_patches = pixel_values.size(0)
        image_flags = torch.tensor([1] * num_patches, dtype=torch.long, device=device)


        return self.model(
                pixel_values=pixel_values.to(device=device, dtype=model_dtype),
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_flags=image_flags.squeeze(-1),
                output_hidden_states=True,
                return_dict=True,
        )

    def forward_last_vla_latent_slots(
        self,
        images: torch.Tensor,
        questions: List[str],
        status_feature: Optional[torch.Tensor] = None,
        high_command_one_hot: Optional[torch.Tensor] = None,
        history_trajectory: Optional[torch.Tensor] = None,
        slot_config: Optional[LastVLALatentSlotConfig] = None,
        structured_mask_mode: str = "stage1_alignment",
        return_image_hidden: bool = True,
        return_answer_logits: bool = False,
        num_patches_list: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.model:
            raise RuntimeError("Backbone model has not been initialized. Call initialize() on the agent first.")
        if self.model_type != "internvl":
            raise NotImplementedError("Strict LaST-VLA latent slots are currently implemented for InternVL backbones only.")
        if slot_config is None:
            slot_config = LastVLALatentSlotConfig(structured_mask_mode=structured_mask_mode)
        if self.last_vla_soft_slots is None or self.last_vla_soft_slots.config != slot_config:
            self.configure_last_vla_latent_slots(slot_config)
        assert self.last_vla_soft_slots is not None
        if num_patches_list is None:
            num_patches_list = [images.shape[0] // len(questions)] * len(questions)
        if not self._model_accepts_inputs_embeds(self.model):
            raise NotImplementedError(
                "Strict LaST-VLA requires the VLM wrapper to accept inputs_embeds. "
                "The current InternVL wrapper forward signature does not expose inputs_embeds; "
                "implement a low-level language_model forward before enabling strict latent slots."
            )

        model_dtype = self._infer_model_compute_dtype(self.model)
        queries = self._build_internvl_queries(images, questions, num_patches_list)
        self.tokenizer.padding_side = 'left'
        model_inputs = self.tokenizer(queries, return_tensors='pt', padding='max_length', max_length=2800)
        device = torch.device(self.device)
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        embeddings = self._language_embedding_layer(self.model)
        token_embeddings = embeddings(input_ids)
        slot_batch = self.last_vla_soft_slots.build_inputs_embeds(
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            build_structured_mask=True,
        )
        StructuredCausalMaskBuilder.require_4d_attention_mask_support(self.model)

        num_patches = images.size(0)
        image_flags = torch.tensor([1] * num_patches, dtype=torch.long, device=device)
        outputs = self.model(
            pixel_values=images.to(device=device, dtype=model_dtype),
            inputs_embeds=slot_batch.inputs_embeds.to(device=device, dtype=token_embeddings.dtype),
            attention_mask=slot_batch.structured_attention_mask,
            position_ids=slot_batch.position_ids,
            image_flags=image_flags.squeeze(-1),
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden_state = outputs.hidden_states[-1]
        slots = self.last_vla_soft_slots.extract_slot_hidden(last_hidden_state, slot_batch.slot_spans)
        base_hidden = last_hidden_state[:, : input_ids.shape[1]]
        image_mask = input_ids == self.img_context_token_id
        text_mask = (attention_mask > 0) & (~image_mask)
        result: Dict[str, torch.Tensor] = {
            "last_hidden_state": last_hidden_state,
            "h_dyn": slots["h_dyn"],
            "h_geo": slots["h_geo"],
            "h_plan": slots["h_plan"],
            "slot_metadata": {
                "dyn_count": slot_config.num_dyn_latent_tokens,
                "geo_count": slot_config.num_geo_latent_tokens,
                "plan_count": slot_config.num_plan_latent_tokens,
                "structured_mask_mode": structured_mask_mode,
            },
        }
        if return_image_hidden:
            result["image_hidden_states"] = self._gather_masked_tokens(base_hidden, image_mask)
        result["text_hidden_states"] = self._gather_masked_tokens(base_hidden, text_mask)
        if return_answer_logits:
            result["answer_logits"] = getattr(outputs, "logits", None)
        return result

    
