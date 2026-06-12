import inspect
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from .two_expert_slots import TwoExpertSoftSlots, extract_two_expert_slot_hidden
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
            raise NotImplementedError("two_expert_slot soft injection requires language_model.get_input_embeddings().")
        embeddings = language_model.get_input_embeddings()
        if embeddings is None:
            raise NotImplementedError("language_model.get_input_embeddings() returned None.")
        return embeddings

    @staticmethod
    def _language_model(model: nn.Module) -> nn.Module:
        language_model = getattr(model, "language_model", None)
        if language_model is None and hasattr(model, "get_base_model"):
            try:
                language_model = getattr(model.get_base_model(), "language_model", None)
            except Exception:
                language_model = None
        if language_model is None:
            raise NotImplementedError("two_expert_slot soft injection requires an InternVL language_model.")
        return language_model

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

    def _apply_two_expert_train_mode(self, train_vlm_mode: str, top_layers: int = 2) -> None:
        if train_vlm_mode not in {"frozen", "lora", "top_layers"}:
            raise ValueError("train_vlm_mode must be 'frozen', 'lora', or 'top_layers'.")
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        if train_vlm_mode == "lora":
            for name, parameter in self.model.named_parameters():
                if "lora_" in name:
                    parameter.requires_grad = True
        elif train_vlm_mode == "top_layers":
            candidates = []
            for name, module in self.model.named_modules():
                lowered = name.lower()
                if "vision" in lowered or "visual" in lowered:
                    continue
                if any(marker in lowered for marker in ("layers.", "layer.", "blocks.")):
                    candidates.append(module)
            for module in candidates[-int(top_layers):]:
                for parameter in module.parameters(recurse=False):
                    parameter.requires_grad = True

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

    def forward_with_two_expert_slots(
        self,
        images: torch.Tensor,
        prompt_inputs: Union[List[str], Dict[str, Any]],
        two_expert_slots: TwoExpertSoftSlots,
        return_image_hidden: bool = True,
        return_raw_hidden: bool = True,
        train_vlm_mode: str = "frozen",
    ) -> Dict[str, Any]:
        if not self.model:
            raise RuntimeError("Backbone model has not been initialized. Call initialize() on the agent first.")
        if self.model_type != "internvl":
            raise NotImplementedError("two_expert_slot VLM soft-slot injection is implemented for InternVL only.")
        use_internvl_low_level_forward = hasattr(self.model, "extract_feature") and hasattr(self.model, "language_model")
        if not use_internvl_low_level_forward and not self._model_accepts_inputs_embeds(self.model):
            raise NotImplementedError(
                "two_expert_slot requires either InternVL extract_feature()+language_model or a wrapper "
                "that accepts inputs_embeds and forwards them through the language/VLM transformer."
            )
        if isinstance(prompt_inputs, dict):
            questions = prompt_inputs.get("questions") or prompt_inputs.get("prompts")
            num_patches_list = prompt_inputs.get("num_patches_list")
        else:
            questions = prompt_inputs
            num_patches_list = None
        if not isinstance(questions, list) or not all(isinstance(item, str) for item in questions):
            raise TypeError("prompt_inputs must be a list[str] or a dict containing questions/prompts.")
        if num_patches_list is None:
            if images.shape[0] % len(questions) != 0:
                raise ValueError("Cannot infer num_patches_list from images and questions.")
            num_patches_list = [images.shape[0] // len(questions)] * len(questions)
        self._apply_two_expert_train_mode(train_vlm_mode)

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
        batch_size, raw_len, hidden_dim = token_embeddings.shape
        if use_internvl_low_level_forward and not hasattr(self.model, "img_context_token_id"):
            self.model.img_context_token_id = self.img_context_token_id
        if use_internvl_low_level_forward:
            vit_embeds = self.model.extract_feature(images.to(device=device, dtype=model_dtype))
            if vit_embeds.shape[-1] != hidden_dim:
                raise ValueError(
                    f"InternVL visual feature dim {vit_embeds.shape[-1]} does not match "
                    f"language hidden dim {hidden_dim}."
                )
            flat_token_embeddings = token_embeddings.reshape(batch_size * raw_len, hidden_dim).clone()
            flat_input_ids = input_ids.reshape(batch_size * raw_len)
            image_selected = flat_input_ids == self.img_context_token_id
            flat_vit_embeds = vit_embeds.reshape(-1, hidden_dim).to(dtype=flat_token_embeddings.dtype)
            selected_count = int(image_selected.sum().item())
            if selected_count <= 0:
                raise ValueError("InternVL prompt contains no IMG_CONTEXT tokens for visual feature injection.")
            if flat_vit_embeds.shape[0] < selected_count:
                raise ValueError(
                    f"Not enough visual tokens for IMG_CONTEXT slots: visual={flat_vit_embeds.shape[0]}, "
                    f"selected={selected_count}."
                )
            flat_token_embeddings[image_selected] = flat_vit_embeds[:selected_count]
            token_embeddings = flat_token_embeddings.reshape(batch_size, raw_len, hidden_dim)

        dyn_slot_embeds, geo_slot_embeds, slot_metadata = two_expert_slots.get_slots(
            batch_size,
            device=token_embeddings.device,
            dtype=token_embeddings.dtype,
        )
        dyn_start = raw_len
        dyn_end = dyn_start + dyn_slot_embeds.shape[1]
        geo_start = dyn_end
        geo_end = geo_start + geo_slot_embeds.shape[1]
        full_inputs_embeds = torch.cat([token_embeddings, dyn_slot_embeds, geo_slot_embeds], dim=1)
        slot_attention = torch.ones(
            batch_size,
            dyn_slot_embeds.shape[1] + geo_slot_embeds.shape[1],
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        full_attention_mask = torch.cat([attention_mask, slot_attention], dim=1)
        extra_offsets = torch.arange(
            full_inputs_embeds.shape[1] - raw_len,
            device=position_ids.device,
            dtype=position_ids.dtype,
        ).unsqueeze(0)
        slot_positions = position_ids.max(dim=1, keepdim=True).values + 1 + extra_offsets
        full_position_ids = torch.cat([position_ids, slot_positions], dim=1)

        absolute_dyn_group_spans = []
        for start, end in slot_metadata["dyn_group_spans"]:
            absolute_dyn_group_spans.append((dyn_start + int(start), dyn_start + int(end)))
        slot_metadata = dict(slot_metadata)
        slot_metadata.update(
            {
                "raw_token_count": raw_len,
                "absolute_dyn_group_spans": absolute_dyn_group_spans,
                "absolute_geo_span": (geo_start, geo_end),
                "slot_insertion": "after_prompt_tokens_before_answer_continuation",
            }
        )

        if use_internvl_low_level_forward:
            language_model = self._language_model(self.model)
            outputs = language_model(
                inputs_embeds=full_inputs_embeds.to(device=device, dtype=token_embeddings.dtype),
                attention_mask=full_attention_mask,
                position_ids=full_position_ids,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            outputs = self.model(
                pixel_values=images.to(device=device, dtype=model_dtype),
                inputs_embeds=full_inputs_embeds.to(device=device, dtype=token_embeddings.dtype),
                attention_mask=full_attention_mask,
                position_ids=full_position_ids,
                output_hidden_states=True,
                return_dict=True,
            )
        full_hidden_state = outputs.hidden_states[-1]
        raw_vlm_hidden = full_hidden_state[:, :raw_len]
        h_dyn, h_geo = extract_two_expert_slot_hidden(full_hidden_state, slot_metadata)
        image_mask = input_ids == self.img_context_token_id
        image_hidden = self._gather_masked_tokens(raw_vlm_hidden, image_mask)
        zero = full_hidden_state.new_zeros(())
        result: Dict[str, Any] = {
            "raw_vlm_hidden": raw_vlm_hidden if return_raw_hidden else None,
            "image_hidden": image_hidden if return_image_hidden else None,
            "h_dyn": h_dyn,
            "h_geo": h_geo,
            "full_hidden_state": full_hidden_state,
            "slot_metadata": slot_metadata,
            "diagnostics": {
                "raw_vlm_hidden_norm": raw_vlm_hidden.detach().float().norm(dim=-1).mean().to(dtype=full_hidden_state.dtype),
                "h_dyn_norm": h_dyn.detach().float().norm(dim=-1).mean().to(dtype=full_hidden_state.dtype),
                "h_geo_norm": h_geo.detach().float().norm(dim=-1).mean().to(dtype=full_hidden_state.dtype),
                "image_hidden_norm": image_hidden.detach().float().norm(dim=-1).mean().to(dtype=full_hidden_state.dtype) if image_hidden.numel() else zero,
                "slot_count_dyn": full_hidden_state.new_tensor(float(h_dyn.shape[1] * h_dyn.shape[2])),
                "slot_count_geo": full_hidden_state.new_tensor(float(h_geo.shape[1])),
            },
        }
        return result

    
