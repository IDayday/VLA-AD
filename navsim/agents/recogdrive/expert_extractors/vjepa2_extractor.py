from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from ..dynamic_tokenizer import DynamicTokenPacker
from .pooling import expand_four_frames_to_eight, pool_vjepa2_tokens


class VJEPA2Extractor:
    teacher_dim = 1024

    def __init__(
        self,
        model_path: str | Path = "facebook/vjepa2-vitl-fpc64-256",
        *,
        device: str | torch.device = "cuda",
        precision: str = "bf16",
        num_tokens: int = 12,
        strict_highcap_jepa: bool = False,
    ) -> None:
        from transformers import AutoModel, AutoVideoProcessor

        self.model_path = str(model_path)
        self.device = torch.device(device)
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        self.num_tokens = int(num_tokens)
        self.strict_highcap_jepa = bool(strict_highcap_jepa)
        self.last_tokenizer_metadata: dict = {}
        self.processor = AutoVideoProcessor.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
        image_size = int(getattr(self.model.config, "image_size", 256))
        patch_size = int(getattr(self.model.config, "patch_size", 16))
        self.spatial_hw = (image_size // patch_size, image_size // patch_size)
        self.highcap_packer = DynamicTokenPacker(
            output_tokens=self.num_tokens,
            teacher_dim=self.teacher_dim,
            strict=self.strict_highcap_jepa,
        ) if self.num_tokens != 12 else None
        if self.dtype != torch.float32:
            self.model = self.model.to(dtype=self.dtype)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @staticmethod
    def _load_frame(frame: str | Path | Image.Image) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")
        return Image.open(frame).convert("RGB")

    def _prepare_inputs(self, frames: Sequence[str | Path | Image.Image]) -> dict:
        images = [self._load_frame(frame) for frame in frames]
        inputs = self.processor(videos=[images], return_tensors="pt")
        prepared = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if value.is_floating_point():
                    value = value.to(device=self.device, dtype=self.dtype)
                else:
                    value = value.to(device=self.device)
            prepared[key] = value
        return prepared

    @torch.no_grad()
    def extract(self, frames4: Sequence[str | Path | Image.Image]) -> torch.Tensor:
        frames8 = expand_four_frames_to_eight(frames4)
        inputs = self._prepare_inputs(frames8)
        if hasattr(self.model, "get_vision_features"):
            outputs = self.model.get_vision_features(**inputs)
            dense = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs
        else:
            outputs = self.model(**inputs)
            dense = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        if not isinstance(dense, torch.Tensor):
            raise TypeError(f"V-JEPA2 output must be a tensor, got {type(dense).__name__}.")
        if self.highcap_packer is not None:
            pooled, metadata = self.highcap_packer.pack(dense.float(), spatial_hw=self.spatial_hw)
            self.last_tokenizer_metadata = metadata
        else:
            pooled = pool_vjepa2_tokens(dense.float(), teacher_dim=self.teacher_dim, spatial_hw=self.spatial_hw)
            self.last_tokenizer_metadata = {
                "output_tokens": 12,
                "input_shape": tuple(int(item) for item in dense.shape),
                "temporal_bins": 4,
                "spatial_tokens_per_bin": 3,
                "tokenizer_mode": "legacy_vjepa2_pool",
            }
        if pooled.shape != (1, self.num_tokens, self.teacher_dim):
            raise RuntimeError(
                f"V-JEPA2 pooled shape {tuple(pooled.shape)} != (1, {self.num_tokens}, {self.teacher_dim})."
            )
        return pooled.squeeze(0).to(dtype=torch.float16, device="cpu")

    def extract_context(self, history_frames4: Sequence[str | Path | Image.Image]) -> torch.Tensor:
        return self.extract(history_frames4)

    def extract_target(self, future_frames4: Sequence[str | Path | Image.Image]) -> torch.Tensor:
        return self.extract(future_frames4)
