from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from .pooling import pool_vggt_tokens


class VGGTExtractor:
    teacher_dim = 2048
    layer_index = 23
    grid_size = (37, 37)

    def __init__(
        self,
        model_path: str | Path = "facebook/VGGT-1B",
        *,
        device: str | torch.device = "cuda",
        precision: str = "bf16",
        image_size: int = 518,
    ) -> None:
        self.model_path = str(model_path)
        self.device = torch.device(device)
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        self.image_size = image_size
        self.model = self._load_model(self.model_path).to(self.device)
        if self.dtype != torch.float32:
            self.model = self.model.to(dtype=self.dtype)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @staticmethod
    def _load_model(model_path: str):
        try:
            from vggt.models.vggt import VGGT
        except Exception:
            try:
                from vggt import VGGT
            except Exception as exc:
                raise ImportError(
                    "Could not import VGGT. Install the VGGT package or make facebook/VGGT-1B code importable."
                ) from exc
        if hasattr(VGGT, "from_pretrained"):
            return VGGT.from_pretrained(model_path)
        return VGGT(model_path)

    def _load_image(self, frame: str | Path | Image.Image) -> torch.Tensor:
        image = frame.convert("RGB") if isinstance(frame, Image.Image) else Image.open(frame).convert("RGB")
        return self.transform(image)

    @staticmethod
    def _unpack_aggregator_output(output: Any) -> tuple[list[torch.Tensor], int]:
        if isinstance(output, dict):
            tokens_list = output.get("aggregated_tokens_list") or output.get("tokens_list") or output.get("tokens")
            patch_start_idx = output.get("patch_start_idx")
        elif isinstance(output, (tuple, list)):
            tokens_list = output[0]
            patch_start_idx = output[1] if len(output) > 1 else None
        else:
            tokens_list = getattr(output, "aggregated_tokens_list", None)
            patch_start_idx = getattr(output, "patch_start_idx", None)
        if tokens_list is None:
            raise KeyError("VGGT aggregator output did not contain aggregated_tokens_list.")
        if patch_start_idx is None:
            raise KeyError("VGGT aggregator output did not contain patch_start_idx.")
        return list(tokens_list), int(patch_start_idx)

    @torch.no_grad()
    def extract(self, current_frame: str | Path | Image.Image) -> torch.Tensor:
        image = self._load_image(current_frame).unsqueeze(0).unsqueeze(0).to(self.device)
        if self.dtype != torch.float32:
            image = image.to(dtype=self.dtype)
        output = self.model.aggregator(image)
        tokens_list, patch_start_idx = self._unpack_aggregator_output(output)
        if len(tokens_list) <= self.layer_index:
            raise IndexError(f"VGGT aggregator returned {len(tokens_list)} layers, need layer {self.layer_index}.")
        layer_tokens = tokens_list[self.layer_index]
        if layer_tokens.ndim == 4:
            layer_tokens = layer_tokens[:, 0]
        if layer_tokens.ndim != 3:
            raise ValueError(f"VGGT layer tokens must have shape [B,N,D], got {tuple(layer_tokens.shape)}.")
        patch_tokens = layer_tokens[:, patch_start_idx:, :]
        pooled = pool_vggt_tokens(patch_tokens.float(), teacher_dim=self.teacher_dim, grid_size=self.grid_size)
        if pooled.shape != (1, 12, self.teacher_dim):
            raise RuntimeError(f"VGGT pooled shape {tuple(pooled.shape)} != (1, 12, {self.teacher_dim}).")
        return pooled.squeeze(0).to(dtype=torch.float16, device="cpu")

    def extract_context(self, current_frame: str | Path | Image.Image) -> torch.Tensor:
        return self.extract(current_frame)

    def extract_target(self, current_frame: str | Path | Image.Image) -> torch.Tensor:
        return self.extract(current_frame)
