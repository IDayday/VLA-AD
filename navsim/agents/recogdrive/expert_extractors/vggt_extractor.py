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
        require_geometry: bool = False,
    ) -> None:
        self.model_path = str(model_path)
        self.device = torch.device(device)
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        self.image_size = image_size
        self.require_geometry = require_geometry
        self.last_geometry_mode = "no_geometry"
        self.model = self._load_model(self.model_path).to(self.device)
        if self.dtype != torch.float32:
            self.model = self.model.to(dtype=self.dtype)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
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

    def _pool_patch_tokens(self, output: Any) -> torch.Tensor:
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
        return pooled

    def _extract_prediction_tensor(self, predictions: Any, names: tuple[str, ...]) -> torch.Tensor | None:
        for name in names:
            if isinstance(predictions, dict):
                value = predictions.get(name)
            else:
                value = getattr(predictions, name, None)
            if isinstance(value, torch.Tensor):
                return value
        return None

    def _compact_geometry_tokens(self, tensor: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        values = tensor.float()
        while values.ndim > 3:
            values = values.flatten(1, -2)
        if values.ndim == 2:
            values = values.unsqueeze(1)
        if values.ndim != 3:
            raise ValueError(f"VGGT geometry tensor could not be compacted from shape {tuple(tensor.shape)}.")
        pooled = torch.nn.functional.adaptive_avg_pool1d(values.transpose(1, 2), 12).transpose(1, 2)
        if pooled.shape[-1] < self.teacher_dim:
            pooled = torch.nn.functional.pad(pooled, (0, self.teacher_dim - pooled.shape[-1]))
        elif pooled.shape[-1] > self.teacher_dim:
            pooled = pooled[..., : self.teacher_dim]
        return pooled.to(device=fallback.device, dtype=fallback.dtype)

    def _try_extract_full_geometry(self, image: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor | None:
        prediction_fns = ("predict", "forward", "__call__")
        predictions = None
        for fn_name in prediction_fns:
            fn = getattr(self.model, fn_name, None)
            if fn is None:
                continue
            try:
                predictions = fn(image)
                break
            except Exception:
                predictions = None
        if predictions is None:
            return None
        geometry = self._extract_prediction_tensor(
            predictions,
            ("point_map", "pointmap", "world_points", "depth", "depth_map", "camera", "camera_tokens"),
        )
        if geometry is None:
            return None
        return self._compact_geometry_tokens(geometry, fallback)

    @torch.no_grad()
    def extract_with_geometry(self, current_frame: str | Path | Image.Image) -> dict[str, torch.Tensor | str]:
        image = self._load_image(current_frame).unsqueeze(0).unsqueeze(0).to(self.device)
        if self.dtype != torch.float32:
            image = image.to(dtype=self.dtype)
        output = self.model.aggregator(image)
        pooled = self._pool_patch_tokens(output)
        geometry = self._try_extract_full_geometry(image, pooled)
        if geometry is None:
            if self.require_geometry:
                raise RuntimeError("VGGT full geometry output is unavailable and require_geometry=True.")
            geometry = pooled
            geometry_mode = "patch_fallback"
        else:
            geometry_mode = "full_geometry"
        self.last_geometry_mode = geometry_mode
        geometry_mode_code = 2 if geometry_mode == "full_geometry" else 1
        return {
            "vggt_context_tokens": pooled.squeeze(0).to(dtype=torch.float16, device="cpu"),
            "vggt_geometry_tokens": geometry.squeeze(0).to(dtype=torch.float16, device="cpu"),
            "vggt_geometry_mode": geometry_mode,
            "vggt_geometry_mode_code": torch.tensor(geometry_mode_code, dtype=torch.int64),
        }

    @torch.no_grad()
    def extract(self, current_frame: str | Path | Image.Image) -> torch.Tensor:
        return self.extract_with_geometry(current_frame)["vggt_context_tokens"]

    def extract_context(self, current_frame: str | Path | Image.Image) -> torch.Tensor:
        return self.extract(current_frame)

    def extract_target(self, current_frame: str | Path | Image.Image) -> torch.Tensor:
        return self.extract(current_frame)

    def extract_geometry(self, current_frame: str | Path | Image.Image) -> torch.Tensor:
        return self.extract_with_geometry(current_frame)["vggt_geometry_tokens"]
