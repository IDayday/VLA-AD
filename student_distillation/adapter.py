"""A single continuous, shared D48 residual adapter for frozen post384 inputs."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn.attention import sdpa_kernel, SDPBackend


def _finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


def validate_inputs(base_post, image_mask, text_mask, state24=None):
    """Check the native layout without cropping or rebuilding padding."""
    if base_post.dtype != torch.float32 or base_post.ndim != 3 or base_post.shape[-1] != 384:
        raise ValueError("base_post must be native FP32 [B,L,384]")
    if image_mask.dtype != torch.bool or text_mask.dtype != torch.bool:
        raise ValueError("image/text masks must be bool")
    if image_mask.shape != base_post.shape[:2] or text_mask.shape != image_mask.shape:
        raise ValueError("masks must preserve the entire native [B,L] layout")
    if image_mask.device != base_post.device or text_mask.device != base_post.device:
        raise ValueError("post and masks must already be on the same device")
    if bool((image_mask & text_mask).any()):
        raise ValueError("image and text masks overlap")
    valid = image_mask | text_mask
    if not bool(valid.any(dim=1).all()):
        raise ValueError("every scene must have at least one valid token")
    _finite(base_post, "base_post")
    if state24 is not None:
        if state24.dtype != torch.float32 or state24.shape != (base_post.shape[0], 24):
            raise ValueError("state24 must be FP32 [B,24]")
        if state24.device != base_post.device:
            raise ValueError("state24 and post must already be on the same device")
        _finite(state24, "state24")
    return valid


class ReadBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_norm = nn.LayerNorm(256)
        self.m_norm = nn.LayerNorm(256)
        self.attn = nn.MultiheadAttention(256, 4, dropout=0.0, batch_first=True)
        self.ff_norm = nn.LayerNorm(256)
        self.ff = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256))

    def forward(self, queries, memory, valid):
        key_value = self.m_norm(memory)
        # Flash/memory-efficient attention backward can differ by a few ULPs
        # across independent restores even with identical RNG and parameters.
        # Scope deterministic math attention to the NEW adapter only; the
        # frozen native DiT remains outside this context and unchanged.
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            attention, _ = self.attn(self.q_norm(queries), key_value, key_value,
                                     key_padding_mask=~valid, need_weights=False)
        queries = queries + attention
        return queries + self.ff(self.ff_norm(queries))


class ContinuousD48Adapter(nn.Module):
    """All scenes use four queries and the same two cross-attention blocks.

    ``basis`` must be the newly refitted teacher's FP64 orthonormal basis.
    Its center is deliberately absent: this module decodes differences only.
    """
    def __init__(self, basis, d0, token_mean, token_std, state_mean, state_std,
                 cap_in_d0):
        super().__init__()
        basis = torch.as_tensor(basis).detach()
        if basis.dtype != torch.float64 or basis.shape != (384, 48):
            raise ValueError("verify the actual FP64 basis[384,48] before loading the FP32 copy")
        _finite(basis, "basis")
        gram = basis.T @ basis
        if not torch.allclose(gram, torch.eye(48, device=basis.device, dtype=torch.float64),
                              atol=1e-7, rtol=1e-7):
            raise ValueError("D48 basis is not orthonormal")
        d0 = float(d0)
        self.cap_in_d0 = float(cap_in_d0)
        if not math.isfinite(d0) or d0 <= 0 or not math.isfinite(self.cap_in_d0) or self.cap_in_d0 <= 0:
            raise ValueError("d0 and the frozen-library cap_in_d0 must be positive and finite")
        self.register_buffer("basis", basis.float().clone())
        self.register_buffer("d0", torch.tensor(d0, dtype=torch.float32))
        for name, value, width in (("token_mean", token_mean, 384), ("token_std", token_std, 384),
                                   ("state_mean", state_mean, 24), ("state_std", state_std, 24)):
            value = torch.as_tensor(value, dtype=torch.float32).detach().clone()
            if value.shape != (width,):
                raise ValueError(f"{name} must have shape [{width}]")
            _finite(value, name)
            if name.endswith("std"):
                if bool((value < 0).any()):
                    raise ValueError(f"{name} cannot be negative")
                value = value.clamp_min(1e-3)
            self.register_buffer(name, value)
        self.input_proj = nn.Linear(384, 256)
        self.role_embed = nn.Embedding(3, 256)  # padding, image, valid text
        self.queries = nn.Parameter(torch.randn(1, 4, 256) * 0.02)
        self.state_encoder = nn.Sequential(nn.Linear(24, 256), nn.GELU(), nn.Linear(256, 256))
        self.blocks = nn.ModuleList([ReadBlock(), ReadBlock()])
        self.output_norm = nn.LayerNorm(1024)
        self.output_hidden = nn.Sequential(nn.Linear(1024, 256), nn.GELU())
        self.output_projection = nn.Linear(256, 48)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, base_post, image_mask, text_mask, state24):
        valid = validate_inputs(base_post, image_mask, text_mask, state24)
        if self.basis.device != base_post.device or self.basis.dtype != torch.float32:
            raise ValueError("adapter must be FP32 on the native input device")
        roles = image_mask.long() + 2 * text_mask.long()
        memory = self.input_proj((base_post - self.token_mean) / self.token_std) + self.role_embed(roles)
        state = (state24 - self.state_mean) / self.state_std
        queries = self.queries.expand(base_post.shape[0], -1, -1) + self.state_encoder(state).unsqueeze(1)
        for block in self.blocks:
            queries = block(queries, memory, valid)
        features = self.output_hidden(self.output_norm(queries.flatten(1)))
        coefficients = self.output_projection(features)
        _finite(coefficients, "adapter coefficients")
        # Clamp before sqrt. In particular, q=0 has scale exactly one and a
        # live derivative, unlike the detached teacher-only native_cap API.
        norm = coefficients.square().sum(-1, keepdim=True).clamp_min(1e-12).sqrt()
        scale = (self.cap_in_d0 / norm).clamp(max=1.0)
        coefficients_capped = coefficients * scale
        delta = (coefficients_capped @ self.basis.T) * (self.d0 * math.sqrt(384.0))
        cap = self.d0.double() * self.cap_in_d0
        tolerance = torch.maximum(cap * 1e-4, cap.new_tensor(1e-7))
        cap_triggered = (scale[:, 0] < 1).detach()
        total_correction = torch.ones_like(scale)
        for corrections in range(9):
            # where preserves all original padding values, including signed zero.
            post = torch.where(valid.unsqueeze(-1), base_post + delta.unsqueeze(1), base_post)
            with torch.no_grad():
                difference = post.double() - base_post.double()
                actual_rms = ((difference.square() * valid.unsqueeze(-1)).sum((1, 2)) /
                              (valid.sum(1).double() * 384.0)).sqrt()
                over = actual_rms > cap + tolerance
                cap_triggered = cap_triggered | over
                if not bool(over.any()):
                    break
                if corrections == 8:
                    raise RuntimeError("Post-rounding RMS cap did not converge after eight corrections")
                correction = torch.where(over, (cap / actual_rms.clamp_min(1e-300)).clamp(max=1),
                                         torch.ones_like(actual_rms)).float().unsqueeze(1)
            delta = delta * correction
            total_correction = total_correction * correction
        _finite(post, "student post")
        return {"post": post, "delta": delta, "coefficients_raw": coefficients,
                "coefficients_capped": coefficients_capped * total_correction,
                "coefficient_scale": scale * total_correction, "actual_rms": actual_rms,
                "valid_mask": valid, "cap_triggered": cap_triggered,
                "cap_corrections": corrections, "cap_rms": cap, "cap_tolerance": tolerance}
