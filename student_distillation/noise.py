"""Stateless scheduling and native-shape, RNG-isolated action noise tapes."""
from __future__ import annotations

import hashlib
import json
from numbers import Integral
from pathlib import Path
import numpy as np
import torch

NATIVE_SHAPE = (1, 8, 3)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        raise ValueError(f"{name} must be a Python integer >= {minimum}")
    return int(value)


def _hash32(payload):
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16)


def train_noise_seed(token: str, training_seed: int, bank: int) -> int:
    training_seed = _integer(training_seed, "training_seed")
    bank = _integer(bank, "bank")
    if not isinstance(token, str) or not token or bank >= 5:
        raise ValueError("training noise requires a nonempty token and bank 0..4")
    return _hash32(f"NRKD_train_v1|{training_seed}|{token}|{bank}")


def scene_seed(token: str, noise_index: int) -> int:
    noise_index = _integer(noise_index, "noise_index")
    if not isinstance(token, str) or not token:
        raise ValueError("scene token must be a nonempty string")
    return _hash32(f"{token}|20261008|{noise_index}")


def training_schedule(token: str, training_seed: int, epoch_index: int) -> dict[str, int]:
    """Visit all 25 time/bank pairs every 25 functional epochs (epochs 4+)."""
    training_seed = _integer(training_seed, "training_seed")
    epoch_index = _integer(epoch_index, "epoch_index")
    if not isinstance(token, str) or not token:
        raise ValueError("scene token must be a nonempty string")
    r = epoch_index - 4
    time_offset = _hash32(f"NRKD_time_v1|{training_seed}|{token}")
    noise_offset = _hash32(f"NRKD_noise_v1|{training_seed}|{token}")
    return {"time_index": int((r + time_offset) % 5),
            "noise_bank": int((r // 5 + noise_offset) % 5)}


def validate_noise_tape(tape, *, device=None):
    """Explicit reference tapes are accepted as-is, without RNG regeneration."""
    if not isinstance(tape, torch.Tensor) or tape.shape != (6,) + NATIVE_SHAPE:
        raise ValueError("noise_tape must have the six native draws [6,1,8,3]")
    if tape.dtype != torch.float32 or not bool(torch.isfinite(tape).all()):
        raise ValueError("noise_tape must be finite FP32")
    if tape.requires_grad:
        raise ValueError("action noise must be detached")
    if device is not None and tape.device != torch.device(device):
        raise ValueError("noise_tape must already be on the post device")
    return tape


def make_noise_tape(seed: int, device) -> torch.Tensor:
    """Draw initial Gaussian then five updates, each at native [1,8,3].

    A device-local generator leaves global CPU/CUDA/shuffle RNG untouched.
    Six separate calls are essential: a combined draw changes CUDA RNG offsets.
    """
    seed = _integer(seed, "seed")
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.stack([torch.randn(NATIVE_SHAPE, generator=generator, device=device,
                                    dtype=torch.float32) for _ in range(6)])


def evaluation_noise_tape(token: str, noise_index: int, device, reference_tape=None):
    """Use a supplied reference tape, or the frozen S0--S9 seed convention."""
    if reference_tape is not None:
        return validate_noise_tape(reference_tape, device=device)
    return make_noise_tape(scene_seed(token, noise_index), device)


class EvaluationNoiseStore:
    """Read the frozen explicit population bank; never regenerate missing noise.

    The Train bank can contain the full historical navtrain population. Exact
    token joins, rather than scene indices, select the requested population.
    """
    def __init__(self, manifest, population, required_indices, expected_tokens=None):
        from .checkpointing import sha256
        manifest = Path(manifest).resolve()
        value = json.loads(manifest.read_text())
        if value.get("schema") != "native_explicit_evaluation_noise_v1":
            raise ValueError("Expected frozen native_explicit_evaluation_noise_v1 manifest")
        entry = value["populations"][population]
        path, tokens_path = Path(entry["path"]), Path(entry["tokens_path"])
        path = path if path.is_absolute() else manifest.parent / path
        tokens_path = tokens_path if tokens_path.is_absolute() else manifest.parent / tokens_path
        if sha256(path) != entry["sha256"] or sha256(tokens_path) != entry["tokens_sha256"]:
            raise ValueError("Explicit evaluation noise bank/token SHA mismatch")
        self.bank = np.load(path, mmap_mode="r", allow_pickle=False)
        indices = entry["noise_indices"]
        if entry.get("dtype") != "float32" or self.bank.dtype != np.float32 or list(self.bank.shape) != entry["shape"]:
            raise ValueError("Explicit evaluation noise bank shape/dtype mismatch")
        if self.bank.ndim != 6 or tuple(self.bank.shape[2:]) != (6, 1, 8, 3) or len(indices) != self.bank.shape[1]:
            raise ValueError("Evaluation tapes must retain six separate native [1,8,3] draws")
        if len(set(indices)) != len(indices) or any(type(j) is not int for j in indices):
            raise ValueError("Noise bank indices must be unique Python integers")
        if not set(required_indices) <= set(indices):
            raise ValueError("Frozen bank does not cover every required evaluation noise; seed fallback is forbidden")
        tokens = json.loads(tokens_path.read_text())
        if not isinstance(tokens, list) or len(tokens) != len(self.bank):
            raise ValueError("Explicit noise token index must match bank row count")
        names = [row["token"] if isinstance(row, dict) else row for row in tokens]
        if len(set(names)) != len(names) or any(not isinstance(token, str) for token in names):
            raise ValueError("Noise token index contains duplicates or invalid tokens")
        self.positions = {token: position for position, token in enumerate(names)}
        self.noise_positions = {j: position for position, j in enumerate(indices)}
        if expected_tokens is not None and not set(expected_tokens) <= set(self.positions):
            raise ValueError("Explicit bank is missing requested population tokens")
        self.identity = {"manifest": str(manifest), "manifest_sha256": sha256(manifest),
                         "population": population, "path": str(path.resolve()), "sha256": entry["sha256"],
                         "tokens_path": str(tokens_path.resolve()), "tokens_sha256": entry["tokens_sha256"],
                         "noise_indices": list(indices), "source": entry.get("source"),
                         "seed_fallback": False}

    def tape(self, token, noise_index, device):
        if token not in self.positions or noise_index not in self.noise_positions:
            raise ValueError("Missing explicit scene/noise tape; seed fallback is forbidden")
        value = np.array(self.bank[self.positions[token], self.noise_positions[noise_index]], copy=True)
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite explicit noise tape")
        return torch.as_tensor(value, device=device, dtype=torch.float32)
