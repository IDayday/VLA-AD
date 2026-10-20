"""Stateless semantic streams, independent of workers, rank and shard layout."""
from dataclasses import dataclass, replace
import hashlib
import json
import torch
from .contracts import NoisePlan

NAMESPACES = {'rollin', 'candidate', 'fit_suffix', 'target_guard', 'diagnostic_test',
              'reference_promotion', 'final_eval', 'selected_step', 'shuffle_value'}


@dataclass(frozen=True)
class RNGKey:
    study_id: str
    train_seed: int
    student_version: int
    reference_version: int
    scene_token: str
    occurrence_id: int
    denoise_index: int

    def seed(self, namespace, draw_id=0):
        if namespace not in NAMESPACES:
            raise ValueError(f'Unregistered stream: {namespace}')
        values = [self.study_id, self.train_seed, self.student_version, self.reference_version,
                  self.scene_token, self.occurrence_id, self.denoise_index, namespace, draw_id]
        return int.from_bytes(hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).digest()[:8], 'little') % (2**63-1)

    def normal(self, shape, namespace, draw_id=0, device='cpu', dtype=torch.float32):
        # One explicit generator; no process/thread global seed changes.
        generator = torch.Generator(device=device).manual_seed(self.seed(namespace, draw_id))
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    def plan(self, namespace, draw_id=0, device='cpu', dtype=torch.float32, steps=5):
        value = self.normal((steps+1, 8, 3), namespace, draw_id, device, dtype)
        return NoisePlan(value[:1], value[1:].unsqueeze(0))

    def at_step(self, index):
        return replace(self, denoise_index=index)


def candidate_noise(key, count, mode='antithetic', device='cpu', dtype=torch.float32, bank=0):
    if count < 1:
        raise ValueError('At least one candidate required')
    if mode == 'iid':
        return key.normal((count, 8, 3), 'candidate', bank, device, dtype)
    if mode != 'antithetic' or count % 2:
        raise ValueError('Antithetic bank requires an even candidate count')
    half = key.normal((count//2, 8, 3), 'candidate', bank, device, dtype)
    return torch.cat([half, -half])
