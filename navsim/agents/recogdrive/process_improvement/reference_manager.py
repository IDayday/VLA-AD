"""Immutable T0 and explicit, versioned reference snapshots; no implicit EMA."""
import copy
from navsim.agents.recogdrive.opd.checkpoint_io import state_hash


class ReferenceManager:
    def __init__(self, teacher):
        self.t0 = teacher
        self.t0.requires_grad_(False).eval()
        self.t0_hash = state_hash(teacher)
        self.reference = teacher
        self.version = 0

    def promote(self, student, evidence):
        if not evidence.get('epoch_complete') or not evidence.get('cal_promotion_only'):
            raise ValueError('Promotion requires a complete epoch and independent promotion set')
        if not (evidence['gain'] >= .002 and evidence['paired_ci_lower'] > 0 and evidence['safety_noninferior']):
            return False
        self.reference = copy.deepcopy(student).requires_grad_(False).eval()
        self.version += 1
        assert state_hash(self.t0) == self.t0_hash
        return True
