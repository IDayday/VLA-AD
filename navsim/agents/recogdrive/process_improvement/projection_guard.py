"""Independent paired acceptance; rejection always restores ordinary reference OPD."""
import numpy as np
import torch
from .candidate_value import evaluate_candidates, evaluate_kernels
from .contracts import TargetRecord
from .improvement_target import proposal
from .rng_plan import candidate_noise


class TargetBuilder:
    def __init__(self, reference_adapter, scorer, config, reference_version=0):
        self.adapter, self.scorer, self.config = reference_adapter, scorer, config
        self.reference_version = reference_version

    @torch.no_grad()
    def build(self, state, context, index, key, metric_cache, *, bank=0, trace=False, ablation=None):
        if len(state) != 1 or key.denoise_index != index or key.reference_version != self.reference_version:
            raise ValueError('One semantically keyed scene/state per target construction')
        cfg = self.config
        stats = self.adapter.transition_stats(state, index, context)
        raw = candidate_noise(key, cfg['candidate_count'], cfg['candidate_mode'], state.device, state.dtype, bank)
        candidates = self.adapter.sample_next(stats, raw)
        fit, _ = evaluate_candidates(self.adapter, context, candidates, index, key, cfg['fit_replicates'],
                                    'fit_suffix', self.scorer, key.scene_token, metric_cache, bank)
        values = torch.from_numpy(fit[:, :, -1].mean(1)).to(device=state.device, dtype=torch.float32)
        if ablation == 'SHUFFLED_VALUE':
            permutation = torch.argsort(key.normal((len(values),), 'shuffle_value', bank, state.device))
            values = values[permutation]
        elif ablation == 'UNIFORM_TARGET':
            values = torch.zeros_like(values)
        proposed = proposal(stats.mean[0], stats.std[0], candidates, values,
                            temperature=cfg['temperature'], rho=cfg['rho'], radius=cfg['radius'],
                            ess_fraction=cfg['ess_fraction'], flat_tolerance=cfg['flat_value_tolerance'])
        native_proposal = proposed.mean.unsqueeze(0).to(stats.mean.dtype)
        guard, _ = evaluate_kernels(self.adapter, context, stats, [stats.mean, native_proposal], key,
                                    cfg['guard_replicates'], 'target_guard', self.scorer,
                                    key.scene_token, metric_cache, bank)
        delta = guard[1]-guard[0]
        gain = float(delta[:, -1].mean())
        safety = delta[:, [0, 1, 3]].mean(0)
        accepted = gain > cfg['guard_margin'] and bool((safety >= -cfg['safety_tolerance']).all())
        if ablation == 'NO_GUARD':
            accepted = True
        reason = proposed.reason if proposed.flat else ('ACCEPTED' if accepted else 'GUARD_REJECTED')
        target = native_proposal if accepted else stats.mean
        fields = dict(temperature=proposed.temperature, ess=proposed.ess, radius_before=proposed.radius_before,
                      radius_after=proposed.radius_after, guard_gain=gain, guard_safety=safety.tolist(),
                      value_mean=float(values.mean()), value_range=float(values.max()-values.min()),
                      requested_shift_rms=float((proposed.mean-stats.mean[0].float()).square().mean().sqrt()),
                      realized_shift_rms=float((native_proposal-stats.mean).float().square().mean().sqrt()),
                      candidate_clip_fraction=float((raw.abs()>stats.noise_clip).float().mean()),
                      fit_namespace='fit_suffix', guard_namespace='target_guard', test_namespace='diagnostic_test',
                      ablation=ablation, candidate_mode=cfg['candidate_mode'], budget=self.scorer.counter.state_dict())
        full = dict(raw_state=state.cpu(), reference_mean=stats.mean.cpu(), actual_std=stats.std.cpu(),
                    raw_candidate_noise=raw.cpu(), candidates=candidates.cpu(), fit_scores=fit,
                    weights=proposed.weights.cpu(), guard_scores=guard, rng_key=key.__dict__) if trace else {}
        return TargetRecord(key.scene_token, key.student_version, key.reference_version, index, stats.timestep,
                            target.detach().clone(), native_proposal.detach().clone(), stats.mean.detach().clone(),
                            stats.std.detach().clone(), accepted, reason, fields, full)
