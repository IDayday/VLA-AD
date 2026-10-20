"""Native DDIM transitions and explicit-noise prefix/suffix execution.

All transition mathematics is delegated to the baseline p_mean_variance.
Native clipping/rounding is applied once, at the same location as get_action.
The final raw transition state is never confused with its clamped final action.
"""
import torch
from navsim.agents.recogdrive.opd.ddim_adapter import DDIMAdapter
from .contracts import TransitionStats, Rollin


def expand_context(context, size):
    result = {}
    for key, value in context.items():
        if value is None:
            result[key] = None
        elif value.shape[0] == size:
            result[key] = value
        elif value.shape[0] == 1:
            result[key] = value.expand(size, *value.shape[1:])
        else:
            raise ValueError('Context expansion requires one scene or aligned batch')
    return result


class NativeAdapter(DDIMAdapter):
    def __init__(self, head, kernel='native_inference', deterministic=False):
        super().__init__(head)
        if kernel not in ('native_inference', 'native_grpo'):
            raise ValueError(kernel)
        self.kernel = kernel
        self.deterministic = deterministic

    def transition_stats(self, state, index, context):
        mean, logvar, x0 = self.transition(state, index, context, self.deterministic)
        # In native code the variance is cast to raw VLM token dtype (float32 here).
        dtype = context['context_embeds'].dtype
        std = (.5 * logvar).exp().to(dtype)
        floor = getattr(self.head, 'eval_min_sampling_denoising_std', .0001) if self.kernel == 'native_inference' else self.head.min_sampling_denoising_std
        std = torch.zeros_like(std) if self.deterministic else std.clamp(min=floor)
        clip = getattr(self.head, 'eval_randn_clip_value', 1.) if self.kernel == 'native_inference' else getattr(self.head, 'randn_clip_value', 5.)
        return TransitionStats(mean, std, x0, index, int(self.head.ddim_t[index]), self.kernel, clip, self.deterministic)

    def sample_next(self, stats, raw_noise, mean=None):
        if raw_noise.shape[-2:] != (8, 3) or not torch.isfinite(raw_noise).all():
            raise ValueError('Invalid complete-trajectory transition noise')
        noise = raw_noise.clamp(-stats.noise_clip, stats.noise_clip)
        # No intermediate-state clamp; preserve baseline dtype promotion.
        return (stats.mean if mean is None else mean) + stats.std * noise

    def finalize(self, raw):
        clip = getattr(self.head, 'final_action_clip_value', 1.)
        return self.head.denorm_odo(raw.clamp(-clip, clip) if clip is not None else raw)

    @torch.no_grad()
    def rollout_suffix(self, context, start_state, next_index, raw_noises, return_states=False):
        count = self.head.ddim_steps
        if not 0 <= next_index <= count:
            raise ValueError('Suffix index outside [0,K]')
        if raw_noises.shape != (len(start_state), count, 8, 3):
            raise ValueError('Suffix noise must retain absolute execution indices')
        context = expand_context(context, len(start_state))
        state = start_state.detach().clone()
        states = [state.clone()]
        for index in range(next_index, count):
            stats = self.transition_stats(state, index, context)
            state = self.sample_next(stats, raw_noises[:, index])
            states.append(state.clone())
        trajectory = self.finalize(state)
        return (trajectory, torch.stack(states, 1)) if return_states else trajectory

    @torch.no_grad()
    def sample_rollin(self, tokens, inputs, noise, student_version=0):
        context = self.prepare_conditioning(tokens, inputs)
        trajectory, states = self.rollout_suffix(context, noise.initial, 0, noise.steps, True)
        return Rollin(states.detach(), trajectory.detach(), student_version,
                      tuple(p._version for p in self.head.parameters()), id(self.head), noise)

    def evaluate_rollin(self, rollin, tokens, inputs, student_version):
        if (rollin.source != id(self.head) or rollin.student_version != student_version or
                rollin.parameter_versions != tuple(p._version for p in self.head.parameters())):
            raise ValueError('Stale/nonstudent rollout cannot be used for on-policy objective')
        # Crucially recompute trainable feature projection and conditioning with gradients.
        context = self.prepare_conditioning(tokens, inputs)
        return torch.stack([self.transition_stats(rollin.states[:, i].detach().clone(), i, context).mean
                            for i in range(self.head.ddim_steps)], 1)
