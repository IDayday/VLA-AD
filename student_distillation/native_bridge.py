"""The frozen get_action DDIM path, starting directly at post384."""
from __future__ import annotations

import math
import torch
from .noise import validate_noise_tape


class NativeDDIMBridge:
    TIMESTEPS = (80, 60, 40, 20, 0)

    def __init__(self, frozen_head):
        self.head = frozen_head.eval().requires_grad_(False)
        config = self.head.config
        if config.sampling_method != "ddim" or self.head.ddim_steps != 5:
            raise ValueError("Round1 requires the native five-step DDIM head")
        # Cache Python integers once; never pass a one-dimensional Tensor as
        # fill_value to the original make_timesteps.
        self.timesteps = tuple(int(t) for t in self.head.ddim_t.detach().cpu().tolist())
        if self.timesteps != self.TIMESTEPS:
            raise ValueError(f"DDIM timesteps differ: {self.timesteps}")
        if config.action_horizon != 8 or config.action_dim != 3:
            raise ValueError("native action shape must remain [1,8,3]")
        self.std_floor = float(getattr(self.head, "eval_min_sampling_denoising_std", 1e-4))
        self.noise_clip = float(getattr(self.head, "eval_randn_clip_value", 1.0))
        self.final_clip = getattr(self.head, "final_action_clip_value", 1.0)
        if self.std_floor != 1e-4 or self.noise_clip != 1.0:
            raise ValueError("get_action requires std floor 1e-4 and noise clip +/-1")
        if self.final_clip is not None:
            self.final_clip = float(self.final_clip)
            if not math.isfinite(self.final_clip) or self.final_clip <= 0:
                raise ValueError("invalid native final action clip")
        parameters = list(self.head.parameters())
        if any(p.is_floating_point() and p.dtype != torch.float32 for p in parameters):
            raise ValueError("the frozen action head must remain FP32")
        probe_device = parameters[0].device if parameters else self.head.ddim_t.device
        with torch.no_grad():
            eta = self.head.eta(torch.zeros((1, 8, 3), device=probe_device, dtype=torch.float32))
        if not bool(torch.isfinite(eta).all()) or not bool((eta == 1.0).all()):
            raise ValueError("actual native stochastic eta must equal one")

    def _post(self, post):
        if post.dtype != torch.float32 or post.ndim != 3 or post.shape[0] != 1 or post.shape[2] != 384:
            raise ValueError("native post must preserve FP32 [1,L,384] including padding")
        if not bool(torch.isfinite(post).all()):
            raise ValueError("non-finite post")
        if self.head.training:
            raise RuntimeError("frozen action head was switched out of eval mode")
        if any(p.requires_grad for p in self.head.parameters()):
            raise RuntimeError("action-head parameters must stay frozen")

    @staticmethod
    def _step(step_index):
        if type(step_index) is not int or not 0 <= step_index < 5:
            raise ValueError("step_index must be a Python int in 0..4")
        return step_index

    def encode_context(self, history12, status8):
        if history12.shape != (1, 12) or status8.shape != (1, 8):
            raise ValueError("native context requires history12[1,12] and status8[1,8]")
        if history12.dtype != torch.float32 or status8.dtype != torch.float32:
            raise ValueError("context must already be FP32")
        if history12.device != status8.device or not bool(torch.isfinite(history12).all()) or not bool(torch.isfinite(status8).all()):
            raise ValueError("context must be finite and on one device")
        if self.head.training:
            raise RuntimeError("frozen action head must remain eval")
        with torch.no_grad():
            history = self.head.his_traj_encoder(history12.unsqueeze(1)).repeat(1, 8, 1)
            status = self.head.ego_status_encoder(status8)
        return history.detach(), status.detach()

    def mean_from_post(self, actions, step_index: int, post, context):
        self._post(post)
        step_index = self._step(step_index)
        if actions.shape != (1, 8, 3) or actions.dtype != torch.float32 or actions.device != post.device:
            raise ValueError("native actions must be FP32 [1,8,3] on the post device")
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("non-finite actions")
        history, status = context
        if history.device != post.device or status.device != post.device:
            raise ValueError("context must already be on the post device")
        t = self.head.make_timesteps(1, self.timesteps[step_index], actions.device)
        index = self.head.make_timesteps(1, step_index, actions.device)
        # Intentionally no no_grad or detach: frozen DiT parameters still give
        # the student condition an autograd path. feature_encoder is not called.
        return self.head.p_mean_variance(actions, t, index, post, history, status,
                                         deterministic=False)

    def states_from_post(self, post, context, noise_tape, stop_before: int | None = None):
        self._post(post)
        validate_noise_tape(noise_tape, device=post.device)
        if stop_before is not None:
            self._step(stop_before)
        current = noise_tape[0].clone()  # the initial Gaussian is never clipped
        states, means, logvars, stds, noises = [], [], [], [], []
        for step_index in range(5):
            if step_index == stop_before:
                return current
            states.append(current)
            mean, logvar, _ = self.mean_from_post(current, step_index, post, context)
            std = torch.exp(0.5 * logvar).to(post.dtype).clamp(min=self.std_floor)
            noise = noise_tape[step_index + 1].clamp(-self.noise_clip, self.noise_clip)
            means.append(mean)
            logvars.append(logvar)
            stds.append(std)
            noises.append(noise)
            current = mean + std * noise  # last step also retains std >= 1e-4
        before_final_clip = current
        if self.final_clip is not None:
            current = current.clamp(-self.final_clip, self.final_clip)
        trajectory = self.head.denorm_odo(current)
        if not bool(torch.isfinite(trajectory).all()):
            raise ValueError("non-finite native trajectory")
        return {"states": states, "means": means, "logvars": logvars, "stds": stds,
                "clipped_noises": noises, "normalized_actions": current,
                "before_final_clip": before_final_clip, "trajectory": trajectory}
