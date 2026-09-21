    def _reduce_chain_logprobs(
        self,
        log_probs: torch.Tensor,
        B: int,
        G: int,
        K: int,
        discount: torch.Tensor,
    ) -> torch.Tensor:
        step_logp = self._chain_step_logprobs(log_probs, B, G, K)

        if self.trajectory_logprob_reduce == "discounted_mean":
            discount = discount.to(device=step_logp.device, dtype=step_logp.dtype)
            discount_norm = discount / discount.sum().clamp(min=1e-8)
            traj_logp = (step_logp * discount_norm.view(1, K)).sum(dim=1)
        elif self.trajectory_logprob_reduce == "mean":
            traj_logp = step_logp.mean(dim=1)
        else:
            raise ValueError(f"Unsupported trajectory_logprob_reduce: {self.trajectory_logprob_reduce!r}")

        # traj_logp: [B * G]
        assert traj_logp.shape == (B * G,)
        return traj_logp
