    def _chain_transition_reference_kl(
        self,
        vl_features: torch.Tensor,
        his_traj_features: torch.Tensor,
        ego_status_features: torch.Tensor,
        chains: torch.Tensor,
        action_input: Optional[BatchFeature],
        B: int,
        G: int,
        num_denoising_steps: int,
        discount: torch.Tensor,
        current_dit_context: Optional[Dict[str, Any]] = None,
        reference_dit_context: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Computes exact per-transition KL(current || frozen reference) on sampled chains."""
        if not hasattr(self, "old_policy"):
            raise RuntimeError("reference_kl_coeff > 0 requires a frozen old_policy reference.")
        total_samples = B * G
        if chains.shape[0] != total_samples:
            raise ValueError(f"Expected chains batch {total_samples}, got {chains.shape[0]}.")

        chunk_size = int(getattr(self, "reference_kl_chunk_size", 0))
        if chunk_size <= 0 or chunk_size >= total_samples:
            current_dist = self._chain_transition_distribution(
                vl_features,
                his_traj_features,
                ego_status_features,
                chains,
                deterministic=False,
                action_input=action_input,
                prepared_dit_context=current_dit_context,
            )
            self.old_policy.eval()
            with torch.no_grad():
                reference_dist = self.old_policy._chain_transition_distribution(
                    vl_features,
                    his_traj_features,
                    ego_status_features,
                    chains,
                    deterministic=False,
                    action_input=action_input,
                    prepared_dit_context=reference_dit_context,
                )
            step_kl = kl_divergence(current_dist, reference_dist).mean(dim=(1, 2))
            step_kl = step_kl.view(total_samples, num_denoising_steps)
            if self.trajectory_logprob_reduce == "discounted_mean":
                discount = discount.to(device=step_kl.device, dtype=step_kl.dtype)
                discount_norm = discount / discount.sum().clamp(min=1e-8)
                traj_kl = (step_kl * discount_norm.view(1, num_denoising_steps)).sum(dim=1)
            elif self.trajectory_logprob_reduce == "mean":
                traj_kl = step_kl.mean(dim=1)
            else:
                raise ValueError(f"Unsupported trajectory_logprob_reduce: {self.trajectory_logprob_reduce!r}")
            return traj_kl.mean()

        kl_sum = chains.new_zeros(())
        self.old_policy.eval()
        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)
            action_input_chunk = self._slice_action_input_batch(action_input, start, end)
            current_dist = self._chain_transition_distribution(
                vl_features[start:end],
                his_traj_features[start:end],
                ego_status_features[start:end],
                chains[start:end],
                deterministic=False,
                action_input=action_input_chunk,
                prepared_dit_context=self._slice_prepared_dit_context(current_dit_context, start, end),
            )
            with torch.no_grad():
                reference_dist = self.old_policy._chain_transition_distribution(
                    vl_features[start:end],
                    his_traj_features[start:end],
                    ego_status_features[start:end],
                    chains[start:end],
                    deterministic=False,
                    action_input=action_input_chunk,
                    prepared_dit_context=self._slice_prepared_dit_context(reference_dit_context, start, end),
                )
            step_kl = kl_divergence(current_dist, reference_dist).mean(dim=(1, 2))
            step_kl = step_kl.view(end - start, num_denoising_steps)
            if self.trajectory_logprob_reduce == "discounted_mean":
                discount_chunk = discount.to(device=step_kl.device, dtype=step_kl.dtype)
                discount_norm = discount_chunk / discount_chunk.sum().clamp(min=1e-8)
                traj_kl = (step_kl * discount_norm.view(1, num_denoising_steps)).sum(dim=1)
            elif self.trajectory_logprob_reduce == "mean":
                traj_kl = step_kl.mean(dim=1)
            else:
                raise ValueError(f"Unsupported trajectory_logprob_reduce: {self.trajectory_logprob_reduce!r}")
            kl_sum = kl_sum + traj_kl.sum()
        return kl_sum / float(total_samples)
