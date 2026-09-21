    def forward_lfp_grpo(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        tokens_list,
        sample_time: Optional[int] = None,
    ) -> BatchFeature:
        """Runs one on-policy trajectory-level LFP-GRPO update."""
        if self.stage3_algorithm != "lfp_grpo":
            raise RuntimeError("forward_lfp_grpo requires stage3_algorithm='lfp_grpo'.")
        if self.lfp_reference_cache is None or self.lfp_metric_adapter is None:
            raise RuntimeError("LFP reference cache/metric adapter is not initialized.")
        self.set_frozen_modules_to_eval_mode()
        B = int(vl_features.shape[0])
        G = int(sample_time if sample_time is not None else self.grpo_sample_time)
        if G <= 0:
            raise ValueError("LFP-GRPO group size must be positive.")
        tokens = [str(token) for token in tokens_list]
        if len(tokens) != B:
            raise ValueError(f"Expected {B} scene tokens, got {len(tokens)}.")

        vl_features_rep = vl_features.repeat_interleave(G, 0)
        his_traj_rep = action_input.his_traj.repeat_interleave(G, 0)
        status_feature_rep = action_input.status_feature.repeat_interleave(G, 0)
        condition_input_rep = self._repeat_expert_action_input(action_input, G)

        self._reset_planning_adapter_forward_count()
        current_dit_context = self._prepare_dit_context(
            vl_features_rep,
            condition_input_rep,
            # Exact KL requires current and frozen policies to see the same condition.
            # DDIM noise remains the sole on-policy exploration source in LFP.
            training=False,
            allow_target_tokens=False,
        )
        with torch.no_grad():
            chains, trajectories = self.sample_chain(
                vl_features_rep,
                his_traj_rep,
                status_feature_rep,
                deterministic=False,
                action_input=condition_input_rep,
                allow_target_tokens=False,
                prepared_dit_context=current_dit_context,
            )

        tokens_rep = [token for token in tokens for _ in range(G)]
        metric_cache: Dict[str, Any] = {}
        if self.lfp_grpo_cfg.benchmark == "navsim_v1":
            for token in set(tokens):
                try:
                    path = self.metric_cache_loader.metric_cache_paths[token]
                except KeyError as error:
                    raise KeyError(f"Stage3 metric cache is missing token {token!r}.") from error
                with lzma.open(path, "rb") as handle:
                    metric_cache[token] = pickle.load(handle)
        metrics = self._evaluate_lfp_rollouts(trajectories, tokens_rep, metric_cache, B, G)
        reference = self.lfp_reference_cache.get(tokens, metrics.scalar.device, metrics.scalar.dtype)
        advantage_output = compute_lfp_advantages(metrics, reference, self.lfp_grpo_cfg)
        if bool(self.lfp_grpo_cfg.curriculum_enabled):
            self._accumulate_lfp_epoch_energy(tokens, advantage_output.energy)

        num_denoising_steps = int(chains.shape[1] - 1)
        discount = self._stage3_discount(
            num_denoising_steps,
            device=chains.device,
            dtype=metrics.scalar.dtype,
        )
        log_probs = self.get_logprobs(
            vl_features_rep,
            his_traj_rep,
            status_feature_rep,
            chains,
            deterministic=False,
            action_input=condition_input_rep,
            prepared_dit_context=current_dit_context,
        )
        trajectory_logp = self._reduce_chain_logprobs(log_probs, B, G, num_denoising_steps, discount)
        policy_loss = trajectory_reinforce_loss(advantage_output.advantages, trajectory_logp)

        self.old_policy.eval()
        with torch.no_grad():
            reference_dit_context = self.old_policy._prepare_dit_context(
                vl_features_rep,
                condition_input_rep,
                training=False,
                allow_target_tokens=False,
            )
        exact_kl = self._chain_transition_reference_kl(
            vl_features_rep,
            his_traj_rep,
            status_feature_rep,
            chains,
            condition_input_rep,
            B,
            G,
            num_denoising_steps,
            discount,
            current_dit_context=current_dit_context,
            reference_dit_context=reference_dit_context,
        ).to(policy_loss)
        kl_coeff = float(self.lfp_grpo_cfg.reference_kl_coeff)
        total_loss = policy_loss + kl_coeff * exact_kl
        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"Non-finite LFP loss: policy={policy_loss.detach().item()}, kl={exact_kl.detach().item()}."
            )

        diagnostics = dict(advantage_output.diagnostics)
        planning_diagnostics = current_dit_context.get("diagnostics", {})
        for key in (
            "planning_token_norm",
            "planning_token_pairwise_cosine",
            "planning_condition_keep_ratio",
            "planning_context_gate",
            "planning_adapter_forward_count",
        ):
            value = planning_diagnostics.get(key)
            if isinstance(value, torch.Tensor):
                diagnostics[key] = value.detach().float().mean().to(total_loss)
        diagnostics.update(
            {
                "lfp_policy_loss": policy_loss.detach(),
                "lfp_exact_kl": exact_kl.detach(),
                "lfp_reference_kl_coeff": total_loss.new_tensor(kl_coeff),
                "lfp_trajectory_logprob_mean": trajectory_logp.detach().mean(),
                "scene_stage_type": total_loss.new_tensor(
                    float(
                        sum(
                            {"first": 1, "followup": 2}.get(
                                str(self.lfp_reference_cache.records[token].get("scene_stage_type", "unknown")),
                                0,
                            )
                            for token in tokens
                        )
                        / max(len(tokens), 1)
                    )
                ),
            }
        )
        zero = total_loss.new_zeros(())
        return BatchFeature(
            data={
                "loss": total_loss,
                "reward": metrics.scalar.mean().detach(),
                "policy_loss": policy_loss,
                "bc_loss": zero,
                "bc_coeff": zero,
                "reference_kl_loss": exact_kl,
                "reference_kl_coeff": total_loss.new_tensor(kl_coeff),
                "trajectory_logp": trajectory_logp.detach().mean(),
                **diagnostics,
            }
        )
