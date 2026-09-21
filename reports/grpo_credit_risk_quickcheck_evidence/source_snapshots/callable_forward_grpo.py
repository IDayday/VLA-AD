    def forward_grpo(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        tokens_list,
        sample_time: Optional[int] = None,
        deterministic=False,
        bc_coeff: Optional[float] = None,
        use_bc_loss: bool = True
    ) -> BatchFeature:
        """Computes the Diffusion-GRPO loss."""
        if self.stage3_algorithm == "lfp_grpo":
            return self.forward_lfp_grpo(
                vl_features,
                action_input,
                tokens_list,
                sample_time=sample_time,
            )
        self.set_frozen_modules_to_eval_mode()
        B = vl_features.shape[0]
        G = int(sample_time if sample_time is not None else getattr(self, "grpo_sample_time", 8))
        if G <= 0:
            raise ValueError("GRPO sample_time must be positive.")

        vl_features_rep = vl_features.repeat_interleave(G, 0)
        his_traj_rep = action_input.his_traj.repeat_interleave(G, 0)
        status_feature_rep = action_input.status_feature.repeat_interleave(G, 0)
        expert_action_input_rep = self._repeat_expert_action_input(action_input, G)

        sampled_from_behavior_policy = False
        behavior_policy_synced = False
        if self.use_trajectory_level_objective and self.use_gspo_ratio and self.behavior_policy_sample:
            if not hasattr(self, "behavior_policy"):
                raise RuntimeError("use_gspo_ratio=True requires a frozen behavior_policy initialized in _init_grpo.")
            sync_interval = max(1, int(self.behavior_policy_sync_interval))
            if int(getattr(self, "grpo_update_counter", 0)) % sync_interval == 0:
                self._sync_behavior_policy()
                behavior_policy_synced = True
            sampled_from_behavior_policy = True
            self.behavior_policy.eval()
            with torch.no_grad():
                chains, trajs = self.behavior_policy.sample_chain(
                    vl_features_rep,
                    his_traj_rep,
                    status_feature_rep,
                    deterministic=False,
                    action_input=expert_action_input_rep,
                )
        else:
            with torch.no_grad():
                chains, trajs = self.sample_chain(
                    vl_features_rep,
                    his_traj_rep,
                    status_feature_rep,
                    deterministic=False,
                    action_input=expert_action_input_rep,
                )

        tokens_rep = [tok for tok in tokens_list for _ in range(G)]
        unique_tokens = set(tokens_list)
        metric_cache = {}
        for token in unique_tokens:
            path = self.metric_cache_loader.metric_cache_paths[token]
            with lzma.open(path, 'rb') as f:
                metric_cache[token] = pickle.load(f)
        offline_cfg = getattr(self, "offline_rl_cfg", None)
        use_grpo_buffer_guidance = (
            offline_cfg is not None
            and bool(offline_cfg.enabled)
            and bool(offline_cfg.grpo_buffer_guidance_enabled)
        )
        grpo_buffer_guidance: Dict[str, Any] = {}
        if use_grpo_buffer_guidance:
            grpo_buffer_guidance = self._load_grpo_buffer_guidance_batch(
                action_input,
                [str(token) for token in tokens_list],
                offline_cfg,
            )

        reward_kwargs: Dict[str, Any] = {}
        if offline_cfg is not None and bool(offline_cfg.enabled):
            reward_kwargs = {
                "strict_submetrics": bool(offline_cfg.strict_reward_submetrics),
                "required_submetrics": offline_cfg.required_reward_submetrics,
                "missing_submetric_policy": str(offline_cfg.missing_submetric_policy),
                "use_batched_pdm_scoring": bool(offline_cfg.use_batched_pdm_scoring),
                "use_exact_array_pdm_state_conversion": bool(offline_cfg.use_exact_array_pdm_state_conversion),
                "use_fast_pdm_scorer": bool(offline_cfg.use_fast_pdm_scorer),
                "pdm_batch_chunk_size": int(offline_cfg.pdm_batch_chunk_size),
                "pdm_shadow_check": bool(offline_cfg.pdm_shadow_check),
                "pdm_shadow_max_samples": int(offline_cfg.pdm_shadow_max_samples),
                "pdm_shadow_max_abs_diff": float(offline_cfg.pdm_shadow_max_abs_diff),
            }
        base_rewards, components = self.reward_fn(
            trajs,
            tokens_rep,
            metric_cache,
            return_components=True,
            **reward_kwargs,
        )
        # rewards: [B * G]
        assert base_rewards.shape == (B * G,)
        components = dict(components)
        components["pdms"] = base_rewards
        grpo_buffer_reward_bonus = base_rewards.new_zeros((B * G,))
        grpo_buffer_diag = {
            "grpo_buffer_reward_bonus_mean": base_rewards.new_zeros(()),
            "grpo_buffer_reward_bonus_max": base_rewards.new_zeros(()),
            "grpo_buffer_target_distance_mean": base_rewards.new_zeros(()),
            "grpo_buffer_guidance_target_ratio": base_rewards.new_zeros(()),
        }
        grpo_buffer_bonus_weight = 0.0
        use_core_pareto = bool(getattr(self, "use_core_pareto_grpo", False))
        use_feasible_pareto = (
            bool(getattr(self, "use_feasible_pareto_grpo", False))
            or str(getattr(self, "reward_mode", "safe_diffgrpo")) == "feasible_pareto"
        )
        if use_core_pareto:
            ref = self._compute_core_pareto_reference_components(
                vl_features,
                action_input,
                [str(token) for token in tokens_list],
                metric_cache,
            )
            components_matrix = self._reshape_components_for_group(components, B, G)
            trajs_matrix = trajs.reshape(B, G, trajs.shape[1], trajs.shape[2])
            rewards_matrix = base_rewards.view(B, G)
            advantages, group_weight, advantage_aux = self._compute_core_pareto_advantages(
                rewards_matrix,
                components_matrix,
                trajs_matrix,
                ref,
            )
            hard_safe_matrix = advantage_aux["core_pareto_valid_mask"].to(device=base_rewards.device).bool()
            hard_safe_mask = hard_safe_matrix.reshape(B * G)
            rewards_matrix = advantage_aux["core_pareto_score"].to(device=base_rewards.device, dtype=base_rewards.dtype)
            rewards = rewards_matrix.reshape(B * G)
            if self.log_safe_diversity or self.use_diversity_reward:
                diversity_bonus = self._compute_safe_diversity_bonus(trajs, hard_safe_mask, base_rewards, B, G)
            else:
                diversity_bonus = torch.zeros_like(base_rewards)
            ep_floor_gap = (
                ref["ref_ep"].to(device=base_rewards.device, dtype=base_rewards.dtype)[:, None]
                - components_matrix["ego_progress"].to(base_rewards)
                - float(self.core_pareto_ep_floor_tolerance)
            ).clamp(min=0.0)
            zero = torch.zeros_like(base_rewards)
            reward_aux = {
                "pdms": base_rewards,
                "ego_progress": components["ego_progress"],
                "diversity_bonus": diversity_bonus,
                "soft_safety_penalty": zero,
                "core_pareto_mode_enabled": base_rewards.new_tensor(1.0),
                "core_pareto_core": self._compute_pdms_core(components_matrix).reshape(B * G).to(base_rewards),
                "core_pareto_pdms_formula": base_rewards,
                "core_pareto_adjusted_core": rewards,
                "core_pareto_ep_floor_gap": ep_floor_gap.reshape(B * G),
                "core_pareto_ep_floor_penalty": (
                    float(self.core_pareto_slow_penalty_weight) * ep_floor_gap
                ).reshape(B * G),
                "core_pareto_ddc_penalty": zero,
                "core_pareto_dual_penalty": zero,
                "core_pareto_nc_dac_feasible_mask": (
                    (
                        components_matrix["no_at_fault_collisions"] >= 1.0
                    )
                    & (components_matrix["drivable_area_compliance"] >= 1.0)
                ).reshape(B * G).to(dtype=base_rewards.dtype),
                "core_pareto_ddc_guard_mask": (
                    (
                        components_matrix["driving_direction_compliance"] >= float(self.core_pareto_ddc_min_absolute)
                    )
                    | (
                        components_matrix["driving_direction_compliance"]
                        >= ref["ref_ddc"].to(device=base_rewards.device)[:, None]
                        - float(self.core_pareto_ddc_drop_tolerance)
                    )
                ).reshape(B * G).to(dtype=base_rewards.dtype),
            }
            for key in (
                "no_at_fault_collisions",
                "drivable_area_compliance",
                "time_to_collision_within_bound",
                "history_comfort",
                "lane_keeping",
                "driving_direction_compliance",
                "traffic_light_compliance",
            ):
                if key in components:
                    reward_aux[key] = components[key]
        elif use_feasible_pareto:
            ref = self._compute_core_pareto_reference_components(
                vl_features,
                action_input,
                [str(token) for token in tokens_list],
                metric_cache,
            )
            components_matrix = self._reshape_components_for_group(components, B, G)
            trajs_matrix = trajs.reshape(B, G, trajs.shape[1], trajs.shape[2])
            rewards_matrix = base_rewards.view(B, G)
            advantages, group_weight, advantage_aux = self._compute_feasible_pareto_advantages(
                rewards_matrix,
                components_matrix,
                trajs_matrix,
                ref,
                support_batch=grpo_buffer_guidance if use_grpo_buffer_guidance else None,
            )
            hard_safe_matrix = advantage_aux["feasible_pareto_valid_mask"].to(device=base_rewards.device).bool()
            hard_safe_mask = hard_safe_matrix.reshape(B * G)
            rewards_matrix = advantage_aux["feasible_pareto_score"].to(device=base_rewards.device, dtype=base_rewards.dtype)
            rewards = rewards_matrix.reshape(B * G)
            if self.log_safe_diversity or self.use_diversity_reward:
                diversity_bonus = self._compute_safe_diversity_bonus(trajs, hard_safe_mask, base_rewards, B, G)
            else:
                diversity_bonus = torch.zeros_like(base_rewards)
            zero = torch.zeros_like(base_rewards)
            reward_aux = {
                "pdms": base_rewards,
                "ego_progress": components["ego_progress"],
                "diversity_bonus": diversity_bonus,
                "soft_safety_penalty": zero,
                "core_pareto_mode_enabled": base_rewards.new_tensor(0.0),
                "core_pareto_core": zero,
                "core_pareto_pdms_formula": zero,
                "core_pareto_adjusted_core": zero,
                "core_pareto_ep_floor_gap": zero,
                "core_pareto_ep_floor_penalty": zero,
                "core_pareto_ddc_penalty": zero,
                "core_pareto_dual_penalty": zero,
                "core_pareto_nc_dac_feasible_mask": hard_safe_mask.to(dtype=base_rewards.dtype),
                "core_pareto_ddc_guard_mask": (
                    components_matrix["driving_direction_compliance"]
                    >= torch.maximum(
                        base_rewards.new_full((B, G), float(self.fp_ddc_min_absolute)),
                        ref["ref_ddc"].to(device=base_rewards.device)[:, None] - float(self.fp_ddc_ref_tolerance),
                    )
                ).reshape(B * G).to(dtype=base_rewards.dtype),
            }
            for key in (
                "no_at_fault_collisions",
                "drivable_area_compliance",
                "time_to_collision_within_bound",
                "history_comfort",
                "lane_keeping",
                "driving_direction_compliance",
                "traffic_light_compliance",
            ):
                if key in components:
                    reward_aux[key] = components[key]
        else:
            rewards, hard_safe_mask, reward_aux = self._compose_stage3_reward(
                base_rewards,
                components,
                trajs,
                B,
                G,
            )
            if use_grpo_buffer_guidance:
                grpo_buffer_reward_bonus, grpo_buffer_diag = self._compute_grpo_buffer_reward_bonus(
                    trajs,
                    B,
                    G,
                    grpo_buffer_guidance,
                    offline_cfg,
                )
                grpo_buffer_bonus_weight = float(offline_cfg.grpo_buffer_reward_bonus_weight)
                if grpo_buffer_bonus_weight > 0.0:
                    rewards = rewards + torch.where(
                        hard_safe_mask,
                        grpo_buffer_bonus_weight * grpo_buffer_reward_bonus.to(rewards),
                        torch.zeros_like(rewards),
                    )
            rewards_matrix = rewards.view(B, G)
            hard_safe_matrix = hard_safe_mask.view(B, G)
            advantages, group_weight, advantage_aux = self._compute_stage3_advantages(
                rewards_matrix,
                hard_safe_matrix,
                reward_aux,
            )
        assert rewards.shape == (B * G,)
        assert hard_safe_mask.shape == (B * G,)
        # advantages: [B * G], group_weight: [B]
        assert advantages.shape == (B * G,)
        assert group_weight.shape == (B,)

        num_denoising_steps = chains.shape[1] - 1
        denoising_indices = torch.arange(num_denoising_steps, device=advantages.device)
        discount = (float(self.gamma_denoising) ** (num_denoising_steps - denoising_indices - 1)).to(
            device=advantages.device,
            dtype=advantages.dtype,
        )

        adv = advantages * group_weight.repeat_interleave(G)
        adv_before = adv.detach().float()
        grpo_advantage_mean_before = adv_before.mean().to(adv)
        grpo_advantage_std_before = adv_before.std(unbiased=False).to(adv)
        grpo_advantage_clip_frac = adv.new_zeros(())
        if bool(getattr(self, "normalize_advantage_batch", False)):
            adv_mean = adv.mean()
            adv_std = adv.std(unbiased=False).clamp(min=1e-6)
            adv = (adv - adv_mean) / adv_std
        advantage_clip_abs = float(getattr(self, "advantage_clip_abs", 0.0))
        if advantage_clip_abs > 0.0:
            clip_mask = adv.detach().abs() > advantage_clip_abs
            grpo_advantage_clip_frac = clip_mask.float().mean().to(adv)
            adv = adv.clamp(min=-advantage_clip_abs, max=advantage_clip_abs)
        adv_after = adv.detach().float()
        grpo_advantage_mean_after = adv_after.mean().to(adv)
        grpo_advantage_std_after = adv_after.std(unbiased=False).to(adv)
        grpo_advantage_min = adv_after.min().to(adv)
        grpo_advantage_max = adv_after.max().to(adv)
        grpo_advantage_positive_ratio = (adv_after > 0.0).float().mean().to(adv)
        grpo_advantage_zero_ratio = (adv_after.abs() <= 1e-8).float().mean().to(adv)

        new_log_probs = self.get_logprobs(
            vl_features_rep,
            his_traj_rep,
            status_feature_rep,
            chains,
            deterministic=False,
            action_input=expert_action_input_rep,
        )
        # raw log_probs: [B * G * K, H, D]
        assert new_log_probs.shape[0] == B * G * num_denoising_steps

        gspo_ratio_mean = new_log_probs.new_tensor(1.0)
        gspo_ratio_clip_frac = new_log_probs.new_tensor(0.0)
        gspo_log_ratio_mean = new_log_probs.new_zeros(())
        gspo_log_ratio_std = new_log_probs.new_zeros(())
        gspo_log_ratio_min = new_log_probs.new_zeros(())
        gspo_log_ratio_max = new_log_probs.new_zeros(())
        gspo_abs_log_ratio_mean = new_log_probs.new_zeros(())
        gspo_ratio_min = new_log_probs.new_tensor(1.0)
        gspo_ratio_max = new_log_probs.new_tensor(1.0)
        gspo_approx_kl = new_log_probs.new_zeros(())
        gspo_reverse_approx_kl = new_log_probs.new_zeros(())

        use_strict_gspo = (
            self.use_trajectory_level_objective
            and self.use_gspo_ratio
            and sampled_from_behavior_policy
        )
        if self.use_trajectory_level_objective:
            new_traj_logp = self._reduce_chain_logprobs(new_log_probs, B, G, num_denoising_steps, discount)
            # traj_logp: [B * G]
            assert new_traj_logp.shape == (B * G,)

            if use_strict_gspo:
                with torch.no_grad():
                    old_log_probs = self.behavior_policy.get_logprobs(
                        vl_features_rep,
                        his_traj_rep,
                        status_feature_rep,
                        chains,
                        deterministic=False,
                        action_input=expert_action_input_rep,
                    )
                    old_traj_logp = self.behavior_policy._reduce_chain_logprobs(
                        old_log_probs,
                        B,
                        G,
                        num_denoising_steps,
                        discount,
                    )

                log_ratio = new_traj_logp - old_traj_logp
                ratio = torch.exp(log_ratio.clamp(min=-20.0, max=20.0))
                ratio_clip_low = 1.0 - float(self.gspo_clip_low)
                ratio_clip_high = 1.0 + float(self.gspo_clip_high)
                clipped_ratio = ratio.clamp(ratio_clip_low, ratio_clip_high)
                surrogate = torch.minimum(ratio * adv, clipped_ratio * adv)
                policy_loss = -torch.mean(surrogate)
                gspo_ratio_mean = ratio.detach().mean()
                gspo_ratio_clip_frac = (
                    ((ratio < ratio_clip_low) | (ratio > ratio_clip_high)).detach().float().mean().to(ratio)
                )
                gspo_ratio_min = ratio.detach().min()
                gspo_ratio_max = ratio.detach().max()
                gspo_approx_kl = (((ratio - 1.0) - log_ratio).detach().float().mean()).to(ratio)
                gspo_reverse_approx_kl = ((old_traj_logp - new_traj_logp).detach().float().mean()).to(ratio)
                log_ratio_detached = log_ratio.detach().float()
                gspo_log_ratio_mean = log_ratio_detached.mean().to(ratio)
                gspo_log_ratio_std = log_ratio_detached.std(unbiased=False).to(ratio)
                gspo_log_ratio_min = log_ratio_detached.min().to(ratio)
                gspo_log_ratio_max = log_ratio_detached.max().to(ratio)
                gspo_abs_log_ratio_mean = log_ratio_detached.abs().mean().to(ratio)
            else:
                policy_loss = -torch.mean(new_traj_logp * adv)
            trajectory_logp = new_traj_logp.detach()
        else:
            step_logp = new_log_probs.clamp(min=-5, max=2).mean(dim=[1, 2])
            adv_steps = adv.view(B, G, 1).expand(-1, -1, num_denoising_steps)
            discount_steps = discount.view(1, 1, num_denoising_steps).expand(B, G, num_denoising_steps)
            adv_weighted_flat = (adv_steps * discount_steps).reshape(-1)
            policy_loss = -torch.mean(step_logp * adv_weighted_flat)
            trajectory_logp = self._reduce_chain_logprobs(
                new_log_probs,
                B,
                G,
                num_denoising_steps,
                discount,
            ).detach()

        total_loss = policy_loss

        reference_kl_coeff = float(getattr(self, "reference_kl_coeff", 0.0))
        reference_kl_loss = policy_loss.new_zeros(())
        if reference_kl_coeff > 0.0:
            reference_kl_loss = self._chain_transition_reference_kl(
                vl_features_rep,
                his_traj_rep,
                status_feature_rep,
                chains,
                expert_action_input_rep,
                B,
                G,
                num_denoising_steps,
                discount,
            ).to(dtype=policy_loss.dtype)
            total_loss = total_loss + reference_kl_coeff * reference_kl_loss

        effective_bc_coeff = self._current_bc_coeff(bc_coeff)
        bc_loss = policy_loss.new_zeros(())
        if use_bc_loss:
            self.old_policy.eval()
            with torch.no_grad():
                teacher_chains, _ = self.old_policy.sample_chain(
                    vl_features,
                    action_input.his_traj,
                    action_input.status_feature,
                    deterministic=False,
                    action_input=action_input,
                )
            bc_logp = self.get_logprobs(
                vl_features,
                action_input.his_traj,
                action_input.status_feature,
                teacher_chains,
                deterministic=False,
                action_input=action_input,
            )
            bc_logp = bc_logp.clamp(min=-5, max=2)
            K_steps = chains.shape[1] - 1
            bc_logp = bc_logp.view(-1, K_steps, chains.shape[2], chains.shape[3]).mean(dim=[1,2,3])
            bc_loss = -bc_logp.mean()
            total_loss = total_loss + effective_bc_coeff * bc_loss

        grpo_buffer_distill_loss = total_loss.new_zeros(())
        grpo_buffer_distill_weight = 0.0
        grpo_buffer_distill_diag = {
            "per_sample_loss_mean": total_loss.new_zeros(()),
            "target_norm_mean": total_loss.new_zeros(()),
            "diffusion_timestep_mean": total_loss.new_zeros(()),
            "diffusion_timestep_min": total_loss.new_zeros(()),
            "diffusion_timestep_max": total_loss.new_zeros(()),
            "effective_weight_sum": total_loss.new_zeros(()),
            "zero_weight_ratio": total_loss.new_zeros(()),
            "zero_weight_batch": total_loss.new_zeros(()),
        }
        if use_grpo_buffer_guidance:
            grpo_buffer_distill_weight = self._current_grpo_buffer_distill_loss_weight(offline_cfg)
            if grpo_buffer_distill_weight > 0.0:
                distill_weights = (
                    grpo_buffer_guidance["target_weights"].to(device=total_loss.device, dtype=torch.float32)
                    * grpo_buffer_guidance["target_mask"].to(device=total_loss.device, dtype=torch.float32)
                )
                grpo_buffer_distill_loss, grpo_buffer_distill_diag = self._weighted_diffusion_loss_on_targets(
                    vl_features,
                    action_input,
                    grpo_buffer_guidance["target_trajs"].to(device=total_loss.device, dtype=action_input.action.dtype),
                    distill_weights,
                    timestep_sampling=str(offline_cfg.grpo_buffer_distill_timestep_sampling),
                )
                total_loss = total_loss + float(grpo_buffer_distill_weight) * grpo_buffer_distill_loss

        grpo_buffer_preference_dpo_loss = total_loss.new_zeros(())
        grpo_buffer_preference_dpo_weight = 0.0
        grpo_buffer_preference_targets: Dict[str, torch.Tensor] = {
            "target_ratio": total_loss.new_zeros(()),
            "il_loser_ratio": total_loss.new_zeros(()),
        }
        grpo_buffer_preference_dpo_diag = self._zero_diffusion_dpo_diag(total_loss.new_zeros(()))
        if use_grpo_buffer_guidance:
            grpo_buffer_preference_dpo_weight = self._current_grpo_buffer_preference_dpo_loss_weight(offline_cfg)
            if grpo_buffer_preference_dpo_weight > 0.0:
                grpo_buffer_preference_targets = self._build_grpo_buffer_preference_dpo_targets(
                    vl_features,
                    action_input,
                    grpo_buffer_guidance,
                    offline_cfg,
                )
                dpo_cfg = copy.copy(offline_cfg)
                dpo_cfg.preference_dpo_loss_weight = float(grpo_buffer_preference_dpo_weight)
                dpo_cfg.preference_dpo_timestep_sampling = str(
                    offline_cfg.grpo_buffer_preference_dpo_timestep_sampling
                )
                dpo_cfg.preference_dpo_beta = float(offline_cfg.grpo_buffer_preference_dpo_beta)
                dpo_cfg.preference_dpo_label_smoothing = float(
                    offline_cfg.grpo_buffer_preference_dpo_label_smoothing
                )
                dpo_cfg.preference_dpo_reference_free = bool(
                    offline_cfg.grpo_buffer_preference_dpo_reference_free
                )
                dpo_cfg.preference_dpo_pair_mode = str(offline_cfg.grpo_buffer_preference_dpo_pair_mode)
                dpo_cfg.preference_dpo_min_reward_gap = float(
                    offline_cfg.grpo_buffer_preference_dpo_min_reward_gap
                )
                dpo_cfg.preference_dpo_max_pairs_per_scene = int(
                    offline_cfg.grpo_buffer_preference_dpo_max_pairs_per_scene
                )
                dpo_cfg.preference_dpo_gap_weight_mode = str(
                    offline_cfg.grpo_buffer_preference_dpo_gap_weight_mode
                )
                dpo_cfg.preference_dpo_gap_weight_scale = float(
                    offline_cfg.grpo_buffer_preference_dpo_gap_weight_scale
                )
                dpo_cfg.preference_dpo_gap_weight_min = float(
                    offline_cfg.grpo_buffer_preference_dpo_gap_weight_min
                )
                dpo_cfg.preference_dpo_gap_weight_max = float(
                    offline_cfg.grpo_buffer_preference_dpo_gap_weight_max
                )
                grpo_buffer_preference_dpo_loss, grpo_buffer_preference_dpo_diag = (
                    self._compute_diffusion_dpo_preference_loss(
                        vl_features,
                        action_input,
                        grpo_buffer_preference_targets["target_trajs"].to(
                            device=total_loss.device,
                            dtype=action_input.action.dtype,
                        ),
                        grpo_buffer_preference_targets["ordering_rewards"].to(
                            device=total_loss.device,
                            dtype=torch.float32,
                        ),
                        grpo_buffer_preference_targets["valid_mask"].to(device=total_loss.device),
                        grpo_buffer_preference_targets["real_mask"].to(device=total_loss.device),
                        grpo_buffer_preference_targets["source_code"].to(device=total_loss.device),
                        dpo_cfg,
                    )
                )
                total_loss = (
                    total_loss
                    + float(grpo_buffer_preference_dpo_weight) * grpo_buffer_preference_dpo_loss
                )

        grpo_self_imitation_loss = total_loss.new_zeros(())
        grpo_self_imitation_weight = 0.0
        grpo_self_imitation_targets: Dict[str, Any] = {}
        grpo_self_imitation_diag = {
            "per_sample_loss_mean": total_loss.new_zeros(()),
            "target_norm_mean": total_loss.new_zeros(()),
            "diffusion_timestep_mean": total_loss.new_zeros(()),
            "diffusion_timestep_min": total_loss.new_zeros(()),
            "diffusion_timestep_max": total_loss.new_zeros(()),
            "effective_weight_sum": total_loss.new_zeros(()),
            "zero_weight_ratio": total_loss.new_zeros(()),
            "zero_weight_batch": total_loss.new_zeros(()),
        }
        if offline_cfg is not None and bool(offline_cfg.enabled):
            grpo_self_imitation_weight = self._current_grpo_self_imitation_loss_weight(offline_cfg)
            if grpo_self_imitation_weight > 0.0:
                grpo_self_imitation_targets = self._build_grpo_self_imitation_targets(
                    trajs,
                    base_rewards,
                    hard_safe_mask,
                    components,
                    B,
                    G,
                    grpo_buffer_guidance,
                    offline_cfg,
                )
                self_imitation_weights = (
                    grpo_self_imitation_targets["target_weights"].to(device=total_loss.device, dtype=torch.float32)
                    * grpo_self_imitation_targets["target_mask"].to(device=total_loss.device, dtype=torch.float32)
                )
                grpo_self_imitation_loss, grpo_self_imitation_diag = self._weighted_diffusion_loss_on_targets(
                    vl_features,
                    action_input,
                    grpo_self_imitation_targets["target_trajs"].to(
                        device=total_loss.device,
                        dtype=action_input.action.dtype,
                    ),
                    self_imitation_weights,
                    timestep_sampling=str(offline_cfg.grpo_self_imitation_timestep_sampling),
                )
                total_loss = total_loss + float(grpo_self_imitation_weight) * grpo_self_imitation_loss

        self.grpo_update_counter = int(getattr(self, "grpo_update_counter", 0)) + 1

        zero_loss = total_loss.new_zeros(())
        hard_safe_ratio = hard_safe_mask.detach().float().mean().to(dtype=total_loss.dtype)
        diversity_bonus = reward_aux["diversity_bonus"].detach()
        if bool(hard_safe_mask.detach().any().item()):
            safe_diversity = diversity_bonus[hard_safe_mask.detach()].mean().to(dtype=total_loss.dtype)
        else:
            safe_diversity = zero_loss

        return BatchFeature(data={
            "loss": total_loss,
            "diffusion_loss": zero_loss,
            "jepa_alignment_loss": zero_loss,
            "vggt_alignment_loss": zero_loss,
            "reward": rewards.mean(),
            "base_reward": base_rewards.mean(),
            "shaped_reward": rewards.mean(),
            "policy_loss": policy_loss,
            "bc_loss": bc_loss,
            "bc_coeff": total_loss.new_tensor(effective_bc_coeff),
            "reference_kl_loss": reference_kl_loss,
            "reference_kl_coeff": total_loss.new_tensor(reference_kl_coeff),
            "reference_kl_chunk_size": total_loss.new_tensor(float(getattr(self, "reference_kl_chunk_size", 0))),
            "grpo_buffer_guidance_enabled": total_loss.new_tensor(float(bool(use_grpo_buffer_guidance))),
            "grpo_buffer_reward_bonus": grpo_buffer_reward_bonus.mean().to(dtype=total_loss.dtype),
            "grpo_buffer_reward_bonus_weight": total_loss.new_tensor(float(grpo_buffer_bonus_weight)),
            "grpo_buffer_reward_bonus_max": grpo_buffer_diag["grpo_buffer_reward_bonus_max"].to(dtype=total_loss.dtype),
            "grpo_buffer_target_distance_mean": grpo_buffer_diag["grpo_buffer_target_distance_mean"].to(dtype=total_loss.dtype),
            "grpo_buffer_guidance_target_ratio": grpo_buffer_diag["grpo_buffer_guidance_target_ratio"].to(dtype=total_loss.dtype),
            "grpo_buffer_selected_valid_ratio": (
                grpo_buffer_guidance.get("selected_valid_ratio", total_loss.new_zeros(()))
                if use_grpo_buffer_guidance else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_buffer_best_valid_minus_gt": (
                grpo_buffer_guidance.get("best_valid_minus_gt_mean", total_loss.new_zeros(()))
                if use_grpo_buffer_guidance else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_buffer_best_valid_minus_il": (
                grpo_buffer_guidance.get("best_valid_minus_il_mean", total_loss.new_zeros(()))
                if use_grpo_buffer_guidance else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_buffer_distill_loss": grpo_buffer_distill_loss,
            "grpo_buffer_distill_weight": total_loss.new_tensor(float(grpo_buffer_distill_weight)),
            "grpo_buffer_distill_weight_sum": grpo_buffer_distill_diag["effective_weight_sum"].to(dtype=total_loss.dtype),
            "grpo_buffer_distill_zero_weight_batch": grpo_buffer_distill_diag["zero_weight_batch"].to(dtype=total_loss.dtype),
            "grpo_buffer_preference_dpo_loss": grpo_buffer_preference_dpo_loss,
            "grpo_buffer_preference_dpo_weight": total_loss.new_tensor(float(grpo_buffer_preference_dpo_weight)),
            "grpo_buffer_preference_dpo_target_ratio": (
                grpo_buffer_preference_targets.get("target_ratio", total_loss.new_zeros(()))
            ).to(dtype=total_loss.dtype),
            "grpo_buffer_preference_dpo_il_loser_ratio": (
                grpo_buffer_preference_targets.get("il_loser_ratio", total_loss.new_zeros(()))
            ).to(dtype=total_loss.dtype),
            "grpo_buffer_preference_dpo_pair_count": (
                grpo_buffer_preference_dpo_diag["preference_dpo_pair_count"].to(total_loss).detach()
            ),
            "grpo_buffer_preference_dpo_active_row_ratio": (
                grpo_buffer_preference_dpo_diag["preference_dpo_active_row_ratio"].to(total_loss).detach()
            ),
            "grpo_buffer_preference_dpo_reward_gap_mean": (
                grpo_buffer_preference_dpo_diag["preference_dpo_reward_gap_mean"].to(total_loss).detach()
            ),
            "grpo_buffer_preference_dpo_gap_weight_mean": (
                grpo_buffer_preference_dpo_diag["preference_dpo_gap_weight_mean"].to(total_loss).detach()
            ),
            "grpo_buffer_preference_dpo_logit_mean": (
                grpo_buffer_preference_dpo_diag["preference_dpo_logit_mean"].to(total_loss).detach()
            ),
            "grpo_buffer_preference_dpo_logit_abs_mean": (
                grpo_buffer_preference_dpo_diag["preference_dpo_logit_abs_mean"].to(total_loss).detach()
            ),
            "grpo_buffer_preference_dpo_implicit_accuracy": (
                grpo_buffer_preference_dpo_diag["preference_dpo_implicit_accuracy"].to(total_loss).detach()
            ),
            "grpo_buffer_preference_dpo_current_margin_mean": (
                grpo_buffer_preference_dpo_diag["preference_dpo_current_margin_mean"].to(total_loss).detach()
            ),
            "grpo_buffer_preference_dpo_reference_margin_mean": (
                grpo_buffer_preference_dpo_diag["preference_dpo_reference_margin_mean"].to(total_loss).detach()
            ),
            "grpo_buffer_preference_dpo_timestep_mean": (
                grpo_buffer_preference_dpo_diag["preference_dpo_timestep_mean"].to(total_loss).detach()
            ),
            "grpo_buffer_preference_dpo_timestep_min": (
                grpo_buffer_preference_dpo_diag["preference_dpo_timestep_min"].to(total_loss).detach()
            ),
            "grpo_buffer_preference_dpo_timestep_max": (
                grpo_buffer_preference_dpo_diag["preference_dpo_timestep_max"].to(total_loss).detach()
            ),
            "grpo_self_imitation_enabled": total_loss.new_tensor(
                float(
                    offline_cfg is not None
                    and bool(offline_cfg.enabled)
                    and float(offline_cfg.grpo_self_imitation_loss_weight) > 0.0
                )
            ),
            "grpo_self_imitation_loss": grpo_self_imitation_loss,
            "grpo_self_imitation_weight": total_loss.new_tensor(float(grpo_self_imitation_weight)),
            "grpo_self_imitation_candidate_ratio": (
                grpo_self_imitation_targets.get("candidate_ratio", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_safety_candidate_ratio": (
                grpo_self_imitation_targets.get("safety_candidate_ratio", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_nc_pass_ratio": (
                grpo_self_imitation_targets.get("nc_pass_ratio", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_dac_pass_ratio": (
                grpo_self_imitation_targets.get("dac_pass_ratio", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_ttc_pass_ratio": (
                grpo_self_imitation_targets.get("ttc_pass_ratio", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_ddc_pass_ratio": (
                grpo_self_imitation_targets.get("ddc_pass_ratio", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_target_ratio": (
                grpo_self_imitation_targets.get("target_ratio", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_pre_cap_target_ratio": (
                grpo_self_imitation_targets.get("pre_cap_target_ratio", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_target_scene_cap_ratio": (
                grpo_self_imitation_targets.get("target_scene_cap_ratio", total_loss.new_ones(()))
                if grpo_self_imitation_targets else total_loss.new_ones(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_target_scene_cap_active": (
                grpo_self_imitation_targets.get("target_scene_cap_active", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_target_reward_mean": (
                grpo_self_imitation_targets.get("target_reward_mean", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_target_reward_max": (
                grpo_self_imitation_targets.get("target_reward_max", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_target_margin_mean": (
                grpo_self_imitation_targets.get("target_margin_mean", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_baseline_mean": (
                grpo_self_imitation_targets.get("baseline_mean", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_baseline_from_buffer": (
                grpo_self_imitation_targets.get("baseline_from_buffer", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_target_nc_mean": (
                grpo_self_imitation_targets.get("target_nc_mean", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_target_dac_mean": (
                grpo_self_imitation_targets.get("target_dac_mean", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_target_ttc_mean": (
                grpo_self_imitation_targets.get("target_ttc_mean", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_target_ep_mean": (
                grpo_self_imitation_targets.get("target_ep_mean", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_target_comfort_mean": (
                grpo_self_imitation_targets.get("target_comfort_mean", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_target_ddc_mean": (
                grpo_self_imitation_targets.get("target_ddc_mean", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_target_tlc_mean": (
                grpo_self_imitation_targets.get("target_tlc_mean", total_loss.new_zeros(()))
                if grpo_self_imitation_targets else total_loss.new_zeros(())
            ).to(dtype=total_loss.dtype),
            "grpo_self_imitation_weight_sum": grpo_self_imitation_diag["effective_weight_sum"].to(dtype=total_loss.dtype),
            "grpo_self_imitation_zero_weight_batch": grpo_self_imitation_diag["zero_weight_batch"].to(dtype=total_loss.dtype),
            "grpo_self_imitation_timestep_mean": grpo_self_imitation_diag["diffusion_timestep_mean"].to(dtype=total_loss.dtype),
            "safe_ratio": hard_safe_ratio,
            "hard_safe_ratio": hard_safe_ratio,
            "mean_ep": reward_aux["ego_progress"].mean(),
            "mean_nc": reward_aux["no_at_fault_collisions"].mean(),
            "mean_dac": reward_aux["drivable_area_compliance"].mean(),
            "mean_ttc": reward_aux["time_to_collision_within_bound"].mean(),
            "mean_comfort": reward_aux["history_comfort"].mean(),
            "mean_ddc": reward_aux["driving_direction_compliance"].mean(),
            "mean_tlc": reward_aux["traffic_light_compliance"].mean(),
            "soft_safety_penalty_mean": reward_aux["soft_safety_penalty"].mean(),
            "soft_safety_penalty_max": reward_aux["soft_safety_penalty"].max(),
            "soft_safety_penalty_weight": total_loss.new_tensor(float(self.soft_safety_penalty_weight)),
            "soft_safety_mode_enabled": total_loss.new_tensor(
                float(str(self.safety_advantage_mode) == "soft_penalty")
            ),
            "nc_pass_ratio": (
                reward_aux["no_at_fault_collisions"] >= float(self.nc_safe_threshold)
            ).detach().float().mean().to(dtype=total_loss.dtype),
            "dac_pass_ratio": (
                reward_aux["drivable_area_compliance"] >= float(self.dac_safe_threshold)
            ).detach().float().mean().to(dtype=total_loss.dtype),
            "ttc_pass_ratio": (
                reward_aux["time_to_collision_within_bound"] >= float(self.ttc_safe_threshold)
            ).detach().float().mean().to(dtype=total_loss.dtype),
            "ddc_pass_ratio": (
                reward_aux["driving_direction_compliance"] >= float(self.ddc_safe_threshold)
            ).detach().float().mean().to(dtype=total_loss.dtype),
            "tlc_pass_ratio": (
                reward_aux["traffic_light_compliance"] >= float(self.tlc_safe_threshold)
            ).detach().float().mean().to(dtype=total_loss.dtype),
            "diversity_bonus": diversity_bonus.mean(),
            "safe_diversity": safe_diversity,
            "group_reward_std": advantage_aux["reward_std"].mean(),
            "safe_count_mean": advantage_aux["safe_count"].mean(),
            "group_weight_mean": group_weight.mean(),
            "group_weight_min": group_weight.min(),
            "group_weight_max": group_weight.max(),
            "mixed_group_ratio": advantage_aux["mixed_group_ratio"],
            "all_safe_group_ratio": advantage_aux["all_safe_group_ratio"],
            "all_unsafe_group_ratio": advantage_aux["all_unsafe_group_ratio"],
            "safe_rpp_advantage_enabled": advantage_aux.get(
                "safe_rpp_advantage_enabled",
                total_loss.new_tensor(0.0),
            ),
            "safe_rpp_centered_batch_std": advantage_aux.get(
                "safe_rpp_centered_batch_std",
                total_loss.new_tensor(0.0),
            ),
            "mean_advantage": advantages.mean(),
            "mean_abs_advantage": advantages.abs().mean(),
            "grpo_advantage_mean_before_transform": grpo_advantage_mean_before.to(dtype=total_loss.dtype),
            "grpo_advantage_std_before_transform": grpo_advantage_std_before.to(dtype=total_loss.dtype),
            "grpo_advantage_mean_after_transform": grpo_advantage_mean_after.to(dtype=total_loss.dtype),
            "grpo_advantage_std_after_transform": grpo_advantage_std_after.to(dtype=total_loss.dtype),
            "grpo_advantage_min": grpo_advantage_min.to(dtype=total_loss.dtype),
            "grpo_advantage_max": grpo_advantage_max.to(dtype=total_loss.dtype),
            "grpo_advantage_positive_ratio": grpo_advantage_positive_ratio.to(dtype=total_loss.dtype),
            "grpo_advantage_zero_ratio": grpo_advantage_zero_ratio.to(dtype=total_loss.dtype),
            "grpo_advantage_clip_frac": grpo_advantage_clip_frac.to(dtype=total_loss.dtype),
            "grpo_advantage_batch_normalized": total_loss.new_tensor(
                float(bool(getattr(self, "normalize_advantage_batch", False)))
            ),
            "grpo_advantage_clip_abs": total_loss.new_tensor(float(advantage_clip_abs)),
            **self._stage3_core_pareto_log_metrics(total_loss, reward_aux, advantage_aux),
            **self._stage3_feasible_pareto_log_metrics(total_loss, advantage_aux),
            "trajectory_logp": trajectory_logp.mean(),
            "gspo_ratio_mean": gspo_ratio_mean.to(dtype=total_loss.dtype),
            "gspo_ratio_min": gspo_ratio_min.to(dtype=total_loss.dtype),
            "gspo_ratio_max": gspo_ratio_max.to(dtype=total_loss.dtype),
            "gspo_ratio_clip_frac": gspo_ratio_clip_frac.to(dtype=total_loss.dtype),
            "gspo_log_ratio_mean": gspo_log_ratio_mean.to(dtype=total_loss.dtype),
            "gspo_log_ratio_std": gspo_log_ratio_std.to(dtype=total_loss.dtype),
            "gspo_log_ratio_min": gspo_log_ratio_min.to(dtype=total_loss.dtype),
            "gspo_log_ratio_max": gspo_log_ratio_max.to(dtype=total_loss.dtype),
            "gspo_abs_log_ratio_mean": gspo_abs_log_ratio_mean.to(dtype=total_loss.dtype),
            "gspo_approx_kl": gspo_approx_kl.to(dtype=total_loss.dtype),
            "gspo_reverse_approx_kl": gspo_reverse_approx_kl.to(dtype=total_loss.dtype),
            "use_gspo_ratio": total_loss.new_tensor(float(bool(self.use_gspo_ratio))),
            "sampled_from_behavior_policy": total_loss.new_tensor(float(bool(sampled_from_behavior_policy))),
            "behavior_policy_synced": total_loss.new_tensor(float(bool(behavior_policy_synced))),
            "behavior_policy_sync_interval": total_loss.new_tensor(float(getattr(self, "behavior_policy_sync_interval", 0))),
            "gspo_clip_low": total_loss.new_tensor(float(self.gspo_clip_low)),
            "gspo_clip_high": total_loss.new_tensor(float(self.gspo_clip_high)),
        })
