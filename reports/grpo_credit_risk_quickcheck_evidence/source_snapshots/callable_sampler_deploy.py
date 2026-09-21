    def get_action(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        init_actions: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> BatchFeature:
        """
        Generates action trajectories via the configured sampling method.

        This method strictly preserves the original logic for each sampler,
        including specific clipping and noise handling for DDPM and DDIM.

        Args:
            vl_features (torch.Tensor): Vision-language features from the backbone.
            action_input (BatchFeature): Input containing conditioning features like
                historical trajectory and ego status.
            init_actions (Optional[torch.Tensor]): An initial trajectory to start
                the denoising from. If None, starts from pure noise.
            deterministic (bool): If True, DDIM sampling will be deterministic (eta=0).

        Returns:
            BatchFeature: A batch containing the final predicted trajectory.
        """
        self._reset_planning_adapter_forward_count()
        self._warn_if_expert_targets_present(action_input, "get_action")
        dit_context = self._prepare_dit_context(vl_features, action_input, training=False, allow_target_tokens=False)
        context_embeds = dit_context["context_tokens"]
        context_mean = dit_context["context_mean"]
        expert_step_condition = dit_context["expert_step_condition"]
        planning_condition_tokens = dit_context.get("planning_condition_tokens")
        cot_condition_tokens = dit_context.get("cot_condition_tokens")
        coarse_prior_norm = None
        if self.config.use_last_vla:
            coarse_prior_norm = dit_context["last_vla_output"].coarse_traj_norm.detach()
        
        history_embeds = self.his_traj_encoder(
            action_input.his_traj.unsqueeze(1)
        ).repeat(1, self.config.action_horizon, 1)

        ego_embeds = self.ego_status_encoder(
            action_input.status_feature
        )

        B, D = context_embeds.shape[0], self.config.action_dim
        device, dtype = context_embeds.device, context_embeds.dtype
        
        current_actions = init_actions if init_actions is not None else torch.randn(
            (B, self.config.action_horizon, D), device=device, dtype=dtype
        )

        if self.config.sampling_method == 'flow':
            dt = 1.0 / self.config.num_inference_steps
            for step in range(self.config.num_inference_steps):
                idx = int(step / self.config.num_inference_steps * self.config.flow_cfg.num_timestep_buckets)
                t = torch.full((B,), idx, device=device, dtype=torch.long)
                if self.config.use_last_rd or self.config.use_last_vla or self.config.use_two_expert_slots:
                    dit_context = self._prepare_dit_context(
                        vl_features,
                        action_input,
                        training=False,
                        noisy_actions=current_actions,
                        diffusion_timestep=t,
                        allow_target_tokens=False,
                        cached_planning_condition_tokens=(
                            planning_condition_tokens if self.planning_adapter is not None else None
                        ),
                    )
                    context_embeds = dit_context["context_tokens"]
                    context_mean = dit_context["context_mean"]
                    expert_step_condition = dit_context["expert_step_condition"]
                    planning_condition_tokens = dit_context.get("planning_condition_tokens")
                    cot_condition_tokens = dit_context.get("cot_condition_tokens")

                action_features = self.action_encoder(current_actions, t)
                if hasattr(self, 'position_embedding'):
                    action_features += self.position_embedding(torch.arange(self.config.action_horizon, device=device))
                
                context_mean_features = context_mean.unsqueeze(1).repeat(1, self.config.action_horizon, 1)
                fused_input = self.fusion_projector(
                    torch.cat((history_embeds, context_mean_features, action_features), dim=2)
                )
                if expert_step_condition is not None:
                    fused_input = self._apply_two_expert_denoise_condition(
                        fused_input,
                        action_features,
                        t,
                        dit_context,
                    )
                
                model_output = self.model(
                    fused_input,
                    context_embeds,
                    ego_embeds,
                    t,
                    cot_condition_tokens=cot_condition_tokens,
                    planning_condition_tokens=planning_condition_tokens,
                    planning_condition_layers=self.config.planning_condition_layers,
                    cot_condition_layers=self.config.last_vla_cot_condition_layers,
                )
                pred = self.action_decoder(model_output)
                
                pred_flow = pred.chunk(2, dim=-1)[0] if self.config.flow_cfg.mean_variance_net else pred
                current_actions = current_actions + dt * pred_flow

        elif self.config.sampling_method == 'ddpm':
            step_size = self.config.ddpm_cfg.num_train_timesteps // self.config.num_inference_steps
            timesteps_to_iterate = list(reversed(range(0, self.config.ddpm_cfg.num_train_timesteps, step_size)))
            
            for i, t_int in enumerate(timesteps_to_iterate):
                t_batch = self.make_timesteps(B, t_int, device)
                index_batch = self.make_timesteps(B, i, device)

                mean, logvar, _ = self.p_mean_variance(
                    current_actions, t_batch, index_batch, context_embeds, history_embeds, ego_embeds, deterministic,
                    context_mean=context_mean,
                    expert_step_condition=expert_step_condition,
                    planning_condition_tokens=planning_condition_tokens,
                    cot_condition_tokens=cot_condition_tokens,
                    vl_features=vl_features,
                    action_input=action_input,
                )

                noise_sample = torch.randn_like(current_actions)
                std = torch.exp(0.5 * logvar)

                std = std.to(dtype)

                if t_int == 0:
                    std.zero_()
                else:
                    std = torch.clamp(std, min=1e-3)

                if hasattr(self, 'eval_randn_clip_value') and self.eval_randn_clip_value is not None:
                    noise_sample.clamp_(-self.eval_randn_clip_value, self.eval_randn_clip_value)

                current_actions = mean + std * noise_sample
                
                if i == len(timesteps_to_iterate) - 1:
                    current_actions = self._clip_output_representation(
                        current_actions,
                        getattr(self, "final_action_clip_value", None),
                    )

        elif self.config.sampling_method == 'ddim':
            eval_min_sampling_denoising_std = getattr(self, 'eval_min_sampling_denoising_std', 0.0001)
            eval_randn_clip_value = getattr(self, 'eval_randn_clip_value', 1.0)
            for i in range(self.ddim_steps):
                t_batch = self.make_timesteps(B, self.ddim_t[i], device)
                index_batch = self.make_timesteps(B, i, device)

                mean, logvar, _ = self.p_mean_variance(
                    current_actions, t_batch, index_batch, context_embeds, history_embeds, ego_embeds, deterministic,
                    context_mean=context_mean,
                    expert_step_condition=expert_step_condition,
                    planning_condition_tokens=planning_condition_tokens,
                    cot_condition_tokens=cot_condition_tokens,
                    vl_features=vl_features,
                    action_input=action_input,
                )

                std = torch.exp(0.5 * logvar)

                std = std.to(dtype)

                noise_sample = torch.randn_like(current_actions)
                
                if deterministic:
                    std.zero_()
                else:
                    std = std.clamp(min=eval_min_sampling_denoising_std)
                
                noise_sample.clamp_(-eval_randn_clip_value, eval_randn_clip_value)
                
                current_actions = mean + std * noise_sample
        else:
            raise ValueError(f"Unsupported sampling method: {self.config.sampling_method}")

        current_actions = self._clip_output_representation(
            current_actions,
            getattr(self, "final_action_clip_value", 1.0),
        )

        residual_alpha = self._last_vla_residual_alpha(training=False)
        residual_anchor_norm = None
        final_norm = current_actions
        if residual_alpha != 0.0:
            residual_anchor_norm, _ = self._last_vla_residual_anchor_norm(
                action_input,
                current_actions,
                required=True,
            )
            assert residual_anchor_norm is not None
            final_norm = current_actions + float(residual_alpha) * residual_anchor_norm.to(current_actions)

        output_data: Dict[str, torch.Tensor] = {}
        if coarse_prior_norm is not None:
            output_data["pred_coarse_traj"] = self._decode_action_target(coarse_prior_norm.to(current_actions))
        if residual_anchor_norm is not None:
            output_data["pred_residual_norm"] = current_actions
            output_data["pred_vlm_text_anchor_traj"] = self._decode_action_target(residual_anchor_norm.to(current_actions))
            output_data["pred_residual_anchor_alpha"] = current_actions.new_tensor(float(residual_alpha))

        final_norm = self._clip_output_representation(
            final_norm,
            getattr(self, "final_action_clip_value", 1.0),
        )

        final_actions = self._decode_action_target(final_norm)
        output_data["pred_traj"] = final_actions

        return BatchFeature(data=output_data)
