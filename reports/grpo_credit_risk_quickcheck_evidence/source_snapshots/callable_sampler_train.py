    def sample_chain(
        self,
        vl_features: torch.Tensor,
        his_traj_features: torch.Tensor,
        ego_status_features: torch.Tensor,
        init_actions: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        action_input: Optional[BatchFeature] = None,
        allow_target_tokens: Optional[bool] = None,
        prepared_dit_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Generates the full denoising chain and the final trajectory.
        This method reuses the logic from get_action but stores intermediate steps.

        Args:
            vl_features (torch.Tensor): Vision-language features from the backbone.
            his_traj_features (torch.Tensor): Encoded historical trajectory features.
            ego_status_features (torch.Tensor): Encoded ego status features.
            init_actions (Optional[torch.Tensor]): An initial trajectory to start from.
                If None, starts from pure noise.
            deterministic (bool): If True, DDIM sampling will be deterministic.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The full denoising chain as a tensor of shape (B, K+1, H, D).
                - The final, denormalized trajectory of shape (B, H, D).
        """
        context_allow_target_tokens = self.training if allow_target_tokens is None else bool(allow_target_tokens)
        if not context_allow_target_tokens:
            self._warn_if_expert_targets_present(action_input, "sample_chain")
        if prepared_dit_context is None:
            self._reset_planning_adapter_forward_count()
            dit_context = self._prepare_dit_context(
                vl_features,
                action_input,
                training=self.training,
                allow_target_tokens=context_allow_target_tokens,
            )
        else:
            dit_context = prepared_dit_context
        context_embeds = dit_context["context_tokens"]
        context_mean = dit_context["context_mean"]
        expert_step_condition = dit_context["expert_step_condition"]
        planning_condition_tokens = dit_context.get("planning_condition_tokens")
        cot_condition_tokens = dit_context.get("cot_condition_tokens")
        B, D = context_embeds.shape[0], self.config.action_dim
        device, dtype = context_embeds.device, context_embeds.dtype
        
        his_traj_features = self.his_traj_encoder(
            his_traj_features.unsqueeze(1)               
        ).repeat(1, self.config.action_horizon, 1) 

        ego_status_features = self.ego_status_encoder(
            ego_status_features       
        )

        current_actions = init_actions if init_actions is not None else torch.randn(
            (B, self.config.action_horizon, D), device=device, dtype=dtype
        )
        denoising_chain = [current_actions.clone()]
        lfp_on_policy = self.stage3_algorithm == "lfp_grpo" and not deterministic
        bounded_final_norm: Optional[torch.Tensor] = None

        if self.config.sampling_method == 'flow':
            dt = 1.0 / self.config.num_inference_steps
            for step in range(self.config.num_inference_steps):
                idx = int(step / self.config.num_inference_steps * self.config.flow_cfg.num_timestep_buckets)
                t_batch = torch.full((B,), idx, device=device, dtype=torch.long)
                if (
                    self.config.use_last_rd
                    or self.config.use_last_vla
                    or self.config.use_two_expert_slots
                ) and action_input is not None:
                    dit_context = self._prepare_dit_context(
                        vl_features,
                        action_input,
                        training=self.training,
                        noisy_actions=current_actions,
                        diffusion_timestep=t_batch,
                        allow_target_tokens=context_allow_target_tokens,
                        cached_planning_condition_tokens=(
                            planning_condition_tokens if self.planning_adapter is not None else None
                        ),
                    )
                    context_embeds = dit_context["context_tokens"]
                    context_mean = dit_context["context_mean"]
                    expert_step_condition = dit_context["expert_step_condition"]
                    planning_condition_tokens = dit_context.get("planning_condition_tokens")
                    cot_condition_tokens = dit_context.get("cot_condition_tokens")
                
                action_features = self.action_encoder(current_actions, t_batch)
                if hasattr(self, 'position_embedding'):
                    action_features += self.position_embedding(torch.arange(self.config.action_horizon, device=device))
                
                context_mean_features = context_mean.unsqueeze(1).repeat(1, self.config.action_horizon, 1)
                fused_input = self.fusion_projector(
                    torch.cat((his_traj_features, context_mean_features, action_features), dim=2)
                )
                if expert_step_condition is not None:
                    fused_input = self._apply_two_expert_denoise_condition(
                        fused_input,
                        action_features,
                        t_batch,
                        dit_context,
                    )
                
                model_output = self.model(
                    fused_input,
                    context_embeds,
                    ego_status_features,
                    t_batch,
                    cot_condition_tokens=cot_condition_tokens,
                    planning_condition_tokens=planning_condition_tokens,
                    planning_condition_layers=self.config.planning_condition_layers,
                    cot_condition_layers=self.config.last_vla_cot_condition_layers,
                )
                pred = self.action_decoder(model_output)
                
                pred_flow = pred.chunk(2, dim=-1)[0] if self.config.flow_cfg.mean_variance_net else pred
                current_actions = current_actions + dt * pred_flow
                denoising_chain.append(current_actions.clone())

        elif self.config.sampling_method in ['ddpm', 'ddim']:
            if self.config.sampling_method == 'ddpm':
                step_size = self.config.ddpm_cfg.num_train_timesteps // self.config.num_inference_steps
                timesteps = list(reversed(range(0, self.config.ddpm_cfg.num_train_timesteps, step_size)))
            else: 
                timesteps = self.ddim_t
            
            for i, t_int in enumerate(timesteps):
                t_batch = self.make_timesteps(B, t_int, device)
                index_batch = self.make_timesteps(B, i, device) if self.config.sampling_method == 'ddim' else t_batch

                mean, logvar, _ = self.p_mean_variance(
                    current_actions, t_batch, index_batch, context_embeds, his_traj_features, ego_status_features, deterministic,
                    context_mean=context_mean,
                    expert_step_condition=expert_step_condition,
                    planning_condition_tokens=planning_condition_tokens,
                    cot_condition_tokens=cot_condition_tokens,
                    vl_features=vl_features if action_input is not None else None,
                    action_input=action_input,
                )

                std = torch.exp(0.5 * logvar).to(dtype)
                noise_sample = torch.randn_like(current_actions)

                if self.config.sampling_method == 'ddim':
                    if deterministic:
                        std.zero_()
                    else:
                        std = std.clamp(min=self.min_sampling_denoising_std)
                else: # ddpm
                    if deterministic and t_int == 0:
                        std = torch.zeros_like(std)
                    elif deterministic:
                        std = std.clamp(min=1e-3)
                    else:
                        std = std.clamp(min=self.min_sampling_denoising_std)
                
                if (
                    not lfp_on_policy
                    and hasattr(self, 'randn_clip_value')
                    and self.randn_clip_value is not None
                ):
                    noise_sample = noise_sample.clamp_(-self.randn_clip_value, self.randn_clip_value)

                current_actions = mean + std * noise_sample
                
                if i == len(timesteps) - 1:
                    bounded_output = self._clip_output_representation(
                        current_actions,
                        getattr(self, "final_action_clip_value", None),
                    )
                    if lfp_on_policy:
                        # The scored action remains bounded, but the sampled transition stored
                        # in the chain must stay Gaussian for exact on-policy log-prob and KL.
                        bounded_final_norm = bounded_output
                    else:
                        current_actions = bounded_output
                
                denoising_chain.append(current_actions.clone())
        else:
            raise ValueError(f"Unsupported sampling method: {self.config.sampling_method}")

        residual_alpha = self._last_vla_residual_alpha(training=self.training)
        final_norm = bounded_final_norm if bounded_final_norm is not None else current_actions
        if residual_alpha != 0.0:
            residual_anchor_norm, _ = self._last_vla_residual_anchor_norm(
                action_input,
                current_actions,
                required=True,
            )
            assert residual_anchor_norm is not None
            final_norm = current_actions + float(residual_alpha) * residual_anchor_norm.to(current_actions)
            final_norm = self._clip_output_representation(
                final_norm,
                getattr(self, "final_action_clip_value", 1.0),
            )
        final_actions = self._decode_action_target(final_norm)
        chain_tensor = torch.stack(denoising_chain, dim=1)
        
        return chain_tensor.detach(), final_actions.detach()
