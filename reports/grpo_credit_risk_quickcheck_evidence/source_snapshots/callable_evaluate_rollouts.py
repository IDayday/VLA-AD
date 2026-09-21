    def _evaluate_lfp_rollouts(
        self,
        trajectories: torch.Tensor,
        tokens_rep: list[str],
        metric_cache: Dict[str, Any],
        B: int,
        G: int,
    ):
        if self.lfp_metric_adapter is None:
            raise RuntimeError("LFP metric adapter is not initialized.")
        if self.lfp_grpo_cfg.benchmark == "navsim_v2":
            evaluator = self.lfp_v2_rollout_evaluator
            if evaluator is None:
                raise RuntimeError(
                    "NAVSIM v2 LFP training requires an official one-stage EPDMS rollout evaluator. "
                    "The local NAVSIM v1 pdm_score backend cannot supply two_frame_extended_comfort/TLC "
                    "and must not be used as a silent approximation."
                )
            if self.lfp_reference_cache is None:
                raise RuntimeError("NAVSIM v2 LFP scoring requires the coherent reference cache.")
            components = evaluator.score(trajectories, tokens_rep, self.lfp_reference_cache)
            if not isinstance(components, dict):
                raise TypeError("lfp_v2_rollout_evaluator must return a metric component dict.")
        else:
            _, components = self.reward_fn(
                trajectories,
                tokens_rep,
                metric_cache,
                return_components=True,
                strict_submetrics=True,
                required_submetrics=(
                    "pdms",
                    "no_at_fault_collisions",
                    "drivable_area_compliance",
                    "time_to_collision_within_bound",
                    "ego_progress",
                    "history_comfort",
                    "driving_direction_compliance",
                ),
                missing_submetric_policy="error",
                use_batched_pdm_scoring=True,
                use_exact_array_pdm_state_conversion=True,
                use_fast_pdm_scorer=True,
            )
        return self.lfp_metric_adapter.canonicalize(
            components,
            batch_size=B,
            group_size=G,
        )
