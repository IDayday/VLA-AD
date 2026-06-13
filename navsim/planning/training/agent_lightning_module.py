import pytorch_lightning as pl

from torch import Tensor
from typing import Dict, Tuple,Any
import torch

from navsim.agents.abstract_agent import AbstractAgent


_CHECKPOINT_EXCLUDED_PREFIXES = (
    "agent.model",
    "agent.action_head.old_policy",
)


def _filter_recogdrive_checkpoint_state_dict(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Skip frozen backbone/reference weights before Lightning serializes checkpoints."""

    def keep_key(key: str) -> bool:
        return not any(key == prefix or key.startswith(f"{prefix}.") for prefix in _CHECKPOINT_EXCLUDED_PREFIXES)

    return {key: value for key, value in state_dict.items() if keep_key(key)}


def _prediction_get(prediction: Any, key: str) -> Any:
    if isinstance(prediction, dict) and key in prediction:
        return prediction[key]
    return getattr(prediction, key, None)


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent

    def _log_optional_recogdrive_metrics(self, prediction: Any, logging_prefix: str) -> None:
        for key in (
            "diffusion_loss",
            "jepa_alignment_loss",
            "vggt_alignment_loss",
            "future_jepa_loss",
            "vggt_geometry_loss",
            "coarse_traj_loss",
            "coarse_heading_loss",
            "risk_loss",
            "policy_kd_loss",
            "jepa_gate_value",
            "vggt_gate_value",
            "branch_weight_vlm",
            "branch_weight_jepa",
            "branch_weight_vggt",
            "last_rd_group_weight_vlm",
            "last_rd_group_weight_dynamic",
            "last_rd_group_weight_geometry",
            "last_rd_group_weight_ego",
            "last_rd_group_weight_risk",
            "last_rd_token_norms",
            "coarse_traj_l1",
            "future_jepa_loss_raw",
            "last_rd_geometry_mode_code",
            "last_vla_geometry_loss",
            "last_vla_dynamic_loss",
            "last_vla_coarse_loss",
            "last_vla_heading_loss",
            "last_vla_progress_loss",
            "last_vla_risk_loss",
            "last_vla_cot_consistency_loss",
            "last_vla_policy_kd_loss",
            "last_vla_coarse_traj_l1",
            "last_vla_dynamic_loss_raw",
            "last_vla_geometry_loss_raw",
            "last_vla_geometry_weight_effective",
            "last_vla_dynamic_weight_effective",
            "last_vla_coarse_weight_effective",
            "last_vla_progress_weight_effective",
            "hidden_anchor_loss",
            "hidden_anchor_loss_weighted",
            "hidden_drift_cosine",
            "hidden_drift_l2",
            "hidden_anchor_computed",
            "hidden_anchor_every_n_steps_tensor",
            "hidden_anchor_step_index",
            "lora_trainable_param_count",
            "lora_matched_module_count",
            "base_reward",
            "shaped_reward",
            "safe_ratio",
            "hard_safe_ratio",
            "mean_ep",
            "mean_ttc",
            "mean_comfort",
            "diversity_bonus",
            "safe_diversity",
            "group_reward_std",
            "mixed_group_ratio",
            "all_safe_group_ratio",
            "all_unsafe_group_ratio",
            "mean_advantage",
            "mean_abs_advantage",
            "trajectory_logp",
            "gspo_ratio_mean",
            "gspo_ratio_clip_frac",
            "bc_coeff",
            "awac_loss",
            "bc_loss_weight_effective",
            "grpo_loss",
            "reward_mean",
            "reward_max",
            "gt_reward_mean",
            "il_reward_mean",
            "best_raw_reward_mean",
            "best_valid_reward_mean",
            "best_selected_reward_mean",
            "best_raw_minus_gt_mean",
            "best_valid_minus_gt_mean",
            "best_selected_minus_gt_mean",
            "best_raw_minus_il_mean",
            "best_valid_minus_il_mean",
            "best_selected_minus_il_mean",
            "best_reward_mean",
            "best_minus_gt_mean",
            "best_minus_il_mean",
            "pct_best_raw_above_gt",
            "pct_best_valid_above_gt",
            "pct_best_selected_above_gt",
            "pct_best_valid_above_il",
            "pct_best_above_gt",
            "pct_best_above_il",
            "pct_candidates_above_gt",
            "pct_candidates_above_il",
            "selected_reward_mean",
            "selected_reward_max",
            "awac_weight_mean",
            "awac_weight_max",
            "awac_advantage_mean",
            "awac_advantage_max",
            "awac_pairwise_rank_loss",
            "awac_invalid_repulsion_loss",
            "awac_dpo_loss",
            "shaped_reward_mean",
            "shaped_reward_delta_mean",
            "shaped_reward_delta_min",
            "shaped_reward_delta_max",
            "component_progress_bonus_mean",
            "component_safety_penalty_mean",
            "component_valid_reward_delta_mean",
            "source_balance_factor_mean",
            "source_balance_factor_min",
            "source_balance_factor_max",
            "source_balance_active_sources",
            "target_blend_alpha_effective",
            "target_blend_l2_to_original",
            "target_blend_anchor_gt_ratio",
            "target_blend_anchor_il_ratio",
            "target_blend_anchor_fallback_gt_ratio",
            "pairwise_rank_pair_count",
            "invalid_repulsion_pair_count",
            "pairwise_rank_active_row_ratio",
            "invalid_repulsion_active_row_ratio",
            "preference_dpo_pair_count",
            "preference_dpo_active_row_ratio",
            "preference_dpo_reward_gap_mean",
            "preference_dpo_gap_weight_mean",
            "preference_dpo_current_logratio_mean",
            "preference_dpo_reference_logratio_mean",
            "preference_dpo_timestep_mean",
            "preference_dpo_timestep_min",
            "preference_dpo_timestep_max",
            "empty_awac_row_ratio",
            "positive_weight_row_ratio",
            "positive_weight_candidate_ratio",
            "target_filter_row_ratio",
            "target_filter_candidate_ratio",
            "has_valid_candidate_ratio",
            "valid_candidate_ratio",
            "selected_valid_ratio",
            "fallback_candidate_ratio",
            "selected_nc_mean",
            "selected_dac_mean",
            "selected_ttc_mean",
            "selected_ep_mean",
            "selected_comfort_mean",
            "selected_ddc_mean",
            "selected_tlc_mean",
            "source_gt_ratio",
            "source_il_ratio",
            "source_policy_ratio",
            "source_progress_ratio",
            "source_lateral_ratio",
            "source_timing_ratio",
            "awac_per_sample_loss_mean",
            "awac_target_norm_mean",
            "awac_timestep_mean",
            "awac_timestep_min",
            "awac_timestep_max",
            "awac_effective_weight_sum",
            "awac_zero_weight_ratio",
            "awac_zero_weight_batch",
        ):
            value = _prediction_get(prediction, key)
            if value is not None:
                self.log(
                    f"{logging_prefix}/{key}",
                    value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )

        action_head = getattr(self.agent, "action_head", None)
        if action_head is None:
            return
        if getattr(action_head.config, "use_expert_features", False) and getattr(action_head.config, "use_expert_gates", False):
            for stream in ("jepa", "vggt"):
                gate = getattr(action_head, f"{stream}_gate", None)
                if gate is not None:
                    self.log(
                        f"{logging_prefix}/{stream}_gate",
                        torch.sigmoid(gate.detach()),
                        on_step=True,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                    )

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets, tokens_list = batch
        prediction = self.agent.forward(features,targets,tokens_list)
        #prediction = self.agent.forward(features,targets)
        loss = self.agent.compute_loss(features, targets, prediction)
        self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{logging_prefix}/total_loss", loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self._log_optional_recogdrive_metrics(prediction, logging_prefix)
        
        return loss
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        每次保存 checkpoint 时，只保留 state_dict 中不以 'agent.model' 开头的条目。
        """
        checkpoint["state_dict"] = _filter_recogdrive_checkpoint_state_dict(checkpoint["state_dict"])

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Tensor]:
        return _filter_recogdrive_checkpoint_state_dict(super().state_dict(*args, **kwargs))

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()


class AgentLightningDiT(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent

    def _log_optional_recogdrive_metrics(self, prediction: Any, logging_prefix: str) -> None:
        AgentLightningModule._log_optional_recogdrive_metrics(self, prediction, logging_prefix)

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets, tokens_list = batch
        prediction = self.agent.forward(features,targets,tokens_list)
        if logging_prefix == 'train':
            predictions = self.agent.compute_loss(features, targets, prediction)
            if isinstance(predictions, torch.Tensor):
                loss = predictions
                metric_source = prediction
            else:
                loss = _prediction_get(predictions, "loss")
                if loss is None:
                    raise KeyError("Training prediction object is missing a 'loss' field.")
                metric_source = predictions
            self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/total_loss", loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            for key in ("reward", "policy_loss", "bc_loss"):
                value = _prediction_get(metric_source, key)
                if value is not None:
                    self.log(
                        f"{logging_prefix}/{key}",
                        value,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=True,
                        sync_dist=True,
                    )
            self._log_optional_recogdrive_metrics(metric_source, logging_prefix)
        else:
            prediction = self.agent.forward(features,targets)
            loss = self.agent.compute_loss(features, targets, prediction)
            self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/total_loss", loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            self._log_optional_recogdrive_metrics(prediction, logging_prefix)
        return loss
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        每次保存 checkpoint 时，只保留 state_dict 中不以 'agent.model' 开头的条目。
        """
        checkpoint["state_dict"] = _filter_recogdrive_checkpoint_state_dict(checkpoint["state_dict"])

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Tensor]:
        return _filter_recogdrive_checkpoint_state_dict(super().state_dict(*args, **kwargs))

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        #print(batch_idx)
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()
