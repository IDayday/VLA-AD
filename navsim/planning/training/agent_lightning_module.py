import pytorch_lightning as pl

from torch import Tensor
from typing import Dict, Tuple,Any
import torch

from navsim.agents.abstract_agent import AbstractAgent


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
        filtered_sd = {
            k: v
            for k, v in checkpoint['state_dict'].items()
            if not k.startswith('agent.model')
        }
        checkpoint['state_dict'] = filtered_sd

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

            loss = predictions.loss
            reward = predictions.reward
            policy_loss = predictions.policy_loss
            bc_loss = predictions.bc_loss
            self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/total_loss", loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f"{logging_prefix}/reward", reward, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/policy_loss", policy_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/bc_loss", bc_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self._log_optional_recogdrive_metrics(predictions, logging_prefix)
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
        filtered_sd = {
            k: v
            for k, v in checkpoint['state_dict'].items()
            if not k.startswith('agent.model')
        }
        checkpoint['state_dict'] = filtered_sd

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
