"""RISK-VLA lightweight model-side scaffold.

RISK-VLA treats BiT-style path/terminal intent as one strategy in a strategy
bank. The core framework is risk state -> strategy routing/modulation ->
diffusion planning.
"""

from .dataclasses import DEFAULT_RISK_CLASS_ORDER, RiskState, StrategyOutput, StrategyWeights
from .bit_adapter import extract_bit_intents
from .risk_state_encoder import RiskStateEncoder
from .risk_losses import (
    ensure_scene_level_labels,
    mvp_labels_to_extended,
    reduce_horizon_labels,
    risk_bce_loss,
    risk_focal_loss,
    strategy_activation_supervision_loss,
    strategy_entropy_loss,
)
from .strategy_bank import (
    ComfortStrategy,
    InteractionSafetyStrategy,
    PathIntentStrategy,
    ProgressStrategy,
    RiskConditionedStrategyBank,
)
from .strategy_router import RiskConditionedStrategyRouter
from .candidate_bank import CandidateBank, CandidateSpec
from .trajectory_risk_critic import TrajectoryRiskCritic, TrajectoryRiskCriticOutput
from .utility_router import RiskVLAv2UtilityRouter, UtilityRouterOutput
from .risk_world_tokens import RiskWorldTokenEncoder

__all__ = [
    "DEFAULT_RISK_CLASS_ORDER",
    "RiskState",
    "StrategyOutput",
    "StrategyWeights",
    "RiskStateEncoder",
    "extract_bit_intents",
    "RiskConditionedStrategyRouter",
    "PathIntentStrategy",
    "InteractionSafetyStrategy",
    "ProgressStrategy",
    "ComfortStrategy",
    "RiskConditionedStrategyBank",
    "risk_bce_loss",
    "risk_focal_loss",
    "strategy_entropy_loss",
    "strategy_activation_supervision_loss",
    "reduce_horizon_labels",
    "mvp_labels_to_extended",
    "ensure_scene_level_labels",
    "CandidateBank",
    "CandidateSpec",
    "TrajectoryRiskCritic",
    "TrajectoryRiskCriticOutput",
    "RiskVLAv2UtilityRouter",
    "UtilityRouterOutput",
    "RiskWorldTokenEncoder",
]
