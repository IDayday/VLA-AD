"""Smoke test for optional ReCogDrive expert-token planner fusion.

This script installs small dependency stubs before importing the planner so it
can run in a lightweight environment without the full training stack.

Run from the repository root:

    python scripts/testing/smoke_recogdrive_expert_planner.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import warnings

import torch
from torch import nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_dependency_stubs() -> None:
    if hasattr(torch, "compile"):
        torch.compile = lambda fn=None, *args, **kwargs: fn if fn is not None else (lambda f: f)

    transformers_mod = sys.modules.get("transformers", ModuleType("transformers"))

    class PretrainedConfig:
        pass

    class _AutoModelStub:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class _AutoTokenizerStub:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    transformers_mod.PretrainedConfig = getattr(transformers_mod, "PretrainedConfig", PretrainedConfig)
    transformers_mod.AutoModel = getattr(transformers_mod, "AutoModel", _AutoModelStub)
    transformers_mod.AutoTokenizer = getattr(transformers_mod, "AutoTokenizer", _AutoTokenizerStub)
    feature_extraction_mod = ModuleType("transformers.feature_extraction_utils")
    modeling_outputs_mod = ModuleType("transformers.modeling_outputs")

    class BatchFeature(dict):
        def __init__(self, data=None, **kwargs):
            super().__init__(data or {}, **kwargs)

        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

    feature_extraction_mod.BatchFeature = BatchFeature
    modeling_outputs_mod.CausalLMOutputWithPast = getattr(
        modeling_outputs_mod,
        "CausalLMOutputWithPast",
        type("CausalLMOutputWithPast", (), {}),
    )
    sys.modules.setdefault("transformers", transformers_mod)
    sys.modules.setdefault("transformers.feature_extraction_utils", feature_extraction_mod)
    sys.modules.setdefault("transformers.modeling_outputs", modeling_outputs_mod)

    timm_mod = ModuleType("timm")
    timm_models_mod = ModuleType("timm.models")
    timm_layers_mod = ModuleType("timm.models.layers")

    class Mlp(nn.Module):
        def __init__(self, in_features, hidden_features=None, out_features=None, norm_layer=None, **kwargs):
            super().__init__()
            hidden_features = hidden_features or in_features
            out_features = out_features or in_features
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = nn.GELU()
            self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
            self.fc2 = nn.Linear(hidden_features, out_features)

        def forward(self, x):
            return self.fc2(self.norm(self.act(self.fc1(x))))

    timm_layers_mod.Mlp = Mlp
    timm_models_mod.layers = timm_layers_mod
    timm_mod.models = timm_models_mod
    sys.modules.setdefault("timm", timm_mod)
    sys.modules.setdefault("timm.models", timm_models_mod)
    sys.modules.setdefault("timm.models.layers", timm_layers_mod)

    diffusers_mod = ModuleType("diffusers")
    diffusers_models_mod = ModuleType("diffusers.models")
    diffusers_embeddings_mod = ModuleType("diffusers.models.embeddings")

    class Timesteps(nn.Module):
        def __init__(self, num_channels, flip_sin_to_cos=True, downscale_freq_shift=1):
            super().__init__()
            self.num_channels = num_channels
            self.flip_sin_to_cos = flip_sin_to_cos

        def forward(self, timesteps):
            half = self.num_channels // 2
            freqs = torch.exp(
                -torch.arange(half, device=timesteps.device, dtype=torch.float32)
                * torch.log(torch.tensor(10000.0, device=timesteps.device))
                / max(half - 1, 1)
            )
            args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
            emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
            if self.flip_sin_to_cos:
                emb = torch.cat([emb[:, half:], emb[:, :half]], dim=1)
            if emb.shape[1] < self.num_channels:
                emb = F.pad(emb, (0, self.num_channels - emb.shape[1]))
            return emb

    class TimestepEmbedding(nn.Module):
        def __init__(self, in_channels, time_embed_dim):
            super().__init__()
            self.linear_1 = nn.Linear(in_channels, time_embed_dim)
            self.act = nn.SiLU()
            self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

        def forward(self, x):
            return self.linear_2(self.act(self.linear_1(x)))

    diffusers_embeddings_mod.Timesteps = Timesteps
    diffusers_embeddings_mod.TimestepEmbedding = TimestepEmbedding
    diffusers_models_mod.embeddings = diffusers_embeddings_mod
    diffusers_mod.models = diffusers_models_mod
    sys.modules.setdefault("diffusers", diffusers_mod)
    sys.modules.setdefault("diffusers.models", diffusers_models_mod)
    sys.modules.setdefault("diffusers.models.embeddings", diffusers_embeddings_mod)

    dataclasses_mod = ModuleType("navsim.common.dataclasses")

    @dataclass
    class Trajectory:
        poses: object

    dataclasses_mod.Trajectory = Trajectory
    sys.modules.setdefault("navsim.common.dataclasses", dataclasses_mod)

    dataloader_mod = ModuleType("navsim.common.dataloader")

    class MetricCacheLoader:
        def __init__(self, path):
            self.metric_cache_paths = {}

    dataloader_mod.MetricCacheLoader = MetricCacheLoader
    sys.modules.setdefault("navsim.common.dataloader", dataloader_mod)

    pdm_score_mod = ModuleType("navsim.evaluate.pdm_score")
    pdm_score_mod.pdm_score = lambda **kwargs: SimpleNamespace(score=0.0)
    pdm_score_mod.get_trajectory_as_array = lambda *args, **kwargs: None
    pdm_score_mod.transform_trajectory = lambda *args, **kwargs: None
    sys.modules.setdefault("navsim.evaluate.pdm_score", pdm_score_mod)

    pdm_score_batch_mod = ModuleType("navsim.evaluate.pdm_score_batch")
    pdm_score_batch_mod.pdm_score_batch_same_cache = lambda *args, **kwargs: {}
    sys.modules.setdefault("navsim.evaluate.pdm_score_batch", pdm_score_batch_mod)

    scorer_mod = ModuleType("navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer")

    @dataclass
    class PDMScorerConfig:
        progress_weight: float = 10.0
        ttc_weight: float = 5.0
        comfortable_weight: float = 2.0

    class PDMScorer:
        def __init__(self, *args, **kwargs):
            pass

    scorer_mod.PDMScorerConfig = PDMScorerConfig
    scorer_mod.PDMScorer = PDMScorer
    sys.modules.setdefault("navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer", scorer_mod)

    fast_scorer_mod = ModuleType("navsim.planning.simulation.planner.pdm_planner.scoring.fast_pdm_scorer")

    class FastPDMScorer:
        def __init__(self, *args, **kwargs):
            pass

    fast_scorer_mod.FastPDMScorer = FastPDMScorer
    sys.modules.setdefault("navsim.planning.simulation.planner.pdm_planner.scoring.fast_pdm_scorer", fast_scorer_mod)

    simulator_mod = ModuleType("navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator")

    class PDMSimulator:
        def __init__(self, proposal_sampling):
            self.proposal_sampling = proposal_sampling

    simulator_mod.PDMSimulator = PDMSimulator
    sys.modules.setdefault("navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator", simulator_mod)

    nuplan_modules = [
        "nuplan",
        "nuplan.planning",
        "nuplan.planning.simulation",
        "nuplan.planning.simulation.trajectory",
    ]
    for module_name in nuplan_modules:
        sys.modules.setdefault(module_name, ModuleType(module_name))

    trajectory_sampling_mod = ModuleType("nuplan.planning.simulation.trajectory.trajectory_sampling")

    class TrajectorySampling:
        def __init__(self, time_horizon=4, interval_length=0.5):
            self.time_horizon = time_horizon
            self.interval_length = interval_length

    trajectory_sampling_mod.TrajectorySampling = TrajectorySampling
    sys.modules.setdefault("nuplan.planning.simulation.trajectory.trajectory_sampling", trajectory_sampling_mod)


_install_dependency_stubs()

from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.recogdrive_diffusion_planner import (  # noqa: E402
    DDIMConfig,
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)


def _tiny_config(
    use_expert_features: bool,
    use_jepa: bool = True,
    use_vggt: bool = True,
    expert_alignment_weight: float = 0.0,
    jepa_alignment_weight: float = 0.0,
    vggt_alignment_weight: float = 0.0,
    alignment_loss_type: str = "normalized_mse",
) -> ReCogDriveDiffusionPlannerConfig:
    return ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg={
            "num_heads": 1,
            "head_dim": 8,
            "num_layers": 2,
            "output_dim": 8,
            "dropout": 0.0,
            "attention_bias": True,
            "norm_eps": 1e-5,
            "interleave_attention": True,
        },
        input_embedding_dim=8,
        hidden_size=16,
        action_dim=3,
        action_horizon=8,
        max_seq_len=8,
        sampling_method="ddim",
        num_inference_steps=2,
        vlm_size="small",
        ddim_cfg=DDIMConfig(num_train_timesteps=10),
        use_expert_features=use_expert_features,
        use_jepa=use_jepa,
        use_vggt=use_vggt,
        jepa_dim=1024 if use_expert_features and use_jepa else 0,
        vggt_dim=2048 if use_expert_features and use_vggt else 0,
        expert_adapter_dim=768,
        num_jepa_tokens=12,
        num_vggt_tokens=12,
        expert_dropout=0.0,
        expert_alignment_weight=expert_alignment_weight,
        jepa_alignment_weight=jepa_alignment_weight,
        vggt_alignment_weight=vggt_alignment_weight,
        alignment_loss_type=alignment_loss_type,
    )


def _features(batch_size: int = 2) -> torch.Tensor:
    return torch.randn(batch_size, 3, 1536)


def _action_input(
    batch_size: int = 2,
    include_action: bool = True,
    include_experts: bool = False,
    include_targets: bool = True,
) -> BatchFeature:
    data = {
        "his_traj": torch.randn(batch_size, 12),
        "status_feature": torch.randn(batch_size, 8),
    }
    if include_action:
        data["action"] = torch.randn(batch_size, 8, 3)
    if include_experts:
        data["jepa_context_tokens"] = torch.randn(batch_size, 12, 1024)
        data["vggt_context_tokens"] = torch.randn(batch_size, 12, 2048)
        if include_targets:
            data["jepa_target_tokens"] = torch.randn(batch_size, 12, 1024)
            data["vggt_target_tokens"] = torch.randn(batch_size, 12, 2048)
    return BatchFeature(data=data)


def main() -> None:
    torch.manual_seed(7)

    baseline = ReCogDriveDiffusionPlanner(_tiny_config(use_expert_features=False))
    baseline.train()
    baseline_loss = baseline(_features(), _action_input(include_experts=False)).loss
    assert torch.isfinite(baseline_loss), "baseline forward loss is not finite"
    baseline_out = baseline(_features(), _action_input(include_experts=False))
    assert baseline_out["jepa_alignment_loss"].item() == 0.0
    assert baseline_out["vggt_alignment_loss"].item() == 0.0

    reloaded = ReCogDriveDiffusionPlanner(_tiny_config(use_expert_features=False))
    reloaded.load_state_dict(baseline.state_dict(), strict=True)

    expert = ReCogDriveDiffusionPlanner(_tiny_config(use_expert_features=True))
    expert.train()
    expert_loss = expert(_features(), _action_input(include_experts=True)).loss
    assert torch.isfinite(expert_loss), "expert forward loss is not finite"
    expert_no_targets = expert(
        _features(),
        _action_input(include_experts=True, include_targets=False),
    )
    assert expert_no_targets["jepa_alignment_loss"].item() == 0.0
    assert expert_no_targets["vggt_alignment_loss"].item() == 0.0

    jepa_only = ReCogDriveDiffusionPlanner(_tiny_config(use_expert_features=True, use_jepa=True, use_vggt=False))
    jepa_only.train()
    jepa_only_input = _action_input(include_experts=False)
    jepa_only_input["jepa_context_tokens"] = torch.randn(2, 12, 1024)
    jepa_only_out = jepa_only(_features(), jepa_only_input)
    assert torch.isfinite(jepa_only_out.loss)

    vggt_only = ReCogDriveDiffusionPlanner(_tiny_config(use_expert_features=True, use_jepa=False, use_vggt=True))
    vggt_only.train()
    vggt_only_input = _action_input(include_experts=False)
    vggt_only_input["vggt_context_tokens"] = torch.randn(2, 12, 2048)
    vggt_only_out = vggt_only(_features(), vggt_only_input)
    assert torch.isfinite(vggt_only_out.loss)

    aligned = ReCogDriveDiffusionPlanner(
        _tiny_config(
            use_expert_features=True,
            expert_alignment_weight=0.5,
            jepa_alignment_weight=0.25,
            vggt_alignment_weight=0.25,
        )
    )
    aligned.train()
    aligned_out = aligned(_features(), _action_input(include_experts=True, include_targets=True))
    assert torch.isfinite(aligned_out["loss"])
    assert torch.isfinite(aligned_out["diffusion_loss"])
    assert aligned_out["jepa_alignment_loss"].item() > 0.0
    assert aligned_out["vggt_alignment_loss"].item() > 0.0

    cosine_aligned = ReCogDriveDiffusionPlanner(
        _tiny_config(
            use_expert_features=True,
            expert_alignment_weight=1.0,
            alignment_loss_type="cosine",
        )
    )
    cosine_out = cosine_aligned(_features(), _action_input(include_experts=True, include_targets=True))
    assert torch.isfinite(cosine_out["jepa_alignment_loss"])
    assert torch.isfinite(cosine_out["vggt_alignment_loss"])

    expert.eval()
    with torch.no_grad():
        current_only_pred = expert.get_action(
            _features(),
            _action_input(include_action=False, include_experts=True, include_targets=False),
            deterministic=True,
        )
    assert current_only_pred["pred_traj"].shape == (2, 8, 3), current_only_pred["pred_traj"].shape

    inference_input = _action_input(include_action=False, include_experts=True, include_targets=False)
    inference_input["jepa_target_tokens"] = object()
    inference_input["vggt_target_tokens"] = object()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.no_grad():
            pred = expert.get_action(
                _features(),
                inference_input,
                deterministic=True,
            )
    assert any("train-only expert target keys" in str(w.message) for w in caught)
    assert pred["pred_traj"].shape == (2, 8, 3), pred["pred_traj"].shape

    try:
        expert(_features(), _action_input(include_experts=False))
    except KeyError as exc:
        assert "jepa_context_tokens" in str(exc)
    else:
        raise AssertionError("missing expert tensors should raise a clear KeyError")

    print("ReCogDrive expert planner smoke test passed.")


if __name__ == "__main__":
    main()
