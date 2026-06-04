from __future__ import annotations

from dataclasses import dataclass
import sys
from types import ModuleType

from omegaconf import OmegaConf
import torch

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

_install_dependency_stubs()

dataclasses_mod = sys.modules["navsim.common.dataclasses"]


@dataclass
class SceneFilter:
    tokens: list[str] | None = None
    log_names: list[str] | None = None


dataclasses_mod.SceneFilter = getattr(dataclasses_mod, "SceneFilter", SceneFilter)

dataloader_mod = sys.modules["navsim.common.dataloader"]


class SceneLoader:
    pass


dataloader_mod.SceneLoader = getattr(dataloader_mod, "SceneLoader", SceneLoader)

abstract_agent_mod = ModuleType("navsim.agents.abstract_agent")


class AbstractAgent:
    pass


abstract_agent_mod.AbstractAgent = AbstractAgent
sys.modules["navsim.agents.abstract_agent"] = abstract_agent_mod

dataset_mod = ModuleType("navsim.planning.training.dataset")


class CacheOnlyDataset:
    pass


class Dataset:
    pass


dataset_mod.CacheOnlyDataset = CacheOnlyDataset
dataset_mod.Dataset = Dataset
sys.modules["navsim.planning.training.dataset"] = dataset_mod

lightning_mod = ModuleType("navsim.planning.training.agent_lightning_module")


class AgentLightningModule:
    pass


lightning_mod.AgentLightningModule = AgentLightningModule
sys.modules["navsim.planning.training.agent_lightning_module"] = lightning_mod

from navsim.planning.script import run_training_recogdrive as training_script
from navsim.planning.script.run_training_recogdrive import _normalized_dataloader_params, custom_collate_fn


def test_normalized_dataloader_params_clear_prefetch_for_zero_workers():
    params = _normalized_dataloader_params(
        OmegaConf.create({"batch_size": 1, "num_workers": 0, "prefetch_factor": 2, "persistent_workers": True})
    )

    assert params["prefetch_factor"] is None
    assert params["persistent_workers"] is False


def _base_features(**extra):
    features = {
        "history_trajectory": torch.zeros(4, 3),
        "high_command_one_hot": torch.tensor([0.0, 1.0, 0.0]),
        "status_feature": torch.zeros(10),
    }
    features.update(extra)
    return features


def test_custom_collate_accepts_online_scene_loader_pairs_with_image_paths():
    targets = {"trajectory": torch.zeros(8, 3)}
    features, collated_targets, tokens = custom_collate_fn(
        [
            (_base_features(image_path_tensor=torch.tensor([47, 97], dtype=torch.long)), targets),
            (_base_features(image_path_tensor=torch.tensor([47, 98, 99], dtype=torch.long)), targets),
        ]
    )

    assert tokens == [None, None]
    assert features["image_path_tensor"].shape == (2, 3)
    assert features["image_path_tensor"][0].tolist() == [47, 97, 0]
    assert "last_hidden_state" not in features
    assert collated_targets["trajectory"].shape == (2, 8, 3)


def test_custom_collate_preserves_cache_tokens_with_hidden_states():
    targets = {"trajectory": torch.zeros(8, 3)}
    features, _, tokens = custom_collate_fn(
        [
            (_base_features(last_hidden_state=torch.zeros(2, 1536)), targets, "token-a"),
            (_base_features(last_hidden_state=torch.zeros(3, 1536)), targets, "token-b"),
        ]
    )

    assert tokens == ["token-a", "token-b"]
    assert features["last_hidden_state"].shape == (2, 3, 1536)
    assert "image_path_tensor" not in features


def test_build_datasets_requests_camera_paths_from_scene_loader(monkeypatch):
    scene_loader_calls = []

    class CapturingSceneLoader:
        def __init__(self, **kwargs):
            scene_loader_calls.append(kwargs)

    class CapturingDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Agent:
        def get_sensor_config(self):
            return "sensor-config"

        def get_feature_builders(self):
            return ["feature-builder"]

        def get_target_builders(self):
            return ["target-builder"]

    def fake_instantiate(_cfg):
        return SceneFilter()

    monkeypatch.setattr(training_script, "SceneLoader", CapturingSceneLoader)
    monkeypatch.setattr(training_script, "Dataset", CapturingDataset)
    monkeypatch.setattr(training_script, "instantiate", fake_instantiate)

    cfg = OmegaConf.create(
        {
            "train_test_split": {"scene_filter": {}},
            "train_logs": ["train-log"],
            "val_logs": ["val-log"],
            "navsim_log_path": "/logs",
            "sensor_blobs_path": "/blobs",
            "cache_path": None,
            "force_cache_computation": False,
            "use_cache_without_dataset": False,
            "agent": {
                "use_expert_features": False,
                "expert_feature_source": "none",
                "expert_cache_dir": None,
            },
        }
    )

    training_script.build_datasets(cfg, Agent())

    assert len(scene_loader_calls) == 2
    assert [call["load_image_path"] for call in scene_loader_calls] == [True, True]
    assert scene_loader_calls[0]["scene_filter"].log_names == ["train-log"]
    assert scene_loader_calls[1]["scene_filter"].log_names == ["val-log"]
