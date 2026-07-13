from __future__ import annotations

import torch

from navsim.agents.recogdrive.recogdrive_backbone import RecogDriveBackbone


class PeftLikeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = torch.nn.Linear(2, 2).float()
        self.base_layer = torch.nn.Linear(2, 2).to(torch.bfloat16)

    def get_base_model(self):
        return self


def test_backbone_compute_dtype_skips_lora_parameters():
    model = PeftLikeModel()

    assert next(model.parameters()).dtype == torch.float32
    assert RecogDriveBackbone._infer_model_compute_dtype(model) == torch.bfloat16


class LanguageModelWithEmbeddings(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(4, 2).to(torch.bfloat16)

    def get_input_embeddings(self):
        return self.embedding


class InternVLLikeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = LanguageModelWithEmbeddings()

    def extract_feature(self, pixel_values):
        return pixel_values.float()


def test_internvl_visual_features_match_language_embedding_dtype():
    model = InternVLLikeModel()

    RecogDriveBackbone._patch_internvl_visual_feature_dtype(model)
    features = model.extract_feature(torch.ones(1, 2, dtype=torch.bfloat16))

    assert features.dtype == torch.bfloat16
