from __future__ import annotations

import pytest
import torch

from navsim.agents.recogdrive.dynamic_tokenizer import DynamicTokenPacker


def test_dynamic_tokenizer_highcap_from_dense_temporal_spatial_input():
    dense = torch.randn(2, 8, 4, 4, 1024)
    packer = DynamicTokenPacker(output_tokens=128, temporal_bins=8, spatial_tokens_per_bin=16, strict=True)

    tokens, metadata = packer.pack(dense)

    assert tokens.shape == (2, 128, 1024)
    assert metadata["output_tokens"] == 128
    assert metadata["temporal_bins"] == 8
    assert metadata["spatial_tokens_per_bin"] == 16
    assert metadata["tokenizer_mode"] == "temporal_spatial_pool"


def test_dynamic_tokenizer_highcap_rejects_old_12_token_cache_in_strict_mode():
    packer = DynamicTokenPacker(output_tokens=128, temporal_bins=8, spatial_tokens_per_bin=16, strict=True)

    with pytest.raises(ValueError, match="old 12-token cache is insufficient"):
        packer.pack(torch.randn(1, 12, 1024))


def test_dynamic_tokenizer_highcap_dense_sequence_records_adaptive_pool_mode():
    packer = DynamicTokenPacker(output_tokens=128, temporal_bins=8, spatial_tokens_per_bin=16, strict=True)
    tokens, metadata = packer.pack(torch.randn(1, 256, 1024), temporal_size=7)

    assert tokens.shape == (1, 128, 1024)
    assert metadata["tokenizer_mode"] == "dense_adaptive_pool"
