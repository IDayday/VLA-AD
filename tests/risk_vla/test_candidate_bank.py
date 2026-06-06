import torch

from navsim.agents.recogdrive.risk_vla.candidate_bank import CandidateBank, load_candidate_cache, save_candidate_cache


def test_candidate_bank_generates_and_caches(tmp_path):
    bank = CandidateBank(horizon=8)
    trajectories, metadata = bank.generate(torch.zeros(2, 8, 3), k=4, seed=7)
    assert trajectories.shape == (2, 4, 8, 3)
    assert len(metadata) == 4
    save_candidate_cache(tmp_path, ["a", "b"], trajectories, metadata)
    tokens, loaded, loaded_metadata = load_candidate_cache(tmp_path)
    assert tokens == ["a", "b"]
    assert loaded.shape == trajectories.shape
    assert loaded_metadata[0]["candidate_id"] == 0
