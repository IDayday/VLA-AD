import torch

from navsim.agents.recogdrive.risk_vla.anchor_grpo_rewards import anchor_grpo_reward, supervised_gates_pass


def test_anchor_grpo_reward_penalizes_safety_regression():
    safe = anchor_grpo_reward(
        utility=torch.tensor([0.5]),
        zero_path_repair=torch.tensor([1.0]),
        ttc_repair=torch.tensor([1.0]),
        progress_recovery=torch.tensor([1.0]),
        comfort_stability=torch.tensor([1.0]),
        nc_regression=torch.tensor([0.0]),
        ttc_regression=torch.tensor([0.0]),
        dac_regression=torch.tensor([0.0]),
        tail_risk=torch.tensor([0.0]),
        positive_anchor_distance=torch.tensor([0.1]),
        negative_anchor_distance=torch.tensor([1.0]),
    )
    unsafe = anchor_grpo_reward(
        utility=torch.tensor([0.5]),
        zero_path_repair=torch.tensor([1.0]),
        ttc_repair=torch.tensor([1.0]),
        progress_recovery=torch.tensor([1.0]),
        comfort_stability=torch.tensor([1.0]),
        nc_regression=torch.tensor([1.0]),
        ttc_regression=torch.tensor([1.0]),
        dac_regression=torch.tensor([1.0]),
        tail_risk=torch.tensor([1.0]),
        positive_anchor_distance=torch.tensor([0.1]),
        negative_anchor_distance=torch.tensor([1.0]),
    )
    assert safe.item() > unsafe.item()


def test_supervised_gates_require_10k_validation():
    gates = {
        "mean_pdms_delta_vs_a0": 0.01,
        "p10_delta_vs_a0": 0.01,
        "zero_delta_vs_a0": 0.0,
        "dac0_delta_vs_a0": 0.0,
        "nc0_delta_vs_a0": 0.0,
        "ttc0_delta_vs_a0": 0.0,
        "critic_pairwise_accuracy": 0.6,
        "validation_samples": 10000,
    }
    assert supervised_gates_pass(gates)
    gates["validation_samples"] = 1024
    assert not supervised_gates_pass(gates)
