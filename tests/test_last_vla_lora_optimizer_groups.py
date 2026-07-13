from __future__ import annotations

from tests.test_last_vla_vlm_lora_trainable_scope import ReCogDriveAgent, TrajectorySampling, DummyBackbone


def test_lora_optimizer_groups_are_separate_and_scoped():
    agent = ReCogDriveAgent(
        trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
        checkpoint_path="",
        allow_random_init=True,
        cache_hidden_state=True,
        use_last_vla=True,
        last_vla_stage="cot_alignment",
        use_last_rd=False,
        train_expert_only=True,
        freeze_base_action_head=True,
        lr_vlm_lora=1e-5,
        lr_last_vla_cot=1e-4,
    )
    agent.initialize()
    agent.backbone = DummyBackbone()
    agent.last_vla_train_vlm_lora = True
    optimizer = agent._build_grouped_optimizer()
    report = agent.get_optimizer_group_report()
    names = [group["name"] for group in optimizer.param_groups]

    assert names == ["last_vla_cot", "vlm_lora"]
    assert [group["lr"] for group in optimizer.param_groups] == [1e-4, 1e-5]
    assert report[0]["parameter_count"] > 0
    assert report[1]["parameter_count"] > 0
    counts = agent.count_trainable_parameters_by_group()
    assert counts["action_base"]["trainable"] == 0
    assert counts["backbone_non_lora"]["trainable"] == 0
