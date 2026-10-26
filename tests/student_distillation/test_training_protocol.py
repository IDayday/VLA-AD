import copy
import json
from pathlib import Path
import random
import numpy as np
import pytest
import torch
from student_distillation.distributed import Environment, epoch_permutation, global_batch_slots
from student_distillation.checkpointing import (clone_nested, load_checkpoint, read_checkpoint,
    rng_state, save_checkpoint)
from student_distillation.train import WarmupCosine, make_optimizer, _assert_same_nested
from student_distillation.experiment_registry import load_config, validate_config
from student_distillation.aggregate import summarize_records, grouped_bootstrap
from student_distillation.evaluate import train_panel


def test_tail_slots_are_53_real_and_11_zero_weight():
    permutation = list(range(85813))
    for world in (1, 2, 4, 8, 16, 32, 64):
        slots = [global_batch_slots(permutation, 1340, rank, world) for rank in range(world)]
        assert {real_count for _, real_count in slots} == {53}
        assert {len(local) for local, _ in slots} == {64 // world}
        assert sum(real for local, _ in slots for _, real in local) == 53
        assert sum((world / 53 if real else 0) / world for local, _ in slots for _, real in local) == pytest.approx(1.)
        assert sorted(index for local, _ in slots for index, real in local if real) == list(range(85760, 85813))


def test_shuffle_is_epoch_specific_and_independent_of_global_rng():
    first = epoch_permutation(100, 20261027, 0)
    random.seed(14)
    np.random.seed(94)
    torch.manual_seed(180)
    assert first == epoch_permutation(100, 20261027, 0)
    assert first != epoch_permutation(100, 20261027, 1)
    assert sorted(first) == list(range(100))


def test_no_decay_and_learning_rate_endpoints():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.queries = torch.nn.Parameter(torch.randn(1, 4, 2))
            self.linear = torch.nn.Linear(2, 2)
            self.norm = torch.nn.LayerNorm(2)
            self.output_projection = torch.nn.Linear(2, 1)
    cfg = {"optimizer": {"lr": 1e-4, "betas": [.9, .999], "eps": 1e-8, "weight_decay": .01},
           "scheduler": {"warmup_updates": 1341, "total_updates": 53640, "min_lr": 1e-5}}
    optimizer, scheduler, groups = make_optimizer(Tiny(), cfg)
    assert groups["decay"] == ["linear.weight"]
    assert set(groups["no_decay"]) == {"queries", "linear.bias", "norm.weight", "norm.bias", "output_projection.weight", "output_projection.bias"}
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4 / 1341)
    scheduler.completed = 1340
    scheduler._set()
    assert optimizer.param_groups[0]["lr"] == 1e-4
    scheduler.completed = 53639
    scheduler._set()
    assert optimizer.param_groups[0]["lr"] == 1e-5


def test_complete_checkpoint_has_independent_adam_rng_and_next_update(tmp_path):
    torch.manual_seed(91)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    scheduler = WarmupCosine(optimizer, 2, 10, .0001)
    x, y = torch.randn(9, 3), torch.randn(9, 2)
    def update():
        optimizer.zero_grad()
        loss = (model(x) - y).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return float(loss)
    update()
    path = tmp_path / "saved.pt"
    save_checkpoint(path, adapter=model, optimizer=optimizer, scheduler=scheduler,
                    progress={"global_step": 1}, sampler={"permutation": [2, 0, 1], "cursor": 1},
                    identities={"teacher": "new_local_teacher"}, config={"loss": "H"},
                    calibration={"h_mse_scale": 1.}, parameter_groups={}, env=Environment())
    state = read_checkpoint(path)
    live_adam = next(iter(optimizer.state.values()))["exp_avg"]
    saved_adam = next(iter(state["optimizer"]["state"].values()))["exp_avg"]
    saved_copy = saved_adam.clone()
    live_adam.add_(100)
    assert torch.equal(saved_adam, saved_copy)
    outcomes = []
    for _ in range(2):
        load_checkpoint(path, model, optimizer, scheduler, Environment(), {"teacher": "new_local_teacher"})
        value = update()
        outcomes.append(clone_nested({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                      "scheduler": scheduler.state_dict(), "loss": value, "rng": rng_state()}))
    _assert_same_nested(*outcomes)
    with pytest.raises(ValueError, match="identity"):
        read_checkpoint(path, {"teacher": "old_ppu_teacher"})


def test_all_four_registered_configs_keep_full_budget():
    root = Path(__file__).resolve().parents[2] / "configs/student_distillation"
    for experiment in ("H_full", "H_gain1", "H_mu_teacher", "H_mu_student"):
        cfg = load_config(root / f"{experiment}.yaml", root / "common.yaml")
        validate_config(cfg)
        cfg["training"]["epochs"] = 1
        with pytest.raises(ValueError):
            validate_config(cfg)


def score(value, nc=1., dac=1., ttc=1.):
    return {"score": value, "no_at_fault_collisions": nc, "drivable_area_compliance": dac,
            "time_to_collision_within_bound": ttc, "ego_progress": value, "comfort": 1., "driving_direction_compliance": 1.}


def test_paired_severe_events_are_not_scene_mean_buckets():
    records = [{"token": "a", "log_id": "L0", "original_choice": 2, "target_is_zero": False,
                "noise": [{"noise_index": 0, "score": score(0, nc=0), "base_score": score(.95)},
                          {"noise_index": 1, "score": score(.8), "base_score": score(.4)}]},
               {"token": "b", "log_id": "L1", "target_is_zero": True,
                "noise": [{"noise_index": 0, "score": score(.9), "base_score": score(.8)}]}]
    summary = summarize_records(records)
    assert summary["events"]["paired_base_gt90_student_zero"] == 1
    assert summary["events"]["new_zero_score"] == 1
    assert summary["groups"]["base_bucket"]["lt70"]["scenes"] == 1
    assert summary["behavior"]["pdms_points"] == pytest.approx(65.)
    assert summary["behavior"]["positive_gain_sum_points"] == pytest.approx(10.)
    assert summary["behavior"]["negative_loss_sum_points"] == pytest.approx(27.5)
    assert summary["bootstrap"]["replicates"] == 2000
    assert grouped_bootstrap(records) == grouped_bootstrap(records)


def test_train_panel_exact_count_and_fixed_hash_selection():
    rows = [{"token": f"T{i}", "log_id": f"L{i % 130}", "base_mean_points": 60. if i % 3 == 0 else 85. if i % 3 == 1 else 95.}
            for i in range(85813)]
    panel = train_panel(rows)
    assert len(panel) == len(set(panel)) == 8192
    assert panel == train_panel(rows)
    assert {rows[index]["log_id"] for index in panel} == {f"L{i}" for i in range(130)}
