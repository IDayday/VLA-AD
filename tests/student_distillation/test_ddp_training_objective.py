"""Two actual Gloo ranks, native scene-loss code, full/tail global updates."""
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest
from student_distillation.distributed import Environment, wrap_adapter
from student_distillation.checkpointing import clone_nested, save_checkpoint, read_checkpoint, load_checkpoint
from student_distillation.train import make_optimizer, train_update, verify_next_update


class SmallAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output_projection = torch.nn.Linear(24, 384)
        torch.nn.init.zeros_(self.output_projection.weight)
        torch.nn.init.zeros_(self.output_projection.bias)

    def forward(self, base_post, image_mask, text_mask, state24):
        delta = self.output_projection(state24)
        valid = image_mask | text_mask
        return {"post": base_post + valid.unsqueeze(-1) * delta.unsqueeze(1), "delta": delta,
                "actual_rms": delta.square().mean(-1).sqrt()}


class SyntheticScenes:
    def __len__(self):
        return 117

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(4567 + index)
        target = torch.randn(384, generator=generator) * .01
        return {"base_post": torch.randn(5, 384, generator=generator),
                "image_mask": torch.tensor([False, True, True, False, False]),
                "text_mask": torch.tensor([False, False, False, True, True]),
                "state24": torch.randn(24, generator=generator), "teacher_actual_shift": target,
                "history12": torch.zeros(12), "status8": torch.zeros(8),
                "metadata": {"token": f"synthetic{index}", "log_id": "synthetic", "target_is_zero": False}}


def config():
    return {"experiment": "H_full", "enable_transition_loss": False, "state_source": "none", "training_seed": 20261027,
            "training": {"global_batch_size": 64, "updates_per_epoch": 2, "epochs": 40},
            "optimizer": {"lr": .0001, "betas": [.9, .999], "eps": 1e-8, "weight_decay": .01, "grad_clip_norm": 1.},
            "scheduler": {"warmup_updates": 1341, "total_updates": 53640, "min_lr": 1e-5},
            "loss": {"additional_keep_weight": 0.}}


def _worker(rank, world, rendezvous, destination):
    torch.set_num_threads(1)
    if world > 1:
        dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world)
    env = Environment(rank, world, rank, "cpu")
    torch.manual_seed(20261027)
    adapter = SmallAdapter()
    cfg = config()
    optimizer, scheduler, groups = make_optimizer(adapter, cfg)
    wrapped = wrap_adapter(adapter, env)
    dataset = SyntheticScenes()
    permutation = list(range(len(dataset)))
    calibration = {"h_mse_scale": .0001}
    metrics = []
    metrics.append(train_update(wrapped, optimizer, scheduler, dataset, permutation, 0, cfg, 0, env, None, calibration))
    checkpoint = Path(destination) / f"checkpoint_w{world}.pt"
    save_checkpoint(checkpoint, adapter=wrapped, optimizer=optimizer, scheduler=scheduler,
                    progress={"global_step": 1, "epoch_index": 0, "global_batch_cursor": 1, "real_scenes_seen": 64},
                    sampler={"permutation": permutation, "epoch_index": 0}, identities={"synthetic": True},
                    config=cfg, calibration=calibration, parameter_groups=groups, env=env)
    verify_next_update(checkpoint, wrapped, optimizer, scheduler, dataset, cfg, env, None,
                       calibration, {"synthetic": True}, Path(destination) / f"RECOVERY_w{world}.json")
    metrics.append(train_update(wrapped, optimizer, scheduler, dataset, permutation, 1, cfg, 0, env, None, calibration))
    if rank == 0:
        torch.save(clone_nested({"parameters": adapter.state_dict(), "gradients": {name: value.grad for name, value in adapter.named_parameters()},
                                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "metrics": metrics}),
                   Path(destination) / f"result_w{world}.pt")
    if world > 1:
        dist.destroy_process_group()


def test_ddp_real_loss_matches_single_and_replays_next_tail_update(tmp_path):
    _worker(0, 1, str(tmp_path / "single_unused"), str(tmp_path))
    mp.spawn(_worker, args=(2, str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    single = torch.load(tmp_path / "result_w1.pt", weights_only=False)
    parallel = torch.load(tmp_path / "result_w2.pt", weights_only=False)
    for name in single["parameters"]:
        torch.testing.assert_close(single["parameters"][name], parallel["parameters"][name], rtol=1e-5, atol=1e-9)
    for name in single["gradients"]:
        torch.testing.assert_close(single["gradients"][name], parallel["gradients"][name], rtol=2e-5, atol=5e-8)
    assert [value["real_scenes"] for value in single["metrics"]] == [64, 53]
    assert single["scheduler"] == parallel["scheduler"]
    for left, right in zip(single["metrics"], parallel["metrics"]):
        assert abs(left["loss"] - right["loss"]) < 1e-7
    assert (tmp_path / "RECOVERY_w1.json").exists()
    assert (tmp_path / "RECOVERY_w2.json").exists()
    state = read_checkpoint(tmp_path / "checkpoint_w2.pt")
    assert state["world_size"] == 2 and len(state["rng_by_rank"]) == 2
    model = SmallAdapter()
    optimizer, scheduler, _ = make_optimizer(model, config())
    with pytest.raises(ValueError, match="checkpoint world size"):
        load_checkpoint(tmp_path / "checkpoint_w2.pt", model, optimizer, scheduler,
                        Environment(world_size=4), {"synthetic": True})
