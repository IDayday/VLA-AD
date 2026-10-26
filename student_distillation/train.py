"""Native FP32 scene-microbatch training with exact global-batch DDP reduction."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
import yaml
from . import distributed as distributed
from .checkpointing import (atomic_json, clone_nested, digest, latest_complete, load_checkpoint,
                            read_checkpoint, restore_rng, rng_state, save_checkpoint, sha256)
from .experiment_registry import identities_for, load_config, validate_config, stamp

QUALIFICATION_CHECKS = ("asset_identity", "zero_identity", "actual_teacher_injection",
                        "native_bridge_replay", "zero_point_gradient", "known_realizable_positive_control",
                        "dit_input_gradient", "ddp_tail_objective", "next_update_recovery")


def verify_formal_config(cfg):
    """Reject ineffective or changed scientific options before any formal work."""
    validate_config(cfg)
    requirements = {
        "data": {"train_scenes": 85813, "validation_scenes": 17475, "test_scenes": 12146,
                 "preserve_native_token_layout": True, "natural_scene_weights": True},
        "model": {"query_count": 4, "hidden_dim": 256, "cross_attention_layers": 2,
                  "attention_heads": 4, "ffn_expansion": 4, "state_dim": 24, "dropout": 0.0,
                  "zero_init_output": True, "freeze_vlm": True, "freeze_feature_encoder": True,
                  "freeze_action_head": True},
        "training": {"epochs": 40, "global_batch_size": 64, "microbatch_per_device": 1,
                     "updates_per_epoch": 1341, "total_optimizer_updates": 53640, "early_stopping": False,
                     "automatic_scientific_stop_on_negative_pdms": False},
        "optimizer": {"name": "adamw", "lr": 1e-4, "betas": [.9, .999], "eps": 1e-8,
                      "weight_decay": .01, "grad_clip_norm": 1.0},
        "scheduler": {"name": "warmup_cosine", "warmup_updates": 1341,
                      "total_updates": 53640, "min_lr": 1e-5},
        "precision": {"adapter": "float32", "action_head": "float32", "loss": "float32",
                      "tf32": False, "autocast": False},
        "loss": {"representation_weight": 1.0, "additional_keep_weight": 0.0,
                 "transition_mean_weight_max": .5, "transition_start_update": 5364, "transition_ramp_updates": 2682},
        "sampler": {"method": "ddim", "timesteps": [80, 60, 40, 20, 0], "deterministic": False,
                    "eta": 1.0, "train_noise_namespace": "NRKD_train_v1", "train_noise_banks": 5},
        "evaluation": {"full_validation_epochs": [8, 16, 24, 32, 40],
                       "monitor_every_epochs": 2, "monitor_scenes": 2048,
                       "full_validation_noise_indices": [0, 1, 2, 3, 4],
                       "new_noise_validation_indices": [5, 6, 7, 8, 9],
                       "bootstrap_unit": "log", "bootstrap_replicates": 2000}}
    for section, values in requirements.items():
        for key, expected in values.items():
            if cfg.get(section, {}).get(key) != expected:
                raise ValueError(f"Formal Round1 requires {section}.{key}={expected!r}")
    targets = {"H_full": ("full", "none", False), "H_gain1": ("gain1", "none", False),
               "H_mu_teacher": ("gain1", "teacher", True), "H_mu_student": ("gain1", "student", True)}
    expected = targets.get(cfg.get("experiment"))
    if expected != (cfg.get("teacher_target"), cfg.get("state_source"), cfg.get("enable_transition_loss")):
        raise ValueError("Formal experiment target/state-source/loss assignment changed")


def verify_qualification(cfg, identities=None):
    """Require nine actual engineering checks bound to this code and these assets."""
    identities = identities or identities_for(cfg)
    path = cfg["paths"].get("qualification_receipt")
    if not path or not Path(path).is_file():
        raise ValueError("Formal training requires the actual CODE_CHECKS_PASS qualification receipt")
    receipt = json.loads(Path(path).read_text())
    if receipt.get("schema") != "nrkd_code_checks_v1" or receipt.get("status") != "CODE_CHECKS_PASS" or receipt.get("formal_asset_qualification") is not True:
        raise ValueError("Synthetic, pending, or partial engineering receipts cannot authorize formal training")
    if receipt.get("code_sha256") != identities["code_files"]:
        raise ValueError("Engineering qualification was produced by different implementation code")
    for key in ("teacher_bundle", "feature_manifest", "targets_manifest", "calibration"):
        expected = identities["files"][key]["sha256"]
        if receipt.get("identities", {}).get(key + "_sha256") != expected:
            raise ValueError(f"Engineering qualification {key} identity differs")
    if receipt.get("identities", {}).get("checkpoint_sha256") != sha256(cfg["paths"]["checkpoint"]):
        raise ValueError("Engineering qualification used a different frozen RL Base")
    for name in QUALIFICATION_CHECKS:
        check = receipt.get("checks", {}).get(name, {})
        if check.get("status") != "PASS":
            raise ValueError(f"Mandatory actual engineering check is incomplete: {name}")
        artifact = check.get("artifact", {})
        artifact_path = Path(artifact.get("path", ""))
        if not artifact_path.is_absolute():
            artifact_path = Path(path).parent / artifact_path
        if not artifact_path.is_file() or sha256(artifact_path) != artifact.get("sha256"):
            raise ValueError(f"Mandatory engineering check evidence SHA mismatch: {name}")
        if name == "next_update_recovery":
            evidence = json.loads(artifact_path.read_text())
            for source in ("teacher", "student"):
                active = evidence.get("active_transition_recovery", {}).get(source, {})
                if active.get("status") != "PASS" or active.get("trained_Adam_state") is not True or active.get("transition_loss_active") is not True or active.get("independent_loads") != 2:
                    raise ValueError(f"Actual updated-Adam active {source}-state transition recovery is not qualified")
        if name == "dit_input_gradient":
            evidence = json.loads(artifact_path.read_text())
            parity = evidence.get("student_forward_native_parity", {})
            if parity.get("status") != "PASS" or parity.get("scenes", 0) < 64 or parity.get("time_indices") != [0, 1, 2, 3, 4] or parity.get("bitwise") is not True:
                raise ValueError("Student deterministic compiled-VJP metadata lacks actual64scene all-five-step native bitwise forward parity")
    from .losses import teacher_query_cache_settings
    cache = teacher_query_cache_settings(cfg, refresh=True)
    if cache is not None and receipt.get("performance_admissions", {}).get("teacher_query_cache") != cache["admission"]:
        raise ValueError("Formal C qualification is missing its exact cache admission artifact and identities")
    result = {"path": str(Path(path).resolve()), "sha256": sha256(path), "checks": list(QUALIFICATION_CHECKS)}
    if cache is not None:
        result["performance_admissions"] = {"teacher_query_cache": cache["admission"]}
    return result


class WarmupCosine:
    """The LR used for update u is deterministic; step follows optimizer.step."""
    def __init__(self, optimizer, warmup_updates, total_updates, min_lr):
        self.optimizer = optimizer
        self.warmup = int(warmup_updates)
        self.total = int(total_updates)
        self.minimum = float(min_lr)
        self.peaks = [group["lr"] for group in optimizer.param_groups]
        self.completed = 0
        self._set()

    def _lr(self, peak):
        if self.completed < self.warmup:
            return peak * ((self.completed + 1) / max(1, self.warmup))
        fraction = (self.completed - self.warmup) / max(1, self.total - self.warmup - 1)
        return self.minimum + (peak - self.minimum) * (1 + math.cos(math.pi * min(1., fraction))) / 2

    def _set(self):
        for peak, group in zip(self.peaks, self.optimizer.param_groups):
            group["lr"] = self._lr(peak)

    def step(self):
        self.completed += 1
        self._set()

    def state_dict(self):
        return {"warmup": self.warmup, "total": self.total, "minimum": self.minimum,
                "peaks": self.peaks.copy(), "completed": self.completed}

    def load_state_dict(self, value):
        for key in ("warmup", "total", "minimum", "peaks", "completed"):
            setattr(self, key, value[key])
        self._set()


def make_optimizer(adapter, cfg):
    no_decay = set()
    for module_name, module in adapter.named_modules():
        if isinstance(module, nn.LayerNorm):
            no_decay.update(f"{module_name}.{name}".lstrip(".") for name, _ in module.named_parameters(recurse=False))
    groups = {"decay": [], "no_decay": []}
    names = {key: [] for key in groups}
    for name, parameter in adapter.named_parameters():
        if not parameter.requires_grad:
            continue
        key = "no_decay" if name in no_decay or name.endswith("bias") or name == "queries" or name.startswith("output_projection.") else "decay"
        groups[key].append(parameter)
        names[key].append(name)
    options = cfg["optimizer"]
    optimizer = torch.optim.AdamW([
        {"params": groups["decay"], "weight_decay": options["weight_decay"], "name": "decay"},
        {"params": groups["no_decay"], "weight_decay": 0., "name": "no_decay"}],
        lr=options["lr"], betas=tuple(options["betas"]), eps=options["eps"])
    schedule = cfg["scheduler"]
    scheduler = WarmupCosine(optimizer, schedule["warmup_updates"], schedule["total_updates"], schedule["min_lr"])
    return optimizer, scheduler, names


def dataset_manifest(cfg):
    return {"feature_manifest": cfg["paths"]["feature_manifest"],
            "targets_manifest": cfg["paths"].get("targets_manifest")}


def make_adapter(cfg, device):
    from .adapter import ContinuousD48Adapter
    from .asset_identity import load_library_bundle
    bundle = load_library_bundle(cfg["paths"]["teacher_bundle"])
    calibration = json.loads(Path(cfg["paths"]["calibration"]).read_text())
    adapter = ContinuousD48Adapter(
        torch.as_tensor(bundle["basis"], dtype=torch.float64), bundle["d0"],
        torch.as_tensor(calibration["token_mean"]), torch.as_tensor(calibration["token_std"]),
        torch.as_tensor(calibration["state_mean"]), torch.as_tensor(calibration["state_std"]),
        cap_in_d0=bundle["cap_in_d0"]).to(device=device, dtype=torch.float32)
    return adapter, calibration, bundle


def make_bridge(cfg, device, out):
    """Load the immutable RL head on this rank without worker.py's cuda:0 override."""
    if not str(device).startswith("cuda"):
        raise RuntimeError("The real frozen native head requires a CUDA device")
    from interface_teacher.runtime import bootstrap
    portable_paths = cfg["paths"].get("paths_json")
    if not portable_paths:
        portable_paths = Path(out) / "runtime_paths.json"
        atomic_json(portable_paths, cfg["paths"])
    bootstrap(portable_paths, out)
    from model.recogdrive_agent import ReCogDriveAgent
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from .native_bridge import NativeDDIMBridge
    baseline = json.loads((Path(__file__).resolve().parents[1] / "configs/repair_final_baseline.json").read_text())
    checkpoint_path = cfg["paths"]["checkpoint"]
    if sha256(checkpoint_path) != baseline["checkpoint_sha256"]:
        raise ValueError("The RL Base checkpoint SHA does not match the frozen evidence")
    torch.cuda.set_device(torch.device(device))
    agent = ReCogDriveAgent(TrajectorySampling(time_horizon=4, interval_length=.5), checkpoint_path=None,
                           vlm_path=None, cache_hidden_state=True, cache_mode=False, dit_type="small",
                           vlm_size="small", sampling_method="ddim", grpo=False, train_backbone=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    active = {name.removeprefix("agent."): value for name, value in checkpoint["state_dict"].items()
              if not name.removeprefix("agent.").startswith("action_head.old_policy.")}
    agent.load_state_dict(active, strict=True)
    head = agent.action_head.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    del checkpoint, active
    if head.ddim_t.tolist() != [80, 60, 40, 20, 0] or head.ddim_steps != 5:
        raise ValueError("Native DDIM sampler identity mismatch")
    eta = float(head.eta(torch.zeros((1, 8, 3), device=device))[0, 0])
    if eta != 1.:
        raise ValueError("Native DDIM eta must be one")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bridge = NativeDDIMBridge(head)
    graph_receipt = cfg["paths"].get("inference_graph_receipt")
    if graph_receipt:
        from .inference_graph import qualified_inference_bridge
        bridge = qualified_inference_bridge(bridge, graph_receipt, feature_manifest=cfg["paths"]["feature_manifest"])
    return bridge


def bridge_factory(paths, device):
    """Calibration CLI factory; accepts a portable paths file or a paths mapping."""
    if isinstance(paths, (str, Path)):
        path_file = str(Path(paths).resolve())
        values = json.loads(Path(paths).read_text())
        values["paths_json"] = path_file
    else:
        values = dict(paths)
    out = Path(values.get("results_root", values.get("runtime_root", "/mnt/project/recogdrive-no-routing-round1-results-20261026/calibration/runtime")))
    return make_bridge({"paths": values}, device, out)


def prepare_scene(item, cfg, epoch, device):
    from .data import batch_scene
    from .noise import training_schedule, train_noise_seed, make_noise_tape
    batch = batch_scene(item, device)
    if cfg["enable_transition_loss"] and epoch >= 4:
        token = item["metadata"]["token"]
        schedule = training_schedule(token, cfg["training_seed"], epoch)
        batch.update(schedule)
        batch["noise_tape"] = make_noise_tape(train_noise_seed(token, cfg["training_seed"], schedule["noise_bank"]), device)
    return batch


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


def _assert_same_nested(left, right, name="state"):
    if torch.is_tensor(left):
        if not torch.equal(left.cpu(), right.cpu()):
            difference = float((left.cpu().double() - right.cpu().double()).abs().max())
            raise AssertionError(f"Next-update recovery mismatch in {name}; max absolute difference {difference}")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"Recovery dictionary keys changed in {name}")
        for key in left:
            _assert_same_nested(left[key], right[key], f"{name}.{key}")
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"Recovery sequence length changed in {name}")
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_same_nested(a, b, f"{name}[{index}]")
    elif isinstance(left, np.ndarray):
        if not np.array_equal(left, right):
            raise AssertionError(f"Recovery NumPy state changed in {name}")
    elif left != right:
        raise AssertionError(f"Recovery scalar changed in {name}: {left!r} != {right!r}")


def verify_next_update(checkpoint, adapter, optimizer, scheduler, dataset, cfg, env,
                       bridge, calibration, identities, out=None):
    """Execute the actual next scene batch twice from two independent full loads.

    This operates on the real adapter, optimizer and training loss (including
    DDP's real tail normalization); it restores the live training state afterward.
    """
    initial = read_checkpoint(checkpoint, identities)
    progress = initial["progress"]
    epoch = progress["epoch_index"]
    if epoch >= cfg["training"]["epochs"]:
        raise ValueError("A finished run has no next formal training update")
    permutation = initial["sampler"].get("permutation")
    if permutation is None or initial["sampler"].get("epoch_index") != epoch:
        permutation = distributed.epoch_permutation(len(dataset), cfg["training_seed"], epoch)
    comparisons = []
    for _ in range(2):
        load_checkpoint(checkpoint, adapter, optimizer, scheduler, env, identities)
        adapter.train()
        metrics = train_update(adapter, optimizer, scheduler, dataset, permutation,
                               progress["global_batch_cursor"], cfg, epoch, env, bridge, calibration)
        comparisons.append(clone_nested({"adapter": distributed.unwrap(adapter).state_dict(),
                                         "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                         "metrics": metrics, "rng": rng_state()}))
    _assert_same_nested(comparisons[0], comparisons[1])
    load_checkpoint(checkpoint, adapter, optimizer, scheduler, env, identities)
    optimizer.zero_grad(set_to_none=True)
    value = {"status": "PASS", "checkpoint_sha256": sha256(checkpoint),
             "world_size": env.world_size, "global_step": progress["global_step"],
             "next_batch": progress["global_batch_cursor"], "epoch_index": epoch,
             "independent_full_loads": 2, "compared": ["parameters", "Adam", "scheduler", "loss", "RNG"],
             "trained_Adam_state": bool(initial["optimizer"]["state"]),
             "actual_transition_scene_count": comparisons[0]["metrics"]["transition_scene_count"],
             "transition_loss_active": comparisons[0]["metrics"]["transition_scene_count"] > 0}
    if out is not None and env.rank == 0:
        atomic_json(out, value)
    distributed.barrier()
    return value


def train_update(adapter, optimizer, scheduler, dataset, permutation, batch_index,
                 cfg, epoch, env, bridge, calibration):
    from .losses import scene_loss, backward_deterministically
    slots, real_count = distributed.global_batch_slots(permutation, batch_index, env.rank, env.world_size,
                                                      cfg["training"]["global_batch_size"])
    optimizer.zero_grad(set_to_none=True)
    update = epoch * cfg["training"]["updates_per_epoch"] + batch_index
    scalar_sums = {"loss": 0., "h": 0., "mu": 0., "zero_h": 0., "repair_h": 0.,
                   "zero_count": 0., "repair_count": 0., "actual_rms": 0., "cap_count": 0., "transition_count": 0.}
    group_sums = {}
    step_sums = [[0., 0.] for _ in range(5)]
    local_cfg = dict(cfg, epoch_index=epoch)
    for microstep, (index, real) in enumerate(slots):
        item = dataset[index]
        batch = prepare_scene(item, cfg, epoch, env.device)
        # no_sync must cover both forward and backward.
        with distributed.synchronization(adapter, microstep == len(slots) - 1):
            loss, stats = scene_loss(batch, adapter, bridge, local_cfg, update, calibration)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite scene loss at update {update}, token {item['metadata']['token']}")
            backward_deterministically(loss * (env.world_size / real_count if real else 0.))
        if real:
            h_value = float(stats.get("h", stats.get("loss_h", 0.)))
            mu_value = float(stats.get("mu", stats.get("loss_mu", 0.)))
            zero = bool(item["metadata"].get("target_is_zero", not bool(item["teacher_actual_shift"].any())))
            scalar_sums["loss"] += float(loss.detach())
            scalar_sums["h"] += h_value
            scalar_sums["mu"] += mu_value
            scalar_sums["zero_h" if zero else "repair_h"] += h_value
            scalar_sums["zero_count" if zero else "repair_count"] += 1
            scalar_sums["actual_rms"] += float(stats.get("actual_rms", 0.))
            scalar_sums["cap_count"] += float(stats.get("cap_triggered", 0.))
            scalar_sums["transition_count"] += int(stats["transition_executed"])
            source = str(item["metadata"].get("original_choice", "unknown"))
            group = group_sums.setdefault(source, [0., 0.])
            group[0] += h_value
            group[1] += 1
            if "time_index" in batch:
                step_sums[batch["time_index"]][0] += mu_value
                step_sums[batch["time_index"]][1] += 1
    grad_norm = torch.nn.utils.clip_grad_norm_(distributed.unwrap(adapter).parameters(), cfg["optimizer"]["grad_clip_norm"],
                                               error_if_nonfinite=True)
    used_lr = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    keys = list(scalar_sums)
    reduced = distributed.sum_tensor(list(scalar_sums.values()), env)
    totals = dict(zip(keys, reduced))
    result = {key: value / real_count for key, value in totals.items()
              if key not in ("zero_count", "repair_count", "zero_h", "repair_h", "transition_count")}
    result.update(zero_h=totals["zero_h"] / max(1, totals["zero_count"]),
                  repair_h=totals["repair_h"] / max(1, totals["repair_count"]),
                  zero_count=int(totals["zero_count"]), repair_count=int(totals["repair_count"]),
                  transition_scene_count=int(totals["transition_count"]),
                  grad_norm=float(grad_norm), grad_clip_ratio=min(1., cfg["optimizer"]["grad_clip_norm"] / max(float(grad_norm), 1e-12)),
                  nonfinite_gradients=0, lr=used_lr, real_scenes=real_count)
    # Choice is used only for retrospective statistics, never for loss weighting.
    if env.world_size > 1:
        import torch.distributed as dist
        all_groups = [None] * env.world_size
        dist.all_gather_object(all_groups, group_sums)
    else:
        all_groups = [group_sums]
    merged = {}
    for groups in all_groups:
        for key, values in groups.items():
            target = merged.setdefault(key, [0., 0.])
            target[0] += values[0]
            target[1] += values[1]
    result["source_groups"] = merged
    result["mu_by_step"] = np.asarray(distributed.sum_tensor(np.asarray(step_sums).reshape(-1).tolist(), env)).reshape(5, 2).tolist()
    return result


def run(cfg, out, resume="auto", formal=True):
    from .data import SceneDataset
    validate_config(cfg, formal=formal)
    if formal:
        verify_formal_config(cfg)
        verification = verify_qualification(cfg)
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("Formal training requires the qualified startup CUBLAS_WORKSPACE_CONFIG=:4096:8")
    env = distributed.initialize()
    torch.set_num_threads(int(cfg.get("runtime", {}).get("torch_threads", 1)))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(cfg["training_seed"])
    np.random.seed(cfg["training_seed"] % 2**32)
    torch.manual_seed(cfg["training_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["training_seed"])
    adapter, calibration, bundle = make_adapter(cfg, env.device)
    optimizer, scheduler, groups = make_optimizer(adapter, cfg)
    dataset = SceneDataset(dataset_manifest(cfg), "train", cfg["teacher_target"])
    if len(dataset) != cfg["data"]["train_scenes"]:
        raise ValueError("The complete Train population must be present")
    identities = identities_for(cfg)
    if formal:
        identities["qualification"] = verification
    parameters = {"trainable": [{"name": name, "shape": list(value.shape), "count": value.numel()}
                                for name, value in adapter.named_parameters() if value.requires_grad],
                  "optimizer_groups": groups, "frozen_head": True, "frozen_vlm": True}
    if env.rank == 0:
        (out / "effective_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        for filename, value in (("PROTOCOL.json", cfg["protocol"]), ("ASSET_IDENTITIES.json", identities),
                                ("CALIBRATION.json", calibration), ("TRAINABLE_PARAMETERS.json", parameters),
                                ("TARGET_IDENTITY.json", {"target": cfg["teacher_target"], "identity": identities["files"]["targets_manifest"]})):
            atomic_json(out / filename, value)
    adapter = distributed.wrap_adapter(adapter, env)
    progress = {"global_step": 0, "epoch_index": 0, "global_batch_cursor": 0, "real_scenes_seen": 0}
    sampler = {}
    checkpoint = latest_complete(out / "checkpoints") if resume == "auto" else (Path(resume) if resume not in (None, "none") else None)
    bridge = make_bridge(cfg, env.device, out) if cfg["enable_transition_loss"] else None
    if checkpoint:
        state = load_checkpoint(checkpoint, adapter, optimizer, scheduler, env, identities)
        progress, sampler = state["progress"], state["sampler"]
    if not (out / "INITIAL_STEP_RECOVERY_CHECK.json").exists() and progress["epoch_index"] < cfg["training"]["epochs"]:
        if checkpoint is None:
            checkpoint = out / "checkpoints/initial.pt"
            save_checkpoint(checkpoint, adapter=adapter, optimizer=optimizer, scheduler=scheduler,
                            progress=progress, sampler=sampler, identities=identities, config=cfg,
                            calibration=calibration, parameter_groups=groups, env=env)
        verify_next_update(checkpoint, adapter, optimizer, scheduler, dataset, cfg, env,
                           bridge, calibration, identities, out / "INITIAL_STEP_RECOVERY_CHECK.json")
    start = time.monotonic()
    interval = cfg["checkpoint"]["interval_updates"]
    first_epoch = progress["epoch_index"]
    for epoch in range(first_epoch, cfg["training"]["epochs"]):
        permutation = sampler.get("permutation") if sampler.get("epoch_index") == epoch else None
        if permutation is None:
            permutation = distributed.epoch_permutation(len(dataset), cfg["training_seed"], epoch)
        cursor = progress["global_batch_cursor"] if epoch == first_epoch else 0
        adapter.train()
        for batch_index in range(cursor, cfg["training"]["updates_per_epoch"]):
            step_start = time.monotonic()
            metrics = train_update(adapter, optimizer, scheduler, dataset, permutation, batch_index,
                                   cfg, epoch, env, bridge, calibration)
            progress.update(global_step=progress["global_step"] + 1, epoch_index=epoch,
                            global_batch_cursor=batch_index + 1,
                            real_scenes_seen=progress["real_scenes_seen"] + metrics["real_scenes"])
            sampler = {"epoch_index": epoch, "permutation": permutation, "next_global_batch": batch_index + 1}
            elapsed = time.monotonic() - step_start
            record = dict(metrics, **progress, epoch=epoch + 1, time=stamp(), step_seconds=elapsed,
                          elapsed_seconds=time.monotonic() - start,
                          eta_seconds=elapsed * (cfg["training"]["total_optimizer_updates"] - progress["global_step"]))
            if env.rank == 0:
                append_jsonl(out / "logs/train_metrics.jsonl", record)
                atomic_json(out / "TRAIN_STATE.json", dict(status="TRAINING", **record))
                if progress["global_step"] % 10 == 0 or batch_index == 0:
                    print(json.dumps({key: record[key] for key in ("epoch", "global_step", "loss", "h", "mu", "step_seconds", "eta_seconds")}), flush=True)
            if progress["global_step"] % interval == 0:
                recovery_checkpoint = out / "checkpoints" / f"update_{progress['global_step']:06d}.pt"
                save_checkpoint(recovery_checkpoint,
                                adapter=adapter, optimizer=optimizer, scheduler=scheduler, progress=progress,
                                sampler=sampler, identities=identities, config=cfg, calibration=calibration,
                                parameter_groups=groups, env=env)
                if progress["global_step"] < cfg["training"]["total_optimizer_updates"]:
                    check_path = out / "NEXT_UPDATE_TRAINED_ADAM_RECOVERY_CHECK.json"
                    if not check_path.exists():
                        verify_next_update(recovery_checkpoint, adapter, optimizer, scheduler, dataset, cfg, env,
                                           bridge, calibration, identities, check_path)
                    if cfg["enable_transition_loss"] and progress["global_step"] > 5364:
                        check_path = out / "NEXT_UPDATE_FUNCTIONAL_RECOVERY_CHECK.json"
                        if not check_path.exists():
                            verify_next_update(recovery_checkpoint, adapter, optimizer, scheduler, dataset, cfg, env,
                                               bridge, calibration, identities, check_path)
        progress.update(epoch_index=epoch + 1, global_batch_cursor=0)
        sampler = {"epoch_index": epoch + 1, "permutation": None, "next_global_batch": 0}
        endpoint = out / "checkpoints" / f"epoch_{epoch + 1:03d}.pt"
        save_checkpoint(endpoint, adapter=adapter, optimizer=optimizer, scheduler=scheduler, progress=progress,
                        sampler=sampler, identities=identities, config=cfg, calibration=calibration,
                        parameter_groups=groups, env=env)
        # last.pt always refers to the most recent complete epoch and has its own SHA marker.
        if env.rank == 0:
            import shutil
            last = out / "checkpoints/last.pt"
            temporary = last.with_suffix(".tmp")
            shutil.copyfile(endpoint, temporary)
            os.replace(temporary, last)
            atomic_json(str(last) + ".COMPLETE.json", json.loads(Path(str(endpoint) + ".COMPLETE.json").read_text()))
        distributed.barrier()
        full_validation = epoch + 1 in cfg["evaluation"]["full_validation_epochs"]
        monitor = (epoch + 1) % cfg["evaluation"]["monitor_every_epochs"] == 0
        if full_validation or monitor:
            from .evaluate import evaluate_model
            training_rng = rng_state()
            eval_bridge = bridge or make_bridge(cfg, env.device, out)
            destination = out / "validation" / (f"epoch_{epoch + 1:03d}" if full_validation else f"monitor_{epoch + 1:03d}")
            evaluate_model(distributed.unwrap(adapter), cfg, eval_bridge, "validation", destination,
                           cfg["evaluation"]["full_validation_noise_indices"] if full_validation else [0], env,
                           checkpoint=endpoint, subset_size=None if full_validation else cfg["evaluation"]["monitor_scenes"])
            if not cfg["enable_transition_loss"]:
                del eval_bridge
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            restore_rng(training_rng)
    if progress["global_step"] != cfg["training"]["total_optimizer_updates"] or progress["real_scenes_seen"] != len(dataset) * cfg["training"]["epochs"]:
        raise RuntimeError("The run did not finish its exact optimizer/exposure budget")
    # An epoch checkpoint is durable before evaluation. If a preemption occurred
    # between those two boundaries, finish that saved endpoint without changing
    # training exposure or requiring a fabricated TRAIN_COMPLETE marker.
    missing = None
    if env.rank == 0:
        missing = [epoch for epoch in range(2, 41, 2)
                   if not (out / "validation" / (f"epoch_{epoch:03d}" if epoch in (8, 16, 24, 32, 40)
                                                  else f"monitor_{epoch:03d}") / "COMPLETE.json").exists()]
    if env.world_size > 1:
        import torch.distributed as dist
        messages = [missing]
        dist.broadcast_object_list(messages, src=0)
        missing = messages[0]
    if missing:
        from .evaluate import evaluate_model
        final_parameters = clone_nested(distributed.unwrap(adapter).state_dict())
        eval_bridge = bridge or make_bridge(cfg, env.device, out)
        for epoch in missing:
            checkpoint = out / "checkpoints" / f"epoch_{epoch:03d}.pt"
            saved = read_checkpoint(checkpoint, identities)
            distributed.unwrap(adapter).load_state_dict(saved["adapter"], strict=True)
            full = epoch in (8, 16, 24, 32, 40)
            evaluate_model(distributed.unwrap(adapter), cfg, eval_bridge, "validation",
                           out / "validation" / (f"epoch_{epoch:03d}" if full else f"monitor_{epoch:03d}"),
                           [0, 1, 2, 3, 4] if full else [0], env, checkpoint=checkpoint,
                           subset_size=None if full else 2048)
        distributed.unwrap(adapter).load_state_dict(final_parameters, strict=True)
        if not cfg["enable_transition_loss"]:
            del eval_bridge
    if env.rank == 0:
        from .evaluate import verify_evaluation
        full_validation_evidence = {}
        for epoch in (8, 16, 24, 32, 40):
            folder = out / "validation" / f"epoch_{epoch:03d}"
            verify_evaluation(folder, {"population": "validation", "scene_count": 17475,
                              "noise_indices": [0, 1, 2, 3, 4], "artifact_kind": "behavior",
                              "checkpoint_sha256": sha256(out / "checkpoints" / f"epoch_{epoch:03d}.pt"),
                              "full_population": True, "mode": "student"})
            full_validation_evidence[str(epoch)] = sha256(folder / "COMPLETE.json")
        recovery = json.loads((out / "NEXT_UPDATE_TRAINED_ADAM_RECOVERY_CHECK.json").read_text())
        if recovery.get("status") != "PASS" or not recovery.get("trained_Adam_state"):
            raise ValueError("Actual trained-Adam next-update recovery is incomplete")
        if cfg["enable_transition_loss"]:
            recovery = json.loads((out / "NEXT_UPDATE_FUNCTIONAL_RECOVERY_CHECK.json").read_text())
            if recovery.get("status") != "PASS" or not recovery.get("trained_Adam_state") or not recovery.get("transition_loss_active"):
                raise ValueError("Actual active C/D next-update recovery is incomplete")
        atomic_json(out / "TRAIN_COMPLETE.json", {"status": "TRAIN_COMPLETE", **progress, "world_size": env.world_size,
                                                  "wall_seconds_this_process": time.monotonic() - start,
                                                  "full_validation_evidence": full_validation_evidence,
                                                  "qualification": verification if formal else None})
    distributed.barrier()
    return progress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--common", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--resume", default="auto")
    args = parser.parse_args()
    cfg = load_config(args.config, args.common)
    if args.training_seed is not None:
        cfg["training_seed"] = args.training_seed
    run(cfg, args.out, args.resume)


if __name__ == "__main__":
    main()
