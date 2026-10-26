"""Expanded configurations and an auditable, atomic eight-run registry."""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import yaml
from .checkpointing import atomic_json, digest, sha256

EXPERIMENTS = ("H_full", "H_gain1", "H_mu_teacher", "H_mu_student")
SEEDS = (20261027, 20261028)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def merge(left, right):
    result = copy.deepcopy(left)
    for key, value in right.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def load_config(path, common=None):
    path = Path(path)
    value = yaml.safe_load(path.read_text())
    if common:
        value = merge(yaml.safe_load(Path(common).read_text()), value)
    return value


def validate_config(cfg, formal=True):
    if cfg["protocol"].get("route_or_gate") is not False:
        raise ValueError("Only a shared continuous adapter is permitted")
    training = cfg["training"]
    count = cfg["data"]["train_scenes"]
    updates = math.ceil(count / training["global_batch_size"])
    if updates != training["updates_per_epoch"] or updates * training["epochs"] != training["total_optimizer_updates"]:
        raise ValueError("Training budget is inconsistent")
    if formal and (count, training["epochs"], training["global_batch_size"], updates) != (85813, 40, 64, 1341):
        raise ValueError("Formal Round1 must execute 85813 scenes x 40 epochs at batch64")
    if training["microbatch_per_device"] != 1:
        raise ValueError("Native scene microbatch must be one")
    experiment = cfg["experiment"]
    transition = experiment in ("H_mu_teacher", "H_mu_student")
    if bool(cfg["enable_transition_loss"]) != transition:
        raise ValueError("A/B must never run transition supervision")
    if cfg["loss"].get("additional_keep_weight", 0) != 0:
        raise ValueError("An additional keep loss is outside this protocol")
    if cfg["precision"].get("autocast") or cfg["precision"].get("tf32"):
        raise ValueError("The adapter and action head require native FP32")


def identities_for(cfg):
    identities = {"protocol_sha256": digest(cfg), "files": {}}
    paths = cfg["paths"]
    for key in ("teacher_bundle", "feature_manifest", "targets_manifest", "calibration"):
        value = paths.get(key)
        if value:
            path = Path(value)
            if not path.is_file():
                raise FileNotFoundError(f"Required verified asset {key} is not ready: {path}")
            identities["files"][key] = {"path": str(path.resolve()), "sha256": sha256(path)}
    root = Path(__file__).resolve().parents[1]
    code = {}
    for path in sorted((root / "student_distillation").glob("*.py")):
        code[str(path.relative_to(root))] = sha256(path)
    identities["code_sha256"] = digest(code)
    identities["code_files"] = code
    identities["code_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    return identities


class Registry:
    def __init__(self, out, experiments=EXPERIMENTS, seeds=SEEDS):
        self.path = Path(out) / "EXPERIMENT_REGISTRY.json"
        if self.path.exists():
            self.value = json.loads(self.path.read_text())
        else:
            self.value = {"schema": 1, "created": stamp(), "status": "WAITING_ASSET_QUALIFICATION", "runs": {},
                          "validation_teacher_diagnostics": {"status": "PENDING"}, "tasks": {}}
        for experiment in experiments:
            for seed in seeds:
                run = f"{experiment}_seed{seed}"
                self.value["runs"].setdefault(run, {"experiment": experiment, "training_seed": seed,
                                                   "status": "PENDING", "global_step": 0, "history": []})
        self.save()

    def save(self):
        self.value["updated"] = stamp()
        atomic_json(self.path, self.value)

    def run_update(self, run, status=None, **fields):
        value = self.value["runs"][run]
        if status and value.get("status") != status:
            value["history"].append({"time": stamp(), "status": status})
            value["status"] = status
        value.update(fields)
        self.save()

    def task_update(self, key, status, **fields):
        self.value["tasks"][key] = {"status": status, "updated": stamp(), **fields}
        self.save()
