"""Observation and seeded native-sampling helpers for the paired evaluation."""
import hashlib
import json
import os
from pathlib import Path

import torch

def run_paths():
    path = Path(os.environ["RECOGDRIVE_UNIFORM_EVAL_CONFIG"]).expanduser().resolve()
    raw = json.loads(path.read_text())
    return {key: (path.parent / Path(value).expanduser()).resolve() for key, value in raw.items() if value}

def make_agent_input(row):
    from navsim.common.dataclasses import AgentInput, Camera, Cameras, EgoStatus
    statuses = [
        EgoStatus(ego_pose=pose, ego_velocity=row["velocity"], ego_acceleration=row["acceleration"],
                  driving_command=row["command"])
        for pose in row["history"]
    ]
    cameras = [Cameras(**{name: Camera() for name in Cameras.__dataclass_fields__}) for _ in statuses]
    cameras[-1].cam_f0.image = Path(row["image"])
    return AgentInput(statuses, cameras, [])

def scene_seed(token, seed):
    return int.from_bytes(hashlib.sha256(f"recogdrive:{seed}:{token}".encode()).digest()[:8], "big") % 2**31

def state_hash(planner):
    digest = hashlib.sha256()
    for name, value in planner.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

class SamplingCaptured(Exception):
    """Stop the native GRPO call after sampling, before loading rewards."""

def draw(planner, hidden, action, token, group, protocol):
    encoded_seed = json.dumps([2026091501, token, "shared_sampling", group],
                              separators=(",", ":"), sort_keys=True).encode()
    torch.manual_seed(int(hashlib.sha256(encoded_seed).hexdigest()[:15], 16) % 2**63)
    if protocol == "eval":
        planner.eval()
        return planner.get_action(hidden.expand(16, -1, -1), action(16), deterministic=False)[
            "pred_traj"].cpu().numpy()
    if protocol != "native_grpo":
        raise ValueError(protocol)
    planner.train()
    planner.set_frozen_modules_to_eval_mode()
    original = planner.sample_chain
    captured = {}

    def sample(*args, **kwargs):
        _, captured["trajectory"] = original(*args, **kwargs)
        raise SamplingCaptured

    planner.sample_chain = sample
    try:
        planner.forward_grpo(hidden, action(1), [token], sample_time=16)
    except SamplingCaptured:
        pass
    finally:
        planner.sample_chain = original
    assert captured["trajectory"].shape == (16, 8, 3)
    return captured["trajectory"].cpu().numpy()
