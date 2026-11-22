"""Evaluate pinned checkpoints using one shared native Stage 1 encoding per scene."""
import argparse
import hashlib
import inspect
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

from eval_support import draw, make_agent_input, run_paths, scene_seed, state_hash

PATHS = run_paths()
SOURCE = Path(__file__).resolve().parents[3]
OUT = PATHS["output"]
NAV = PATHS["nav_manifest"]
VLM = PATHS["stage1"]
TRAIN = PATHS["train_args"]
FEATURE_ROOT = PATHS.get("shared_features", OUT / "stage1_features")

from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from navsim.agents.recogdrive.recogdrive_diffusion_planner import GRPOConfig
from navsim.agents.recogdrive.recogdrive_features import ReCogDriveFeatureBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

def read(path):
    return json.loads(path.read_text())

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def initialize():
    OUT.mkdir(parents=True, exist_ok=True)
    psi = read(PATHS["psi_models"])["psi_sft"]
    planner_hash = sha(SOURCE / "navsim/agents/recogdrive/recogdrive_diffusion_planner.py")
    assert planner_hash == psi["runtime_planner_sha256"]
    latest = PATHS["checkpoint"]
    checkpoint = torch.load(latest, map_location="cpu", mmap=True, weights_only=False)
    assert checkpoint["state_dict"]
    psi_path = PATHS["psi_checkpoint"]
    assert sha(psi_path) == psi["sha256"]
    models = [
        dict(name=latest.stem, file=str(latest), sha256=sha(latest),
             epoch_zero_based=checkpoint["epoch"], global_step=checkpoint["global_step"]),
        dict(name="psi_sft_epoch011", file=str(psi_path), sha256=psi["sha256"]),
    ]
    nav = read(NAV)
    nav_rows = nav["rows"]
    index = PATHS["nav_metric_index"]
    metric = {r["token"]: r for r in read(index)}
    assert len(nav_rows) == len(metric) == 12146
    assert {r["token"]: r["log"] for r in nav_rows} == {t: r["log"] for t, r in metric.items()}
    assert len({r["log"] for r in nav_rows}) == 136
    stage1 = nav["identity"]["stage1"]
    assert sha(VLM / "model.safetensors") == stage1["sha256"]
    assert sha(VLM / "config.json") == stage1["config_sha256"]
    diag = PATHS["diagnostic_manifest"]
    value = dict(models=models, nav_manifest=str(NAV), nav_manifest_sha256=sha(NAV),
                 nav_metric=str(index), nav_metric_sha256=sha(index),
                 diag_manifest=str(diag), diag_manifest_sha256=sha(diag),
                 planner_sha256=psi["runtime_planner_sha256"], stage1=nav["identity"]["stage1"],
                 nav_seed=42, nav_precision="BF16 Stage 1 and FP32 head",
                 diagnostic_precision="FP32 Stage 1 and FP32 head with TF32 disabled",
                 diagnostic_seed=2026091501, diagnostic_groups=4, group_size=16)
    reference = PATHS.get("psi_nav_reference")
    if reference is not None:
        previous = read(reference / "identity.json")
        reference_hash = sha(reference / "identity.json")
        previous_nav = read(reference / "nav/COMPLETE.json")
        assert previous_nav["paired_valid_scenes"] == previous_nav["expected_scenes"] == 12146
        assert previous_nav["identity_sha256"] == reference_hash
        assert previous["models"][1]["sha256"] == psi["sha256"]
        for key in ("nav_manifest_sha256", "nav_metric_sha256", "stage1", "nav_seed", "planner_sha256"):
            assert previous[key] == value[key], key
        value["nav_PSI_reference"] = dict(out=str(reference), identity_sha256=reference_hash)
    path = OUT / "identity.json"
    if path.exists():
        assert read(path) == value
    else:
        path.write_text(json.dumps(value, indent=2) + "\n")

def models_for(identity, phase, historical_psi=False):
    params = inspect.signature(ReCogDriveAgent.__init__).parameters
    psi = read(PATHS["psi_models"])["psi_sft"]
    train = read(TRAIN)["agent"]
    models = []
    for item in identity["models"][:1] if historical_psi else identity["models"]:
        config = dict(psi["config"] if item["name"].startswith("psi") else train)
        config.update(trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=.5),
                      checkpoint_path="", allow_random_init=True, cache_hidden_state=True,
                      cache_mode=False, dit_type="small", vlm_type="internvl", vlm_path="", grpo=False)
        model = ReCogDriveAgent(**{k: v for k, v in config.items() if k in params}).float()
        raw = torch.load(item["file"], map_location="cpu", weights_only=False, mmap=True)
        if "epoch" in raw:
            print("CHECKPOINT", item["name"], raw["epoch"], raw.get("global_step"), flush=True)
        assert all(k.startswith("agent.") for k in raw["state_dict"])
        model.load_state_dict({k[len("agent."):]: v for k, v in raw["state_dict"].items()}, strict=True)
        model.cuda().eval().requires_grad_(False)
        planner = model.action_head
        assert planner.ddim_steps == 5
        if phase == "diag":
            config = GRPOConfig()
            run = psi["grpo_config"]
            for key in vars(config):
                if "grpo_" + key in run:
                    setattr(config, key, run["grpo_" + key])
                elif key in run:
                    setattr(config, key, run[key])
            for key in ("denoised_clip_value", "eval_randn_clip_value", "randn_clip_value",
                        "final_action_clip_value", "eps_clip_value", "eval_min_sampling_denoising_std",
                        "min_sampling_denoising_std", "min_logprob_denoising_std", "gamma_denoising"):
                if hasattr(config, key):
                    setattr(planner, key, getattr(config, key))
            planner.grpo_sample_time = 16
            planner.use_trajectory_level_objective = True
            planner.use_gspo_ratio = bool(run.get("grpo_use_gspo_ratio", False))
            planner.behavior_policy_sample = bool(run.get("grpo_behavior_policy_sample", False))
            assert not planner.use_gspo_ratio
        models.append(model)
    return models

def diagnostic_inputs(f):
    hidden = f["last_hidden_state"].unsqueeze(0).cuda().float()
    history = f["history_trajectory"].reshape(1, -1).cuda().float()
    status = f["status_feature"].unsqueeze(0).cuda().float()
    command = f["high_command_one_hot"].unsqueeze(0).cuda().float()
    state = torch.cat([status, history], dim=1)

    def action(n):
        return BatchFeature(data=dict(his_traj=history.expand(n, -1),
                                      history_trajectory=history.reshape(1, 4, 3).expand(n, -1, -1),
                                      status_feature=status.expand(n, -1),
                                      high_command_one_hot=command.expand(n, -1), state=state.expand(n, -1)))
    return hidden, action

def publish(path, token, identity_hash, values):
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        np.savez(handle, token=token, identity_hash=identity_hash, **values)
    file_hash = sha(tmp)
    tmp.replace(path)
    record = path.with_suffix(".sha256")
    tmp_record = path.with_suffix(".sha256.tmp")
    tmp_record.write_text(file_hash + "\n")
    tmp_record.replace(record)


def build_vlm(phase):
    if phase == "diag":
        from transformers import AutoModel
        load_original = AutoModel.from_pretrained

        def fp32_load(*args, **kwargs):
            kwargs.update(torch_dtype=torch.float32, use_flash_attn=False)
            return load_original(*args, **kwargs)

        AutoModel.from_pretrained = fp32_load
    builder = ReCogDriveFeatureBuilder(
        cache_hidden_state=True, cache_mode=True, model_type="internvl",
        checkpoint_path=str(VLM), device="cuda:0", use_expert_features=False, use_jepa=False, use_vggt=False,
    )
    builder.backbone.eval().requires_grad_(False)
    assert next(builder.backbone.model.parameters()).dtype == (torch.float32 if phase == "diag" else torch.bfloat16)
    return builder

def diagnostic_features(builder, row, token, raw_logs):
    from navsim.common.dataclasses import AgentInput, SensorConfig
    log = row["log"]
    if log not in raw_logs:
        with (PATHS["navtrain_logs"] / f"{log}.pkl").open("rb") as handle:
            raw_logs[log] = pickle.load(handle)
    begin = row["frame_start"]
    frames = raw_logs[log][begin:begin + 4]
    assert len(frames) == 4 and frames[-1]["token"] == token
    sensor = SensorConfig(cam_f0=True, cam_l0=False, cam_l1=False, cam_r0=False,
                          cam_r1=False, cam_l2=False, cam_r2=False, cam_b0=False, lidar_pc=False)
    observation = AgentInput.from_scene_dict_list(
        frames, PATHS["navtrain_sensors"], 4, sensor, load_image_path=True,
    )
    return builder.compute_features(observation)

def fetch_features(path, identity, token, phase, build, read_only):
    if path.is_file():
        data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        assert data["identity"] == identity and data["token"] == token
        return {key: value.float() for key, value in data["features"].items()}
    if read_only:
        raise FileNotFoundError(path)
    features = build()
    packed = dict(features)
    if phase == "nav":
        hidden = features["last_hidden_state"]
        packed["last_hidden_state"] = hidden.bfloat16()
        assert torch.equal(hidden, packed["last_hidden_state"].float())
    temp = path.with_suffix(".tmp")
    torch.save(dict(identity=identity, token=token, features=packed), temp)
    temp.replace(path)
    return features

def work(args):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(0)
    identity = read(OUT / "identity.json")
    ident = sha(OUT / "identity.json")
    phase = args.phase
    models = models_for(identity, phase, args.historical_psi)
    builder = build_vlm(phase) if not args.read_features else None
    features_dir = FEATURE_ROOT / phase
    if not args.read_features:
        features_dir.mkdir(parents=True, exist_ok=True)
    feature_id = identity["stage1"]["sha256"] + ":" + phase + ":" + identity[
        "nav_manifest_sha256" if phase == "nav" else "diag_manifest_sha256"]
    raw_logs = {}
    if phase == "nav":
        rows = read(NAV)["rows"]
    else:
        rows = read(Path(identity["diag_manifest"]))["scenes"]
        before = [state_hash(m.action_head) for m in models]
    if args.token_plan:
        plan = read(args.token_plan)
        assert plan["identity_hash"] == ident and len(plan["shards"]) == args.world
        by_token = {row["token"]: row for row in rows}
        rows = [by_token[token] for token in plan["shards"][args.rank]]
    else:
        rows = rows[args.rank::args.world]
    if args.limit:
        rows = rows[:args.limit]
    directory = OUT / (phase + ("_smoke" if args.limit else "")) / "predictions"
    directory.mkdir(exist_ok=True, parents=True)
    started = time.monotonic()
    with torch.inference_mode():
        for i, row in enumerate(rows, 1):
            token = row["token"]
            path = directory / f"{token}.npz"
            if path.is_file() and path.with_suffix(".sha256").is_file():
                continue
            if phase == "nav":
                agent_input = make_agent_input(row)
                native_features = fetch_features(
                    features_dir / f"{token}.pt", feature_id, token, phase,
                    lambda: builder.compute_features(agent_input), args.read_features,
                )
                features = {k: v.unsqueeze(0).cuda() for k, v in native_features.items()}
                result = []
                for model in models:
                    torch.manual_seed(scene_seed(token, 42))
                    trajectory = model.forward(dict(features))["pred_traj"].float().cpu().squeeze(0).numpy()
                    result.append(trajectory)
                if i == 1 and args.limit and builder is not None:
                    for model, trajectory in zip(models, result):
                        model.get_feature_builders = lambda: [builder]
                        torch.manual_seed(scene_seed(token, 42))
                        native = model.compute_trajectory(agent_input).poses
                        assert np.array_equal(trajectory, np.asarray(native))
                if args.historical_psi:
                    reference = Path(identity["nav_PSI_reference"]["out"])
                    prior = reference / "nav/predictions" / f"{token}.npz"
                    assert sha(prior) == prior.with_suffix(".sha256").read_text().strip()
                    with np.load(prior, allow_pickle=False) as payload:
                        assert str(payload["token"]) == token
                        assert str(payload["identity_hash"]) == identity["nav_PSI_reference"]["identity_sha256"]
                        assert payload["trajectories"].shape == (2, 8, 3)
                        result.append(payload["trajectories"][1].copy())
                values = dict(trajectories=np.stack(result))
            else:
                features = fetch_features(
                    features_dir / f"{token}.pt", feature_id, token, phase,
                    lambda: diagnostic_features(builder, row, token, raw_logs), args.read_features,
                )
                hidden, action = diagnostic_inputs(features)
                values = {protocol: np.stack([
                    np.stack([draw(model.action_head, hidden, action, token, g, protocol)
                              for g in range(4)]) for model in models]) for protocol in ("eval", "native_grpo")}
            assert all(np.isfinite(v).all() and v.shape[-2:] == (8, 3) for v in values.values())
            publish(path, token, ident, values)
            if i == 1 or i % 25 == 0:
                print("PROGRESS", phase, args.rank, i, len(rows), round(time.monotonic()-started, 2), flush=True)
    if phase == "diag":
        assert before == [state_hash(m.action_head) for m in models]
        print("STATE_UNCHANGED", json.dumps(before), flush=True)
    print("COMPLETE", phase, args.rank, len(rows), round(time.monotonic()-started, 2),
          "peak_mib", torch.cuda.max_memory_reserved() // 2**20, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["init", "nav", "diag"])
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--token-plan", type=Path)
    parser.add_argument("--historical-psi", action="store_true")
    parser.add_argument("--read-features", action="store_true")
    args = parser.parse_args()
    initialize() if args.phase == "init" else work(args)
