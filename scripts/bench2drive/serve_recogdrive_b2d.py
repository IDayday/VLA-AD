#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from transformers.feature_extraction_utils import BatchFeature
except ModuleNotFoundError:
    from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

    _install_dependency_stubs()
    from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.recogdrive_features import ReCogDriveFeatureBuilder
from navsim.common.dataclasses import AgentInput, Camera, Cameras, EgoStatus, Lidar
from scripts.train_recogdrive_expert_chunked import build_planner, shape_safe_load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve ReCogDrive trajectory predictions for Bench2Drive closed-loop.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--config", type=Path, default=Path("configs/bench2drive_recogdrive_il.yaml"))
    parser.add_argument("--planner-checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, default=Path("checkpoints/recogdrive/ReCogDrive-VLM-2B"))
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--init-action-mode", choices=("token_noise", "zeros", "random"), default="token_noise")
    parser.add_argument("--init-action-seed", type=int, default=20260709)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile-every", type=int, default=50)
    parser.add_argument("--visual-cache-limit", type=int, default=512)
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dtype_from_precision(precision: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]


def make_planner_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        vlm_feature_dim=None,
        precision=args.precision,
        jepa_align_weight=None,
        vggt_align_weight=None,
        diffusion_loss_weight=None,
        vlm_adapter_type=None,
        vlm_adapter_dropout=None,
        use_action_aware_aux=None,
        action_aware_aux_weight=None,
        action_aware_aux_space=None,
        expert_gate_init=None,
    )


def token_seed(token: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def make_init_actions(
    sample_tokens: Sequence[str],
    *,
    mode: str,
    seed: int,
    horizon: int,
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    shape = (len(sample_tokens), horizon, action_dim)
    if mode == "random":
        return None
    if mode == "zeros":
        return torch.zeros(shape, device=device, dtype=dtype)
    rows = []
    for token in sample_tokens:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(token_seed(token, seed))
        rows.append(torch.randn((horizon, action_dim), generator=generator, dtype=torch.float32))
    return torch.stack(rows, dim=0).to(device=device, dtype=dtype)


def batch_to_device(batch: BatchFeature, *, device: torch.device, dtype: torch.dtype) -> BatchFeature:
    return BatchFeature(
        data={
            key: (value.to(device=device, dtype=dtype) if isinstance(value, torch.Tensor) else value)
            for key, value in batch.items()
        }
    )


def build_agent_input(payload: Dict[str, Any]) -> AgentInput:
    image_path = Path(str(payload["image_path"]))
    if not image_path.is_file():
        raise FileNotFoundError(f"Front camera image not found: {image_path}")
    history = torch.as_tensor(payload["history_trajectory"], dtype=torch.float32)
    high_command = torch.as_tensor(payload["high_command_one_hot"], dtype=torch.float32)
    status = torch.as_tensor(payload["status_feature"], dtype=torch.float32)
    empty_camera = Camera()
    cameras = Cameras(
        cam_f0=Camera(image=image_path),
        cam_l0=empty_camera,
        cam_l1=empty_camera,
        cam_l2=empty_camera,
        cam_r0=empty_camera,
        cam_r1=empty_camera,
        cam_r2=empty_camera,
        cam_b0=empty_camera,
    )
    ego_statuses = [
        EgoStatus(
            ego_pose=row.detach().cpu().numpy().astype("float32"),
            ego_velocity=status[3:5].detach().cpu().numpy().astype("float32"),
            ego_acceleration=status[5:8].detach().cpu().numpy().astype("float32"),
            driving_command=high_command.detach().cpu().numpy().astype("float32"),
        )
        for row in history
    ]
    return AgentInput(
        ego_statuses=ego_statuses,
        cameras=[cameras for _ in ego_statuses],
        lidars=[Lidar() for _ in ego_statuses],
        token=str(payload.get("sample_token", image_path.stem)),
        log_name="bench2drive_closed_loop",
        scene_token="bench2drive_closed_loop",
    )


class ReCogDriveB2DPredictor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.cfg = load_yaml(args.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype_from_precision(args.precision) if self.device.type == "cuda" else torch.float32
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.feature_builder = ReCogDriveFeatureBuilder(
            cache_hidden_state=True,
            model_type="internvl",
            checkpoint_path=str(args.vlm_path),
            device=str(self.device),
            cache_mode=True,
            use_expert_features=False,
        )
        self.planner = build_planner(self.cfg, make_planner_args(args)).to(self.device)
        if self.dtype != torch.float32:
            self.planner = self.planner.to(dtype=self.dtype)
        self.load_report = shape_safe_load(self.planner, args.planner_checkpoint, strict_original=False)
        self.planner.eval()
        self.horizon = int(getattr(self.planner.config, "action_horizon", 8))
        self.action_dim = int(getattr(self.planner.config, "action_dim", 3))
        self.visual_cache: Dict[str, torch.Tensor] = {}
        self.profile_count = 0
        self.profile_totals: Dict[str, float] = {}
        print(
            json.dumps(
                {
                    "event": "recogdrive_b2d_server_ready",
                    "device": str(self.device),
                    "dtype": str(self.dtype).replace("torch.", ""),
                    "config": str(args.config),
                    "planner_checkpoint": str(args.planner_checkpoint),
                    "vlm_path": str(args.vlm_path),
                    "load_report": self.load_report,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        required = ("history_trajectory", "high_command_one_hot", "status_feature")
        for key in required:
            if key not in payload:
                raise KeyError(f"Request missing required key: {key}")
        start = time.monotonic()
        timings: Dict[str, float] = {}
        sample_token = str(payload.get("sample_token") or Path(str(payload.get("image_path", "frame"))).stem)
        visual_cache_key = str(payload.get("visual_cache_key") or sample_token)
        image_path_value = payload.get("image_path")
        visual_refreshed = bool(image_path_value)

        if image_path_value:
            t0 = time.monotonic()
            agent_input = build_agent_input(payload)
            timings["build_agent_input"] = time.monotonic() - t0
            t0 = time.monotonic()
            with torch.inference_mode():
                features = self.feature_builder.compute_features(agent_input)
            timings["visual_features"] = time.monotonic() - t0
            hidden_state = features["last_hidden_state"].detach()
            self.visual_cache[visual_cache_key] = hidden_state
            if len(self.visual_cache) > int(self.args.visual_cache_limit):
                self.visual_cache.pop(next(iter(self.visual_cache)))
        else:
            if visual_cache_key not in self.visual_cache:
                raise KeyError(
                    f"No cached visual features for key '{visual_cache_key}'. "
                    "Send image_path on the first request or reduce visual_refresh_interval_steps."
                )
            hidden_state = self.visual_cache[visual_cache_key]

        t0 = time.monotonic()
        last_hidden_state = hidden_state.unsqueeze(0).to(device=self.device, dtype=self.dtype)
        history = torch.as_tensor(payload["history_trajectory"], dtype=torch.float32)
        high_command = torch.as_tensor(payload["high_command_one_hot"], dtype=torch.float32)
        status = torch.as_tensor(payload["status_feature"], dtype=torch.float32)
        action_input = BatchFeature(
            data={
                "his_traj": history.reshape(1, -1),
                "history_trajectory": history.unsqueeze(0),
                "high_command_one_hot": high_command.unsqueeze(0),
                "status_feature": status.unsqueeze(0),
            }
        )
        action_input = batch_to_device(action_input, device=self.device, dtype=self.dtype)
        init_actions = make_init_actions(
            [sample_token],
            mode=self.args.init_action_mode,
            seed=self.args.init_action_seed,
            horizon=self.horizon,
            action_dim=self.action_dim,
            device=self.device,
            dtype=self.dtype,
        )
        timings["build_planner_input"] = time.monotonic() - t0
        t0 = time.monotonic()
        with torch.inference_mode():
            output = self.planner.get_action(
                last_hidden_state,
                action_input,
                init_actions=init_actions,
                deterministic=bool(self.args.deterministic),
            )
        timings["planner"] = time.monotonic() - t0
        trajectory = output["pred_traj"].detach().float().cpu().squeeze(0)
        if trajectory.shape != (8, 3):
            raise RuntimeError(f"Planner returned shape {tuple(trajectory.shape)}, expected (8, 3).")
        if not torch.isfinite(trajectory).all():
            raise RuntimeError("Planner returned non-finite trajectory.")
        timings["total"] = time.monotonic() - start
        self._record_profile(timings)
        return {
            "trajectory": trajectory.tolist(),
            "elapsed_seconds": timings["total"],
            "sample_token": sample_token,
            "visual_refreshed": visual_refreshed,
            "timings": timings,
        }

    def _record_profile(self, timings: Dict[str, float]) -> None:
        self.profile_count += 1
        for key, value in timings.items():
            self.profile_totals[key] = self.profile_totals.get(key, 0.0) + float(value)
        if self.args.profile_every <= 0 or self.profile_count % self.args.profile_every != 0:
            return
        averages = {key: value / self.profile_count for key, value in sorted(self.profile_totals.items())}
        print(
            json.dumps(
                {
                    "event": "recogdrive_b2d_profile",
                    "requests": self.profile_count,
                    "avg_seconds": averages,
                    "visual_cache_entries": len(self.visual_cache),
                },
                sort_keys=True,
            ),
            flush=True,
        )


class Handler(BaseHTTPRequestHandler):
    predictor: ReCogDriveB2DPredictor

    def _send(self, status: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, {"ok": True, "device": str(self.predictor.device)})
            return
        self._send(404, {"ok": False, "error": "unknown endpoint"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/predict":
            self._send(404, {"ok": False, "error": "unknown endpoint"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = self.predictor.predict(payload)
            self._send(200, result)
        except Exception as exc:  # keep CARLA client error visible
            traceback.print_exc()
            self._send(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[ReCogDriveB2DServer] {self.address_string()} - {format % args}", flush=True)


def main() -> int:
    args = parse_args()
    Handler.predictor = ReCogDriveB2DPredictor(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"event": "http_listen", "host": args.host, "port": args.port}, sort_keys=True), flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
