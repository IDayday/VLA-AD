from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import tempfile
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from .stage3_reference_cache import Stage3ReferenceCache


class OfficialNAVSIMV2MetricEvaluator:
    """Persistent isolated bridge to the official NAVSIM v2 scorer."""

    def __init__(self, cfg: Any, metric_cache_path: str) -> None:
        root = Path(str(cfg.v2_official_navsim_root)).expanduser().resolve()
        if not (root / "navsim" / "evaluate" / "pdm_score.py").is_file():
            raise FileNotFoundError(
                "NAVSIM v2 LFP requires an official v2 source checkout via "
                f"lfp_grpo_cfg.v2_official_navsim_root; got {root}."
            )
        config_dir_value = str(getattr(cfg, "v2_official_config_dir", "") or "")
        config_dir = (
            Path(config_dir_value).expanduser().resolve()
            if config_dir_value
            else root / "navsim" / "planning" / "script" / "config" / "pdm_scoring"
        )
        self.timeout_s = float(getattr(cfg, "v2_evaluator_timeout_s", 600.0))
        self._temporary = tempfile.TemporaryDirectory(prefix="lfp-v2-worker-")
        temporary_root = Path(self._temporary.name)
        self._socket_path = str(temporary_root / "metric.sock")
        self._authkey = secrets.token_bytes(32)
        self._listener = Listener(self._socket_path, family="AF_UNIX", authkey=self._authkey)
        listener_socket = getattr(getattr(self._listener, "_listener", None), "_socket", None)
        if listener_socket is not None:
            listener_socket.settimeout(self.timeout_s)
        repo_root = Path(__file__).resolve().parents[3]
        worker = repo_root / "scripts" / "stage3" / "lfp_v2_metric_worker.py"
        self._log_handle = (temporary_root / "worker.log").open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(worker),
            "--socket",
            self._socket_path,
            "--authkey",
            self._authkey.hex(),
            "--navsim-root",
            str(root),
            "--config-dir",
            str(config_dir),
            "--config-name",
            str(getattr(cfg, "v2_official_config_name", "default_run_pdm_score")),
            "--metric-cache-path",
            str(Path(metric_cache_path).expanduser().resolve()),
            "--overrides-json",
            json.dumps(list(getattr(cfg, "v2_official_overrides", ()))),
        ]
        self._process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=os.environ.copy(),
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
        )
        try:
            self._connection = self._listener.accept()
            if not self._connection.poll(self.timeout_s):
                raise TimeoutError("Timed out waiting for official NAVSIM v2 scorer startup.")
            ready = self._connection.recv()
            if not ready.get("ready"):
                raise RuntimeError(f"Official NAVSIM v2 scorer failed to initialize: {ready}")
        except Exception:
            self.close()
            raise

    def score(
        self,
        trajectories: torch.Tensor,
        tokens: list[str],
        reference_cache: Stage3ReferenceCache,
    ) -> Dict[str, torch.Tensor]:
        if len(tokens) != trajectories.shape[0]:
            raise ValueError("NAVSIM v2 evaluator token/trajectory batch mismatch.")
        previous_tokens = []
        previous_trajectories = []
        for token in tokens:
            record = reference_cache.records[str(token)]
            previous = record.get("previous_token")
            previous_trajectory = record.get("previous_stage2_trajectory")
            if not previous or previous_trajectory is None:
                raise KeyError(
                    f"LFP v2 reference for token {token!r} lacks a coherent previous Stage2 trajectory "
                    "required by official extended comfort."
                )
            previous_tokens.append(str(previous))
            previous_trajectories.append(previous_trajectory)
        self._connection.send(
            {
                "command": "score",
                "trajectories": trajectories.detach().float().cpu().numpy(),
                "tokens": [str(token) for token in tokens],
                "previous_tokens": previous_tokens,
                "previous_trajectories": np.asarray(previous_trajectories, dtype=np.float32),
            }
        )
        if not self._connection.poll(self.timeout_s):
            raise TimeoutError("Official NAVSIM v2 rollout scoring timed out.")
        response = self._connection.recv()
        if "error" in response:
            detail = response.get("traceback", response["error"])
            raise RuntimeError(f"Official NAVSIM v2 rollout scoring failed:\n{detail}")
        return {
            key: torch.as_tensor(value, device=trajectories.device, dtype=trajectories.dtype)
            for key, value in response["metrics"].items()
        }

    def close(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            try:
                connection.send({"command": "close"})
                if connection.poll(5.0):
                    connection.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
            connection.close()
            self._connection = None
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        listener = getattr(self, "_listener", None)
        if listener is not None:
            listener.close()
            self._listener = None
        log_handle = getattr(self, "_log_handle", None)
        if log_handle is not None:
            log_handle.close()
            self._log_handle = None
        temporary = getattr(self, "_temporary", None)
        if temporary is not None:
            temporary.cleanup()
            self._temporary = None

    def __del__(self) -> None:
        self.close()
