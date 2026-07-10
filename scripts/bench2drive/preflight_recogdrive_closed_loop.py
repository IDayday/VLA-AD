#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import urlopen
from xml.etree import ElementTree as ET

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight checks for ReCogDrive Bench2Drive closed-loop evaluation.")
    parser.add_argument("--bench2drive-root", type=Path, default=Path(os.environ.get("BENCH2DRIVE_ROOT", "")))
    parser.add_argument("--carla-root", type=Path, default=Path(os.environ.get("CARLA_ROOT", "")))
    parser.add_argument("--agent-config", type=Path, default=Path("configs/bench2drive_recogdrive_closed_loop.remote.yaml"))
    parser.add_argument("--routes", type=Path, default=None)
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--require-server", action="store_true")
    return parser.parse_args()


def add(result: List[Dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    result.append({"name": name, "ok": bool(ok), "detail": detail})


def import_check(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def count_routes(path: Path) -> Optional[int]:
    if not path.is_file():
        return None
    root = ET.parse(path).getroot()
    return len(root.findall(".//route"))


def server_health(url: str, timeout: float = 2.0) -> Dict[str, Any]:
    try:
        with urlopen(f"{url.rstrip('/')}/health", timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": str(exc)}


def main() -> int:
    args = parse_args()
    checks: List[Dict[str, Any]] = []

    bench2drive_root = args.bench2drive_root.resolve() if str(args.bench2drive_root) else Path("")
    carla_root = args.carla_root.resolve() if str(args.carla_root) else Path("")
    egg = carla_root / "PythonAPI" / "carla" / "dist" / "carla-0.9.15-py3.7-linux-x86_64.egg"
    for path in (
        carla_root / "PythonAPI",
        carla_root / "PythonAPI" / "carla",
        egg,
        bench2drive_root / "leaderboard",
        bench2drive_root / "scenario_runner",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    add(checks, "python_version", sys.version_info[:2] == (3, 8), f"{sys.version_info.major}.{sys.version_info.minor}")
    add(checks, "BENCH2DRIVE_ROOT", bench2drive_root.is_dir(), str(bench2drive_root))
    add(checks, "CARLA_ROOT", carla_root.is_dir(), str(carla_root))
    add(checks, "CarlaUE4.sh", (carla_root / "CarlaUE4.sh").is_file(), str(carla_root / "CarlaUE4.sh"))
    add(checks, "carla_egg", egg.is_file(), str(egg))

    evaluator = bench2drive_root / "leaderboard" / "leaderboard" / "leaderboard_evaluator.py"
    scenario_runner = bench2drive_root / "scenario_runner"
    add(checks, "leaderboard_evaluator", evaluator.is_file(), str(evaluator))
    add(checks, "scenario_runner", scenario_runner.is_dir(), str(scenario_runner))
    routes = args.routes or bench2drive_root / "leaderboard" / "data" / "bench2drive220.xml"
    route_count = count_routes(routes) if routes.is_file() else None
    add(checks, "routes_xml", routes.is_file(), f"{routes} routes={route_count}")
    add(checks, "weather_xml", (bench2drive_root / "leaderboard" / "data" / "weather.xml").is_file(), "leaderboard/data/weather.xml")

    for package in ("carla", "leaderboard", "srunner", "cv2", "numpy", "yaml", "requests", "scipy"):
        add(checks, f"import_{package}", import_check(package), package)

    cfg_path = args.agent_config
    add(checks, "agent_config", cfg_path.is_file(), str(cfg_path))
    server_url = args.server_url
    if cfg_path.is_file():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        server_url = server_url or cfg.get("server_url")
        add(checks, "agent_mode_remote", cfg.get("mode", "remote") == "remote", str(cfg.get("mode")))
    if server_url:
        health = server_health(str(server_url))
        add(checks, "recogdrive_server", bool(health.get("ok")), json.dumps(health, sort_keys=True))
    elif args.require_server:
        add(checks, "recogdrive_server", False, "server_url missing")

    ok = all(item["ok"] for item in checks if args.require_server or item["name"] != "recogdrive_server")
    print(json.dumps({"ok": ok, "checks": checks}, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
