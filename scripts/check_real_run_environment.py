#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check real ReCogDrive/NAVSIM run environment.")
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/project/VLA-AD"))
    parser.add_argument("--navsim-root", type=Path, default=Path("/mnt/navsim"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("/mnt/project/VLA-AD/checkpoints"))
    parser.add_argument("--require-modelscope", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/project/VLA-AD/experiments/recogdrive_expert"))
    return parser.parse_args()


def import_status(module: str, *, required: bool = True) -> Dict[str, Any]:
    try:
        imported = importlib.import_module(module)
        return {"module": module, "ok": True, "version": getattr(imported, "__version__", None), "required": required}
    except Exception as exc:
        return {"module": module, "ok": False, "error": repr(exc), "required": required}


def torch_status() -> Dict[str, Any]:
    status: Dict[str, Any] = import_status("torch")
    if not status["ok"]:
        return status
    import torch

    gpus = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gpus.append({
                "index": idx,
                "name": props.name,
                "total_memory_bytes": int(props.total_memory),
                "major": props.major,
                "minor": props.minor,
            })
    status.update({
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu_count": len(gpus),
        "gpus": gpus,
    })
    return status


def path_status(path: Path) -> Dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "is_dir": path.is_dir()}


def write_md(path: Path, report: Dict[str, Any]) -> None:
    lines = ["# Real Run Environment Check", ""]
    lines.append(f"Python: `{report['python']['version']}`")
    lines.append(f"Executable: `{report['python']['executable']}`")
    lines.append("")
    lines.append("## Torch / CUDA")
    lines.append("")
    torch = report["torch"]
    lines.append(f"- torch import: {torch.get('ok')}")
    lines.append(f"- torch version: {torch.get('version')}")
    lines.append(f"- CUDA available: {torch.get('cuda_available')}")
    lines.append(f"- GPU count: {torch.get('gpu_count')}")
    for gpu in torch.get("gpus", []):
        gb = gpu["total_memory_bytes"] / 1024**3
        lines.append(f"- GPU {gpu['index']}: {gpu['name']} ({gb:.2f} GiB)")
    lines.append("")
    lines.append("## Imports")
    lines.append("")
    for item in report["imports"]:
        status = "OK" if item["ok"] else "MISSING"
        required = "required" if item["required"] else "optional"
        err = f" - {item.get('error')}" if not item["ok"] else ""
        lines.append(f"- `{item['module']}`: {status} ({required}){err}")
    lines.append("")
    lines.append("## Paths")
    lines.append("")
    for key, item in report["paths"].items():
        lines.append(f"- {key}: `{item['path']}` exists={item['exists']} dir={item['is_dir']}")
    lines.append("")
    lines.append(f"Overall OK: {report['ok']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if str(args.project_root) not in sys.path:
        sys.path.insert(0, str(args.project_root))
    imports = [
        import_status("transformers"),
        import_status("huggingface_hub"),
        import_status("safetensors"),
        import_status("navsim"),
        import_status("nuplan"),
        import_status("vggt", required=False),
        import_status("modelscope", required=args.require_modelscope),
    ]
    paths = {
        "project_root": path_status(args.project_root),
        "navsim_root": path_status(args.navsim_root),
        "checkpoint_root": path_status(args.checkpoint_root),
        "cache_root": path_status(args.project_root / "cache"),
        "experiments_root": path_status(args.project_root / "experiments"),
    }
    report = {
        "python": {"version": platform.python_version(), "executable": sys.executable, "platform": platform.platform()},
        "torch": torch_status(),
        "imports": imports,
        "paths": paths,
    }
    required_imports_ok = all(item["ok"] or not item["required"] for item in imports)
    paths_ok = all(item["exists"] and item["is_dir"] for item in paths.values())
    report["ok"] = bool(report["torch"].get("ok") and required_imports_ok and paths_ok)
    args.output_root.mkdir(parents=True, exist_ok=True)
    json_path = args.output_root / "environment_check.json"
    md_path = args.output_root / "environment_check.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_md(md_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
