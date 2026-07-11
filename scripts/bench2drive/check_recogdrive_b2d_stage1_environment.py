#!/usr/bin/env python3
"""Validate the locked, closest-public software stack used for B2D Stage1."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Sequence


# ReCogDrive's vendored InternVL declares exact versions for most of these
# packages. Torch, torchvision, accelerate and FlashAttention are public
# compatibility locks chosen for this reproduction, not author-confirmed pins.
EXPECTED_PACKAGES = {
    "torch": "2.2.2",
    "torchvision": "0.17.2",
    "transformers": "4.37.2",
    "tokenizers": "0.15.1",
    "accelerate": "0.28.0",
    "deepspeed": "0.13.5",
    "timm": "0.9.12",
    "sentencepiece": "0.1.99",
    "einops": "0.6.1",
    "einops-exts": "0.0.4",
    "peft": "0.10.0",
    "flash-attn": "2.5.8",
    "numpy": "1.26.4",
    "opencv-python": "4.9.0.80",
    "Pillow": "10.3.0",
    "decord": "0.6.0",
    "imageio": "2.34.1",
    "orjson": "3.10.3",
    "bitsandbytes": "0.42.0",
    "huggingface-hub": "0.23.4",
    "safetensors": "0.4.3",
    "tensorboardX": "2.6.2.2",
    "pydantic": "2.6.4",
    "hjson": "3.1.0",
    "psutil": "5.9.8",
    "py-cpuinfo": "9.0.0",
    "nvidia-nvjitlink-cu12": "12.1.105",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-gpus", type=int, default=1)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="Diagnostic-only escape hatch; mismatches remain recorded in the report.",
    )
    return parser.parse_args(argv)


def normalize_version(version: str) -> str:
    """Remove a local build suffix such as ``+cu121`` before comparison."""

    return version.split("+", 1)[0]


def package_versions() -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}
    for distribution, expected in EXPECTED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(distribution)
            error = None
        except importlib.metadata.PackageNotFoundError:
            actual = None
            error = "not installed"
        checks[distribution] = {
            "actual": actual,
            "expected": expected,
            "ok": actual is not None and normalize_version(actual) == expected,
            "error": error,
        }
    return checks


def _import_check(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        return {
            "ok": True,
            "module": module_name,
            "path": str(getattr(module, "__file__", "")),
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - depends on the CUDA runtime
        return {
            "ok": False,
            "module": module_name,
            "path": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_checks(min_gpus: int, allow_version_mismatch: bool) -> dict[str, Any]:
    versions = package_versions()
    pip_check_process = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        check=False,
        text=True,
    )
    pip_check = {
        "ok": pip_check_process.returncode == 0,
        "returncode": pip_check_process.returncode,
        "stdout": pip_check_process.stdout.strip(),
        "stderr": pip_check_process.stderr.strip(),
    }
    imports = {
        "deepspeed": _import_check("deepspeed"),
        "flash_attn": _import_check("flash_attn"),
    }

    cuda: dict[str, Any] = {
        "available": False,
        "device_count": 0,
        "minimum_required": min_gpus,
        "bf16_supported": [],
        "devices": [],
        "nccl_available": False,
        "flash_attention_forward": False,
        "flash_attention_error": None,
    }
    try:
        import torch

        cuda["torch_version"] = torch.__version__
        cuda["torch_cuda_version"] = torch.version.cuda
        cuda["available"] = bool(torch.cuda.is_available())
        cuda["device_count"] = int(torch.cuda.device_count())
        cuda["nccl_available"] = bool(torch.distributed.is_nccl_available())
        if cuda["available"]:
            cuda["devices"] = [torch.cuda.get_device_name(i) for i in range(cuda["device_count"])]
            cuda["bf16_supported"] = [
                torch.cuda.get_device_capability(i)[0] >= 8 for i in range(cuda["device_count"])
            ]
        if imports["flash_attn"]["ok"] and cuda["available"]:
            from flash_attn import flash_attn_func

            q = torch.randn(1, 16, 2, 64, device="cuda:0", dtype=torch.bfloat16)
            out = flash_attn_func(q, q, q, dropout_p=0.0, causal=True)
            cuda["flash_attention_forward"] = bool(
                out.shape == q.shape and torch.isfinite(out).all().item()
            )
            del q, out
            torch.cuda.empty_cache()
    except Exception as exc:  # pragma: no cover - depends on the CUDA runtime
        cuda["flash_attention_error"] = f"{type(exc).__name__}: {exc}"

    versions_ok = all(item["ok"] for item in versions.values())
    python_ok = sys.version_info[:2] == (3, 9)
    cuda_runtime_ok = cuda.get("torch_cuda_version") == "12.1"
    cuda["expected_torch_cuda_version"] = "12.1"
    cuda["torch_cuda_version_ok"] = cuda_runtime_ok
    cuda_ok = (
        cuda["available"]
        and cuda["device_count"] >= min_gpus
        and cuda["nccl_available"]
        and len(cuda["bf16_supported"]) >= min_gpus
        and all(cuda["bf16_supported"][:min_gpus])
        and cuda["flash_attention_forward"]
        and cuda_runtime_ok
    )
    import_ok = all(item["ok"] for item in imports.values())
    return {
        "ok": (
            import_ok
            and cuda_ok
            and python_ok
            and pip_check["ok"]
            and (versions_ok or allow_version_mismatch)
        ),
        "classification": "closest-public environment compatibility lock",
        "author_confirmed_torch_version": False,
        "allow_version_mismatch": allow_version_mismatch,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "expected_major_minor": "3.9",
            "ok": python_ok,
        },
        "versions_ok": versions_ok,
        "pip_check": pip_check,
        "packages": versions,
        "imports": imports,
        "cuda": cuda,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.min_gpus < 1:
        print("--min-gpus must be positive", file=sys.stderr)
        return 2
    result = run_checks(args.min_gpus, args.allow_version_mismatch)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
