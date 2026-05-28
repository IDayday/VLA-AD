#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_PROJECT_ROOT = Path("/mnt/project/VLA-AD")
DEFAULT_CHECKPOINT_ROOT = DEFAULT_PROJECT_ROOT / "checkpoints"
DEFAULT_EXPERIMENT_ROOT = DEFAULT_PROJECT_ROOT / "experiments" / "recogdrive_expert"
DEFAULT_CACHE_ROOT = DEFAULT_PROJECT_ROOT / "cache" / "recogdrive_expert_chunks"
REQUIRED = {
    "recogdrive_2b_il": "recogdrive_2b_il_path",
    "recogdrive_vlm_2b": "recogdrive_vlm_2b_path",
    "vjepa2": "vjepa2_path",
    "vggt_1b": "vggt_1b_path",
}
CONFIG_SUFFIXES = {".json", ".yaml", ".yml", ".py", ".txt", ".model"}
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".ckpt", ".pth", ".model"}
PARTIAL_SUFFIXES = (".incomplete", ".partial", ".tmp", ".download")
IGNORED_WEIGHT_NAMES = {"training_args.bin", "trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report the current real experiment gate from resolved weight paths.")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--resolved-paths", type=Path, default=DEFAULT_CHECKPOINT_ROOT / "resolved_weight_paths.json")
    return parser.parse_args()


def load_resolved(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"records": {}, "missing_file": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def is_plausible_weight_file(path: Path) -> bool:
    lower = path.name.lower()
    if ".cache" in path.parts or lower.endswith(PARTIAL_SUFFIXES):
        return False
    if lower in IGNORED_WEIGHT_NAMES:
        return False
    if path.suffix.lower() not in WEIGHT_SUFFIXES:
        return False
    if path.suffix.lower() == ".model":
        return True
    return path.stat().st_size >= 1024 * 1024


def local_dir_complete(key: str, local_dir_value: Any) -> tuple[bool, List[str], int]:
    missing: List[str] = []
    if not local_dir_value:
        return False, ["local_dir"], 0
    local_dir = Path(str(local_dir_value))
    if not local_dir.is_dir():
        return False, ["directory"], 0
    weights = [path for path in local_dir.rglob("*") if path.is_file() and is_plausible_weight_file(path)]
    configs = [
        path for path in local_dir.rglob("*")
        if path.is_file() and ".cache" not in path.parts and path.suffix.lower() in CONFIG_SUFFIXES
    ]
    if not weights:
        missing.append("model weight files")
    if key not in {"recogdrive_2b_il"} and not configs:
        missing.append("config/tokenizer/source files")
    return not missing, missing, len(weights)


def resolved_entry_complete(key: str, resolved: Dict[str, Any], json_key: str) -> Dict[str, Any]:
    records = resolved.get("records") or {}
    record = records.get(key) or {}
    selected_path = resolved.get(json_key) or record.get("selected_path")
    selected_exists = bool(selected_path and Path(str(selected_path)).exists())
    dir_complete, missing, weight_count = local_dir_complete(key, record.get("local_dir"))
    complete = selected_exists and dir_complete
    if not selected_exists:
        missing = [*missing, "resolved selected path"]
    return {
        "json_key": json_key,
        "path": selected_path,
        "local_dir": record.get("local_dir"),
        "selected_exists": selected_exists,
        "weight_file_count": weight_count,
        "complete": complete,
        "missing_critical_files": missing,
    }


def determine_gate(complete: Dict[str, bool]) -> str:
    if all(complete.values()):
        return "W4"
    if complete["recogdrive_2b_il"] and complete["recogdrive_vlm_2b"] and complete["vjepa2"]:
        return "W3"
    if complete["recogdrive_2b_il"] and complete["recogdrive_vlm_2b"]:
        return "W2"
    if complete["recogdrive_2b_il"]:
        return "W1"
    return "W0"


def command_block(lines: List[str]) -> str:
    return " \\\n  ".join(lines)


def next_command(gate: str, args: argparse.Namespace) -> str:
    project = args.project_root
    ckpt = args.checkpoint_root
    cache = args.cache_root
    exp = args.experiment_root
    if gate == "W0":
        return command_block([
            "python scripts/run_weight_download_priority_plan.py",
            f"--output-root {ckpt}",
            "--source huggingface",
            "--proxy http://127.0.0.1:7890",
            "--timeout 300",
        ])
    if gate == "W1":
        return command_block([
            "python scripts/check_real_checkpoint_loading.py",
            f"--base-il-checkpoint {ckpt / 'recogdrive' / 'ReCogDrive-2B-IL'}",
            "--config configs/recogdrive2b_expert768_il.yaml",
            f"--output {exp / 'checkpoint_loading_report.md'}",
        ])
    if gate == "W2":
        vlm_chunk = cache / "navtest_vlm_chunk_000000"
        a0_dir = exp / "ablations" / "A0_base_no_expert_eval_256"
        if not (vlm_chunk / "metadata.json").is_file():
            return command_block([
                "python scripts/build_recogdrive_chunk_cache.py",
                "--navsim-root /mnt/navsim",
                "--split navtest",
                "--chunk-index 0",
                "--chunk-size 256",
                f"--output-dir {vlm_chunk}",
                "--build-vlm-hidden",
                f"--recogdrive-vlm-path {ckpt / 'recogdrive' / 'ReCogDrive-VLM-2B'}",
                "--precision bf16",
                "--device cuda",
                "--num-gpus 8",
            ])
        if not (a0_dir / "metrics.json").is_file():
            return command_block([
                "python scripts/eval_recogdrive_expert_pdm.py",
                "--config configs/ablations/recogdrive2b_A0_base_no_expert.yaml",
                f"--checkpoint {ckpt / 'recogdrive' / 'ReCogDrive-2B-IL'}",
                f"--chunk-cache-dir {vlm_chunk}",
                "--split navtest",
                "--max-samples 256",
                f"--output-dir {a0_dir}",
            ])
        return command_block([
            "python scripts/download_required_weights.py",
            f"--output-root {ckpt}",
            "--source hf-mirror",
            "--only vjepa2",
            "--repo-workers 1",
            "--file-workers 8",
            "--timeout 300",
        ])
    if gate == "W3":
        train_chunk = cache / "train_jepa_chunk_000000"
        eval_chunk = cache / "navtest_jepa_chunk_000000"
        train_dir = exp / "ablations" / "A1_jepa_only_train_1024"
        if not (train_chunk / "metadata.json").is_file():
            return command_block([
                "python scripts/build_recogdrive_chunk_cache.py",
                "--navsim-root /mnt/navsim",
                "--split navtrain",
                "--chunk-index 0",
                "--chunk-size 1024",
                f"--output-dir {train_chunk}",
                "--build-vlm-hidden",
                "--build-jepa",
                f"--recogdrive-vlm-path {ckpt / 'recogdrive' / 'ReCogDrive-VLM-2B'}",
                f"--jepa-model-path {ckpt / 'teachers' / 'vjepa2-vitl-fpc64-256'}",
                "--precision bf16",
                "--device cuda",
                "--num-gpus 8",
                "--log-every 10",
            ])
        if not (eval_chunk / "metadata.json").is_file():
            return command_block([
                "python scripts/build_recogdrive_chunk_cache.py",
                "--navsim-root /mnt/navsim",
                "--split navtest",
                "--chunk-index 0",
                "--chunk-size 256",
                f"--output-dir {eval_chunk}",
                "--build-vlm-hidden",
                "--build-jepa",
                f"--recogdrive-vlm-path {ckpt / 'recogdrive' / 'ReCogDrive-VLM-2B'}",
                f"--jepa-model-path {ckpt / 'teachers' / 'vjepa2-vitl-fpc64-256'}",
                "--precision bf16",
                "--device cuda",
                "--num-gpus 8",
            ])
        if not (train_dir / "best.ckpt").is_file():
            return command_block([
                "python scripts/train_recogdrive_expert_chunked.py",
                "--config configs/ablations/recogdrive2b_A1_jepa_only.yaml",
                f"--base-il-checkpoint {ckpt / 'recogdrive' / 'ReCogDrive-2B-IL'}",
                f"--chunk-cache-dir {train_chunk}",
                f"--output-dir {train_dir}",
                "--max-samples 1024",
                "--batch-size 2",
                "--gradient-accumulation-steps 2",
                "--num-steps 1000",
                "--lr-expert 1e-4",
                "--lr-action-head 2e-5",
                "--jepa-align-weight 0.03",
                "--vggt-align-weight 0.0",
                "--precision bf16",
                "--log-every 20",
                "--save-every 250",
            ])
        a1_eval = exp / "ablations" / "A1_jepa_only_eval_256"
        if not (a1_eval / "metrics.json").is_file():
            return command_block([
                "python scripts/eval_recogdrive_expert_pdm.py",
                "--config configs/ablations/recogdrive2b_A1_jepa_only.yaml",
                f"--checkpoint {train_dir / 'best.ckpt'}",
                f"--chunk-cache-dir {eval_chunk}",
                "--split navtest",
                "--max-samples 256",
                f"--output-dir {a1_eval}",
            ])
        return command_block([
            "python scripts/download_required_weights.py",
            f"--output-root {ckpt}",
            "--source hf-mirror",
            "--only vggt_1b",
            "--repo-workers 1",
            "--file-workers 4",
            "--timeout 300",
        ])
    return command_block([
        "python scripts/build_recogdrive_chunk_cache.py",
        "--navsim-root /mnt/navsim",
        "--split navtrain",
        "--chunk-index 0",
        "--chunk-size 1024",
        f"--output-dir {cache / 'train_full_chunk_000000'}",
        "--build-vlm-hidden",
        "--build-jepa",
        "--build-vggt",
        f"--recogdrive-vlm-path {ckpt / 'recogdrive' / 'ReCogDrive-VLM-2B'}",
        f"--jepa-model-path {ckpt / 'teachers' / 'vjepa2-vitl-fpc64-256'}",
        f"--vggt-model-path {ckpt / 'teachers' / 'VGGT-1B'}",
        "--precision bf16",
        "--device cuda",
        "--num-gpus 8",
    ])


def actions_for(gate: str) -> tuple[List[str], List[str]]:
    allowed = ["environment check", "NAVSIM data inspection", "weight download/status"]
    blocked: List[str] = []
    if gate in {"W1", "W2", "W3", "W4"}:
        allowed.append("Base-IL checkpoint loading test")
    else:
        blocked.append("Base-IL checkpoint loading test")
    if gate in {"W2", "W3", "W4"}:
        allowed.extend(["A0 same-pipeline baseline eval", "VLM-hidden chunk generation"])
    else:
        blocked.extend(["A0 same-pipeline baseline eval", "VLM-hidden chunk generation"])
    if gate in {"W3", "W4"}:
        allowed.append("JEPA-only real chunk/training/eval")
    else:
        blocked.append("JEPA-only real chunk/training/eval")
    if gate == "W4":
        allowed.append("full JEPA+VGGT pilot and ablations")
    else:
        blocked.append("full JEPA+VGGT pilot and ablations")
    blocked.append("RL training/evaluation")
    return allowed, blocked


def write_outputs(args: argparse.Namespace, payload: Dict[str, Any]) -> None:
    args.experiment_root.mkdir(parents=True, exist_ok=True)
    (args.experiment_root / "current_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Current Real Experiment Gate",
        "",
        f"Created: {payload['created_at']}",
        f"Resolved paths file: `{payload['resolved_paths']}`",
        f"Current gate: **{payload['gate']}**",
        "",
        "## Weight Completeness",
        "",
        "| Target | Complete | Resolved Path |",
        "|---|---:|---|",
    ]
    for key, record in payload["weights"].items():
        missing = ", ".join(record.get("missing_critical_files") or []) or "none"
        lines.append(f"| `{key}` | {record['complete']} | `{record['path']}` |")
        if missing != "none":
            lines.append(f"<!-- {key} missing: {missing} -->")
    lines.extend(["", "## Allowed Next Actions", ""])
    lines.extend([f"- {item}" for item in payload["allowed_actions"]])
    lines.extend(["", "## Blocked Actions", ""])
    lines.extend([f"- {item}" for item in payload["blocked_actions"]])
    lines.extend(["", "## Exact Next Command", "", "```bash", payload["next_command"], "```", ""])
    (args.experiment_root / "current_gate.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    resolved = load_resolved(args.resolved_paths)
    weights = {
        key: resolved_entry_complete(key, resolved, json_key)
        for key, json_key in REQUIRED.items()
    }
    complete = {key: bool(record["complete"]) for key, record in weights.items()}
    gate = determine_gate(complete)
    allowed, blocked = actions_for(gate)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "resolved_paths": str(args.resolved_paths),
        "gate": gate,
        "weights": weights,
        "allowed_actions": allowed,
        "blocked_actions": blocked,
        "next_command": next_command(gate, args),
    }
    write_outputs(args, payload)
    print(f"Current gate: {gate}")
    print("Allowed next actions:")
    for item in allowed:
        print(f"  - {item}")
    print("Blocked actions:")
    for item in blocked:
        print(f"  - {item}")
    print("\nExact next command:")
    print(payload["next_command"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
