#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path("/mnt/project/VLA-AD")
DEFAULT_OUTPUT_PARENT = PROJECT_ROOT / "experiments/recogdrive_expert"
STAGE1_BASE = PROJECT_ROOT / "checkpoints/recogdrive/ReCogDrive-VLM-2B"
CHUNK_ROOT = PROJECT_ROOT / "cache/recogdrive_expert_chunks/full_v1"
METRIC_CACHE = PROJECT_ROOT / "cache/metric_cache_navtest_full_v1"
PYTHON = "/root/miniconda3/envs/navsim/bin/python"

RUNS: Dict[str, Dict[str, Any]] = {
    "a0_no_expert": {
        "label": "A0 no-expert",
        "config": "configs/ablations/recogdrive2b_A0_base_no_expert.yaml",
        "lr_expert": 0.0,
        "jepa_align_weight": 0.0,
        "vggt_align_weight": 0.0,
    },
    "a1_jepa_only": {
        "label": "A1 JEPA-only",
        "config": "configs/ablations/recogdrive2b_A1_jepa_only.yaml",
        "lr_expert": 1e-4,
        "jepa_align_weight": 0.03,
        "vggt_align_weight": 0.0,
    },
    "a2_vggt_only": {
        "label": "A2 VGGT-only",
        "config": "configs/ablations/recogdrive2b_A2_vggt_only.yaml",
        "lr_expert": 1e-4,
        "jepa_align_weight": 0.0,
        "vggt_align_weight": 0.05,
    },
    "a3_context_only": {
        "label": "A3 context-only",
        "config": "configs/ablations/recogdrive2b_A3_context_only.yaml",
        "lr_expert": 1e-4,
        "jepa_align_weight": 0.0,
        "vggt_align_weight": 0.0,
    },
    "a4_jepa_vggt": {
        "label": "A4 JEPA+VGGT",
        "config": "configs/ablations/recogdrive2b_A4_full.yaml",
        "lr_expert": 1e-4,
        "jepa_align_weight": 0.03,
        "vggt_align_weight": 0.05,
    },
    "a4_v2": {
        "label": "A4-v2 JEPA+VGGT horizon residual",
        "config": "configs/ablations/recogdrive2b_A4_v2.yaml",
        "lr_expert": 1e-4,
        "jepa_align_weight": 0.03,
        "vggt_align_weight": 0.05,
        "alignment_ramp_start_epoch": 3,
        "alignment_ramp_end_epoch": 20,
        "expert_context_ramp_start_epoch": 0,
        "expert_context_ramp_end_epoch": 10,
    },
}


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    parser = argparse.ArgumentParser(description="Launch full-epoch ReCogDrive expert-token runs from the Stage1 VLM base cache.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_PARENT / f"stage2_a4v2_stage1base_full200_{stamp}")
    parser.add_argument("--stage1-base-path", type=Path, default=STAGE1_BASE)
    parser.add_argument("--init-policy-checkpoint", type=Path, default=None)
    parser.add_argument("--base-il-checkpoint", type=Path, default=None, help="Deprecated alias for --init-policy-checkpoint.")
    parser.add_argument("--chunk-cache-root", type=Path, default=CHUNK_ROOT)
    parser.add_argument("--metric-cache-dir", type=Path, default=METRIC_CACHE)
    parser.add_argument("--python", default=PYTHON)
    parser.add_argument("--torchrun", default=None)
    parser.add_argument("--only", default="a0_no_expert,a4_v2")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--master-port", type=int, default=29671)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--global-epochs", type=int, default=200)
    parser.add_argument("--official-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--flat-global-dataset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="bf16")
    parser.add_argument("--eval-precision", choices=("fp16", "bf16", "fp32"), default="fp32")
    parser.add_argument("--lr-scheduler-epochs", type=int, default=200)
    parser.add_argument("--lr-scheduler-start-epoch", type=int, default=0)
    parser.add_argument("--lr-warmup-epochs", type=int, default=3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--launch-eval-watcher", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.base_il_checkpoint and args.init_policy_checkpoint is None:
        args.init_policy_checkpoint = args.base_il_checkpoint
    return args


def selected_runs(args: argparse.Namespace) -> List[str]:
    selected = [item.strip() for item in args.only.split(",") if item.strip()]
    unknown = [item for item in selected if item not in RUNS]
    if unknown:
        raise ValueError(f"Unknown runs: {unknown}")
    return selected


def torchrun_path(args: argparse.Namespace) -> str:
    if args.torchrun:
        return args.torchrun
    return str(Path(args.python).with_name("torchrun"))


def gpu_for_run(args: argparse.Namespace, index: int) -> str:
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(gpus) < len(selected_runs(args)):
        raise ValueError(f"Need at least {len(selected_runs(args))} GPUs, got {gpus}")
    return gpus[index]


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run_dir(args: argparse.Namespace, run: str) -> Path:
    return args.output_root / run


def master_port_for_run(args: argparse.Namespace, run_index: int) -> int:
    return int(args.master_port) + int(run_index)


def train_command(args: argparse.Namespace, run: str, *, master_port: int | None = None) -> List[str]:
    spec = RUNS[run]
    script_cmd = [
        args.python,
        "scripts/train_recogdrive_expert_chunked.py",
        "--config",
        spec["config"],
        "--recogdrive-vlm-path",
        str(args.stage1_base_path),
        "--chunk-cache-root",
        str(args.chunk_cache_root),
        "--chunk-name-pattern",
        "train_*chunk_*",
        "--global-epochs",
        str(args.global_epochs),
        "--batch-size",
        str(args.batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--num-workers",
        str(args.num_workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--lr-expert",
        str(spec["lr_expert"]),
        "--lr-action-head",
        str(args.lr),
        "--jepa-align-weight",
        str(spec["jepa_align_weight"]),
        "--vggt-align-weight",
        str(spec["vggt_align_weight"]),
        "--alignment-ramp-start-epoch",
        str(spec.get("alignment_ramp_start_epoch", 0)),
        "--alignment-ramp-end-epoch",
        str(spec.get("alignment_ramp_end_epoch", 0)),
        "--expert-context-ramp-start-epoch",
        str(spec.get("expert_context_ramp_start_epoch", 0)),
        "--expert-context-ramp-end-epoch",
        str(spec.get("expert_context_ramp_end_epoch", 0)),
        "--precision",
        args.precision,
        "--lr-scheduler",
        "official-cosine",
        "--lr-scheduler-epochs",
        str(args.lr_scheduler_epochs),
        "--lr-scheduler-start-epoch",
        str(args.lr_scheduler_start_epoch),
        "--lr-warmup-epochs",
        str(args.lr_warmup_epochs),
        "--min-lr",
        str(args.min_lr),
        "--log-every",
        str(args.log_every),
        "--save-every",
        str(args.save_every),
        "--final-check-precision",
        "fp32",
        "--output-dir",
        str(run_dir(args, run)),
    ]
    if args.init_policy_checkpoint is not None:
        script_cmd.extend(["--init-policy-checkpoint", str(args.init_policy_checkpoint)])
    if args.flat_global_dataset:
        script_cmd.append("--flat-global-dataset")
    if args.nproc_per_node <= 1:
        return script_cmd
    return [
        torchrun_path(args),
        "--nproc_per_node",
        str(args.nproc_per_node),
        "--master_port",
        str(args.master_port if master_port is None else master_port),
        *script_cmd[1:],
    ]


def run_specs(args: argparse.Namespace) -> Dict[str, Dict[str, str | None]]:
    specs: Dict[str, Dict[str, str | None]] = {}
    for run in RUNS:
        specs[run] = {
            "config": str(args.project_root / RUNS[run]["config"]),
            "checkpoint": str(run_dir(args, run) / "latest.ckpt"),
            "train_dir": str(run_dir(args, run)),
        }
    return specs


def eval_command(args: argparse.Namespace) -> List[str]:
    return [
        args.python,
        "scripts/run_recogdrive_full_pdm_suite.py",
        "--run-spec-json",
        str(args.output_root / "run_specs.json"),
        "--output-root",
        str(args.output_root / "pdm_full_navtest_latest"),
        "--runs",
        ",".join(selected_runs(args)),
        "--chunk-cache-root",
        str(args.chunk_cache_root),
        "--chunk-name-pattern",
        "navtest_full_chunk_*",
        "--metric-cache-dir",
        str(args.metric_cache_dir),
        "--expected-metric-caches",
        "12138",
        "--precision",
        args.eval_precision,
        "--root-report-name",
        "FINAL_STAGE1BASE_FULL200_PDM_REPORT.md",
        "--wait",
        "--poll-seconds",
        "300",
    ]


def write_protocol(args: argparse.Namespace, selected: List[str]) -> None:
    effective_batch = args.batch_size * args.gradient_accumulation_steps * max(args.nproc_per_node, 1)
    init_policy = str(args.init_policy_checkpoint) if args.init_policy_checkpoint else "none; policy/action-head initialized from config"
    lines = [
        "# Stage1-Base Full-Epoch ReCogDrive Expert Training",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Purpose",
        "",
        "不再从官方 `ReCogDrive-2B-IL` policy checkpoint 微调；本轮使用 Stage1 预训练后的 `ReCogDrive-VLM-2B` 作为缓存特征来源，policy/action-head 与专家分支按当前配置从头训练完整 epoch。",
        "",
        "## Training Choices",
        "",
        f"- Stage1 VLM base path: `{args.stage1_base_path}`",
        f"- Init policy checkpoint: `{init_policy}`",
        f"- Official Stage2 reference epochs: `{args.official_epochs}`",
        f"- This run epochs: `{args.global_epochs}`",
        f"- Per-device-style batch size: `{args.batch_size}`",
        f"- GPUs per experiment: `{args.nproc_per_node}`",
        f"- Gradient accumulation: `{args.gradient_accumulation_steps}`",
        f"- Effective batch size: `{effective_batch}`",
        f"- DataLoader workers per rank: `{args.num_workers}`, prefetch_factor=`{args.prefetch_factor}`",
        f"- Sampling/DataLoader: `{'flat global all-chunk sampler' if args.flat_global_dataset else 'per-chunk sampler'}`",
        f"- Action-head LR: `{args.lr}` with AdamW weight_decay=1e-4 betas=(0.9,0.95)",
        "- Expert LR: per-run table below (`0.0` for A0, `1e-4` for expert runs)",
        f"- LR scheduler: official-style warmup cosine, epochs=`{args.lr_scheduler_epochs}`, start_epoch=`{args.lr_scheduler_start_epoch}`, warmup=`{args.lr_warmup_epochs}`, min_lr=`{args.min_lr}`",
        f"- Train precision: `{args.precision}`",
        f"- Primary checkpoint for PDM: `latest.ckpt` after final epoch, not train-loss `best.ckpt`",
        f"- Full-navtest eval precision: `{args.eval_precision}`",
        "",
        "## Runs",
        "",
        "| Run | Config | Expert LR | JEPA align | VGGT align | Align ramp | Expert context ramp |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for run in selected:
        spec = RUNS[run]
        align_ramp = "{}-{}".format(spec.get("alignment_ramp_start_epoch", 0), spec.get("alignment_ramp_end_epoch", 0))
        context_ramp = "{}-{}".format(spec.get("expert_context_ramp_start_epoch", 0), spec.get("expert_context_ramp_end_epoch", 0))
        lines.append("| {} | `{}` | {} | {} | {} | {} | {} |".format(
            run,
            spec["config"],
            spec["lr_expert"],
            spec["jepa_align_weight"],
            spec["vggt_align_weight"],
            align_ramp,
            context_ramp,
        ))
    lines.extend([
        "",
        "## Decision Rule",
        "",
        "A4-v2 必须在 full navtest PDMS 上超过同样从 Stage1 base/full-epoch 训练的 A0 no-expert control，才能声称 JEPA+VGGT 注入带来有效增益。",
    ])
    (args.output_root / "PROTOCOL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_artifacts(args: argparse.Namespace, selected: List[str]) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "launcher_args.json").write_text(
        json.dumps(vars(args), default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "run_specs.json").write_text(json.dumps(run_specs(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_protocol(args, selected)
    shell_lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd {shlex.quote(str(args.project_root))}", ""]
    for idx, run in enumerate(selected):
        cmd = train_command(args, run, master_port=master_port_for_run(args, idx))
        shell_lines.append(f"# {run}: {RUNS[run]['label']}")
        if args.nproc_per_node <= 1:
            prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu_for_run(args, idx))} "
        else:
            prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(args.gpus)} "
        shell_lines.append(prefix + f"PYTHONPATH={shlex.quote(str(args.project_root))}:${{PYTHONPATH:-}} " + shlex.join(cmd))
        shell_lines.append("")
    shell_path = args.output_root / "run_train_foreground.sh"
    shell_path.write_text("\n".join(shell_lines), encoding="utf-8")
    shell_path.chmod(0o755)
    sequence_path = sequence_shell_path(args)
    sequence_path.write_text("\n".join(shell_lines), encoding="utf-8")
    sequence_path.chmod(0o755)
    parallel_lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd {shlex.quote(str(args.project_root))}", ""]
    parallel_lines.append("pids=()")
    for idx, run in enumerate(selected):
        cmd = train_command(args, run, master_port=master_port_for_run(args, idx))
        out_dir = run_dir(args, run)
        parallel_lines.append(f"mkdir -p {shlex.quote(str(out_dir))}")
        parallel_lines.append(f"# {run}: {RUNS[run]['label']}")
        if args.nproc_per_node <= 1:
            prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu_for_run(args, idx))} "
        else:
            prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(args.gpus)} "
        parallel_lines.append(
            prefix
            + f"PYTHONPATH={shlex.quote(str(args.project_root))}:${{PYTHONPATH:-}} "
            + shlex.join(cmd)
            + f" > {shlex.quote(str(out_dir / 'console.log'))} 2>&1 &"
        )
        parallel_lines.append(f"echo $! > {shlex.quote(str(out_dir / 'pid.txt'))}")
        parallel_lines.append("pids+=(\"$!\")")
        parallel_lines.append("")
    parallel_lines.append("wait \"${pids[@]}\"")
    parallel_path = args.output_root / "run_train_8gpu_parallel.sh"
    parallel_path.write_text("\n".join(parallel_lines), encoding="utf-8")
    parallel_path.chmod(0o755)
    eval_shell = args.output_root / "run_pdm_full_wait.sh"
    eval_shell.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(args.project_root))}\n"
        f"PYTHONPATH={shlex.quote(str(args.project_root))}:${{PYTHONPATH:-}} {shlex.join(eval_command(args))}\n",
        encoding="utf-8",
    )
    eval_shell.chmod(0o755)


def sequence_shell_path(args: argparse.Namespace) -> Path:
    return args.output_root / "run_train_8gpu_sequence.sh"


def launch_run(args: argparse.Namespace, run: str, gpu: str, *, master_port: int | None = None) -> Dict[str, Any]:
    out_dir = run_dir(args, run)
    out_dir.mkdir(parents=True, exist_ok=True)
    pid_path = out_dir / "pid.txt"
    if pid_path.is_file():
        try:
            old_pid = int(pid_path.read_text().strip())
        except ValueError:
            old_pid = -1
        if old_pid > 0 and is_alive(old_pid) and not args.force:
            return {"run": run, "pid": old_pid, "status": "already_running", "gpu": gpu}
    cmd = train_command(args, run, master_port=master_port)
    (out_dir / "command.txt").write_text(" ".join(shlex.quote(part) for part in cmd) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = f"{args.project_root}:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    log_fp = (out_dir / "console.log").open("a", encoding="utf-8")
    process = subprocess.Popen(cmd, cwd=args.project_root, env=env, stdout=log_fp, stderr=subprocess.STDOUT, start_new_session=True)
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    return {"run": run, "pid": process.pid, "status": "launched", "gpu": gpu}


def launch_parallel(args: argparse.Namespace, selected: List[str]) -> List[Dict[str, Any]]:
    launches: List[Dict[str, Any]] = []
    for idx, run in enumerate(selected):
        gpu = args.gpus if args.nproc_per_node > 1 else gpu_for_run(args, idx)
        launches.append(launch_run(args, run, gpu, master_port=master_port_for_run(args, idx)))
    return launches


def launch_sequence(args: argparse.Namespace) -> Dict[str, Any]:
    shell_path = sequence_shell_path(args)
    pid_path = args.output_root / "sequence_pid.txt"
    if pid_path.is_file():
        try:
            old_pid = int(pid_path.read_text().strip())
        except ValueError:
            old_pid = -1
        if old_pid > 0 and is_alive(old_pid) and not args.force:
            return {"pid": old_pid, "status": "already_running", "script": str(shell_path)}
    log_fp = (args.output_root / "sequence_console.log").open("a", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(["bash", str(shell_path)], cwd=args.project_root, env=env, stdout=log_fp, stderr=subprocess.STDOUT, start_new_session=True)
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    return {"pid": process.pid, "status": "launched", "script": str(shell_path)}


def launch_eval_watcher(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = args.output_root / "pdm_full_navtest_latest"
    out_dir.mkdir(parents=True, exist_ok=True)
    pid_path = out_dir / "pid.txt"
    if pid_path.is_file():
        try:
            old_pid = int(pid_path.read_text().strip())
        except ValueError:
            old_pid = -1
        if old_pid > 0 and is_alive(old_pid) and not args.force:
            return {"pid": old_pid, "status": "already_running"}
    cmd = eval_command(args)
    (out_dir / "command.txt").write_text(" ".join(shlex.quote(part) for part in cmd) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{args.project_root}:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    log_fp = (out_dir / "console.log").open("a", encoding="utf-8")
    process = subprocess.Popen(cmd, cwd=args.project_root, env=env, stdout=log_fp, stderr=subprocess.STDOUT, start_new_session=True)
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    return {"pid": process.pid, "status": "launched"}


def main() -> int:
    args = parse_args()
    selected = selected_runs(args)
    write_artifacts(args, selected)
    result: Dict[str, Any] = {
        "output_root": str(args.output_root),
        "selected_runs": selected,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps * max(args.nproc_per_node, 1),
        "launch": [],
    }
    if args.launch:
        if args.parallel:
            result["launch"] = launch_parallel(args, selected)
        elif args.nproc_per_node > 1:
            result["launch_sequence"] = launch_sequence(args)
        else:
            for idx, run in enumerate(selected):
                result["launch"].append(launch_run(args, run, gpu_for_run(args, idx), master_port=master_port_for_run(args, idx)))
        if args.launch_eval_watcher:
            result["eval_watcher"] = launch_eval_watcher(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    (args.output_root / "launch_status.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
