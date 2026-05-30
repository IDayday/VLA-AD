#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path("/mnt/project/VLA-AD")
DEFAULT_OUTPUT_PARENT = PROJECT_ROOT / "experiments/recogdrive_expert"
STAGE1_BASE = PROJECT_ROOT / "checkpoints/recogdrive/ReCogDrive-VLM-2B"
CHUNK_ROOT = PROJECT_ROOT / "cache/recogdrive_expert_chunks/full_v1"
METRIC_CACHE = PROJECT_ROOT / "cache/metric_cache_navtest_full_v1"
PYTHON = "/root/miniconda3/envs/navsim/bin/python"

ALIGN_CONFIG = "configs/ablations/recogdrive2b_A4_align_first_stage1.yaml"
STAGE2_CONFIG = "configs/ablations/recogdrive2b_A4_align_first_stage2.yaml"


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    parser = argparse.ArgumentParser(description="Launch A4 align-first ReCogDrive expert-token training.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_PARENT / f"a4_align_first_{stamp}")
    parser.add_argument("--stage1-base-path", type=Path, default=STAGE1_BASE)
    parser.add_argument("--init-policy-checkpoint", type=Path, default=None)
    parser.add_argument("--base-il-checkpoint", type=Path, default=None, help="Deprecated alias for --init-policy-checkpoint.")
    parser.add_argument("--chunk-cache-root", type=Path, default=CHUNK_ROOT)
    parser.add_argument("--chunk-name-pattern", default="train_*chunk_*")
    parser.add_argument("--metric-cache-dir", type=Path, default=METRIC_CACHE)
    parser.add_argument("--python", default=PYTHON)
    parser.add_argument("--torchrun", default=None)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--master-port", type=int, default=29731)
    parser.add_argument("--align-epochs", type=int, default=20)
    parser.add_argument("--stage2-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--flat-global-dataset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--align-lr-expert", type=float, default=1e-4)
    parser.add_argument("--stage2-lr-expert", type=float, default=1e-4)
    parser.add_argument("--stage2-lr-action-head", type=float, default=1e-4)
    parser.add_argument("--jepa-align-weight", type=float, default=1.0)
    parser.add_argument("--vggt-align-weight", type=float, default=1.0)
    parser.add_argument("--stage2-freeze-expert", action="store_true")
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="bf16")
    parser.add_argument("--eval-precision", choices=("fp16", "bf16", "fp32"), default="fp32")
    parser.add_argument("--lr-warmup-epochs", type=int, default=3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--stage2-expert-context-ramp-end-epoch", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-steps", default="50000,60000,80000,100000,120000,140000,160000")
    parser.add_argument("--expected-metric-caches", type=int, default=12138)
    parser.add_argument("--staged-eval-gpu", default="7")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--launch-eval-watcher", action="store_true")
    parser.add_argument("--launch-staged-eval-watcher", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.base_il_checkpoint and args.init_policy_checkpoint is None:
        args.init_policy_checkpoint = args.base_il_checkpoint
    return args


def torchrun_path(args: argparse.Namespace) -> str:
    if args.torchrun:
        return args.torchrun
    return str(Path(args.python).with_name("torchrun"))


def torchrun_command(args: argparse.Namespace, script_cmd: List[str], *, master_port: int) -> List[str]:
    if args.nproc_per_node <= 1:
        return script_cmd
    return [
        torchrun_path(args),
        "--nproc_per_node",
        str(args.nproc_per_node),
        "--master_port",
        str(master_port),
        *script_cmd[1:],
    ]


def common_train_args(args: argparse.Namespace, output_dir: Path, config: str, epochs: int) -> List[str]:
    cmd = [
        args.python,
        "scripts/train_recogdrive_expert_chunked.py",
        "--config",
        config,
        "--recogdrive-vlm-path",
        str(args.stage1_base_path),
        "--chunk-cache-root",
        str(args.chunk_cache_root),
        "--chunk-name-pattern",
        args.chunk_name_pattern,
        "--global-epochs",
        str(epochs),
        "--batch-size",
        str(args.batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--num-workers",
        str(args.num_workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--precision",
        args.precision,
        "--lr-scheduler",
        "official-cosine",
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
        str(output_dir),
    ]
    if args.flat_global_dataset:
        cmd.append("--flat-global-dataset")
    return cmd


def align_dir(args: argparse.Namespace) -> Path:
    return args.output_root / "stage1_align"


def stage2_dir(args: argparse.Namespace) -> Path:
    return args.output_root / "stage2_condition"


def align_command(args: argparse.Namespace) -> List[str]:
    cmd = common_train_args(args, align_dir(args), ALIGN_CONFIG, args.align_epochs)
    cmd.extend([
        "--lr-scheduler-epochs",
        str(args.align_epochs),
        "--lr-expert",
        str(args.align_lr_expert),
        "--lr-action-head",
        "0",
        "--jepa-align-weight",
        str(args.jepa_align_weight),
        "--vggt-align-weight",
        str(args.vggt_align_weight),
        "--diffusion-loss-weight",
        "0",
        "--train-expert-only",
        "--freeze-base-action-head",
    ])
    if args.init_policy_checkpoint is not None:
        cmd.extend(["--init-policy-checkpoint", str(args.init_policy_checkpoint)])
    return torchrun_command(args, cmd, master_port=args.master_port)


def stage2_command(args: argparse.Namespace) -> List[str]:
    cmd = common_train_args(args, stage2_dir(args), STAGE2_CONFIG, args.stage2_epochs)
    cmd.extend([
        "--resume-from",
        str(align_dir(args) / "latest.ckpt"),
        "--lr-scheduler-epochs",
        str(args.stage2_epochs),
        "--lr-expert",
        str(args.stage2_lr_expert),
        "--lr-action-head",
        str(args.stage2_lr_action_head),
        "--jepa-align-weight",
        "0",
        "--vggt-align-weight",
        "0",
        "--diffusion-loss-weight",
        "1",
        "--alignment-ramp-start-epoch",
        "0",
        "--alignment-ramp-end-epoch",
        "0",
        "--expert-context-ramp-start-epoch",
        "0",
        "--expert-context-ramp-end-epoch",
        str(args.stage2_expert_context_ramp_end_epoch),
    ])
    if args.stage2_freeze_expert:
        cmd.append("--freeze-expert")
    return torchrun_command(args, cmd, master_port=args.master_port + 1)


def run_specs(args: argparse.Namespace) -> Dict[str, Dict[str, str | None]]:
    return {
        "a4_align_first": {
            "config": str(args.project_root / STAGE2_CONFIG),
            "checkpoint": str(stage2_dir(args) / "latest.ckpt"),
            "train_dir": str(stage2_dir(args)),
        }
    }


def eval_command(args: argparse.Namespace) -> List[str]:
    return [
        args.python,
        "scripts/run_recogdrive_full_pdm_suite.py",
        "--run-spec-json",
        str(args.output_root / "run_specs.json"),
        "--output-root",
        str(args.output_root / "pdm_full_navtest_latest"),
        "--runs",
        "a4_align_first",
        "--chunk-cache-root",
        str(args.chunk_cache_root),
        "--chunk-name-pattern",
        "navtest_full_chunk_*",
        "--metric-cache-dir",
        str(args.metric_cache_dir),
        "--expected-metric-caches",
        str(args.expected_metric_caches),
        "--precision",
        args.eval_precision,
        "--root-report-name",
        "FINAL_A4_ALIGN_FIRST_PDM_REPORT.md",
        "--wait",
        "--poll-seconds",
        "300",
    ]


def staged_eval_command(args: argparse.Namespace) -> List[str]:
    return [
        args.python,
        "scripts/watch_recogdrive_staged_pdm.py",
        "--experiment-root",
        str(args.output_root),
        "--run-spec-json",
        str(args.output_root / "run_specs.json"),
        "--output-root",
        str(args.output_root / "pdm_staged_navtest"),
        "--runs",
        "a4_align_first",
        "--checkpoint-steps",
        args.checkpoint_steps,
        "--include-final",
        "--chunk-cache-root",
        str(args.chunk_cache_root),
        "--chunk-name-pattern",
        "navtest_full_chunk_*",
        "--metric-cache-dir",
        str(args.metric_cache_dir),
        "--expected-metric-caches",
        str(args.expected_metric_caches),
        "--precision",
        args.eval_precision,
        "--cuda-visible-devices",
        args.staged_eval_gpu,
        "--poll-seconds",
        "300",
    ]


def write_protocol(args: argparse.Namespace) -> None:
    effective_batch = args.batch_size * args.gradient_accumulation_steps * max(args.nproc_per_node, 1)
    init_policy = str(args.init_policy_checkpoint) if args.init_policy_checkpoint else "none; policy/action-head initialized from config"
    lines = [
        "# A4 Align-First ReCogDrive Expert Training",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Method",
        "",
        "Stage 1 trains the JEPA/VGGT expert branch with alignment losses only. The diffusion loss weight is 0, and the base action head is frozen.",
        "",
        "Stage 2 resumes from the Stage 1 checkpoint and trains ReCogDrive with the aligned expert branch injected as condition. Alignment losses are disabled.",
        "",
        "## Paths",
        "",
        f"- Project root: `{args.project_root}`",
        f"- NAVSIM data root: `/mnt/navsim`",
        f"- Stage1 VLM base path: `{args.stage1_base_path}`",
        f"- Init policy checkpoint: `{init_policy}`",
        f"- Chunk cache root: `{args.chunk_cache_root}`",
        f"- Metric cache dir: `{args.metric_cache_dir}`",
        "",
        "## Training",
        "",
        f"- Align epochs: `{args.align_epochs}`",
        f"- Stage2 epochs: `{args.stage2_epochs}`",
        f"- Effective batch size: `{effective_batch}`",
        f"- Train precision: `{args.precision}`",
        f"- Stage2 freeze expert: `{args.stage2_freeze_expert}`",
        f"- Stage2 checkpoint steps for PDM: `{args.checkpoint_steps},final`",
        f"- Stage1 config: `{ALIGN_CONFIG}`",
        f"- Stage2 config: `{STAGE2_CONFIG}`",
    ]
    (args.output_root / "PROTOCOL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_shell(path: Path, args: argparse.Namespace, commands: List[List[str]]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(args.project_root))}",
        "",
    ]
    prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(args.gpus)} "
    env = f"PYTHONPATH={shlex.quote(str(args.project_root))}:${{PYTHONPATH:-}} "
    for command in commands:
        lines.append(prefix + env + shlex.join(command))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def write_artifacts(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "launcher_args.json").write_text(
        json.dumps(vars(args), default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "run_specs.json").write_text(json.dumps(run_specs(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_protocol(args)
    write_shell(args.output_root / "run_align_stage.sh", args, [align_command(args)])
    write_shell(args.output_root / "run_stage2_condition.sh", args, [stage2_command(args)])
    write_shell(args.output_root / "run_sequence.sh", args, [align_command(args), stage2_command(args)])
    eval_shell = args.output_root / "run_pdm_full_wait.sh"
    write_shell(eval_shell, args, [eval_command(args)])
    staged_eval_shell = args.output_root / "run_pdm_staged_wait.sh"
    write_shell(staged_eval_shell, args, [staged_eval_command(args)])


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def launch_sequence(args: argparse.Namespace) -> Dict[str, Any]:
    shell_path = args.output_root / "run_sequence.sh"
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


def launch_staged_eval_watcher(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = args.output_root / "pdm_staged_navtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    pid_path = out_dir / "pid.txt"
    if pid_path.is_file():
        try:
            old_pid = int(pid_path.read_text().strip())
        except ValueError:
            old_pid = -1
        if old_pid > 0 and is_alive(old_pid) and not args.force:
            return {"pid": old_pid, "status": "already_running"}
    cmd = staged_eval_command(args)
    (out_dir / "command.txt").write_text(" ".join(shlex.quote(part) for part in cmd) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{args.project_root}:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    log_fp = (out_dir / "watcher_console.log").open("a", encoding="utf-8")
    process = subprocess.Popen(cmd, cwd=args.project_root, env=env, stdout=log_fp, stderr=subprocess.STDOUT, start_new_session=True)
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    return {"pid": process.pid, "status": "launched"}


def main() -> int:
    args = parse_args()
    write_artifacts(args)
    result: Dict[str, Any] = {
        "output_root": str(args.output_root),
        "align_dir": str(align_dir(args)),
        "stage2_dir": str(stage2_dir(args)),
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps * max(args.nproc_per_node, 1),
        "launch": {},
    }
    if args.launch:
        result["launch"]["sequence"] = launch_sequence(args)
        if args.launch_eval_watcher:
            result["launch"]["eval_watcher"] = launch_eval_watcher(args)
        if args.launch_staged_eval_watcher:
            result["launch"]["staged_eval_watcher"] = launch_staged_eval_watcher(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    (args.output_root / "launch_status.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
