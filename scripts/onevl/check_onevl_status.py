#!/usr/bin/env python3
"""Summarize local OneVL/NAVSIM readiness in this workspace."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/project")
MODELS = ROOT / "onevl_models"
DATA = ROOT / "onevl_navsim_data"
ANSWER_RUN = ROOT / "onevl_navsim_exp/answer_full_20260624_184133"
ANSWER_SWIFT = ANSWER_RUN / "swift_output/v0-20260624-184217"
ALIGNMENT_REPORT = ROOT / "onevl_navsim_data/navsim_test_alignment.report.json"
REMOTE_AR_EVAL = ROOT / "onevl_navsim_exp/remote_training_rl_zt3_ar_answer_latest_navtest_eval_20260624_204037"
LOCAL_POST_TRAIN_EVAL = ROOT / "onevl_navsim_exp/local_post_train_ar_answer_eval_latest"

REPOS = {
    "Qwen3-VL-4B-Instruct": MODELS / "Qwen3-VL-4B-Instruct",
    "Baseline_cot_NAVSIM": MODELS / "Baseline_cot_NAVSIM",
    "OneVL_NAVSIM": MODELS / "OneVL_NAVSIM",
    "OneVL_mlp_NAVSIM": MODELS / "OneVL_mlp_NAVSIM",
    "OneVL_visual_decoder_pt": MODELS / "OneVL_visual_decoder_pt",
    "Emu3.5-VisionTokenizer": MODELS / "Emu3.5-VisionTokenizer",
}


def print_section(name: str) -> None:
    print(f"\n== {name} ==")


def selected_repos() -> dict[str, Path]:
    requested = os.environ.get("ONEVL_REPOS")
    if not requested:
        return REPOS
    selected: dict[str, Path] = {}
    aliases = {name: name for name in REPOS}
    aliases.update(
        {
            "Qwen/Qwen3-VL-4B-Instruct": "Qwen3-VL-4B-Instruct",
            "xiaomi-research/Baseline_cot_NAVSIM": "Baseline_cot_NAVSIM",
            "xiaomi-research/OneVL_NAVSIM": "OneVL_NAVSIM",
            "xiaomi-research/OneVL_mlp_NAVSIM": "OneVL_mlp_NAVSIM",
            "xiaomi-research/OneVL_visual_decoder_pt": "OneVL_visual_decoder_pt",
            "BAAI/Emu3.5-VisionTokenizer": "Emu3.5-VisionTokenizer",
        }
    )
    for raw_name in requested.split(","):
        name = raw_name.strip()
        if not name:
            continue
        repo_name = aliases.get(name)
        if repo_name is None:
            known = ", ".join(REPOS)
            raise SystemExit(f"Unknown ONEVL_REPOS entry {name!r}. Known local names: {known}")
        selected[repo_name] = REPOS[repo_name]
    return selected


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def parse_step_pair(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, str) or "/" not in value:
        return None
    left, right = value.split("/", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return None


def read_text_tail(path: Path, max_lines: int = 8) -> list[str]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "")
    lines = text.splitlines()
    return lines[-max_lines:]


def read_pid(path: Path) -> int | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    try:
        return int(text)
    except ValueError:
        return None


def pid_status(pid: int | None) -> str:
    if pid is None:
        return "missing"
    result = run_command(["ps", "-p", str(pid), "-o", "stat=", "-o", "etime="])
    if result.returncode != 0:
        return f"not_running pid={pid}"
    fields = result.stdout.strip().split(None, 1)
    if not fields:
        return f"unknown pid={pid}"
    if len(fields) == 1:
        return f"running pid={pid} stat={fields[0]}"
    return f"running pid={pid} stat={fields[0]} elapsed={fields[1]}"


def check_download_processes() -> None:
    print_section("Downloads")
    result = run_command(["ps", "-eo", "pid,ppid,stat,cmd"])
    needles = ("aria2c", "run_exact_weight_download", "huggingface-cli", "snapshot_download", "hf_hub_download")
    rows = [line for line in result.stdout.splitlines() if any(needle in line for needle in needles)]
    rows = [line for line in rows if "check_onevl_status.py" not in line]
    if rows:
        print("active_download_processes=yes")
        for row in rows[:20]:
            print(row)
    else:
        print("active_download_processes=no")


def check_index(repo: str, path: Path) -> None:
    index_path = path / "model.safetensors.index.json"
    data = json.loads(index_path.read_text())
    expected = sorted(set(data["weight_map"].values()))
    missing = [name for name in expected if not (path / name).exists()]
    empty = [name for name in expected if (path / name).exists() and (path / name).stat().st_size == 0]
    aria2 = sorted(p.name for p in path.glob("*.aria2"))
    status = "ok" if not missing and not empty and not aria2 else "incomplete"
    print(f"{repo}: status={status} shards={len(expected)} missing={len(missing)} empty={len(empty)} aria2={len(aria2)}")
    if missing:
        print("  missing:", ", ".join(missing[:20]))
    if aria2:
        print("  aria2:", ", ".join(aria2[:20]))


def check_plain(repo: str, path: Path, names: list[str]) -> None:
    missing = [name for name in names if not (path / name).exists()]
    aria2 = sorted(p.name for p in path.glob("*.aria2"))
    status = "ok" if not missing and not aria2 else "incomplete"
    print(f"{repo}: status={status} required_files={len(names)} missing={len(missing)} aria2={len(aria2)}")
    if missing:
        print("  missing:", ", ".join(missing))
    if aria2:
        print("  aria2:", ", ".join(aria2[:20]))


def check_models() -> None:
    print_section("Models")
    for repo, path in selected_repos().items():
        if not path.exists():
            print(f"{repo}: status=missing directory={path}")
            continue
        if (path / "model.safetensors.index.json").exists():
            check_index(repo, path)
        elif repo == "Emu3.5-VisionTokenizer":
            check_plain(repo, path, ["config.yaml", "model.ckpt"])
        else:
            aria2 = sorted(p.name for p in path.glob("*.aria2"))
            print(f"{repo}: status=no_index aria2={len(aria2)} directory={path}")


def load_latest_logging(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    latest = None
    with path.open(encoding="utf-8") as src:
        for line in src:
            if line.strip():
                latest = json.loads(line)
    return latest


def check_answer_training() -> None:
    print_section("AR Answer Training")
    log_path = ANSWER_SWIFT / "logging.jsonl"
    latest = load_latest_logging(log_path)
    if latest is None:
        print(f"status=missing_log log={log_path}")
    else:
        step_pair = parse_step_pair(latest.get("global_step/max_steps"))
        train_speed = latest.get("train_speed(s/it)")
        print(
            "status=running_or_recent "
            f"step={latest.get('global_step/max_steps')} "
            f"loss={latest.get('loss')} "
            f"token_acc={latest.get('token_acc')} "
            f"memory_GiB={latest.get('memory(GiB)')} "
            f"remaining={latest.get('remaining_time')}"
        )
        if step_pair is not None:
            step, max_steps = step_pair
            next_checkpoint = ((step // 500) + 1) * 500
            if next_checkpoint > max_steps:
                next_checkpoint = max_steps
            steps_to_next = max(0, next_checkpoint - step)
            eta_seconds = None
            if isinstance(train_speed, (int, float)):
                eta_seconds = steps_to_next * float(train_speed)
            print(
                f"next_checkpoint=checkpoint-{next_checkpoint} "
                f"steps_to_next={steps_to_next} "
                f"eta_to_next={format_duration(eta_seconds)}"
            )
    checkpoints = sorted(ANSWER_SWIFT.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if checkpoints:
        print("checkpoints=" + ", ".join(p.name for p in checkpoints))
    else:
        print("checkpoints=none")

    validation_pid = read_pid(ANSWER_RUN / "checkpoint_validation_watcher.pid")
    print(f"checkpoint_validation_watcher={pid_status(validation_pid)}")
    validation_reports = sorted(
        ANSWER_RUN.glob("checkpoint-*.validation.json"),
        key=lambda p: int(p.name.split("-")[1].split(".")[0]),
    )
    valid_names: list[str] = []
    invalid_names: list[str] = []
    stale_names: list[str] = []
    for report_path in validation_reports:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            invalid_names.append(report_path.name.replace(".validation.json", ""))
            continue
        checkpoint_name = report_path.name.replace(".validation.json", "")
        model_path = Path(str(report.get("model_path", "")))
        if report.get("ok") is True and model_path.is_dir():
            valid_names.append(checkpoint_name)
        elif report.get("ok") is True:
            stale_names.append(checkpoint_name)
        else:
            invalid_names.append(checkpoint_name)
    print("validated_checkpoints=" + (", ".join(valid_names) if valid_names else "none"))
    if valid_names:
        print(f"latest_valid_checkpoint={valid_names[-1]}")
    if stale_names:
        print("stale_validation_reports=" + ", ".join(stale_names))
    if invalid_names:
        print("invalid_checkpoints=" + ", ".join(invalid_names))


def model_ready(repo: str) -> bool:
    path = REPOS[repo]
    if not path.exists() or list(path.glob("*.aria2")):
        return False
    if (path / "model.safetensors.index.json").exists():
        try:
            index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        return all((path / name).is_file() for name in set(index.get("weight_map", {}).values()))
    if repo == "Emu3.5-VisionTokenizer":
        return (path / "config.yaml").is_file() and (path / "model.ckpt").is_file()
    return any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))


def count_jsonl(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", errors="replace") as src:
        return sum(1 for line in src if line.strip())


def check_onevl_latent_readiness() -> None:
    print_section("OneVL Latent Readiness")
    required_models = {
        "teacher_cot": "Baseline_cot_NAVSIM",
        "future_image_tokens": "Emu3.5-VisionTokenizer",
        "stage0_visual_decoder": "OneVL_visual_decoder_pt",
        "base_qwen": "Qwen3-VL-4B-Instruct",
    }
    blocking: list[str] = []
    for purpose, repo in required_models.items():
        ready = model_ready(repo)
        print(f"{purpose}={repo} ready={str(ready).lower()}")
        if not ready:
            blocking.append(f"{purpose}:{repo}")

    data_files = {
        "future_complete_answer_rows": DATA / "navsim_answer_official_paths_future_complete.jsonl",
        "future_manifest_rows": DATA / "navsim_future_frame_manifest_future_complete.jsonl",
        "future_image_token_dryrun_rows": DATA / "navsim_future_image_tokens_dryrun100.jsonl",
        "latent_skeleton_smoke_rows": DATA / "navsim_onevl_latent_skeleton_smoke16.jsonl",
        "demo_answer_to_latent_check_rows": DATA / "demo_answer_to_onevl_latent_check.jsonl",
    }
    for label, path in data_files.items():
        count = count_jsonl(path)
        if count is None:
            print(f"{label}=missing path={path}")
        else:
            print(f"{label}={count} path={path}")

    if blocking:
        print("status=blocked_by_paused_or_incomplete_weights " + ",".join(blocking))
    else:
        print("status=ready_for_full_latent_dataset_build")


def check_remote_ar_eval() -> None:
    print_section("Remote AR Answer Eval")
    if not REMOTE_AR_EVAL.exists():
        print(f"status=not_configured out_root={REMOTE_AR_EVAL}")
        return

    retry_pid = read_pid(REMOTE_AR_EVAL / "retry_remote_launcher.pid")
    remote_pid = read_pid(REMOTE_AR_EVAL / "remote_launcher.pid")
    print(f"out_root={REMOTE_AR_EVAL}")
    print(f"retry_launcher={pid_status(retry_pid)}")
    print(f"remote_launcher={pid_status(remote_pid)}")

    report_path = REMOTE_AR_EVAL / "pdm_eval.report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(
            "pdm_eval=done "
            f"average_score={report.get('average_score')} "
            f"num_valid={report.get('num_valid')} "
            f"num_failed={report.get('num_failed')} "
            f"csv={report.get('output_csv')}"
        )
    elif (REMOTE_AR_EVAL / "navsim_results_submission.pkl").is_file():
        print("pdm_eval=pending_or_running submission=present")
    else:
        print("pdm_eval=not_started")

    retry_tail = read_text_tail(REMOTE_AR_EVAL / "logs/retry_remote_launcher.log", max_lines=5)
    if retry_tail:
        print("retry_log_tail:")
        for line in retry_tail:
            print(f"  {line}")
    remote_tail = read_text_tail(REMOTE_AR_EVAL / "logs/remote_launcher.outer.log", max_lines=5)
    if remote_tail:
        print("remote_outer_tail:")
        for line in remote_tail:
            print(f"  {line}")


def check_local_post_train_eval() -> None:
    print_section("Local Post-Train AR Answer Eval")
    if not LOCAL_POST_TRAIN_EVAL.exists():
        print(f"status=not_configured latest_link={LOCAL_POST_TRAIN_EVAL}")
        return

    out_root = LOCAL_POST_TRAIN_EVAL.resolve()
    watch_pid = read_pid(out_root / "watch_then_eval.pid")
    eval_pid = read_pid(out_root / "eval_launcher.pid")
    print(f"out_root={out_root}")
    print(f"watcher={pid_status(watch_pid)}")
    print(f"eval_launcher={pid_status(eval_pid)}")

    report_path = out_root / "pdm_eval.report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(
            "pdm_eval=done "
            f"average_score={report.get('average_score')} "
            f"num_valid={report.get('num_valid')} "
            f"num_failed={report.get('num_failed')} "
            f"csv={report.get('output_csv')}"
        )
    elif (out_root / "navsim_results_submission.pkl").is_file():
        print("pdm_eval=pending_or_running submission=present")
    elif eval_pid is not None:
        print("pdm_eval=eval_launcher_started")
    else:
        print("pdm_eval=waiting_for_training_or_gpu")

    watch_tail = read_text_tail(out_root / "logs/watch_then_eval.log", max_lines=5)
    if watch_tail:
        print("watch_log_tail:")
        for line in watch_tail:
            print(f"  {line}")
    eval_tail = read_text_tail(out_root / "logs/eval_launcher.outer.log", max_lines=5)
    if eval_tail:
        print("eval_outer_tail:")
        for line in eval_tail:
            print(f"  {line}")


def check_alignment_report() -> None:
    print_section("NAVSIM Test Alignment")
    if not ALIGNMENT_REPORT.is_file():
        print(f"status=missing report={ALIGNMENT_REPORT}")
        return
    report = json.loads(ALIGNMENT_REPORT.read_text(encoding="utf-8"))
    keys = [
        "rows",
        "missing_local_images",
        "missing_log_images",
        "matched_tokens",
        "metric_cache_tokens",
        "missing_cache_tokens",
        "cache_tokens_not_in_test",
        "duplicate_image_keys",
        "duplicate_tokens",
    ]
    print("status=ok" if all(int(report.get(k, 0)) == 0 for k in keys if k.startswith(("missing", "cache_tokens_not", "duplicate"))) else "status=check")
    for key in keys:
        print(f"{key}={report.get(key)}")


def check_navsim_images() -> None:
    print_section("Demo/Test Images")
    files = [
        ROOT / "OneVL_training/demo_data/navsim/navsim_vis4_text2_demo100.jsonl",
        ROOT / "onevl/test_data/navsim_test.json",
    ]
    for file_path in files:
        total = missing = 0
        if not file_path.is_file():
            print(f"{file_path}: missing file")
            continue
        if file_path.suffix == ".jsonl":
            rows = (json.loads(line) for line in file_path.read_text().splitlines() if line.strip())
        else:
            rows = iter(json.loads(file_path.read_text()))
        for row in rows:
            for image in row.get("images", []):
                total += 1
                candidate = ROOT / "OneVL_training" / image
                if not candidate.exists():
                    mapped = image.replace(
                        "navsim_v1.1_all/dataset/sensor_blobs/trainval",
                        "/mnt/navsim/trainval_sensor_blobs/trainval",
                    ).replace(
                        "navsim_v1.1_all/dataset/sensor_blobs/test",
                        "/mnt/navsim/test_sensor_blobs/test",
                    )
                    if not Path(mapped).exists():
                        missing += 1
        print(f"{file_path}: images={total} missing={missing}")


def check_gpu() -> None:
    print_section("GPU")
    result = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode != 0:
        print(result.stdout.strip())
        return
    print("index,memory.used,memory.total,utilization.gpu")
    print(result.stdout.strip())


def main() -> None:
    check_download_processes()
    check_models()
    check_onevl_latent_readiness()
    check_answer_training()
    check_remote_ar_eval()
    check_local_post_train_eval()
    check_alignment_report()
    check_navsim_images()
    check_gpu()
    print_section("Environment")
    print(f"navsim_dir_exists={Path('/mnt/navsim').is_dir()}")
    print("PYTHONPATH for NAVSIM evaluation should include /mnt")


if __name__ == "__main__":
    main()
