#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


RUNS = ("a0", "b5", "b6")
METRIC_MAP = {
    "mean": "mean_pdms",
    "median": "median_pdms",
    "p10": "p10_pdms",
    "zero": "zero_score_count",
    "dac0": "drivable_area_compliance_zero_count",
    "nc0": "no_at_fault_collision_zero_count",
    "ttc0": "time_to_collision_zero_count",
    "ego": "ego_progress_mean",
}


def shell_join(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def resolve_checkpoint(value: str | Path) -> Path:
    text = str(value)
    matches = sorted(Path(item) for item in glob.glob(text))
    if matches:
        candidates = matches
    else:
        path = Path(text)
        candidates = [path]
    expanded: List[Path] = []
    for candidate in candidates:
        if candidate.is_file():
            expanded.append(candidate)
        elif candidate.is_dir():
            for name in ("best.ckpt", "latest.ckpt"):
                child = candidate / name
                if child.is_file():
                    expanded.append(child)
            for suffix in ("*.ckpt", "*.pt", "*.pth", "*.safetensors", "*.bin"):
                expanded.extend(sorted(candidate.rglob(suffix)))
    if not expanded:
        raise FileNotFoundError(f"No checkpoint found for {value}")

    def score(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        value = (1000 if name == "best.ckpt" else 0) + (500 if name == "latest.ckpt" else 0)
        value += 100 if "il" in name else 0
        return value, path.stat().st_size, str(path)

    return sorted(set(expanded), key=score, reverse=True)[0]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(metrics: Dict[str, Any], name: str) -> Optional[float]:
    key = METRIC_MAP[name]
    value = metrics.get(key)
    return None if value is None else float(value)


def eval_cmd(
    *,
    config: Path,
    checkpoint: Path,
    output_dir: Path,
    split: str,
    chunk_cache_root: Path,
    chunk_name_pattern: str,
    metric_cache_dir: Path,
    max_samples: int,
    seed: int,
    mode: str,
    precision: str,
) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/eval_bit_drive_pdm.py",
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--split",
        split,
        "--chunk-cache-root",
        str(chunk_cache_root),
        "--chunk-name-pattern",
        chunk_name_pattern,
        "--metric-cache-dir",
        str(metric_cache_dir),
        "--max-samples",
        str(max_samples),
        "--output-dir",
        str(output_dir),
        "--precision",
        precision,
        "--seed",
        str(seed),
    ]
    if mode == "stochastic":
        cmd.append("--stochastic")
    else:
        cmd.append("--deterministic")
    return cmd


def mean_std(values: List[float]) -> tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def bootstrap_ci(values: List[float], *, samples: int = 1000, alpha: float = 0.05) -> tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = random.Random(20260601)
    means = []
    for _ in range(samples):
        draw = [rng.choice(values) for _item in values]
        means.append(sum(draw) / len(draw))
    means.sort()
    lo = means[int((alpha / 2.0) * len(means))]
    hi = means[min(len(means) - 1, int((1.0 - alpha / 2.0) * len(means)))]
    return float(lo), float(hi)


def write_raw(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "run",
        "mode",
        "metric",
        "mean",
        "std",
        "ci_low",
        "ci_high",
        "delta_vs_a0_mean",
        "delta_vs_a0_std",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_md(path: Path, rows: List[Dict[str, Any]], raw_rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# BiT Seed Stability Summary",
        "",
        "Each row reports mean/std across requested seeds for a fixed deterministic/stochastic mode.",
        "",
        "| Run | Mode | Mean | P10 | Zero | DAC0 | NC0 | TTC0 | Ego | Delta NC0 vs A0 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_key = {(row["run"], row["mode"], row["metric"]): row for row in rows}
    for run in RUNS:
        for mode in sorted({row["mode"] for row in raw_rows}):
            def val(metric_name: str) -> str:
                row = by_key.get((run, mode, metric_name))
                return "n/a" if row is None or row.get("mean") is None else f"{float(row['mean']):.4f}"

            delta_nc = by_key.get((run, mode, "nc0"), {}).get("delta_vs_a0_mean")
            lines.append(
                f"| {run.upper()} | {mode} | {val('mean')} | {val('p10')} | {val('zero')} | "
                f"{val('dac0')} | {val('nc0')} | {val('ttc0')} | {val('ego')} | "
                f"{'n/a' if delta_nc is None else f'{float(delta_nc):.4f}'} |"
            )
    lines.extend([
        "",
        "NC regression is considered unstable if it disappears under seed/mode changes; persistent positive NC0 delta indicates a real safety regression.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_eval_jobs(jobs: List[Dict[str, Any]], *, max_parallel: int, parallel_gpus: List[str]) -> None:
    pending = [job for job in jobs if job["needs_run"]]
    if not pending:
        return
    if max_parallel <= 1:
        for job in pending:
            subprocess.run([str(part) for part in job["cmd"]], check=True)
        return

    active: List[Dict[str, Any]] = []
    next_gpu = 0

    def launch(job: Dict[str, Any]) -> None:
        nonlocal next_gpu
        env = os.environ.copy()
        gpu = None
        if parallel_gpus:
            gpu = parallel_gpus[next_gpu % len(parallel_gpus)]
            next_gpu += 1
            env["CUDA_VISIBLE_DEVICES"] = gpu
        job["output_dir"].mkdir(parents=True, exist_ok=True)
        log_path = job["output_dir"] / "seed_stability_eval.log"
        log_file = log_path.open("w", encoding="utf-8")
        log_file.write(shell_join(job["cmd"]) + "\n")
        log_file.flush()
        print(
            f"launch {job['run']} {job['mode']} seed={job['seed']}"
            + (f" gpu={gpu}" if gpu is not None else ""),
            flush=True,
        )
        proc = subprocess.Popen([str(part) for part in job["cmd"]], stdout=log_file, stderr=subprocess.STDOUT, env=env)
        active.append({"proc": proc, "log_file": log_file, "job": job})

    def reap_one(*, wait: bool) -> bool:
        for item in list(active):
            proc = item["proc"]
            if wait:
                returncode = proc.wait()
            else:
                returncode = proc.poll()
                if returncode is None:
                    continue
            item["log_file"].close()
            active.remove(item)
            job = item["job"]
            if returncode != 0:
                raise subprocess.CalledProcessError(returncode, [str(part) for part in job["cmd"]])
            return True
        return False

    for job in pending:
        while len(active) >= max_parallel:
            reap_one(wait=True)
        launch(job)
    while active:
        reap_one(wait=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A0/B5/B6 seed-stability evals on a matched NAVSIM subset.")
    parser.add_argument("--a0-config", type=Path, required=True)
    parser.add_argument("--b5-config", type=Path, required=True)
    parser.add_argument("--b6-config", type=Path, required=True)
    parser.add_argument("--a0-checkpoint", required=True)
    parser.add_argument("--b5-checkpoint", required=True)
    parser.add_argument("--b6-checkpoint", required=True)
    parser.add_argument("--eval-cache-or-split", default="navtest")
    parser.add_argument("--chunk-cache-root", type=Path, default=Path("/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1"))
    parser.add_argument("--chunk-name-pattern", default="navtest_full_chunk_*")
    parser.add_argument("--metric-cache-dir", type=Path, default=Path("/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1"))
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--modes", default="deterministic")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="fp32")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-if-exists", action="store_true")
    parser.add_argument("--parallel-gpus", default="", help="Comma-separated GPU ids for parallel eval, e.g. 0,1,2,3.")
    parser.add_argument("--max-parallel", type=int, default=1, help="Maximum concurrent eval processes.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configs = {"a0": args.a0_config, "b5": args.b5_config, "b6": args.b6_config}
    checkpoints = {
        "a0": resolve_checkpoint(args.a0_checkpoint),
        "b5": resolve_checkpoint(args.b5_checkpoint),
        "b6": resolve_checkpoint(args.b6_checkpoint),
    }
    seeds = [int(item) for item in parse_csv(args.seeds)]
    modes = parse_csv(args.modes)
    if any(mode not in {"deterministic", "stochastic"} for mode in modes):
        raise ValueError("--modes entries must be deterministic or stochastic.")
    parallel_gpus = parse_csv(args.parallel_gpus)
    if parallel_gpus and args.max_parallel <= 1:
        args.max_parallel = len(parallel_gpus)

    jobs: List[Dict[str, Any]] = []
    for mode in modes:
        for seed in seeds:
            for run in RUNS:
                out_dir = args.output_dir / f"eval_{run}_{mode}_seed{seed}"
                metrics_path = out_dir / "aggregate_metrics.json"
                cmd = eval_cmd(
                    config=configs[run],
                    checkpoint=checkpoints[run],
                    output_dir=out_dir,
                    split=args.eval_cache_or_split,
                    chunk_cache_root=args.chunk_cache_root,
                    chunk_name_pattern=args.chunk_name_pattern,
                    metric_cache_dir=args.metric_cache_dir,
                    max_samples=args.max_samples,
                    seed=seed,
                    mode=mode,
                    precision=args.precision,
                )
                print(shell_join(cmd), flush=True)
                jobs.append({
                    "run": run,
                    "mode": mode,
                    "seed": seed,
                    "output_dir": out_dir,
                    "metrics_path": metrics_path,
                    "cmd": cmd,
                    "needs_run": not (args.skip_if_exists and metrics_path.is_file()),
                })

    if args.dry_run:
        return 0

    run_eval_jobs(jobs, max_parallel=args.max_parallel, parallel_gpus=parallel_gpus)

    raw_rows: List[Dict[str, Any]] = []
    for mode in modes:
        for seed in seeds:
            seed_metrics: Dict[str, Dict[str, Any]] = {}
            for run in RUNS:
                out_dir = args.output_dir / f"eval_{run}_{mode}_seed{seed}"
                metrics_path = out_dir / "aggregate_metrics.json"
                seed_metrics[run] = load_json(metrics_path)
            a0_metrics = seed_metrics["a0"]
            for run in RUNS:
                metrics = seed_metrics[run]
                row: Dict[str, Any] = {
                    "run": run,
                    "mode": mode,
                    "seed": seed,
                    "output_dir": str(args.output_dir / f"eval_{run}_{mode}_seed{seed}"),
                }
                for metric_name in METRIC_MAP:
                    value = metric(metrics, metric_name)
                    row[metric_name] = value
                    base = metric(a0_metrics, metric_name)
                    row[f"delta_{metric_name}_vs_a0"] = None if value is None or base is None else value - base
                raw_rows.append(row)

    write_raw(args.output_dir / "seed_stability_raw.jsonl", raw_rows)
    summary_rows: List[Dict[str, Any]] = []
    for run in RUNS:
        for mode in modes:
            subset = [row for row in raw_rows if row["run"] == run and row["mode"] == mode]
            for metric_name in METRIC_MAP:
                values = [float(row[metric_name]) for row in subset if row.get(metric_name) is not None]
                deltas = [
                    float(row[f"delta_{metric_name}_vs_a0"])
                    for row in subset
                    if row.get(f"delta_{metric_name}_vs_a0") is not None
                ]
                avg, std = mean_std(values)
                delta_avg, delta_std = mean_std(deltas)
                ci_low, ci_high = bootstrap_ci(values)
                summary_rows.append({
                    "run": run,
                    "mode": mode,
                    "metric": metric_name,
                    "mean": avg,
                    "std": std,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "delta_vs_a0_mean": delta_avg,
                    "delta_vs_a0_std": delta_std,
                })
    write_csv(args.output_dir / "seed_stability_summary.csv", summary_rows)
    write_md(args.output_dir / "seed_stability_summary.md", summary_rows, raw_rows)
    print(json.dumps({"output_dir": str(args.output_dir), "raw_rows": len(raw_rows)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
