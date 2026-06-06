from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def shard_dirs(root: Path, pattern: str, max_shards: Optional[int]) -> List[Path]:
    dirs = sorted(path for path in root.glob(pattern) if path.is_dir())
    if max_shards is not None:
        dirs = dirs[: int(max_shards)]
    if not dirs:
        raise FileNotFoundError(f"No shard directories matching {pattern!r} under {root}")
    return dirs


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Launch sharded eval_bit_drive_pdm.py jobs.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--shard-pattern", default="shard_[0-9][0-9]")
    parser.add_argument("--metric-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="navtest")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="fp32")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--cuda-devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--max-samples-per-shard", type=int, default=None)
    parser.add_argument("--sleep-between-launches", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    shards = shard_dirs(args.shard_root, args.shard_pattern, args.max_shards)
    if args.start_shard:
        shards = shards[int(args.start_shard) :]
    if args.num_shards is not None:
        shards = shards[: int(args.num_shards)]
    if not shards:
        raise FileNotFoundError("No shards selected after applying --start-shard/--num-shards")
    devices = [item.strip() for item in args.cuda_devices.split(",") if item.strip()]
    if not devices:
        devices = ["0"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    launches: List[Dict[str, object]] = []
    pids: List[str] = []
    for idx, shard in enumerate(shards):
        out = args.output_dir / shard.name
        done = out / "aggregate_metrics.json"
        if args.skip_existing and done.is_file():
            launches.append({"shard": shard.name, "status": "skipped_existing", "output_dir": str(out)})
            continue
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python,
            "scripts/eval_bit_drive_pdm.py",
            "--config",
            str(args.config),
            "--checkpoint",
            str(args.checkpoint),
            "--split",
            args.split,
            "--output-dir",
            str(out),
            "--precision",
            args.precision,
            "--chunk-cache-dir",
            str(shard),
            "--metric-cache-dir",
            str(args.metric_cache_dir),
        ]
        if args.max_samples_per_shard is not None:
            cmd.extend(["--max-samples", str(args.max_samples_per_shard)])
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{args.project_root}:{env.get('PYTHONPATH', '')}"
        env["CUDA_VISIBLE_DEVICES"] = devices[idx % len(devices)]
        launch_row: Dict[str, object] = {
            "shard": shard.name,
            "command": cmd,
            "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
            "output_dir": str(out),
        }
        if args.dry_run:
            launch_row["status"] = "dry_run"
        else:
            log = (out / "run.log").open("w", encoding="utf-8")
            process = subprocess.Popen(cmd, cwd=args.project_root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            log.close()
            launch_row["pid"] = process.pid
            launch_row["status"] = "launched"
            pids.append(f"{shard.name}\t{process.pid}\t{env['CUDA_VISIBLE_DEVICES']}")
            if args.sleep_between_launches > 0:
                time.sleep(float(args.sleep_between_launches))
        launches.append(launch_row)
    (args.output_dir / "launch_manifest.json").write_text(json.dumps(launches, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "pids.tsv").write_text("\n".join(pids) + ("\n" if pids else ""), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "num_shards": len(shards), "num_launched": len(pids), "dry_run": args.dry_run}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
