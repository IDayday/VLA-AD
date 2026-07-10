#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_checkpoint_meta(path: Path) -> Optional[Dict[str, int]]:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(ckpt, dict):
        return None
    epoch = ckpt.get("epoch")
    global_step = ckpt.get("global_step")
    if epoch is None or global_step is None:
        return None
    return {"completed_epoch": int(epoch) + 1, "global_step": int(global_step)}


def stable_file(path: Path, stable_seconds: float) -> bool:
    if not path.is_file():
        return False
    try:
        stat_a = path.stat()
        if stat_a.st_size <= 0 or time.time() - stat_a.st_mtime < stable_seconds:
            return False
        time.sleep(min(max(stable_seconds, 1.0), 15.0))
        stat_b = path.stat()
        return stat_a.st_size == stat_b.st_size and stat_a.st_mtime == stat_b.st_mtime
    except OSError:
        return False


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def copy_checkpoint(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def log(log_path: Path, message: str) -> None:
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror Lightning last.ckpt into fixed epoch_XXX.ckpt files.")
    parser.add_argument("--last-ckpt", type=Path, required=True)
    parser.add_argument("--mirror-dir", type=Path, required=True)
    parser.add_argument("--min-epoch", type=int, required=True)
    parser.add_argument("--max-epoch", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--stable-seconds", type=float, default=20.0)
    parser.add_argument("--state-json", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.mirror_dir.mkdir(parents=True, exist_ok=True)
    state_json = args.state_json or args.mirror_dir / "mirror_state.json"
    log_file = args.log_file or args.mirror_dir / "mirror.log"
    copied: Dict[str, Any] = {}

    while True:
        if stable_file(args.last_ckpt, args.stable_seconds):
            meta = read_checkpoint_meta(args.last_ckpt)
            if meta is not None:
                epoch = meta["completed_epoch"]
                if epoch >= args.min_epoch and (args.max_epoch <= 0 or epoch <= args.max_epoch):
                    dst = args.mirror_dir / f"epoch_{epoch:03d}.ckpt"
                    if not dst.is_file():
                        log(log_file, f"copy epoch={epoch} global_step={meta['global_step']} src={args.last_ckpt} dst={dst}")
                        copy_checkpoint(args.last_ckpt, dst)
                    copied[str(epoch)] = {"checkpoint": str(dst), **meta, "updated_at": utc_now()}
                    write_json(
                        state_json,
                        {
                            "last_ckpt": str(args.last_ckpt),
                            "mirror_dir": str(args.mirror_dir),
                            "min_epoch": args.min_epoch,
                            "max_epoch": args.max_epoch,
                            "copied": copied,
                            "updated_at": utc_now(),
                        },
                    )
                    if args.max_epoch > 0 and epoch >= args.max_epoch:
                        log(log_file, f"reached max_epoch={args.max_epoch}; exiting")
                        return 0
        if args.once:
            return 0
        time.sleep(max(args.poll_seconds, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())
