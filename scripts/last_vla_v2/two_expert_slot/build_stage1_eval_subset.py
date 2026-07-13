#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import write_json  # noqa: E402
from scripts.last_vla_v2.two_expert_slot.two_expert_cache_utils import load_path_index  # noqa: E402


def _str_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _load_index_tokens(root: Optional[Path], *, label: str) -> set[str]:
    if root is None or not Path(root).exists():
        return set()
    try:
        return set(load_path_index(Path(root)).keys())
    except FileNotFoundError:
        print(f"[build_stage1_eval_subset] missing {label}: {root}", file=sys.stderr)
        return set()


def _read_tokens(path: Optional[Path]) -> list[str]:
    if path is None or not Path(path).is_file():
        return []
    tokens: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        token = line.strip().split()[0] if line.strip() else ""
        if token:
            tokens.append(token)
    return tokens


def _stable_order(tokens: Iterable[str], *, seed: int = 0) -> list[str]:
    def key(token: str) -> str:
        return hashlib.sha1(f"{seed}:{token}".encode("utf-8")).hexdigest()

    return sorted(set(str(token) for token in tokens), key=key)


def _write_md(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# Stage1 Eval Subset",
        "",
        f"- status: `{payload['status']}`",
        f"- eval_count: `{payload['eval_count']}`",
        f"- heldout_source: `{payload['heldout_source']}`",
        f"- strict_heldout: `{payload['strict_heldout']}`",
        f"- train_overlap_count: `{payload['train_overlap_count']}`",
        f"- base_chunk_root: `{payload.get('base_chunk_root')}`",
        f"- jepa_cache_root: `{payload.get('jepa_cache_root')}`",
        f"- vggt_cache_root: `{payload.get('vggt_cache_root')}`",
        f"- replay_cache_root: `{payload.get('replay_cache_root')}`",
        "",
        "## Counts",
        "",
        "| source | count |",
        "| --- | ---: |",
    ]
    for key, value in payload["counts"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Notes", ""])
    for note in payload.get("notes", []):
        lines.append(f"- {note}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> Dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_tokens = _load_index_tokens(args.base_chunk_root, label="base chunks")
    jepa_tokens = _load_index_tokens(args.jepa_cache_root, label="JEPA cache")
    vggt_tokens = _load_index_tokens(args.vggt_cache_root, label="VGGT cache")
    replay_tokens = _load_index_tokens(args.replay_cache_root, label="replay cache")

    required_sets = [jepa_tokens, vggt_tokens]
    if replay_tokens:
        required_sets.append(replay_tokens)
    if base_tokens:
        required_sets.append(base_tokens)
    intersection = set.intersection(*required_sets) if required_sets else set()

    preferred = _read_tokens(args.preferred_val_token_file)
    notes: list[str] = []
    strict_heldout = False
    heldout_source = "unknown"
    if preferred:
        preferred_set = set(preferred)
        candidates = [token for token in preferred if token in intersection]
        heldout_source = "preferred_val_token_file"
        strict_heldout = True
        notes.append(f"Preferred token file supplied with {len(preferred_set)} unique tokens.")
    else:
        candidates = _stable_order(intersection, seed=int(args.seed))
        heldout_source = "train_cache_deterministic_subset_not_true_holdout"
        notes.append(
            "No independent validation token file or split metadata was available; "
            "using a deterministic subset from the available cache intersection."
        )

    if not base_tokens:
        notes.append("Base chunk root missing or empty; subset uses JEPA/VGGT/replay intersection.")
    if bool(args.exclude_train_replay_tokens) and not strict_heldout:
        notes.append(
            "--exclude-train-replay-tokens was requested, but replay cache is the only token source; "
            "tokens are retained and train overlap is reported explicitly."
        )

    selected = list(candidates[: int(args.max_samples)])
    train_overlap_count = len(set(selected) & replay_tokens) if replay_tokens else 0
    status = "READY" if len(selected) >= int(args.min_samples) else "FAIL"
    payload: Dict[str, Any] = {
        "status": status,
        "eval_count": len(selected),
        "min_samples": int(args.min_samples),
        "max_samples": int(args.max_samples),
        "heldout_source": heldout_source,
        "strict_heldout": bool(strict_heldout),
        "train_overlap_count": int(train_overlap_count),
        "train_overlap_fraction": float(train_overlap_count / max(1, len(selected))),
        "base_chunk_root": str(args.base_chunk_root) if args.base_chunk_root else None,
        "jepa_cache_root": str(args.jepa_cache_root),
        "vggt_cache_root": str(args.vggt_cache_root),
        "replay_cache_root": str(args.replay_cache_root) if args.replay_cache_root else None,
        "preferred_val_token_file": str(args.preferred_val_token_file) if args.preferred_val_token_file else None,
        "counts": {
            "base": len(base_tokens),
            "jepa": len(jepa_tokens),
            "vggt": len(vggt_tokens),
            "replay": len(replay_tokens),
            "intersection": len(intersection),
            "selected": len(selected),
        },
        "notes": notes,
    }
    (args.output_dir / "stage1_eval_tokens.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
    write_json(args.output_dir / "stage1_eval_subset.json", payload)
    _write_md(args.output_dir / "stage1_eval_subset.md", payload)
    if status != "READY":
        raise RuntimeError(f"Stage1 eval subset has only {len(selected)} samples; required >= {args.min_samples}.")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Stage1-v2 evaluation token subset.")
    parser.add_argument("--base-chunk-root", type=Path, default=None)
    parser.add_argument("--jepa-cache-root", type=Path, required=True)
    parser.add_argument("--vggt-cache-root", type=Path, required=True)
    parser.add_argument("--replay-cache-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--min-samples", type=int, default=1024)
    parser.add_argument("--preferred-val-token-file", type=Path, default=None)
    parser.add_argument("--exclude-train-replay-tokens", nargs="?", const=True, default=True, type=_str_bool)
    parser.add_argument("--no-exclude-train-replay-tokens", dest="exclude_train_replay_tokens", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    result = build(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
