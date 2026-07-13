#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import atomic_torch_save, load_sample, write_index, write_json  # noqa: E402
from navsim.agents.recogdrive.trajectory_text_replay import try_parse_trajectory_answer  # noqa: E402
from scripts.last_vla_v2.two_expert_slot.run_two_expert_vlm_sft import decode_path_tensor  # noqa: E402
from scripts.last_vla_v2.two_expert_slot.two_expert_cache_utils import iter_indexed_records  # noqa: E402


def _image_key(path: Any) -> str:
    if isinstance(path, list):
        path = path[0] if path else ""
    text = str(path).strip()
    text = text.replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    marker = "sensor_blobs/"
    if marker in text:
        return text[text.index(marker) :]
    for prefix in ("NAVSIM/dataset/", "dataset/"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _resolve_image_path(raw_path: Any, navsim_dataset_root: Optional[Path], base_path: Optional[str]) -> str:
    if base_path:
        return base_path
    if isinstance(raw_path, list):
        raw_path = raw_path[0] if raw_path else ""
    text = str(raw_path).strip()
    if text.startswith("./NAVSIM/dataset/"):
        rel = text[len("./NAVSIM/dataset/") :]
    elif text.startswith("NAVSIM/dataset/"):
        rel = text[len("NAVSIM/dataset/") :]
    elif text.startswith("./dataset/"):
        rel = text[len("./dataset/") :]
    elif text.startswith("dataset/"):
        rel = text[len("dataset/") :]
    else:
        rel = text
    if navsim_dataset_root is not None and rel:
        return str(navsim_dataset_root / rel)
    return text


def _build_base_image_index(
    base_chunk_root: Optional[Path],
    *,
    chunk_name_pattern: Optional[str],
    max_base_samples: Optional[int],
) -> Dict[str, Tuple[str, str]]:
    if base_chunk_root is None:
        return {}
    mapping: Dict[str, Tuple[str, str]] = {}
    for _, sample_path, record in iter_indexed_records(base_chunk_root, pattern=chunk_name_pattern, max_records=max_base_samples):
        token = str(record.get("sample_token") or sample_path.stem)
        sample = load_sample(sample_path)
        if "image_path_tensor" not in sample:
            continue
        image_path = decode_path_tensor(sample["image_path_tensor"])
        mapping.setdefault(_image_key(image_path), (token, image_path))
    return mapping


def _build_teacher_image_index(
    jepa_cache_root: Optional[Path],
    *,
    max_teacher_samples: Optional[int],
    wanted_keys: Optional[set[str]] = None,
) -> Dict[str, Tuple[str, str]]:
    if jepa_cache_root is None:
        return {}
    mapping: Dict[str, Tuple[str, str]] = {}
    scanned = 0
    for _, sample_path, record in iter_indexed_records(jepa_cache_root, max_records=max_teacher_samples):
        scanned += 1
        token = str(record.get("sample_token") or sample_path.stem)
        payload = load_sample(sample_path)
        diagnostics = payload.get("jepa_dynamic_teacher_diagnostics")
        if not isinstance(diagnostics, dict):
            continue
        source_paths = diagnostics.get("source_image_paths")
        if not isinstance(source_paths, list) or not source_paths:
            continue
        image_path = str(source_paths[0])
        key = _image_key(image_path)
        if scanned % 10000 == 0:
            print(f"[prepare_replay] scanned_teacher={scanned} matched_teacher={len(mapping)}", file=sys.stderr, flush=True)
        if wanted_keys is not None and key not in wanted_keys:
            continue
        mapping.setdefault(key, (token, image_path))
        if wanted_keys is not None and len(mapping) >= len(wanted_keys):
            break
    return mapping


def _collect_json_image_keys(path: Path, max_samples: Optional[int]) -> set[str]:
    keys = set()
    for row in _iter_jsonl(path, max_samples):
        keys.add(_image_key(row.get("image")))
    return keys


def _conversation_pairs(conversations: List[Dict[str, Any]]) -> Iterable[Tuple[str, str, Optional[str], str]]:
    system_text: Optional[str] = None
    pending_human: Optional[str] = None
    for turn in conversations:
        role = str(turn.get("from", turn.get("role", ""))).lower()
        value = str(turn.get("value", turn.get("content", "")))
        if role == "system":
            system_text = value
        elif role in {"human", "user"}:
            pending_human = value
        elif role in {"gpt", "assistant"} and pending_human is not None:
            yield pending_human, value, system_text, role
            pending_human = None


def _iter_jsonl(path: Path, max_samples: Optional[int]) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples is not None and idx >= int(max_samples):
                return
            line = line.strip()
            if line:
                yield json.loads(line)


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    args.output_root.mkdir(parents=True, exist_ok=True)
    samples_dir = args.output_root / "samples"
    samples_dir.mkdir(exist_ok=True)
    base_image_index = _build_base_image_index(
        args.base_chunk_root,
        chunk_name_pattern=args.chunk_name_pattern,
        max_base_samples=args.max_base_samples,
    )
    wanted_keys = _collect_json_image_keys(args.official_navsim_traj_jsonl, args.max_samples) if args.jepa_cache_root else None
    teacher_image_index = _build_teacher_image_index(
        args.jepa_cache_root,
        max_teacher_samples=args.max_teacher_samples,
        wanted_keys=wanted_keys,
    )
    for key, value in teacher_image_index.items():
        base_image_index.setdefault(key, value)
    records = []
    seen_tokens = set()
    stats = {
        "json_rows": 0,
        "written": 0,
        "matched_base_image": 0,
        "matched_teacher_image": 0,
        "skipped_unmatched": 0,
        "skipped_nonparseable": 0,
        "duplicates": 0,
    }
    source_path = args.official_navsim_traj_jsonl
    for row in _iter_jsonl(source_path, args.max_samples):
        stats["json_rows"] += 1
        raw_image = row.get("image")
        key = _image_key(raw_image)
        matched = base_image_index.get(key)
        if matched is None and args.require_base_match:
            stats["skipped_unmatched"] += 1
            continue
        sample_token = matched[0] if matched is not None else f"official_navsim_traj_{row.get('id', stats['json_rows'] - 1)}"
        if sample_token in seen_tokens:
            stats["duplicates"] += 1
            continue
        conversations = row.get("conversations")
        if not isinstance(conversations, list):
            continue
        first_pair = next(_conversation_pairs(conversations), None)
        if first_pair is None:
            continue
        prompt, answer_text, system_text, answer_role = first_pair
        parsed = try_parse_trajectory_answer(answer_text)
        if not parsed.parse_ok and args.require_trajectory_parse:
            stats["skipped_nonparseable"] += 1
            continue
        image_path = _resolve_image_path(raw_image, args.navsim_dataset_root, matched[1] if matched is not None else None)
        payload: Dict[str, Any] = {
            "sample_token": sample_token,
            "image_path": image_path,
            "prompt": prompt,
            "answer_text": answer_text,
            "trajectory": parsed.trajectory if parsed.trajectory is not None else None,
            "history_trajectory": None,
            "high_command_one_hot": None,
            "status_feature": None,
            "replay_source": "official_navsim_traj",
            "official_recogdrive_stage1": True,
            "prompt_version": "recogdrive_official_navsim_traj",
            "answer_format_version": "recogdrive_official_pt_v1",
            "parse_ok": bool(parsed.parse_ok),
            "parse_error": parsed.error,
            "official_jsonl_row_id": row.get("id"),
            "official_answer_role": answer_role,
            "official_system_text": system_text,
        }
        out_path = samples_dir / f"{sample_token}.pt"
        atomic_torch_save(payload, out_path)
        records.append({"sample_token": sample_token, "path": str(out_path.relative_to(args.output_root)), "replay_source": "official_navsim_traj"})
        seen_tokens.add(sample_token)
        stats["written"] += 1
        if matched is not None:
            if key in teacher_image_index:
                stats["matched_teacher_image"] += 1
            else:
                stats["matched_base_image"] += 1
    write_index(args.output_root, records)
    metadata = {
        "schema": "recogdrive_stage1_replay_cache_v1",
        "source": "official_navsim_traj",
        "official_recogdrive_stage1": True,
        "official_navsim_traj_jsonl": str(source_path),
        "base_chunk_root": str(args.base_chunk_root) if args.base_chunk_root else None,
        "jepa_cache_root": str(args.jepa_cache_root) if args.jepa_cache_root else None,
        "require_base_match": bool(args.require_base_match),
        "require_trajectory_parse": bool(args.require_trajectory_parse),
        "stats": stats,
    }
    write_json(args.output_root / "metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare indexed Stage1 replay cache from official NAVSIM-Traj JSONL.")
    parser.add_argument("--official-navsim-traj-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-chunk-root", type=Path, default=None)
    parser.add_argument("--jepa-cache-root", type=Path, default=None)
    parser.add_argument("--chunk-name-pattern", default=None)
    parser.add_argument("--navsim-dataset-root", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-base-samples", type=int, default=None)
    parser.add_argument("--max-teacher-samples", type=int, default=None)
    parser.add_argument("--require-base-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-trajectory-parse", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = prepare(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if int(result["stats"]["written"]) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
