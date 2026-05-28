#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_recogdrive_chunk_cache import autodetect_data_paths, cam_f0_sensor_config, load_scene_filter, split_paths
from navsim.common.dataloader import SceneLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create an indexed local recovery manifest for ReCogDrive cache generation.')
    parser.add_argument('--navsim-root', type=Path, default=Path('/mnt/navsim'))
    parser.add_argument('--split', choices=('navtrain',), default='navtrain')
    parser.add_argument('--current-blobs', type=Path, default=Path('/mnt/navsim/trainval_sensor_blobs/trainval'))
    parser.add_argument('--recovery-blobs', type=Path, default=Path('/mnt/navsim/trainval_all/trainval_sensor_blobs/trainval'))
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'experiments/recogdrive_expert/cache_generation/local_union/data_completeness')
    parser.add_argument('--chunk-size', type=int, default=4096)
    parser.add_argument('--cache-output-root', type=Path, default=REPO_ROOT / 'cache/recogdrive_expert_chunks/full_v1_local_union')
    parser.add_argument('--report-root', type=Path, default=REPO_ROOT / 'experiments/recogdrive_expert/cache_generation/local_union')
    parser.add_argument('--progress-every', type=int, default=5000)
    return parser.parse_args()


def cam_rel_path(frame: Dict[str, Any], camera_key: str = 'cam_f0') -> Path:
    for raw_key, camera in frame['cams'].items():
        if raw_key.lower() == camera_key:
            return Path(camera['data_path'])
    raise KeyError("Camera {} not found. Available: {}".format(camera_key, list(frame["cams"].keys())))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, openscene, _ = autodetect_data_paths(args.navsim_root)
    logs, detected_blobs = split_paths(openscene, args.split)
    scene_filter, scene_filter_path = load_scene_filter(args.split, None)
    loader = SceneLoader(
        data_path=logs,
        sensor_blobs_path=args.recovery_blobs,
        scene_filter=scene_filter,
        sensor_config=cam_f0_sensor_config(),
        load_image_path=True,
    )

    counts: Counter[str] = Counter()
    raw_chunk_counts: Dict[int, Counter[str]] = defaultdict(Counter)
    recoverable_backfill_counter = 0
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    manifest_path = args.output_dir / f'{args.split}_local_recovery_manifest.jsonl'
    recoverable_path = args.output_dir / f'{args.split}_recoverable_backfill_manifest.jsonl'
    unresolved_path = args.output_dir / f'{args.split}_unresolved_missing_manifest.jsonl'

    with manifest_path.open('w', encoding='utf-8') as manifest_fp,             recoverable_path.open('w', encoding='utf-8') as recoverable_fp,             unresolved_path.open('w', encoding='utf-8') as unresolved_fp:
        for raw_index, token in enumerate(loader.tokens):
            if args.progress_every and raw_index and raw_index % args.progress_every == 0:
                print(f'scanned {raw_index}/{len(loader.tokens)}', flush=True)
            counts['tokens'] += 1
            frames = loader.scene_frames_dicts[token]
            h = scene_filter.num_history_frames
            history_frames = frames[:h]
            future_frames = frames[h:h + 4]
            if len(history_frames) < 4 or len(future_frames) < 4:
                status = 'short_required_window'
                record = {
                    'split': args.split,
                    'raw_index': raw_index,
                    'raw_chunk_index': raw_index // args.chunk_size,
                    'loader_token': token,
                    'status': status,
                    'history_frames': len(history_frames),
                    'future_frames': len(future_frames),
                }
                counts[status] += 1
                raw_chunk_counts[raw_index // args.chunk_size][status] += 1
                manifest_fp.write(json.dumps(record, sort_keys=True) + '\n')
                unresolved_fp.write(json.dumps(record, sort_keys=True) + '\n')
                continue

            roles = []
            rels = []
            for i, frame in enumerate(history_frames[-4:]):
                roles.append(f'history_cam_f0_{i}')
                rels.append(cam_rel_path(frame))
            for i, frame in enumerate(future_frames[:4]):
                roles.append(f'future_cam_f0_{i}')
                rels.append(cam_rel_path(frame))

            current_missing = []
            recovery_missing = []
            union_missing = []
            frame_states = []
            for role, rel in zip(roles, rels):
                current_exists = (args.current_blobs / rel).is_file()
                recovery_exists = (args.recovery_blobs / rel).is_file()
                if not current_exists:
                    current_missing.append(str(rel))
                if not recovery_exists:
                    recovery_missing.append(str(rel))
                if not (current_exists or recovery_exists):
                    union_missing.append(str(rel))
                if not (current_exists and recovery_exists):
                    frame_states.append({
                        'role': role,
                        'rel_path': str(rel),
                        'current_exists': current_exists,
                        'recovery_exists': recovery_exists,
                    })

            current_valid = not current_missing
            recovery_valid = not recovery_missing
            union_valid = not union_missing
            if current_valid and recovery_valid:
                status = 'valid_both_roots'
            elif current_valid and not recovery_valid:
                status = 'valid_current_only'
            elif (not current_valid) and recovery_valid:
                status = 'valid_recovery_only'
            elif union_valid:
                status = 'valid_union_only'
            else:
                status = 'missing_both_roots'

            counts[status] += 1
            raw_chunk_index = raw_index // args.chunk_size
            raw_chunk_counts[raw_chunk_index][status] += 1

            is_old_root_skipped_but_local = (not current_valid) and union_valid
            record = {
                'split': args.split,
                'raw_index': raw_index,
                'raw_chunk_index': raw_chunk_index,
                'raw_chunk_start': raw_chunk_index * args.chunk_size,
                'loader_token': token,
                'sample_token': frames[h - 1]['token'],
                'scene_token': frames[h - 1].get('scene_token', token),
                'log_name': frames[h - 1].get('log_name'),
                'status': status,
                'current_valid': current_valid,
                'recovery_valid': recovery_valid,
                'union_valid': union_valid,
                'old_root_skipped_but_local_recoverable': is_old_root_skipped_but_local,
                'current_missing_rel': current_missing,
                'recovery_missing_rel': recovery_missing,
                'union_missing_rel': union_missing,
                'frame_states': frame_states,
            }
            if is_old_root_skipped_but_local:
                record['backfill_chunk_index'] = recoverable_backfill_counter // args.chunk_size
                record['backfill_offset'] = recoverable_backfill_counter % args.chunk_size
                recoverable_backfill_counter += 1

            if status != 'valid_both_roots':
                manifest_fp.write(json.dumps(record, sort_keys=True) + '\n')
                if len(examples[status]) < 5:
                    examples[status].append(record)
            if is_old_root_skipped_but_local:
                recoverable_fp.write(json.dumps(record, sort_keys=True) + '\n')
            if not union_valid or status == 'short_required_window':
                unresolved_fp.write(json.dumps(record, sort_keys=True) + '\n')

    chunk_rows = []
    for chunk_index in sorted(raw_chunk_counts):
        row = {'raw_chunk_index': chunk_index, 'raw_chunk_start': chunk_index * args.chunk_size}
        row.update(dict(raw_chunk_counts[chunk_index]))
        chunk_rows.append(row)

    summary = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'split': args.split,
        'navsim_root': str(args.navsim_root),
        'logs': str(logs),
        'detected_blobs_after_patch': str(detected_blobs),
        'current_blobs': str(args.current_blobs),
        'recovery_blobs': str(args.recovery_blobs),
        'scene_filter': str(scene_filter_path),
        'chunk_size': args.chunk_size,
        'total_tokens': len(loader.tokens),
        'counts': dict(counts),
        'old_root_skipped_but_local_recoverable': recoverable_backfill_counter,
        'union_valid_total': counts['valid_both_roots'] + counts['valid_current_only'] + counts['valid_recovery_only'] + counts['valid_union_only'],
        'union_missing_total': counts['missing_both_roots'] + counts['short_required_window'],
        'estimated_union_train_chunks': math.ceil((counts['valid_both_roots'] + counts['valid_current_only'] + counts['valid_recovery_only'] + counts['valid_union_only']) / args.chunk_size),
        'paths': {
            'manifest': str(manifest_path),
            'recoverable_backfill_manifest': str(recoverable_path),
            'unresolved_missing_manifest': str(unresolved_path),
            'raw_chunk_summary': str(args.output_dir / f'{args.split}_raw_chunk_summary.json'),
        },
        'generation_task': {
            'strategy': 'regenerate full cache with local union sensor roots',
            'cache_output_root': str(args.cache_output_root),
            'report_root': str(args.report_root),
            'primary_root': str(args.recovery_blobs),
            'fallback_roots': [str(args.current_blobs)],
        },
        'examples': examples,
    }
    write_json(args.output_dir / f'{args.split}_summary.json', summary)
    write_json(args.output_dir / f'{args.split}_raw_chunk_summary.json', {'chunks': chunk_rows})
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
