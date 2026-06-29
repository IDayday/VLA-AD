#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
    PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator


_WORKER_STATE: Dict[str, Any] = {}


def _build_scorer_objects() -> tuple[TrajectorySampling, PDMSimulator, PDMScorer]:
    future_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator = PDMSimulator(proposal_sampling=future_sampling)
    scorer = PDMScorer(proposal_sampling=future_sampling, config=PDMScorerConfig())
    return future_sampling, simulator, scorer


def _load_submission(submission_file_path: Path) -> Dict[str, Any]:
    with open(submission_file_path, "rb") as f:
        payload = pickle.load(f)
    predictions = payload.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != 1:
        raise ValueError("Expected submission pickle with exactly one predictions entry")
    if not isinstance(predictions[0], dict):
        raise ValueError("Expected predictions[0] to be a token -> trajectory dictionary")
    return predictions[0]


def _init_worker(submission_file_path: str, metric_cache_path: str) -> None:
    agent_output = _load_submission(Path(submission_file_path))
    metric_cache_loader = MetricCacheLoader(Path(metric_cache_path), require_existing=True)
    future_sampling, simulator, scorer = _build_scorer_objects()
    _WORKER_STATE.update(
        {
            "agent_output": agent_output,
            "metric_cache_loader": metric_cache_loader,
            "future_sampling": future_sampling,
            "simulator": simulator,
            "scorer": scorer,
        }
    )


def _score_chunk(tokens: List[str]) -> List[Dict[str, Any]]:
    agent_output = _WORKER_STATE["agent_output"]
    metric_cache_loader = _WORKER_STATE["metric_cache_loader"]
    future_sampling = _WORKER_STATE["future_sampling"]
    simulator = _WORKER_STATE["simulator"]
    scorer = _WORKER_STATE["scorer"]

    rows: List[Dict[str, Any]] = []
    for token in tokens:
        score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            trajectory = agent_output[token]
            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=future_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            score_row.update(asdict(pdm_result))
        except Exception:
            print(f"----------- Agent failed for token {token}:", file=sys.stderr)
            traceback.print_exc()
            score_row["valid"] = False
        rows.append(score_row)
    return rows


def _chunk_tokens(tokens: List[str], num_chunks: int) -> List[List[str]]:
    num_chunks = max(1, min(num_chunks, len(tokens)))
    return [tokens[i::num_chunks] for i in range(num_chunks) if tokens[i::num_chunks]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel PDM scoring for a NAVSIM submission pickle.")
    parser.add_argument("--submission-file-path", required=True)
    parser.add_argument("--metric-cache-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--chunk-multiplier", type=int, default=4)
    parser.add_argument("--only-submission-tokens", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--report-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    submission_file_path = Path(args.submission_file_path)
    metric_cache_path = Path(args.metric_cache_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if args.chunk_multiplier < 1:
        raise ValueError("--chunk-multiplier must be >= 1")

    metric_cache_loader = MetricCacheLoader(metric_cache_path, require_existing=True)
    tokens = metric_cache_loader.tokens
    if args.only_submission_tokens:
        agent_output = _load_submission(submission_file_path)
        tokens = [token for token in tokens if token in agent_output]
    if args.max_tokens > 0:
        tokens = tokens[: args.max_tokens]
    if not tokens:
        raise ValueError(f"No metric-cache tokens found under {metric_cache_path}")

    num_workers = min(args.num_workers, len(tokens))
    chunks = _chunk_tokens(tokens, num_workers * args.chunk_multiplier)

    if num_workers == 1:
        _init_worker(str(submission_file_path), str(metric_cache_path))
        chunk_results = [_score_chunk(chunk) for chunk in chunks]
    else:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(str(submission_file_path), str(metric_cache_path)),
        ) as executor:
            chunk_results = list(executor.map(_score_chunk, chunks))

    score_rows: List[Dict[str, Any]] = []
    for chunk_result in chunk_results:
        score_rows.extend(chunk_result)

    pdm_score_df = pd.DataFrame(score_rows)
    numeric_columns = [column for column in pdm_score_df.columns if column not in {"token", "valid"}]
    average_row = pdm_score_df[numeric_columns].mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = bool(pdm_score_df["valid"].all())
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    csv_path = output_dir / f"{timestamp}.csv"
    pdm_score_df.to_csv(csv_path, index=False)

    report = {
        "submission_file_path": str(submission_file_path),
        "metric_cache_path": str(metric_cache_path),
        "output_csv": str(csv_path),
        "num_workers": num_workers,
        "num_tokens": len(tokens),
        "num_valid": int(pdm_score_df.iloc[:-1]["valid"].sum()),
        "num_failed": int((~pdm_score_df.iloc[:-1]["valid"]).sum()),
        "average_valid": bool(average_row["valid"]),
        "average_score": float(average_row["score"]) if "score" in average_row else None,
    }
    print(json.dumps(report, indent=2))
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
