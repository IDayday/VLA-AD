#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import lzma
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.offline_rl_buffer import save_elite_record
from navsim.agents.recogdrive.pareto_support import (
    CandidateRecord,
    build_archive_record,
    select_feasible_pareto_support,
)
from navsim.agents.recogdrive.pareto_vector_scorer import ParetoVectorScorer


def _load_records(path: Path):
    for item in sorted(path.rglob("*.pkl.xz")):
        with lzma.open(item, "rb") as f:
            yield pickle.load(f)


def _candidate_records(record: dict) -> list[CandidateRecord]:
    candidates = np.asarray(record["candidates"], dtype=np.float32)
    components = record["components"]
    feasibility = record.get("feasibility", {})
    sources = record.get("sources", ["archive"] * candidates.shape[0])
    selection = np.asarray(record.get("selection_score", record["rewards"]), dtype=np.float32)
    out = []
    for idx in range(candidates.shape[0]):
        comp = {key: float(np.asarray(value)[idx]) for key, value in components.items()}
        feas = {key: float(np.asarray(value)[idx]) for key, value in feasibility.items()}
        out.append(
            CandidateRecord(
                trajectory=candidates[idx],
                source=str(sources[idx]),
                token=str(record["token"]),
                components=comp,
                reward=float(comp.get("pdms", record["rewards"][idx])),
                feas=feas,
                selection_score=float(selection[idx]),
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Refine support with a scorer proposal step. This script only writes labels already present in "
            "the input archive; scorer predictions are not written as final evaluator labels."
        )
    )
    parser.add_argument("--candidate_archive_path", required=True)
    parser.add_argument("--scorer_checkpoint_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--pred_top_k", type=int, default=24)
    parser.add_argument("--support_top_m", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = ParetoVectorScorer().to(device)
    ckpt = torch.load(args.scorer_checkpoint_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    output = Path(args.output_path)
    output.mkdir(parents=True, exist_ok=True)
    regrets = []
    oracle_gaps = []
    for record in _load_records(Path(args.candidate_archive_path)):
        candidates = _candidate_records(record)
        traj = torch.as_tensor(np.stack([c.trajectory for c in candidates], axis=0), dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(
                torch.zeros((traj.shape[0], 1, 16), device=device),
                torch.zeros((traj.shape[0], 8), device=device),
                torch.zeros((traj.shape[0], 3), device=device),
                traj,
            )
            score = (pred["utility"] + pred["pdms"]).detach().cpu().numpy()
        keep_idx = np.argsort(-score)[: min(int(args.pred_top_k), len(candidates))]
        proposed = [candidates[int(i)] for i in keep_idx]
        selected = select_feasible_pareto_support(proposed, proposed[0].components if proposed else {}, {"support_top_m": args.support_top_m})
        oracle = max((float(c.reward) for c in candidates), default=0.0)
        best_proposed = max((float(c.reward) for c in proposed), default=0.0)
        best_selected = max((float(c.reward) for c in selected), default=0.0)
        regrets.append(oracle - best_selected)
        oracle_gaps.append(oracle - best_proposed)
        refined = build_archive_record(str(record["token"]), proposed, selected, ref=proposed[0].components if proposed else {}, cfg={"support_top_m": args.support_top_m})
        save_elite_record(output, str(record["token"]), refined)

    summary = {
        "scene_count": len(regrets),
        "scorer_selection_regret": float(np.mean(regrets)) if regrets else 0.0,
        "oracle_topk_gap": float(np.mean(oracle_gaps)) if oracle_gaps else 0.0,
    }
    with open(output / "refine_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
