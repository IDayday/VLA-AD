#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.pareto_support.scorer_dataset import ParetoScorerDataset
from navsim.agents.recogdrive.pareto_support.vector_scorer import ParetoVectorScorer, scorer_loss


def _collate(batch):
    out = {}
    for key in batch[0]:
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Pareto vector scorer on evaluator-labeled archives.")
    parser.add_argument("--archive_dir", required=True)
    parser.add_argument("--output_ckpt", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset = ParetoScorerDataset(args.archive_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"No scorer records found in {args.archive_dir}.")
    val_len = max(1, int(0.1 * len(dataset))) if len(dataset) > 10 else 1
    train_len = max(1, len(dataset) - val_len)
    train_ds, val_ds = random_split(dataset, [train_len, len(dataset) - train_len], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate)
    model = ParetoVectorScorer().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch in train_loader:
            trajectory = batch["trajectory"].to(args.device)
            outputs = model(batch["scene_tokens"].to(args.device), None, None, trajectory, batch["source_id"].to(args.device))
            targets = {k: v.to(args.device) for k, v in batch.items() if k not in {"trajectory", "source_id", "scene_tokens"}}
            loss, _ = scorer_loss(outputs, targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss += float(loss.detach()) * trajectory.shape[0]
            train_count += trajectory.shape[0]
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                trajectory = batch["trajectory"].to(args.device)
                outputs = model(batch["scene_tokens"].to(args.device), None, None, trajectory, batch["source_id"].to(args.device))
                targets = {k: v.to(args.device) for k, v in batch.items() if k not in {"trajectory", "source_id", "scene_tokens"}}
                loss, _ = scorer_loss(outputs, targets)
                val_loss += float(loss.detach()) * trajectory.shape[0]
                val_count += trajectory.shape[0]
        row = {"epoch": epoch, "train_loss": train_loss / max(train_count, 1), "val_loss": val_loss / max(val_count, 1)}
        history.append(row)
        print(json.dumps(row, sort_keys=True))
    Path(args.output_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "history": history}, args.output_ckpt)


if __name__ == "__main__":
    main()
