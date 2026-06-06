from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.risk_vla.safe_alignment_losses import safe_alignment_total_loss


def forbidden_training_split(split: str) -> bool:
    split_l = split.lower()
    return "test" in split_l or "navtest" in split_l


def load_pairs(path: Path, purpose: str) -> Dict[str, Dict[str, Any]]:
    pairs: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            split = str(row.get("split") or "")
            if purpose == "training" and forbidden_training_split(split):
                raise RuntimeError(f"Refusing to train safe alignment from split={split!r} at {path}:{line_number}")
            token = str(row.get("sample_token") or row.get("token_id"))
            pairs[token] = row
    return pairs


class SafeAlignmentTrajectoryAdapter(nn.Module):
    def __init__(self, horizon: int = 8, action_dim: int = 3, hidden_dim: int = 128, residual_scale: float = 0.5) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.residual_scale = float(residual_scale)
        dim = self.horizon * self.action_dim
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, base_traj: torch.Tensor) -> torch.Tensor:
        flat = base_traj.reshape(base_traj.shape[0], -1)
        residual = self.net(flat).reshape_as(base_traj)
        return base_traj + residual * self.residual_scale


class SafeAlignmentDataset(Dataset):
    def __init__(self, candidate_npz: Path, pairs_jsonl: Path, purpose: str) -> None:
        data = np.load(candidate_npz, allow_pickle=False)
        self.tokens = [str(token) for token in data["sample_tokens"].tolist()]
        self.trajectories = data["trajectories"].astype(np.float32)
        pairs = load_pairs(pairs_jsonl, purpose)
        self.items: List[Dict[str, Any]] = []
        num_candidates = int(self.trajectories.shape[1])
        for index, token in enumerate(self.tokens):
            pair = pairs.get(token)
            if not pair:
                continue
            positive_id = int(pair.get("positive_candidate_id", 0))
            negative_id = int(pair.get("negative_candidate_id", 0))
            if not (0 <= positive_id < num_candidates and 0 <= negative_id < num_candidates):
                continue
            self.items.append({"index": index, "token": token, "pair": pair, "positive_id": positive_id, "negative_id": negative_id})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        row = self.items[item]
        traj = torch.from_numpy(self.trajectories[row["index"]]).float()
        pair = row["pair"]
        safety = bool(pair.get("tail_risk_label") or pair.get("negative_regresses_nc") or pair.get("negative_regresses_ttc"))
        return {
            "token": row["token"],
            "base": traj[0],
            "positive": traj[row["positive_id"]],
            "negative": traj[row["negative_id"]],
            "safety_mask": torch.tensor(safety, dtype=torch.bool),
            "path_repair_mask": torch.tensor(bool(pair.get("positive_repairs_path")), dtype=torch.bool),
        }


def collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "tokens": [item["token"] for item in batch],
        "base": torch.stack([item["base"] for item in batch], dim=0),
        "positive": torch.stack([item["positive"] for item in batch], dim=0),
        "negative": torch.stack([item["negative"] for item in batch], dim=0),
        "safety_mask": torch.stack([item["safety_mask"] for item in batch], dim=0),
        "path_repair_mask": torch.stack([item["path_repair_mask"] for item in batch], dim=0),
    }


@torch.no_grad()
def evaluate(model: SafeAlignmentTrajectoryAdapter, loader: DataLoader, device: torch.device, weights: Dict[str, float]) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    kd_losses: List[float] = []
    margin_losses: List[float] = []
    for batch in loader:
        base = batch["base"].to(device)
        pred = model(base)
        out = safe_alignment_total_loss(
            pred,
            batch["positive"].to(device),
            batch["negative"].to(device),
            batch["safety_mask"].to(device),
            path_repair_mask=batch["path_repair_mask"].to(device),
            path_anchor_traj=batch["positive"].to(device),
            weights=weights,
        )
        losses.append(float(out["safealign_total_loss"].detach().cpu()))
        kd_losses.append(float(out["safealign_positive_kd_loss"].detach().cpu()))
        margin_losses.append(float(out["safealign_negative_margin_loss"].detach().cpu()))
    return {
        "loss": mean(losses) if losses else math.nan,
        "positive_kd": mean(kd_losses) if kd_losses else math.nan,
        "negative_margin": mean(margin_losses) if margin_losses else math.nan,
        "num_batches": len(losses),
    }


def write_report(path: Path, summary: Dict[str, Any]) -> None:
    best = summary["best_val"]
    lines = [
        "# Safe Alignment SFT Report",
        "",
        f"Candidate cache: `{summary['candidate_npz']}`",
        f"Pairs: `{summary['pairs_jsonl']}`",
        f"Train/val pairs: `{summary['num_train']}` / `{summary['num_val']}`",
        f"Best epoch: `{summary['best_epoch']}`",
        f"Best val loss: `{best['loss']:.6f}`",
        f"Best positive KD: `{best['positive_kd']:.6f}`",
        f"Best negative margin: `{best['negative_margin']:.6f}`",
        "",
        "GRPO is not run here. This is the supervised safety gate stage.",
        "Leakage guard: training mode rejects split names containing `test` or `navtest` when present in pair rows.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train supervised negative-enhanced safe alignment from candidate anchors.")
    parser.add_argument("--candidate-npz", type=Path, required=True)
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--purpose", choices=("training", "analysis"), default="training")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-real-samples", type=int, default=0)
    parser.add_argument("--debug-allow-small", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    dataset = SafeAlignmentDataset(args.candidate_npz, args.pairs_jsonl, args.purpose)
    if args.dry_run:
        print(json.dumps({"pairs": len(dataset), "would_write": str(args.output_dir)}, sort_keys=True))
        return 0
    if len(dataset) == 0:
        raise RuntimeError("No valid safe-alignment pairs matched the candidate cache.")
    if args.min_real_samples and len(dataset) < args.min_real_samples and not args.debug_allow_small:
        raise RuntimeError(
            f"Refusing formal safe-alignment SFT with {len(dataset)} matched pairs; "
            f"minimum is {args.min_real_samples}. Use --debug-allow-small only for debug-only runs."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    num_val = max(1, int(round(len(indices) * float(args.val_fraction)))) if len(indices) > 1 else 1
    val_indices = indices[:num_val]
    train_indices = indices[num_val:] or val_indices
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_indices), batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    horizon = int(dataset.trajectories.shape[2])
    action_dim = int(dataset.trajectories.shape[3])
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = SafeAlignmentTrajectoryAdapter(horizon=horizon, action_dim=action_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    weights = {"positive_kd": 1.0, "negative_margin": 1.0, "path_anchor": 0.25}
    best_val: Optional[Dict[str, float]] = None
    best_epoch = -1
    with (args.output_dir / "train_log.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_losses: List[float] = []
            for batch in train_loader:
                pred = model(batch["base"].to(device))
                out = safe_alignment_total_loss(
                    pred,
                    batch["positive"].to(device),
                    batch["negative"].to(device),
                    batch["safety_mask"].to(device),
                    path_repair_mask=batch["path_repair_mask"].to(device),
                    path_anchor_traj=batch["positive"].to(device),
                    weights=weights,
                )
                optimizer.zero_grad(set_to_none=True)
                out["safealign_total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                train_losses.append(float(out["safealign_total_loss"].detach().cpu()))
            val = evaluate(model, val_loader, device, weights)
            row = {"epoch": epoch, "train_loss": mean(train_losses) if train_losses else math.nan, **{f"val_{k}": v for k, v in val.items()}}
            log.write(json.dumps(row, sort_keys=True) + "\n")
            log.flush()
            if best_val is None or val["loss"] < best_val["loss"]:
                best_val = val
                best_epoch = epoch
                torch.save({"model": model.state_dict(), "args": vars(args), "val": val, "epoch": epoch}, args.output_dir / "best_safealign.pt")
    final_val = evaluate(model, val_loader, device, weights)
    torch.save({"model": model.state_dict(), "args": vars(args), "val": final_val, "epoch": args.epochs}, args.output_dir / "latest_safealign.pt")
    summary = {
        "candidate_npz": str(args.candidate_npz),
        "pairs_jsonl": str(args.pairs_jsonl),
        "output_dir": str(args.output_dir),
        "num_train": len(train_indices),
        "num_val": len(val_indices),
        "best_epoch": best_epoch,
        "best_val": best_val if best_val is not None else final_val,
        "final_val": final_val,
        "loss_weights": weights,
        "device": str(device),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(args.output_dir / "safe_alignment_sft_report.md", summary)
    print(json.dumps({"summary": str(args.output_dir / "summary.json"), "best_epoch": best_epoch, "best_val": summary["best_val"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
