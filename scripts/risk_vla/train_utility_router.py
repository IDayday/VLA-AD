from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.risk_vla.dataclasses import RiskState
from navsim.agents.recogdrive.risk_vla.utility_router import RiskVLAv2UtilityRouter


FEATURE_KEYS = (
    "pdm_score",
    "score",
    "dac",
    "nc",
    "ttc",
    "progress",
    "prog",
    "comfort",
    "delta_score",
    "delta_dac",
    "delta_nc",
    "delta_ttc",
    "delta_progress",
    "delta_comfort",
)


def forbidden_training_split(split: str) -> bool:
    split_l = split.lower()
    return "test" in split_l or "navtest" in split_l


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_groups(labels_jsonl: Path, purpose: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    with labels_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            split = str(row.get("split") or "")
            if purpose == "training" and forbidden_training_split(split):
                raise RuntimeError(f"Refusing to train utility router from split={split!r} at {labels_jsonl}:{line_number}")
            token = str(row.get("token_id") or row.get("sample_token"))
            groups.setdefault(token, []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda item: int(item.get("candidate_id", 0)))
    return groups


def build_strategy_map(groups: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
    names = sorted({str(row.get("strategy_name") or "unknown") for rows in groups.values() for row in rows})
    return {name: index for index, name in enumerate(names)}


def fit_dim(values: torch.Tensor, dim: int) -> torch.Tensor:
    if values.shape[-1] == dim:
        return values
    if values.shape[-1] > dim:
        return values[..., :dim]
    return torch.cat([values, values.new_zeros(*values.shape[:-1], dim - values.shape[-1])], dim=-1)


class UtilityRouterLabelDataset(Dataset):
    def __init__(self, groups: Dict[str, List[Dict[str, Any]]], tokens: Sequence[str], strategy_map: Dict[str, int], planner_dim: int) -> None:
        self.groups = groups
        self.tokens = list(tokens)
        self.strategy_map = strategy_map
        self.planner_dim = int(planner_dim)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        token = self.tokens[index]
        rows = self.groups[token]
        positive_id = str(rows[0].get("positive_anchor"))
        positive = next((row for row in rows if str(row.get("candidate_id")) == positive_id), rows[0])
        raw_features: List[float] = []
        for row in rows:
            raw_features.extend(as_float(row.get(key)) for key in FEATURE_KEYS)
            raw_features.extend(
                [
                    float(bool(row.get("unsafe_any"))),
                    float(bool(row.get("regresses_nc"))),
                    float(bool(row.get("regresses_ttc"))),
                    float(bool(row.get("tail_risk_label"))),
                ]
            )
        feature = fit_dim(torch.tensor(raw_features, dtype=torch.float32), self.planner_dim)
        target = self.strategy_map[str(positive.get("strategy_name") or "unknown")]
        risk_probs = torch.tensor(
            [
                float(any(bool(row.get("tail_risk_label")) for row in rows)),
                float(any(bool(row.get("unsafe_path")) for row in rows)),
                float(any(bool(row.get("unsafe_nc")) for row in rows)),
                float(any(bool(row.get("unsafe_ttc")) for row in rows)),
                float(any(as_float(row.get("progress"), 1.0) < 0.5 for row in rows)),
                float(any(as_float(row.get("comfort"), 1.0) < 0.5 for row in rows)),
            ],
            dtype=torch.float32,
        )
        return {"risk_embedding": feature, "risk_probs": risk_probs, "target_strategy": torch.tensor(target, dtype=torch.long)}


def collate(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "risk_embedding": torch.stack([item["risk_embedding"] for item in batch], dim=0),
        "risk_probs": torch.stack([item["risk_probs"] for item in batch], dim=0),
        "target_strategy": torch.stack([item["target_strategy"] for item in batch], dim=0),
    }


@torch.no_grad()
def evaluate(model: RiskVLAv2UtilityRouter, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    correct = 0
    total = 0
    for batch in loader:
        risk_state = RiskState(
            risk_logits=torch.logit(batch["risk_probs"].to(device).clamp(1e-4, 1 - 1e-4)),
            risk_probs=batch["risk_probs"].to(device),
            risk_embedding=batch["risk_embedding"].to(device),
            uncertainty=None,
            diagnostics={},
        )
        out = model(risk_state, mode="learned_strategy_softmax")
        target = batch["target_strategy"].to(device)
        loss = F.nll_loss(out.strategy_weights.clamp_min(1e-8).log(), target)
        losses.append(float(loss.detach().cpu()))
        correct += int((out.strategy_weights.argmax(dim=1) == target).sum().item())
        total += target.numel()
    return {"loss": mean(losses) if losses else math.nan, "accuracy": correct / max(total, 1), "num_samples": total}


def write_report(path: Path, summary: Dict[str, Any]) -> None:
    best = summary["best_val"]
    lines = [
        "# Utility Router Training Report",
        "",
        f"Labels: `{summary['labels_jsonl']}`",
        f"Train/val tokens: `{summary['num_train']}` / `{summary['num_val']}`",
        f"Strategies: `{summary['strategy_map']}`",
        f"Best epoch: `{summary['best_epoch']}`",
        f"Best val loss: `{best['loss']:.6f}`",
        f"Best val accuracy: `{best['accuracy']:.6f}`",
        "",
        "Leakage guard: training mode rejects split names containing `test` or `navtest`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train RISK-VLA v2 learned utility strategy router from utility labels.")
    parser.add_argument("--labels-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--purpose", choices=("training", "analysis"), default="training")
    parser.add_argument("--planner-dim", type=int, default=384)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-real-samples", type=int, default=0)
    parser.add_argument("--debug-allow-small", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    groups = load_groups(args.labels_jsonl, args.purpose)
    strategy_map = build_strategy_map(groups)
    if args.dry_run:
        print(json.dumps({"tokens": len(groups), "strategies": strategy_map, "would_write": str(args.output_dir)}, sort_keys=True))
        return 0
    if not groups:
        raise RuntimeError(f"No utility-label rows found in {args.labels_jsonl}")
    if args.min_real_samples and len(groups) < args.min_real_samples and not args.debug_allow_small:
        raise RuntimeError(
            f"Refusing formal utility-router training with {len(groups)} tokens; "
            f"minimum is {args.min_real_samples}. Use --debug-allow-small only for debug-only runs."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokens = sorted(groups)
    random.shuffle(tokens)
    num_val = max(1, int(round(len(tokens) * float(args.val_fraction)))) if len(tokens) > 1 else 1
    val_tokens = sorted(tokens[:num_val])
    train_tokens = sorted(tokens[num_val:]) or val_tokens
    train_loader = DataLoader(
        UtilityRouterLabelDataset(groups, train_tokens, strategy_map, args.planner_dim),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        UtilityRouterLabelDataset(groups, val_tokens, strategy_map, args.planner_dim),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = RiskVLAv2UtilityRouter(
        planner_dim=args.planner_dim,
        num_strategies=max(1, len(strategy_map)),
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val: Optional[Dict[str, float]] = None
    best_epoch = -1
    with (args.output_dir / "train_log.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_losses: List[float] = []
            for batch in train_loader:
                risk_state = RiskState(
                    risk_logits=torch.logit(batch["risk_probs"].to(device).clamp(1e-4, 1 - 1e-4)),
                    risk_probs=batch["risk_probs"].to(device),
                    risk_embedding=batch["risk_embedding"].to(device),
                    uncertainty=None,
                    diagnostics={},
                )
                out = model(risk_state, mode="learned_strategy_softmax")
                target = batch["target_strategy"].to(device)
                loss = F.nll_loss(out.strategy_weights.clamp_min(1e-8).log(), target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
            val = evaluate(model, val_loader, device)
            row = {"epoch": epoch, "train_loss": mean(train_losses) if train_losses else math.nan, **{f"val_{k}": v for k, v in val.items()}}
            log.write(json.dumps(row, sort_keys=True) + "\n")
            log.flush()
            if best_val is None or val["loss"] < best_val["loss"]:
                best_val = val
                best_epoch = epoch
                torch.save({"model": model.state_dict(), "args": vars(args), "strategy_map": strategy_map, "val": val}, args.output_dir / "best_router.pt")
    final_val = evaluate(model, val_loader, device)
    torch.save({"model": model.state_dict(), "args": vars(args), "strategy_map": strategy_map, "val": final_val}, args.output_dir / "latest_router.pt")
    summary = {
        "labels_jsonl": str(args.labels_jsonl),
        "output_dir": str(args.output_dir),
        "num_train": len(train_tokens),
        "num_val": len(val_tokens),
        "strategy_map": strategy_map,
        "best_epoch": best_epoch,
        "best_val": best_val if best_val is not None else final_val,
        "final_val": final_val,
        "device": str(device),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(args.output_dir / "utility_router_report.md", summary)
    print(json.dumps({"summary": str(args.output_dir / "summary.json"), "best_epoch": best_epoch, "best_val": summary["best_val"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
