from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.risk_vla.critic_losses import critic_total_loss
from navsim.agents.recogdrive.risk_vla.trajectory_risk_critic import TrajectoryRiskCritic


RISK_KEYS = ("low_score", "path_dac", "interaction_nc", "ttc", "progress", "comfort")
DELTA_KEYS = ("delta_score", "delta_dac", "delta_nc", "delta_ttc", "delta_progress", "delta_comfort")


def split_is_forbidden(split: str) -> bool:
    split_l = split.lower()
    return "test" in split_l or "navtest" in split_l


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_label_rows(path: Path, purpose: str) -> Dict[str, List[Dict[str, Any]]]:
    rows_by_token: Dict[str, List[Dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            split = str(row.get("split") or "")
            if purpose == "training" and split_is_forbidden(split):
                raise RuntimeError(f"Refusing to train critic from split={split!r} at {path}:{line_number}")
            token = str(row["sample_token"])
            rows_by_token.setdefault(token, []).append(row)
    for token, rows in rows_by_token.items():
        rows.sort(key=lambda item: int(item["candidate_id"]))
    return rows_by_token


def risk_vector(row: Dict[str, Any], low_score_threshold: float, progress_threshold: float, comfort_threshold: float, zero_eps: float) -> List[float]:
    score = as_float(row.get("pdm_score"))
    dac = as_float(row.get("dac"), 1.0)
    nc = as_float(row.get("nc"), 1.0)
    ttc = as_float(row.get("ttc"), 1.0)
    progress = as_float(row.get("progress"), 1.0)
    comfort = as_float(row.get("comfort"), 1.0)
    return [
        float(score <= low_score_threshold),
        float(dac <= zero_eps),
        float(nc <= zero_eps),
        float(ttc <= zero_eps),
        float(progress < progress_threshold),
        float(comfort < comfort_threshold),
    ]


def metric_vector(row: Dict[str, Any]) -> List[float]:
    return [
        as_float(row.get("pdm_score")),
        as_float(row.get("dac"), 1.0),
        as_float(row.get("nc"), 1.0),
        as_float(row.get("ttc"), 1.0),
        as_float(row.get("progress"), 1.0),
        as_float(row.get("comfort"), 1.0),
    ]


class CandidateAssetDataset(Dataset):
    def __init__(
        self,
        candidate_npz: Path,
        labels_jsonl: Path,
        indices: Sequence[int],
        purpose: str,
        low_score_threshold: float,
        progress_threshold: float,
        comfort_threshold: float,
        zero_eps: float,
    ) -> None:
        data = np.load(candidate_npz, allow_pickle=False)
        self.tokens = [str(item) for item in data["sample_tokens"].tolist()]
        self.trajectories = data["trajectories"].astype(np.float32)
        self.labels = load_label_rows(labels_jsonl, purpose)
        self.indices = list(indices)
        self.low_score_threshold = float(low_score_threshold)
        self.progress_threshold = float(progress_threshold)
        self.comfort_threshold = float(comfort_threshold)
        self.zero_eps = float(zero_eps)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        idx = self.indices[item]
        token = self.tokens[idx]
        traj = self.trajectories[idx]
        rows = self.labels[token]
        num_candidates = traj.shape[0]
        if len(rows) != num_candidates:
            raise ValueError(f"Token {token} has {len(rows)} labels but {num_candidates} trajectories")
        scene_labels = np.asarray(
            [risk_vector(row, self.low_score_threshold, self.progress_threshold, self.comfort_threshold, self.zero_eps) for row in rows],
            dtype=np.float32,
        )
        deltas = np.asarray([[as_float(row.get(key)) for key in DELTA_KEYS] for row in rows], dtype=np.float32)
        metrics = np.asarray([metric_vector(row) for row in rows], dtype=np.float32)
        positive = int(rows[0].get("positive_anchor"))
        negative = int(rows[0].get("negative_anchor"))
        return {
            "token": token,
            "trajectories": torch.from_numpy(traj),
            "scene_labels": torch.from_numpy(scene_labels),
            "horizon_labels": torch.from_numpy(np.repeat(scene_labels[:, None, :], traj.shape[1], axis=1)),
            "deltas": torch.from_numpy(deltas),
            "metrics": torch.from_numpy(metrics),
            "positive": torch.tensor(positive, dtype=torch.long),
            "negative": torch.tensor(negative, dtype=torch.long),
        }


def collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "tokens": [item["token"] for item in batch],
        "trajectories": torch.stack([item["trajectories"] for item in batch], dim=0),
        "scene_labels": torch.stack([item["scene_labels"] for item in batch], dim=0),
        "horizon_labels": torch.stack([item["horizon_labels"] for item in batch], dim=0),
        "deltas": torch.stack([item["deltas"] for item in batch], dim=0),
        "metrics": torch.stack([item["metrics"] for item in batch], dim=0),
        "positive": torch.stack([item["positive"] for item in batch], dim=0),
        "negative": torch.stack([item["negative"] for item in batch], dim=0),
    }


def average_precision(labels: Sequence[float], scores: Sequence[float]) -> Optional[float]:
    positives = sum(1 for value in labels if value > 0.5)
    if positives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    hit = 0
    total = 0.0
    for rank, idx in enumerate(order, start=1):
        if labels[idx] > 0.5:
            hit += 1
            total += hit / rank
    return total / positives


def roc_auc(labels: Sequence[float], scores: Sequence[float]) -> Optional[float]:
    pairs = [(float(score), float(label)) for label, score in zip(labels, scores)]
    positives = sum(1 for _, label in pairs if label > 0.5)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None
    pairs.sort(key=lambda item: item[0])
    rank_sum = 0.0
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and pairs[end][0] == pairs[idx][0]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        rank_sum += avg_rank * sum(1 for _, label in pairs[idx:end] if label > 0.5)
        idx = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def metric_summary(values: torch.Tensor) -> Dict[str, float]:
    arr = values.detach().cpu().float().numpy()
    return {
        "pdms": float(arr[:, 0].mean()),
        "dac0": int((arr[:, 1] <= 1e-9).sum()),
        "nc0": int((arr[:, 2] <= 1e-9).sum()),
        "ttc0": int((arr[:, 3] <= 1e-9).sum()),
        "progress": float(arr[:, 4].mean()),
        "comfort": float(arr[:, 5].mean()),
        "zero_score": int((arr[:, 0] <= 1e-9).sum()),
    }


@torch.no_grad()
def evaluate(
    model: TrajectoryRiskCritic,
    loader: DataLoader,
    device: torch.device,
    planner_dim: int,
    loss_weights: Dict[str, float],
) -> Dict[str, Any]:
    model.eval()
    losses: List[float] = []
    pair_correct = 0
    selection_correct = 0
    total = 0
    all_risk_labels: List[List[float]] = [[] for _ in RISK_KEYS]
    all_risk_scores: List[List[float]] = [[] for _ in RISK_KEYS]
    selected_metrics: List[torch.Tensor] = []
    base_metrics: List[torch.Tensor] = []
    oracle_metrics: List[torch.Tensor] = []
    for batch in loader:
        traj = batch["trajectories"].to(device)
        bsz = traj.shape[0]
        vlm = torch.zeros(bsz, 1, planner_dim, device=device)
        status = torch.zeros(bsz, 8, device=device)
        history = torch.zeros(bsz, 4, 3, device=device)
        command = torch.zeros(bsz, 3, device=device)
        out = model(vlm, status, history, command, traj)
        losses_dict = critic_total_loss(
            out.utility_score,
            out.risk_logits_scene,
            out.risk_logits_horizon,
            out.submetric_delta_pred,
            scene_labels=batch["scene_labels"].to(device),
            horizon_labels=batch["horizon_labels"].to(device),
            submetric_delta_target=batch["deltas"].to(device),
            positive_index=batch["positive"].to(device),
            negative_index=batch["negative"].to(device),
            weights=loss_weights,
        )
        losses.append(float(losses_dict["total_loss"].detach().cpu()))
        pos = batch["positive"].to(device)
        neg = batch["negative"].to(device)
        pair_correct += int((out.utility_score.gather(1, pos.view(-1, 1)).squeeze(1) > out.utility_score.gather(1, neg.view(-1, 1)).squeeze(1)).sum().item())
        selected = out.utility_score.argmax(dim=1)
        selection_correct += int((selected == pos).sum().item())
        total += bsz
        risk_prob = torch.sigmoid(out.risk_logits_scene).detach().cpu()
        labels = batch["scene_labels"].detach().cpu()
        for risk_idx in range(len(RISK_KEYS)):
            all_risk_labels[risk_idx].extend(labels[..., risk_idx].reshape(-1).tolist())
            all_risk_scores[risk_idx].extend(risk_prob[..., risk_idx].reshape(-1).tolist())
        metrics = batch["metrics"]
        selected_metrics.append(metrics[torch.arange(bsz), selected.cpu()])
        base_metrics.append(metrics[:, 0])
        oracle_metrics.append(metrics[torch.arange(bsz), pos.cpu()])
    selected_cat = torch.cat(selected_metrics, dim=0) if selected_metrics else torch.zeros(0, 6)
    base_cat = torch.cat(base_metrics, dim=0) if base_metrics else torch.zeros(0, 6)
    oracle_cat = torch.cat(oracle_metrics, dim=0) if oracle_metrics else torch.zeros(0, 6)
    risk_metrics = {}
    for idx, key in enumerate(RISK_KEYS):
        risk_metrics[f"{key}_auroc"] = roc_auc(all_risk_labels[idx], all_risk_scores[idx])
        risk_metrics[f"{key}_auprc"] = average_precision(all_risk_labels[idx], all_risk_scores[idx])
    return {
        "loss": mean(losses) if losses else math.nan,
        "pairwise_accuracy": pair_correct / max(total, 1),
        "selection_accuracy": selection_correct / max(total, 1),
        "selected_metrics": metric_summary(selected_cat),
        "base_metrics": metric_summary(base_cat),
        "oracle_metrics": metric_summary(oracle_cat),
        "risk_metrics": risk_metrics,
        "num_samples": total,
    }


def write_report(path: Path, summary: Dict[str, Any]) -> None:
    val = summary["best_val"]
    lines = [
        "# Trajectory Risk Critic Training Report",
        "",
        f"Labels: `{summary['labels_jsonl']}`",
        f"Candidates: `{summary['candidate_npz']}`",
        f"Train/val tokens: `{summary['num_train']}` / `{summary['num_val']}`",
        f"Best epoch: `{summary['best_epoch']}`",
        f"Best val loss: `{val['loss']:.6f}`",
        f"Pairwise accuracy: `{val['pairwise_accuracy']:.6f}`",
        f"Selection accuracy vs constrained anchor: `{val['selection_accuracy']:.6f}`",
        "",
        "| Selection | PDMS | Zero | DAC0 | NC0 | TTC0 | Progress | Comfort |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("base_metrics", "selected_metrics", "oracle_metrics"):
        item = val[name]
        label = name.replace("_metrics", "")
        lines.append(
            f"| {label} | {item['pdms']:.6f} | {item['zero_score']} | {item['dac0']} | {item['nc0']} | "
            f"{item['ttc0']} | {item['progress']:.6f} | {item['comfort']:.6f} |"
        )
    lines.extend(["", "Risk AUROC/AUPRC:"])
    for key, value in sorted(val["risk_metrics"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "Leakage guard: training mode rejects split names containing `test` or `navtest`."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train a trajectory-conditioned RiskCritic from candidate-bank utility assets.")
    parser.add_argument("--candidate-npz", type=Path, required=True)
    parser.add_argument("--labels-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--purpose", choices=("training", "analysis"), default="training")
    parser.add_argument("--planner-dim", type=int, default=384)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--low-score-threshold", type=float, default=0.5)
    parser.add_argument("--progress-threshold", type=float, default=0.5)
    parser.add_argument("--comfort-threshold", type=float, default=0.5)
    parser.add_argument("--zero-eps", type=float, default=1e-9)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--min-real-samples", type=int, default=0)
    parser.add_argument("--debug-allow-small", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = np.load(args.candidate_npz, allow_pickle=False)
    num_tokens = len(data["sample_tokens"])
    if args.dry_run:
        print(
            json.dumps(
                {
                    "candidate_npz": str(args.candidate_npz),
                    "labels_jsonl": str(args.labels_jsonl),
                    "num_tokens": num_tokens,
                    "min_real_samples": args.min_real_samples,
                    "would_write": str(args.output_dir),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.min_real_samples and num_tokens < args.min_real_samples and not args.debug_allow_small:
        raise RuntimeError(
            f"Refusing formal critic training with {num_tokens} samples; "
            f"minimum is {args.min_real_samples}. Use --debug-allow-small only for debug-only runs."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = list(range(num_tokens))
    random.shuffle(indices)
    num_val = max(1, int(round(num_tokens * float(args.val_fraction))))
    val_indices = sorted(indices[:num_val])
    train_indices = sorted(indices[num_val:])

    dataset_kwargs = dict(
        candidate_npz=args.candidate_npz,
        labels_jsonl=args.labels_jsonl,
        purpose=args.purpose,
        low_score_threshold=args.low_score_threshold,
        progress_threshold=args.progress_threshold,
        comfort_threshold=args.comfort_threshold,
        zero_eps=args.zero_eps,
    )
    train_dataset = CandidateAssetDataset(indices=train_indices, **dataset_kwargs)
    val_dataset = CandidateAssetDataset(indices=val_indices, **dataset_kwargs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = TrajectoryRiskCritic(
        planner_dim=args.planner_dim,
        hidden_dim=args.hidden_dim,
        num_risk_classes=len(RISK_KEYS),
        horizon=int(data["trajectories"].shape[2]),
        action_dim=int(data["trajectories"].shape[3]),
        num_submetrics=len(DELTA_KEYS),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_weights = {
        "scene_risk_loss": 1.0,
        "horizon_risk_loss": 0.25,
        "submetric_delta_loss": 1.0,
        "pairwise_preference_loss": 2.0,
        "constrained_safety_loss": 0.25,
        "cvar_tail_risk_loss": 0.1,
    }
    train_log_path = output_dir / "train_log.jsonl"
    best_val: Optional[Dict[str, Any]] = None
    best_epoch = -1
    with train_log_path.open("w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_losses: List[float] = []
            for batch in train_loader:
                traj = batch["trajectories"].to(device)
                bsz = traj.shape[0]
                vlm = torch.zeros(bsz, 1, args.planner_dim, device=device)
                status = torch.zeros(bsz, 8, device=device)
                history = torch.zeros(bsz, 4, 3, device=device)
                command = torch.zeros(bsz, 3, device=device)
                out = model(vlm, status, history, command, traj)
                losses = critic_total_loss(
                    out.utility_score,
                    out.risk_logits_scene,
                    out.risk_logits_horizon,
                    out.submetric_delta_pred,
                    scene_labels=batch["scene_labels"].to(device),
                    horizon_labels=batch["horizon_labels"].to(device),
                    submetric_delta_target=batch["deltas"].to(device),
                    positive_index=batch["positive"].to(device),
                    negative_index=batch["negative"].to(device),
                    weights=loss_weights,
                )
                optimizer.zero_grad(set_to_none=True)
                losses["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                epoch_losses.append(float(losses["total_loss"].detach().cpu()))
            val = evaluate(model, val_loader, device, args.planner_dim, loss_weights)
            row = {
                "epoch": epoch,
                "train_loss": mean(epoch_losses) if epoch_losses else math.nan,
                "val_loss": val["loss"],
                "val_pairwise_accuracy": val["pairwise_accuracy"],
                "val_selection_accuracy": val["selection_accuracy"],
                "val_selected_pdms": val["selected_metrics"]["pdms"],
                "val_base_pdms": val["base_metrics"]["pdms"],
                "val_oracle_pdms": val["oracle_metrics"]["pdms"],
            }
            log.write(json.dumps(row, sort_keys=True) + "\n")
            log.flush()
            if epoch % max(1, args.log_every) == 0:
                print(json.dumps(row, sort_keys=True), flush=True)
            if best_val is None or val["loss"] < best_val["loss"]:
                best_val = val
                best_epoch = epoch
                torch.save({"model": model.state_dict(), "args": vars(args), "val": val, "epoch": epoch}, output_dir / "best_critic.pt")
    final_val = evaluate(model, val_loader, device, args.planner_dim, loss_weights)
    torch.save({"model": model.state_dict(), "args": vars(args), "val": final_val, "epoch": args.epochs}, output_dir / "latest_critic.pt")
    summary = {
        "labels_jsonl": str(args.labels_jsonl),
        "candidate_npz": str(args.candidate_npz),
        "output_dir": str(output_dir),
        "num_train": len(train_dataset),
        "num_val": len(val_dataset),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val": best_val if best_val is not None else final_val,
        "final_val": final_val,
        "loss_weights": loss_weights,
        "device": str(device),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(output_dir / "trajectory_risk_critic_report.md", summary)
    print(json.dumps({"summary": str(output_dir / "summary.json"), "best_epoch": best_epoch, "best_val": summary["best_val"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
