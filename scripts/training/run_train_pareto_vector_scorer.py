#!/usr/bin/env python
from __future__ import annotations

import argparse
import lzma
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.pareto_vector_scorer import ParetoVectorScorer


class ArchiveCandidateDataset(Dataset):
    def __init__(self, archive_path: str):
        self.rows = []
        for path in sorted(Path(archive_path).rglob("*.pkl.xz")):
            with lzma.open(path, "rb") as f:
                record = pickle.load(f)
            candidates = np.asarray(record["candidates"], dtype=np.float32)
            components = record["components"]
            feasibility = record.get("feasibility", {})
            front = np.asarray(record.get("pareto_front_mask", np.zeros(candidates.shape[0])), dtype=np.float32)
            utility = np.asarray(record.get("utility", record["rewards"]), dtype=np.float32)
            for idx in range(candidates.shape[0]):
                target = {
                    "nc": float(np.asarray(components["no_at_fault_collisions"])[idx]),
                    "dac": float(np.asarray(components["drivable_area_compliance"])[idx]),
                    "ttc": float(np.asarray(components["time_to_collision_within_bound"])[idx]),
                    "ep": float(np.asarray(components["ego_progress"])[idx]),
                    "comfort": float(np.asarray(components["history_comfort"])[idx]),
                    "ddc": float(np.asarray(components["driving_direction_compliance"])[idx]),
                    "tlc": float(np.asarray(components["traffic_light_compliance"])[idx]),
                    "feas_cost": float(np.asarray(feasibility.get("feas_cost", np.zeros(candidates.shape[0])))[idx]),
                    "pdms": float(np.asarray(components["pdms"])[idx]),
                    "front": float(front[idx]),
                    "utility": float(utility[idx]),
                }
                self.rows.append((candidates[idx], target))
        if not self.rows:
            raise ValueError(f"No archive candidates found under {archive_path}.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        traj, target = self.rows[idx]
        return torch.as_tensor(traj, dtype=torch.float32), {k: torch.tensor(v, dtype=torch.float32) for k, v in target.items()}


def collate(batch):
    traj = torch.stack([row[0] for row in batch], dim=0)
    targets = {key: torch.stack([row[1][key] for row in batch], dim=0) for key in batch[0][1]}
    return traj, targets


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    count = 0
    for traj, target in loader:
        traj = traj.to(device)
        target = {k: v.to(device) for k, v in target.items()}
        b = traj.shape[0]
        scene = torch.zeros((b, 1, 16), device=device)
        status = torch.zeros((b, 8), device=device)
        command = torch.zeros((b, 3), device=device)
        out = model(scene, status, command, traj)
        metric_loss = (
            F.binary_cross_entropy_with_logits(out["nc_logit"], target["nc"].clamp(0, 1))
            + F.binary_cross_entropy_with_logits(out["dac_logit"], target["dac"].clamp(0, 1))
            + F.binary_cross_entropy_with_logits(out["tlc_logit"], target["tlc"].clamp(0, 1))
            + F.smooth_l1_loss(out["ttc"], target["ttc"].clamp(0, 1))
            + F.smooth_l1_loss(out["_ep"], target["ep"].clamp(0, 1))
            + F.smooth_l1_loss(out["comfort"], target["comfort"].clamp(0, 1))
            + F.smooth_l1_loss(out["ddc"], target["ddc"].clamp(0, 1))
            + F.smooth_l1_loss(out["feas_cost"], target["feas_cost"].clamp_min(0))
            + F.smooth_l1_loss(out["pdms"], target["pdms"].clamp(0, 1))
            + F.binary_cross_entropy_with_logits(out["pareto_front_logit"], target["front"].clamp(0, 1))
            + F.smooth_l1_loss(out["utility"], target["utility"])
        )
        formula = torch.sigmoid(out["nc_logit"]) * torch.sigmoid(out["dac_logit"]) * (
            5.0 * out["_ep"] + 5.0 * out["ttc"] + 2.0 * out["comfort"]
        ) / 12.0
        loss = metric_loss + 0.1 * F.l1_loss(out["pdms"], formula.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu()) * b
        count += b
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Pareto-vector scorer on SG-FPS archive records.")
    parser.add_argument("--archive_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset = ArchiveCandidateDataset(args.archive_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    device = torch.device(args.device)
    model = ParetoVectorScorer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, loader, optimizer, device)
        print(f"epoch={epoch} loss={loss:.6f}")
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, output)
    print(f"saved scorer checkpoint to {output}")


if __name__ == "__main__":
    main()
