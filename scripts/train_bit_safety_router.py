#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.bit_safety_router import BitSafetyRouter  # noqa: E402


FEATURES = [
    "endpoint_dx",
    "endpoint_dy",
    "endpoint_dheading",
    "early_x_delta_mean",
    "early_y_delta_mean",
    "early_step_distance_delta",
    "heading_delta_mean",
    "curvature_proxy_delta",
    "a0_score",
    "bit_score",
    "a0_dac",
    "bit_dac",
    "a0_nc",
    "bit_nc",
    "a0_ttc",
    "bit_ttc",
]


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_csv(path: Path) -> tuple[torch.Tensor, torch.Tensor, List[str]]:
    rows = []
    labels = []
    tokens = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            features = [as_float(row.get(name)) for name in FEATURES]
            a0_nc = as_float(row.get("a0_nc"))
            bit_nc = as_float(row.get("bit_nc"))
            a0_ttc = as_float(row.get("a0_ttc"))
            bit_ttc = as_float(row.get("bit_ttc"))
            delta = as_float(row.get("delta_score"))
            safety_ok = not (a0_nc > 1e-9 and bit_nc <= 1e-9) and not (a0_ttc > 1e-9 and bit_ttc <= 1e-9)
            label = 1.0 if delta > 0.0 and safety_ok else 0.0
            rows.append(features)
            labels.append(label)
            tokens.append(str(row.get("sample_token") or row.get("scene_token")))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return torch.tensor(rows, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32), tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple BiT safety router from counterfactual eval features.")
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    x, y, _tokens = load_csv(args.features_csv)
    model = BitSafetyRouter(x.shape[1], hidden_dim=args.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for step in range(args.steps):
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 100 == 0:
            pred = (torch.sigmoid(logits) >= 0.5).float()
            acc = float((pred == y).float().mean().item())
            print(f"step={step + 1} loss={float(loss.item()):.6f} acc={acc:.4f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_dim": x.shape[1], "features": FEATURES}, args.output)
    report = {
        "features_csv": str(args.features_csv),
        "output": str(args.output),
        "samples": int(x.shape[0]),
        "positive_rate": float(y.mean().item()),
    }
    (args.output.parent / "router_train_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
