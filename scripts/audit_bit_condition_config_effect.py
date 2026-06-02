#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers.feature_extraction_utils import BatchFeature  # noqa: E402
from scripts.eval_bit_drive_pdm import build_planner  # noqa: E402


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def tensor_summary(value: torch.Tensor) -> Dict[str, Any]:
    detached = value.detach().float().cpu()
    return {
        "shape": list(detached.shape),
        "mean": float(detached.mean().item()) if detached.numel() else 0.0,
        "std": float(detached.std(unbiased=False).item()) if detached.numel() else 0.0,
        "min": float(detached.min().item()) if detached.numel() else 0.0,
        "max": float(detached.max().item()) if detached.numel() else 0.0,
        "l2": float(torch.linalg.norm(detached).item()) if detached.numel() else 0.0,
        "preview": detached.reshape(-1)[:12].tolist(),
    }


def run_config(path: Path, *, seed: int, batch_size: int, context_tokens: int) -> Dict[str, Any]:
    cfg = load_yaml(path)
    torch.manual_seed(seed)
    planner = build_planner(cfg)
    planner.eval()

    generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    dim = int(cfg.get("planner_dim", 384))
    context = torch.randn(batch_size, context_tokens, dim, generator=generator)
    context_mean = context.mean(dim=1)
    action_input = BatchFeature(data={
        "status_feature": torch.randn(batch_size, 8, generator=generator),
        "his_traj": torch.randn(batch_size, 12, generator=generator),
        "action": torch.full((batch_size, 8, 3), float("nan")),
    })

    with torch.no_grad():
        output = planner._build_bit_condition(
            context_mean,
            context,
            action_input,
            gt_actions=None,
            training=False,
        )
        terminal_token, path_tokens, _summary = planner.bit_condition_encoder(
            output["bit_selected_terminal"],
            output["bit_selected_path_anchors"],
            use_path_anchors=bool(planner.config.bit_use_path_anchors and planner.config.bit_enable_path_condition),
        )

    before_tokens = context.shape[1]
    after_tokens = output["context_tokens"].shape[1]
    action_condition = output.get("bit_action_condition")
    return {
        "config": str(path),
        "bit_condition_coordinate_mode": planner.config.bit_condition_coordinate_mode,
        "bit_condition_axis_scale_x": float(planner.config.bit_condition_axis_scale_x),
        "bit_condition_axis_scale_y": float(planner.config.bit_condition_axis_scale_y),
        "bit_condition_axis_scale_heading": float(planner.config.bit_condition_axis_scale_heading),
        "bit_enable_terminal_condition": bool(planner.config.bit_enable_terminal_condition),
        "bit_enable_path_condition": bool(planner.config.bit_enable_path_condition),
        "bit_context_condition_strength": float(planner.config.bit_context_condition_strength),
        "bit_action_condition_strength": float(planner.config.bit_action_condition_strength),
        "selected_terminal_summary": tensor_summary(output["bit_selected_terminal"]),
        "selected_path_summary": tensor_summary(output["bit_selected_path_anchors"]),
        "terminal_token_norm": float(torch.linalg.norm(terminal_token.detach().float()).item()),
        "path_token_norm": float(torch.linalg.norm(path_tokens.detach().float()).item()),
        "context_token_count_before": int(before_tokens),
        "context_token_count_after": int(after_tokens),
        "condition_token_count": int(after_tokens - before_tokens),
        "action_add_enabled": bool(action_condition is not None),
        "context_gate_value": tensor_summary(output["bit_context_gate_value"]),
        "action_gate_value": tensor_summary(output["bit_action_gate_value"]),
    }


def pairwise_diffs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    diffs = []
    tensors = []
    for row in rows:
        terminal = torch.tensor(row["selected_terminal_summary"]["preview"], dtype=torch.float32)
        path = torch.tensor(row["selected_path_summary"]["preview"], dtype=torch.float32)
        tensors.append((terminal, path))
    for i, lhs in enumerate(rows):
        for j in range(i + 1, len(rows)):
            rhs = rows[j]
            t_l, p_l = tensors[i]
            t_r, p_r = tensors[j]
            diffs.append({
                "lhs": Path(lhs["config"]).name,
                "rhs": Path(rhs["config"]).name,
                "terminal_preview_l2": float(torch.linalg.norm(t_l - t_r).item()),
                "path_preview_l2": float(torch.linalg.norm(p_l - p_r).item()),
                "condition_token_count_delta": int(lhs["condition_token_count"] - rhs["condition_token_count"]),
            })
    return diffs


def write_md(path: Path, rows: List[Dict[str, Any]], diffs: List[Dict[str, Any]]) -> None:
    lines = [
        "# BiT v3 Config Effect Audit",
        "",
        "| Config | Mode | Axis Scale | Terminal | Path | Context Strength | Action Strength | Condition Tokens | Action Add | Terminal L2 | Path L2 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {Path(row['config']).name} | {row['bit_condition_coordinate_mode']} | "
            f"{row['bit_condition_axis_scale_x']}/{row['bit_condition_axis_scale_y']}/{row['bit_condition_axis_scale_heading']} | "
            f"{row['bit_enable_terminal_condition']} | {row['bit_enable_path_condition']} | "
            f"{row['bit_context_condition_strength']} | {row['bit_action_condition_strength']} | "
            f"{row['condition_token_count']} | {row['action_add_enabled']} | "
            f"{row['selected_terminal_summary']['l2']:.6f} | {row['selected_path_summary']['l2']:.6f} |"
        )
    lines.extend(["", "## Pairwise Preview Differences", "", "| LHS | RHS | Terminal L2 | Path L2 | Token Count Delta |", "| --- | --- | ---: | ---: | ---: |"])
    for row in diffs:
        lines.append(
            f"| {row['lhs']} | {row['rhs']} | {row['terminal_preview_l2']:.6f} | "
            f"{row['path_preview_l2']:.6f} | {row['condition_token_count_delta']} |"
        )
    meaningful = any(
        diff["terminal_preview_l2"] > 1e-6 or diff["path_preview_l2"] > 1e-6 or diff["condition_token_count_delta"] != 0
        for diff in diffs
    )
    lines.extend([
        "",
        f"Meaningfully different condition tensors: `{meaningful}`.",
        "",
        "This audit uses fixed random weights and a fixed fake batch; differences therefore reflect config plumbing rather than training randomness.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether BiT v3 configs actually change conditioning tensors.")
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--context-tokens", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_config(path, seed=args.seed, batch_size=args.batch_size, context_tokens=args.context_tokens) for path in args.configs]
    diffs = pairwise_diffs(rows)
    payload = {"configs": rows, "pairwise_diffs": diffs}
    (args.output_dir / "config_effect_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_md(args.output_dir / "config_effect_audit.md", rows, diffs)
    print(json.dumps({"output_dir": str(args.output_dir), "configs": len(rows)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
