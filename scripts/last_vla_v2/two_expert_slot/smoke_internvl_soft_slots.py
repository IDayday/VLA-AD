#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.recogdrive_backbone import RecogDriveBackbone  # noqa: E402
from navsim.agents.recogdrive.two_expert_slots import TwoExpertSlotConfig, TwoExpertSoftSlots  # noqa: E402


def _assert_span_valid(span, total: int, name: str) -> None:
    start, end = int(span[0]), int(span[1])
    if not (0 <= start < end <= total):
        raise AssertionError(f"{name} span {span} invalid for sequence length {total}")


def run_smoke(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device(args.device)
    backbone = RecogDriveBackbone(model_type="internvl", checkpoint_path=str(args.vlm_path), device=str(device))
    slots = TwoExpertSoftSlots(TwoExpertSlotConfig(vlm_hidden_dim=int(args.vlm_hidden_dim))).to(device)
    questions = ["<image>\nPredict the ego vehicle trajectory."]
    pixel_values = torch.randn(1, 3, int(args.image_size), int(args.image_size), device=device, dtype=torch.float32)
    zero_values = torch.zeros_like(pixel_values)

    with torch.no_grad():
        normal = backbone.forward(pixel_values, questions, [1])
        out = backbone.forward_with_two_expert_slots(pixel_values, {"questions": questions, "num_patches_list": [1]}, slots)
        out_zero_image = backbone.forward_with_two_expert_slots(zero_values, {"questions": questions, "num_patches_list": [1]}, slots)
        zero_slots = TwoExpertSoftSlots(TwoExpertSlotConfig(vlm_hidden_dim=int(args.vlm_hidden_dim))).to(device)
        for parameter in zero_slots.parameters():
            parameter.zero_()
        out_zero_slots = backbone.forward_with_two_expert_slots(
            pixel_values,
            {"questions": questions, "num_patches_list": [1]},
            zero_slots,
        )

    if out["h_dyn"].shape != (1, 3, 12, int(args.vlm_hidden_dim)):
        raise AssertionError(f"h_dyn shape mismatch: {tuple(out['h_dyn'].shape)}")
    if out["h_geo"].shape != (1, 12, int(args.vlm_hidden_dim)):
        raise AssertionError(f"h_geo shape mismatch: {tuple(out['h_geo'].shape)}")
    if out["image_hidden"] is None or out["image_hidden"].numel() == 0:
        raise AssertionError("image_hidden is empty; inputs_embeds path may not inject vision tokens")
    if out["raw_vlm_hidden"] is None or out["raw_vlm_hidden"].numel() == 0:
        raise AssertionError("raw_vlm_hidden is empty")
    total = int(out["full_hidden_state"].shape[1])
    metadata = out["slot_metadata"]
    for idx, span in enumerate(metadata["absolute_dyn_group_spans"]):
        _assert_span_valid(span, total, f"dyn_group_{idx}")
    _assert_span_valid(metadata["absolute_geo_span"], total, "geo")

    image_delta = (out["raw_vlm_hidden"].float() - out_zero_image["raw_vlm_hidden"].float()).abs().mean().item()
    slot_delta = (out["h_dyn"].float() - out_zero_slots["h_dyn"].float()).abs().mean().item()
    geo_slot_delta = (out["h_geo"].float() - out_zero_slots["h_geo"].float()).abs().mean().item()
    if image_delta <= float(args.threshold):
        raise AssertionError(f"raw hidden did not change with image input: delta={image_delta}")
    if max(slot_delta, geo_slot_delta) <= float(args.threshold):
        raise AssertionError(f"slot hidden did not change when slots were zeroed: dyn={slot_delta}, geo={geo_slot_delta}")

    return {
        "ok": True,
        "normal_forward_hidden_states": bool(getattr(normal, "hidden_states", None)),
        "h_dyn_shape": list(out["h_dyn"].shape),
        "h_geo_shape": list(out["h_geo"].shape),
        "raw_vlm_hidden_shape": list(out["raw_vlm_hidden"].shape),
        "image_hidden_shape": list(out["image_hidden"].shape),
        "image_delta": image_delta,
        "slot_delta": slot_delta,
        "geo_slot_delta": geo_slot_delta,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test real InternVL two-expert soft slot injection.")
    parser.add_argument("--vlm-path", type=Path, default=os.getenv("VLM_PATH"))
    parser.add_argument("--device", default=os.getenv("DEVICE", "cuda"))
    parser.add_argument("--vlm-hidden-dim", type=int, default=1536)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--threshold", type=float, default=1e-6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if os.getenv("RUN_SMOKE", "0") != "1":
        payload = {"ok": False, "status": "not_run", "reason": "Set RUN_SMOKE=1 to load the real InternVL model."}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.vlm_path is None:
        payload = {"ok": False, "status": "not_run", "reason": "VLM_PATH or --vlm-path is required."}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    print(json.dumps(run_smoke(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
