#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import write_json  # noqa: E402
from navsim.agents.recogdrive.recogdrive_backbone import RecogDriveBackbone  # noqa: E402
from navsim.agents.recogdrive.two_expert_slots import TwoExpertSlotConfig  # noqa: E402
from navsim.agents.recogdrive.two_expert_vlm_sft import TwoExpertVLMSFTConfig, TwoExpertVLMSFTModule  # noqa: E402
from scripts.last_vla_v2.two_expert_slot.build_two_expert_hidden_cache import (  # noqa: E402
    load_stage1_state_for_hidden_cache,
)
from scripts.last_vla_v2.two_expert_slot.run_two_expert_vlm_sft import (  # noqa: E402
    TwoExpertStage1Dataset,
    make_collate,
    resolve_cache_roots,
    resolve_vggt_feature_dim,
)


def _load_checkpoint_payload(path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Stage1 checkpoint must be a dict: {path}")
    return payload


def _load_stage1_modules(module: TwoExpertVLMSFTModule, checkpoint: Path) -> Dict[str, Any]:
    payload = _load_checkpoint_payload(checkpoint)
    loaded: Dict[str, Any] = {"checkpoint_schema": payload.get("checkpoint_schema")}
    if isinstance(payload.get("two_expert_slots"), dict):
        module.two_expert_slots.load_state_dict(payload["two_expert_slots"], strict=True)
        loaded["two_expert_slots"] = True
    else:
        loaded["two_expert_slots"] = False
    for attr, key in (
        ("dynamic_adapter", "dynamic_adapter"),
        ("geometry_adapter", "geometry_adapter"),
        ("trajectory_probe", "trajectory_probe"),
    ):
        state = payload.get(key)
        if not isinstance(state, dict):
            raise KeyError(f"Stage1 checkpoint {checkpoint} missing {key}.")
        getattr(module, attr).load_state_dict(state, strict=True)
        loaded[key] = True
    return loaded


def _autocast(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision in {"bf16", "fp16"}
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _metric(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().float()
        if value.numel() != 1:
            value = value.mean()
        return float(value.cpu().item())
    return float(value)


class RunningMean:
    def __init__(self) -> None:
        self.sum: Dict[str, float] = {}
        self.count: Dict[str, int] = {}

    def add(self, name: str, value: Any, weight: int = 1) -> None:
        self.sum[name] = self.sum.get(name, 0.0) + _metric(value) * int(weight)
        self.count[name] = self.count.get(name, 0) + int(weight)

    def result(self) -> Dict[str, float]:
        return {key: self.sum[key] / max(1, self.count[key]) for key in sorted(self.sum)}


def _deterministic_mask_tokens(tokens: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    if mask_ratio <= 0.0:
        return tokens
    if tokens.ndim != 3:
        raise ValueError(f"Expected [B,N,D] tokens, got {tuple(tokens.shape)}")
    length = int(tokens.shape[1])
    if length == 0:
        return tokens
    mask_count = min(length, max(1, int(round(length * float(mask_ratio)))))
    # Stable pseudo-random order independent of process RNG state.
    positions = torch.arange(length, device=tokens.device)
    scores = (positions * 1103515245 + 12345) % 2147483647
    masked_positions = torch.argsort(scores)[:mask_count]
    mask = torch.zeros(length, device=tokens.device, dtype=torch.bool)
    mask[masked_positions] = True
    return torch.where(mask.view(1, length, 1), torch.zeros_like(tokens), tokens)


def _adapter_probe_from_vlm(
    module: TwoExpertVLMSFTModule,
    batch: Dict[str, Any],
    vlm_out: Dict[str, torch.Tensor],
    *,
    image_mode: str = "normal",
    dyn_mode: str = "normal",
    geo_mode: str = "normal",
    high_mask_ratio: float = 0.9,
) -> Dict[str, torch.Tensor]:
    image_hidden = vlm_out["image_hidden"]
    if image_mode == "zero":
        image_hidden = torch.zeros_like(image_hidden)
    elif image_mode == "high_mask":
        image_hidden = _deterministic_mask_tokens(image_hidden, high_mask_ratio)
    elif image_mode != "normal":
        raise ValueError(f"Unknown image_mode={image_mode!r}")
    h_dyn = vlm_out["h_dyn"]
    h_geo = vlm_out["h_geo"]
    if dyn_mode == "zero":
        h_dyn = torch.zeros_like(h_dyn)
    elif dyn_mode != "normal":
        raise ValueError(f"Unknown dyn_mode={dyn_mode!r}")
    if geo_mode == "zero":
        h_geo = torch.zeros_like(h_geo)
    elif geo_mode != "normal":
        raise ValueError(f"Unknown geo_mode={geo_mode!r}")
    dyn_target = batch["jepa_dynamic_teacher_tokens"]
    geo_target = batch["vggt_feature23_tokens"]
    dyn_out = module.dynamic_adapter(image_hidden, h_dyn, dyn_target)
    geo_out = module.geometry_adapter(image_hidden, h_geo, geo_target)
    probe_out = module.trajectory_probe(
        h_dyn,
        h_geo,
        status_feature=batch.get("status_feature"),
        high_command_one_hot=batch.get("high_command_one_hot"),
        history_trajectory=batch.get("history_trajectory"),
        target_action_norm=module._target_action_norm(batch, h_dyn),
    )
    probe_losses = probe_out["losses"]
    hidden_anchor = vlm_out["raw_vlm_hidden"].new_zeros(())
    total = (
        float(module.config.dyn_loss_weight) * dyn_out["loss"]
        + float(module.config.geo_loss_weight) * geo_out["loss"]
        + float(module.config.probe_traj_loss_weight) * probe_losses["probe_loss"]
        + float(module.config.probe_dyn_loss_weight) * probe_losses.get("probe_dyn_loss", hidden_anchor)
        + float(module.config.probe_geo_loss_weight) * probe_losses.get("probe_geo_loss", hidden_anchor)
        + float(module.config.probe_heading_loss_weight) * probe_losses["probe_heading_loss"]
        + float(module.config.probe_progress_loss_weight) * probe_losses["probe_progress_loss"]
        + float(module.config.hidden_anchor_weight) * hidden_anchor
    )
    return {
        "loss": total,
        "dyn_loss": dyn_out["loss"],
        "geo_loss": geo_out["loss"],
        "probe_loss": probe_losses["probe_loss"],
        "probe_dyn_loss": probe_losses.get("probe_dyn_loss", hidden_anchor),
        "probe_geo_loss": probe_losses.get("probe_geo_loss", hidden_anchor),
        "probe_heading_loss": probe_losses["probe_heading_loss"],
        "probe_progress_loss": probe_losses["probe_progress_loss"],
        "h_dyn_norm": h_dyn.detach().float().norm(dim=-1).mean().to(dtype=total.dtype),
        "h_geo_norm": h_geo.detach().float().norm(dim=-1).mean().to(dtype=total.dtype),
        "dyn_pred_pool": dyn_out["pred"].detach().float().mean(dim=(1, 2)),
        "dyn_target_pool": dyn_target.detach().float().mean(dim=(1, 2)),
        "geo_pred_pool": geo_out["pred"].detach().float().mean(dim=1),
        "geo_target_pool": geo_target.detach().float().mean(dim=1),
        **dyn_out["diagnostics"],
        **geo_out["diagnostics"],
    }


def _add_outputs(acc: RunningMean, prefix: str, outputs: Dict[str, Any], batch_size: int) -> None:
    for key, value in outputs.items():
        if key.endswith("_pool"):
            continue
        if isinstance(value, torch.Tensor) and value.numel() >= 1 and torch.is_floating_point(value):
            acc.add(f"{prefix}{key}", value, weight=batch_size)


def _collect_retrieval(store: Dict[str, Dict[str, List[torch.Tensor]]], prefix: str, outputs: Dict[str, Any]) -> None:
    bucket = store.setdefault(prefix, {"dyn_pred": [], "dyn_target": [], "geo_pred": [], "geo_target": []})
    bucket["dyn_pred"].append(outputs["dyn_pred_pool"].detach().cpu())
    bucket["dyn_target"].append(outputs["dyn_target_pool"].detach().cpu())
    bucket["geo_pred"].append(outputs["geo_pred_pool"].detach().cpu())
    bucket["geo_target"].append(outputs["geo_target_pool"].detach().cpu())


def _retrieval_metrics(pred_parts: List[torch.Tensor], target_parts: List[torch.Tensor]) -> Dict[str, float]:
    if not pred_parts or not target_parts:
        return {}
    pred = torch.cat(pred_parts, dim=0).float()
    target = torch.cat(target_parts, dim=0).float()
    count = int(min(pred.shape[0], target.shape[0]))
    if count <= 1:
        return {"count": float(count)}
    pred = F.normalize(pred[:count], dim=-1, eps=1e-6)
    target = F.normalize(target[:count], dim=-1, eps=1e-6)
    sim = pred @ target.t()
    labels = torch.arange(count)
    order = torch.argsort(sim, dim=1, descending=True)
    matches = order.eq(labels.view(-1, 1))
    ranks = matches.float().argmax(dim=1) + 1
    top1 = (ranks <= 1).float().mean()
    top5 = (ranks <= min(5, count)).float().mean()
    pos = sim[labels, labels]
    masked = sim.masked_fill(torch.eye(count, dtype=torch.bool), -1e9)
    hardest_neg = masked.max(dim=1).values
    return {
        "count": float(count),
        "chance_top1": 1.0 / float(count),
        "top1": float(top1.item()),
        "top5": float(top5.item()),
        "mean_rank": float(ranks.float().mean().item()),
        "mrr": float((1.0 / ranks.float()).mean().item()),
        "positive_cosine": float(pos.mean().item()),
        "hardest_negative_cosine": float(hardest_neg.mean().item()),
        "positive_margin": float((pos - hardest_neg).mean().item()),
    }


def _summarize_retrieval(store: Dict[str, Dict[str, List[torch.Tensor]]]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for prefix, items in store.items():
        result[f"{prefix}_dyn"] = _retrieval_metrics(items["dyn_pred"], items["dyn_target"])
        result[f"{prefix}_geo"] = _retrieval_metrics(items["geo_pred"], items["geo_target"])
    return result


def _encode_with_slots(module: TwoExpertVLMSFTModule, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    return module.backbone.forward_with_two_expert_slots(
        batch["images"],
        batch["prompt_inputs"],
        two_expert_slots=module.two_expert_slots,
        return_image_hidden=True,
        return_raw_hidden=True,
        train_vlm_mode=module.config.train_mode,
    )


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    jepa_root, vggt_root = resolve_cache_roots(args)
    from scripts.last_vla_v2.two_expert_slot.run_two_expert_vlm_sft import load_teacher_index

    # Build full path indexes for teachers. This reads only index rows, not
    # tensors; applying --max-samples here can accidentally destroy the
    # intersection if base and teacher indexes are ordered differently.
    jepa_index = load_teacher_index(jepa_root, "jepa_dynamic_teacher_tokens", max_samples=None)
    vggt_index = load_teacher_index(vggt_root, "vggt_feature23_tokens", max_samples=None)
    vggt_dim = resolve_vggt_feature_dim(vggt_root, vggt_index, args.vggt_feature_dim)
    dataset = TwoExpertStage1Dataset(
        args.base_chunk_root,
        jepa_index,
        vggt_index,
        max_samples=args.max_samples,
        chunk_name_pattern=args.chunk_name_pattern,
        teacher_lru_size=int(args.teacher_lru_size),
        require_strict_teachers=not bool(args.allow_dev_fallback_teachers),
        allow_minimal_prompt=bool(args.allow_minimal_prompt),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=make_collate(args.max_image_patches),
    )
    backbone = RecogDriveBackbone(model_type=args.vlm_type, checkpoint_path=str(args.vlm_path), device=str(device))
    slot_config = TwoExpertSlotConfig(vlm_hidden_dim=int(args.vlm_hidden_dim), planner_dim=384)
    stage1_args = argparse.Namespace(
        stage1_train_mode=args.stage1_train_mode,
        train_vlm_mode=args.stage1_train_mode or "frozen",
        vlm_lora_adapter_dir=args.vlm_lora_adapter_dir,
    )
    loaded_slots, stage1_load_report = load_stage1_state_for_hidden_cache(
        checkpoint=args.stage1_checkpoint,
        config=slot_config,
        backbone=backbone,
        args=stage1_args,
    )
    stage1_mode = str(stage1_load_report.get("stage1_train_mode") or args.stage1_train_mode or "frozen")
    trained = TwoExpertVLMSFTModule(
        backbone,
        TwoExpertVLMSFTConfig(
            train_mode=stage1_mode,
            vggt_feature_dim=int(vggt_dim),
            hidden_anchor_weight=0.0,
            random_mask_ratio=0.3,
        ),
        slot_config=slot_config,
    ).to(device)
    trained.two_expert_slots.load_state_dict(loaded_slots.state_dict(), strict=True)
    module_load_report = _load_stage1_modules(trained, args.stage1_checkpoint)
    trained.eval()
    random_module: Optional[TwoExpertVLMSFTModule] = None
    if args.include_random_baseline:
        random_module = TwoExpertVLMSFTModule(
            backbone,
            TwoExpertVLMSFTConfig(
                train_mode=stage1_mode,
                vggt_feature_dim=int(vggt_dim),
                hidden_anchor_weight=0.0,
                random_mask_ratio=0.3,
            ),
            slot_config=slot_config,
        ).to(device)
        random_module.eval()

    acc = RunningMean()
    retrieval_store: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    samples = 0
    finite = True
    start = time.time()
    with torch.inference_mode():
        for batch in dataloader:
            batch = _to_device(batch, device)
            batch_size = int(batch["jepa_dynamic_teacher_tokens"].shape[0])
            with _autocast(device, args.precision):
                trained_vlm = _encode_with_slots(trained, batch)
                normal = _adapter_probe_from_vlm(trained, batch, trained_vlm)
                slot_only = _adapter_probe_from_vlm(
                    trained,
                    batch,
                    trained_vlm,
                    image_mode="zero",
                )
                image_only = _adapter_probe_from_vlm(
                    trained,
                    batch,
                    trained_vlm,
                    dyn_mode="zero",
                    geo_mode="zero",
                )
                no_signal = _adapter_probe_from_vlm(
                    trained,
                    batch,
                    trained_vlm,
                    image_mode="zero",
                    dyn_mode="zero",
                    geo_mode="zero",
                )
                high_mask = _adapter_probe_from_vlm(
                    trained,
                    batch,
                    trained_vlm,
                    image_mode="high_mask",
                    high_mask_ratio=float(args.high_mask_ratio),
                )
                zero_dyn = _adapter_probe_from_vlm(trained, batch, trained_vlm, dyn_mode="zero")
                zero_geo = _adapter_probe_from_vlm(trained, batch, trained_vlm, geo_mode="zero")
                random_outputs = None
                if random_module is not None:
                    random_vlm = _encode_with_slots(random_module, batch)
                    random_outputs = _adapter_probe_from_vlm(random_module, batch, random_vlm)
            for prefix, outputs in (
                ("trained_", normal),
                ("slot_only_", slot_only),
                ("image_only_", image_only),
                ("no_signal_", no_signal),
                ("high_mask_", high_mask),
                ("zero_dyn_", zero_dyn),
                ("zero_geo_", zero_geo),
            ):
                _add_outputs(acc, prefix, outputs, batch_size)
                if prefix in {"trained_", "slot_only_", "image_only_", "no_signal_", "high_mask_"}:
                    _collect_retrieval(retrieval_store, prefix.rstrip("_"), outputs)
            if random_outputs is not None:
                _add_outputs(acc, "random_", random_outputs, batch_size)
                _collect_retrieval(retrieval_store, "random", random_outputs)
            finite = finite and all(torch.isfinite(value.detach().float()).all().item() for value in normal.values() if isinstance(value, torch.Tensor))
            samples += batch_size
            if args.max_eval_batches is not None and samples >= int(args.max_eval_batches) * int(args.batch_size):
                break
    metrics = acc.result()
    retrieval = _summarize_retrieval(retrieval_store)
    comparisons = {}
    for key in ("loss", "dyn_loss", "geo_loss", "probe_loss"):
        trained_key = f"trained_{key}"
        if trained_key in metrics:
            for prefix in ("random", "slot_only", "image_only", "no_signal", "high_mask", "zero_dyn", "zero_geo"):
                other_key = f"{prefix}_{key}"
                if other_key in metrics:
                    comparisons[f"{other_key}_minus_{trained_key}"] = metrics[other_key] - metrics[trained_key]
                    if metrics[trained_key] != 0.0:
                        comparisons[f"{other_key}_over_{trained_key}"] = metrics[other_key] / metrics[trained_key]
    image_only_dyn_ratio = comparisons.get("image_only_dyn_loss_over_trained_dyn_loss", 0.0)
    image_only_geo_ratio = comparisons.get("image_only_geo_loss_over_trained_geo_loss", 0.0)
    slot_only_dyn_ratio = metrics.get("no_signal_dyn_loss", 0.0) / max(metrics.get("slot_only_dyn_loss", 0.0), 1e-12)
    slot_only_geo_ratio = metrics.get("no_signal_geo_loss", 0.0) / max(metrics.get("slot_only_geo_loss", 0.0), 1e-12)
    evidence = {
        "teacher_alignment_better_than_random": bool(
            comparisons.get("random_dyn_loss_over_trained_dyn_loss", 0.0) > 1.5
            and comparisons.get("random_geo_loss_over_trained_geo_loss", 0.0) > 1.5
        ),
        "slot_only_beats_no_signal": bool(slot_only_dyn_ratio > 1.05 and slot_only_geo_ratio > 1.05),
        "strong_slot_teacher_contribution_over_image_only": bool(
            image_only_dyn_ratio > 1.25 and image_only_geo_ratio > 1.25
        ),
        "planner_probe_uses_slots": bool(
            comparisons.get("zero_dyn_probe_loss_over_trained_probe_loss", 0.0) > 2.0
            and comparisons.get("zero_geo_probe_loss_over_trained_probe_loss", 0.0) > 2.0
        ),
    }
    evidence["overall_stage1_representation_supported"] = bool(
        evidence["teacher_alignment_better_than_random"]
        and evidence["slot_only_beats_no_signal"]
        and evidence["planner_probe_uses_slots"]
    )
    evidence["assessment"] = (
        "strong"
        if evidence["overall_stage1_representation_supported"]
        and evidence["strong_slot_teacher_contribution_over_image_only"]
        else "partial_positive_not_decisive"
        if evidence["overall_stage1_representation_supported"]
        else "not_supported"
    )
    return {
        "ok": bool(finite and samples > 0),
        "num_samples": int(samples),
        "elapsed_sec": float(time.time() - start),
        "stage1_checkpoint": str(args.stage1_checkpoint),
        "stage1_load_report": stage1_load_report,
        "module_load_report": module_load_report,
        "vggt_feature_dim": int(vggt_dim),
        "metrics": metrics,
        "comparisons": comparisons,
        "retrieval": retrieval,
        "evidence": evidence,
        "notes": [
            "This evaluates Stage1 teacher-alignment/probe quality only; it is not a PDMS estimate.",
            "Adapters are in eval mode; high_mask uses deterministic zero masking over image_hidden tokens for stress testing.",
            "slot_only keeps H_dyn/H_geo and zeros image_hidden; image_only keeps image_hidden and zeros H_dyn/H_geo.",
            "The current full Stage1 run used the same navtrain teacher set; use a held-out teacher cache for true generalization.",
        ],
    }


def write_markdown(path: Path, result: Dict[str, Any]) -> None:
    metrics = result["metrics"]
    comparisons = result["comparisons"]
    retrieval = result.get("retrieval", {})
    evidence = result.get("evidence", {})
    lines = [
        "# Two-Expert Stage1 Checkpoint Evaluation",
        "",
        f"- ok: `{result['ok']}`",
        f"- samples: `{result['num_samples']}`",
        f"- checkpoint: `{result['stage1_checkpoint']}`",
        f"- VGGT feature dim: `{result['vggt_feature_dim']}`",
        "",
        "## Evaluation Design",
        "",
        "- `trained`: full Stage1 checkpoint with normal image hidden and H_dyn/H_geo.",
        "- `random`: same VLM/LoRA backbone but random slots/adapters/probe.",
        "- `slot_only`: keep H_dyn/H_geo, zero image_hidden before adapters.",
        "- `image_only`: keep image_hidden, zero H_dyn/H_geo.",
        "- `no_signal`: zero both image_hidden and H_dyn/H_geo.",
        "- `high_mask`: keep H_dyn/H_geo and deterministically mask most image_hidden tokens.",
        "- Retrieval computes whether predicted teacher features identify the matching teacher sample within the evaluated batch.",
        "",
        "## Evidence Flags",
        "",
        "| flag | value |",
        "| --- | --- |",
    ]
    for key in sorted(evidence):
        lines.append(f"| `{key}` | `{evidence[key]}` |")
    lines.extend(
        [
            "",
            "## Core Metrics",
            "",
            "| metric | value |",
            "| --- | ---: |",
        ]
    )
    interesting_prefixes = (
        "trained_",
        "random_",
        "slot_only_",
        "image_only_",
        "no_signal_",
        "high_mask_",
        "zero_dyn_",
        "zero_geo_",
    )
    interesting_suffixes = ("loss", "dyn_loss", "geo_loss", "probe_loss", "h_dyn_norm", "h_geo_norm")
    for key in sorted(metrics):
        if key.startswith(interesting_prefixes) and key.endswith(interesting_suffixes):
            lines.append(f"| `{key}` | {metrics[key]:.8g} |")
    lines.extend(["", "## Retrieval", "", "| route | count | top1 | chance | mrr | margin |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for key in sorted(retrieval):
        item = retrieval[key]
        if not item:
            continue
        lines.append(
            f"| `{key}` | {item.get('count', 0):.0f} | {item.get('top1', 0):.6g} | "
            f"{item.get('chance_top1', 0):.6g} | {item.get('mrr', 0):.6g} | {item.get('positive_margin', 0):.6g} |"
        )
    lines.extend(["", "## Comparisons", "", "| comparison | value |", "| --- | ---: |"])
    for key in sorted(comparisons):
        lines.append(f"| `{key}` | {comparisons[key]:.8g} |")
    lines.extend(["", "## Notes", ""])
    for note in result.get("notes", []):
        lines.append(f"- {note}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a two_expert_slot Stage1 checkpoint without training.")
    parser.add_argument("--base-chunk-root", type=Path, required=True)
    parser.add_argument("--teacher-cache-root", type=Path, default=None)
    parser.add_argument("--jepa-cache-root", type=Path, default=None)
    parser.add_argument("--vggt-cache-root", type=Path, default=None)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-lora-adapter-dir", type=Path, default=None)
    parser.add_argument("--stage1-train-mode", choices=("frozen", "lora", "top_layers", "full"), default=None)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--vlm-type", default="internvl")
    parser.add_argument("--chunk-name-pattern", default=None)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--teacher-lru-size", type=int, default=16)
    parser.add_argument("--max-image-patches", type=int, default=12)
    parser.add_argument("--vggt-feature-dim", type=int, default=None)
    parser.add_argument("--vlm-hidden-dim", type=int, default=1536)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--high-mask-ratio", type=float, default=0.9)
    parser.add_argument("--allow-dev-fallback-teachers", action="store_true")
    parser.add_argument("--allow-minimal-prompt", action="store_true")
    parser.add_argument("--include-random-baseline", action="store_true", default=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if os.getenv("RUN_EVAL", "0") != "1":
        payload = {
            "status": "dry_run",
            "message": "Set RUN_EVAL=1 to evaluate Stage1 checkpoint.",
            "stage1_checkpoint": str(args.stage1_checkpoint),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output_json, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    result = evaluate(args)
    write_json(args.output_json, result)
    if args.output_md is not None:
        write_markdown(args.output_md, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
