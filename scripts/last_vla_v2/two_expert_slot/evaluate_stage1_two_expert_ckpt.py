#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import write_json  # noqa: E402
from navsim.agents.recogdrive.recogdrive_backbone import RecogDriveBackbone  # noqa: E402
from navsim.agents.recogdrive.trajectory_text_replay import try_parse_trajectory_answer  # noqa: E402
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
from scripts.last_vla_v2.two_expert_slot.two_expert_cache_utils import load_path_index  # noqa: E402


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
        incompatible = module.two_expert_slots.load_state_dict(payload["two_expert_slots"], strict=False)
        loaded["two_expert_slots"] = True
        loaded["two_expert_slots_missing"] = sorted(getattr(incompatible, "missing_keys", []))
        loaded["two_expert_slots_unexpected"] = sorted(getattr(incompatible, "unexpected_keys", []))
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
        if key == "trajectory_probe":
            state = _compat_probe_state(state)
        incompatible = getattr(module, attr).load_state_dict(state, strict=False)
        loaded[key] = True
        loaded[f"{key}_missing"] = sorted(getattr(incompatible, "missing_keys", []))
        loaded[f"{key}_unexpected"] = sorted(getattr(incompatible, "unexpected_keys", []))
    return loaded


def _compat_probe_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    mapped = dict(state)
    for key, value in list(state.items()):
        if key.startswith("head."):
            mapped.setdefault("fused_head." + key[len("head.") :], value)
    return mapped


def _read_token_file(path: Optional[Path]) -> Optional[list[str]]:
    if path is None:
        return None
    tokens: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        token = line.strip().split()[0] if line.strip() else ""
        if token:
            tokens.append(token)
    return tokens


def _filter_index(index: Dict[str, Path], tokens: Optional[list[str]]) -> Dict[str, Path]:
    if tokens is None:
        return index
    token_set = set(tokens)
    return {token: index[token] for token in tokens if token in token_set and token in index}


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


def _normalized_mse_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.detach().to(device=pred.device, dtype=pred.dtype)
    pred_norm = F.normalize(pred.float(), p=2, dim=-1, eps=1e-6)
    target_norm = F.normalize(target.float(), p=2, dim=-1, eps=1e-6)
    return F.mse_loss(pred_norm, target_norm, reduction="none").flatten(1).mean(dim=1).to(dtype=pred.dtype)


def _smooth_l1_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.detach().to(device=pred.device, dtype=pred.dtype)
    return F.smooth_l1_loss(pred.float(), target.float(), reduction="none").flatten(1).mean(dim=1).to(dtype=pred.dtype)


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
    elif dyn_mode == "random":
        h_dyn = torch.randn_like(h_dyn) * h_dyn.detach().float().std().clamp_min(1e-6).to(h_dyn)
    elif dyn_mode == "shuffle":
        if h_dyn.shape[0] > 1:
            h_dyn = h_dyn[torch.roll(torch.arange(h_dyn.shape[0], device=h_dyn.device), shifts=1)]
        else:
            h_dyn = torch.randn_like(h_dyn) * h_dyn.detach().float().std().clamp_min(1e-6).to(h_dyn)
    elif dyn_mode != "normal":
        raise ValueError(f"Unknown dyn_mode={dyn_mode!r}")
    if geo_mode == "zero":
        h_geo = torch.zeros_like(h_geo)
    elif geo_mode == "random":
        h_geo = torch.randn_like(h_geo) * h_geo.detach().float().std().clamp_min(1e-6).to(h_geo)
    elif geo_mode == "shuffle":
        if h_geo.shape[0] > 1:
            h_geo = h_geo[torch.roll(torch.arange(h_geo.shape[0], device=h_geo.device), shifts=1)]
        else:
            h_geo = torch.randn_like(h_geo) * h_geo.detach().float().std().clamp_min(1e-6).to(h_geo)
    elif geo_mode != "normal":
        raise ValueError(f"Unknown geo_mode={geo_mode!r}")
    dyn_target = batch["jepa_dynamic_teacher_tokens"]
    geo_target = batch["vggt_feature23_tokens"]
    dyn_out = module.dynamic_adapter(image_hidden, h_dyn, dyn_target)
    geo_out = module.geometry_adapter(image_hidden, h_geo, geo_target)
    target_action_norm = module._target_action_norm(batch, h_dyn)
    probe_out = module.trajectory_probe(
        h_dyn,
        h_geo,
        status_feature=batch.get("status_feature"),
        high_command_one_hot=batch.get("high_command_one_hot"),
        history_trajectory=batch.get("history_trajectory"),
        target_action_norm=target_action_norm,
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
        "dyn_loss_sample": _normalized_mse_per_sample(dyn_out["pred"], dyn_target),
        "geo_loss_sample": _normalized_mse_per_sample(geo_out["pred"], geo_target),
        "probe_loss_sample": _smooth_l1_per_sample(probe_out["probe_traj_norm"], target_action_norm),
        "probe_dyn_loss_sample": _smooth_l1_per_sample(probe_out["probe_dyn_traj_norm"], target_action_norm),
        "probe_geo_loss_sample": _smooth_l1_per_sample(probe_out["probe_geo_traj_norm"], target_action_norm),
        "dyn_pred_pool": dyn_out["pred"].detach().float().mean(dim=(1, 2)),
        "dyn_target_pool": dyn_target.detach().float().mean(dim=(1, 2)),
        "geo_pred_pool": geo_out["pred"].detach().float().mean(dim=1),
        "geo_target_pool": geo_target.detach().float().mean(dim=1),
        **dyn_out["diagnostics"],
        **geo_out["diagnostics"],
    }


def _add_outputs(acc: RunningMean, prefix: str, outputs: Dict[str, Any], batch_size: int) -> None:
    for key, value in outputs.items():
        if key.endswith("_pool") or key.endswith("_sample"):
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
        "chance_top5": min(5, count) / float(count),
        "top1": float(top1.item()),
        "top5": float(top5.item()),
        "mean_rank": float(ranks.float().mean().item()),
        "mrr": float((1.0 / ranks.float()).mean().item()),
        "positive_cosine": float(pos.mean().item()),
        "hardest_negative_cosine": float(hardest_neg.mean().item()),
        "positive_margin": float((pos - hardest_neg).mean().item()),
    }


def _retrieval_metrics_and_rows(
    pred_parts: List[torch.Tensor],
    target_parts: List[torch.Tensor],
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    if not pred_parts or not target_parts:
        return {}, []
    pred = torch.cat(pred_parts, dim=0).float()
    target = torch.cat(target_parts, dim=0).float()
    count = int(min(pred.shape[0], target.shape[0]))
    if count <= 1:
        return {"count": float(count)}, []
    pred = F.normalize(pred[:count], dim=-1, eps=1e-6)
    target = F.normalize(target[:count], dim=-1, eps=1e-6)
    sim = pred @ target.t()
    labels = torch.arange(count)
    order = torch.argsort(sim, dim=1, descending=True)
    matches = order.eq(labels.view(-1, 1))
    ranks = matches.float().argmax(dim=1) + 1
    pos = sim[labels, labels]
    masked = sim.masked_fill(torch.eye(count, dtype=torch.bool), -1e9)
    hardest_neg = masked.max(dim=1).values
    rows = []
    for idx in range(count):
        rank = int(ranks[idx].item())
        rows.append(
            {
                "rank": float(rank),
                "top1": float(rank <= 1),
                "top5": float(rank <= min(5, count)),
                "rr": float(1.0 / rank),
                "positive_cosine": float(pos[idx].item()),
                "hardest_negative_cosine": float(hardest_neg[idx].item()),
                "positive_margin": float((pos[idx] - hardest_neg[idx]).item()),
            }
        )
    return _retrieval_metrics([pred], [target]), rows


def _summarize_retrieval(store: Dict[str, Dict[str, List[torch.Tensor]]]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for prefix, items in store.items():
        result[f"{prefix}_dyn"] = _retrieval_metrics(items["dyn_pred"], items["dyn_target"])
        result[f"{prefix}_geo"] = _retrieval_metrics(items["geo_pred"], items["geo_target"])
    return result


def _disable_adapter_context(model: Any):
    disable_adapter = getattr(model, "disable_adapter", None)
    if disable_adapter is not None:
        return disable_adapter()
    class _Null:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc, tb): return False
    return _Null()


def _chat_callable(model: Any):
    if hasattr(model, "chat"):
        return model.chat
    base_model = getattr(model, "base_model", None)
    inner = getattr(base_model, "model", None)
    if inner is not None and hasattr(inner, "chat"):
        return inner.chat
    return None


def _trajectory_direct_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred = pred.detach().cpu().float()
    target = target.detach().cpu().float()
    return {
        "direct_traj_l1": float((pred - target).abs().mean().item()),
        "direct_traj_heading_l1": float((pred[:, 2] - target[:, 2]).abs().mean().item()),
        "direct_traj_endpoint_l2": float(torch.linalg.norm(pred[-1, :2] - target[-1, :2]).item()),
        "direct_traj_progress_error": float(abs((pred[-1, 0] - pred[0, 0]) - (target[-1, 0] - target[0, 0])).item()),
    }


def _mean_dict(rows: List[Dict[str, float]], *, prefix: str = "") -> Dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    out: Dict[str, float] = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row]
        if vals:
            out[prefix + key] = sum(vals) / len(vals)
    return out


def _try_direct_generation(
    backbone: RecogDriveBackbone,
    batch: Dict[str, Any],
    *,
    generation_config: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    chat = _chat_callable(backbone.model)
    if chat is None:
        return None, "model_chat_unavailable"
    replay_inputs = batch.get("replay_prompt_inputs")
    if not isinstance(replay_inputs, dict):
        return None, "missing_replay_prompt_inputs"
    prompts = replay_inputs.get("prompts") or replay_inputs.get("questions")
    num_patches_list = replay_inputs.get("num_patches_list")
    if not isinstance(prompts, list) or len(prompts) != 1:
        return None, "direct_generation_requires_batch_size_1"
    try:
        response = chat(
            backbone.tokenizer,
            batch["images"],
            str(prompts[0]),
            generation_config=dict(generation_config),
            num_patches_list=num_patches_list,
        )
        return str(response), None
    except Exception as exc:  # direct generation support varies across wrappers.
        return None, repr(exc)


def _slice_one_sample(batch: Dict[str, Any], index: int) -> Dict[str, Any]:
    sliced: Dict[str, Any] = {}
    num_patches = batch.get("prompt_inputs", {}).get("num_patches_list")
    if not isinstance(num_patches, list):
        raise TypeError("prompt_inputs.num_patches_list is required for single-sample slicing.")
    start = int(sum(int(item) for item in num_patches[:index]))
    end = start + int(num_patches[index])
    for key, value in batch.items():
        if key == "images" and isinstance(value, torch.Tensor):
            sliced[key] = value[start:end]
        elif key in {"prompt_inputs", "replay_prompt_inputs"} and isinstance(value, dict):
            item: Dict[str, Any] = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, list):
                    item[sub_key] = [sub_value[index]]
                else:
                    item[sub_key] = sub_value
            sliced[key] = item
        elif isinstance(value, torch.Tensor) and value.shape[:1] == (len(num_patches),):
            sliced[key] = value[index : index + 1]
        elif isinstance(value, list) and len(value) == len(num_patches):
            sliced[key] = [value[index]]
        else:
            sliced[key] = value
    return sliced


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
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    jepa_root, vggt_root = resolve_cache_roots(args)
    from scripts.last_vla_v2.two_expert_slot.run_two_expert_vlm_sft import load_teacher_index

    eval_tokens = _read_token_file(args.eval_token_file)
    jepa_index = load_teacher_index(jepa_root, "jepa_dynamic_teacher_tokens", max_samples=None)
    vggt_index = load_teacher_index(vggt_root, "vggt_feature23_tokens", max_samples=None)
    jepa_index = _filter_index(jepa_index, eval_tokens)
    vggt_index = _filter_index(vggt_index, eval_tokens)
    replay_index = load_path_index(args.replay_cache_root) if args.replay_cache_root is not None else {}
    replay_index = _filter_index(replay_index, eval_tokens)
    vggt_dim = resolve_vggt_feature_dim(vggt_root, vggt_index, args.vggt_feature_dim)
    allow_replay_only = bool(args.allow_replay_only_base or (args.base_chunk_root is None or not Path(args.base_chunk_root).exists()))
    dataset = TwoExpertStage1Dataset(
        args.base_chunk_root,
        jepa_index,
        vggt_index,
        max_samples=args.max_samples,
        chunk_name_pattern=args.chunk_name_pattern,
        teacher_lru_size=int(args.teacher_lru_size),
        require_strict_teachers=not bool(args.allow_dev_fallback_teachers),
        allow_minimal_prompt=bool(args.allow_minimal_prompt),
        replay_index=replay_index,
        require_replay=bool(args.replay_cache_root is not None),
        allow_replay_only_base=allow_replay_only,
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
    sample_tokens: List[str] = []
    per_sample: List[Dict[str, Any]] = []
    replay_ce_rows: List[Dict[str, float]] = []
    base_replay_ce_rows: List[Dict[str, float]] = []
    direct_rows: List[Dict[str, float]] = []
    base_direct_rows: List[Dict[str, float]] = []
    hidden_rows: List[Dict[str, float]] = []
    direct_errors: List[str] = []
    samples = 0
    last_progress_samples = 0
    finite = True
    start = time.time()
    generation_config = {
        "max_new_tokens": int(args.direct_max_new_tokens),
        "do_sample": False,
    }
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
                dyn_only = _adapter_probe_from_vlm(
                    trained,
                    batch,
                    trained_vlm,
                    image_mode="zero",
                    geo_mode="zero",
                )
                geo_only = _adapter_probe_from_vlm(
                    trained,
                    batch,
                    trained_vlm,
                    image_mode="zero",
                    dyn_mode="zero",
                )
                random_slots = _adapter_probe_from_vlm(
                    trained,
                    batch,
                    trained_vlm,
                    dyn_mode="shuffle",
                    geo_mode="shuffle",
                )
                random_outputs = None
                if random_module is not None:
                    random_vlm = _encode_with_slots(random_module, batch)
                    random_outputs = _adapter_probe_from_vlm(random_module, batch, random_vlm)
                if args.replay_cache_root is not None and samples < int(args.replay_ce_max_samples):
                    replay_ce = trained.backbone.forward_replay_ce(
                        batch["images"],
                        batch["replay_prompt_inputs"],
                        train_vlm_mode=stage1_mode,
                        max_length=int(args.replay_max_length),
                    )
                    replay_ce_rows.append(
                        {
                            "replay_ce_loss": _metric(replay_ce["loss"]),
                            "replay_token_count": _metric(replay_ce["token_count"]),
                        }
                    )
                    if args.include_base_vlm_replay:
                        with _disable_adapter_context(trained.backbone.model):
                            base_ce = trained.backbone.forward_replay_ce(
                                batch["images"],
                                batch["replay_prompt_inputs"],
                                train_vlm_mode=stage1_mode,
                                max_length=int(args.replay_max_length),
                            )
                        base_replay_ce_rows.append(
                            {
                                "replay_ce_loss": _metric(base_ce["loss"]),
                                "replay_token_count": _metric(base_ce["token_count"]),
                            }
                        )
                if samples < int(args.hidden_drift_max_samples):
                    frozen = trained._online_frozen_raw_hidden(batch["images"], batch["prompt_inputs"])
                    if isinstance(frozen, torch.Tensor):
                        cur = trained_vlm["raw_vlm_hidden"].float().mean(dim=1)
                        ref = frozen.float().mean(dim=1)
                        hidden_rows.append(
                            {
                                "hidden_drift_cosine": float(F.cosine_similarity(cur, ref, dim=-1).mean().item()),
                                "hidden_drift_l2": float(torch.linalg.norm(cur - ref, dim=-1).mean().item()),
                                "hidden_anchor_loss": float((1.0 - F.cosine_similarity(cur, ref, dim=-1)).mean().item()),
                                "raw_hidden_mean_norm": float(cur.norm(dim=-1).mean().item()),
                                "pooled_raw_hidden_cosine_to_base": float(F.cosine_similarity(cur, ref, dim=-1).mean().item()),
                            }
                        )
                if args.include_direct_generation and len(direct_rows) < int(args.direct_max_samples):
                    for local_idx in range(batch_size):
                        if len(direct_rows) >= int(args.direct_max_samples):
                            break
                        single_batch = _slice_one_sample(batch, local_idx)
                        response, error = _try_direct_generation(trained.backbone, single_batch, generation_config=generation_config)
                        target = single_batch.get("trajectory")
                        if response is not None and isinstance(target, torch.Tensor):
                            parsed = try_parse_trajectory_answer(response)
                            row = {"direct_traj_parse_ok_ratio": float(parsed.parse_ok)}
                            if parsed.parse_ok and parsed.trajectory is not None:
                                row.update(_trajectory_direct_metrics(parsed.trajectory, target[0].detach().cpu()))
                            direct_rows.append(row)
                        else:
                            direct_rows.append({"direct_traj_parse_ok_ratio": 0.0})
                            direct_errors.append(error or "unknown_direct_generation_error")
                        if args.include_base_vlm_replay:
                            with _disable_adapter_context(trained.backbone.model):
                                response, error = _try_direct_generation(trained.backbone, single_batch, generation_config=generation_config)
                            target = single_batch.get("trajectory")
                            if response is not None and isinstance(target, torch.Tensor):
                                parsed = try_parse_trajectory_answer(response)
                                row = {"direct_traj_parse_ok_ratio": float(parsed.parse_ok)}
                                if parsed.parse_ok and parsed.trajectory is not None:
                                    row.update(_trajectory_direct_metrics(parsed.trajectory, target[0].detach().cpu()))
                                base_direct_rows.append(row)
                            else:
                                base_direct_rows.append({"direct_traj_parse_ok_ratio": 0.0})
                                direct_errors.append("base:" + (error or "unknown_direct_generation_error"))
            for prefix, outputs in (
                ("trained_", normal),
                ("slot_only_", slot_only),
                ("image_only_", image_only),
                ("no_signal_", no_signal),
                ("high_mask_", high_mask),
                ("zero_dyn_", zero_dyn),
                ("zero_geo_", zero_geo),
                ("dyn_only_", dyn_only),
                ("geo_only_", geo_only),
                ("random_slots_", random_slots),
            ):
                _add_outputs(acc, prefix, outputs, batch_size)
                if prefix in {"trained_", "slot_only_", "image_only_", "no_signal_", "high_mask_"}:
                    _collect_retrieval(retrieval_store, prefix.rstrip("_"), outputs)
            if random_outputs is not None:
                _add_outputs(acc, "random_", random_outputs, batch_size)
                _collect_retrieval(retrieval_store, "random", random_outputs)
            token_list = [str(token) for token in batch.get("sample_token", [])]
            sample_tokens.extend(token_list)
            sample_tensors = {
                "normal_dyn_loss": normal["dyn_loss_sample"],
                "normal_geo_loss": normal["geo_loss_sample"],
                "normal_probe_loss": normal["probe_loss_sample"],
                "image_only_dyn_loss": image_only["dyn_loss_sample"],
                "image_only_geo_loss": image_only["geo_loss_sample"],
                "slot_only_dyn_loss": slot_only["dyn_loss_sample"],
                "slot_only_geo_loss": slot_only["geo_loss_sample"],
                "no_signal_dyn_loss": no_signal["dyn_loss_sample"],
                "no_signal_geo_loss": no_signal["geo_loss_sample"],
                "zero_dyn_probe_loss": zero_dyn["probe_loss_sample"],
                "zero_geo_probe_loss": zero_geo["probe_loss_sample"],
                "dyn_only_probe_loss": dyn_only["probe_loss_sample"],
                "geo_only_probe_loss": geo_only["probe_loss_sample"],
                "no_signal_probe_loss": no_signal["probe_loss_sample"],
            }
            if random_outputs is not None:
                sample_tensors["random_dyn_loss"] = random_outputs["dyn_loss_sample"]
                sample_tensors["random_geo_loss"] = random_outputs["geo_loss_sample"]
            for local_idx, token in enumerate(token_list):
                row = {"sample_token": token}
                for key, tensor in sample_tensors.items():
                    row[key] = _metric(tensor[local_idx])
                per_sample.append(row)
            finite = finite and all(torch.isfinite(value.detach().float()).all().item() for value in normal.values() if isinstance(value, torch.Tensor))
            samples += batch_size
            if int(args.progress_every) > 0 and samples - last_progress_samples >= int(args.progress_every):
                elapsed = time.time() - start
                print(
                    f"[stage1-eval] samples={samples}/{args.max_samples} "
                    f"elapsed_sec={elapsed:.1f} direct={len(direct_rows)} replay_ce={len(replay_ce_rows)} hidden={len(hidden_rows)}",
                    flush=True,
                )
                last_progress_samples = samples
            if args.max_eval_batches is not None and samples >= int(args.max_eval_batches) * int(args.batch_size):
                break
    metrics = acc.result()
    retrieval: Dict[str, Dict[str, float]] = {}
    retrieval_rows: Dict[str, List[Dict[str, float]]] = {}
    for prefix, items in retrieval_store.items():
        retrieval[f"{prefix}_dyn"], retrieval_rows[f"{prefix}_dyn"] = _retrieval_metrics_and_rows(items["dyn_pred"], items["dyn_target"])
        retrieval[f"{prefix}_geo"], retrieval_rows[f"{prefix}_geo"] = _retrieval_metrics_and_rows(items["geo_pred"], items["geo_target"])
    for idx, row in enumerate(per_sample):
        for retrieval_name in ("trained_dyn", "trained_geo"):
            rows = retrieval_rows.get(retrieval_name, [])
            if idx < len(rows):
                for key, value in rows[idx].items():
                    row[f"{retrieval_name}_{key}"] = value
    comparisons = {}
    for key in ("loss", "dyn_loss", "geo_loss", "probe_loss", "probe_dyn_loss", "probe_geo_loss"):
        trained_key = f"trained_{key}"
        if trained_key in metrics:
            for prefix in ("random", "slot_only", "image_only", "no_signal", "high_mask", "zero_dyn", "zero_geo", "dyn_only", "geo_only", "random_slots"):
                other_key = f"{prefix}_{key}"
                if other_key in metrics:
                    comparisons[f"{other_key}_minus_{trained_key}"] = metrics[other_key] - metrics[trained_key]
                    if metrics[trained_key] != 0.0:
                        comparisons[f"{other_key}_over_{trained_key}"] = metrics[other_key] / metrics[trained_key]
    comparisons["slot_only_dyn_gain"] = metrics.get("no_signal_dyn_loss", 0.0) / max(metrics.get("slot_only_dyn_loss", 0.0), 1e-12)
    comparisons["slot_only_geo_gain"] = metrics.get("no_signal_geo_loss", 0.0) / max(metrics.get("slot_only_geo_loss", 0.0), 1e-12)
    comparisons["zero_dyn_probe_ratio"] = metrics.get("zero_dyn_probe_loss", 0.0) / max(metrics.get("trained_probe_loss", 0.0), 1e-12)
    comparisons["zero_geo_probe_ratio"] = metrics.get("zero_geo_probe_loss", 0.0) / max(metrics.get("trained_probe_loss", 0.0), 1e-12)
    image_only_dyn_ratio = comparisons.get("image_only_dyn_loss_over_trained_dyn_loss", 0.0)
    image_only_geo_ratio = comparisons.get("image_only_geo_loss_over_trained_geo_loss", 0.0)
    slot_only_dyn_ratio = metrics.get("no_signal_dyn_loss", 0.0) / max(metrics.get("slot_only_dyn_loss", 0.0), 1e-12)
    slot_only_geo_ratio = metrics.get("no_signal_geo_loss", 0.0) / max(metrics.get("slot_only_geo_loss", 0.0), 1e-12)
    replay = _mean_dict(replay_ce_rows)
    replay["replay_eval_count"] = float(len(replay_ce_rows))
    base_replay = _mean_dict(base_replay_ce_rows, prefix="base_")
    replay.update(base_replay)
    direct = _mean_dict(direct_rows)
    direct["direct_eval_count"] = float(len(direct_rows))
    if direct_errors:
        direct["direct_generation_error_count"] = float(len(direct_errors))
        direct["direct_generation_first_error"] = direct_errors[0]
    base_direct = _mean_dict(base_direct_rows, prefix="base_")
    direct.update(base_direct)
    hidden = _mean_dict(hidden_rows)
    hidden["hidden_eval_count"] = float(len(hidden_rows))
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
        "replay_ce_measured": bool(replay.get("replay_eval_count", 0.0) > 0.0),
        "hidden_drift_measured": bool(hidden.get("hidden_eval_count", 0.0) > 0.0),
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
        "replay": replay,
        "direct": direct,
        "hidden": hidden,
        "per_sample": per_sample,
        "eval_token_file": str(args.eval_token_file) if args.eval_token_file else None,
        "replay_cache_root": str(args.replay_cache_root) if args.replay_cache_root else None,
        "base_chunk_root": str(args.base_chunk_root) if args.base_chunk_root else None,
        "allow_replay_only_base": bool(allow_replay_only),
        "evidence": evidence,
        "notes": [
            "This evaluates Stage1 teacher-alignment/probe quality only; it is not a PDMS estimate.",
            "Adapters are in eval mode; high_mask uses deterministic zero masking over image_hidden tokens for stress testing.",
            "slot_only keeps H_dyn/H_geo and zeros image_hidden; image_only keeps image_hidden and zeros H_dyn/H_geo.",
            "no_signal keeps trajectory-probe status/history/command context, so it is also the history/context-only probe baseline.",
            "If eval subset metadata reports train overlap, results are not a strict held-out Stage1 conclusion.",
        ],
    }


def write_markdown(path: Path, result: Dict[str, Any]) -> None:
    metrics = result["metrics"]
    comparisons = result["comparisons"]
    retrieval = result.get("retrieval", {})
    replay = result.get("replay", {})
    direct = result.get("direct", {})
    hidden = result.get("hidden", {})
    evidence = result.get("evidence", {})
    lines = [
        "# Two-Expert Stage1 Checkpoint Evaluation",
        "",
        f"- ok: `{result['ok']}`",
        f"- samples: `{result['num_samples']}`",
        f"- checkpoint: `{result['stage1_checkpoint']}`",
        f"- VGGT feature dim: `{result['vggt_feature_dim']}`",
        f"- replay cache: `{result.get('replay_cache_root')}`",
        f"- eval token file: `{result.get('eval_token_file')}`",
        f"- allow replay-only base: `{result.get('allow_replay_only_base')}`",
        "",
        "## Evaluation Design",
        "",
        "- `trained`: full Stage1 checkpoint with normal image hidden and H_dyn/H_geo.",
        "- `random`: same VLM/LoRA backbone but random slots/adapters/probe.",
        "- `slot_only`: keep H_dyn/H_geo, zero image_hidden before adapters.",
        "- `image_only`: keep image_hidden, zero H_dyn/H_geo.",
        "- `no_signal`: zero both image_hidden and H_dyn/H_geo.",
        "- `high_mask`: keep H_dyn/H_geo and deterministically mask most image_hidden tokens.",
        "- `dyn_only`: keep H_dyn only, zero image_hidden and H_geo.",
        "- `geo_only`: keep H_geo only, zero image_hidden and H_dyn.",
        "- `random_slots`: keep image_hidden and adapters, replace H_dyn/H_geo with shuffled/random slot hidden.",
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
        "dyn_only_",
        "geo_only_",
        "random_slots_",
    )
    interesting_suffixes = ("loss", "dyn_loss", "geo_loss", "probe_loss", "probe_dyn_loss", "probe_geo_loss", "h_dyn_norm", "h_geo_norm")
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
    lines.extend(["", "## Replay CE / Direct Trajectory", "", "| metric | value |", "| --- | ---: |"])
    for key in sorted(replay):
        value = replay[key]
        if isinstance(value, (int, float)):
            lines.append(f"| `replay.{key}` | {float(value):.8g} |")
    for key in sorted(direct):
        value = direct[key]
        if isinstance(value, (int, float)):
            lines.append(f"| `direct.{key}` | {float(value):.8g} |")
        else:
            lines.append(f"| `direct.{key}` | `{value}` |")
    lines.extend(["", "## Hidden Drift", "", "| metric | value |", "| --- | ---: |"])
    for key in sorted(hidden):
        value = hidden[key]
        if isinstance(value, (int, float)):
            lines.append(f"| `{key}` | {float(value):.8g} |")
    lines.extend(["", "## Notes", ""])
    for note in result.get("notes", []):
        lines.append(f"- {note}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_flat_csv(path: Path, result: Dict[str, Any]) -> None:
    flat: Dict[str, Any] = {
        "ok": result.get("ok"),
        "num_samples": result.get("num_samples"),
        "stage1_checkpoint": result.get("stage1_checkpoint"),
        "eval_token_file": result.get("eval_token_file"),
    }
    for section in ("metrics", "comparisons", "replay", "direct", "hidden"):
        values = result.get(section, {})
        if isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    flat[f"{section}.{key}"] = value
    retrieval = result.get("retrieval", {})
    if isinstance(retrieval, dict):
        for route, values in retrieval.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    if isinstance(value, (int, float, bool)) or value is None:
                        flat[f"retrieval.{route}.{key}"] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(flat))
        writer.writeheader()
        writer.writerow(flat)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a two_expert_slot Stage1 checkpoint without training.")
    parser.add_argument("--base-chunk-root", type=Path, default=None)
    parser.add_argument("--teacher-cache-root", type=Path, default=None)
    parser.add_argument("--jepa-cache-root", type=Path, default=None)
    parser.add_argument("--vggt-cache-root", type=Path, default=None)
    parser.add_argument("--replay-cache-root", type=Path, default=None)
    parser.add_argument("--stage1-checkpoint", "--checkpoint", dest="stage1_checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-lora-adapter-dir", type=Path, default=None)
    parser.add_argument("--stage1-train-mode", choices=("frozen", "lora", "top_layers", "full"), default=None)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--vlm-type", default="internvl")
    parser.add_argument("--chunk-name-pattern", default=None)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--eval-token-file", type=Path, default=None)
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
    parser.add_argument("--allow-replay-only-base", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-random-baseline", action="store_true", default=True)
    parser.add_argument("--include-base-vlm-replay", action="store_true")
    parser.add_argument("--include-direct-generation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--replay-ce-max-samples", type=int, default=4096)
    parser.add_argument("--hidden-drift-max-samples", type=int, default=512)
    parser.add_argument("--direct-max-samples", type=int, default=64)
    parser.add_argument("--direct-max-new-tokens", type=int, default=192)
    parser.add_argument("--replay-max-length", type=int, default=3200)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
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
    if args.output_csv is not None:
        write_flat_csv(args.output_csv, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
