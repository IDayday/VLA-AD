from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from torch import nn


LORA_PRESETS = {"attention_only", "attention_mlp", "all_linear", "vision_last_n", "custom"}
LORA_SCOPES = {"llm", "vision", "llm_vision", "projector", "all"}
LORA_CATEGORIES = ("llm_attention", "llm_mlp", "vision_attention", "vision_mlp", "projector", "other")
ATTENTION_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_SUFFIXES = ("gate_proj", "up_proj", "down_proj", "fc1", "fc2")
EXCLUDED_NAME_TOKENS = ("lm_head", "action_head")
VISION_NAME_TOKENS = ("vision_tower", "visual", "vision_model", "vision_encoder", "vit", "clip_vision")
PROJECTOR_NAME_TOKENS = ("mm_projector", "projector", "mlp1", "vision_proj", "visual_proj", "multi_modal_projector")


def _split_targets(value: str | Sequence[str] | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _module_name_suffix(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _is_linear(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear)


def _is_excluded(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in EXCLUDED_NAME_TOKENS)


def _is_vision_name(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in VISION_NAME_TOKENS)


def _is_projector_name(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in PROJECTOR_NAME_TOKENS)


def _matches_scope(name: str, scope: str) -> bool:
    is_vision = _is_vision_name(name)
    is_projector = _is_projector_name(name)
    if scope == "all":
        return not _is_excluded(name)
    if scope == "llm":
        return not is_vision and not is_projector and not _is_excluded(name)
    if scope == "vision":
        return is_vision and not is_projector and not _is_excluded(name)
    if scope == "llm_vision":
        return (is_vision or (not is_projector and not _is_excluded(name)))
    if scope == "projector":
        return is_projector and not _is_excluded(name)
    raise ValueError(f"Unknown LoRA scope: {scope!r}")


def _matches_any_suffix(name: str, suffixes: Sequence[str]) -> bool:
    return _module_name_suffix(name) in set(suffixes)


def _unique_suffixes(module_names: Iterable[str]) -> List[str]:
    suffixes = sorted({_module_name_suffix(name) for name in module_names})
    return suffixes


def infer_lora_module_category(module_name: str) -> str:
    suffix = _module_name_suffix(module_name)
    if _is_projector_name(module_name):
        return "projector"
    is_vision = _is_vision_name(module_name)
    if suffix in ATTENTION_SUFFIXES:
        return "vision_attention" if is_vision else "llm_attention"
    if suffix in MLP_SUFFIXES:
        return "vision_mlp" if is_vision else "llm_mlp"
    return "other"


def _named_linear_modules(model: nn.Module) -> List[str]:
    return [name for name, module in model.named_modules() if name and _is_linear(module)]


def _suffix_target_matches(model: nn.Module, suffixes: Sequence[str], scope: str) -> List[str]:
    matched = []
    for name in _named_linear_modules(model):
        if _matches_scope(name, scope) and _matches_any_suffix(name, suffixes):
            matched.append(name)
    return matched


def _vision_last_n_matches(model: nn.Module, *, vision_last_n: int, scope: str) -> List[str]:
    if int(vision_last_n) <= 0:
        raise ValueError("preset='vision_last_n' requires last_vla_vlm_lora_vision_last_n > 0.")
    vision_names = [name for name in _named_linear_modules(model) if _is_vision_name(name) and _matches_scope(name, scope)]
    if not vision_names:
        available = [name for name in _named_linear_modules(model) if _is_vision_name(name)][:50]
        raise ValueError(
            "Could not resolve vision_last_n LoRA targets: no vision-like linear modules matched. "
            f"Available vision-like modules: {available}"
        )
    block_indices: Dict[int, List[str]] = {}
    import re

    for name in vision_names:
        match = re.search(r"(?:layers|blocks|encoder\.layers|encoder\.blocks)\.(\d+)", name)
        if match is not None:
            block_indices.setdefault(int(match.group(1)), []).append(name)
    if not block_indices:
        raise ValueError(
            "Could not infer vision block indices for vision_last_n. "
            f"Available vision-like linear modules: {vision_names[:50]}"
        )
    selected_blocks = sorted(block_indices)[-int(vision_last_n):]
    suffixes = (*ATTENTION_SUFFIXES, *MLP_SUFFIXES)
    return [
        name
        for block in selected_blocks
        for name in block_indices[block]
        if _matches_any_suffix(name, suffixes)
    ]


def resolve_lora_target_modules(
    model: nn.Module,
    *,
    preset: str,
    scope: str,
    custom_target_modules: str,
    vision_last_n: int,
) -> list[str] | str:
    preset = str(preset)
    scope = str(scope)
    if preset not in LORA_PRESETS:
        raise ValueError(f"Unknown LoRA preset {preset!r}. Allowed: {sorted(LORA_PRESETS)}")
    if scope not in LORA_SCOPES:
        raise ValueError(f"Unknown LoRA scope {scope!r}. Allowed: {sorted(LORA_SCOPES)}")
    if preset == "custom":
        targets = _split_targets(custom_target_modules)
        if not targets:
            raise ValueError("preset='custom' requires last_vla_vlm_lora_target_modules.")
        audit = audit_lora_target_modules(model, target_modules=targets, scope=scope, preset=preset)
        if audit["matched_total"] == 0:
            raise ValueError(f"Custom LoRA targets matched zero modules: {targets!r}")
        return targets
    if preset == "attention_only":
        matched = _suffix_target_matches(model, ATTENTION_SUFFIXES, scope)
        targets = _unique_suffixes(matched)
    elif preset == "attention_mlp":
        matched = _suffix_target_matches(model, (*ATTENTION_SUFFIXES, *MLP_SUFFIXES), scope)
        targets = _unique_suffixes(matched)
    elif preset == "all_linear":
        matched = [name for name in _named_linear_modules(model) if _matches_scope(name, scope)]
        targets = _unique_suffixes(matched)
        if not targets:
            targets = "all-linear"
    elif preset == "vision_last_n":
        matched = _vision_last_n_matches(model, vision_last_n=vision_last_n, scope=scope)
        targets = sorted(matched)
    else:
        raise AssertionError(preset)
    audit = audit_lora_target_modules(model, target_modules=targets, scope=scope, preset=preset)
    if audit["matched_total"] == 0:
        raise ValueError(
            f"LoRA preset={preset!r} scope={scope!r} matched zero modules. "
            f"Resolved targets: {targets!r}"
        )
    return targets


def _target_matches_name(name: str, target_modules: list[str] | str) -> bool:
    if isinstance(target_modules, str):
        if target_modules == "all-linear":
            return True
        return _module_name_suffix(name) == target_modules or name.endswith(target_modules)
    targets = set(str(item) for item in target_modules)
    suffix = _module_name_suffix(name)
    return name in targets or suffix in targets or any(name.endswith(f".{target}") for target in targets)


def audit_lora_target_modules(
    model: nn.Module,
    *,
    target_modules: list[str] | str,
    scope: str,
    preset: str | None = None,
) -> dict:
    matched = [
        name
        for name in _named_linear_modules(model)
        if (preset == "custom" or _matches_scope(name, scope)) and _target_matches_name(name, target_modules)
    ]
    matched_by_category = {category: 0 for category in LORA_CATEGORIES}
    for name in matched:
        matched_by_category[infer_lora_module_category(name)] += 1
    trainable_lora = 0
    base_params = 0
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        base_params += count
        if "lora_" in name and parameter.requires_grad:
            trainable_lora += count
    warnings_list = []
    if preset == "attention_mlp" and matched_by_category["llm_mlp"] + matched_by_category["vision_mlp"] == 0:
        warnings_list.append("preset=attention_mlp matched no MLP modules.")
    return {
        "preset": preset,
        "scope": scope,
        "resolved_target_modules": target_modules,
        "matched_module_names": matched,
        "matched_total": int(len(matched)),
        "matched_by_category": matched_by_category,
        "trainable_lora_param_count": int(trainable_lora),
        "base_model_param_count": int(base_params),
        "trainable_ratio": float(trainable_lora / base_params) if base_params else 0.0,
        "high_risk_warnings": warnings_list,
    }


def validate_lora_scope_audit(audit: dict, *, allow_mixed_scope: bool = False) -> None:
    scope = str(audit.get("scope"))
    by_cat = audit.get("matched_by_category") or {}
    if int(audit.get("matched_total", 0)) <= 0:
        raise ValueError("LoRA target audit matched zero modules.")
    if allow_mixed_scope:
        return
    vision_count = int(by_cat.get("vision_attention", 0)) + int(by_cat.get("vision_mlp", 0))
    llm_count = int(by_cat.get("llm_attention", 0)) + int(by_cat.get("llm_mlp", 0))
    if scope == "llm" and vision_count > 0:
        raise ValueError("LoRA scope=llm matched vision modules; set allow_mixed_scope only for explicit ablations.")
    if scope == "vision" and llm_count > 0:
        raise ValueError("LoRA scope=vision matched LLM modules; set allow_mixed_scope only for explicit ablations.")


def peft_lora_config_kwargs_supported() -> set[str]:
    try:
        from peft import LoraConfig
    except ImportError:
        return set()
    try:
        signature = inspect.signature(LoraConfig)
        params = set(signature.parameters)
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            params.update({"use_rslora", "use_dora", "init_lora_weights"})
        return params
    except (TypeError, ValueError):
        return set()


def build_lora_config_payload(
    *,
    preset: str,
    scope: str,
    target_modules: list[str] | str,
    r: int,
    alpha: int,
    dropout: float,
    bias: str,
    use_rslora: bool,
    use_dora: bool,
    init: str,
    vision_last_n: int,
    peft_version: str | None = None,
) -> Dict[str, Any]:
    return {
        "preset": preset,
        "scope": scope,
        "target_modules": target_modules,
        "resolved_target_modules": target_modules,
        "r": int(r),
        "alpha": int(alpha),
        "lora_alpha": int(alpha),
        "dropout": float(dropout),
        "lora_dropout": float(dropout),
        "bias": str(bias),
        "use_rslora": bool(use_rslora),
        "use_dora": bool(use_dora),
        "init": str(init),
        "vision_last_n": int(vision_last_n),
        "peft_version": peft_version,
    }


def stable_config_hash(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_lora_adapter_metadata(adapter_dir: str | Path) -> Dict[str, Any]:
    adapter_dir = Path(adapter_dir)
    metadata_path = adapter_dir / "lora_metadata.json"
    config_path = adapter_dir / "adapter_config.json"
    payload: Dict[str, Any] = {}
    if config_path.is_file():
        payload.update(json.loads(config_path.read_text(encoding="utf-8")))
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload.update(metadata)
    if not payload:
        raise FileNotFoundError(f"No adapter_config.json or lora_metadata.json found under {adapter_dir}")
    if "target_modules" not in payload and "resolved_target_modules" in payload:
        payload["target_modules"] = payload["resolved_target_modules"]
    if "resolved_target_modules" not in payload and "target_modules" in payload:
        payload["resolved_target_modules"] = payload["target_modules"]
    if "alpha" not in payload and "lora_alpha" in payload:
        payload["alpha"] = payload["lora_alpha"]
    if "dropout" not in payload and "lora_dropout" in payload:
        payload["dropout"] = payload["lora_dropout"]
    return payload
