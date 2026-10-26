"""Scene-balanced H and same-state DDIM transition-mean objectives."""
from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import contextmanager
from functools import lru_cache
import json
from pathlib import Path
import torch
from .adapter import validate_inputs


@contextmanager
def deterministic_backward_context():
    """Choose deterministic backward kernels without changing native forward.

    Enter only after constructing the loss/mean. PyTorch's Flash SDPA backward
    reads this runtime flag; its original forward attention remains untouched.
    CUDA processes also require their qualified startup cuBLAS workspace.
    """
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def backward_deterministically(loss):
    with deterministic_backward_context():
        loss.backward()


@contextmanager
def deterministic_student_forward_context():
    """Capture native compiled VJPs with the same flag used by backward.

    This metadata scope is confined to the grad-enabled student mean query.
    Formal admission requires real all-five-step bitwise parity against the
    untouched no-grad native query. Teacher/state/get_action paths stay outside.
    """
    with deterministic_backward_context():
        yield


@lru_cache(maxsize=16)
def _frozen_teacher_cache_settings(feature_manifest, targets_manifest, teacher_bundle,
                                   target_version, receipt_path, max_entries):
    from .checkpointing import sha256
    from .teacher_query_cache import validate_cache_receipt
    path = Path(receipt_path).resolve()
    receipt = json.loads(path.read_text())
    admission = validate_cache_receipt(receipt)
    root = Path(__file__).resolve().parents[1]
    feature_sha256 = sha256(feature_manifest)
    if admission.get("feature_manifest_sha256") != feature_sha256:
        raise ValueError("Teacher-query cache qualification used different real feature inputs")
    module = Path(__file__).with_name("teacher_query_cache.py")
    if admission.get("cache_module_sha256") != sha256(module):
        raise ValueError("Qualified teacher-query cache implementation changed")
    sources = admission.get("native_source_sha256", {})
    if not sources:
        raise ValueError("Teacher-query cache admission has no frozen native source identities")
    for name, checksum in sources.items():
        source = Path(name)
        source = source if source.is_absolute() else root / source
        if not source.is_file() or sha256(source) != checksum:
            raise ValueError("Qualified teacher-query cache native source changed")
    return {"identity": {"feature_manifest_sha256": feature_sha256,
                         "targets_manifest_sha256": sha256(targets_manifest),
                         "teacher_bundle_sha256": sha256(teacher_bundle), "teacher_target": target_version,
                         "experiment": "H_mu_teacher", "state_source": "teacher"},
            "receipt": receipt, "max_entries": max_entries,
            "admission": {"path": str(path), "sha256": sha256(path), "identity": admission}}


def teacher_query_cache_settings(cfg, *, refresh=False):
    """Resolve a qualified, scene-independent C-only pure-function cache."""
    performance = cfg.get("performance", {})
    if cfg.get("experiment") != "H_mu_teacher" or cfg.get("state_source") != "teacher" or performance.get("frozen_teacher_query_cache") is not True:
        return None
    paths = cfg["paths"]
    receipt = paths.get("teacher_query_cache_receipt", performance.get("cache_receipt"))
    if not receipt:
        raise ValueError("C teacher-query caching requires its actual qualified cache receipt")
    receipt = Path(receipt)
    if not receipt.is_absolute():
        receipt = Path(paths["results_root"]) / receipt
    maximum = performance.get("teacher_query_cache_max_entries", 500000)
    if type(maximum) is not int or not 1 <= maximum <= 500000:
        raise ValueError("Teacher-query cache entry bound must be an integer in 1..500000")
    if refresh:
        _frozen_teacher_cache_settings.cache_clear()
    return _frozen_teacher_cache_settings(str(Path(paths["feature_manifest"]).resolve()),
        str(Path(paths["targets_manifest"]).resolve()), str(Path(paths["teacher_bundle"]).resolve()),
        cfg["teacher_target"], str(receipt.resolve()), maximum)


def transition_weight(cfg, update: int) -> float:
    if type(update) is not int or update < 0:
        raise ValueError("update must be a nonnegative Python integer")
    if not isinstance(cfg, Mapping) or type(cfg.get("enable_transition_loss")) is not bool:
        raise ValueError("cfg must explicitly specify enable_transition_loss as bool")
    enabled = cfg["enable_transition_loss"]
    experiment = cfg.get("experiment")
    if experiment in ("H_full", "H_gain1") and enabled:
        raise ValueError("A/B may never enable functional supervision")
    if experiment in ("H_mu_teacher", "H_mu_student") and not enabled:
        raise ValueError("C/D must retain their registered transition curriculum")
    if not enabled:
        return 0.0
    if cfg.get("state_source") not in ("teacher", "student"):
        raise ValueError("C/D state_source must be teacher or student")
    if experiment == "H_mu_teacher" and cfg["state_source"] != "teacher":
        raise ValueError("C must query teacher states")
    if experiment == "H_mu_student" and cfg["state_source"] != "student":
        raise ValueError("D must query freshly generated student states")
    loss_cfg = cfg.get("loss", {})
    start = int(loss_cfg.get("transition_start_update", 5364))
    ramp = int(loss_cfg.get("transition_ramp_updates", 2682))
    maximum = float(loss_cfg.get("transition_mean_weight_max", 0.5))
    if (start, ramp, maximum) != (5364, 2682, 0.5):
        raise ValueError("Round1 transition curriculum must be 5364/2682/0.5")
    return maximum * min(max((update - start) / ramp, 0.0), 1.0)


def _scale(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive, finite, and frozen from Train")
    return max(value, 1e-12), value < 1e-12


def scene_loss(batch, adapter, bridge, cfg, update, calibration):
    """Return differentiable scalar loss and JSON-safe diagnostics.

    Only ordinary input tensors are passed to ``adapter``. Target provenance
    and scene/noise identities never enter its network or change loss weights.
    The trainer implements DDP's true-scene weighting outside this function.
    """
    weight_mu = transition_weight(cfg, update)
    if float(cfg.get("loss", {}).get("additional_keep_weight", 0.0)) != 0.0:
        raise ValueError("Round1 has no additional keep loss")
    h0 = batch["base_post"]
    valid = validate_inputs(h0, batch["image_mask"], batch["text_mask"], batch["state24"])
    if h0.shape[0] != 1:
        raise ValueError("native training microbatch must contain one scene")
    # Frozen Base, target, and ordinary state never become trainable inputs.
    h0 = h0.detach()
    delta_t = batch["teacher_actual_shift"].detach()
    if delta_t.shape != (1, 384) or delta_t.dtype != torch.float32 or delta_t.device != h0.device:
        raise ValueError("teacher_actual_shift must be FP32 [1,384] on the post device")
    if not bool(torch.isfinite(delta_t).all()):
        raise ValueError("teacher target contains non-finite shift")
    h_t = torch.where(valid.unsqueeze(-1), h0 + delta_t.unsqueeze(1), h0).detach()
    result = adapter(h0, batch["image_mask"], batch["text_mask"], batch["state24"].detach())
    h_s = result["post"]
    if h_s.shape != h0.shape or h_s.dtype != torch.float32 or h_s.device != h0.device:
        raise ValueError("adapter changed native interface shape/dtype/device")
    h_scale, protected = _scale(calibration["h_mse_scale"], "h_mse_scale")
    mse_h = ((h_s - h_t).square() * valid.unsqueeze(-1)).sum() / (valid.sum() * 384.0)
    loss_h = mse_h / h_scale
    stats = {"h": float(loss_h.detach()), "h_mse": float(mse_h.detach()),
             "mu": 0.0, "mu_weight": weight_mu, "transition_executed": False,
             "h_scale_protection": protected, "target_is_zero": bool((delta_t == 0).all()),
             "actual_rms": float(result["actual_rms"].mean()),
             "cap_triggered": bool(result.get("cap_triggered", torch.tensor(False)).any()),
             "cap_corrections": int(result.get("cap_corrections", 0))}
    if weight_mu == 0.0:
        if not bool(torch.isfinite(loss_h)):
            raise ValueError("non-finite representation loss")
        return loss_h, stats
    if bridge is None:
        raise ValueError("functional supervision needs the frozen native bridge")
    j = batch["time_index"]
    if type(j) is not int or not 0 <= j < 5:
        raise ValueError("time_index must remain a Python int in 0..4")
    bank = batch.get("noise_bank")
    if type(bank) is not int or not 0 <= bank < 5:
        raise ValueError("noise_bank must remain a Python int in 0..4")
    mu_scales = calibration["mu_mse_scale"]
    if len(mu_scales) != 5:
        raise ValueError("mu_mse_scale must have five fixed Train scales")
    mu_scale, mu_protected = _scale(mu_scales[j], f"mu_mse_scale[{j}]")
    context = bridge.encode_context(batch["history12"].detach(), batch["status8"].detach())
    teacher_cache_hit = False
    with torch.no_grad():
        options = teacher_query_cache_settings(cfg)
        if options is not None:
            from .teacher_query_cache import cached_teacher_query
            metadata = batch.get("metadata", {})
            token = batch.get("token", metadata.get("token"))
            if not isinstance(token, str) or not token:
                raise ValueError("Teacher-query caching requires the scene token solely as a cache key")
            identity = dict(options["identity"])
            for name in ("actual_Base_interface_sha256", "token_layout_sha256"):
                if name in metadata:
                    identity[name] = metadata[name]
            a_j, teacher_mean, teacher_cache_hit = cached_teacher_query(
                bridge, h_t, context, batch["noise_tape"], j, token=token,
                training_seed=cfg["training_seed"], noise_bank=bank, identity=identity,
                max_entries=options["max_entries"], receipt=options["receipt"])
            a_j, teacher_mean = a_j.detach(), teacher_mean.detach()
        else:
            state_post = h_t if cfg["state_source"] == "teacher" else h_s.detach()
            a_j = bridge.states_from_post(state_post, context, batch["noise_tape"], stop_before=j).detach()
            teacher_mean, _, _ = bridge.mean_from_post(a_j, j, h_t, context)
    # The exact same a_j and j go to both branches. State generation is a
    # fresh semi-gradient target each visit; only the student condition trains.
    with deterministic_student_forward_context():
        student_mean, _, _ = bridge.mean_from_post(a_j, j, h_s, context)
    mse_mu = (student_mean - teacher_mean.detach()).square().mean()
    loss_mu = mse_mu / mu_scale
    loss = loss_h + weight_mu * loss_mu
    if not bool(torch.isfinite(loss)):
        raise ValueError("non-finite scene loss")
    stats.update(mu=float(loss_mu.detach()), mu_mse=float(mse_mu.detach()),
                 time_index=j, noise_bank=bank, transition_executed=True,
                 mu_scale_protection=mu_protected, teacher_query_cache_hit=bool(teacher_cache_hit))
    return loss, stats
