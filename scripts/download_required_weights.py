#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import shlex
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

DEFAULT_WEIGHTS_CONFIG = Path("configs/weights.yaml")
DEFAULT_OUTPUT_ROOT = Path("/mnt/project/VLA-AD/checkpoints")
DEFAULT_HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
REQUIRED_KEYS = ["recogdrive_2b_il", "recogdrive_vlm_2b", "vjepa2", "vggt_1b"]
OPTIONAL_KEYS = ["recogdrive_2b_rl"]
ALL_KNOWN_KEYS = [*REQUIRED_KEYS, *OPTIONAL_KEYS]

ALLOW_PATTERNS = [
    "*.safetensors", "*.bin", "*.pt", "*.ckpt", "*.pth", "*.json", "*.yaml", "*.yml", "*.py",
    "*.txt", "*.md", "*.model", "*.tiktoken", "tokenizer*", "sentencepiece*", "merges.txt",
    "vocab.*", "preprocessor_config.json", "processor_config.json", "generation_config.json",
    "special_tokens_map.json", "added_tokens.json", "config.json", "model_index.json",
]
IGNORE_PATTERNS = [
    "*.log", "logs/*", "wandb/*", "tensorboard/*", "events.out.tfevents*", "*.mp4", "*.avi",
    "*.zip", "*.tar", "*.tar.gz",
]
CONFIG_SUFFIXES = {".json", ".yaml", ".yml", ".py", ".txt", ".model"}
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".ckpt", ".pth", ".model"}
PARTIAL_SUFFIXES = (".incomplete", ".partial", ".tmp", ".download")
PARTIAL_NAME_FRAGMENTS = ("incomplete", "partial", "tmp")
IGNORED_WEIGHT_NAMES = {"training_args.bin", "trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"}
PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download required ReCogDrive expert-token weights.")
    parser.add_argument("--weights-config", type=Path, default=DEFAULT_WEIGHTS_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source", choices=("auto", "modelscope", "huggingface", "hf-original", "hf-mirror"), default="auto")
    parser.add_argument("--hf-endpoint", default=None, help="Override HF_ENDPOINT. For --source hf-mirror, defaults to https://hf-mirror.com.")
    parser.add_argument("--modelscope-cache", type=Path, default=Path(os.environ.get("MODELSCOPE_CACHE", "/mnt/project/VLA-AD/modelscope_cache")))
    parser.add_argument("--hf-cache", type=Path, default=Path(os.environ.get("HF_HUB_CACHE", "/mnt/project/VLA-AD/hf_home/hub")))
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--http-proxy", default=None)
    parser.add_argument("--https-proxy", default=None)
    parser.add_argument("--all-proxy", default=None)
    parser.add_argument("--use-env-proxy", action="store_true", help="Keep proxy variables already present in the environment.")
    parser.add_argument("--proxy-if-slow", action="store_true", help="Accepted for workflow compatibility; proxy is still only enabled when a proxy value is supplied.")
    parser.add_argument("--min-speed-mbps", type=float, default=5.0)
    parser.add_argument("--timeout", "--timeout-sec", dest="timeout_sec", type=int, default=300)
    parser.add_argument("--etag-timeout", dest="etag_timeout_sec", type=int, default=None, help="HF_HUB_ETAG_TIMEOUT. Defaults to --timeout.")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--repo-workers", type=int, default=4)
    parser.add_argument("--file-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="Reuse local partial files and HF cache fragments. This is the default.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Disable explicit Hugging Face resume mode for this run.")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--include-recogdrive-rl", action="store_true")
    parser.add_argument("--include-rl", action="store_true", help="Alias for --include-recogdrive-rl.")
    parser.add_argument("--skip-vlm", action="store_true")
    parser.add_argument("--skip-teachers", action="store_true")
    parser.add_argument("--only", default=None, help="Comma-separated targets to download, e.g. recogdrive_2b_il,vjepa2")
    parser.add_argument("--skip", default=None, help="Comma-separated targets to skip.")
    parser.add_argument("--clean-partials", action="store_true", help="Remove incomplete/temp fragments for selected repos before downloading.")
    parser.add_argument("--list-targets", action="store_true")
    parser.add_argument("--status", action="store_true", help="Write and print local download status without downloading.")
    parser.add_argument("--revision", default=None)
    return parser.parse_args()


def parse_name_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_weights_config(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"weights config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"weights config must be a mapping, got {type(data).__name__}")
    return data


def local_dir_for(spec: Dict[str, Any], output_root: Path) -> Path:
    configured = Path(str(spec["local_dir"])).expanduser()
    if not configured.is_absolute():
        return output_root / configured
    parts = configured.parts
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        return output_root.joinpath(*parts[idx + 1 :])
    return configured


def configured_targets(config: Dict[str, Dict[str, Any]], output_root: Path) -> Dict[str, Dict[str, Any]]:
    targets: Dict[str, Dict[str, Any]] = {}
    for key, spec in config.items():
        if not isinstance(spec, dict):
            continue
        targets[key] = {
            "key": key,
            "hf_repo": spec.get("hf_repo"),
            "modelscope_repo": spec.get("modelscope_repo"),
            "local_dir": str(local_dir_for(spec, output_root)),
            "optional": bool(spec.get("optional", key in OPTIONAL_KEYS)),
        }
    return targets


def selected_keys(args: argparse.Namespace, config: Dict[str, Dict[str, Any]]) -> List[str]:
    only = parse_name_list(args.only)
    skips = set(parse_name_list(args.skip))
    known = [key for key in ALL_KNOWN_KEYS if key in config]
    if args.include_recogdrive_rl or args.include_rl:
        keys = known
    else:
        keys = [key for key in known if key not in OPTIONAL_KEYS]
    if only:
        unknown = [key for key in only if key not in config]
        if unknown:
            raise ValueError(f"Unknown --only target(s): {unknown}. Use --list-targets.")
        keys = only
    if args.skip_vlm:
        skips.add("recogdrive_vlm_2b")
    if args.skip_teachers:
        skips.update({"vjepa2", "vggt_1b"})
    unknown_skips = [key for key in skips if key not in config]
    if unknown_skips:
        raise ValueError(f"Unknown --skip target(s): {unknown_skips}. Use --list-targets.")
    keys = [key for key in keys if key not in skips]
    if not keys and not (args.status or args.list_targets):
        raise ValueError("No targets selected.")
    return keys


def is_plausible_weight_file(path: Path) -> bool:
    lower = path.name.lower()
    if ".cache" in path.parts or lower.endswith(PARTIAL_SUFFIXES):
        return False
    if lower in IGNORED_WEIGHT_NAMES:
        return False
    if path.suffix.lower() not in WEIGHT_SUFFIXES:
        return False
    if path.suffix.lower() == ".model":
        return True
    return path.stat().st_size >= 1024 * 1024


def is_config_file(path: Path) -> bool:
    if ".cache" in path.parts:
        return False
    return path.is_file() and path.suffix.lower() in CONFIG_SUFFIXES


def iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (path for path in root.rglob("*") if path.is_file())


def partial_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    files: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if lower.endswith(PARTIAL_SUFFIXES) or any(fragment in lower for fragment in PARTIAL_NAME_FRAGMENTS):
            files.append(path)
    return sorted(files)


def weight_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and is_plausible_weight_file(path))


def config_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if is_config_file(path))


def dir_total_size(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                pass
    return total


def target_status(key: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    local_dir = Path(spec["local_dir"])
    weights = weight_files(local_dir)
    configs = config_files(local_dir)
    partials = partial_files(local_dir)
    requires_config = key not in {"recogdrive_2b_il", "recogdrive_2b_rl"}
    missing: List[str] = []
    if not local_dir.is_dir():
        missing.append("directory")
    if requires_config and not configs:
        missing.append("config/tokenizer/source files")
    if not weights:
        missing.append("model weight files")
    complete = local_dir.is_dir() and bool(weights) and (bool(configs) or not requires_config)
    return {
        "target": key,
        "hf_repo": spec.get("hf_repo"),
        "modelscope_repo": spec.get("modelscope_repo"),
        "local_dir": str(local_dir),
        "exists": local_dir.is_dir(),
        "config_file_count": len(configs),
        "weight_file_count": len(weights),
        "partial_file_count": len(partials),
        "total_size_bytes": dir_total_size(local_dir),
        "weight_size_bytes": sum(path.stat().st_size for path in weights),
        "complete_plausible": complete,
        "missing_critical_files": missing,
        "weight_files": [str(path) for path in weights],
        "partial_files": [str(path) for path in partials[:50]],
    }


def all_status(config: Dict[str, Dict[str, Any]], output_root: Path, keys: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    targets = configured_targets(config, output_root)
    ordered = list(keys) if keys is not None else [key for key in ALL_KNOWN_KEYS if key in targets]
    records = [target_status(key, targets[key]) for key in ordered if key in targets]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "records": records,
    }


def write_status_reports(output_root: Path, payload: Dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "download_status.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Weight Download Status", "", f"Created: {payload['created_at']}", f"Output root: `{payload['output_root']}`", ""]
    lines.extend(["| Target | Exists | Weight Files | Total Size | Complete | Missing Critical Files |", "|---|---:|---:|---:|---:|---|"])
    for record in payload["records"]:
        missing = ", ".join(record["missing_critical_files"]) or "none"
        lines.append(
            f"| {record['target']} | {record['exists']} | {record['weight_file_count']} | {record['total_size_bytes']} | {record['complete_plausible']} | {missing} |"
        )
    lines.append("")
    for record in payload["records"]:
        lines.append(f"## {record['target']}")
        lines.append(f"- local dir: `{record['local_dir']}`")
        lines.append(f"- hf repo: `{record.get('hf_repo')}`")
        lines.append(f"- modelscope repo: `{record.get('modelscope_repo')}`")
        lines.append(f"- partial files: {record['partial_file_count']}")
        if record["weight_files"]:
            lines.append("- weight files:")
            for path in record["weight_files"]:
                lines.append(f"  - `{path}`")
        if record["partial_files"]:
            lines.append("- partial fragments sample:")
            for path in record["partial_files"][:20]:
                lines.append(f"  - `{path}`")
        lines.append("")
    (output_root / "download_status.md").write_text("\n".join(lines), encoding="utf-8")


def print_status(payload: Dict[str, Any]) -> None:
    for record in payload["records"]:
        print(f"{record['target']}: {record['local_dir']}")
        print(f"  exists: {'yes' if record['exists'] else 'no'}")
        print(f"  weight files: {record['weight_file_count']}")
        print(f"  total size: {record['total_size_bytes']} bytes")
        print(f"  complete plausible: {'yes' if record['complete_plausible'] else 'no'}")
        print(f"  missing critical files: {', '.join(record['missing_critical_files']) or 'none'}")


def clean_partial_files(local_dir: Path) -> Dict[str, Any]:
    removed: List[str] = []
    errors: List[str] = []
    for path in partial_files(local_dir):
        try:
            path.unlink()
            removed.append(str(path))
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    return {"removed_count": len(removed), "removed_files": removed, "errors": errors}


def hf_cache_repo_dir(hf_cache: Path, repo_id: str) -> Path:
    return hf_cache / f"models--{repo_id.replace('/', '--')}"


def clean_selected_repo_partials(local_dir: Path, hf_cache: Path, hf_repo: str) -> Dict[str, Any]:
    roots = [local_dir, hf_cache_repo_dir(hf_cache, hf_repo)]
    removed: List[str] = []
    errors: List[str] = []
    scanned_roots: List[str] = []
    for root in roots:
        scanned_roots.append(str(root))
        report = clean_partial_files(root)
        removed.extend(report["removed_files"])
        errors.extend(report["errors"])
    return {
        "removed_count": len(removed),
        "removed_files": removed,
        "errors": errors,
        "scanned_roots": scanned_roots,
    }


def proxy_enabled(args: argparse.Namespace) -> bool:
    return bool(args.proxy or args.http_proxy or args.https_proxy or args.all_proxy or args.use_env_proxy)


def effective_hf_endpoint(args: argparse.Namespace) -> Optional[str]:
    if args.hf_endpoint:
        return args.hf_endpoint
    if args.source == "hf-mirror":
        return DEFAULT_HF_MIRROR_ENDPOINT
    return None


def set_env_before_import(args: argparse.Namespace) -> None:
    if not args.use_env_proxy and not any([args.proxy, args.http_proxy, args.https_proxy, args.all_proxy]):
        for key in PROXY_ENV_KEYS:
            os.environ.pop(key, None)
    if args.proxy:
        if not args.http_proxy:
            os.environ["HTTP_PROXY"] = args.proxy
        if not args.https_proxy:
            os.environ["HTTPS_PROXY"] = args.proxy
        if not args.all_proxy:
            os.environ["ALL_PROXY"] = args.proxy
    if args.http_proxy:
        os.environ["HTTP_PROXY"] = args.http_proxy
    if args.https_proxy:
        os.environ["HTTPS_PROXY"] = args.https_proxy
    if args.all_proxy:
        os.environ["ALL_PROXY"] = args.all_proxy
    endpoint = effective_hf_endpoint(args)
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    else:
        os.environ.pop("HF_ENDPOINT", None)
    os.environ["HF_HOME"] = str(args.hf_cache.parent)
    os.environ["HF_HUB_CACHE"] = str(args.hf_cache)
    os.environ.setdefault("HF_XET_CACHE", str(args.hf_cache.parent / "xet"))
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "16")
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(args.timeout_sec)
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(args.etag_timeout_sec or args.timeout_sec)
    os.environ["MODELSCOPE_CACHE"] = str(args.modelscope_cache)


def repo_jobs(args: argparse.Namespace, config: Dict[str, Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    targets = configured_targets(config, args.output_root)
    jobs: List[Dict[str, Any]] = []
    for key in keys:
        spec = targets[key]
        hf_repo = spec.get("hf_repo")
        if not hf_repo:
            raise ValueError(f"weights config entry {key} requires hf_repo")
        clean_report = None
        if args.clean_partials:
            clean_report = clean_selected_repo_partials(Path(spec["local_dir"]), args.hf_cache, hf_repo)
        jobs.append({
            "key": key,
            "hf_repo": hf_repo,
            "modelscope_repo": spec.get("modelscope_repo"),
            "local_dir": spec["local_dir"],
            "revision": args.revision,
            "source": args.source,
            "hf_endpoint": effective_hf_endpoint(args),
            "hf_token": args.hf_token or os.environ.get("HF_TOKEN"),
            "file_workers": args.file_workers,
            "dry_run": args.dry_run,
            "force_download": args.force_download,
            "resume": args.resume,
            "local_files_only": args.local_files_only,
            "allow_patterns": ALLOW_PATTERNS,
            "ignore_patterns": IGNORE_PATTERNS,
            "retries": args.retries,
            "timeout_sec": args.timeout_sec,
            "etag_timeout_sec": args.etag_timeout_sec or args.timeout_sec,
            "modelscope_cache": str(args.modelscope_cache),
            "clean_partials": clean_report,
        })
    return jobs


def hf_snapshot(job: Dict[str, Any]) -> Dict[str, Any]:
    from huggingface_hub import snapshot_download

    kwargs = dict(
        repo_id=job["hf_repo"],
        repo_type="model",
        revision=job["revision"],
        local_dir=job["local_dir"],
        token=job["hf_token"],
        max_workers=job["file_workers"],
        allow_patterns=job["allow_patterns"],
        ignore_patterns=job["ignore_patterns"],
        local_files_only=job["local_files_only"],
        force_download=job["force_download"],
        resume_download=bool(job.get("resume", True)),
    )
    if job["dry_run"]:
        kwargs["dry_run"] = True
    try:
        result = snapshot_download(**kwargs)
    except TypeError:
        kwargs.pop("resume_download", None)
        if job["dry_run"]:
            kwargs.pop("dry_run", None)
            # Older huggingface_hub versions do not dry-run. Verify API reachability through model_info instead.
            from huggingface_hub import HfApi
            HfApi(token=job["hf_token"]).model_info(job["hf_repo"], revision=job["revision"])
            return {"snapshot_path": None, "dry_run": True, "dry_run_mode": "model_info"}
        result = snapshot_download(**kwargs)
    return {"snapshot_path": None if job["dry_run"] else str(result), "dry_run": bool(job["dry_run"])}


def modelscope_snapshot(job: Dict[str, Any]) -> Dict[str, Any]:
    model_id = job.get("modelscope_repo")
    if not model_id:
        raise RuntimeError("No ModelScope mirror id configured for this model; skipping mirror.")
    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
    except Exception as exc:
        raise RuntimeError("ModelScope is not installed; install modelscope or use --source huggingface.") from exc
    if job["dry_run"]:
        return {"snapshot_path": None, "dry_run": True}
    local_dir = Path(job["local_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)
    result = ms_snapshot_download(
        model_id=model_id,
        revision=job["revision"] or "master",
        cache_dir=job["modelscope_cache"],
        local_dir=str(local_dir),
    )
    return {"snapshot_path": str(result), "dry_run": False}


def source_sequence(job: Dict[str, Any]) -> List[str]:
    source = job["source"]
    if source == "auto":
        sequence: List[str] = []
        if job.get("modelscope_repo"):
            sequence.append("modelscope")
        sequence.append("huggingface")
        return sequence
    if source in {"hf-original", "hf-mirror"}:
        return ["huggingface"]
    return [source]


def run_one(job: Dict[str, Any]) -> Dict[str, Any]:
    started = time.time()
    record: Dict[str, Any] = {
        "key": job["key"],
        "repo": job["hf_repo"],
        "modelscope_repo": job.get("modelscope_repo"),
        "local_dir": job["local_dir"],
        "requested_source": job["source"],
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "clean_partials": job.get("clean_partials"),
        "ok": False,
    }
    initial_status = target_status(job["key"], {"local_dir": job["local_dir"], "hf_repo": job["hf_repo"], "modelscope_repo": job.get("modelscope_repo")})
    record["initial_status"] = initial_status
    if initial_status.get("complete_plausible") and not job["force_download"] and not job.get("clean_partials"):
        record["ok"] = True
        record["skipped"] = True
        record["skip_reason"] = "already complete plausible"
        record["resolved_source"] = "local"
        record["elapsed_sec"] = round(time.time() - started, 3)
        record["final_status"] = initial_status
        return record
    errors: List[Dict[str, Any]] = []
    for source in source_sequence(job):
        for attempt in range(1, max(1, int(job.get("retries", 1))) + 1):
            try:
                result = modelscope_snapshot(job) if source == "modelscope" else hf_snapshot(job)
                verification = {"ok": True} if job["dry_run"] else target_status(job["key"], {"local_dir": job["local_dir"], "hf_repo": job["hf_repo"], "modelscope_repo": job.get("modelscope_repo")})
                if not verification.get("complete_plausible", verification.get("ok", False)):
                    raise RuntimeError(f"Downloaded files did not pass verification: {verification}")
                record.update(result)
                record["verification"] = verification
                record["resolved_source"] = source
                record["attempt"] = attempt
                record["ok"] = True
                break
            except Exception as exc:
                errors.append({"source": source, "attempt": attempt, "error": str(exc), "traceback": traceback.format_exc()})
                if attempt < max(1, int(job.get("retries", 1))):
                    time.sleep(min(30, 2 ** (attempt - 1)))
        if record["ok"]:
            break
    record["elapsed_sec"] = round(time.time() - started, 3)
    record["final_status"] = target_status(job["key"], {"local_dir": job["local_dir"], "hf_repo": job["hf_repo"], "modelscope_repo": job.get("modelscope_repo")})
    if errors:
        record["errors"] = errors
    if not record["ok"]:
        record["error"] = errors[-1]["error"] if errors else "unknown error"
    return record


def write_manifest(output_root: Path, records: List[Dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "records": records,
        "env": {key: os.environ.get(key) for key in [
            "HF_ENDPOINT", "HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "HF_HUB_DOWNLOAD_TIMEOUT",
            "HF_HUB_ETAG_TIMEOUT", "MODELSCOPE_CACHE", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        ] if os.environ.get(key)},
    }
    (output_root / "download_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_failure_report(output_root: Path, args: argparse.Namespace, records: List[Dict[str, Any]]) -> None:
    failed = [record for record in records if not record.get("ok")]
    if not failed:
        return
    output_root.mkdir(parents=True, exist_ok=True)
    lines = ["# Weight Download Failure Report", "", f"Created: {datetime.now(timezone.utc).isoformat()}", ""]
    lines.append("## Command")
    lines.extend(["", f"```bash\n{shlex.join([sys.executable, *sys.argv])}\n```", ""])
    lines.append("## Failures")
    lines.append("")
    for record in failed:
        lines.append(f"- key: `{record.get('key')}`")
        lines.append(f"  repo: `{record.get('repo')}`")
        lines.append(f"  requested source: `{record.get('requested_source')}`")
        lines.append(f"  local dir: `{record.get('local_dir')}`")
        lines.append(f"  final error: `{record.get('error')}`")
        status = record.get("final_status") or {}
        lines.append(f"  current complete plausible: `{status.get('complete_plausible')}`")
        lines.append(f"  current weight files: `{status.get('weight_file_count')}`")
        lines.append(f"  partial fragments: `{status.get('partial_file_count')}`")
        for error in record.get("errors", []):
            lines.append(f"  - source={error.get('source')} attempt={error.get('attempt')}: {error.get('error')}")
    lines.extend(["", "## Suggested Retry", ""])
    for record in failed:
        retry_source = "huggingface" if args.source in {"auto", "modelscope"} else args.source
        retry = [
            sys.executable, "scripts/download_required_weights.py",
            "--output-root", str(args.output_root),
            "--weights-config", str(args.weights_config),
            "--source", retry_source,
            "--only", str(record.get("key")),
            "--repo-workers", "1",
            "--file-workers", str(args.file_workers),
            "--timeout", str(args.timeout_sec),
        ]
        if args.proxy:
            retry += ["--proxy", args.proxy]
        if args.http_proxy:
            retry += ["--http-proxy", args.http_proxy]
        if args.https_proxy:
            retry += ["--https-proxy", args.https_proxy]
        if args.all_proxy:
            retry += ["--all-proxy", args.all_proxy]
        lines.append(f"```bash\n{shlex.join(retry)}\n```")
    (output_root / "download_failure_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_targets(config: Dict[str, Dict[str, Any]], output_root: Path) -> None:
    targets = configured_targets(config, output_root)
    for key in [*ALL_KNOWN_KEYS, *sorted(set(targets) - set(ALL_KNOWN_KEYS))]:
        if key not in targets:
            continue
        spec = targets[key]
        print(f"{key}")
        print(f"  hf_repo: {spec.get('hf_repo')}")
        print(f"  modelscope_repo: {spec.get('modelscope_repo')}")
        print(f"  local_dir: {spec.get('local_dir')}")
        print(f"  optional: {spec.get('optional')}")


def main() -> int:
    args = parse_args()
    config = load_weights_config(args.weights_config)
    if args.list_targets:
        print_targets(config, args.output_root)
        return 0
    if args.status:
        payload = all_status(config, args.output_root)
        write_status_reports(args.output_root, payload)
        print_status(payload)
        return 0

    keys = selected_keys(args, config)
    set_env_before_import(args)
    print(f"Proxy enabled: {'yes' if proxy_enabled(args) else 'no'}")
    if proxy_enabled(args):
        print(f"  HTTP_PROXY={os.environ.get('HTTP_PROXY')}")
        print(f"  HTTPS_PROXY={os.environ.get('HTTPS_PROXY')}")
        print(f"  ALL_PROXY={os.environ.get('ALL_PROXY')}")
    print(f"HF endpoint: {os.environ.get('HF_ENDPOINT') or '<default huggingface.co>'}")
    jobs = repo_jobs(args, config, keys)
    print(f"Downloading/checking {len(jobs)} repos to {args.output_root} using source={args.source}")
    for job in jobs:
        mirror = job.get("modelscope_repo") or "<not configured>"
        partial_count = len(partial_files(Path(job["local_dir"])))
        print(f"  - {job['key']}: {job['hf_repo']} -> {job['local_dir']} (modelscope={mirror}, partials={partial_count})")
        if job.get("clean_partials"):
            print(f"    cleaned partials: {job['clean_partials']['removed_count']}")
            for root in job["clean_partials"].get("scanned_roots", []):
                print(f"    scanned partial root: {root}")
    records: List[Dict[str, Any]] = []
    if args.repo_workers <= 1:
        for job in jobs:
            records.append(run_one(job))
    else:
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.repo_workers, mp_context=ctx) as executor:
            futures = [executor.submit(run_one, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                try:
                    records.append(future.result())
                except Exception as exc:
                    records.append({
                        "key": "<worker>",
                        "repo": "<unknown>",
                        "local_dir": "<unknown>",
                        "requested_source": args.source,
                        "ok": False,
                        "error": f"worker failure: {exc}",
                        "traceback": traceback.format_exc(),
                    })
    records.sort(key=lambda item: item.get("key", ""))
    write_manifest(args.output_root, records)
    write_failure_report(args.output_root, args, records)
    status_payload = all_status(config, args.output_root)
    write_status_reports(args.output_root, status_payload)
    for record in records:
        status = "OK" if record.get("ok") else "FAILED"
        source = record.get("resolved_source", record.get("requested_source"))
        print(f"[{status}] {record['key']} {record['repo']} via {source} -> {record['local_dir']}")
        if record.get("error"):
            print(f"  error: {record['error']}", file=sys.stderr)
    return 0 if all(record.get("ok") for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
