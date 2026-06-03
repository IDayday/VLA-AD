#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_A0_CONFIG = Path("configs/bit_drive/bit_ablation_base_no_bit.yaml")
DEFAULT_B3_CONFIG = Path("configs/bit_drive/v3/bit_v3_C1_lateral_terminal.yaml")
EVALUATOR = Path("navsim/planning/script/run_pdm_score_recogdrive.py")
REQUIRED_PDM_COLUMNS = [
    "token",
    "score",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
]


def _existing_path(value: Optional[str | Path]) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        return None
    path = Path(value).expanduser()
    return path if path.exists() else None


def _as_path(value: Optional[str | Path]) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        return None
    return Path(value).expanduser()


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_format_value(item) for item in value) + "]"
    return str(value)


def yaml_agent_overrides(path: Path) -> List[str]:
    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return []
    return [f"agent.{key}={_format_value(value)}" for key, value in sorted(data.items())]


def infer_vlm_path(checkpoint_root: Optional[Path], explicit: Optional[Path]) -> Optional[Path]:
    if explicit and explicit.exists():
        return explicit
    if checkpoint_root:
        candidate = checkpoint_root / "recogdrive" / "ReCogDrive-VLM-2B"
        if candidate.exists():
            return candidate
    return None


def shell_script_for_job(
    *,
    name: str,
    checkpoint: Path,
    config_path: Path,
    output_dir: Path,
    split: str,
    navsim_log_path: Path,
    sensor_blobs_path: Path,
    metric_cache_path: Path,
    vlm_path: Path,
    devices: int,
    master_port: int,
    max_samples: int,
    analysis_only: bool,
) -> str:
    run_dir = output_dir / name / "run"
    final_csv = output_dir / name / "pdm.csv"
    log_file = output_dir / name / "pdm_eval.log"
    overrides = [
        f"train_test_split={split}",
        "agent=recogdrive_agent",
        f"agent.checkpoint_path={checkpoint}",
        f"agent.vlm_path={vlm_path}",
        "agent.cam_type=single",
        "agent.grpo=false",
        "agent.cache_hidden_state=false",
        "agent.vlm_type=internvl",
        "agent.dit_type=small",
        "agent.vlm_size=small",
        "agent.sampling_method=ddim",
        f"metric_cache_path={metric_cache_path}",
        f"navsim_log_path={navsim_log_path}",
        f"sensor_blobs_path={sensor_blobs_path}",
        f"output_dir={run_dir}",
        f"experiment_name=risk_vla_round1_{name}",
        f"train_test_split.scene_filter.max_scenes=${{MAX_SAMPLES:-{max_samples}}}",
    ]
    overrides.extend(yaml_agent_overrides(config_path))
    body = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# RISK-VLA small PDM generation for {name}.",
        "# Default is dry-run/no execution; set EXECUTE=1 to run.",
        "# Oracle router is not involved here. A0/B3 PDMs are analysis artifacts; train/val PDMs are for train/val risk labels.",
        f"# analysis_only={str(analysis_only).lower()}",
        f"MAX_SAMPLES=\"${{MAX_SAMPLES:-{max_samples}}}\"",
        f"EXECUTE=\"${{EXECUTE:-0}}\"",
        f"DEVICES=\"${{DEVICES:-{devices}}}\"",
        f"MASTER_PORT=\"${{MASTER_PORT:-{master_port}}}\"",
        f"export NAVSIM_EXP_ROOT=\"${{NAVSIM_EXP_ROOT:-{output_dir / 'navsim_exp'}}}\"",
        "export OPENSCENE_DATA_ROOT=\"${OPENSCENE_DATA_ROOT:-/mnt/navsim/openscene}\"",
        "export NUPLAN_MAPS_ROOT=\"${NUPLAN_MAPS_ROOT:-/mnt/navsim/maps}\"",
        f"mkdir -p {shlex.quote(str(run_dir))} {shlex.quote(str(final_csv.parent))}",
        "CMD=(",
        "  torchrun",
        "  --nproc_per_node=\"${DEVICES}\"",
        "  --master_port=\"${MASTER_PORT}\"",
        f"  {shlex.quote(str(EVALUATOR))}",
    ]
    for part in overrides:
        if part.startswith("train_test_split.scene_filter.max_scenes="):
            body.append("  \"train_test_split.scene_filter.max_scenes=${MAX_SAMPLES}\"")
        else:
            body.append(f"  {shlex.quote(part)}")
    body.extend(
        [
            ")",
            "printf '[small_pdm] command: '; printf '%q ' \"${CMD[@]}\"; echo",
            "if [[ \"${EXECUTE}\" != \"1\" ]]; then",
            "  echo '[small_pdm] EXECUTE=0; no NAVSIM PDM evaluation launched.'",
            "  exit 0",
            "fi",
            f"\"${{CMD[@]}}\" 2>&1 | tee {shlex.quote(str(log_file))}",
            f"latest_csv=$(ls -t {shlex.quote(str(run_dir))}/*.csv 2>/dev/null | head -n 1 || true)",
            "if [[ -z \"${latest_csv}\" ]]; then",
            "  echo '[small_pdm] evaluator produced no CSV.' >&2",
            "  exit 1",
            "fi",
            f"cp \"${{latest_csv}}\" {shlex.quote(str(final_csv))}",
            f"python - <<'PY'\nimport pandas as pd\nfrom pathlib import Path\npath = Path({str(final_csv)!r})\nrequired = {REQUIRED_PDM_COLUMNS!r}\ndf = pd.read_csv(path)\nmissing = [c for c in required if c not in df.columns]\nif missing:\n    raise SystemExit(f'Missing required PDM columns in {{path}}: {{missing}}')\nprint(f'Validated PDM CSV columns: {{path}}')\nPY",
        ]
    )
    return "\n".join(body) + "\n"


def blocker_report(blockers: Iterable[str]) -> str:
    lines = ["# RISK-VLA Small PDM Generation Blockers", ""]
    blockers = list(blockers)
    if blockers:
        lines.extend(f"- {item}" for item in blockers)
    else:
        lines.append("_None._")
    lines.append("")
    return "\n".join(lines)


def evaluator_report(args: argparse.Namespace, blockers: List[str], *, vlm_path: Optional[Path]) -> str:
    lines = [
        "# RISK-VLA PDM Evaluator Path Report",
        "",
        "PDM CSVs are generated evaluation artifacts, not repo source files. This report records the exact evaluator path for small-scale Round 1 materialization.",
        "",
        "## Evaluator Command",
        "",
        f"- entrypoint: `{EVALUATOR}`",
        "- launcher: `torchrun`",
        "- agent config: `agent=recogdrive_agent` plus method-specific `agent.*` overrides",
        "",
        "## Required Config Overrides",
        "",
        "- `train_test_split=<split>`",
        "- `train_test_split.scene_filter.max_scenes=${MAX_SAMPLES}` for small-scale limiting",
        "- `agent.checkpoint_path=<checkpoint>`",
        "- `agent.vlm_path=<ReCogDrive-VLM-2B>`",
        "- `metric_cache_path=<metric cache>`",
        "- `navsim_log_path=<NAVSIM logs>`",
        "- `sensor_blobs_path=<sensor blobs>`",
        "- `output_dir=<isolated output dir>`",
        "",
        "## Token Subsetting / Max Samples",
        "",
        "This branch supports small-scale limiting through `train_test_split.scene_filter.max_scenes`. Token-list overrides are possible through scene-filter tokens, but this builder uses `max_scenes` by default.",
        "",
        "## Output CSVs",
        "",
        f"- `A0_base`: `{Path(args.output_dir) / 'A0_base' / 'pdm.csv'}`",
        f"- `B3_direct_bit`: `{Path(args.output_dir) / 'B3_direct_bit' / 'pdm.csv'}`",
        f"- `train`: `{Path(args.output_dir) / 'train' / 'pdm.csv'}`",
        f"- `val`: `{Path(args.output_dir) / 'val' / 'pdm.csv'}`",
        "",
        "## Resolved Inputs",
        "",
        f"- A0 checkpoint: `{args.a0_checkpoint}`",
        f"- B3 checkpoint: `{args.b3_checkpoint}`",
        f"- VLM path: `{vlm_path}`",
        f"- metric cache: `{args.metric_cache_path}`",
        f"- navsim logs: `{args.navsim_log_path}`",
        f"- sensor blobs: `{args.sensor_blobs_path}`",
        "",
        "## Missing Inputs",
        "",
    ]
    lines.extend(f"- {item}" for item in blockers) if blockers else lines.append("_None._")
    lines.append("")
    return "\n".join(lines)


def build_commands(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    commands_dir = output_dir / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_root = _as_path(os.getenv("CHECKPOINT_ROOT", "/mnt/project/VLA-AD/checkpoints"))
    vlm_path = infer_vlm_path(checkpoint_root, _as_path(args.vlm_path))
    a0_checkpoint = _existing_path(args.a0_checkpoint)
    b3_checkpoint = _existing_path(args.b3_checkpoint)
    metric_cache = _existing_path(args.metric_cache_path)
    navsim_logs = _existing_path(args.navsim_log_path)
    sensor_blobs = _existing_path(args.sensor_blobs_path)

    blockers: List[str] = []
    if not EVALUATOR.is_file():
        blockers.append(f"Evaluator entrypoint missing: {EVALUATOR}")
    if a0_checkpoint is None:
        blockers.append("A0 checkpoint is missing or does not exist.")
    if b3_checkpoint is None:
        blockers.append("B3 checkpoint is missing or does not exist.")
    if vlm_path is None:
        blockers.append("ReCogDrive VLM path is missing; set --vlm-path or CHECKPOINT_ROOT/recogdrive/ReCogDrive-VLM-2B.")
    if metric_cache is None:
        blockers.append("Metric cache path is missing or does not exist.")
    if navsim_logs is None:
        blockers.append("NAVSIM log path is missing or does not exist.")
    if sensor_blobs is None:
        blockers.append("Sensor blobs path is missing or does not exist.")
    if args.cache_path and not Path(args.cache_path).exists():
        blockers.append("Chunk cache path is missing or does not exist; PDM generation can be planned, but R0 overlay creation will be blocked.")

    (output_dir / "pdm_generation_blockers.md").write_text(blocker_report(blockers), encoding="utf-8")
    (output_dir / "evaluator_path_report.md").write_text(evaluator_report(args, blockers, vlm_path=vlm_path), encoding="utf-8")

    jobs = [
        ("generate_a0_pdm.sh", "A0_base", a0_checkpoint, DEFAULT_A0_CONFIG, args.split, True),
        ("generate_b3_pdm.sh", "B3_direct_bit", b3_checkpoint, DEFAULT_B3_CONFIG, args.split, True),
        ("generate_train_pdm.sh", "train", b3_checkpoint or a0_checkpoint, DEFAULT_B3_CONFIG if b3_checkpoint else DEFAULT_A0_CONFIG, "navtrain", False),
        ("generate_val_pdm.sh", "val", b3_checkpoint or a0_checkpoint, DEFAULT_B3_CONFIG if b3_checkpoint else DEFAULT_A0_CONFIG, "navval", False),
    ]
    command_rows: List[Dict[str, Any]] = []
    for script_name, name, checkpoint, config, split, analysis_only in jobs:
        path = commands_dir / script_name
        if blockers or checkpoint is None or vlm_path is None or metric_cache is None or navsim_logs is None or sensor_blobs is None:
            path.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n"
                "echo 'RISK-VLA PDM generation blocked. See pdm_generation_blockers.md.' >&2\n"
                "exit 1\n",
                encoding="utf-8",
            )
        else:
            path.write_text(
                shell_script_for_job(
                    name=name,
                    checkpoint=checkpoint,
                    config_path=config,
                    output_dir=output_dir,
                    split=split,
                    navsim_log_path=navsim_logs,
                    sensor_blobs_path=sensor_blobs,
                    metric_cache_path=metric_cache,
                    vlm_path=vlm_path,
                    devices=int(args.devices),
                    master_port=int(args.master_port),
                    max_samples=int(args.max_samples),
                    analysis_only=analysis_only,
                ),
                encoding="utf-8",
            )
        path.chmod(0o755)
        command_rows.append(
            {
                "name": name,
                "script": str(path),
                "output_csv": str(output_dir / name / "pdm.csv"),
                "split": split,
                "analysis_only": analysis_only,
                "config": str(config),
            }
        )

    lines = ["# RISK-VLA Small PDM Eval Commands", "", f"Output dir: `{output_dir}`", ""]
    lines.append("These scripts default to `EXECUTE=0` and include `train_test_split.scene_filter.max_scenes=${MAX_SAMPLES}`. They must not be used for full NAVSIM evaluation without explicit review.")
    lines.extend(["", "| name | split | analysis_only | script | output_csv |", "| --- | --- | --- | --- | --- |"])
    for row in command_rows:
        lines.append(f"| {row['name']} | {row['split']} | {row['analysis_only']} | `{row['script']}` | `{row['output_csv']}` |")
    lines.extend(["", "## Blockers", "", blocker_report(blockers)])
    (output_dir / "small_pdm_eval_commands.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {"output_dir": str(output_dir), "has_blockers": bool(blockers), "blockers": blockers, "commands": command_rows}
    (output_dir / "small_pdm_eval_commands.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dry-run-first small-scale PDM generation commands for RISK-VLA Round 1.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--a0-checkpoint", default=None)
    parser.add_argument("--b3-checkpoint", default=None)
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--metric-cache-path", default=None)
    parser.add_argument("--navsim-log-path", default=None)
    parser.add_argument("--sensor-blobs-path", default=None)
    parser.add_argument("--vlm-path", default=None)
    parser.add_argument("--split", default="navval")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=29671)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_commands(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
