#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


ISOLATED_ROOT = Path("/mnt/project/bit_drive_left_tail")
DEFAULT_CODE_ROOT = ISOLATED_ROOT / "VLA-AD" if (ISOLATED_ROOT / "VLA-AD").is_dir() else ISOLATED_ROOT
DEFAULT_EXP_ROOT = ISOLATED_ROOT / "experiments/bit_drive"
DEFAULT_NAVSIM_ROOT = Path("/mnt/navsim")
SHARED_ROOT = Path("/mnt/project/VLA-AD")
SHARED_EXPERIMENT_ROOT = SHARED_ROOT / "experiments"

STAGES = {
    "2k": 2000,
    "8k": 8000,
    "30k": 30000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged D5-M2 final active safety mining.")
    parser.add_argument("--stage", choices=("2k", "8k", "30k", "all"), default="2k")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-if-low-yield", action="store_true")
    parser.add_argument("--min-nc-per-1k", type=float, default=0.625)
    parser.add_argument("--min-ttc-per-1k", type=float, default=6.25)
    parser.add_argument("--min-dac-fix-per-1k", type=float, default=12.5)
    parser.add_argument("--candidate-modes", default="base_det,bit_det,bit_stochastic_seed0,bit_stochastic_seed1,bit_stochastic_seed2")
    parser.add_argument("--sampling-mode", default="mixed")
    parser.add_argument("--sampling-pool-multiplier", type=float, default=1.5)
    parser.add_argument("--max-sampling-pool-scenes", type=int, default=60000)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--code-root", type=Path, default=DEFAULT_CODE_ROOT)
    parser.add_argument("--exp-root", type=Path, default=DEFAULT_EXP_ROOT)
    parser.add_argument("--navsim-root", type=Path, default=DEFAULT_NAVSIM_ROOT)
    parser.add_argument("--base-config", default="configs/bit_drive/bit_ablation_base_no_bit.yaml")
    parser.add_argument("--base-checkpoint", type=Path, default=SHARED_ROOT / "checkpoints/recogdrive/ReCogDrive-2B-IL")
    parser.add_argument("--bit-config", default="configs/bit_drive/v3/bit_v3_C1_lateral_terminal.yaml")
    parser.add_argument("--bit-checkpoint", type=Path, default=None)
    parser.add_argument("--metric-cache-dir", type=Path, action="append", default=None)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def assert_not_shared_output(path: Path) -> None:
    resolved = path.resolve()
    shared = SHARED_EXPERIMENT_ROOT.resolve()
    if resolved == shared or shared in resolved.parents:
        raise RuntimeError(f"Refusing to write outputs under shared experiment root: {shared}")


def stage_sequence(stage: str) -> List[str]:
    if stage == "all":
        return ["2k", "8k", "30k"]
    return [stage]


def discover_bit_checkpoint(exp_root: Path, explicit: Optional[Path]) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            exp_root / "v3/C1_lateral_terminal/best.ckpt",
            exp_root / "v2/B5_terminal_only_context/best.ckpt",
            SHARED_ROOT / "experiments/bit_drive/v3/C1_lateral_terminal/best.ckpt",
            SHARED_ROOT / "experiments/bit_drive/v2/B5_terminal_only_context/best.ckpt",
        ]
    )
    for path in candidates:
        if path.is_file():
            return path
    return explicit


def metric_cache_dirs(args: argparse.Namespace) -> List[Path]:
    if args.metric_cache_dir:
        return args.metric_cache_dir
    for path in (args.exp_root / "d5_mining/mining_summary.json", args.exp_root / "d5_mining/metadata.json"):
        data = read_json(path)
        dirs = data.get("metric_cache_dirs")
        if isinstance(dirs, list) and dirs:
            return [Path(item) for item in dirs]
    return []


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def count(summary: Dict[str, Any], key: str) -> int:
    return int(summary.get(key, 0) or 0)


def add_rates(summary: Dict[str, Any]) -> Dict[str, Any]:
    scenes = max(1, int(summary.get("scenes_processed", 0) or 0))
    enriched = dict(summary)
    enriched["hard_nc_per_1k"] = 1000.0 * count(summary, "hard_nc_regressions") / scenes
    enriched["hard_ttc_per_1k"] = 1000.0 * count(summary, "hard_ttc_regressions") / scenes
    enriched["dac_fixes_per_1k"] = 1000.0 * count(summary, "dac_fixes") / scenes
    return enriched


def gate(summary: Dict[str, Any]) -> Dict[str, bool]:
    hard_nc = count(summary, "hard_nc_regressions")
    hard_ttc = count(summary, "hard_ttc_regressions")
    soft = max(count(summary, "soft_safety_regressions"), hard_nc + hard_ttc)
    dac = count(summary, "dac_fixes")
    ideal = hard_nc >= 50 and hard_ttc >= 150 and dac >= 200
    acceptable = hard_nc >= 30 and hard_ttc >= 100 and soft >= 500 and dac >= 200
    return {"ideal_gate_passed": ideal, "acceptable_gate_passed": acceptable, "gate_passed": ideal or acceptable}


def low_yield(summary: Dict[str, Any], args: argparse.Namespace) -> bool:
    scenes = max(1, int(summary.get("scenes_processed", 0) or 0))
    if scenes < 8000:
        return False
    return (
        count(summary, "hard_nc_regressions") < args.min_nc_per_1k * scenes / 1000.0
        or count(summary, "hard_ttc_regressions") < args.min_ttc_per_1k * scenes / 1000.0
        or count(summary, "dac_fixes") < args.min_dac_fix_per_1k * scenes / 1000.0
    )


def promising_yield(summary: Dict[str, Any], args: argparse.Namespace) -> bool:
    scenes = max(1, int(summary.get("scenes_processed", 0) or 0))
    if scenes <= 0 or summary.get("missing"):
        return False
    return (
        count(summary, "hard_nc_regressions") >= args.min_nc_per_1k * scenes / 1000.0
        and count(summary, "hard_ttc_regressions") >= args.min_ttc_per_1k * scenes / 1000.0
        and count(summary, "dac_fixes") >= args.min_dac_fix_per_1k * scenes / 1000.0
    )


def should_pivot(summary: Dict[str, Any], args: argparse.Namespace) -> bool:
    if summary.get("gate_passed"):
        return False
    if summary.get("stage") == "2k" and not promising_yield(summary, args):
        return True
    return low_yield(summary, args) or summary.get("stage") == "30k"


def command_for_stage(args: argparse.Namespace, output_root: Path, max_scenes: int) -> List[Any]:
    bit_checkpoint = discover_bit_checkpoint(args.exp_root, args.bit_checkpoint)
    cmd: List[Any] = [
        args.python,
        "scripts/mine_bit_counterfactual_safety_cases_v2.py",
        "--navsim-root",
        args.navsim_root,
        "--base-config",
        args.base_config,
        "--base-checkpoint",
        args.base_checkpoint,
        "--bit-config",
        args.bit_config,
        "--split",
        "navtrain",
        "--output-dir",
        output_root,
        "--max-total-scenes",
        max_scenes,
        "--sampling-pool-multiplier",
        args.sampling_pool_multiplier,
        "--max-sampling-pool-scenes",
        args.max_sampling_pool_scenes,
        "--target-hard-nc",
        50,
        "--target-hard-ttc",
        150,
        "--target-soft-safety",
        500,
        "--target-dac-fix",
        200,
        "--candidate-modes",
        args.candidate_modes,
        "--sampling-mode",
        args.sampling_mode,
        "--precision",
        args.precision,
        "--device",
        args.device,
        "--num-gpus",
        args.num_gpus,
        "--log-every",
        args.log_every,
    ]
    if bit_checkpoint is not None:
        cmd.extend(["--bit-checkpoint", bit_checkpoint])
    for metric_dir in metric_cache_dirs(args):
        cmd.extend(["--metric-cache-dir", metric_dir])
    return cmd


def run_cmd(cmd: List[Any], *, dry_run: bool, cwd: Path) -> None:
    print("RUN", " ".join(str(part) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run([str(part) for part in cmd], cwd=cwd, check=True)


def load_stage_summary(output_root: Path, stage_name: str) -> Dict[str, Any]:
    summary = read_json(output_root / "mining_summary.json")
    if not summary:
        return {"stage": stage_name, "missing": True}
    enriched = add_rates(summary)
    enriched["stage"] = stage_name
    enriched.update(gate(enriched))
    return enriched


def old_sparse_summary(exp_root: Path) -> Dict[str, Any]:
    audit = read_json(exp_root / "d5_mining_audit/audit_summary.json")
    if audit:
        return audit
    summary = read_json(exp_root / "d5_mining/mining_summary.json")
    if not summary:
        return {}
    hard_nc = int((summary.get("tag_counts") or {}).get("bit_nc_regression", 0))
    hard_ttc = int((summary.get("tag_counts") or {}).get("bit_ttc_regression", 0))
    dac = int((summary.get("tag_counts") or {}).get("bit_dac_fix", 0))
    return {
        "total_rows": summary.get("total_rows"),
        "hard_nc_regressions": hard_nc,
        "hard_ttc_regressions": hard_ttc,
        "soft_safety_regressions": hard_nc + hard_ttc,
        "dac_fixes": dac,
        "split_counts": summary.get("split_counts"),
        "training_allowed": summary.get("training_allowed"),
    }


def table_from_yields(title: str, yields: Dict[str, Dict[str, Any]]) -> List[str]:
    lines = [
        f"## {title}",
        "",
        "| Name | Candidate rows | Scenes attempted | Scenes valid | Hard NC | Hard TTC | Soft safety | DAC fixes | Zero fixes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not yields:
        lines.append("| pending | - | - | - | - | - | - | - | - |")
        return lines
    for name, row in sorted(yields.items()):
        lines.append(
            f"| {name} | {int(row.get('candidate_rows', 0))} | {int(row.get('scenes_attempted', 0))} | "
            f"{int(row.get('scenes_valid', 0))} | {int(row.get('hard_nc', 0))} | {int(row.get('hard_ttc', 0))} | "
            f"{int(row.get('soft_safety', 0))} | {int(row.get('dac_fixes', 0))} | {int(row.get('zero_fixes', 0))} |"
        )
    return lines


def next_command(args: argparse.Namespace, stage_results: List[Dict[str, Any]], final_summary: Dict[str, Any]) -> str:
    if final_summary.get("gate_passed"):
        return (
            "python scripts/run_bit_d5_conservative_plan.py "
            "--project-root /mnt/project/bit_drive_left_tail/VLA-AD "
            "--exp-root /mnt/project/bit_drive_left_tail/experiments/bit_drive/v5_d5 "
            "--counterfactual-label-jsonl /mnt/project/bit_drive_left_tail/experiments/bit_drive/d5_mining_v2_final/selected_training_labels.jsonl "
            "--max-train-samples 4096 --max-eval-samples 1024 --num-steps 1000 --batch-size 8 --precision bf16 --only D5A,D5E"
        )
    if should_pivot(final_summary, args):
        return "# D5 training blocked. Review /mnt/project/bit_drive_left_tail/experiments/bit_drive/BIT_D5_PIVOT_PLAN.md"
    stages_done = {row.get("stage") for row in stage_results if not row.get("missing")}
    next_stage = "8k" if "2k" in stages_done and "8k" not in stages_done else "30k" if "8k" in stages_done and "30k" not in stages_done else None
    if next_stage is None:
        return "Write pivot plan; do not train D5."
    stop_flag = " --stop-if-low-yield" if next_stage in {"8k", "30k"} else ""
    return (
        "python scripts/run_d5_m2_final_mining_plan.py "
        "--code-root /mnt/project/bit_drive_left_tail/VLA-AD "
        "--exp-root /mnt/project/bit_drive_left_tail/experiments/bit_drive "
        "--navsim-root /mnt/navsim "
        f"--stage {next_stage} --candidate-modes {args.candidate_modes} --sampling-mode {args.sampling_mode} --resume{stop_flag}"
    )


def write_final_report(args: argparse.Namespace, output_root: Path, stage_results: List[Dict[str, Any]], mode_report: Dict[str, Any]) -> None:
    final_summary = stage_results[-1] if stage_results else {}
    old = old_sparse_summary(args.exp_root)
    true_multi_candidate = bool(mode_report.get("v2_multi_candidate_mining_ran")) or (
        int(final_summary.get("candidate_rows", 0) or 0) > int(final_summary.get("scenes_processed", 0) or 0)
        and len(final_summary.get("candidate_modes") or []) > 1
    )
    lines = [
        "# BIT D5-M2 Final Mining Report",
        "",
        "No D5 training and no GRPO were run by this mining plan.",
        "",
        "## Old Sparse Mining Result",
        "",
        f"- Rows: `{old.get('total_rows')}`",
        f"- Hard NC regressions: `{old.get('hard_nc_regressions')}`",
        f"- Hard TTC regressions: `{old.get('hard_ttc_regressions')}`",
        f"- Soft safety-family rows: `{old.get('soft_safety_regressions')}`",
        f"- DAC fixes: `{old.get('dac_fixes')}`",
        f"- Split distribution: `{old.get('split_counts')}`",
        f"- Training allowed: `{old.get('training_allowed')}`",
        "",
        "## Multi-Candidate Mining Status",
        "",
        f"True multi-candidate mining ran: `{true_multi_candidate}`",
        f"Previously completed v2 mining before this stage: `{mode_report.get('v2_multi_candidate_mining_ran')}`",
        "",
        "## Stage Results",
        "",
        "| Stage | Scenes | Candidate rows | Hard NC | Hard TTC | Soft safety | DAC fixes | Zero fixes | NC/1k | TTC/1k | DAC/1k | Gate | Continue worthwhile |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not stage_results:
        lines.append("| pending | - | - | - | - | - | - | - | - | - | - | false | false |")
    for row in stage_results:
        if row.get("missing"):
            lines.append(f"| {row.get('stage')} | missing | - | - | - | - | - | - | - | - | - | false | false |")
            continue
        worthwhile = bool(row.get("gate_passed")) or promising_yield(row, args)
        lines.append(
            f"| {row.get('stage')} | {int(row.get('scenes_processed', 0))} | {int(row.get('candidate_rows', 0))} | "
            f"{int(row.get('hard_nc_regressions', 0))} | {int(row.get('hard_ttc_regressions', 0))} | "
            f"{int(row.get('soft_safety_regressions', 0))} | {int(row.get('dac_fixes', 0))} | {int(row.get('zero_fixes', 0))} | "
            f"{float(row.get('hard_nc_per_1k', 0.0)):.3f} | {float(row.get('hard_ttc_per_1k', 0.0)):.3f} | "
            f"{float(row.get('dac_fixes_per_1k', 0.0)):.3f} | {row.get('gate_passed')} | {worthwhile} |"
        )
    lines.extend(
        [
            "",
            "## D5 Gate Status",
            "",
            "Ideal gate: hard NC >= 50, hard TTC >= 150, DAC fixes >= 200.",
            "",
            "Acceptable gate: hard NC >= 30, hard TTC >= 100, soft safety >= 500, DAC fixes >= 200.",
            "",
            f"Current gate passed: `{final_summary.get('gate_passed', False)}`",
            "",
            "## Decision",
            "",
        ]
    )
    if final_summary.get("gate_passed"):
        decision = "Run only D5A and D5E first."
    elif should_pivot(final_summary, args):
        decision = "Stop supervised D5 and pivot; do not train D5."
    else:
        decision = "Continue staged active mining; do not train D5 yet."
    lines.extend([decision, "", "## Exact Next Command", "", "```bash", next_command(args, stage_results, final_summary), "```", ""])
    lines.extend(table_from_yields("Candidate Mode Yield Table", final_summary.get("candidate_mode_yields", {})))
    lines.extend([""])
    lines.extend(table_from_yields("Sampling Stratum Yield Table", final_summary.get("sampling_stratum_yields", {})))
    (args.exp_root / "BIT_D5_M2_FINAL_MINING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pivot_plan(args: argparse.Namespace, final_summary: Dict[str, Any]) -> None:
    lines = [
        "# BIT D5 Pivot Plan",
        "",
        "Supervised D5 is not supported because non-test counterfactual safety-regression samples are too sparse.",
        "",
        "No D5 training and no GRPO should be run from the sparse label set.",
        "",
        "## Recommended Direction",
        "",
        "Choose B. BiT + interaction-risk teacher.",
        "",
        "- Train a risk head from VLM/JEPA or privileged train-only interaction annotations.",
        "- Use annotations only as supervision during training, never as inference features.",
        "- Target NC/TTC directly with interaction-aware dynamic-risk labels.",
        "- Keep BiT terminal/path behavior as auxiliary signal when it helps DAC, without claiming NC/TTC success.",
        "",
        "## Alternatives",
        "",
        "A. BiT as DAC/path auxiliary only: keep terminal/path auxiliary loss, avoid direct conditioning or keep it very weak, and do not claim NC/TTC resolution.",
        "",
        "C. BiT + RL: use online PDM/GRPO to generate a safety signal only after a safe IL initialization is selected.",
        "",
        "## Final Mining Counts",
        "",
        f"- Stage: `{final_summary.get('stage')}`",
        f"- Scenes: `{final_summary.get('scenes_processed')}`",
        f"- Hard NC regressions: `{final_summary.get('hard_nc_regressions')}`",
        f"- Hard TTC regressions: `{final_summary.get('hard_ttc_regressions')}`",
        f"- Soft safety regressions: `{final_summary.get('soft_safety_regressions')}`",
        f"- DAC fixes: `{final_summary.get('dac_fixes')}`",
    ]
    (args.exp_root / "BIT_D5_PIVOT_PLAN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.exp_root.mkdir(parents=True, exist_ok=True)
    assert_not_shared_output(args.exp_root)
    output_root = args.output_root or (args.exp_root / "d5_mining_v2_final")
    assert_not_shared_output(output_root)
    mode_report = read_json(args.exp_root / "D5_MINING_MODE_REPORT.json")
    stage_results: List[Dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)

    for stage_name in stage_sequence(args.stage):
        max_scenes = STAGES[stage_name]
        existing = load_stage_summary(output_root, stage_name)
        if args.resume and not args.dry_run and not existing.get("missing") and int(existing.get("scenes_processed", 0) or 0) >= max_scenes:
            print(f"SKIP {stage_name}: existing summary has {existing.get('scenes_processed')} scenes.", flush=True)
            stage_results.append(existing)
        else:
            cmd = command_for_stage(args, output_root, max_scenes)
            run_cmd(cmd, dry_run=args.dry_run, cwd=args.code_root)
            if args.dry_run:
                stage_results.append({"stage": stage_name, "planned_scenes": max_scenes, "missing": True})
            else:
                stage_results.append(load_stage_summary(output_root, stage_name))

        current = stage_results[-1]
        if current.get("gate_passed"):
            break
        if args.stop_if_low_yield and low_yield(current, args):
            print("Stopping staged mining: low-yield rule triggered.", flush=True)
            break

    write_final_report(args, output_root, stage_results, mode_report)
    final_summary = stage_results[-1] if stage_results else {}
    if (not args.dry_run) and should_pivot(final_summary, args):
        write_pivot_plan(args, final_summary)
    print(json.dumps({"stage_results": stage_results, "report": str(args.exp_root / "BIT_D5_M2_FINAL_MINING_REPORT.md")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
