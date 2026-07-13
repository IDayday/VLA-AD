#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not Path(path).is_file():
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        value = json.load(f)
    return value if isinstance(value, dict) else {}


def _get(data: Dict[str, Any], section: str, key: str, default: float = 0.0) -> float:
    obj = data.get(section, {}) if isinstance(data.get(section), dict) else {}
    try:
        return float(obj.get(key, default))
    except (TypeError, ValueError):
        return default


def _retr(data: Dict[str, Any], route: str, key: str, default: float = 0.0) -> float:
    retrieval = data.get("retrieval", {}) if isinstance(data.get("retrieval"), dict) else {}
    obj = retrieval.get(route, {}) if isinstance(retrieval.get(route), dict) else {}
    try:
        return float(obj.get(key, default))
    except (TypeError, ValueError):
        return default


def _ci(boot: Dict[str, Any], name: str) -> str:
    metrics = boot.get("metrics", {}) if isinstance(boot.get("metrics"), dict) else {}
    item = metrics.get(name, {}) if isinstance(metrics.get(name), dict) else {}
    if not item:
        return "n/a"
    return f"{item.get('mean', 0):.4g} [{item.get('ci95_low', 0):.4g}, {item.get('ci95_high', 0):.4g}]"


def build_markdown(eval_data: Dict[str, Any], gate: Dict[str, Any], boot: Dict[str, Any], old: Dict[str, Any]) -> str:
    metrics = eval_data.get("metrics", {}) if isinstance(eval_data.get("metrics"), dict) else {}
    comparisons = eval_data.get("comparisons", {}) if isinstance(eval_data.get("comparisons"), dict) else {}
    status = str(gate.get("status", "UNKNOWN"))
    allow_hidden = status in {"READY", "CONDITIONAL"}
    lines = [
        "# Stage1-v2 External Knowledge Utility Summary",
        "",
        f"- checkpoint: `{eval_data.get('stage1_checkpoint')}`",
        f"- samples: `{eval_data.get('num_samples')}`",
        f"- eval token file: `{eval_data.get('eval_token_file')}`",
        f"- gate status: `{status}`",
        f"- ALLOW_HIDDEN_CACHE_BUILD: `{str(allow_hidden).lower()}`",
        "- ALLOW_STAGE2: `false` until hidden-cache smoke later passes",
        "",
        "## Q1: JEPA Dynamic Teacher",
        "",
        f"- trained dyn loss: `{metrics.get('trained_dyn_loss')}`",
        f"- random dyn loss: `{metrics.get('random_dyn_loss')}`",
        f"- dyn top1/top5/MRR: `{_retr(eval_data, 'trained_dyn', 'top1'):.4g}` / `{_retr(eval_data, 'trained_dyn', 'top5'):.4g}` / `{_retr(eval_data, 'trained_dyn', 'mrr'):.4g}`",
        f"- dyn positive margin: `{_retr(eval_data, 'trained_dyn', 'positive_margin'):.4g}`",
        f"- image-only dyn ratio: `{comparisons.get('image_only_dyn_loss_over_trained_dyn_loss')}`; CI {_ci(boot, 'image_only_dyn_ratio')}",
        f"- slot-only dyn gain: `{comparisons.get('slot_only_dyn_gain')}`; CI {_ci(boot, 'slot_only_dyn_gain')}",
        "",
        "## Q2: VGGT Geometry Teacher",
        "",
        f"- trained geo loss: `{metrics.get('trained_geo_loss')}`",
        f"- random geo loss: `{metrics.get('random_geo_loss')}`",
        f"- geo top1/top5/MRR: `{_retr(eval_data, 'trained_geo', 'top1'):.4g}` / `{_retr(eval_data, 'trained_geo', 'top5'):.4g}` / `{_retr(eval_data, 'trained_geo', 'mrr'):.4g}`",
        f"- geo positive margin: `{_retr(eval_data, 'trained_geo', 'positive_margin'):.4g}`",
        f"- image-only geo ratio: `{comparisons.get('image_only_geo_loss_over_trained_geo_loss')}`; CI {_ci(boot, 'image_only_geo_ratio')}",
        f"- slot-only geo gain: `{comparisons.get('slot_only_geo_gain')}`; CI {_ci(boot, 'slot_only_geo_gain')}",
        "",
        "## Q3: Shortcut Diagnostics",
        "",
        f"- high-mask dyn ratio: `{comparisons.get('high_mask_dyn_loss_over_trained_dyn_loss')}`",
        f"- high-mask geo ratio: `{comparisons.get('high_mask_geo_loss_over_trained_geo_loss')}`",
        f"- random-slots dyn ratio: `{comparisons.get('random_slots_dyn_loss_over_trained_dyn_loss')}`",
        f"- random-slots geo ratio: `{comparisons.get('random_slots_geo_loss_over_trained_geo_loss')}`",
        "",
        "## Q4: Planning Relevance",
        "",
        f"- normal probe loss: `{metrics.get('trained_probe_loss')}`",
        f"- zero dyn probe ratio: `{comparisons.get('zero_dyn_probe_ratio')}`; CI {_ci(boot, 'zero_dyn_probe_ratio')}",
        f"- zero geo probe ratio: `{comparisons.get('zero_geo_probe_ratio')}`; CI {_ci(boot, 'zero_geo_probe_ratio')}",
        f"- dyn-only probe loss: `{metrics.get('dyn_only_probe_loss')}`",
        f"- geo-only probe loss: `{metrics.get('geo_only_probe_loss')}`",
        f"- no-signal/history-only probe loss: `{metrics.get('no_signal_probe_loss')}`",
        "",
        "## Q5: Replay / Direct Trajectory",
        "",
        f"- replay CE loss: `{_get(eval_data, 'replay', 'replay_ce_loss')}`",
        f"- replay token count: `{_get(eval_data, 'replay', 'replay_token_count')}`",
        f"- direct parse OK: `{_get(eval_data, 'direct', 'direct_traj_parse_ok_ratio')}`; CI {_ci(boot, 'direct_traj_parse_ok_ratio')}",
        f"- direct L1: `{_get(eval_data, 'direct', 'direct_traj_l1')}`; CI {_ci(boot, 'direct_traj_l1')}",
        "",
        "## Q6: Hidden Drift",
        "",
        f"- hidden drift cosine: `{_get(eval_data, 'hidden', 'hidden_drift_cosine')}`; CI {_ci(boot, 'hidden_drift_cosine')}",
        f"- hidden drift L2: `{_get(eval_data, 'hidden', 'hidden_drift_l2')}`",
        "",
        "## Old Stage1-v1 Comparison",
        "",
    ]
    if old:
        lines.extend(
            [
                f"- old checkpoint: `{old.get('stage1_checkpoint')}`",
                f"- old image-only dyn ratio: `{_get(old, 'comparisons', 'image_only_dyn_loss_over_trained_dyn_loss')}`",
                f"- old image-only geo ratio: `{_get(old, 'comparisons', 'image_only_geo_loss_over_trained_geo_loss')}`",
                f"- old slot-only dyn gain: `{_get(old, 'comparisons', 'slot_only_dyn_gain')}`",
                f"- old slot-only geo gain: `{_get(old, 'comparisons', 'slot_only_geo_gain')}`",
            ]
        )
    else:
        lines.append("- old_stage1_comparison: `missing`")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- external knowledge injection useful: `{'true' if status in {'READY', 'CONDITIONAL'} else 'false'}`",
            f"- allowed to proceed to hidden cache smoke plan: `{str(allow_hidden).lower()}`",
            "- full hidden cache executed: `false`",
            "- Stage2 launched: `false`",
            "- Stage3 launched: `false`",
            "- residual diffusion enabled: `false`",
            "- action-side CoT restored: `false`",
            "- token counts changed: `false`",
            "- baseline A0 full navtest PDMS: `0.864891`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage1-v2 external knowledge utility.")
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--gate-json", type=Path, required=True)
    parser.add_argument("--bootstrap-json", type=Path, required=True)
    parser.add_argument("--old-stage1-eval-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    text = build_markdown(_load(args.eval_json), _load(args.gate_json), _load(args.bootstrap_json), _load(args.old_stage1_eval_json))
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
