from __future__ import annotations

from typing import Any

from .dataclasses import ParetoSupportArchive


def archive_summary_row(archive: ParetoSupportArchive) -> dict[str, Any]:
    return {
        "scene_token": archive.scene_token,
        "support_count": len(archive.support_set),
        "true_eval_count": len(archive.evaluated_candidates),
        "valid_count": sum(1 for c in archive.evaluated_candidates if c.valid_mask),
        "archive_hypervolume": float(archive.archive_hypervolume),
        "best_pdms": float(archive.oracle_best.get("best_pdms", 0.0)),
        "best_utility": float(archive.oracle_best.get("best_utility", 0.0)),
        "best_ep": float(archive.oracle_best.get("best_ep", 0.0)),
        "best_ddc": float(archive.oracle_best.get("best_ddc", 0.0)),
    }
