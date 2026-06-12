from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


METRIC_ALIASES = {
    "pdms": ("pdms", "pdm_score", "score", "final_score"),
    "nc": ("nc", "no_at_fault_collisions"),
    "dac": ("dac", "drivable_area_compliance", "drivable_area"),
    "ttc": ("ttc", "time_to_collision_within_bound", "time_to_collision"),
    "ep": ("ep", "ego_progress", "progress"),
    "comfort": ("comfort", "history_comfort", "comfortable"),
    "ddc": ("ddc", "driving_direction_compliance"),
    "tlc": ("tlc", "traffic_light_compliance"),
}


def _find_metric_column(columns: Iterable[str], aliases: tuple[str, ...]) -> Optional[str]:
    normalized = {str(column).lower(): column for column in columns}
    for alias in aliases:
        if alias.lower() in normalized:
            return normalized[alias.lower()]
    return None


def _candidate_csvs(eval_dir: Path) -> list[Path]:
    paths = []
    for path in eval_dir.rglob("*.csv"):
        if path.name in {"checkpoint_eval_summary.tsv", "checkpoint_eval_submetrics.tsv"}:
            continue
        paths.append(path)
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def _load_best_metric_csv(eval_dir: Path) -> tuple[Path, pd.DataFrame]:
    for path in _candidate_csvs(eval_dir):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if _find_metric_column(df.columns, METRIC_ALIASES["pdms"]) is not None:
            return path, df
    raise FileNotFoundError(f"No PDM metric CSV with a PDMS/score column found under {eval_dir}")


def _scene_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = df.copy()
    if "token" in rows.columns:
        token = rows["token"].astype("string")
        rows = rows[token.notna() & (token.str.lower() != "average")]
    if "valid" in rows.columns:
        rows = rows[rows["valid"].astype(bool)]
    return rows


def summarize_eval(eval_dir: Path, checkpoint_id: str) -> Dict[str, object]:
    csv_path, df = _load_best_metric_csv(eval_dir)
    rows = _scene_rows(df)
    summary: Dict[str, object] = {
        "checkpoint_id": checkpoint_id,
        "eval_dir": str(eval_dir),
        "csv_path": str(csv_path),
        "num_rows": int(len(df)),
        "num_valid_rows": int(len(rows)),
    }
    for metric_name, aliases in METRIC_ALIASES.items():
        column = _find_metric_column(rows.columns, aliases)
        if column is None or rows.empty:
            summary[f"{metric_name}_mean"] = ""
            summary[f"{metric_name}_std"] = ""
            continue
        values = pd.to_numeric(rows[column], errors="coerce").dropna()
        summary[f"{metric_name}_mean"] = float(values.mean()) if not values.empty else ""
        summary[f"{metric_name}_std"] = float(values.std(ddof=0)) if len(values) > 1 else 0.0

    seed_column = _find_metric_column(rows.columns, ("seed", "eval_seed", "sample_seed"))
    pdms_column = _find_metric_column(rows.columns, METRIC_ALIASES["pdms"])
    if seed_column is not None and pdms_column is not None and not rows.empty:
        seed_means = rows.groupby(seed_column)[pdms_column].mean()
        summary["seed_mean_pdms_mean"] = float(seed_means.mean())
        summary["seed_mean_pdms_std"] = float(seed_means.std(ddof=0)) if len(seed_means) > 1 else 0.0
        summary["num_seeds"] = int(len(seed_means))
    else:
        summary["seed_mean_pdms_mean"] = summary.get("pdms_mean", "")
        summary["seed_mean_pdms_std"] = ""
        summary["num_seeds"] = ""
    return summary


def append_summary(summary_tsv: Path, row: Dict[str, object]) -> None:
    summary_tsv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "checkpoint_id",
        "eval_dir",
        "csv_path",
        "num_rows",
        "num_valid_rows",
        "pdms_mean",
        "pdms_std",
        "nc_mean",
        "nc_std",
        "dac_mean",
        "dac_std",
        "ttc_mean",
        "ttc_std",
        "ep_mean",
        "ep_std",
        "comfort_mean",
        "comfort_std",
        "ddc_mean",
        "ddc_std",
        "tlc_mean",
        "tlc_std",
        "seed_mean_pdms_mean",
        "seed_mean_pdms_std",
        "num_seeds",
    ]
    exists = summary_tsv.is_file()
    with summary_tsv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})

    try:
        df = pd.read_csv(summary_tsv, sep="\t")
        df["pdms_mean_numeric"] = pd.to_numeric(df["pdms_mean"], errors="coerce")
        best = df.sort_values("pdms_mean_numeric", ascending=False).iloc[0]
        best_path = summary_tsv.parent / "best_checkpoint_by_pdms.txt"
        best_path.write_text(
            f"checkpoint_id={best['checkpoint_id']}\n"
            f"pdms_mean={best['pdms_mean']}\n"
            f"eval_dir={best['eval_dir']}\n",
            encoding="utf-8",
        )
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--summary-tsv", required=True)
    args = parser.parse_args()

    row = summarize_eval(Path(args.eval_dir), args.checkpoint_id)
    append_summary(Path(args.summary_tsv), row)
    print(
        f"checkpoint_id={row['checkpoint_id']} "
        f"pdms_mean={row.get('pdms_mean', '')} "
        f"num_valid_rows={row.get('num_valid_rows', '')}"
    )


if __name__ == "__main__":
    main()
