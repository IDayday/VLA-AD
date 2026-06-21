#!/usr/bin/env python3
"""Analyze ReCogDrive Stage3 navtest checkpoint summaries without changing scoring."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


METRICS = ("pdms", "nc", "dac", "ttc", "ep", "comfort", "ddc", "tlc")


def _checkpoint_sort_key(checkpoint_id: str) -> tuple[int, int, str]:
    text = str(checkpoint_id)
    step_match = re.search(r"step[-_=](\d+)", text)
    epoch_match = re.search(r"epoch[-_=](\d+)", text)
    epoch = int(epoch_match.group(1)) if epoch_match else -1
    step = int(step_match.group(1)) if step_match else -1
    return (step, epoch, text)


def _read_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df["summary_path"] = str(path)
    for metric in METRICS:
        column = f"{metric}_mean"
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "num_valid_rows" in df.columns:
        df["num_valid_rows"] = pd.to_numeric(df["num_valid_rows"], errors="coerce")
    return df


def _summary_files(run_root: Path) -> list[Path]:
    if run_root.is_file():
        return [run_root]
    return sorted(run_root.rglob("checkpoint_eval_submetrics.tsv"))


def _format_float(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.6f}"
    except Exception:
        return ""


def _write_markdown(
    grouped: pd.DataFrame,
    all_rows: pd.DataFrame,
    output_md: Path,
    *,
    early_gate: float,
    original_final: float,
    safe_diffgrpo_best: float,
) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# ReCogDrive Stage3 Navtest PDMS Analysis")
    lines.append("")
    lines.append("This report reads existing navtest evaluation summaries only. It must not be used as a training reward/cache.")
    lines.append("")
    if grouped.empty:
        lines.append("No checkpoint rows found.")
        output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    best = grouped.sort_values("pdms_mean", ascending=False).iloc[0]
    latest = grouped.sort_values(["sort_step", "sort_epoch"], ascending=True).iloc[-1]
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Best checkpoint: `{best['checkpoint_id']}` PDMS `{_format_float(best['pdms_mean'])}`")
    lines.append(f"- Latest checkpoint: `{latest['checkpoint_id']}` PDMS `{_format_float(latest['pdms_mean'])}`")
    lines.append(f"- Early gate delta (`{early_gate:.4f}`): `{_format_float(latest['pdms_mean'] - early_gate)}`")
    lines.append(
        f"- Original Stage3 epoch9-step13300 long-horizon target delta "
        f"(`{original_final:.4f}`): `{_format_float(latest['pdms_mean'] - original_final)}`"
    )
    lines.append(
        f"- Safe DiffGRPO best delta (`{safe_diffgrpo_best:.6f}`): "
        f"`{_format_float(latest['pdms_mean'] - safe_diffgrpo_best)}`"
    )
    lines.append("")

    lines.append("## Checkpoint Trend")
    lines.append("")
    header = [
        "checkpoint",
        "evals",
        "PDMS",
        "delta",
        "NC",
        "DAC",
        "TTC",
        "EP",
        "comfort",
        "DDC",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|")
    prev_pdms = None
    for _, row in grouped.sort_values(["sort_step", "sort_epoch", "checkpoint_id"]).iterrows():
        delta = "" if prev_pdms is None else _format_float(row["pdms_mean"] - prev_pdms)
        prev_pdms = row["pdms_mean"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['checkpoint_id']}`",
                    str(int(row["eval_count"])),
                    _format_float(row["pdms_mean"]),
                    delta,
                    _format_float(row.get("nc_mean")),
                    _format_float(row.get("dac_mean")),
                    _format_float(row.get("ttc_mean")),
                    _format_float(row.get("ep_mean")),
                    _format_float(row.get("comfort_mean")),
                    _format_float(row.get("ddc_mean")),
                ]
            )
            + " |"
        )
    lines.append("")

    if len(grouped) >= 2:
        first = grouped.sort_values(["sort_step", "sort_epoch", "checkpoint_id"]).iloc[0]
        last = grouped.sort_values(["sort_step", "sort_epoch", "checkpoint_id"]).iloc[-1]
        lines.append("## First To Latest Delta")
        lines.append("")
        for metric in ("pdms", "nc", "dac", "ttc", "ep", "comfort", "ddc"):
            column = f"{metric}_mean"
            if column in grouped.columns:
                lines.append(f"- {metric.upper()}: `{_format_float(last[column] - first[column])}`")
        lines.append("")

    duplicate_rows = all_rows.groupby("checkpoint_id").size()
    duplicate_rows = duplicate_rows[duplicate_rows > 1]
    if not duplicate_rows.empty:
        lines.append("## Duplicate Eval Rows")
        lines.append("")
        lines.append("Grouped rows above use the mean across duplicate eval summaries for the same checkpoint.")
        for checkpoint_id, count in duplicate_rows.items():
            values = all_rows[all_rows["checkpoint_id"] == checkpoint_id]["pdms_mean"].dropna()
            lines.append(
                f"- `{checkpoint_id}`: {int(count)} rows, PDMS mean `{_format_float(values.mean())}`, "
                f"std `{_format_float(values.std(ddof=0) if len(values) > 1 else 0.0)}`"
            )
        lines.append("")

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(
    summary_files: Iterable[Path],
    *,
    output_tsv: Path,
    output_md: Path,
    early_gate: float,
    original_final: float,
    safe_diffgrpo_best: float,
) -> pd.DataFrame:
    frames = [_read_summary(path) for path in summary_files if path.is_file()]
    if frames:
        all_rows = pd.concat(frames, ignore_index=True)
    else:
        all_rows = pd.DataFrame(columns=["checkpoint_id"])
    if all_rows.empty:
        grouped = pd.DataFrame()
    else:
        grouped_data = []
        for checkpoint_id, rows in all_rows.groupby("checkpoint_id", sort=False):
            item = {
                "checkpoint_id": checkpoint_id,
                "eval_count": len(rows),
                "num_valid_rows": rows.get("num_valid_rows", pd.Series(dtype=float)).max(),
            }
            for metric in METRICS:
                column = f"{metric}_mean"
                if column in rows.columns:
                    item[column] = rows[column].mean()
                    item[f"{metric}_std_across_evals"] = rows[column].std(ddof=0) if len(rows[column].dropna()) > 1 else 0.0
            step, epoch, _ = _checkpoint_sort_key(str(checkpoint_id))
            item["sort_step"] = step
            item["sort_epoch"] = epoch
            item["early_gate_delta"] = item.get("pdms_mean", float("nan")) - early_gate
            item["original_final_delta"] = item.get("pdms_mean", float("nan")) - original_final
            item["safe_diffgrpo_best_delta"] = item.get("pdms_mean", float("nan")) - safe_diffgrpo_best
            grouped_data.append(item)
        grouped = pd.DataFrame(grouped_data).sort_values(["sort_step", "sort_epoch", "checkpoint_id"])

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_tsv, sep="\t", index=False)
    _write_markdown(
        grouped,
        all_rows,
        output_md,
        early_gate=early_gate,
        original_final=original_final,
        safe_diffgrpo_best=safe_diffgrpo_best,
    )
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-tsv", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--early-gate", type=float, default=0.88)
    parser.add_argument(
        "--original-final",
        type=float,
        default=0.9055,
        help="Long-horizon original Stage3 reference, obtained at epoch9-step13300.",
    )
    parser.add_argument(
        "--safe-diffgrpo-best",
        type=float,
        default=0.906184,
        help="Long-horizon Safe DiffGRPO reference, obtained at epoch_12-step_17290.",
    )
    args = parser.parse_args()

    files = _summary_files(args.run_root)
    grouped = analyze(
        files,
        output_tsv=args.output_tsv,
        output_md=args.output_md,
        early_gate=args.early_gate,
        original_final=args.original_final,
        safe_diffgrpo_best=args.safe_diffgrpo_best,
    )
    if grouped.empty:
        print("No navtest summary rows found.")
    else:
        best = grouped.sort_values("pdms_mean", ascending=False).iloc[0]
        latest = grouped.sort_values(["sort_step", "sort_epoch"]).iloc[-1]
        print(
            f"best={best['checkpoint_id']} pdms={best['pdms_mean']:.6f} "
            f"latest={latest['checkpoint_id']} pdms={latest['pdms_mean']:.6f}"
        )


if __name__ == "__main__":
    main()
