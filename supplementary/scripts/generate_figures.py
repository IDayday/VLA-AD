#!/usr/bin/env python3
"""Generate 300-dpi PNG and vector PDF figures from declarative specifications.

No figure is emitted when its source records are absent.  Instead, the command
writes an explicit generation report, preventing a polished empty chart from
being confused with experimental evidence.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_common import (
    ANALYSIS_SEED,
    atomic_write_json,
    atomic_write_text,
    build_provenance,
    configure_logging,
    load_json,
    markdown_table,
    read_table,
    relative_display,
    repository_root,
    supplementary_root,
)


OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # bluish green
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]


def parse_args() -> argparse.Namespace:
    supp = supplementary_root()
    parser = argparse.ArgumentParser(
        description="Generate colorblind-friendly PDF/PNG appendix figures from verified data."
    )
    parser.add_argument(
        "--figure-config", type=Path, default=supp / "configs/figure_generation.json"
    )
    parser.add_argument(
        "--default-input",
        type=Path,
        default=supp / "derived/canonical_scene_metrics.parquet",
    )
    parser.add_argument(
        "--validation-status",
        type=Path,
        default=supp / "derived/data_validation_status.json",
    )
    parser.add_argument("--output-dir", type=Path, default=supp / "figures/generated")
    parser.add_argument(
        "--report", type=Path, default=supp / "derived/figure_generation_report.md"
    )
    parser.add_argument(
        "--provenance", type=Path, default=supp / "derived/figure_generation_provenance.json"
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--seed", type=int, default=ANALYSIS_SEED)
    parser.add_argument(
        "--strict-empty", action="store_true", help="Fail if a configured figure has no records."
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def selector_mask(frame: pd.DataFrame, selector: dict[str, Any]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, expected in selector.items():
        if column not in frame:
            raise ValueError(f"figure filter references absent column {column!r}")
        if isinstance(expected, list):
            mask &= frame[column].isin(expected)
        elif expected is None:
            mask &= frame[column].isna()
        else:
            mask &= frame[column].astype(str).eq(str(expected))
    return mask


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column and column not in frame]
    if missing:
        raise ValueError(f"figure {name!r} requires absent columns: {missing}")


def configure_style(config: dict[str, Any]) -> None:
    plt.rcParams.update(
        {
            "font.family": config.get("font_family", "DejaVu Serif"),
            "font.size": float(config.get("font_size", 8.0)),
            "axes.labelsize": float(config.get("label_size", 8.0)),
            "axes.titlesize": float(config.get("title_size", 8.5)),
            "legend.fontsize": float(config.get("legend_size", 7.0)),
            "xtick.labelsize": float(config.get("tick_size", 7.0)),
            "ytick.labelsize": float(config.get("tick_size", 7.0)),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def group_values(frame: pd.DataFrame, group: str | None) -> list[tuple[str, pd.DataFrame]]:
    if not group:
        return [("", frame)]
    return [(str(value), subset) for value, subset in frame.groupby(group, dropna=False, sort=True)]


def numeric(frame: pd.DataFrame, column: str, name: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    invalid = frame[column].notna() & values.isna()
    if invalid.any():
        raise ValueError(f"figure {name!r}: column {column!r} contains non-numeric values")
    return values


def plot_scatter(ax: plt.Axes, frame: pd.DataFrame, spec: dict[str, Any]) -> None:
    name = str(spec["name"])
    x, y = str(spec["x"]), str(spec["y"])
    color = spec.get("color")
    group = spec.get("group")
    label_column = spec.get("annotation")
    require_columns(frame, [x, y, color, group, label_column], name)
    x_values, y_values = numeric(frame, x, name), numeric(frame, y, name)
    valid = x_values.notna() & y_values.notna()
    frame = frame.loc[valid].copy()
    frame["__x"] = x_values[valid]
    frame["__y"] = y_values[valid]
    if color:
        colors = numeric(frame, str(color), name)
        scatter = ax.scatter(
            frame["__x"],
            frame["__y"],
            c=colors,
            cmap=spec.get("cmap", "viridis"),
            s=float(spec.get("marker_size", 34)),
            edgecolor="white",
            linewidth=0.45,
        )
        colorbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        colorbar.set_label(spec.get("color_label", str(color)))
    else:
        for index, (label, subset) in enumerate(group_values(frame, str(group) if group else None)):
            ax.scatter(
                subset["__x"],
                subset["__y"],
                label=label or None,
                color=OKABE_ITO[index % len(OKABE_ITO)],
                s=float(spec.get("marker_size", 34)),
                edgecolor="white",
                linewidth=0.45,
            )
    if label_column:
        for _, row in frame.iterrows():
            ax.annotate(
                str(row[label_column]),
                (row["__x"], row["__y"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=float(spec.get("annotation_size", 6.5)),
            )
    if group and not color:
        ax.legend(frameon=False)


def aggregate_line(frame: pd.DataFrame, x: str, y: str, group: str | None, aggregation: str | None) -> pd.DataFrame:
    keys = [x] + ([group] if group else [])
    if frame.duplicated(keys).any():
        if aggregation not in {"mean", "median"}:
            raise ValueError(
                "line figure has repeated x/group records; set aggregation to mean or median explicitly"
            )
        return frame.groupby(keys, dropna=False, as_index=False)[y].agg(aggregation)
    return frame


def plot_line(ax: plt.Axes, frame: pd.DataFrame, spec: dict[str, Any]) -> None:
    name = str(spec["name"])
    x, y = str(spec["x"]), str(spec["y"])
    group = str(spec["group"]) if spec.get("group") else None
    require_columns(frame, [x, y, group], name)
    frame = frame.copy()
    frame[x], frame[y] = numeric(frame, x, name), numeric(frame, y, name)
    frame = frame.dropna(subset=[x, y])
    frame = aggregate_line(frame, x, y, group, spec.get("aggregation"))
    for index, (label, subset) in enumerate(group_values(frame, group)):
        subset = subset.sort_values(x)
        ax.plot(
            subset[x],
            subset[y],
            marker=spec.get("marker", "o"),
            markersize=float(spec.get("marker_size", 4)),
            linewidth=float(spec.get("line_width", 1.5)),
            color=OKABE_ITO[index % len(OKABE_ITO)],
            label=label or None,
        )
    if group:
        ax.legend(frameon=False)


def plot_bar(ax: plt.Axes, frame: pd.DataFrame, spec: dict[str, Any]) -> None:
    name = str(spec["name"])
    x, y = str(spec["x"]), str(spec["y"])
    require_columns(frame, [x, y], name)
    values = numeric(frame, y, name)
    valid = values.notna()
    labels = frame.loc[valid, x].astype(str).tolist()
    ax.bar(
        np.arange(len(labels)),
        values[valid],
        color=[OKABE_ITO[index % len(OKABE_ITO)] for index in range(len(labels))],
        width=0.72,
    )
    ax.set_xticks(np.arange(len(labels)), labels, rotation=float(spec.get("rotation", 0)))


def plot_histogram(ax: plt.Axes, frame: pd.DataFrame, spec: dict[str, Any]) -> None:
    name = str(spec["name"])
    x = str(spec["x"])
    group = str(spec["group"]) if spec.get("group") else None
    require_columns(frame, [x, group], name)
    for index, (label, subset) in enumerate(group_values(frame, group)):
        values = numeric(subset, x, name).dropna()
        ax.hist(
            values,
            bins=int(spec.get("bins", 10)),
            color=OKABE_ITO[index % len(OKABE_ITO)],
            alpha=0.75 if group else 0.9,
            label=label or None,
            edgecolor="white",
            linewidth=0.5,
        )
    if group:
        ax.legend(frameon=False)


def plot_stacked_fraction(ax: plt.Axes, frame: pd.DataFrame, spec: dict[str, Any]) -> None:
    name = str(spec["name"])
    x, category = str(spec["x"]), str(spec["category"])
    require_columns(frame, [x, category], name)
    counts = frame.groupby([x, category], dropna=False).size().unstack(fill_value=0).sort_index()
    fractions = counts.div(counts.sum(axis=1), axis=0)
    bottom = np.zeros(len(fractions))
    positions = np.arange(len(fractions))
    for index, column in enumerate(fractions.columns):
        values = fractions[column].to_numpy(float)
        ax.bar(
            positions,
            values,
            bottom=bottom,
            label=str(column),
            color=OKABE_ITO[index % len(OKABE_ITO)],
            width=0.82,
        )
        bottom += values
    ax.set_xticks(positions, [str(value) for value in fractions.index])
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)


def plot_transition_matrix(ax: plt.Axes, frame: pd.DataFrame, spec: dict[str, Any]) -> None:
    name = str(spec["name"])
    require_columns(frame, ["initial_failure", "recovered", "new_failure"], name)
    evaluable = frame.dropna(subset=["initial_failure", "recovered", "new_failure"])
    initial_failure = evaluable["initial_failure"].astype(bool)
    final_failure = (initial_failure & ~evaluable["recovered"].astype(bool)) | (
        ~initial_failure & evaluable["new_failure"].astype(bool)
    )
    matrix = np.array(
        [
            [int((initial_failure & final_failure).sum()), int((initial_failure & ~final_failure).sum())],
            [int((~initial_failure & final_failure).sum()), int((~initial_failure & ~final_failure).sum())],
        ]
    )
    image = ax.imshow(matrix, cmap=spec.get("cmap", "Blues"), aspect="auto")
    plt.colorbar(image, ax=ax, pad=0.02, label="Scenes")
    for row in range(2):
        for column in range(2):
            threshold = matrix.max() / 2 if matrix.size else 0
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    ax.set_xticks([0, 1], ["Final failure", "Final success"])
    ax.set_yticks([0, 1], ["Initial failure", "Initial success"])


PLOTTERS = {
    "scatter": plot_scatter,
    "line": plot_line,
    "bar": plot_bar,
    "histogram": plot_histogram,
    "stacked_fraction": plot_stacked_fraction,
    "transition_matrix": plot_transition_matrix,
}


def resolve_input(value: str | None, default_input: Path) -> Path:
    if value is None:
        return default_input
    path = Path(value)
    return path if path.is_absolute() else repository_root() / path


def save_figure(fig: plt.Figure, output_dir: Path, name: str, dpi: int) -> list[Path]:
    if not name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in name):
        raise ValueError(f"unsafe figure name {name!r}; use letters, digits, underscore, or hyphen")
    pdf = output_dir / f"{name}.pdf"
    png = output_dir / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    return [pdf, png]


def render_report(records: list[dict[str, Any]], configured: int, seed: int, dpi: int) -> str:
    completed = sum(record["status"] == "generated" for record in records)
    lines = [
        "# Figure Generation Report",
        "",
        f"- Configured figures: {configured}",
        f"- Generated figures: {completed}",
        f"- PNG resolution: {dpi} dpi",
        f"- Analysis seed (plotting only): {seed}",
        "",
    ]
    if configured == 0:
        lines.append(
            "No evidence-backed figure specification is configured. No placeholder image was produced."
        )
    elif records:
        lines.extend(
            [
                markdown_table(
                    ["Figure", "Type", "Input", "Rows", "Status", "Note"],
                    [
                        (
                            record["name"],
                            record["type"],
                            record["input"],
                            record["rows"],
                            record["status"],
                            record.get("note", ""),
                        )
                        for record in records
                    ],
                ),
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    if args.seed < 0 or args.dpi < 72:
        raise ValueError("seed must be non-negative and dpi must be at least 72")
    validation = load_json(args.validation_status)
    if validation.get("status") == "invalid":
        raise RuntimeError("refusing to generate figures from invalid canonical data")
    config = load_json(args.figure_config)
    specs = config.get("figures", [])
    if not isinstance(specs, list):
        raise ValueError("figure config 'figures' must be a list")
    configure_style(config.get("style", {}))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    input_paths: set[Path] = {args.figure_config, args.validation_status}
    outputs: list[Path] = []
    for index, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise ValueError(f"figure specification {index} must be an object")
        name = str(spec.get("name", ""))
        plot_type = str(spec.get("type", ""))
        if plot_type not in PLOTTERS:
            raise ValueError(f"figure {name!r} has unsupported type {plot_type!r}")
        source = resolve_input(spec.get("input"), args.default_input)
        input_paths.add(source)
        if not source.exists():
            note = "source file is absent"
            records.append(
                {
                    "name": name,
                    "type": plot_type,
                    "input": relative_display(source, repository_root()),
                    "rows": 0,
                    "status": "pending",
                    "note": note,
                }
            )
            if args.strict_empty:
                raise FileNotFoundError(f"figure {name!r}: {note}: {source}")
            continue
        frame = read_table(source)
        frame = frame.loc[selector_mask(frame, spec.get("filters", {}))].copy()
        if frame.empty:
            note = "no verified records matched the figure filters"
            records.append(
                {
                    "name": name,
                    "type": plot_type,
                    "input": relative_display(source, repository_root()),
                    "rows": 0,
                    "status": "pending",
                    "note": note,
                }
            )
            if args.strict_empty:
                raise RuntimeError(f"figure {name!r}: {note}")
            continue

        width = float(spec.get("width_inches", 3.35))
        height = float(spec.get("height_inches", 2.35))
        if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
            raise ValueError(f"figure {name!r} has invalid dimensions")
        fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
        PLOTTERS[plot_type](ax, frame, spec)
        ax.set_xlabel(spec.get("xlabel", spec.get("x", "")))
        ax.set_ylabel(spec.get("ylabel", spec.get("y", "")))
        title = spec.get("title")
        if title:
            ax.set_title(title)
        if spec.get("grid", True):
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
            ax.set_axisbelow(True)
        figure_outputs = save_figure(fig, args.output_dir, name, args.dpi)
        plt.close(fig)
        outputs.extend(figure_outputs)
        records.append(
            {
                "name": name,
                "type": plot_type,
                "input": relative_display(source, repository_root()),
                "rows": int(len(frame)),
                "status": "generated",
                "note": "PDF and 300-dpi PNG share the same source records.",
            }
        )
        logging.info("generated figure %s from %d rows", name, len(frame))

    atomic_write_text(args.report, render_report(records, len(specs), args.seed, args.dpi))
    outputs.append(args.report)
    existing_inputs = [path for path in input_paths if path.exists()]
    status = "complete" if specs and all(record["status"] == "generated" for record in records) else "pending"
    provenance = build_provenance(
        seed=args.seed,
        inputs=existing_inputs,
        outputs=outputs,
        status=status,
        notes=[record.get("note", "") for record in records if record["status"] != "generated"],
    )
    atomic_write_json(args.provenance, provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
