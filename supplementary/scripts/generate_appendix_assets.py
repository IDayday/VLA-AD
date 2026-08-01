#!/usr/bin/env python3
"""Generate evidence-linked appendix tables and publication figures.

Paper-reported aggregate CSVs are treated as claims, canonical scene exports as
reproduced evidence, and reproduction-only settings as proposals.  The three
identities are never pooled to estimate uncertainty.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_common import (
    ANALYSIS_SEED,
    atomic_write_json,
    atomic_write_text,
    repository_root,
    sha256_file,
    supplementary_root,
)


PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]


def parse_args() -> argparse.Namespace:
    root = repository_root()
    supp = supplementary_root()
    parser = argparse.ArgumentParser(
        description="Generate CSV/LaTeX appendix tables and PDF/300-dpi PNG figures from audited evidence."
    )
    parser.add_argument("--source-dir", type=Path, default=supp / "tables/sources")
    parser.add_argument("--canonical", type=Path, default=supp / "derived/canonical_scene_metrics.parquet")
    parser.add_argument("--stats", type=Path, default=supp / "derived/stats.json")
    parser.add_argument(
        "--hard658", type=Path,
        default=root / "outputs/navtest_hard_stable_zero658_recogdrive_stage3_vs_pdms914505_20260727/per_token_comparison.csv",
    )
    parser.add_argument(
        "--qualitative-manifest", type=Path,
        default=root / "outputs/ampt9145_vs_recogdrive_rl_ddv2_navtest_cases_20260728/visualizations/manifest.csv",
    )
    parser.add_argument("--table-dir", type=Path, default=supp / "tables/generated")
    parser.add_argument("--figure-dir", type=Path, default=supp / "figures/generated")
    parser.add_argument("--manifest", type=Path, default=supp / "derived/all_appendix_tables_manifest.json")
    parser.add_argument("--seed", type=int, default=ANALYSIS_SEED, help="Plot/bootstrap analysis seed only.")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def latex_escape(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "not recorded"
    text = str(value)
    for source, target in (
        ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
        ("_", r"\_"), ("#", r"\#"), ("$", r"\$"),
        ("{", r"\{"), ("}", r"\}"),
    ):
        text = text.replace(source, target)
    return text


def frame_to_tex(
    frame: pd.DataFrame, *, caption: str, label: str, wide: bool = False,
    headers: dict[str, str] | None = None, align: str | None = None,
) -> str:
    display = frame.rename(columns=headers or {})
    environment = "table*" if wide else "table"
    width = r"\textwidth" if wide else r"\linewidth"
    align = align or ("l" + "c" * (len(display.columns) - 1))
    rows = []
    for record in display.itertuples(index=False, name=None):
        rows.append(" & ".join(latex_escape(value) for value in record) + r" \\")
    return "\n".join(
        [
            rf"\begin{{{environment}}}[t]", r"\centering", r"\small",
            rf"\caption{{{caption}}}", rf"\label{{{label}}}",
            rf"\resizebox{{{width}}}{{!}}{{%", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
            " & ".join(latex_escape(column) for column in display.columns) + r" \\", r"\midrule",
            *rows, r"\bottomrule", r"\end{tabular}%", r"}", rf"\end{{{environment}}}", "",
        ]
    )


def write_table(
    frame: pd.DataFrame, stem: str, table_dir: Path, *, caption: str,
    label: str, wide: bool = False, headers: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / f"{stem}.csv"
    tex_path = table_dir / f"{stem}.tex"
    frame.to_csv(csv_path, index=False)
    atomic_write_text(tex_path, frame_to_tex(frame, caption=caption, label=label, wide=wide, headers=headers))
    return csv_path, tex_path


def wilson(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - radius, center + radius


def validate_reported_sources(source_dir: Path) -> None:
    mismatch = pd.read_csv(source_dir / "supervision_mismatch_reported.csv")
    computed = mismatch["post_scalar_grpo_pdms"] - mismatch["il_pdms"]
    if not np.allclose(computed, mismatch["grpo_change_points"], atol=1e-9):
        raise ValueError("supervision mismatch deltas do not equal post-GRPO minus IL")
    rounds = pd.read_csv(source_dir / "apr_rounds_reported.csv")
    expected = rounds["pdms"].diff().fillna(0)
    if not np.allclose(expected, rounds["gain_points"], atol=1e-9):
        raise ValueError("APR gains do not equal consecutive reported scores")


def hard658_tables(hard: pd.DataFrame, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    before = pd.to_numeric(hard["orig_PDMS"], errors="raise").to_numpy()
    after = pd.to_numeric(hard["old_v2_PDMS"], errors="raise").to_numpy()
    before_positive, after_positive = before > 1e-12, after > 1e-12
    matrix = np.array(
        [
            [np.sum(~before_positive & ~after_positive), np.sum(~before_positive & after_positive)],
            [np.sum(before_positive & ~after_positive), np.sum(before_positive & after_positive)],
        ], dtype=int,
    )
    expected = np.array([[198, 93], [20, 347]])
    if not np.array_equal(matrix, expected):
        raise ValueError(f"paper hard-658 transition mismatch: {matrix.tolist()}")
    transition = pd.DataFrame(
        [
            {"Scalar-GRPO state": "zero", "AMPT zero": 198, "AMPT positive": 93, "row total": 291},
            {"Scalar-GRPO state": "positive", "AMPT zero": 20, "AMPT positive": 347, "row total": 367},
        ]
    )
    scalar_count, ampt_count = int(before_positive.sum()), int(after_positive.sum())
    recovery_rows = []
    for method, count in (("Scalar GRPO", scalar_count), ("AMPT recovery checkpoint", ampt_count)):
        low, high = wilson(count, len(hard))
        recovery_rows.append(
            {"Method": method, "Positive": count, "Zero": len(hard) - count,
             "Recovery rate (%)": f"{100 * count / len(hard):.1f}",
             "Wilson 95% CI (%)": f"[{100 * low:.1f}, {100 * high:.1f}]",
             "Runs": "1 archived comparison", "Scenes": len(hard)}
        )
    recovery = pd.DataFrame(recovery_rows)

    delta = after_positive.astype(int) - before_positive.astype(int)
    draws = np.empty(10_000)
    for index in range(len(draws)):
        draws[index] = rng.choice(delta, size=len(delta), replace=True).mean()
    gain_ci = np.quantile(draws, [0.025, 0.975])
    recovery_gain = pd.DataFrame(
        [{"Absolute gain (pp)": f"{100 * delta.mean():.1f}", "Paired 95% CI (pp)": f"[{100 * gain_ci[0]:.1f}, {100 * gain_ci[1]:.1f}]",
          "Zero-to-positive": 93, "Positive-to-zero": 20, "Net repair": 73, "Scenes": len(hard), "Runs": "1 archived comparison"}]
    )

    cause_rows = []
    mappings = [
        ("DAC_all_fail", "DAC-only/all-fail"),
        ("NC_all_fail", "collision-only/all-fail"),
        ("NC_and_DAC_all_fail", "collision + DAC"),
    ]
    for raw_name, label in mappings:
        subset = hard.loc[hard["root_cause_group"].eq(raw_name)]
        b = subset["orig_PDMS"].to_numpy() > 1e-12
        a = subset["old_v2_PDMS"].to_numpy() > 1e-12
        repairs = int(np.sum(~b & a))
        regressions = int(np.sum(b & ~a))
        cause_rows.append(
            {"Failure type": label, "Scenes": len(subset), "Repairs": repairs,
             "Regressions": regressions, "Net repair": repairs - regressions}
        )
    cause = pd.DataFrame(cause_rows)
    if cause[["Scenes", "Repairs", "Regressions", "Net repair"]].sum().tolist() != [658, 93, 20, 73]:
        raise ValueError("failure-cause totals do not close to the 658-scene transition")
    return {"transition": transition, "recovery": recovery, "recovery_gain": recovery_gain, "cause": cause}


def failure_case_table(hard: pd.DataFrame, qualitative_manifest: Path) -> pd.DataFrame:
    before = hard["orig_PDMS"].gt(1e-12)
    after = hard["old_v2_PDMS"].gt(1e-12)
    hard = hard.copy()
    hard["transition"] = np.select(
        [~before & after, before & ~after, ~before & ~after],
        ["repaired", "regressed", "persistent failure"], default="retained positive",
    )
    desired = ["repaired", "repaired", "regressed", "persistent failure", "retained positive"]
    chosen = []
    used: set[str] = set()
    for kind in desired:
        candidates = hard.loc[hard["transition"].eq(kind)].sort_values(["root_cause_group", "token"])
        row = next((record for _, record in candidates.iterrows() if str(record["token"]) not in used), None)
        if row is None:
            raise ValueError(f"no hard-658 case available for {kind}")
        used.add(str(row["token"]))
        chosen.append(row)
    command_map: dict[str, str] = {}
    if qualitative_manifest.exists():
        manifest = pd.read_csv(qualitative_manifest)
        command_map = dict(zip(manifest["token"].astype(str), manifest["command"].astype(str)))
    rows = []
    for row in chosen:
        token = str(row["token"])
        rows.append(
            {"Scene ID": token, "Command": command_map.get(token, "not archived"),
             "Failure type": str(row["root_cause_group"]).replace("_", " "),
             "Scalar PDMS": f"{100 * float(row['orig_PDMS']):.1f}",
             "AMPT PDMS": f"{100 * float(row['old_v2_PDMS']):.1f}",
             "Outcome": str(row["transition"]),
             "Diagnosis": "hard feasibility remains limiting" if row["transition"] == "persistent failure" else "credit assignment changes feasibility state"}
        )
    return pd.DataFrame(rows)


def reproduced_results(canonical: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("NAVSIM v1", "paper_final", ["NC", "DAC", "TTC", "C", "EP", "aggregate_score"], 12138),
        ("NAVSIM v2", "paper_final", ["NC", "DAC", "DDC", "TLC", "TTC", "EP", "LK", "HC", "EC", "aggregate_score"], 12146),
    ]
    rows = []
    for benchmark, variant, metrics, expected in specs:
        subset = canonical.loc[
            canonical["benchmark"].eq(benchmark) & canonical["method"].eq("AMPT")
            & canonical["stage"].eq("APR") & canonical["variant"].eq(variant)
        ]
        if subset["scene_id"].nunique() != expected:
            raise ValueError(f"{benchmark}: expected {expected} final scenes")
        row: dict[str, Any] = {"Benchmark": benchmark, "Scenes": expected, "Inference": "1 trajectory", "Runs": "1 inference seed"}
        for metric in metrics:
            label = "PDMS" if benchmark.endswith("v1") and metric == "aggregate_score" else ("EPDMS" if metric == "aggregate_score" else metric)
            row[label] = f"{100 * pd.to_numeric(subset[metric], errors='coerce').mean():.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, stem: str, directory: Path, dpi: int) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    pdf = directory / f"{stem}.pdf"
    png = directory / f"{stem}.png"
    # Suppress wall-clock metadata so identical inputs produce identical PDFs.
    fig.savefig(pdf, bbox_inches="tight", metadata={"Creator": "AMPT supplementary pipeline", "CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=dpi, bbox_inches="tight", metadata={"Software": "AMPT supplementary pipeline"})
    plt.close(fig)
    return pdf, png


def configure_plots() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def plot_supervision(frame: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    for index, row in frame.iterrows():
        ax.scatter(row["il_pdms"], row["grpo_change_points"], s=45, color=PALETTE[index], edgecolor="white", zorder=3)
        ax.annotate(row["method"], (row["il_pdms"], row["grpo_change_points"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.axhline(0, color="#666666", linewidth=.8, linestyle="--")
    ax.set_xlabel("Imitation-learning PDMS")
    ax.set_ylabel("Change after scalar GRPO (points)")
    ax.grid(alpha=.2)
    fig.tight_layout()
    return fig


def plot_apr(frame: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    ax.plot(frame["round"], frame["pdms"], marker="o", color=PALETTE[0], linewidth=1.7)
    for _, row in frame.iterrows():
        ax.annotate(f"{row['pdms']:.2f}", (row["round"], row["pdms"]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)
    ax.set_xticks(frame["round"])
    ax.set_xlabel("APR round")
    ax.set_ylabel("PDMS")
    ax.set_ylim(frame["pdms"].min() - .08, frame["pdms"].max() + .10)
    ax.grid(alpha=.2)
    fig.tight_layout()
    return fig


def plot_transition() -> plt.Figure:
    values = np.array([[198, 93], [20, 347]])
    fig, ax = plt.subplots(figsize=(3.1, 2.6))
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=values.max())
    for (row, column), value in np.ndenumerate(values):
        ax.text(column, row, str(value), ha="center", va="center", color="white" if value > 220 else "black", fontsize=10)
    ax.set_xticks([0, 1], ["zero", "positive"])
    ax.set_yticks([0, 1], ["zero", "positive"])
    ax.set_xlabel("AMPT recovery checkpoint")
    ax.set_ylabel("Scalar GRPO")
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04, label="Scenes")
    fig.tight_layout()
    return fig


def plot_failure_causes(cause: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.0, 2.55))
    x = np.arange(len(cause))
    ax.bar(x - .18, cause["Repairs"], width=.36, color=PALETTE[2], label="zero to positive")
    ax.bar(x + .18, cause["Regressions"], width=.36, color=PALETTE[3], label="positive to zero")
    ax.set_xticks(x, ["DAC", "collision", "both"])
    ax.set_ylabel("Scenes")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    return fig


def plot_metric_profiles(canonical: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.55))
    specifications = [
        ("NAVSIM v1", ["NC", "DAC", "TTC", "C", "EP", "aggregate_score"], "PDMS"),
        ("NAVSIM v2", ["NC", "DAC", "DDC", "TLC", "TTC", "EP", "LK", "HC", "EC", "aggregate_score"], "EPDMS"),
    ]
    for ax, (benchmark, metrics, aggregate_name) in zip(axes, specifications):
        subset = canonical.loc[(canonical["benchmark"] == benchmark) & (canonical["method"] == "AMPT") & (canonical["stage"] == "APR") & (canonical["variant"] == "paper_final")]
        values = [100 * pd.to_numeric(subset[metric], errors="coerce").mean() for metric in metrics]
        labels = [aggregate_name if metric == "aggregate_score" else metric for metric in metrics]
        colors = [PALETTE[0]] * (len(values) - 1) + [PALETTE[1]]
        ax.bar(np.arange(len(values)), values, color=colors)
        ax.set_xticks(np.arange(len(values)), labels, rotation=45 if len(values) > 7 else 0, ha="right" if len(values) > 7 else "center")
        ax.set_ylim(80, 101)
        ax.set_ylabel("Score (%)")
        ax.set_title(benchmark)
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    return fig


def qualitative_contact_sheet(path: Path) -> plt.Figure | None:
    if not path.exists():
        return None
    manifest = pd.read_csv(path)
    selected = manifest.sort_values(["command", "case_rank"]).groupby("command", as_index=False).first()
    order = {"left": 0, "straight": 1, "right": 2}
    selected = selected.assign(order=selected["command"].map(order)).sort_values("order")
    if len(selected) != 3:
        return None
    fig, axes = plt.subplots(3, 1, figsize=(6.9, 8.9))
    for ax, (_, row) in zip(axes, selected.iterrows()):
        image_path = Path(str(row["front_bev_path"]))
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        ax.imshow(plt.imread(image_path))
        ax.axis("off")
    fig.tight_layout(pad=.2)
    return fig


def main() -> int:
    args = parse_args()
    if args.seed < 0 or args.dpi < 72:
        raise ValueError("--seed must be non-negative and --dpi at least 72")
    for path in (args.source_dir, args.canonical, args.stats, args.hard658):
        if not path.exists():
            raise FileNotFoundError(path)
    validate_reported_sources(args.source_dir)
    canonical = pd.read_parquet(args.canonical)
    hard = pd.read_csv(args.hard658)
    rng = np.random.default_rng(args.seed)
    hard_tables = hard658_tables(hard, rng)
    outputs: list[Path] = []

    source_specs = {
        "candidate_source_statistics": ("Candidate-source and audit budgets for PC-MTS on the NAVSIM training universe. Counts tagged proposed are reproduction settings rather than observations; higher/lower is not applicable.", "tab:candidate_sources", True),
        "metric_partition": ("NAVSIM v1/v2 metric partition used by PC-MTS and FF-PGRPO. Hard feasibility is tested before protected non-regression and Pareto quality.", "tab:metric_partition", False),
        "supervision_mismatch_reported": ("Supervision--optimization comparison on NAVSIM v1 under single-trajectory inference. All four rows are one paper-reported run; higher PDMS and GRPO change are better, and no variance is implied.", "tab:supervision_mismatch", False),
        "stage_ablation_reported": ("Stage-wise NAVSIM v1 ablation under single-trajectory inference. Higher is better. Rows are one paper-reported run; only the final AMPT row is independently reproduced from 12,138 scene scores.", "tab:stage_ablation", True),
        "apr_rounds_reported": ("Paper-reported NAVSIM v1 APR progression under single-trajectory inference. Higher is better. These are one-run round summaries without a training-seed confidence interval.", "tab:apr_rounds", False),
        "stage_hyperparameters": ("Stage-wise training protocol. Reported/reproduced settings are distinguished from bounded reproduction settings; `not recorded' is an explicit provenance value, not a blank. Inference always emits one trajectory.", "tab:stage_hyperparameters", True),
        "baseline_protocol": ("Fair-comparison protocol for main-paper baselines. All methods use one inference candidate and no test-time scorer in the paper table; unreported backbone/sensor/data details are not inferred. Result identity distinguishes reported from reproduced.", "tab:baseline_protocol", True),
        "apr_teacher_pool_audited": ("Audited APR teacher-pool construction on NAVSIM navtrain. Counts are from one archived teacher audit; the 28,910 accepted teachers follow interpolation and exact re-evaluation. Higher acceptance is not inherently better because guards constrain quality.", "tab:apr_teacher_pool", False),
        "apr_fixed_teacher_diagnostic": ("APR fixed-teacher diagnostic on NAVSIM v1. The archived fixed-set continuation decreases from its 91.2766 anchor, whereas the 91.45 endpoint is the paper's compressed dynamic-round report; the rows are diagnostic rather than a step-matched causal ablation.", "tab:apr_fixed_teacher", False),
    }
    for stem, (caption, label, wide) in source_specs.items():
        source = args.source_dir / f"{stem}.csv"
        frame = pd.read_csv(source, keep_default_na=False)
        outputs.extend(write_table(frame, stem.replace("_reported", ""), args.table_dir, caption=caption, label=label, wide=wide))

    outputs.extend(write_table(
        hard_tables["transition"], "failure_transition", args.table_dir,
        caption="Two-by-two transition on the fixed 658 NAVSIM v1 initial-failure scenes. The paper-aligned archived pair compares scalar GRPO with the AMPT recovery checkpoint; counts sum to 658. Positive PDMS is better.",
        label="tab:failure_transition"))
    outputs.extend(write_table(
        hard_tables["recovery"], "failure_recovery", args.table_dir,
        caption="Recovery on the same 658 NAVSIM v1 scenes under one archived comparison. Rates use Wilson 95% intervals over scenes; this interval does not represent training-seed uncertainty. Higher recovery is better.",
        label="tab:failure_recovery"))
    outputs.extend(write_table(
        hard_tables["recovery_gain"], "failure_net_gain", args.table_dir,
        caption="Paired recovery change on 658 NAVSIM v1 scenes. The 11.1-point gain has a scene-paired bootstrap interval (10,000 resamples, analysis seed 2027); higher net repair is better. One archived model pair is available.",
        label="tab:failure_net_gain"))
    outputs.extend(write_table(
        hard_tables["cause"], "failure_cause_breakdown", args.table_dir,
        caption="Failure-type decomposition for the fixed 658 NAVSIM v1 scenes. Repairs and regressions use the scalar-GRPO to AMPT transition; net repairs sum to 73. One archived comparison; higher net repair is better.",
        label="tab:failure_causes"))
    cases = failure_case_table(hard, args.qualitative_manifest)
    outputs.extend(write_table(
        cases, "failure_cases", args.table_dir,
        caption="Mechanism-organized examples from the fixed 658-scene set. Scene IDs and PDMS values are reproduced from the archived comparison; command is explicitly marked when not present in the qualitative manifest. These selected cases are not an unbiased benchmark sample.",
        label="tab:failure_cases", wide=True))
    verified = reproduced_results(canonical)
    outputs.extend(write_table(
        verified, "reproduced_main_results", args.table_dir,
        caption="Scene-level reproduction of the AMPT main results under one-trajectory inference without test-time scoring or reranking. All metrics are percentages and higher is better. NAVSIM v1 has 12,138 scored scenes; v2 has 12,146.",
        label="tab:reproduced_main", wide=True))

    configure_plots()
    mismatch = pd.read_csv(args.source_dir / "supervision_mismatch_reported.csv")
    apr = pd.read_csv(args.source_dir / "apr_rounds_reported.csv")
    for stem, fig in (
        ("supervision_mismatch", plot_supervision(mismatch)),
        ("apr_rounds", plot_apr(apr)),
        ("failure_transition", plot_transition()),
        ("failure_cause_recovery", plot_failure_causes(hard_tables["cause"])),
        ("ampt_metric_profiles", plot_metric_profiles(canonical)),
    ):
        outputs.extend(save_figure(fig, stem, args.figure_dir, args.dpi))
    contact = qualitative_contact_sheet(args.qualitative_manifest)
    if contact is not None:
        outputs.extend(save_figure(contact, "qualitative_selected_cases", args.figure_dir, args.dpi))

    payload = {
        "schema_version": 1,
        "status": "complete",
        "analysis_seed": args.seed,
        "analysis_seed_scope": "bootstrap resampling and deterministic plotting only",
        "hard658_policy": "paper-aligned scalar GRPO versus historical AMPT recovery checkpoint; later 91.45 hard-set columns excluded",
        "inputs": [
            {"path": path.relative_to(repository_root()).as_posix(), "sha256": sha256_file(path)}
            for path in [args.canonical, args.stats, args.hard658, *sorted(args.source_dir.glob("*.csv"))]
        ],
        "outputs": [
            {"path": path.relative_to(repository_root()).as_posix(), "sha256": sha256_file(path)} for path in outputs
        ],
    }
    atomic_write_json(args.manifest, payload)
    print(json.dumps({"status": "complete", "tables": len([p for p in outputs if p.suffix == '.csv']), "figures": len([p for p in outputs if p.suffix == '.png'])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
