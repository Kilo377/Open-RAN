"""Single-panel paper figures for Experiment 2.

This script reads the preserved CSV outputs and creates one metric per figure
with minimal in-figure text.  It does not rerun the experiment.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METHOD_ORDER = [
    "schema_only",
    "orchestran_like",
    "oreo_like",
    "pacifista_like",
    "repair_wo_memory",
    "repair_wo_l3",
    "full_repair",
    "conservative_fallback",
]

METHOD_LABELS = {
    "schema_only": "Schema-only",
    "orchestran_like": "OrchestRAN",
    "oreo_like": "OREO",
    "pacifista_like": "PACIFISTA",
    "repair_wo_memory": "Repair w/o memory",
    "repair_wo_l3": "Repair w/o L3",
    "full_repair": "Full repair",
    "conservative_fallback": "Fallback",
}

METHOD_COLORS = {
    "schema_only": "#9AA4B2",
    "orchestran_like": "#D28A22",
    "oreo_like": "#B66AA5",
    "pacifista_like": "#C94C4C",
    "repair_wo_memory": "#5B8CCB",
    "repair_wo_l3": "#7B76CF",
    "full_repair": "#147D64",
    "conservative_fallback": "#6B7280",
}

ABLATION_ORDER = ["repair_wo_memory", "repair_wo_l3", "full_repair"]
ABLATION_LABELS = ["No memory", "L2 scoring", "Full repair"]

OUTCOME_GROUPS = [
    ("executable_without_repair", "Initial executable", "#9BCB73"),
    ("repaired_and_executable", "Repaired executable", "#147D64"),
    ("rejected", "Rejected", "#C94C4C"),
    ("residual_or_invalid", "Residual/invalid", "#9AA4B2"),
]

RESIDUAL_FLAG_COLUMNS = [
    "hard_conflict_escape",
    "resource_overflow_escape",
    "unsupported_parameter_residual",
    "empty_bound_residual",
    "unsafe_bound_residual",
    "policy_violation_residual",
    "contract_inconsistency",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _summary_by_mode(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row.get("mode", "")): row for row in rows}


def _ordered_modes(summary: dict[str, dict[str, str]]) -> list[str]:
    ordered = [mode for mode in METHOD_ORDER if mode in summary]
    ordered.extend(mode for mode in summary if mode not in ordered)
    return ordered


def _percent_axis(ax) -> None:
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
    ax.set_xlabel("Rate (%)")
    ax.grid(axis="x", color="#E6EAF0", linewidth=0.8)


def _save(fig, output_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"bbox_inches": "tight", "facecolor": fig.get_facecolor()}
        if suffix == "png":
            kwargs["dpi"] = 420
        fig.savefig(output_dir / f"{stem}.{suffix}", **kwargs)


def configure_style() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#D8DEE9",
            "axes.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.titlesize": 10.8,
            "axes.labelsize": 9.4,
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "xtick.color": "#475467",
            "ytick.color": "#344054",
            "axes.labelcolor": "#344054",
            "text.color": "#1F2937",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_method_bar(
    summary: dict[str, dict[str, str]],
    metric: str,
    title: str,
    output_dir: Path,
    stem: str,
    *,
    lower_is_better: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    modes = _ordered_modes(summary)
    labels = [METHOD_LABELS.get(mode, mode) for mode in modes]
    values = [_float(summary[mode].get(metric)) for mode in modes]
    colors = [METHOD_COLORS.get(mode, "#6B7280") for mode in modes]
    y_positions = list(range(len(modes)))

    fig, ax = plt.subplots(figsize=(5.7, 3.15))
    bars = ax.barh(y_positions, values, color=colors, edgecolor="white", linewidth=0.7)
    for bar, value, mode in zip(bars, values, modes):
        if mode == "full_repair":
            bar.set_edgecolor("#0F513F")
            bar.set_linewidth(1.5)
        x_text = min(0.985, value + 0.018)
        ax.text(x_text, bar.get_y() + bar.get_height() / 2, f"{100 * value:.1f}", va="center", fontsize=7.8)

    if title:
        ax.set_title(title, loc="left", pad=6)
    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    _percent_axis(ax)
    if lower_is_better:
        ax.text(0.985, -0.65, "lower is better", ha="right", va="center", fontsize=7.5, color="#667085")
    fig.tight_layout(pad=0.8)
    _save(fig, output_dir, stem)
    plt.close(fig)


def _candidate_outcome_groups(case_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in case_rows:
        grouped.setdefault(str(row.get("mode", "")), []).append(row)

    result: dict[str, dict[str, float]] = {}
    for mode, rows in grouped.items():
        counts = {key: 0 for key, _label, _color in OUTCOME_GROUPS}
        for row in rows:
            bucket = str(row.get("outcome_bucket", ""))
            if bucket == "executable_without_repair":
                counts["executable_without_repair"] += 1
            elif bucket == "repaired_and_executable":
                counts["repaired_and_executable"] += 1
            elif bucket.startswith("rejected"):
                counts["rejected"] += 1
            else:
                counts["residual_or_invalid"] += 1
        denom = max(1, len(rows))
        result[mode] = {key: value / denom for key, value in counts.items()}
    return result


def plot_candidate_outcomes(
    summary: dict[str, dict[str, str]],
    case_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    modes = _ordered_modes(summary)
    x_positions = list(range(len(modes)))
    fractions = _candidate_outcome_groups(case_rows)

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    bottom = [0.0 for _mode in modes]
    for group, label, color in OUTCOME_GROUPS:
        values = [fractions.get(mode, {}).get(group, 0.0) for mode in modes]
        ax.bar(x_positions, values, bottom=bottom, label=label, color=color, edgecolor="white", linewidth=0.45)
        bottom = [old + value for old, value in zip(bottom, values)]

    ax.set_ylabel("Fraction (%)")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
    ax.set_xticks(x_positions, [METHOD_LABELS.get(mode, mode) for mode in modes], rotation=28, ha="right")
    ax.grid(axis="y", color="#E6EAF0", linewidth=0.8)
    ax.legend(ncol=2, bbox_to_anchor=(0.5, -0.32), loc="upper center", fontsize=7.6)
    fig.tight_layout(pad=0.8)
    _save(fig, output_dir, "fig2_candidate_outcomes")
    plt.close(fig)


def _any_residual_violation_rates(case_rows: list[dict[str, str]]) -> dict[str, float]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in case_rows:
        grouped.setdefault(str(row.get("mode", "")), []).append(row)

    rates: dict[str, float] = {}
    for mode, rows in grouped.items():
        count = 0
        for row in rows:
            if any(str(row.get(column, "0")) == "1" for column in RESIDUAL_FLAG_COLUMNS):
                count += 1
        rates[mode] = count / float(max(1, len(rows)))
    return rates


def plot_any_residual_violation(
    summary: dict[str, dict[str, str]],
    case_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    modes = _ordered_modes(summary)
    values_by_mode = _any_residual_violation_rates(case_rows)
    values = [values_by_mode.get(mode, 0.0) for mode in modes]
    colors = [METHOD_COLORS.get(mode, "#6B7280") for mode in modes]
    y_positions = list(range(len(modes)))

    fig, ax = plt.subplots(figsize=(5.7, 3.15))
    bars = ax.barh(y_positions, values, color=colors, edgecolor="white", linewidth=0.7)
    for bar, value, mode in zip(bars, values, modes):
        if mode == "full_repair":
            bar.set_edgecolor("#0F513F")
            bar.set_linewidth(1.5)
        ax.text(min(0.985, value + 0.018), bar.get_y() + bar.get_height() / 2, f"{100 * value:.1f}", va="center", fontsize=7.8)
    ax.set_yticks(y_positions, [METHOD_LABELS.get(mode, mode) for mode in modes])
    ax.invert_yaxis()
    _percent_axis(ax)
    ax.text(0.985, -0.65, "lower is better", ha="right", va="center", fontsize=7.5, color="#667085")
    fig.tight_layout(pad=0.8)
    _save(fig, output_dir, "fig2_residual_violation_rate")
    plt.close(fig)


def plot_ablation_contract_success(summary: dict[str, dict[str, str]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    modes = [mode for mode in ABLATION_ORDER if mode in summary]
    labels = [ABLATION_LABELS[ABLATION_ORDER.index(mode)] for mode in modes]
    values = [_float(summary[mode].get("contract_compilation_success_rate")) for mode in modes]
    x_positions = list(range(len(modes)))

    fig, ax = plt.subplots(figsize=(4.3, 2.75))
    ax.plot(x_positions, values, color="#147D64", lw=2.2, marker="o", markersize=5.5)
    for x, value in zip(x_positions, values):
        ax.text(x, min(0.985, value + 0.035), f"{100 * value:.1f}", ha="center", va="bottom", fontsize=8.0)
    ax.set_xticks(x_positions, labels)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
    ax.set_ylabel("Rate (%)")
    ax.grid(axis="y", color="#E6EAF0", linewidth=0.8)
    fig.tight_layout(pad=0.8)
    _save(fig, output_dir, "fig2_ablation_contract_success")
    plt.close(fig)


def plot_l3_tightening(summary: dict[str, dict[str, str]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    modes = [mode for mode in ABLATION_ORDER if mode in summary]
    labels = [ABLATION_LABELS[ABLATION_ORDER.index(mode)] for mode in modes]
    values = [_float(summary[mode].get("risk_bound_tightening_rate")) for mode in modes]
    colors = [METHOD_COLORS.get(mode, "#6B7280") for mode in modes]
    x_positions = list(range(len(modes)))

    fig, ax = plt.subplots(figsize=(4.3, 2.75))
    bars = ax.bar(x_positions, values, color=colors, edgecolor="white", linewidth=0.7)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, min(0.985, value + 0.035), f"{100 * value:.1f}", ha="center", va="bottom", fontsize=8.0)
    ax.set_xticks(x_positions, labels)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
    ax.set_ylabel("Rate (%)")
    ax.grid(axis="y", color="#E6EAF0", linewidth=0.8)
    fig.tight_layout(pad=0.8)
    _save(fig, output_dir, "fig2_l3_risk_bound_tightening")
    plt.close(fig)


def write_html_board(output_dir: Path) -> None:
    stems = [
        "fig2_utility_preserving_executable",
        "fig2_contract_success",
        "fig2_ablation_contract_success",
        "fig2_l3_risk_bound_tightening",
        "fig2_residual_violation_rate",
        "fig2_plan_diversity",
        "fig2_candidate_outcomes",
    ]
    cards = "\n".join(
        f'<section><h2>{stem}.png</h2><img src="{stem}.png" alt="{stem}"/></section>'
        for stem in stems
        if (output_dir / f"{stem}.png").exists()
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Experiment 2 Single-Panel Paper Figures</title>
  <style>
    body{{margin:0;background:#f5f6f8;color:#1f2937;font-family:Segoe UI,Arial,sans-serif}}
    header{{padding:28px 40px 8px}}
    h1{{margin:0;font-size:24px}}
    main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:24px;padding:20px 40px 40px}}
    section{{background:white;border:1px solid #e5e7eb;border-radius:10px;padding:14px}}
    h2{{font-size:14px;margin:0 0 10px}}
    img{{display:block;width:100%;height:auto;border:1px solid #edf0f5}}
  </style>
</head>
<body>
  <header><h1>Experiment 2 Single-Panel Paper Figures</h1></header>
  <main>{cards}</main>
</body>
</html>
"""
    (output_dir / "figure_board.html").write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate single-panel paper figures for Exp. 2.")
    parser.add_argument(
        "--input-dir",
        default="outputs/candidate_repair/experiment2_candidate_repair_contract_v1",
        help="Directory containing summary.csv and case_results.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to INPUT_DIR/figures_paper_single.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if str(args.output_dir).strip() else input_dir / "figures_paper_single"
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    summary = _summary_by_mode(_read_csv(input_dir / "summary.csv"))
    case_rows = _read_csv(input_dir / "case_results.csv")

    plot_method_bar(
        summary,
        "utility_preserving_executable_ratio",
        "",
        output_dir,
        "fig2_utility_preserving_executable",
    )
    plot_method_bar(
        summary,
        "contract_compilation_success_rate",
        "",
        output_dir,
        "fig2_contract_success",
    )
    plot_ablation_contract_success(summary, output_dir)
    plot_l3_tightening(summary, output_dir)
    plot_any_residual_violation(summary, case_rows, output_dir)
    plot_method_bar(summary, "plan_diversity", "", output_dir, "fig2_plan_diversity")
    plot_candidate_outcomes(summary, case_rows, output_dir)
    write_html_board(output_dir)
    print(f"Wrote single-panel paper figures to {output_dir}")


if __name__ == "__main__":
    main()
