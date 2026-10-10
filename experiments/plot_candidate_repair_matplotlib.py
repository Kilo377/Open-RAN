"""Matplotlib paper figures for Experiment 2.

Reads the preserved Exp. 2 CSV artifacts and writes publication-friendly PNG,
PDF, and SVG figures.  The script does not rerun the simulator or modify the
data tables.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


METHOD_ORDER = [
    "schema_only",
    "repair_wo_memory",
    "repair_wo_l3",
    "full_repair",
    "orchestran_like",
    "oreo_like",
    "pacifista_like",
    "conservative_fallback",
]

METHOD_LABELS = {
    "schema_only": "Schema-only",
    "repair_wo_memory": "Repair w/o memory",
    "repair_wo_l3": "Repair w/o L3",
    "full_repair": "Full repair",
    "orchestran_like": "OrchestRAN",
    "oreo_like": "OREO",
    "pacifista_like": "PACIFISTA",
    "conservative_fallback": "Conservative fallback",
}

METHOD_COLORS = {
    "schema_only": "#8A94A6",
    "repair_wo_memory": "#5B8CCB",
    "repair_wo_l3": "#7B76CF",
    "full_repair": "#147D64",
    "orchestran_like": "#D28A22",
    "oreo_like": "#B66AA5",
    "pacifista_like": "#C94C4C",
    "conservative_fallback": "#5D6678",
}

OUTCOME_BUCKETS = [
    ("executable_without_repair", "Executable without repair", "#8DBF67"),
    ("repaired_and_executable", "Repaired and executable", "#2C8C7D"),
    ("rejected_due_to_hard_conflict", "Rejected: hard conflict", "#C94C4C"),
    ("rejected_due_to_empty_bounds", "Rejected: empty bounds", "#E4A84D"),
    ("rejected_due_to_resource_overflow", "Rejected: resource overflow", "#B869A5"),
    ("rejected_due_to_policy_violation", "Rejected: policy", "#7861A7"),
    ("residual_resource_overflow", "Residual resource overflow", "#934E60"),
    ("residual_hard_conflict", "Residual hard conflict", "#8F2D2D"),
    ("residual_policy_violation", "Residual policy", "#5C4A83"),
    ("residual_other", "Residual other", "#949CA8"),
    ("rejected_other", "Rejected other", "#C8CDD5"),
]

OVERVIEW_METRICS = [
    ("final_executable_plan_ratio", "Executable ratio"),
    ("contract_compilation_success_rate", "Contract success"),
    ("utility_preserving_executable_ratio", "Utility-preserving"),
    ("plan_diversity", "Plan diversity"),
]

CONTRACT_METRICS = [
    ("contract_compilation_success_rate", "Contract success", True),
    ("projectable_constraint_completeness", "Projectable constraints", True),
    ("kpi_guardrail_coverage", "KPI guardrail coverage", True),
    ("admissible_region_volume_ratio", "Action-region volume", True),
    ("over_restriction_rate", "Over-restriction", False),
    ("contract_inconsistency_rate", "Contract inconsistency", False),
]

RESIDUAL_METRICS = [
    ("hard_conflict_residual_rate", "Hard conflict", "#C94C4C"),
    ("resource_overflow_residual_rate", "Resource overflow", "#D28A22"),
    ("policy_violation_residual_rate", "Policy violation", "#7861A7"),
    ("unsafe_bound_residual_rate", "Unsafe bounds", "#B869A5"),
]

ABLATION_ORDER = ["repair_wo_memory", "repair_wo_l3", "full_repair"]

ABLATION_STAGE_LABELS = {
    "repair_wo_memory": "No memory",
    "repair_wo_l3": "L2 memory\nscoring",
    "full_repair": "L2 + L3\nbound repair",
}

ABLATION_QUALITY_METRICS = [
    ("contract_compilation_success_rate", "Contract success", "#147D64", False),
    ("utility_preserving_executable_ratio", "Utility-preserving", "#5B8CCB", False),
    ("protected_kpi_uncontrolled_rate", "Protected KPI controlled", "#7B76CF", True),
]

ABLATION_TRACE_METRICS = [
    ("l2_penalty_applied_rate", "L2 trace", "#7B76CF"),
    ("risk_bound_tightening_rate", "L3 bound tightening", "#147D64"),
]

BOUND_PATTERN = re.compile(r"(?P<target>[^.;]+)\.(?P<field>[^=\[]+)=\[(?P<lower>[^,]+),(?P<upper>[^\]]+)\]")


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


def _labels(modes: list[str]) -> list[str]:
    return [METHOD_LABELS.get(mode, mode) for mode in modes]


def _save(fig, output_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"bbox_inches": "tight", "facecolor": fig.get_facecolor()}
        if suffix == "png":
            kwargs["dpi"] = 360
        fig.savefig(output_dir / f"{stem}.{suffix}", **kwargs)


def configure_style() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "#F7F8FB",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#D9DEE8",
            "axes.linewidth": 0.9,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "xtick.color": "#475467",
            "ytick.color": "#475467",
            "axes.labelcolor": "#344054",
            "text.color": "#1F2937",
            "grid.color": "#E9EDF3",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "savefig.dpi": 360,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_overview(summary: dict[str, dict[str, str]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    modes = _ordered_modes(summary)
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.2), sharey=True)
    axes = list(axes.ravel())
    y_positions = list(range(len(modes)))
    colors = [METHOD_COLORS.get(mode, "#6B7280") for mode in modes]
    for ax, (metric, title) in zip(axes, OVERVIEW_METRICS):
        values = [_float(summary[mode].get(metric)) for mode in modes]
        bars = ax.barh(y_positions, values, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(title, loc="left", pad=10)
        ax.set_xlim(0, 1.04)
        ax.grid(axis="x")
        ax.set_yticks(y_positions, labels=_labels(modes))
        ax.invert_yaxis()
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
        for bar, value, mode in zip(bars, values, modes):
            label = f"{100 * value:.1f}%"
            x_text = min(value + 0.018, 0.97)
            ax.text(x_text, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=8.5)
            if mode == "full_repair":
                bar.set_edgecolor("#073B2F")
                bar.set_linewidth(1.8)
    fig.suptitle(
        "Experiment 2: Feasibility-preserving repair and contract compilation",
        x=0.02,
        y=1.02,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.965,
        "Full repair is the only method that combines high executable ratio, reliable contracts, useful plans, and non-collapsed diversity.",
        ha="left",
        color="#667085",
    )
    legend_handles = [Patch(facecolor=METHOD_COLORS["full_repair"], label="Full repair")]
    axes[0].legend(handles=legend_handles, loc="lower right")
    fig.tight_layout(pad=2.0)
    _save(fig, output_dir, "exp2_matplotlib_overview")
    plt.close(fig)


def _outcome_fractions(case_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in case_rows:
        grouped.setdefault(str(row.get("mode", "")), []).append(row)
    result: dict[str, dict[str, float]] = {}
    for mode, rows in grouped.items():
        denom = max(1, len(rows))
        result[mode] = {
            bucket: sum(1 for row in rows if str(row.get("outcome_bucket", "")) == bucket) / denom
            for bucket, _label, _color in OUTCOME_BUCKETS
        }
    return result


def plot_candidate_outcomes(
    summary: dict[str, dict[str, str]],
    case_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    modes = _ordered_modes(summary)
    fractions = _outcome_fractions(case_rows)
    x_positions = list(range(len(modes)))
    fig, ax = plt.subplots(figsize=(13.8, 6.7))
    bottom = [0.0 for _ in modes]
    for bucket, label, color in OUTCOME_BUCKETS:
        values = [fractions.get(mode, {}).get(bucket, 0.0) for mode in modes]
        ax.bar(
            x_positions,
            values,
            bottom=bottom,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.45,
        )
        bottom = [old + value for old, value in zip(bottom, values)]
    ax.set_title("Candidate outcome decomposition", loc="left", pad=14)
    ax.text(
        -0.48,
        1.075,
        "Same 3240 candidates per method; stacked bars show how each candidate exits validation, repair, or contract compilation.",
        color="#667085",
        transform=ax.transData,
    )
    ax.set_ylabel("Candidate fraction")
    ax.set_ylim(0, 1)
    ax.set_xticks(x_positions, _labels(modes), rotation=28, ha="right")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
    ax.grid(axis="y")
    for x, mode in zip(x_positions, modes):
        executable = _float(summary[mode].get("final_executable_plan_ratio"))
        ax.text(x, 1.015, f"{100 * executable:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.legend(ncol=3, bbox_to_anchor=(0.5, -0.24), loc="upper center", fontsize=8.5)
    fig.tight_layout()
    _save(fig, output_dir, "exp2_matplotlib_candidate_outcomes")
    plt.close(fig)


def plot_contract_heatmap(summary: dict[str, dict[str, str]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    modes = _ordered_modes(summary)
    values: list[list[float]] = []
    annotations: list[list[str]] = []
    for mode in modes:
        row: list[float] = []
        anno_row: list[str] = []
        for metric, _label, higher_is_better in CONTRACT_METRICS:
            raw = max(0.0, min(1.0, _float(summary[mode].get(metric))))
            score = raw if higher_is_better else 1.0 - raw
            row.append(score)
            anno_row.append(f"{100 * raw:.1f}%")
        values.append(row)
        annotations.append(anno_row)

    cmap = LinearSegmentedColormap.from_list("contract_quality", ["#C94C4C", "#E4A84D", "#5B8CCB", "#147D64"])
    fig, ax = plt.subplots(figsize=(13.4, 6.8))
    ax.imshow(values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_title("Guidance-contract quality matrix", loc="left", pad=14)
    ax.text(
        -0.5,
        -1.08,
        "Green cells are desirable. Raw percentages are shown; lower is better for over-restriction and inconsistency.",
        color="#667085",
        transform=ax.transData,
    )
    ax.set_xticks(range(len(CONTRACT_METRICS)), [label for _metric, label, _higher in CONTRACT_METRICS], rotation=28, ha="right")
    ax.set_yticks(range(len(modes)), _labels(modes))
    for row_index, mode in enumerate(modes):
        for col_index in range(len(CONTRACT_METRICS)):
            text_color = "white" if values[row_index][col_index] < 0.58 or values[row_index][col_index] > 0.78 else "#1F2937"
            ax.text(
                col_index,
                row_index,
                annotations[row_index][col_index],
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color=text_color,
            )
        if mode == "full_repair":
            ax.add_patch(plt.Rectangle((-0.5, row_index - 0.5), len(CONTRACT_METRICS), 1.0, fill=False, ec="#073B2F", lw=2.2))
    ax.set_xticks([x - 0.5 for x in range(1, len(CONTRACT_METRICS))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(modes))], minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    _save(fig, output_dir, "exp2_matplotlib_contract_heatmap")
    plt.close(fig)


def plot_residuals(summary: dict[str, dict[str, str]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    modes = _ordered_modes(summary)
    y_positions = list(range(len(modes)))
    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    offsets = [-0.27, -0.09, 0.09, 0.27]
    height = 0.16
    for offset, (metric, label, color) in zip(offsets, RESIDUAL_METRICS):
        values = [_float(summary[mode].get(metric)) for mode in modes]
        ax.barh(
            [y + offset for y in y_positions],
            values,
            height=height,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.6,
        )
    ax.set_title("Residual violation rates after method-specific handling", loc="left", pad=14)
    ax.text(
        0,
        -0.95,
        "Lower is better. Full repair removes hard, resource, policy, and unsafe-bound residuals in this static repair benchmark.",
        color="#667085",
    )
    ax.set_xlim(0, max(0.32, max(_float(summary[mode].get("resource_overflow_residual_rate")) for mode in modes) + 0.05))
    ax.set_yticks(y_positions, _labels(modes))
    ax.invert_yaxis()
    ax.set_xlabel("Residual rate")
    ax.grid(axis="x")
    ax.legend(ncol=4, bbox_to_anchor=(0.5, -0.14), loc="upper center")
    fig.tight_layout()
    _save(fig, output_dir, "exp2_matplotlib_residuals")
    plt.close(fig)


def _extract_bound_values(text: str, field_name: str) -> list[tuple[float, float]]:
    values: list[tuple[float, float]] = []
    for match in BOUND_PATTERN.finditer(text or ""):
        if match.group("field") != field_name:
            continue
        lower = _float(match.group("lower"), math.nan)
        upper = _float(match.group("upper"), math.nan)
        if not math.isnan(lower) and not math.isnan(upper):
            values.append((lower, upper))
    return values


def _mean(values: list[float]) -> float:
    return sum(values) / float(max(1, len(values)))


def plot_risk_bound_repair(
    summary: dict[str, dict[str, str]],
    case_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    full_rows = [
        row
        for row in case_rows
        if row.get("mode") == "full_repair"
        and row.get("risk_conflict_present") == "1"
        and row.get("risk_bound_tightened") == "1"
    ]
    steering_raw: list[float] = []
    steering_repaired: list[float] = []
    hyst_raw: list[float] = []
    hyst_repaired: list[float] = []
    for row in full_rows:
        steering_raw.extend(upper for _lower, upper in _extract_bound_values(row.get("requested_bounds", ""), "steering_aggressiveness"))
        steering_repaired.extend(upper for _lower, upper in _extract_bound_values(row.get("output_bounds", ""), "steering_aggressiveness"))
        hyst_raw.extend(lower for lower, _upper in _extract_bound_values(row.get("requested_bounds", ""), "steering_hysteresis_db"))
        hyst_repaired.extend(lower for lower, _upper in _extract_bound_values(row.get("output_bounds", ""), "steering_hysteresis_db"))

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), gridspec_kw={"width_ratios": [1.05, 1.0]})
    ax = axes[0]
    ax.set_title("L3-supported bound repair example", loc="left", pad=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    examples = [
        ("steering_aggressiveness upper", _mean(steering_raw), _mean(steering_repaired), "#5B8CCB", "target <= 0.60"),
        ("steering_hysteresis_db lower", _mean(hyst_raw), _mean(hyst_repaired), "#147D64", "target >= 3.00 dB"),
    ]
    for index, (label, before, after, color, target) in enumerate(examples):
        y = 0.68 - index * 0.36
        ax.text(0.02, y + 0.11, label, fontweight="bold")
        ax.scatter([0.24, 0.68], [y, y], s=[780, 780], color=[color, color], alpha=[0.95, 0.45])
        arrow = FancyArrowPatch((0.31, y), (0.61, y), arrowstyle="-|>", mutation_scale=18, color="#667085", lw=2.0)
        ax.add_patch(arrow)
        ax.text(0.24, y - 0.13, f"raw mean\n{before:.2f}", ha="center", va="top", fontsize=9)
        ax.text(0.68, y - 0.13, f"repaired mean\n{after:.2f}", ha="center", va="top", fontsize=9)
        ax.text(0.79, y + 0.01, target, color="#667085", fontsize=9)

    ax2 = axes[1]
    modes = _ordered_modes(summary)
    values = [_float(summary[mode].get("risk_bound_tightening_rate")) for mode in modes]
    colors = [METHOD_COLORS.get(mode, "#6B7280") for mode in modes]
    y_positions = list(range(len(modes)))
    ax2.barh(y_positions, values, color=colors, edgecolor="white", linewidth=0.7)
    ax2.set_title("Risk-bound tightening rate", loc="left", pad=12)
    ax2.set_xlim(0, 1)
    ax2.set_yticks(y_positions, _labels(modes))
    ax2.invert_yaxis()
    ax2.grid(axis="x")
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
    for y, value in zip(y_positions, values):
        ax2.text(min(0.98, value + 0.02), y, f"{100 * value:.1f}%", va="center", fontsize=8.5)
    fig.suptitle("Risk-conflict repair converts unsafe candidate regions into admissible guidance contracts", x=0.02, y=1.03, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, output_dir, "exp2_matplotlib_risk_bound_repair")
    plt.close(fig)


def plot_ablation_ladder(summary: dict[str, dict[str, str]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    modes = [mode for mode in ABLATION_ORDER if mode in summary]
    if not modes:
        return

    x_positions = list(range(len(modes)))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.8, 5.8),
        gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.22},
    )

    ax = axes[0]
    width = 0.22
    offsets = [-width, 0.0, width]
    for offset, (metric, label, color, invert) in zip(offsets, ABLATION_QUALITY_METRICS):
        raw_values = [_float(summary[mode].get(metric)) for mode in modes]
        values = [1.0 - value if invert else value for value in raw_values]
        bars = ax.bar(
            [x + offset for x in x_positions],
            values,
            width=width * 0.92,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.7,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(1.015, value + 0.025),
                f"{100 * value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    contract_values = [_float(summary[mode].get("contract_compilation_success_rate")) for mode in modes]
    ax.plot(
        x_positions,
        contract_values,
        color="#0F513F",
        lw=2.0,
        marker="o",
        markersize=6,
        label="Contract-success trajectory",
        zorder=5,
    )
    for left, right in zip(x_positions[:-1], x_positions[1:]):
        delta = contract_values[right] - contract_values[left]
        y = min(1.04, max(contract_values[left], contract_values[right]) + 0.075)
        ax.text(
            (left + right) / 2,
            y,
            f"+{100 * delta:.1f} pts",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#0F513F",
            fontweight="bold",
        )

    ax.set_title("Ablation ladder", loc="left", pad=8)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x_positions, [ABLATION_STAGE_LABELS.get(mode, mode) for mode in modes])
    ax.set_ylabel("Rate")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
    ax.grid(axis="y")
    ax.legend(ncol=2, bbox_to_anchor=(0.5, -0.2), loc="upper center", fontsize=8.5)

    ax2 = axes[1]
    trace_width = 0.32
    trace_offsets = [-trace_width / 2, trace_width / 2]
    for offset, (metric, label, color) in zip(trace_offsets, ABLATION_TRACE_METRICS):
        values = [_float(summary[mode].get(metric)) for mode in modes]
        bars = ax2.bar(
            [x + offset for x in x_positions],
            values,
            width=trace_width * 0.86,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.7,
        )
        for bar, value in zip(bars, values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                min(1.015, value + 0.025),
                f"{100 * value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    ax2.set_title("Mechanism trace", loc="left", pad=8)
    ax2.set_ylim(0, 1.08)
    ax2.set_xticks(x_positions, [ABLATION_STAGE_LABELS.get(mode, mode) for mode in modes])
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
    ax2.grid(axis="y")
    ax2.legend(ncol=1, loc="upper left", fontsize=8.5)

    fig.suptitle(
        "Repair ablation: from deterministic feasibility to memory-guided contract compilation",
        x=0.07,
        y=0.98,
        ha="left",
        fontsize=13.5,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.905,
        "Executable ratio is matched across these ablations; the separation appears in contract success, KPI control, and L2/L3 traces.",
        color="#667085",
        fontsize=9.5,
    )
    fig.subplots_adjust(top=0.78, bottom=0.25, left=0.07, right=0.98, wspace=0.28)
    _save(fig, output_dir, "exp2_matplotlib_ablation_ladder")
    plt.close(fig)


def write_html_board(output_dir: Path) -> None:
    stems = [
        "exp2_matplotlib_overview",
        "exp2_matplotlib_ablation_ladder",
        "exp2_matplotlib_candidate_outcomes",
        "exp2_matplotlib_contract_heatmap",
        "exp2_matplotlib_residuals",
        "exp2_matplotlib_risk_bound_repair",
    ]
    cards = []
    for stem in stems:
        cards.append(
            f'<section><h2>{stem}.png</h2><img src="{stem}.png" alt="{stem}"/></section>'
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Experiment 2 Matplotlib Figures</title>
  <style>
    body{{margin:0;background:#f3f5f8;color:#1f2937;font-family:Segoe UI,Arial,sans-serif}}
    header{{padding:30px 44px 10px}}
    h1{{margin:0 0 8px;font-size:28px}}
    p{{color:#667085}}
    main{{display:grid;gap:28px;padding:20px 44px 44px}}
    section{{background:white;border:1px solid #e5e7eb;border-radius:18px;padding:18px;box-shadow:0 10px 30px rgba(16,24,40,.08)}}
    h2{{font-size:16px;margin:0 0 12px}}
    img{{display:block;width:100%;height:auto;border-radius:10px;border:1px solid #edf0f5}}
  </style>
</head>
<body>
  <header>
    <h1>Experiment 2 Matplotlib Figure Board</h1>
    <p>PNG previews with matching PDF and SVG files in the same directory.</p>
  </header>
  <main>{''.join(cards)}</main>
</body>
</html>
"""
    (output_dir / "exp2_matplotlib_figure_board.html").write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate matplotlib figures for Exp. 2.")
    parser.add_argument(
        "--input-dir",
        default="outputs/candidate_repair/experiment2_candidate_repair_contract_v1",
        help="Directory containing summary.csv and case_results.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to INPUT_DIR/figures_matplotlib.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if str(args.output_dir).strip() else input_dir / "figures_matplotlib"
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    summary = _summary_by_mode(_read_csv(input_dir / "summary.csv"))
    case_rows = _read_csv(input_dir / "case_results.csv")

    plot_overview(summary, output_dir)
    plot_ablation_ladder(summary, output_dir)
    plot_candidate_outcomes(summary, case_rows, output_dir)
    plot_contract_heatmap(summary, output_dir)
    plot_residuals(summary, output_dir)
    plot_risk_bound_repair(summary, case_rows, output_dir)
    write_html_board(output_dir)
    print(f"Wrote matplotlib figures to {output_dir}")


if __name__ == "__main__":
    main()
