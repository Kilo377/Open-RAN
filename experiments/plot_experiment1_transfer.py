"""Plot Experiment 1 warm-up/transfer headline results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


VARIANT_LABELS = {
    "fixed_plan": "Fixed plan",
    "no_memory_planner": "No-memory planner",
    "success_only_memory": "Success-only memory",
    "orchestran_like_resource_aware": "OrchestRAN",
    "intent_hrl_like": "Intent-HRL",
    "oreo_like_service_graph": "OREO",
    "full_method": "Full method",
}


def _variant_label(variant_name: str) -> str:
    return VARIANT_LABELS.get(str(variant_name), str(variant_name))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Experiment 1 transfer results.")
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="E:/oran_slot_sim_outputs/paper_eval/experiment1_cross_scenario_transfer_v1",
    )
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except ValueError:
        return 0.0


def _plot_bar(*, rows: list[dict[str, str]], metric: str, ylabel: str, title: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    variant_names = [row.get("variant_name", "") for row in rows]
    labels = [_variant_label(variant_name) for variant_name in variant_names]
    values = [_float(row, metric) for row in rows]
    colors = [
        "#6b7280" if variant_name != "full_method" else "#0f766e"
        for variant_name in variant_names
    ]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_recovery_curve(
    *,
    rows: list[dict[str, str]],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> bool:
    if len(rows) == 0:
        return False
    import matplotlib.pyplot as plt

    order = [
        "fixed_plan",
        "no_memory_planner",
        "success_only_memory",
        "orchestran_like_resource_aware",
        "intent_hrl_like",
        "oreo_like_service_graph",
        "full_method",
    ]
    by_variant: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_variant.setdefault(str(row.get("variant_name", "")), []).append(row)
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    plotted = False
    for variant_name in order:
        selected = sorted(
            by_variant.get(variant_name, []),
            key=lambda item: _float(item, "phase_repeat"),
        )
        if len(selected) == 0:
            continue
        x_values = [_float(item, "phase_repeat") for item in selected]
        y_values = [_float(item, metric) for item in selected]
        color = "#0f766e" if variant_name == "full_method" else None
        linewidth = 3.0 if variant_name == "full_method" else 1.7
        ax.plot(
            x_values,
            y_values,
            marker="o",
            label=_variant_label(variant_name),
            color=color,
            linewidth=linewidth,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_title(title)
    ax.set_xlabel("Transfer repeat")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return True


def _plot_high_risk_grouped(
    *,
    rows: list[dict[str, str]],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> bool:
    if len(rows) == 0:
        return False
    import matplotlib.pyplot as plt
    import numpy as np

    variants = [
        "no_memory_planner",
        "success_only_memory",
        "intent_hrl_like",
        "full_method",
    ]
    buckets = sorted({str(row.get("risk_bucket", "")) for row in rows if str(row.get("risk_bucket", ""))})
    if len(buckets) == 0:
        return False
    values_by_key = {
        (str(row.get("risk_bucket", "")), str(row.get("variant_name", ""))): _float(row, metric)
        for row in rows
    }
    x_positions = np.arange(len(buckets))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for index, variant_name in enumerate(variants):
        values = [values_by_key.get((bucket, variant_name), 0.0) for bucket in buckets]
        offset = (index - (len(variants) - 1) / 2.0) * width
        color = "#0f766e" if variant_name == "full_method" else None
        ax.bar(x_positions + offset, values, width=width, label=_variant_label(variant_name), color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(buckets, rotation=22, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return True


def main() -> None:
    args = _parse_args()
    experiment_dir = Path(str(args.experiment_dir)).resolve()
    rows = _load_rows(experiment_dir / "transfer_headline_results.csv")
    if len(rows) == 0:
        raise FileNotFoundError(f"No transfer_headline_results.csv found under {experiment_dir}")

    order = [
        "fixed_plan",
        "no_memory_planner",
        "success_only_memory",
        "orchestran_like_resource_aware",
        "intent_hrl_like",
        "oreo_like_service_graph",
        "full_method",
    ]
    rows_by_name = {row.get("variant_name", ""): row for row in rows}
    ordered_rows = [rows_by_name[name] for name in order if name in rows_by_name]
    figures_dir = experiment_dir / "figures_transfer"

    plots = [
        (
            "cross_scenario_adaptation_score",
            "Adaptation score (lower is better)",
            "Experiment 1 degradation-adjusted transfer adaptation",
            "exp1_transfer_adaptation_score.png",
        ),
        (
            "cross_scenario_degradation_score",
            "Warm-up to transfer degradation (lower is better)",
            "Experiment 1 cross-scenario degradation",
            "exp1_transfer_degradation_score.png",
        ),
        (
            "degradation_safety_first_delta",
            "Safety-first score delta after transfer",
            "Experiment 1 warm-up to transfer safety-first degradation",
            "exp1_transfer_degradation_safety_first_delta.png",
        ),
        (
            "transfer_high_risk_safety_first_score",
            "High-risk safety-first score (lower is better)",
            "Experiment 1 high-risk transfer safety-first score",
            "exp1_transfer_high_risk_safety_first.png",
        ),
        (
            "transfer_high_risk_safety_violation_rate",
            "High-risk safety violation rate",
            "Experiment 1 high-risk transfer safety violations",
            "exp1_transfer_high_risk_safety.png",
        ),
        (
            "transfer_recovery_gain_final_vs_first",
            "Recovery gain vs first transfer repeat",
            "Experiment 1 recovery across transfer repeats",
            "exp1_transfer_recovery_gain.png",
        ),
        (
            "transfer_safety_first_score",
            "Safety-first transfer score (lower is better)",
            "Experiment 1 safety-first cross-scenario orchestration score",
            "exp1_transfer_safety_first_score.png",
        ),
        (
            "transfer_tail_robustness_score",
            "Tail-robust transfer score (lower is better)",
            "Experiment 1 tail-robust cross-scenario orchestration score",
            "exp1_transfer_tail_robustness_score.png",
        ),
        (
            "transfer_p90_case_safety_first_score",
            "Worst-tail case score (lower is better)",
            "Experiment 1 worst-tail case safety-first score",
            "exp1_transfer_p90_case_safety_first_score.png",
        ),
        (
            "transfer_memory_sensitive_tail_robustness_score",
            "Memory-sensitive tail score (lower is better)",
            "Experiment 1 memory-sensitive transfer tail score",
            "exp1_transfer_memory_sensitive_tail_score.png",
        ),
        (
            "transfer_memory_sensitive_mean_safety_violation_rate",
            "Memory-sensitive safety violation rate",
            "Experiment 1 memory-sensitive transfer safety violations",
            "exp1_transfer_memory_sensitive_safety.png",
        ),
        (
            "transfer_safety_first_gain_vs_no_memory",
            "Gain vs no-memory planner",
            "Experiment 1 safety-first gain over no-memory planning",
            "exp1_transfer_gain_vs_no_memory.png",
        ),
        (
            "transfer_cross_scenario_score",
            "Legacy cross-scenario score (lower is better)",
            "Experiment 1 legacy transfer cross-scenario orchestration score",
            "exp1_transfer_cross_scenario_score.png",
        ),
        (
            "transfer_mean_intent_loss_normalized",
            "Normalized intent loss",
            "Experiment 1 transfer normalized intent loss",
            "exp1_transfer_intent_loss.png",
        ),
        (
            "transfer_mean_hard_target_fulfillment",
            "Hard target fulfillment ratio",
            "Experiment 1 transfer hard KPI target fulfillment",
            "exp1_transfer_hard_target_fulfillment.png",
        ),
        (
            "transfer_mean_hard_intent_fulfillment",
            "Hard intent fulfillment ratio",
            "Experiment 1 transfer hard intent fulfillment",
            "exp1_transfer_hard_intent_fulfillment.png",
        ),
        (
            "transfer_mean_safety_violation_rate",
            "Safety violation rate",
            "Experiment 1 transfer safety violation rate",
            "exp1_transfer_safety.png",
        ),
        (
            "transfer_safety_success_rate",
            "Safety success rate",
            "Experiment 1 transfer safety success rate",
            "exp1_transfer_safety_success.png",
        ),
        (
            "transfer_mean_safe_baseline_fallback_rate",
            "Near-RT fallback rate",
            "Experiment 1 Near-RT safe-baseline fallback rate",
            "exp1_transfer_near_rt_fallback.png",
        ),
        (
            "transfer_mean_forced_non_rt_reselection_count",
            "Forced Non-RT reselections",
            "Experiment 1 forced Non-RT recovery reselections",
            "exp1_transfer_forced_non_rt_reselection.png",
        ),
        (
            "transfer_mean_throughput_mbps",
            "Throughput (Mbps)",
            "Experiment 1 transfer throughput",
            "exp1_transfer_throughput.png",
        ),
        (
            "transfer_mean_switching_cost",
            "Switching cost",
            "Experiment 1 transfer switching cost",
            "exp1_transfer_switching.png",
        ),
    ]
    generated: list[str] = []
    for metric, ylabel, title, filename in plots:
        output_path = figures_dir / filename
        _plot_bar(
            rows=ordered_rows,
            metric=metric,
            ylabel=ylabel,
            title=title,
            output_path=output_path,
        )
        generated.append(str(output_path))

    recovery_rows = _load_rows(experiment_dir / "transfer_repeat_recovery.csv")
    recovery_plot_path = figures_dir / "exp1_transfer_recovery_safety_first_curve.png"
    if _plot_recovery_curve(
        rows=recovery_rows,
        metric="safety_first_score",
        ylabel="Safety-first score (lower is better)",
        title="Experiment 1 recovery over repeated transfer exposure",
        output_path=recovery_plot_path,
    ):
        generated.append(str(recovery_plot_path))
    recovery_gain_plot_path = figures_dir / "exp1_transfer_recovery_gain_curve.png"
    if _plot_recovery_curve(
        rows=recovery_rows,
        metric="safety_first_gain_vs_first_repeat",
        ylabel="Gain vs first transfer repeat",
        title="Experiment 1 cumulative recovery gain over transfer repeats",
        output_path=recovery_gain_plot_path,
    ):
        generated.append(str(recovery_gain_plot_path))

    high_risk_rows = _load_rows(experiment_dir / "transfer_high_risk_results.csv")
    high_risk_plot_path = figures_dir / "exp1_transfer_high_risk_bucket_safety_first.png"
    if _plot_high_risk_grouped(
        rows=high_risk_rows,
        metric="safety_first_score",
        ylabel="Safety-first score (lower is better)",
        title="Experiment 1 high-risk transfer categories",
        output_path=high_risk_plot_path,
    ):
        generated.append(str(high_risk_plot_path))

    summary = {
        "headline_row_count": len(ordered_rows),
        "figures": generated,
    }
    (figures_dir / "plot_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Plotted Experiment 1 transfer figures under {figures_dir}")


if __name__ == "__main__":
    main()
