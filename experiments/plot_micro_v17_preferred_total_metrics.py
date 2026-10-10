"""Create preferred total-metric figures for Experiment 1 micro-v17 chunks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ORDER = [
    ("full_method", "Full"),
    ("success_only_memory", "Success-only"),
    ("no_memory_planner", "No-memory"),
    ("orchestran_like_resource_aware", "OrchestRAN-like"),
    ("intent_hrl_like", "Intent-HRL-like"),
    ("oreo_like_service_graph", "OREO-like"),
    ("fixed_plan", "Fixed"),
]

COLORS = {
    "full_method": "#0f7a55",
    "success_only_memory": "#d95f02",
    "no_memory_planner": "#0877b2",
    "orchestran_like_resource_aware": "#c875a7",
    "intent_hrl_like": "#858585",
    "oreo_like_service_graph": "#eda900",
    "fixed_plan": "#4d4d4d",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot preferred total metrics for exp1 micro-v17.")
    parser.add_argument("--root", default="E:/oran_slot_sim_outputs/paper_eval")
    parser.add_argument("--through", type=int, required=True)
    return parser.parse_args()


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    args = _parse_args()
    root = Path(str(args.root)).resolve()
    through = int(args.through)
    macro_path = root / f"exp1_micro_v17_aggregated_metrics_through_chunk{through:03d}.csv"
    wide_path = root / "exp1_micro_v17_per_chunk_metrics_wide.csv"
    out_dir = root / "figures_micro_v17"
    out_dir.mkdir(parents=True, exist_ok=True)

    macro = {row["variant"]: row for row in _load_csv(macro_path)}
    wide = _load_csv(wide_path)
    variants = [variant for variant, _ in ORDER if variant in macro]
    labels = [label for variant, label in ORDER if variant in macro]
    x_positions = np.arange(len(variants))

    def values(metric: str) -> list[float]:
        return [float(macro[variant][metric]) for variant in variants]

    def bar_panel(ax, metric: str, title: str, ylabel: str, *, higher: bool = False) -> None:
        vals = values(metric)
        bars = ax.bar(
            x_positions,
            vals,
            color=[COLORS[variant] for variant in variants],
            width=0.72,
        )
        ax.set_title(title, fontsize=13, pad=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
        best_index = int(np.argmax(vals) if higher else np.argmin(vals))
        for index, (bar, val) in enumerate(zip(bars, vals)):
            label = f"{int(val)}" if metric == "safety_first_wins" else f"{val:.3f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold" if index == best_index else "normal",
            )
        ymin, ymax = ax.get_ylim()
        if ymax > 0:
            ax.set_ylim(ymin, ymax * 1.14)

    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))
    fig.suptitle(
        f"Experiment 1 Aggregated Metrics Across {through} Micro Experiments",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    bar_panel(axes[0, 0], "mean_safety_first", "Safety-first transfer score (lower is better)", "score")
    bar_panel(axes[0, 1], "mean_intent_loss", "Normalized intent loss (lower is better)", "loss")
    bar_panel(axes[0, 2], "mean_safety_violation", "Safety violation rate (lower is better)", "rate")
    bar_panel(axes[1, 0], "mean_hard_fulfill", "Hard KPI fulfillment (higher is better)", "ratio", higher=True)
    bar_panel(axes[1, 1], "safety_first_wins", "Per-chunk safety-first wins (higher is better)", "wins", higher=True)

    ax = axes[1, 2]
    chunk_ids = [int(float(row["chunk_id"])) for row in wide if int(float(row["chunk_id"])) <= through]
    selected_wide = [row for row in wide if int(float(row["chunk_id"])) <= through]
    for variant, label in [
        ("full_method", "Full"),
        ("success_only_memory", "Success-only"),
        ("no_memory_planner", "No-memory"),
        ("oreo_like_service_graph", "OREO-like"),
        ("orchestran_like_resource_aware", "OrchestRAN-like"),
        ("fixed_plan", "Fixed"),
    ]:
        key = f"safety_first__{variant}"
        vals = [float(row.get(key, "nan")) for row in selected_wide]
        ax.plot(
            chunk_ids,
            vals,
            marker="o",
            linewidth=3 if variant == "full_method" else 1.8,
            label=label,
            color=COLORS.get(variant),
        )
    ax.set_title("Safety-first score by micro experiment", fontsize=13, pad=8)
    ax.set_xlabel("chunk id")
    ax.set_ylabel("score")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2, loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out_main = out_dir / f"exp1_micro_v17_preferred_total_metrics_through_chunk{through:03d}.png"
    out_pdf = out_dir / f"exp1_micro_v17_preferred_total_metrics_through_chunk{through:03d}.pdf"
    fig.savefig(out_main, dpi=260)
    fig.savefig(out_pdf)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    fig.suptitle(
        f"Aggregated Metrics Across {through} Micro Experiments",
        fontsize=16,
        fontweight="bold",
        y=1.03,
    )
    bar_panel(axes[0], "mean_intent_loss", "Intent loss (lower is better)", "loss")
    bar_panel(axes[1], "mean_hard_fulfill", "Hard KPI fulfillment (higher is better)", "ratio", higher=True)
    bar_panel(axes[2], "mean_safety_violation", "Safety violation (lower is better)", "rate")
    bar_panel(axes[3], "mean_safety_first", "Safety-first score (lower is better)", "score")
    fig.tight_layout()
    out_compact = out_dir / f"exp1_micro_v17_preferred_total_metrics_compact_through_chunk{through:03d}.png"
    out_compact_pdf = out_dir / f"exp1_micro_v17_preferred_total_metrics_compact_through_chunk{through:03d}.pdf"
    fig.savefig(out_compact, dpi=260, bbox_inches="tight")
    fig.savefig(out_compact_pdf, bbox_inches="tight")
    plt.close(fig)

    print(out_main)
    print(out_compact)


if __name__ == "__main__":
    main()
