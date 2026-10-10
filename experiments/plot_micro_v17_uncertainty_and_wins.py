"""Plot uncertainty and per-chunk win summaries for Experiment 1 micro-v17."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
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

METRICS = [
    ("intent_loss", "Intent loss", "lower"),
    ("hard_fulfill", "Hard KPI fulfillment", "higher"),
    ("safety_violation", "Safety violation", "lower"),
    ("safety_first", "Safety-first score", "lower"),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot exp1 micro-v17 uncertainty and wins.")
    parser.add_argument("--root", default="E:/oran_slot_sim_outputs/paper_eval")
    parser.add_argument("--through", type=int, required=True)
    return parser.parse_args()


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _tcrit_975(df: int) -> float:
    # Two-sided 95% Student-t critical values. Fall back to normal for large df.
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    if df <= 0:
        return 0.0
    return table.get(int(df), 1.960)


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = _mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / float(len(values) - 1))


def _load_rows(path: Path, through: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        for raw in csv.DictReader(input_file):
            chunk_id = int(_float(raw.get("chunk_id"), 0.0))
            if chunk_id <= 0 or chunk_id > through:
                continue
            rows.append(
                {
                    "chunk": str(raw.get("chunk", "")),
                    "chunk_id": chunk_id,
                    "label": str(raw.get("label", "")),
                    "variant": str(raw.get("variant", "")),
                    "intent_loss": _float(raw.get("intent_loss")),
                    "hard_fulfill": _float(raw.get("hard_fulfill")),
                    "safety_violation": _float(raw.get("safety_violation")),
                    "safety_first": _float(raw.get("safety_first")),
                    "rank_by_safety_first": int(_float(raw.get("rank_by_safety_first"), 0.0)),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _uncertainty_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_variant: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row["variant"])].append(row)
    out: list[dict[str, object]] = []
    for variant, _label in ORDER:
        selected = by_variant.get(variant, [])
        if not selected:
            continue
        for metric, metric_label, direction in METRICS:
            values = [float(row[metric]) for row in selected]
            n = len(values)
            avg = _mean(values)
            std = _sample_std(values)
            sem = std / math.sqrt(float(n)) if n > 0 else 0.0
            ci_half_width = _tcrit_975(n - 1) * sem if n > 1 else 0.0
            out.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "metric_label": metric_label,
                    "direction": direction,
                    "n_chunks": n,
                    "mean": avg,
                    "std": std,
                    "sem": sem,
                    "ci95_half_width": ci_half_width,
                    "ci95_low": avg - ci_half_width,
                    "ci95_high": avg + ci_half_width,
                }
            )
    return out


def _uncertainty_wide(long_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_variant: dict[str, dict[str, object]] = defaultdict(dict)
    for row in long_rows:
        variant = str(row["variant"])
        metric = str(row["metric"])
        by_variant[variant]["variant"] = variant
        by_variant[variant]["n_chunks"] = row["n_chunks"]
        for field_name in ("mean", "std", "sem", "ci95_half_width", "ci95_low", "ci95_high"):
            by_variant[variant][f"{metric}_{field_name}"] = row[field_name]
    return [by_variant[variant] for variant, _label in ORDER if variant in by_variant]


def _winner_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_chunk: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_chunk[int(row["chunk_id"])].append(row)
    per_chunk: list[dict[str, object]] = []
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for chunk_id in sorted(by_chunk):
        selected = by_chunk[chunk_id]
        line: dict[str, object] = {
            "chunk": selected[0]["chunk"],
            "chunk_id": chunk_id,
            "label": selected[0]["label"],
        }
        for metric, _metric_label, direction in METRICS:
            values = [(float(row[metric]), str(row["variant"])) for row in selected]
            best_value = max(values)[0] if direction == "higher" else min(values)[0]
            winners = sorted(
                variant
                for value, variant in values
                if abs(float(value) - float(best_value)) <= 1.0e-12
            )
            winner = winners[0] if winners else ""
            line[f"{metric}_winner"] = winner
            line[f"{metric}_winner_count_tied"] = len(winners)
            line[f"{metric}_best_value"] = best_value
            counts[(metric, winner)] += 1
        per_chunk.append(line)
    count_rows: list[dict[str, object]] = []
    for variant, _label in ORDER:
        line: dict[str, object] = {"variant": variant}
        for metric, _metric_label, _direction in METRICS:
            line[f"{metric}_wins"] = int(counts.get((metric, variant), 0))
        count_rows.append(line)
    return per_chunk, count_rows


def _plot_errorbars(
    *,
    figures_dir: Path,
    uncertainty: list[dict[str, object]],
    through: int,
    field_name: str,
    suffix: str,
    title_suffix: str,
) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    by_variant_metric = {
        (str(row["variant"]), str(row["metric"])): row
        for row in uncertainty
    }
    variants = [variant for variant, _label in ORDER if (variant, "safety_first") in by_variant_metric]
    labels = [label for variant, label in ORDER if variant in variants]
    x_positions = np.arange(len(variants))

    fig, axes = plt.subplots(2, 2, figsize=(17, 9.5))
    fig.suptitle(
        f"Experiment 1 Mean {title_suffix} Across {through} Micro Experiments",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )
    panels = [
        ("intent_loss", "Intent loss (lower is better)", "loss", False),
        ("hard_fulfill", "Hard KPI fulfillment (higher is better)", "ratio", True),
        ("safety_violation", "Safety violation (lower is better)", "rate", False),
        ("safety_first", "Safety-first score (lower is better)", "score", False),
    ]
    for ax, (metric, title, ylabel, higher) in zip(axes.ravel(), panels):
        means = [float(by_variant_metric[(variant, metric)]["mean"]) for variant in variants]
        errors = [float(by_variant_metric[(variant, metric)][field_name]) for variant in variants]
        bars = ax.bar(
            x_positions,
            means,
            yerr=errors,
            capsize=5,
            color=[COLORS[variant] for variant in variants],
            width=0.72,
            error_kw={"elinewidth": 1.2},
        )
        best_index = int(np.argmax(means) if higher else np.argmin(means))
        ax.set_title(title, fontsize=13, pad=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.grid(axis="y", alpha=0.25)
        for index, (bar, mean_value) in enumerate(zip(bars, means)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{mean_value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold" if index == best_index else "normal",
            )
        ymin, ymax = ax.get_ylim()
        if ymax > 0:
            ax.set_ylim(ymin, ymax * 1.15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = figures_dir / f"exp1_micro_v17_mean_{suffix}_through_chunk{through:03d}.png"
    fig.savefig(out_path, dpi=260)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path


def _plot_wins(
    *,
    figures_dir: Path,
    rows: list[dict[str, object]],
    win_counts: list[dict[str, object]],
    through: int,
) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    by_chunk = defaultdict(list)
    for row in rows:
        by_chunk[int(row["chunk_id"])].append(row)
    chunk_ids = sorted(by_chunk)
    variants = [variant for variant, _label in ORDER]
    labels = [label for _variant, label in ORDER]
    variant_to_index = {variant: index for index, variant in enumerate(variants)}
    color_list = [COLORS[variant] for variant in variants]

    safety_winners: list[str] = []
    cumulative = {variant: [] for variant in variants}
    running_counts = {variant: 0 for variant in variants}
    for chunk_id in chunk_ids:
        selected = by_chunk[chunk_id]
        winner = min(selected, key=lambda item: float(item["safety_first"]))
        winner_variant = str(winner["variant"])
        safety_winners.append(winner_variant)
        running_counts[winner_variant] += 1
        for variant in variants:
            cumulative[variant].append(running_counts[variant])

    fig = plt.figure(figsize=(17, 8.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.1], width_ratios=[2.2, 1.0])
    ax_line = fig.add_subplot(grid[0, 0])
    ax_bar = fig.add_subplot(grid[0, 1])
    ax_strip = fig.add_subplot(grid[1, :])

    for variant, label in ORDER:
        ax_line.plot(
            chunk_ids,
            cumulative[variant],
            marker="o",
            linewidth=3.0 if variant == "full_method" else 1.7,
            color=COLORS[variant],
            label=label,
        )
    ax_line.set_title("Cumulative per-chunk safety-first wins")
    ax_line.set_xlabel("chunk id")
    ax_line.set_ylabel("cumulative wins")
    ax_line.grid(alpha=0.25)
    ax_line.legend(fontsize=8, ncol=2, loc="upper left")

    win_by_variant = {str(row["variant"]): int(row.get("safety_first_wins", 0) or 0) for row in win_counts}
    bar_values = [win_by_variant.get(variant, 0) for variant in variants]
    bars = ax_bar.bar(labels, bar_values, color=color_list, width=0.72)
    ax_bar.set_title("Safety-first wins")
    ax_bar.set_ylabel("chunks")
    ax_bar.tick_params(axis="x", rotation=35)
    ax_bar.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, bar_values):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold" if value == max(bar_values) else "normal",
        )
    ymin, ymax = ax_bar.get_ylim()
    ax_bar.set_ylim(ymin, max(ymax, max(bar_values) + 1.0))

    strip_values = np.array([[variant_to_index[winner] for winner in safety_winners]], dtype=float)
    cmap = plt.matplotlib.colors.ListedColormap(color_list)
    ax_strip.imshow(strip_values, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(variants) - 0.5)
    ax_strip.set_title("Safety-first winner by chunk")
    ax_strip.set_yticks([])
    ax_strip.set_xticks(range(len(chunk_ids)))
    ax_strip.set_xticklabels([str(chunk_id) for chunk_id in chunk_ids])
    for index, winner in enumerate(safety_winners):
        ax_strip.text(
            index,
            0,
            labels[variant_to_index[winner]].split("-")[0],
            ha="center",
            va="center",
            fontsize=7,
            color="white" if winner in {"full_method", "fixed_plan", "no_memory_planner"} else "black",
            fontweight="bold" if winner == "full_method" else "normal",
        )
    fig.suptitle(f"Experiment 1 Per-Chunk Wins Through Chunk {through:03d}", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = figures_dir / f"exp1_micro_v17_per_chunk_wins_through_chunk{through:03d}.png"
    fig.savefig(out_path, dpi=260)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path


def main() -> None:
    args = _parse_args()
    root = Path(str(args.root)).resolve()
    through = int(args.through)
    figures_dir = root / "figures_micro_v17"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(root / "exp1_micro_v17_completed_chunks_summary.csv", through)
    uncertainty = _uncertainty_rows(rows)
    uncertainty_wide = _uncertainty_wide(uncertainty)
    per_chunk_winners, win_counts = _winner_rows(rows)

    _write_csv(root / f"exp1_micro_v17_uncertainty_summary_through_chunk{through:03d}.csv", uncertainty)
    _write_csv(root / f"exp1_micro_v17_uncertainty_summary_wide_through_chunk{through:03d}.csv", uncertainty_wide)
    _write_csv(root / f"exp1_micro_v17_per_chunk_winners_through_chunk{through:03d}.csv", per_chunk_winners)
    _write_csv(root / f"exp1_micro_v17_win_counts_through_chunk{through:03d}.csv", win_counts)

    mean_std = _plot_errorbars(
        figures_dir=figures_dir,
        uncertainty=uncertainty,
        through=through,
        field_name="std",
        suffix="std_errorbars",
        title_suffix="+/- Std",
    )
    mean_ci = _plot_errorbars(
        figures_dir=figures_dir,
        uncertainty=uncertainty,
        through=through,
        field_name="ci95_half_width",
        suffix="ci95_errorbars",
        title_suffix="with 95% CI",
    )
    wins = _plot_wins(
        figures_dir=figures_dir,
        rows=rows,
        win_counts=win_counts,
        through=through,
    )
    print(mean_std)
    print(mean_ci)
    print(wins)


if __name__ == "__main__":
    main()
