"""Artifact helpers for paper evaluation scripts.

The helpers intentionally keep plotting dependencies narrow and local to the
experiment scripts.  They write both machine-readable CSV/JSON data and
publication draft figures under each run directory.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if len(rows) == 0:
        path.write_text("", encoding="utf-8")
        return
    field_names: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in field_names:
                field_names.append(key)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, ensure_ascii=False)


def mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if len(items) == 0:
        return 0.0
    return float(sum(items) / len(items))


def percentile(values: Iterable[float], q: float) -> float:
    items = sorted(float(value) for value in values)
    if len(items) == 0:
        return 0.0
    if len(items) == 1:
        return float(items[0])
    position = (len(items) - 1) * max(0.0, min(1.0, float(q)))
    lower = int(position)
    upper = min(lower + 1, len(items) - 1)
    weight = position - lower
    return float(items[lower] * (1.0 - weight) + items[upper] * weight)


def safe_div(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) <= 1e-12:
        return 0.0
    return float(numerator) / float(denominator)


def configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    return plt


def save_bar_chart(
    *,
    output_path: Path,
    labels: list[str],
    series: dict[str, list[float]],
    title: str,
    ylabel: str,
    rotate_labels: bool = True,
) -> None:
    plt = configure_matplotlib()
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.7 * len(labels)), 3.6))
    x_positions = list(range(len(labels)))
    series_items = list(series.items())
    width = 0.75 / max(1, len(series_items))
    for idx, (name, values) in enumerate(series_items):
        offset = (idx - (len(series_items) - 1) / 2.0) * width
        ax.bar([x + offset for x in x_positions], values, width=width, label=name)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=30 if rotate_labels else 0, ha="right" if rotate_labels else "center")
    if len(series_items) > 1:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_line_chart(
    *,
    output_path: Path,
    x_values: list[float],
    series: dict[str, list[float]],
    title: str,
    ylabel: str,
    xlabel: str = "slot",
    event_lines: dict[float, str] | None = None,
) -> None:
    plt = configure_matplotlib()
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for name, values in series.items():
        ax.plot(x_values, values, label=name, linewidth=1.7)
    for x_value, label in dict(event_lines or {}).items():
        ax.axvline(float(x_value), linestyle="--", linewidth=0.9, alpha=0.55)
        ax.text(float(x_value), ax.get_ylim()[1], label, rotation=90, va="top", ha="right", fontsize=7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(series) > 1:
        ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
