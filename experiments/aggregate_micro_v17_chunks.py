"""Aggregate Experiment 1 micro-v17 chunk headline CSVs and plot summary figures."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import re


VARIANT_ORDER = [
    "full_method",
    "success_only_memory",
    "no_memory_planner",
    "orchestran_like_resource_aware",
    "intent_hrl_like",
    "oreo_like_service_graph",
    "fixed_plan",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate exp1 micro-v17 chunks.")
    parser.add_argument("--root", default="E:/oran_slot_sim_outputs/paper_eval")
    parser.add_argument("--through", type=int, default=0)
    return parser.parse_args()


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _chunk_number(path: Path) -> int:
    match = re.search(r"chunk(\d{3})", path.name)
    return int(match.group(1)) if match else 0


def _chunk_label(path: Path, chunk_number: int) -> str:
    known = {
        1: "mobility_fast",
        2: "interference_fast",
        3: "stale_cumulative_fast",
        4: "hotspot_cumulative_fast",
        5: "blockage_cumulative_fast",
        6: "energy_cumulative_fast",
        7: "mixed_slice_cumulative_fast",
        8: "premium_urllc_cumulative_fast",
        9: "handover_video_cumulative_fast",
        10: "interference_urllc_cumulative_fast",
        11: "recovery_handover_cumulative_v1",
        12: "recovery_interference_urllc_v1",
        13: "recovery_handover_blockage_v1",
        14: "stale_memory_recovery_v1",
        15: "premium_policy_recovery_v1",
        16: "conflicting_energy_recovery_v1",
        17: "audit_blockage_repair_v1",
        18: "load_balance_stability_v1",
    }
    return known.get(int(chunk_number), path.stem)


def _select_latest_headlines(root: Path, through: int) -> list[Path]:
    grouped: dict[int, list[Path]] = defaultdict(list)
    for path in root.glob("exp1_micro_v17_chunk*_combined_headline*.csv"):
        chunk_number = _chunk_number(path)
        if chunk_number <= 0:
            continue
        if through > 0 and chunk_number > through:
            continue
        grouped[chunk_number].append(path)
    selected = []
    for chunk_number, paths in grouped.items():
        selected.append(max(paths, key=lambda item: item.stat().st_mtime))
    return sorted(selected, key=_chunk_number)


def _read_chunk_rows(path: Path) -> list[dict[str, object]]:
    chunk_number = _chunk_number(path)
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        for raw in csv.DictReader(input_file):
            raw = {
                str(key or "").strip().strip('"'): value
                for key, value in dict(raw or {}).items()
            }
            variant = str(raw.get("variant", raw.get("variant_name", "")) or "")
            if variant == "":
                continue
            rows.append(
                {
                    "chunk": f"chunk{chunk_number:03d}",
                    "chunk_id": chunk_number,
                    "label": _chunk_label(path, chunk_number),
                    "variant": variant,
                    "intent_loss": _float(raw.get("intent_loss", raw.get("transfer_mean_intent_loss_normalized"))),
                    "hard_fulfill": _float(raw.get("hard_fulfill", raw.get("transfer_mean_hard_target_fulfillment"))),
                    "safety_violation": _float(raw.get("safety_violation", raw.get("transfer_mean_safety_violation_rate"))),
                    "safety_first": _float(raw.get("safety_first", raw.get("transfer_safety_first_score"))),
                    "result_file": str(path),
                }
            )
    rows.sort(key=lambda item: float(item["safety_first"]))
    for index, row in enumerate(rows, start=1):
        row["rank_by_safety_first"] = index
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


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_variant: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row["variant"])].append(row)
    out = []
    for variant in VARIANT_ORDER:
        selected = by_variant.get(variant, [])
        if not selected:
            continue
        n = float(len(selected))
        out.append(
            {
                "variant": variant,
                "chunks": len(selected),
                "mean_intent_loss": sum(float(r["intent_loss"]) for r in selected) / n,
                "mean_hard_fulfill": sum(float(r["hard_fulfill"]) for r in selected) / n,
                "mean_safety_violation": sum(float(r["safety_violation"]) for r in selected) / n,
                "mean_safety_first": sum(float(r["safety_first"]) for r in selected) / n,
                "safety_first_wins": sum(1 for r in selected if int(r["rank_by_safety_first"]) == 1),
            }
        )
    out.sort(key=lambda item: float(item["mean_safety_first"]))
    return out


def _wide(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_chunk: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_chunk[int(row["chunk_id"])].append(row)
    out = []
    for chunk_id in sorted(by_chunk):
        selected = by_chunk[chunk_id]
        line = {"chunk": f"chunk{chunk_id:03d}", "chunk_id": chunk_id, "label": selected[0]["label"]}
        for metric in ("intent_loss", "hard_fulfill", "safety_violation", "safety_first", "rank_by_safety_first"):
            for row in selected:
                line[f"{metric}__{row['variant']}"] = row[metric]
        out.append(line)
    return out


def _plot(root: Path, macro_rows: list[dict[str, object]], wide_rows: list[dict[str, object]], through: int) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    figures_dir = root / "figures_micro_v17"
    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(row["variant"]) for row in macro_rows]
    color = ["#0f766e" if label == "full_method" else "#6b7280" for label in labels]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, metric, title in [
        (axes[0, 0], "mean_intent_loss", "Mean intent loss"),
        (axes[0, 1], "mean_safety_violation", "Mean safety violation"),
        (axes[1, 0], "mean_safety_first", "Mean safety-first score"),
        (axes[1, 1], "safety_first_wins", "Safety-first wins"),
    ]:
        values = [float(row[metric]) for row in macro_rows]
        ax.bar(labels, values, color=color)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / f"exp1_micro_v17_metric_dashboard_through_chunk{through:03d}.png", dpi=220)
    fig.savefig(figures_dir / "exp1_micro_v17_metric_dashboard_latest.png", dpi=220)
    plt.close(fig)

    full_margins = []
    chunk_labels = []
    for row in wide_rows:
        full_score = _float(row.get("safety_first__full_method"))
        competitors = [
            _float(value)
            for key, value in row.items()
            if key.startswith("safety_first__") and key != "safety_first__full_method"
        ]
        if competitors:
            full_margins.append(min(competitors) - full_score)
            chunk_labels.append(str(row["chunk"]))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.bar(chunk_labels, full_margins, color=["#0f766e" if value > 0 else "#b91c1c" for value in full_margins])
    ax.set_title("Full-method safety-first advantage by chunk")
    ax.set_ylabel("Best competitor score - full score")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "exp1_micro_v17_full_advantage_by_chunk.png", dpi=220)
    plt.close(fig)

    heatmap_values = []
    chunk_names = []
    heatmap_variants = VARIANT_ORDER
    for row in wide_rows:
        chunk_names.append(str(row["chunk"]))
        heatmap_values.append([
            _float(row.get(f"rank_by_safety_first__{variant}"), np.nan)
            for variant in heatmap_variants
        ])
    fig, ax = plt.subplots(figsize=(12, 5.5))
    image = ax.imshow(np.array(heatmap_values, dtype=float), aspect="auto", cmap="viridis_r", vmin=1, vmax=7)
    ax.set_xticks(range(len(heatmap_variants)))
    ax.set_xticklabels(heatmap_variants, rotation=25, ha="right")
    ax.set_yticks(range(len(chunk_names)))
    ax.set_yticklabels(chunk_names)
    ax.set_title("Safety-first rank by chunk")
    fig.colorbar(image, ax=ax, label="rank")
    fig.tight_layout()
    fig.savefig(figures_dir / "exp1_micro_v17_safety_first_rank_heatmap.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    root = Path(str(args.root)).resolve()
    selected_files = _select_latest_headlines(root, int(args.through or 0))
    if not selected_files:
        raise FileNotFoundError(f"No chunk headline files found under {root}")
    through = max(_chunk_number(path) for path in selected_files)
    rows: list[dict[str, object]] = []
    for path in selected_files:
        rows.extend(_read_chunk_rows(path))
    macro_rows = _aggregate(rows)
    wide_rows = _wide(rows)
    _write_csv(root / "exp1_micro_v17_completed_chunks_summary.csv", rows)
    _write_csv(root / f"exp1_micro_v17_macro_summary_through_chunk{through:03d}.csv", macro_rows)
    _write_csv(root / f"exp1_micro_v17_aggregated_metrics_through_chunk{through:03d}.csv", macro_rows)
    _write_csv(root / "exp1_micro_v17_aggregated_metrics_latest.csv", macro_rows)
    _write_csv(root / "exp1_micro_v17_per_chunk_metrics_wide.csv", wide_rows)
    _plot(root, macro_rows, wide_rows, through)
    print(f"Aggregated {len(selected_files)} chunks through chunk{through:03d}.")
    print(root / f"exp1_micro_v17_aggregated_metrics_through_chunk{through:03d}.csv")


if __name__ == "__main__":
    main()
