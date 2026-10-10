"""Summarize paired Experiment 1 task-seed cells from split-run headlines."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import statistics
from typing import Any


METRICS = {
    "safety_first_gain": ("transfer_safety_first_score", "other_minus_main"),
    "intent_loss_gain": ("transfer_mean_intent_loss_normalized", "other_minus_main"),
    "safety_gain": ("transfer_mean_safety_violation_rate", "other_minus_main"),
    "hard_kpi_gain": ("transfer_mean_hard_target_fulfillment", "main_minus_other"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute paired task-seed win rates and bootstrap CIs for Experiment 1."
    )
    parser.add_argument("--manifest", required=True, help="Split task_manifest.csv path.")
    parser.add_argument(
        "--additional-manifest",
        action="append",
        default=[],
        help=(
            "Additional split task_manifest.csv path. Variant rows from matching task/seed cells "
            "are merged, keeping the first copy of repeated reference methods."
        ),
    )
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--main-method", default="full_method")
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        help="Baseline variant name. Repeat to select explicit baselines.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260521)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _float(row: dict[str, str], field_name: str) -> float:
    return float(row.get(field_name, 0.0) or 0.0)


def _headline_rows(output_dir: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(output_dir / "transfer_headline_results.csv")
    return {
        str(row.get("variant_name", "") or ""): row
        for row in rows
        if str(row.get("variant_name", "") or "").strip() != ""
    }


def _paired_cells(manifest_paths: list[Path]) -> list[dict[str, Any]]:
    cells_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for manifest_path in manifest_paths:
        for manifest_row in _read_csv(manifest_path):
            output_dir = Path(str(manifest_row.get("output_dir", "") or "")).resolve()
            headline_path = output_dir / "transfer_headline_results.csv"
            if not headline_path.exists():
                continue
            task_id = str(manifest_row.get("task_id", "") or "")
            seed = int(float(manifest_row.get("seed", 0) or 0))
            key = (task_id, seed)
            cell = cells_by_key.setdefault(
                key,
                {
                    "task_id": task_id,
                    "seed": seed,
                    "rows": {},
                },
            )
            cell_rows = dict(cell["rows"])
            for variant_name, row in _headline_rows(output_dir).items():
                cell_rows.setdefault(variant_name, row)
            cell["rows"] = cell_rows
    return [cells_by_key[key] for key in sorted(cells_by_key)]


def _bootstrap_ci(
    values: list[float],
    *,
    samples: int,
    rng: random.Random,
) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    if len(values) == 1 or samples <= 0:
        return float(values[0]), float(values[0])
    means = [
        sum(values[rng.randrange(len(values))] for _ in range(len(values))) / len(values)
        for _ in range(int(samples))
    ]
    means.sort()
    lower_index = max(0, int(0.025 * len(means)))
    upper_index = min(len(means) - 1, int(0.975 * len(means)))
    return float(means[lower_index]), float(means[upper_index])


def _gain(main_row: dict[str, str], other_row: dict[str, str], *, field_name: str, direction: str) -> float:
    if direction == "main_minus_other":
        return _float(main_row, field_name) - _float(other_row, field_name)
    return _float(other_row, field_name) - _float(main_row, field_name)


def _selected_baselines(
    cells: list[dict[str, Any]],
    *,
    main_method: str,
    requested: list[str],
) -> list[str]:
    requested_names = [str(name).strip() for name in requested if str(name).strip() != ""]
    if requested_names:
        return list(dict.fromkeys(requested_names))
    variants = sorted(
        {
            variant_name
            for cell in cells
            for variant_name in dict(cell["rows"]).keys()
            if str(variant_name) != str(main_method)
        }
    )
    return variants


def _summary_rows(
    cells: list[dict[str, Any]],
    *,
    main_method: str,
    baselines: list[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for baseline_index, baseline in enumerate(baselines):
        pairs = [
            (dict(cell["rows"])[main_method], dict(cell["rows"])[baseline])
            for cell in cells
            if main_method in dict(cell["rows"]) and baseline in dict(cell["rows"])
        ]
        summary: dict[str, Any] = {
            "main_method": str(main_method),
            "baseline": str(baseline),
            "paired_cell_count": int(len(pairs)),
            "safety_first_win_count": sum(
                _float(main_row, "transfer_safety_first_score")
                < _float(other_row, "transfer_safety_first_score")
                for main_row, other_row in pairs
            ),
            "intent_loss_win_count": sum(
                _float(main_row, "transfer_mean_intent_loss_normalized")
                < _float(other_row, "transfer_mean_intent_loss_normalized")
                for main_row, other_row in pairs
            ),
            "hard_kpi_win_count": sum(
                _float(main_row, "transfer_mean_hard_target_fulfillment")
                > _float(other_row, "transfer_mean_hard_target_fulfillment")
                for main_row, other_row in pairs
            ),
            "lower_safety_count": sum(
                _float(main_row, "transfer_mean_safety_violation_rate")
                < _float(other_row, "transfer_mean_safety_violation_rate")
                for main_row, other_row in pairs
            ),
            "safety_tie_count": sum(
                abs(
                    _float(main_row, "transfer_mean_safety_violation_rate")
                    - _float(other_row, "transfer_mean_safety_violation_rate")
                )
                <= 1e-12
                for main_row, other_row in pairs
            ),
        }
        for metric_index, (metric_name, (field_name, direction)) in enumerate(METRICS.items()):
            gains = [
                _gain(main_row, other_row, field_name=field_name, direction=direction)
                for main_row, other_row in pairs
            ]
            rng = random.Random(int(bootstrap_seed) + 100 * baseline_index + metric_index)
            lower, upper = _bootstrap_ci(gains, samples=bootstrap_samples, rng=rng)
            summary[f"{metric_name}_mean"] = statistics.mean(gains) if gains else 0.0
            summary[f"{metric_name}_ci95_low"] = lower
            summary[f"{metric_name}_ci95_high"] = upper
        summaries.append(summary)
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    field_names: list[str] = []
    for row in rows:
        for field_name in row.keys():
            if field_name not in field_names:
                field_names.append(field_name)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    manifest_paths = [
        Path(str(raw_path)).resolve()
        for raw_path in [args.manifest, *list(args.additional_manifest or [])]
    ]
    cells = _paired_cells(manifest_paths)
    baselines = _selected_baselines(
        cells,
        main_method=str(args.main_method),
        requested=list(args.baseline or []),
    )
    summaries = _summary_rows(
        cells,
        main_method=str(args.main_method),
        baselines=baselines,
        bootstrap_samples=max(0, int(args.bootstrap_samples)),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    output_path = Path(str(args.output)).resolve()
    _write_csv(output_path, summaries)
    print(f"Wrote {len(summaries)} paired Experiment 1 summaries from {len(cells)} cells to {output_path}.")


if __name__ == "__main__":
    main()
