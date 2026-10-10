"""Generate figures for cross-scenario generalization artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any


project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from core.config_loader import load_yaml_file
from experiments.paper_eval_artifacts import save_bar_chart, write_json
from experiments.postprocess_generalization import postprocess_generalization_artifacts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot cross-scenario generalization outputs.")
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="outputs/paper_eval/oran_paper_eval_v1",
        help="Directory produced by run_cross_scenario_generalization.py.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Optional manifest path. Defaults to <experiment-dir>/scenario_manifest.csv.",
    )
    parser.add_argument(
        "--reference-scenario-id",
        type=str,
        default="",
        help="Optional reference scenario override.",
    )
    return parser.parse_args()


def _safe_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def plot_generalization(experiment_dir: Path, *, manifest_path: Path, reference_scenario_id: str) -> dict[str, Any]:
    config_snapshot = load_yaml_file(str(experiment_dir / "config_snapshot.yaml"))
    variant_names = list(config_snapshot.get("variants", []) or [])
    artifacts = postprocess_generalization_artifacts(
        experiment_dir=experiment_dir,
        manifest_path=manifest_path,
        reference_scenario_id=reference_scenario_id,
        variant_names=variant_names or None,
    )

    summary_rows = [
        row
        for row in _read_csv(experiment_dir / "aggregated_results.csv")
        if int(float(row.get("case_count", "0") or 0)) > 0
    ]
    macro_rows = [
        row
        for row in _read_csv(experiment_dir / "macro_average_results.csv")
        if int(float(row.get("intent_count", "0") or 0)) > 0
    ]
    if len(summary_rows) == 0:
        return {"case_count": 0, "figure_count": 0}

    labels = [row["variant_name"] for row in summary_rows]
    figures_dir = experiment_dir / "figures"
    save_bar_chart(
        output_path=figures_dir / "cross_scenario_intent_loss.png",
        labels=labels,
        series={
            "seen": [_safe_float(row.get("seen_mean_intent_loss_normalized")) for row in summary_rows],
            "unseen": [_safe_float(row.get("unseen_mean_intent_loss_normalized")) for row in summary_rows],
        },
        title="Cross-scenario normalized intent loss",
        ylabel="Normalized intent loss",
    )
    save_bar_chart(
        output_path=figures_dir / "cross_scenario_safety_stability.png",
        labels=labels,
        series={
            "safety violation": [_safe_float(row.get("all_mean_safety_violation_rate")) for row in summary_rows],
            "switching cost": [_safe_float(row.get("all_mean_xapp_switching_cost_mean")) for row in summary_rows],
        },
        title="Safety and switching behavior across scenarios",
        ylabel="Rate / cost",
    )
    save_bar_chart(
        output_path=figures_dir / "cross_scenario_throughput.png",
        labels=labels,
        series={
            "seen": [_safe_float(row.get("seen_mean_mean_observable_throughput_mbps")) for row in summary_rows],
            "unseen": [_safe_float(row.get("unseen_mean_mean_observable_throughput_mbps")) for row in summary_rows],
        },
        title="Observable throughput across scenario splits",
        ylabel="Throughput (Mbps)",
    )
    if len(macro_rows) > 0:
        macro_labels = [row["variant_name"] for row in macro_rows]
        save_bar_chart(
            output_path=figures_dir / "macro_average_intent_loss.png",
            labels=macro_labels,
            series={
                "macro avg": [
                    _safe_float(row.get("macro_mean_intent_loss_normalized"))
                    for row in macro_rows
                ],
            },
            title="Macro-average normalized intent loss across intents",
            ylabel="Normalized intent loss",
        )
    payload = {
        "case_count": len(artifacts["case_records"]),
        "behavior_row_count": len(artifacts["behavior_rows"]),
        "summary_row_count": len(summary_rows),
        "figures": [
            str((figures_dir / "cross_scenario_intent_loss.png").relative_to(project_root)).replace("\\", "/"),
            str((figures_dir / "cross_scenario_safety_stability.png").relative_to(project_root)).replace("\\", "/"),
            str((figures_dir / "cross_scenario_throughput.png").relative_to(project_root)).replace("\\", "/"),
            str((figures_dir / "macro_average_intent_loss.png").relative_to(project_root)).replace("\\", "/"),
        ],
    }
    write_json(experiment_dir / "figures" / "plot_summary.json", payload)
    return payload


def main() -> None:
    args = _parse_args()
    experiment_dir = (project_root / str(args.experiment_dir)).resolve()
    manifest_path = (
        (project_root / str(args.manifest)).resolve()
        if str(args.manifest).strip() != ""
        else experiment_dir / "scenario_manifest.csv"
    )
    config_snapshot = load_yaml_file(str(experiment_dir / "config_snapshot.yaml"))
    reference = str(args.reference_scenario_id or config_snapshot.get("reference_scenario_id", ""))
    payload = plot_generalization(
        experiment_dir=experiment_dir,
        manifest_path=manifest_path,
        reference_scenario_id=reference,
    )
    print(f"Plotted {payload.get('case_count', 0)} completed case(s) under {experiment_dir / 'figures'}")


if __name__ == "__main__":
    main()
