"""Run Experiment 3-C contract and deployment robustness conditions."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
from typing import Any


project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from experiments.experiment3_simulator_bridge import (  # noqa: E402
    BRIDGE_EVENT_NAMES,
    BRIDGE_ROBUSTNESS_PROFILES,
    _aggregate,
    run_experiment3_simulator_bridge,
)
from experiments.paper_eval_artifacts import mean, percentile, write_csv, write_json  # noqa: E402


FULL_METHOD = "full_contract_constrained_ddpg"
PROJECTION_BASELINE = "ddpg_projection_only"
ROBUSTNESS_METRICS = {
    "recovery_time_slots": "lower",
    "auv": "lower",
    "mean_queue_bytes": "lower",
    "mean_urllc_queue_bytes": "lower",
    "mean_throughput_mbps": "higher",
    "mean_delivered_throughput_mbps": "higher",
    "latency_violation_rate": "lower",
    "urllc_latency_violation_rate": "lower",
    "raw_action_violation_rate": "lower",
    "executed_action_violation_rate": "lower",
    "guardrail_violation_rate": "lower",
}


def _bootstrap_ci(values: list[float], *, seed: int, samples: int) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    draws = [
        mean(values[rng.randrange(len(values))] for _ in range(len(values)))
        for _ in range(max(100, int(samples)))
    ]
    return percentile(draws, 0.025), percentile(draws, 0.975)


def _condition_degradation(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    keyed = {
        (
            str(row["robustness_condition"]),
            str(row["method"]),
            str(row["event_name"]),
            str(row["severity"]),
            int(row["seed"]),
        ): row
        for row in rows
    }
    cells = sorted(
        {
            (str(row["method"]), str(row["event_name"]), str(row["severity"]), int(row["seed"]))
            for row in rows
            if str(row["robustness_condition"]) == "nominal"
        }
    )
    conditions = sorted(
        {str(row["robustness_condition"]) for row in rows}
        - {"nominal"}
    )
    out: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(conditions):
        for metric_index, (metric, direction) in enumerate(ROBUSTNESS_METRICS.items()):
            for method in sorted({cell[0] for cell in cells}):
                degradations: list[float] = []
                nominal_values: list[float] = []
                stress_values: list[float] = []
                for cell_method, event_name, severity, seed in cells:
                    if cell_method != method:
                        continue
                    nominal = keyed.get(("nominal", method, event_name, severity, seed))
                    stress = keyed.get((condition, method, event_name, severity, seed))
                    if nominal is None or stress is None:
                        continue
                    nominal_value = float(nominal[metric])
                    stress_value = float(stress[metric])
                    degradation = (
                        stress_value - nominal_value
                        if direction == "lower"
                        else nominal_value - stress_value
                    )
                    nominal_values.append(nominal_value)
                    stress_values.append(stress_value)
                    degradations.append(degradation)
                ci_low, ci_high = _bootstrap_ci(
                    degradations,
                    seed=3101 + 131 * condition_index + 17 * metric_index,
                    samples=bootstrap_samples,
                )
                out.append(
                    {
                        "robustness_condition": condition,
                        "method": method,
                        "metric": metric,
                        "preferred_direction": direction,
                        "paired_cell_count": len(degradations),
                        "nominal_mean": round(mean(nominal_values), 6),
                        "stress_mean": round(mean(stress_values), 6),
                        "mean_degradation": round(mean(degradations), 6),
                        "ci95_low": round(ci_low, 6),
                        "ci95_high": round(ci_high, 6),
                        "stress_worse_rate": round(
                            mean(float(value > 1e-12) for value in degradations), 6
                        ),
                    }
                )
    return out


def _method_comparison(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    keyed = {
        (
            str(row["robustness_condition"]),
            str(row["method"]),
            str(row["event_name"]),
            str(row["severity"]),
            int(row["seed"]),
        ): row
        for row in rows
    }
    cells = sorted(
        {
            (
                str(row["robustness_condition"]),
                str(row["event_name"]),
                str(row["severity"]),
                int(row["seed"]),
            )
            for row in rows
            if str(row["method"]) == FULL_METHOD
        }
    )
    out: list[dict[str, Any]] = []
    for metric_index, (metric, direction) in enumerate(ROBUSTNESS_METRICS.items()):
        for condition_index, condition in enumerate(sorted({cell[0] for cell in cells})):
            advantages: list[float] = []
            full_values: list[float] = []
            baseline_values: list[float] = []
            for cell_condition, event_name, severity, seed in cells:
                if cell_condition != condition:
                    continue
                full = keyed.get((condition, FULL_METHOD, event_name, severity, seed))
                baseline = keyed.get((condition, PROJECTION_BASELINE, event_name, severity, seed))
                if full is None or baseline is None:
                    continue
                full_value = float(full[metric])
                baseline_value = float(baseline[metric])
                advantage = (
                    full_value - baseline_value
                    if direction == "higher"
                    else baseline_value - full_value
                )
                full_values.append(full_value)
                baseline_values.append(baseline_value)
                advantages.append(advantage)
            ci_low, ci_high = _bootstrap_ci(
                advantages,
                seed=4101 + 149 * condition_index + 19 * metric_index,
                samples=bootstrap_samples,
            )
            out.append(
                {
                    "robustness_condition": condition,
                    "metric": metric,
                    "preferred_direction": direction,
                    "paired_cell_count": len(advantages),
                    "full_mean": round(mean(full_values), 6),
                    "projection_only_mean": round(mean(baseline_values), 6),
                    "mean_full_advantage": round(mean(advantages), 6),
                    "ci95_low": round(ci_low, 6),
                    "ci95_high": round(ci_high, 6),
                    "full_win_rate": round(
                        mean(float(value > 1e-12) for value in advantages), 6
                    ),
                }
            )
    return out


def run_contract_robustness(
    output_dir: Path,
    *,
    conditions: list[str],
    seeds: list[int],
    training_seeds: list[int],
    episode_length: int,
    training_episodes: int,
    events: list[str],
    severities: list[str],
    methods: list[str],
    bootstrap_samples: int,
    write_time_series: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_time_series: list[dict[str, Any]] = []
    all_episode_metrics: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    for condition in conditions:
        result = run_experiment3_simulator_bridge(
            output_dir / "conditions" / condition,
            seeds=seeds,
            training_seeds=training_seeds,
            episode_length=episode_length,
            training_episodes=training_episodes,
            method_names=methods,
            event_names=events,
            severity_names=severities,
            robustness_condition=condition,
            write_time_series=write_time_series,
        )
        all_time_series.extend(result["time_series"])
        all_episode_metrics.extend(result["episode_metrics"])
        all_diagnostics.extend(
            {"robustness_condition": condition, **row}
            for row in result["training_diagnostics"]
        )

    condition_summary = _aggregate(all_episode_metrics, ("robustness_condition", "method"))
    condition_event_summary = _aggregate(
        all_episode_metrics,
        ("robustness_condition", "event_name", "severity", "method"),
    )
    degradation = _condition_degradation(
        all_episode_metrics,
        bootstrap_samples=bootstrap_samples,
    )
    method_comparison = _method_comparison(
        all_episode_metrics,
        bootstrap_samples=bootstrap_samples,
    )
    if write_time_series:
        write_csv(output_dir / "time_series.csv", all_time_series)
    write_csv(output_dir / "episode_metrics.csv", all_episode_metrics)
    write_csv(output_dir / "condition_summary.csv", condition_summary)
    write_csv(output_dir / "condition_event_summary.csv", condition_event_summary)
    write_csv(output_dir / "condition_degradation_paired.csv", degradation)
    write_csv(output_dir / "method_comparison_paired.csv", method_comparison)
    write_csv(output_dir / "training_diagnostics.csv", all_diagnostics)
    run_plan = {
        "experiment": "experiment3_contract_robustness",
        "conditions": {
            name: BRIDGE_ROBUSTNESS_PROFILES[name]
            for name in conditions
        },
        "seeds": seeds,
        "training_seeds": training_seeds,
        "episode_length": episode_length,
        "training_episodes": training_episodes,
        "events": events,
        "severities": severities,
        "methods": methods,
        "planned_episode_count": len(conditions) * len(seeds) * len(events) * len(severities) * len(methods),
        "paired_bootstrap_samples": bootstrap_samples,
    }
    write_json(output_dir / "run_plan.json", run_plan)
    write_json(
        output_dir / "summary.json",
        {
            "run_plan": run_plan,
            "condition_summary": condition_summary,
            "condition_degradation_paired": degradation,
            "method_comparison_paired": method_comparison,
        },
    )
    return {
        "episode_metrics": all_episode_metrics,
        "condition_summary": condition_summary,
        "condition_degradation_paired": degradation,
        "method_comparison_paired": method_comparison,
        "run_plan": run_plan,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_eval/experiment3_contract_robustness_5seed_v1",
    )
    parser.add_argument("--conditions", nargs="+", default=list(BRIDGE_ROBUSTNESS_PROFILES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--training-seeds",
        nargs="+",
        type=int,
        default=[1001, 1002, 1003, 1004, 1005],
    )
    parser.add_argument("--episode-length", type=int, default=120)
    parser.add_argument("--training-episodes", type=int, default=40)
    parser.add_argument("--events", nargs="+", default=list(BRIDGE_EVENT_NAMES))
    parser.add_argument("--severities", nargs="+", default=["high"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[FULL_METHOD, PROJECTION_BASELINE],
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--no-time-series", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    unknown_conditions = sorted(set(args.conditions) - set(BRIDGE_ROBUSTNESS_PROFILES))
    if unknown_conditions:
        raise ValueError(f"Unknown robustness conditions: {unknown_conditions}")
    output_dir = (project_root / str(args.output_dir)).resolve()
    result = run_contract_robustness(
        output_dir,
        conditions=[str(name) for name in args.conditions],
        seeds=[int(seed) for seed in args.seeds],
        training_seeds=[int(seed) for seed in args.training_seeds],
        episode_length=int(args.episode_length),
        training_episodes=int(args.training_episodes),
        events=[str(name) for name in args.events],
        severities=[str(name) for name in args.severities],
        methods=[str(name) for name in args.methods],
        bootstrap_samples=int(args.bootstrap_samples),
        write_time_series=not bool(args.no_time_series),
    )
    print(f"Generated Experiment 3-C contract robustness under {output_dir}")
    print(f"- episodes: {len(result['episode_metrics'])}")
    print(f"- paired degradation: {output_dir / 'condition_degradation_paired.csv'}")
    print(f"- method comparison: {output_dir / 'method_comparison_paired.csv'}")


if __name__ == "__main__":
    main()
