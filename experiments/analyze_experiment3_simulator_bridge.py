"""Build paired confidence intervals for Experiment 3-B simulator results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import sys
from typing import Any


project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.paper_eval_artifacts import mean, percentile, write_csv


FULL_METHOD = "full_contract_constrained_ddpg"
METHOD_LABELS = {
    "no_near_rt_tuning": "No Near-RT tuning",
    "rule_based_bounded_tuner": "Rule-based bounded",
    "ddpg_projection_only": "DDPG + projection only",
    "full_contract_constrained_ddpg": "Full CC-DDPG",
}
METRICS = {
    "recovery_time_slots": "lower",
    "auv": "lower",
    "mean_queue_bytes": "lower",
    "mean_urllc_queue_bytes": "lower",
    "mean_throughput_mbps": "higher",
    "mean_delivered_throughput_mbps": "higher",
    "mean_estimated_queue_delay_ms": "lower",
    "latency_violation_rate": "lower",
    "urllc_latency_violation_rate": "lower",
    "raw_action_violation_rate": "lower",
    "executed_action_violation_rate": "lower",
    "mean_policy_projection_distance": "lower",
    "mean_projection_distance": "lower",
    "fallback_frequency": "lower",
    "guardrail_violation_rate": "lower",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _bootstrap_ci(values: list[float], *, seed: int, samples: int) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    draws = [
        mean(values[rng.randrange(len(values))] for _ in range(len(values)))
        for _ in range(max(100, int(samples)))
    ]
    return percentile(draws, 0.025), percentile(draws, 0.975)


def analyze_bridge(
    *,
    input_path: Path,
    output_path: Path,
    bootstrap_samples: int = 5000,
) -> list[dict[str, Any]]:
    rows = _read_csv(input_path)
    keyed = {
        (
            row["event_name"],
            row["severity"],
            int(row["seed"]),
            row["method"],
        ): row
        for row in rows
    }
    cells = sorted(
        {
            (row["event_name"], row["severity"], int(row["seed"]))
            for row in rows
            if row["method"] == FULL_METHOD
        }
    )
    baselines = sorted(
        {row["method"] for row in rows if row["method"] != FULL_METHOD}
    )
    results: list[dict[str, Any]] = []
    for baseline_index, baseline in enumerate(baselines):
        for metric_index, (metric, direction) in enumerate(METRICS.items()):
            advantages: list[float] = []
            full_values: list[float] = []
            baseline_values: list[float] = []
            wins = 0
            ties = 0
            for event_name, severity, seed in cells:
                full_row = keyed.get((event_name, severity, seed, FULL_METHOD))
                baseline_row = keyed.get((event_name, severity, seed, baseline))
                if full_row is None or baseline_row is None:
                    continue
                full_value = float(full_row[metric])
                baseline_value = float(baseline_row[metric])
                advantage = (
                    full_value - baseline_value
                    if direction == "higher"
                    else baseline_value - full_value
                )
                full_values.append(full_value)
                baseline_values.append(baseline_value)
                advantages.append(advantage)
                if advantage > 1e-12:
                    wins += 1
                elif abs(advantage) <= 1e-12:
                    ties += 1
            ci_low, ci_high = _bootstrap_ci(
                advantages,
                seed=1701 + 97 * baseline_index + metric_index,
                samples=bootstrap_samples,
            )
            count = len(advantages)
            results.append(
                {
                    "full_method": FULL_METHOD,
                    "full_method_label": METHOD_LABELS[FULL_METHOD],
                    "baseline": baseline,
                    "baseline_label": METHOD_LABELS.get(baseline, baseline),
                    "metric": metric,
                    "preferred_direction": direction,
                    "paired_cell_count": count,
                    "full_mean": round(mean(full_values), 6),
                    "baseline_mean": round(mean(baseline_values), 6),
                    "mean_full_advantage": round(mean(advantages), 6),
                    "ci95_low": round(ci_low, 6),
                    "ci95_high": round(ci_high, 6),
                    "full_win_rate": round(float(wins) / float(count), 6) if count else 0.0,
                    "tie_rate": round(float(ties) / float(count), 6) if count else 0.0,
                    "ci_excludes_zero_in_full_favor": int(ci_low > 0.0),
                }
            )
    write_csv(output_path, results)
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="outputs/paper_eval/experiment3_simulator_bridge_5seed_v1/episode_metrics.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/paper_eval/experiment3_simulator_bridge_5seed_v1/paired_comparison.csv",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = (project_root / args.input).resolve()
    output_path = (project_root / args.output).resolve()
    results = analyze_bridge(
        input_path=input_path,
        output_path=output_path,
        bootstrap_samples=int(args.bootstrap_samples),
    )
    print(f"Wrote {len(results)} paired comparison rows to {output_path}")


if __name__ == "__main__":
    main()
