"""Build candidate-level paired confidence intervals for Experiment 2."""

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


from experiments.paper_eval_artifacts import mean, percentile, write_csv  # noqa: E402


FULL_METHOD = "full_repair"
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
METRICS = {
    "final_feasible": "higher",
    "repair_success": "higher",
    "contract_success": "higher",
    "utility_preserving_executable": "higher",
    "protected_kpi_coverage": "higher",
    "intent_coverage": "higher",
    "predicted_intent_loss": "lower",
    "hard_conflict_escape": "lower",
    "resource_overflow_escape": "lower",
    "policy_violation_residual": "lower",
    "candidate_rejected": "lower",
    "near_rt_action_blocked_rate": "lower",
    "average_repair_distance": "lower",
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


def analyze_candidate_pairs(
    *,
    input_path: Path,
    output_path: Path,
    bootstrap_samples: int = 5000,
) -> list[dict[str, Any]]:
    rows = _read_csv(input_path)
    keyed = {(row["candidate_id"], row["mode"]): row for row in rows}
    candidate_ids = sorted(
        row["candidate_id"]
        for row in rows
        if row["mode"] == FULL_METHOD
    )
    baselines = sorted({row["mode"] for row in rows if row["mode"] != FULL_METHOD})
    out: list[dict[str, Any]] = []
    for baseline_index, baseline in enumerate(baselines):
        for metric_index, (metric, direction) in enumerate(METRICS.items()):
            advantages: list[float] = []
            full_values: list[float] = []
            baseline_values: list[float] = []
            wins = 0
            ties = 0
            for candidate_id in candidate_ids:
                full = keyed.get((candidate_id, FULL_METHOD))
                other = keyed.get((candidate_id, baseline))
                if full is None or other is None:
                    continue
                full_value = float(full[metric])
                baseline_value = float(other[metric])
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
                seed=5101 + 101 * baseline_index + metric_index,
                samples=bootstrap_samples,
            )
            pair_count = len(advantages)
            out.append(
                {
                    "full_method": FULL_METHOD,
                    "full_method_label": METHOD_LABELS[FULL_METHOD],
                    "baseline": baseline,
                    "baseline_label": METHOD_LABELS.get(baseline, baseline),
                    "metric": metric,
                    "preferred_direction": direction,
                    "paired_candidate_count": pair_count,
                    "full_mean": round(mean(full_values), 6),
                    "baseline_mean": round(mean(baseline_values), 6),
                    "mean_full_advantage": round(mean(advantages), 6),
                    "ci95_low": round(ci_low, 6),
                    "ci95_high": round(ci_high, 6),
                    "full_win_rate": round(float(wins) / float(pair_count), 6) if pair_count else 0.0,
                    "tie_rate": round(float(ties) / float(pair_count), 6) if pair_count else 0.0,
                    "ci_excludes_zero_in_full_favor": int(ci_low > 0.0),
                }
            )
    write_csv(output_path, out)
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="outputs/candidate_repair/experiment2_candidate_repair_llm_corpus_v1/case_results.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/candidate_repair/experiment2_candidate_repair_llm_corpus_v1/paired_comparison.csv",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = (project_root / str(args.output)).resolve()
    results = analyze_candidate_pairs(
        input_path=(project_root / str(args.input)).resolve(),
        output_path=output_path,
        bootstrap_samples=int(args.bootstrap_samples),
    )
    print(f"Wrote {len(results)} paired Experiment 2 comparison rows to {output_path}")


if __name__ == "__main__":
    main()
