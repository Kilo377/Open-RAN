"""Run Experiment 3-B on the slot-driven simulator."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from experiments.experiment3_simulator_bridge import (  # noqa: E402
    BRIDGE_EVENT_NAMES,
    BRIDGE_METHOD_NAMES,
    BRIDGE_ROBUSTNESS_PROFILES,
    run_experiment3_simulator_bridge,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 3-B full simulator bridge.")
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_eval/experiment3_simulator_bridge_v1",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--training-seeds",
        nargs="+",
        type=int,
        default=[1001, 1002, 1003, 1004, 1005],
    )
    parser.add_argument("--episode-length", type=int, default=120)
    parser.add_argument("--event-time-fraction", type=float, default=0.35)
    parser.add_argument("--shock-fraction", type=float, default=0.12)
    parser.add_argument("--recovery-window", type=int, default=6)
    parser.add_argument("--training-episodes", type=int, default=40)
    parser.add_argument("--methods", nargs="+", default=list(BRIDGE_METHOD_NAMES))
    parser.add_argument("--events", nargs="+", default=list(BRIDGE_EVENT_NAMES))
    parser.add_argument(
        "--severities",
        nargs="+",
        choices=["low", "medium", "high"],
        default=["low", "medium", "high"],
    )
    parser.add_argument(
        "--robustness-condition",
        choices=sorted(BRIDGE_ROBUSTNESS_PROFILES),
        default="nominal",
    )
    parser.add_argument("--no-time-series", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = (project_root / str(args.output_dir)).resolve()
    result = run_experiment3_simulator_bridge(
        output_dir,
        seeds=[int(seed) for seed in args.seeds],
        training_seeds=[int(seed) for seed in args.training_seeds],
        episode_length=int(args.episode_length),
        event_time_fraction=float(args.event_time_fraction),
        shock_fraction=float(args.shock_fraction),
        recovery_window=int(args.recovery_window),
        training_episodes=int(args.training_episodes),
        method_names=[str(name) for name in args.methods],
        event_names=[str(name) for name in args.events],
        severity_names=[str(name) for name in args.severities],
        robustness_condition=str(args.robustness_condition),
        write_time_series=not bool(args.no_time_series),
    )
    print(f"Generated Experiment 3-B simulator bridge under {output_dir}")
    print(f"- episodes: {len(result['episode_metrics'])}")
    print(f"- summary: {output_dir / 'summary.csv'}")
    print(f"- recovery curve: {output_dir / 'recovery_curve.csv'}")


if __name__ == "__main__":
    main()
