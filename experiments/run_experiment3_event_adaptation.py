"""Run Experiment 3 event-driven contract-constrained adaptation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from experiments.event_driven_contract_adaptation import (  # noqa: E402
    run_event_driven_contract_adaptation_experiment,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Experiment 3: Event-Driven Contract-Constrained Near-RT Adaptation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/paper_eval/experiment3_event_adaptation_neural_v2",
        help="Output directory for Experiment 3 artifacts.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(1, 11)),
        help="Random seeds to evaluate.",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=300,
        help="Number of slots in each event episode.",
    )
    parser.add_argument(
        "--event-time-fraction",
        type=float,
        default=0.40,
        help="Fraction of the episode at which the event is injected.",
    )
    parser.add_argument(
        "--shock-fraction",
        type=float,
        default=0.10,
        help="Fraction of the episode used for the event shock phase.",
    )
    parser.add_argument(
        "--recovery-window",
        type=int,
        default=12,
        help="Consecutive slots required for recovery/stabilization metrics.",
    )
    parser.add_argument(
        "--method-suite",
        choices=["main", "ablation", "algorithm", "literature", "paper", "full"],
        default="paper",
        help="Method suite to run. 'paper' includes main baselines, ablations, TD3/SAC, and literature-inspired profiles.",
    )
    parser.add_argument(
        "--training-episodes",
        type=int,
        default=80,
        help="Synthetic training episodes for projection diagnostics.",
    )
    parser.add_argument(
        "--evaluation-protocol",
        choices=["all", "heldout"],
        default="heldout",
        help="Evaluation protocol. 'heldout' separates training, validation, IID test, and OOD test cases.",
    )
    parser.add_argument(
        "--training-seeds",
        nargs="+",
        type=int,
        default=[1001, 1002, 1003, 1004, 1005],
        help="Training-only seeds used to initialize neural policies under the held-out protocol.",
    )
    parser.add_argument(
        "--validation-seeds",
        nargs="+",
        type=int,
        default=[2001, 2002, 2003],
        help="Validation-only seeds under the held-out protocol.",
    )
    parser.add_argument(
        "--no-time-series",
        action="store_true",
        help="Skip writing per-slot time_series.csv while still computing summaries and figures.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip matplotlib figure generation while still writing CSV and JSON summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = (project_root / str(args.output_dir)).resolve()
    result = run_event_driven_contract_adaptation_experiment(
        output_dir,
        seeds=[int(seed) for seed in args.seeds],
        episode_length=int(args.episode_length),
        event_time_fraction=float(args.event_time_fraction),
        shock_fraction=float(args.shock_fraction),
        recovery_window=int(args.recovery_window),
        method_suite=str(args.method_suite),
        training_episodes=int(args.training_episodes),
        write_time_series=not bool(args.no_time_series),
        write_figures=not bool(args.no_figures),
        evaluation_protocol=str(args.evaluation_protocol),
        training_seeds=[int(seed) for seed in args.training_seeds],
        validation_seeds=[int(seed) for seed in args.validation_seeds],
    )
    print(f"Generated Experiment 3 artifacts under {output_dir}")
    print(f"- methods: {len(result['methods'])}")
    print(f"- episode rows: {len(result['episode_metrics'])}")
    print(f"- summary: {output_dir / 'summary.csv'}")
    if not bool(args.no_figures):
        print(f"- figures: {output_dir / 'figures'}")


if __name__ == "__main__":
    main()
