"""Run paper sandbox experiments and generate data/figures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from experiments.paper_sandbox_eval import run_selected_experiments


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper sandbox evaluations.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/paper_eval/paper_sandbox_eval_v1",
        help="Root output directory for sandbox paper experiments.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["2", "3", "5", "6", "7", "8", "9"],
        help="Experiment ids or aliases to run.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Random seeds used for deterministic trace generation.",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="deepseek",
        help="LLM provider label recorded in LLM-safety artifacts.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="deepseek-v4-pro",
        help="LLM model label recorded in LLM-safety artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = (project_root / str(args.output_dir)).resolve()
    experiment_outputs = run_selected_experiments(
        root_output_dir=output_dir,
        seeds=[int(seed) for seed in args.seeds],
        experiments=[str(item) for item in args.experiments],
        llm_provider=str(args.llm_provider),
        llm_model=str(args.llm_model),
    )
    print(f"Generated sandbox paper artifacts under {output_dir}")
    for name, path in experiment_outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
