"""Run the typed conflict taxonomy ablation benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from core.config_loader import load_yaml_file
from experiments.conflict_taxonomy_eval import (
    DEFAULT_CONFLICT_MODES,
    build_default_conflict_fixtures,
    evaluate_conflict_taxonomy_modes,
    write_conflict_taxonomy_outputs,
)
from experiments.candidate_repair_eval import build_default_available_xapps
from ric.non_rt.orchestration_feasibility import OrchestrationFeasibilityProjector


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run conflict taxonomy evaluation.")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/conflict_taxonomy_eval.yaml",
        help="Conflict taxonomy evaluation config path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print planned case/mode count without writing outputs.",
    )
    return parser.parse_args()


def _modes_from_config(config: dict[str, object]) -> list[str]:
    modes = config.get("conflict_modes", DEFAULT_CONFLICT_MODES)
    if not isinstance(modes, list):
        return list(DEFAULT_CONFLICT_MODES)
    return [str(item).strip() for item in modes if str(item).strip() != ""]


def main() -> None:
    args = _parse_args()
    config_path = project_root / str(args.config)
    config = load_yaml_file(str(config_path))

    modes = _modes_from_config(config)
    fixtures = build_default_conflict_fixtures()
    available_xapps = build_default_available_xapps()
    projector_config = dict(config.get("projector", {}) or {})
    projector = OrchestrationFeasibilityProjector(
        max_selected_xapps=int(projector_config.get("max_selected_xapps", 4)),
        max_cpu_cost=float(projector_config.get("max_cpu_cost", 4.6)),
        max_mem_cost=float(projector_config.get("max_mem_cost", 4.2)),
    )

    output_subdir = str(config.get("output_subdir", "conflict_taxonomy") or "conflict_taxonomy")
    run_name = str(config.get("run_name", "conflict_taxonomy_eval_v1") or "conflict_taxonomy_eval_v1")
    output_dir = project_root / "outputs" / output_subdir / run_name

    run_plan = {
        "config": str(config_path),
        "conflict_modes": modes,
        "case_count": len(fixtures),
        "planned_evaluations": len(modes) * len(fixtures),
        "output_dir": str(output_dir),
        "available_xapp_count": len(available_xapps),
        "projector": {
            "max_selected_xapps": projector.max_selected_xapps,
            "max_cpu_cost": projector.max_cpu_cost,
            "max_mem_cost": projector.max_mem_cost,
        },
    }
    if args.dry_run:
        print(json.dumps(run_plan, indent=2, sort_keys=True))
        return

    results, summary = evaluate_conflict_taxonomy_modes(
        modes=modes,
        fixtures=fixtures,
        projector=projector,
        available_xapp_names=available_xapps,
    )
    write_conflict_taxonomy_outputs(
        output_dir=output_dir,
        results=results,
        summary=summary,
        config={**config, "run_plan": run_plan},
    )

    print(f"Conflict taxonomy evaluation completed: {run_plan['planned_evaluations']} mode/case evaluations")
    print(f"Outputs written to: {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
