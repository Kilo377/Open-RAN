"""Run the candidate-level feasibility repair benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from core.config_loader import load_yaml_file
from experiments.candidate_repair_eval import (
    DEFAULT_REPAIR_MODES,
    build_default_available_xapps,
    build_default_candidate_fixtures,
    build_exp2_candidate_pool,
    evaluate_candidate_repair_modes,
    load_candidate_fixture_jsonl,
    write_candidate_repair_outputs,
)
from ric.non_rt.orchestration_feasibility import OrchestrationFeasibilityProjector


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run candidate feasibility repair evaluation.")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/candidate_repair_eval.yaml",
        help="Candidate repair evaluation config path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print the planned candidate/mode count without writing outputs.",
    )
    return parser.parse_args()


def _repair_modes_from_config(config: dict[str, object]) -> list[str]:
    modes = config.get("repair_modes", DEFAULT_REPAIR_MODES)
    if not isinstance(modes, list):
        return list(DEFAULT_REPAIR_MODES)
    return [str(item).strip() for item in modes if str(item).strip() != ""]


def main() -> None:
    args = _parse_args()
    config_path = project_root / str(args.config)
    config = load_yaml_file(str(config_path))

    modes = _repair_modes_from_config(config)
    available_xapps = build_default_available_xapps()
    pool_config = dict(config.get("candidate_pool", {}) or {})
    use_exp2_pool = bool(pool_config.get("use_exp2_pool", False))
    candidates_per_context = int(pool_config.get("candidates_per_context", 20))
    candidate_seed = int(pool_config.get("seed", 20260517))
    corpus_path_text = str(pool_config.get("corpus_path", "") or "").strip()
    corpus_path = Path(corpus_path_text)
    if corpus_path_text != "" and not corpus_path.is_absolute():
        corpus_path = project_root / corpus_path
    if corpus_path_text != "":
        fixtures = load_candidate_fixture_jsonl(corpus_path)
    elif use_exp2_pool:
        fixtures = build_exp2_candidate_pool(
            candidates_per_context=candidates_per_context,
            seed=candidate_seed,
            available_xapp_names=available_xapps,
        )
    else:
        fixtures = build_default_candidate_fixtures()
    projector_config = dict(config.get("projector", {}) or {})
    projector = OrchestrationFeasibilityProjector(
        max_selected_xapps=int(projector_config.get("max_selected_xapps", 4)),
        max_cpu_cost=float(projector_config.get("max_cpu_cost", 4.6)),
        max_mem_cost=float(projector_config.get("max_mem_cost", 4.2)),
    )

    output_subdir = str(config.get("output_subdir", "candidate_repair") or "candidate_repair")
    run_name = str(config.get("run_name", "candidate_repair_eval_v1") or "candidate_repair_eval_v1")
    output_dir = project_root / "outputs" / output_subdir / run_name

    run_plan = {
        "config": str(config_path),
        "repair_modes": modes,
        "candidate_count": len(fixtures),
        "planned_evaluations": len(modes) * len(fixtures),
        "output_dir": str(output_dir),
        "available_xapp_count": len(available_xapps),
        "candidate_pool": {
            "use_exp2_pool": use_exp2_pool,
            "candidates_per_context": candidates_per_context,
            "seed": candidate_seed,
            "corpus_path": str(corpus_path) if corpus_path_text != "" else "",
        },
        "projector": {
            "max_selected_xapps": projector.max_selected_xapps,
            "max_cpu_cost": projector.max_cpu_cost,
            "max_mem_cost": projector.max_mem_cost,
        },
    }
    if args.dry_run:
        print(json.dumps(run_plan, indent=2, sort_keys=True))
        return

    results, summary = evaluate_candidate_repair_modes(
        modes=modes,
        fixtures=fixtures,
        projector=projector,
        available_xapp_names=available_xapps,
        use_exp2_pool=use_exp2_pool,
        candidates_per_context=candidates_per_context,
        seed=candidate_seed,
    )
    write_candidate_repair_outputs(
        output_dir=output_dir,
        results=results,
        summary=summary,
        config={**config, "run_plan": run_plan},
    )

    print(f"Candidate repair evaluation completed: {run_plan['planned_evaluations']} mode/candidate cases")
    print(f"Outputs written to: {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
