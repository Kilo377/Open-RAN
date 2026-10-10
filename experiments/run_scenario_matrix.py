"""
Batch experiment entry point for scenario matrices and parameter sweeps.
"""

from __future__ import annotations

from pathlib import Path
import sys


project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from core.config_loader import load_all_configs, load_yaml_file
from core.debug import DebugConfig
from experiments.scenario_scan_utils import build_case_summary_record, run_matrix_case, write_matrix_summary
from scenarios.scenario_matrix_registry import ScenarioMatrixRegistry


def _resolve_run_settings(args: list[str], matrix_config: dict[str, object]) -> tuple[str, set[str], int | None, str]:
    matrix_name = str(
        args[1] if len(args) > 1 else matrix_config.get("matrix_name", "scenario_catalog")
    ).strip()
    selected_cases_raw = matrix_config.get("selected_case_ids", []) or []
    selected_cases_config = selected_cases_raw if isinstance(selected_cases_raw, list) else []
    selected_case_ids = {str(item).strip() for item in selected_cases_config if str(item).strip() != ""}
    if len(args) > 2 and str(args[2]).strip() != "":
        selected_case_ids = {str(item).strip() for item in str(args[2]).split(",") if str(item).strip() != ""}
    max_cases_value = matrix_config.get("max_cases", None)
    if len(args) > 3 and str(args[3]).strip() != "":
        max_cases_value = int(args[3])
    max_cases = int(max_cases_value) if isinstance(max_cases_value, (int, float, str)) and str(max_cases_value) != "" else None
    output_subdir = str(matrix_config.get("output_subdir", "scenario_matrix")).strip() or "scenario_matrix"
    return matrix_name, selected_case_ids, max_cases, output_subdir


def main() -> None:
    DebugConfig.DEBUG = False

    configs = load_all_configs(str(project_root / "configs"))
    phy_config = load_yaml_file(str(project_root / "configs" / "phy.yaml"))
    matrix_config = load_yaml_file(str(project_root / "configs" / "scenario_matrix.yaml"))

    matrix_name, selected_case_ids, max_cases, output_subdir = _resolve_run_settings(sys.argv, matrix_config)
    matrix_registry = ScenarioMatrixRegistry()
    matrix_definition = matrix_registry.get_matrix(matrix_name)
    matrix_cases = matrix_registry.expand_matrix(matrix_name)

    if len(selected_case_ids) > 0:
        matrix_cases = [case for case in matrix_cases if case.case_id in selected_case_ids]
    if max_cases is not None:
        matrix_cases = matrix_cases[: max(0, int(max_cases))]

    matrix_output_dir = project_root / "outputs" / output_subdir / matrix_name
    case_records: list[dict[str, object]] = []
    for index, case in enumerate(matrix_cases, start=1):
        print(f"[matrix] running {index}/{len(matrix_cases)}: {case.case_id}")
        case_output_dir = matrix_output_dir / case.case_id
        final_summary = run_matrix_case(
            base_configs=configs,
            phy_config=phy_config,
            case=case,
            output_dir=str(case_output_dir),
        )
        case_records.append(
            build_case_summary_record(
                case=case,
                final_summary=final_summary,
                output_dir=str(case_output_dir),
            )
        )

    write_matrix_summary(
        output_dir=str(matrix_output_dir),
        matrix_name=matrix_definition.name,
        matrix_description=matrix_definition.description,
        case_records=case_records,
    )
    print(f"Scenario matrix '{matrix_definition.name}' completed with {len(case_records)} cases.")


if __name__ == "__main__":
    main()
