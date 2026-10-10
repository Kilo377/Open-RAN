"""
Registry for scenario matrices and sweep expansion.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import product
import re
from typing import Any

from models.scenario_matrix import ScenarioMatrix, ScenarioMatrixCase, ScenarioSweepDimension, ScenarioSweepValue
from scenarios.scenario_matrix_templates import build_builtin_scenario_matrices
from scenarios.scenario_registry import deep_merge_dict


def _build_nested_override(path: str, value: Any) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    current = nested
    path_parts = [part for part in str(path).split(".") if str(part).strip() != ""]
    if len(path_parts) == 0:
        return nested
    for path_part in path_parts[:-1]:
        current = current.setdefault(path_part, {})
    current[path_parts[-1]] = value
    return nested


def _sanitize_token(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value).strip()).strip("_").lower()


class ScenarioMatrixRegistry:
    def __init__(self) -> None:
        self._matrices = build_builtin_scenario_matrices()

    def list_matrix_names(self) -> list[str]:
        return sorted(self._matrices.keys())

    def get_matrix(self, name: str) -> ScenarioMatrix:
        matrix_name = str(name or "scenario_catalog").strip()
        if matrix_name not in self._matrices:
            available = ", ".join(self.list_matrix_names())
            raise ValueError(f"Unknown scenario matrix '{matrix_name}'. Available: {available}")
        return deepcopy(self._matrices[matrix_name])

    def _apply_dimension(
        self,
        *,
        case: ScenarioMatrixCase,
        dimension: ScenarioSweepDimension,
        sweep_value: ScenarioSweepValue,
    ) -> None:
        target = str(dimension.target).strip().lower()
        nested_override = _build_nested_override(dimension.path, sweep_value.value)
        if target == "template":
            case.template_overrides = deep_merge_dict(case.template_overrides, nested_override)
            return
        if target == "sim":
            case.sim_overrides = deep_merge_dict(case.sim_overrides, nested_override)
            return
        if target == "cell":
            case.cell_overrides = deep_merge_dict(case.cell_overrides, nested_override)
            return
        if target == "ue":
            case.ue_overrides = deep_merge_dict(case.ue_overrides, nested_override)
            return
        if target == "ric":
            case.ric_overrides = deep_merge_dict(case.ric_overrides, nested_override)
            return
        if target == "beamforming":
            case.beamforming_overrides = deep_merge_dict(case.beamforming_overrides, nested_override)
            return
        if target == "scenario":
            case.scenario_overrides = deep_merge_dict(case.scenario_overrides, nested_override)
            return
        raise ValueError(f"Unsupported scenario sweep target '{dimension.target}'")

    def expand_matrix(self, name: str) -> list[ScenarioMatrixCase]:
        matrix = self.get_matrix(name)
        expanded_cases: list[ScenarioMatrixCase] = []

        for case in matrix.cases:
            concrete_case = deepcopy(case)
            concrete_case.tags = sorted(set(list(concrete_case.tags) + [matrix.name]))
            expanded_cases.append(concrete_case)

        if matrix.base_case is None:
            return expanded_cases

        if len(matrix.sweep_dimensions) == 0:
            concrete_case = deepcopy(matrix.base_case)
            concrete_case.tags = sorted(set(list(concrete_case.tags) + [matrix.name]))
            expanded_cases.append(concrete_case)
            return expanded_cases

        dimension_values = [list(dimension.values) for dimension in matrix.sweep_dimensions]
        for combo in product(*dimension_values):
            case = deepcopy(matrix.base_case)
            coordinates: dict[str, Any] = {}
            tokens: list[str] = []
            for dimension, sweep_value in zip(matrix.sweep_dimensions, combo):
                self._apply_dimension(case=case, dimension=dimension, sweep_value=sweep_value)
                coordinates[dimension.name] = sweep_value.value
                tokens.append(_sanitize_token(sweep_value.label))
            case.coordinates = coordinates
            case.tags = sorted(set(list(case.tags) + [matrix.name]))
            case.case_id = f"{case.case_id}__{'__'.join(tokens)}"
            expanded_cases.append(case)

        return expanded_cases
