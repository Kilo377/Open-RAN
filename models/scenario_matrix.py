"""
Scenario-matrix models for reusable scene catalogs and parameter sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScenarioSweepValue:
    label: str
    value: Any


@dataclass
class ScenarioSweepDimension:
    name: str
    target: str
    path: str
    values: list[ScenarioSweepValue] = field(default_factory=list)


@dataclass
class ScenarioMatrixCase:
    case_id: str
    description: str
    scenario_name: str
    template_overrides: dict[str, Any] = field(default_factory=dict)
    scenario_overrides: dict[str, Any] = field(default_factory=dict)
    sim_overrides: dict[str, Any] = field(default_factory=dict)
    cell_overrides: dict[str, Any] = field(default_factory=dict)
    ue_overrides: dict[str, Any] = field(default_factory=dict)
    ric_overrides: dict[str, Any] = field(default_factory=dict)
    beamforming_overrides: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    coordinates: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioMatrix:
    name: str
    description: str
    base_case: ScenarioMatrixCase | None = None
    sweep_dimensions: list[ScenarioSweepDimension] = field(default_factory=list)
    cases: list[ScenarioMatrixCase] = field(default_factory=list)
