"""
Registry for named scene templates.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

from models.scenario import (
    ChannelProfile,
    ControlProfile,
    RadioProfile,
    ScenarioTemplate,
    TopologyProfile,
    TrafficProfile,
)
from scenarios.scenario_templates import build_builtin_scenario_templates


def deep_merge_dict(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


class ScenarioRegistry:
    def __init__(self) -> None:
        self._templates = build_builtin_scenario_templates()

    def list_template_names(self) -> list[str]:
        return sorted(self._templates.keys())

    def get_template(self, name: str) -> ScenarioTemplate:
        template_name = str(name or "baseline_light_load_static").strip()
        if template_name not in self._templates:
            available = ", ".join(self.list_template_names())
            raise ValueError(f"Unknown scenario template '{template_name}'. Available: {available}")
        return deepcopy(self._templates[template_name])

    def resolve(self, scenario_config: dict[str, Any] | None = None) -> ScenarioTemplate:
        config = dict(scenario_config or {})
        template = self.get_template(str(config.get("scenario_name", "baseline_light_load_static")))
        template_overrides = dict(config.get("template_overrides", {}) or {})
        if len(template_overrides) == 0:
            return template

        template_dict = asdict(template)
        merged = deep_merge_dict(template_dict, template_overrides)
        return ScenarioTemplate(
            name=str(merged.get("name", template.name)),
            description=str(merged.get("description", template.description)),
            topology=TopologyProfile(**dict(merged.get("topology", {}))),
            radio=RadioProfile(**dict(merged.get("radio", {}))),
            channel=ChannelProfile(**dict(merged.get("channel", {}))),
            traffic=TrafficProfile(**dict(merged.get("traffic", {}))),
            control=ControlProfile(**dict(merged.get("control", {}))),
            metadata=dict(merged.get("metadata", {})),
        )
