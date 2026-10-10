"""Validator for intent-to-KPI translation candidates."""

from __future__ import annotations

import re
from typing import Any

from models.intent_orchestration import ValidationReport
from models.observation import Observation


KPI_UNIT_BY_NAME: dict[str, str] = {
    "total_throughput_mbps": "mbps",
    "avg_bler": "ratio",
    "queue_bytes": "bytes",
    "total_queue_bytes": "bytes",
    "sinr": "db",
    "handover_risk": "score",
    "power_cost": "dbm_like",
}

KPI_DIRECTION_BY_NAME: dict[str, set[str]] = {
    "total_throughput_mbps": {"maximize", "max", "target"},
    "avg_bler": {"minimize", "min", "target"},
    "queue_bytes": {"minimize", "min", "target"},
    "total_queue_bytes": {"minimize", "min", "target"},
    "sinr": {"maximize", "max", "target"},
    "handover_risk": {"minimize", "min", "target"},
    "power_cost": {"minimize", "min", "target"},
}


def coerce_numeric(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match is None:
        return None
    return float(match.group(0))


def normalize_kpi_name(raw_name: object) -> str:
    normalized = str(raw_name or "").strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    alias_map = {
        "throughput": "total_throughput_mbps",
        "total_throughput": "total_throughput_mbps",
        "throughput_mbps": "total_throughput_mbps",
        "total_throughput_mbps": "total_throughput_mbps",
        "bler": "avg_bler",
        "avg_bler": "avg_bler",
        "sinr": "sinr",
        "sinr_db": "sinr",
        "avg_sinr_db": "sinr",
        "queue": "queue_bytes",
        "queue_bytes": "queue_bytes",
        "total_queue": "total_queue_bytes",
        "total_queue_bytes": "total_queue_bytes",
        "handover": "handover_risk",
        "handover_risk": "handover_risk",
        "mobility_risk": "handover_risk",
        "power": "power_cost",
        "power_cost": "power_cost",
        "energy": "power_cost",
    }
    return alias_map.get(normalized, normalized)


def normalize_objective(raw_value: object) -> str | None:
    normalized = str(raw_value or "").strip().lower()
    alias_map = {
        "maximize": "maximize",
        "max": "maximize",
        "increase": "maximize",
        "boost": "maximize",
        "minimize": "minimize",
        "min": "minimize",
        "decrease": "minimize",
        "reduce": "minimize",
        "target": "target",
        "track": "target",
    }
    return alias_map.get(normalized)


def feasible_target_bounds(kpi_name: str, observation: Observation) -> tuple[float, float]:
    throughput = float(getattr(observation, "total_throughput_mbps", 0.0) or 0.0)
    queue_bytes = float(getattr(observation, "total_queue_bytes", 0.0) or 0.0)
    if kpi_name == "total_throughput_mbps":
        return (0.0, max(10.0, throughput * 2.5 + 10.0))
    if kpi_name == "avg_bler":
        return (0.0, 1.0)
    if kpi_name in {"queue_bytes", "total_queue_bytes"}:
        return (0.0, max(1000.0, queue_bytes * 2.0 + 1000.0))
    if kpi_name == "sinr":
        return (-20.0, 80.0)
    if kpi_name == "handover_risk":
        return (0.0, 1.0)
    if kpi_name == "power_cost":
        return (0.0, 50.0)
    return (-1.0e12, 1.0e12)


class IntentTemplateValidator:
    """Structured validator for raw LLM KPI templates."""

    def __init__(self, *, name: str = "intent_template_validator") -> None:
        self.name = str(name or "intent_template_validator")

    def validate(self, candidate: object, observation: Observation) -> ValidationReport:
        report = ValidationReport(validator_name=self.name)
        if not isinstance(candidate, dict):
            report.schema_ok = False
            report.issues.append("candidate_root_not_object")
            return report

        raw_kpis = candidate.get("kpi_targets")
        if not isinstance(raw_kpis, list):
            report.schema_ok = False
            report.issues.append("kpi_targets_missing_or_not_list")
            return report

        if len(raw_kpis) == 0:
            report.admissibility_ok = False
            report.issues.append("kpi_targets_empty")
            return report

        seen_kpis: set[str] = set()
        has_admissible_kpi = False

        for index, entry in enumerate(raw_kpis):
            if not isinstance(entry, dict):
                report.schema_ok = False
                report.rejected_kpis.append(f"index:{index}:non_object")
                continue

            normalized_kpi_name = normalize_kpi_name(entry.get("kpi_name", ""))
            if normalized_kpi_name == "":
                report.schema_ok = False
                report.rejected_kpis.append(f"index:{index}:missing_kpi_name")
                continue
            if normalized_kpi_name != str(entry.get("kpi_name", "") or ""):
                report.normalized_fields.append(f"kpi_targets[{index}].kpi_name")

            if normalized_kpi_name in seen_kpis:
                report.duplicate_kpis.append(normalized_kpi_name)
            else:
                seen_kpis.add(normalized_kpi_name)

            if normalized_kpi_name not in KPI_UNIT_BY_NAME:
                report.admissibility_ok = False
                report.rejected_kpis.append(normalized_kpi_name)
                continue

            has_admissible_kpi = True

            normalized_objective = normalize_objective(entry.get("objective", ""))
            if normalized_objective is None:
                report.direction_ok = False
                report.issues.append(f"invalid_objective:{normalized_kpi_name}")
            elif normalized_objective not in KPI_DIRECTION_BY_NAME[normalized_kpi_name]:
                report.direction_ok = False
                report.issues.append(f"inadmissible_direction:{normalized_kpi_name}:{normalized_objective}")

            target_value = coerce_numeric(entry.get("target_value"))
            if "target_value" in entry and target_value is None:
                report.schema_ok = False
                report.issues.append(f"non_numeric_target:{normalized_kpi_name}")
            elif target_value is not None:
                lower_bound, upper_bound = feasible_target_bounds(normalized_kpi_name, observation)
                if target_value < lower_bound or target_value > upper_bound:
                    report.range_ok = False
                    report.warnings.append(
                        f"target_out_of_range:{normalized_kpi_name}:{target_value}:{lower_bound}:{upper_bound}"
                    )

            weight_value = coerce_numeric(entry.get("weight"))
            if "weight" in entry and weight_value is None:
                report.schema_ok = False
                report.issues.append(f"non_numeric_weight:{normalized_kpi_name}")

            priority_value = coerce_numeric(entry.get("priority"))
            if "priority" in entry and priority_value is None:
                report.schema_ok = False
                report.issues.append(f"non_numeric_priority:{normalized_kpi_name}")

            candidate_unit = str(entry.get("unit", "") or "").strip().lower()
            expected_unit = KPI_UNIT_BY_NAME[normalized_kpi_name]
            if candidate_unit != "" and candidate_unit != expected_unit:
                report.warnings.append(
                    f"unit_mismatch:{normalized_kpi_name}:{candidate_unit}:{expected_unit}"
                )

        if report.duplicate_kpis:
            report.admissibility_ok = False

        if not has_admissible_kpi:
            report.admissibility_ok = False

        return report
