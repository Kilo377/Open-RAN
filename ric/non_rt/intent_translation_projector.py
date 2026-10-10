"""Projector/repair layer for intent translation candidates."""

from __future__ import annotations

from typing import Any

from models.intent_orchestration import GoalKpiTarget, ProjectionReport, StructuredIntentGoal
from models.observation import Observation
from ric.non_rt.intent_orchestration_helpers import stable_intent_id
from ric.non_rt.intent_translation_validator import (
    KPI_UNIT_BY_NAME,
    coerce_numeric,
    feasible_target_bounds,
    normalize_kpi_name,
    normalize_objective,
)


def _normalize_constraints(raw_constraints: object) -> dict[str, object]:
    if isinstance(raw_constraints, dict):
        return dict(raw_constraints)
    if isinstance(raw_constraints, list):
        return {
            f"constraint_{index + 1}": str(item)
            for index, item in enumerate(raw_constraints)
            if str(item).strip() != ""
        }
    if raw_constraints is None:
        return {}
    text = str(raw_constraints).strip()
    if text == "":
        return {}
    return {"constraint_1": text}


class IntentTemplateProjector:
    """Repairs raw candidates into an admissible structured goal when possible."""

    def __init__(self, *, name: str = "intent_template_projector") -> None:
        self.name = str(name or "intent_template_projector")

    def project(
        self,
        candidate: object,
        *,
        observation: Observation,
        intent_text: str,
        active_intent_id: str,
    ) -> tuple[StructuredIntentGoal | None, ProjectionReport, dict[str, object]]:
        report = ProjectionReport(
            projected=False,
            projector_name=self.name,
            projection_target="structured_intent_goal",
        )
        projector_output: dict[str, object] = {
            "retained_kpis": [],
            "dropped_kpis": [],
            "repair_actions": [],
        }

        if not isinstance(candidate, dict):
            report.notes.append("candidate_root_not_object")
            return None, report, projector_output

        raw_kpi_targets = candidate.get("kpi_targets")
        if not isinstance(raw_kpi_targets, list) or len(raw_kpi_targets) == 0:
            report.notes.append("kpi_targets_missing_or_empty")
            return None, report, projector_output

        projected_kpis: list[GoalKpiTarget] = []
        seen_kpis: set[str] = set()

        for index, entry in enumerate(raw_kpi_targets):
            if not isinstance(entry, dict):
                report.dropped_fields.append(f"kpi_targets[{index}]")
                projector_output["dropped_kpis"].append({"index": index, "reason": "non_object"})
                continue

            normalized_kpi_name = normalize_kpi_name(entry.get("kpi_name", ""))
            if normalized_kpi_name not in KPI_UNIT_BY_NAME:
                report.dropped_fields.append(f"kpi_targets[{index}].kpi_name")
                projector_output["dropped_kpis"].append(
                    {"index": index, "kpi_name": normalized_kpi_name, "reason": "inadmissible_kpi"}
                )
                continue

            if normalized_kpi_name in seen_kpis:
                report.dropped_fields.append(f"kpi_targets[{index}]")
                projector_output["dropped_kpis"].append(
                    {"index": index, "kpi_name": normalized_kpi_name, "reason": "duplicate_kpi"}
                )
                continue
            seen_kpis.add(normalized_kpi_name)

            normalized_objective = normalize_objective(entry.get("objective", ""))
            if normalized_objective is None:
                report.dropped_fields.append(f"kpi_targets[{index}].objective")
                projector_output["dropped_kpis"].append(
                    {"index": index, "kpi_name": normalized_kpi_name, "reason": "invalid_objective"}
                )
                continue

            target_value = coerce_numeric(entry.get("target_value"))
            if target_value is None:
                report.dropped_fields.append(f"kpi_targets[{index}].target_value")
                projector_output["dropped_kpis"].append(
                    {"index": index, "kpi_name": normalized_kpi_name, "reason": "missing_numeric_target"}
                )
                continue

            lower_bound, upper_bound = feasible_target_bounds(normalized_kpi_name, observation)
            projected_target = min(max(target_value, lower_bound), upper_bound)
            if projected_target != target_value:
                report.projected = True
                report.repaired_fields.append(f"kpi_targets[{index}].target_value")
                projector_output["repair_actions"].append(
                    {
                        "field": f"kpi_targets[{index}].target_value",
                        "action": "clip_to_feasible_range",
                        "from": target_value,
                        "to": projected_target,
                        "bounds": [lower_bound, upper_bound],
                    }
                )

            projected_weight = coerce_numeric(entry.get("weight"))
            if projected_weight is None or projected_weight <= 0.0:
                projected_weight = 1.0
                report.repaired_fields.append(f"kpi_targets[{index}].weight")
                projector_output["repair_actions"].append(
                    {
                        "field": f"kpi_targets[{index}].weight",
                        "action": "default_weight",
                        "to": projected_weight,
                    }
                )

            projected_priority_value = coerce_numeric(entry.get("priority"))
            projected_priority = int(projected_priority_value or 0)
            if projected_priority_value is None:
                report.repaired_fields.append(f"kpi_targets[{index}].priority")
                projector_output["repair_actions"].append(
                    {
                        "field": f"kpi_targets[{index}].priority",
                        "action": "default_priority",
                        "to": projected_priority,
                    }
                )

            projected_tolerance = coerce_numeric(entry.get("tolerance"))
            candidate_unit = str(entry.get("unit", "") or "").strip().lower()
            canonical_unit = KPI_UNIT_BY_NAME[normalized_kpi_name]
            if candidate_unit != canonical_unit:
                report.repaired_fields.append(f"kpi_targets[{index}].unit")
                projector_output["repair_actions"].append(
                    {
                        "field": f"kpi_targets[{index}].unit",
                        "action": "normalize_unit",
                        "from": candidate_unit,
                        "to": canonical_unit,
                    }
                )

            projected_kpi = GoalKpiTarget(
                kpi_name=normalized_kpi_name,
                objective=normalized_objective,
                target_value=projected_target,
                tolerance=projected_tolerance,
                weight=float(projected_weight),
                unit=canonical_unit,
                priority=projected_priority,
            )
            projected_kpis.append(projected_kpi)
            projector_output["retained_kpis"].append(
                {
                    "index": index,
                    "kpi_name": normalized_kpi_name,
                    "objective": normalized_objective,
                    "target_value": projected_target,
                }
            )

        if len(projected_kpis) == 0:
            report.notes.append("no_projectable_kpis")
            return None, report, projector_output

        horizon_slots = int(coerce_numeric(candidate.get("horizon_slots", 12)) or 12)
        if horizon_slots < 1:
            horizon_slots = 1
            report.repaired_fields.append("horizon_slots")
            projector_output["repair_actions"].append(
                {
                    "field": "horizon_slots",
                    "action": "clamp_min",
                    "to": horizon_slots,
                }
            )

        raw_target_entities = candidate.get("target_entities", ["cluster"])
        if isinstance(raw_target_entities, list) and len(raw_target_entities) > 0:
            target_entities = [str(item) for item in raw_target_entities if str(item).strip() != ""]
        elif str(raw_target_entities).strip() != "":
            target_entities = [str(raw_target_entities).strip()]
        else:
            target_entities = ["cluster"]

        intent_id = str(active_intent_id or stable_intent_id(intent_text))
        goal_id = str(candidate.get("goal_id", "") or f"goal:{intent_id.split(':', 1)[-1]}")
        projected_goal = StructuredIntentGoal(
            goal_id=goal_id,
            intent_id=intent_id,
            intent_text=str(intent_text or ""),
            summary=str(
                candidate.get("summary", "Projected RAN goal derived from active intent")
                or "Projected RAN goal derived from active intent"
            ),
            horizon_slots=max(1, horizon_slots),
            risk_posture=str(candidate.get("risk_posture", "balanced") or "balanced"),
            target_entities=target_entities,
            kpi_targets=projected_kpis,
            constraints=_normalize_constraints(candidate.get("constraints", {})),
            generated_by="validator_projector",
        )
        projector_output["projected_goal"] = {
            "goal_id": projected_goal.goal_id,
            "intent_id": projected_goal.intent_id,
            "kpi_count": len(projected_goal.kpi_targets),
        }
        return projected_goal, report, projector_output
