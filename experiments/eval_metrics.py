"""
Unified experiment metrics for intent-driven orchestration studies.
"""

from __future__ import annotations

import json
import math
from typing import Any

from models.intent_orchestration import ValidatedIntentTemplate
from models.observation import Observation
from ric.non_rt.intent_orchestration_helpers import (
    INTENT_LOSS_DEFINITION_VERSION,
    observation_kpi_values,
)

INTENT_LOSS_NORMALIZATION_RULE = (
    "per_kpi_denom=normalization_coefficient_or_tolerance_or_max(abs(reference),1.0); "
    "per_kpi_normalized_gap_clipped_to_2.0"
)
INTENT_LOSS_NORMALIZED_GAP_CLIP = 2.0
INTENT_LOSS_APPROXIMATION_NOTES = [
    "The current implementation follows the paper-style weighted squared normalized intent loss.",
    "maximize/minimize objectives use one-sided squared penalties.",
    "target semantics use a symmetric squared normalized penalty.",
    "A robust per-KPI normalized-gap clip prevents a single malformed target from dominating macro averages.",
]

STABILITY_PENALTY_DEFINITION_VERSION = "v1_slotwise_effective_control_profile_delta"
STABILITY_CONTROL_FIELDS = [
    "scheduler_policy_hint",
    "latency_bias",
    "slice_weight_adjustment",
    "beam_candidate_preference",
    "power_cap_dbm",
    "traffic_steering_preference",
    "mobility_bias_db",
    "measurement_event_offset_db",
]
STABILITY_NUMERIC_SCALES = {
    "latency_bias": 1.0,
    "power_cap_dbm": 2.0,
    "mobility_bias_db": 3.0,
    "measurement_event_offset_db": 1.0,
}

SAFETY_VIOLATION_DEFINITION_VERSION = "v1_health_gate_threshold_check"
GENERALIZATION_GAP_DEFINITION_VERSION = "v1_reference_case_mean_difference"


def _safe_float(value: object | None, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, bool):
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: object | None, default: int = 0) -> int:
    if value is None:
        return int(default)
    if isinstance(value, bool):
        return int(default)
    if isinstance(value, int):
        return int(value)
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return int(default)


def _safe_dict(value: object | None) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _safe_list(value: object | None) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return []


def _observation_value(observation: Observation | dict[str, Any] | None, field_name: str) -> Any:
    if observation is None:
        return None
    if isinstance(observation, dict):
        return observation.get(field_name)
    return getattr(observation, field_name, None)


def _extract_observation_kpi_values(
    observation: Observation | dict[str, Any] | None,
) -> dict[str, float]:
    if isinstance(observation, Observation):
        return observation_kpi_values(observation)
    queue_bytes = _safe_float(
        _observation_value(observation, "total_queue_bytes")
        if _observation_value(observation, "total_queue_bytes") is not None
        else _observation_value(observation, "queue_bytes"),
        0.0,
    )
    throughput_mbps = _safe_float(
        _observation_value(observation, "total_throughput_mbps")
        if _observation_value(observation, "total_throughput_mbps") is not None
        else _observation_value(observation, "throughput_mbps"),
        0.0,
    )
    avg_bler = _safe_float(
        _observation_value(observation, "avg_bler")
        if _observation_value(observation, "avg_bler") is not None
        else _observation_value(observation, "bler"),
        0.0,
    )
    avg_sinr_db = _safe_float(
        _observation_value(observation, "avg_sinr_db")
        if _observation_value(observation, "avg_sinr_db") is not None
        else _observation_value(observation, "sinr_db"),
        0.0,
    )
    connected_ue_count = max(
        1.0,
        _safe_float(_observation_value(observation, "connected_ue_count"), 1.0),
    )
    handover_pending_ratio = (
        _safe_float(_observation_value(observation, "handover_pending_ue_count"), 0.0)
        / connected_ue_count
    )
    power_cost = _safe_float(
        _observation_value(observation, "power_cap_dbm")
        if _observation_value(observation, "power_cap_dbm") is not None
        else _observation_value(observation, "tx_power_dbm"),
        30.0,
    )
    return {
        "queue_bytes": queue_bytes,
        "total_queue_bytes": queue_bytes,
        "total_throughput_mbps": throughput_mbps,
        "avg_bler": avg_bler,
        "sinr": avg_sinr_db,
        "handover_risk": handover_pending_ratio,
        "power_cost": power_cost,
    }


def _objective_to_semantic_type(objective: object | None) -> str:
    normalized = str(objective or "maximize").strip().lower()
    if normalized in {"maximize", "max"}:
        return "max"
    if normalized in {"minimize", "min"}:
        return "min"
    if normalized == "target":
        return "target"
    return "max"


def _iter_template_targets(
    *,
    validated_intent_template: ValidatedIntentTemplate | dict[str, Any] | None = None,
    structured_goal: Any | None = None,
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    if validated_intent_template is not None:
        kpi_template = _safe_list(
            getattr(validated_intent_template, "kpi_template", None)
            if not isinstance(validated_intent_template, dict)
            else validated_intent_template.get("kpi_template")
        )
        for item in kpi_template:
            normalization = (
                getattr(item, "normalization", None)
                if not isinstance(item, dict)
                else item.get("normalization")
            )
            targets.append(
                {
                    "kpi_name": str(getattr(item, "kpi_name", "") if not isinstance(item, dict) else item.get("kpi_name", "")),
                    "semantic_type": str(
                        getattr(item, "semantic_type", "max")
                        if not isinstance(item, dict)
                        else item.get("semantic_type", "max")
                    ).strip().lower(),
                    "target_value": (
                        getattr(item, "target_value", None)
                        if not isinstance(item, dict)
                        else item.get("target_value")
                    ),
                    "threshold_value": (
                        getattr(item, "threshold_value", None)
                        if not isinstance(item, dict)
                        else item.get("threshold_value")
                    ),
                    "weight": _safe_float(
                        getattr(item, "weight", 1.0) if not isinstance(item, dict) else item.get("weight", 1.0),
                        1.0,
                    ),
                    "priority": int(
                        _safe_float(
                            getattr(item, "priority", 0)
                            if not isinstance(item, dict)
                            else item.get("priority", 0),
                            0.0,
                        )
                    ),
                    "hard": bool(
                        getattr(item, "hard", False)
                        if not isinstance(item, dict)
                        else item.get("hard", False)
                    ),
                    "normalization_value": _safe_float(
                        getattr(normalization, "normalization_coefficient", None)
                        if normalization is not None and not isinstance(normalization, dict)
                        else _safe_dict(normalization).get("normalization_coefficient"),
                        0.0,
                    ),
                    "tolerance": (
                        getattr(item, "tolerance", None)
                        if not isinstance(item, dict)
                        else item.get("tolerance")
                    ),
                }
            )
        if len(targets) > 0:
            return targets

    if structured_goal is None:
        return targets

    for item in list(getattr(structured_goal, "kpi_targets", []) or []):
        semantic_type = _objective_to_semantic_type(getattr(item, "objective", "maximize"))
        target_value = getattr(item, "target_value", None) if semantic_type == "target" else None
        threshold_value = getattr(item, "target_value", None) if semantic_type != "target" else None
        targets.append(
            {
                "kpi_name": str(getattr(item, "kpi_name", "") or ""),
                "semantic_type": semantic_type,
                "target_value": target_value,
                "threshold_value": threshold_value,
                "weight": _safe_float(getattr(item, "weight", 1.0), 1.0),
                "priority": int(_safe_float(getattr(item, "priority", 0), 0.0)),
                "hard": bool(getattr(item, "hard", False)),
                "normalization_value": 0.0,
                "tolerance": getattr(item, "tolerance", None),
            }
        )
    return targets


def compute_intent_loss(
    *,
    observation: Observation | dict[str, Any] | None,
    validated_intent_template: ValidatedIntentTemplate | dict[str, Any] | None = None,
    structured_goal: Any | None = None,
) -> dict[str, Any]:
    targets = _iter_template_targets(
        validated_intent_template=validated_intent_template,
        structured_goal=structured_goal,
    )
    if observation is None or len(targets) == 0:
        return {
            "intent_loss_raw": 0.0,
            "intent_loss_normalized": 0.0,
            "intent_loss_definition_version": INTENT_LOSS_DEFINITION_VERSION,
            "intent_loss_normalization_rule": INTENT_LOSS_NORMALIZATION_RULE,
            "intent_loss_component_count": 0,
            "intent_target_count": 0,
            "intent_target_fulfillment_ratio": 1.0,
            "hard_intent_target_count": 0,
            "hard_intent_target_violation_count": 0,
            "hard_intent_target_fulfillment_ratio": 1.0,
            "hard_intent_fulfilled": True,
            "intent_loss_breakdown": [],
            "intent_loss_approximation_notes": list(INTENT_LOSS_APPROXIMATION_NOTES),
        }

    values_by_kpi = _extract_observation_kpi_values(observation)
    breakdown: list[dict[str, Any]] = []
    weighted_raw_sum = 0.0
    weighted_normalized_sum = 0.0
    target_count = 0
    fulfilled_count = 0
    hard_target_count = 0
    hard_fulfilled_count = 0

    for target in targets:
        kpi_name = str(target.get("kpi_name", "") or "").strip().lower()
        semantic_type = str(target.get("semantic_type", "max") or "max").strip().lower()
        if kpi_name == "":
            continue
        reference_value = (
            target.get("target_value")
            if semantic_type == "target"
            else target.get("threshold_value")
        )
        if reference_value is None:
            continue
        reference_float = _safe_float(reference_value, 0.0)
        actual_value = _safe_float(values_by_kpi.get(kpi_name), 0.0)
        denominator = max(
            _safe_float(target.get("normalization_value"), 0.0),
            _safe_float(target.get("tolerance"), 0.0),
            abs(reference_float),
            1.0,
        )
        if semantic_type == "min":
            raw_gap = max(0.0, actual_value - reference_float)
        elif semantic_type == "target":
            raw_gap = actual_value - reference_float
        else:
            raw_gap = max(0.0, reference_float - actual_value)
        tolerance_value = max(0.0, _safe_float(target.get("tolerance"), 0.0))
        if semantic_type == "target":
            satisfied = abs(float(raw_gap)) <= max(tolerance_value, 1.0e-9)
        else:
            satisfied = float(raw_gap) <= max(tolerance_value, 1.0e-9)
        priority = int(_safe_float(target.get("priority"), 0.0))
        hard = bool(target.get("hard", False)) or priority >= 5
        target_count += 1
        if satisfied:
            fulfilled_count += 1
        if hard:
            hard_target_count += 1
            if satisfied:
                hard_fulfilled_count += 1
        normalized_gap_unclipped = raw_gap / denominator
        normalized_gap = max(
            -INTENT_LOSS_NORMALIZED_GAP_CLIP,
            min(INTENT_LOSS_NORMALIZED_GAP_CLIP, float(normalized_gap_unclipped)),
        )
        penalty = float(normalized_gap) * float(normalized_gap)
        weight = max(0.0, _safe_float(target.get("weight"), 1.0))
        weighted_raw_sum += raw_gap * weight
        weighted_normalized_sum += penalty * weight
        breakdown.append(
            {
                "kpi_name": kpi_name,
                "semantic_type": semantic_type,
                "actual_value": actual_value,
                "reference_value": reference_float,
                "raw_gap": raw_gap,
                "normalized_gap": normalized_gap,
                "normalized_gap_unclipped": normalized_gap_unclipped,
                "normalized_gap_clip": INTENT_LOSS_NORMALIZED_GAP_CLIP,
                "penalty": penalty,
                "weight": weight,
                "priority": priority,
                "hard": bool(hard),
                "satisfied": bool(satisfied),
            }
        )
    hard_violation_count = int(hard_target_count - hard_fulfilled_count)

    return {
        "intent_loss_raw": float(weighted_raw_sum),
        "intent_loss_normalized": float(weighted_normalized_sum),
        "intent_loss_definition_version": INTENT_LOSS_DEFINITION_VERSION,
        "intent_loss_normalization_rule": INTENT_LOSS_NORMALIZATION_RULE,
        "intent_loss_component_count": int(len(breakdown)),
        "intent_target_count": int(target_count),
        "intent_target_fulfillment_ratio": (
            float(fulfilled_count) / float(target_count) if target_count > 0 else 1.0
        ),
        "hard_intent_target_count": int(hard_target_count),
        "hard_intent_target_violation_count": int(hard_violation_count),
        "hard_intent_target_fulfillment_ratio": (
            float(hard_fulfilled_count) / float(hard_target_count)
            if hard_target_count > 0
            else 1.0
        ),
        "hard_intent_fulfilled": bool(hard_violation_count == 0),
        "intent_loss_breakdown": breakdown,
        "intent_loss_approximation_notes": list(INTENT_LOSS_APPROXIMATION_NOTES),
    }


def compute_xapp_switching_cost(orchestration_epoch_history: list[dict[str, Any]]) -> dict[str, Any]:
    transitions = 0
    total_switch_cost = 0.0
    previous_selected_xapps: set[str] | None = None

    for entry in list(orchestration_epoch_history or []):
        current_selected_xapps = {
            str(item)
            for item in list(entry.get("selected_xapps", []) or [])
            if str(item).strip() != ""
        }
        if previous_selected_xapps is not None:
            transitions += 1
            total_switch_cost += float(
                len(current_selected_xapps.symmetric_difference(previous_selected_xapps))
            )
        previous_selected_xapps = current_selected_xapps

    return {
        "xapp_switching_cost_total": float(total_switch_cost),
        "xapp_switching_cost_mean": (
            float(total_switch_cost) / float(transitions) if transitions > 0 else 0.0
        ),
        "xapp_switching_transition_count": int(transitions),
    }


def _normalize_control_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_control_value(child_value)
            for key, child_value in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [_normalize_control_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_control_value(item) for item in value]
    if isinstance(value, float):
        return round(float(value), 6)
    return value


def _extract_control_profile_value(profile: dict[str, Any], field_name: str) -> Any:
    cells = _safe_dict(profile.get("cells"))
    if len(cells) == 0:
        return None
    normalized_values = [
        _normalize_control_value(_safe_dict(cell_profile).get(field_name))
        for _, cell_profile in sorted(cells.items(), key=lambda item: str(item[0]))
    ]
    if len(normalized_values) == 1:
        return normalized_values[0]
    return normalized_values


def _control_delta(field_name: str, previous_value: Any, current_value: Any) -> float:
    if previous_value is None and current_value is None:
        return 0.0
    if previous_value is None or current_value is None:
        return 1.0
    if isinstance(previous_value, (int, float)) and isinstance(current_value, (int, float)):
        denominator = max(float(STABILITY_NUMERIC_SCALES.get(field_name, 1.0)), 1e-6)
        return abs(float(current_value) - float(previous_value)) / denominator
    if previous_value == current_value:
        return 0.0
    return 1.0


def compute_stability_penalty(
    slot_records: list[dict[str, Any]],
    *,
    control_fields: list[str] | None = None,
) -> dict[str, Any]:
    field_names = list(control_fields or STABILITY_CONTROL_FIELDS)
    penalties: list[float] = []

    previous_profile = None
    for record in list(slot_records or []):
        current_profile = _safe_dict(_safe_dict(record.get("control_trace")).get("effective_control_profile"))
        if previous_profile is None or len(current_profile) == 0:
            previous_profile = current_profile
            continue
        transition_penalty = 0.0
        for field_name in field_names:
            transition_penalty += _control_delta(
                field_name,
                _extract_control_profile_value(previous_profile, field_name),
                _extract_control_profile_value(current_profile, field_name),
            )
        penalties.append(transition_penalty / float(len(field_names)) if len(field_names) > 0 else 0.0)
        previous_profile = current_profile

    total_penalty = float(sum(penalties))
    transition_count = int(len(penalties))
    return {
        "stability_penalty_total": total_penalty,
        "stability_penalty_mean": (
            total_penalty / float(transition_count) if transition_count > 0 else 0.0
        ),
        "stability_transition_count": transition_count,
        "stability_penalty_definition_version": STABILITY_PENALTY_DEFINITION_VERSION,
        "stability_control_fields": list(field_names),
    }


def _resolve_epoch_gate(
    slot_id: int,
    orchestration_epoch_history: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_entry: dict[str, Any] | None = None
    for entry in list(orchestration_epoch_history or []):
        if _safe_int(entry.get("slot_id"), -1) <= int(slot_id):
            selected_entry = entry
        else:
            break
    if selected_entry is None:
        return {}
    return _safe_dict(selected_entry.get("exploration_health_gate"))


def compute_safety_violation_metrics(
    slot_records: list[dict[str, Any]],
    *,
    orchestration_epoch_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    evaluated_slot_count = 0
    violation_count = 0
    health_gate_trigger_count = 0
    safe_baseline_fallback_count = 0

    for record in list(slot_records or []):
        slot_id = _safe_int(record.get("slot_id"), -1)
        control_trace = _safe_dict(record.get("control_trace"))
        observable_metrics = _safe_dict(record.get("observable_metrics"))
        gate = _resolve_epoch_gate(slot_id, list(orchestration_epoch_history or []))
        if len(gate) > 0:
            evaluated_slot_count += 1
            violated = False
            max_avg_bler = gate.get("max_avg_bler")
            max_total_queue_bytes = gate.get("max_total_queue_bytes")
            min_total_throughput_mbps = gate.get("min_total_throughput_mbps")
            max_handover_pending_ue_count = gate.get("max_handover_pending_ue_count")
            if max_avg_bler is not None and _safe_float(observable_metrics.get("bler"), 0.0) > _safe_float(max_avg_bler, math.inf):
                violated = True
            if max_total_queue_bytes is not None and _safe_int(observable_metrics.get("queue_bytes"), 0) > _safe_int(max_total_queue_bytes, 0):
                violated = True
            if min_total_throughput_mbps is not None and _safe_float(observable_metrics.get("throughput_mbps"), 0.0) < _safe_float(min_total_throughput_mbps, 0.0):
                violated = True
            if (
                max_handover_pending_ue_count is not None
                and _safe_int(observable_metrics.get("handover_pending_ue_count"), 0)
                > _safe_int(max_handover_pending_ue_count, 0)
            ):
                violated = True
            if violated:
                violation_count += 1

        if len(_safe_list(control_trace.get("rl_health_gate_block_reasons"))) > 0:
            health_gate_trigger_count += 1
        if bool(control_trace.get("rl_safe_baseline_fallback", False)):
            safe_baseline_fallback_count += 1

    slot_count = int(len(slot_records or []))
    return {
        "safety_violation_rate": (
            float(violation_count) / float(evaluated_slot_count)
            if evaluated_slot_count > 0
            else 0.0
        ),
        "safety_violation_count": int(violation_count),
        "safety_violation_evaluated_slot_count": int(evaluated_slot_count),
        "health_gate_trigger_rate": (
            float(health_gate_trigger_count) / float(slot_count)
            if slot_count > 0
            else 0.0
        ),
        "health_gate_trigger_count": int(health_gate_trigger_count),
        "safe_baseline_fallback_rate": (
            float(safe_baseline_fallback_count) / float(slot_count)
            if slot_count > 0
            else 0.0
        ),
        "safe_baseline_fallback_count": int(safe_baseline_fallback_count),
        "safety_violation_definition_version": SAFETY_VIOLATION_DEFINITION_VERSION,
    }


def _series_summary(values: list[float]) -> dict[str, float | None]:
    if len(values) == 0:
        return {"mean": None, "min": None, "max": None, "final": None}
    return {
        "mean": float(sum(values) / float(len(values))),
        "min": float(min(values)),
        "max": float(max(values)),
        "final": float(values[-1]),
    }


def compute_kpi_summary(slot_records: list[dict[str, Any]]) -> dict[str, Any]:
    observable_throughput = [
        _safe_float(_safe_dict(record.get("observable_metrics")).get("throughput_mbps"), 0.0)
        for record in list(slot_records or [])
    ]
    observable_bler = [
        _safe_float(_safe_dict(record.get("observable_metrics")).get("bler"), 0.0)
        for record in list(slot_records or [])
    ]
    observable_queue = [
        _safe_float(_safe_dict(record.get("observable_metrics")).get("queue_bytes"), 0.0)
        for record in list(slot_records or [])
    ]
    observable_sinr = [
        _safe_float(_safe_dict(record.get("observable_metrics")).get("sinr_db"), 0.0)
        for record in list(slot_records or [])
        if _safe_dict(record.get("observable_metrics")).get("sinr_db") is not None
    ]
    observable_hol_delay_ms = [
        _safe_float(_safe_dict(record.get("scheduler_trace")).get("demand_hol_delay_ms_mean"), 0.0)
        for record in list(slot_records or [])
        if _safe_dict(record.get("scheduler_trace")).get("demand_hol_delay_ms_mean") is not None
    ]
    return {
        "mean_observable_throughput_mbps": _series_summary(observable_throughput)["mean"],
        "final_observable_throughput_mbps": _series_summary(observable_throughput)["final"],
        "mean_observable_bler": _series_summary(observable_bler)["mean"],
        "final_observable_bler": _series_summary(observable_bler)["final"],
        "mean_observable_queue_bytes": _series_summary(observable_queue)["mean"],
        "final_observable_queue_bytes": _series_summary(observable_queue)["final"],
        "mean_observable_sinr_db": _series_summary(observable_sinr)["mean"],
        "final_observable_sinr_db": _series_summary(observable_sinr)["final"],
        "mean_scheduler_hol_delay_ms": _series_summary(observable_hol_delay_ms)["mean"],
        "final_scheduler_hol_delay_ms": _series_summary(observable_hol_delay_ms)["final"],
    }


def build_run_metric_summary(
    *,
    slot_records: list[dict[str, Any]],
    orchestration_epoch_history: list[dict[str, Any]],
) -> dict[str, Any]:
    intent_loss_entries = list(orchestration_epoch_history or [])
    raw_values = [
        _safe_float(entry.get("intent_loss_raw"), 0.0)
        for entry in intent_loss_entries
        if entry.get("intent_loss_raw") is not None
    ]
    normalized_values = [
        _safe_float(entry.get("intent_loss_normalized"), 0.0)
        for entry in intent_loss_entries
        if entry.get("intent_loss_normalized") is not None
    ]
    target_fulfillment_values = [
        _safe_float(entry.get("intent_target_fulfillment_ratio"), 1.0)
        for entry in intent_loss_entries
        if entry.get("intent_target_fulfillment_ratio") is not None
    ]
    hard_target_fulfillment_values = [
        _safe_float(entry.get("hard_intent_target_fulfillment_ratio"), 1.0)
        for entry in intent_loss_entries
        if entry.get("hard_intent_target_fulfillment_ratio") is not None
    ]
    hard_target_violation_counts = [
        _safe_float(entry.get("hard_intent_target_violation_count"), 0.0)
        for entry in intent_loss_entries
        if entry.get("hard_intent_target_violation_count") is not None
    ]
    hard_intent_fulfilled_values = [
        1.0 if bool(entry.get("hard_intent_fulfilled", True)) else 0.0
        for entry in intent_loss_entries
        if entry.get("hard_intent_fulfilled") is not None
    ]
    switching_summary = compute_xapp_switching_cost(orchestration_epoch_history)
    stability_summary = compute_stability_penalty(slot_records)
    safety_summary = compute_safety_violation_metrics(
        slot_records,
        orchestration_epoch_history=orchestration_epoch_history,
    )
    kpi_summary = compute_kpi_summary(slot_records)
    return {
        "intent_loss_raw": raw_values[-1] if len(raw_values) > 0 else 0.0,
        "intent_loss_normalized": normalized_values[-1] if len(normalized_values) > 0 else 0.0,
        "intent_loss_mean_raw": (
            float(sum(raw_values) / float(len(raw_values))) if len(raw_values) > 0 else 0.0
        ),
        "intent_loss_mean_normalized": (
            float(sum(normalized_values) / float(len(normalized_values)))
            if len(normalized_values) > 0
            else 0.0
        ),
        "intent_loss_max_normalized": max(normalized_values) if len(normalized_values) > 0 else 0.0,
        "intent_target_fulfillment_ratio": (
            target_fulfillment_values[-1] if len(target_fulfillment_values) > 0 else 1.0
        ),
        "intent_target_fulfillment_mean": (
            float(sum(target_fulfillment_values) / float(len(target_fulfillment_values)))
            if len(target_fulfillment_values) > 0
            else 1.0
        ),
        "hard_target_fulfillment_ratio": (
            hard_target_fulfillment_values[-1]
            if len(hard_target_fulfillment_values) > 0
            else 1.0
        ),
        "hard_target_fulfillment_mean": (
            float(
                sum(hard_target_fulfillment_values)
                / float(len(hard_target_fulfillment_values))
            )
            if len(hard_target_fulfillment_values) > 0
            else 1.0
        ),
        "hard_target_violation_count_mean": (
            float(sum(hard_target_violation_counts) / float(len(hard_target_violation_counts)))
            if len(hard_target_violation_counts) > 0
            else 0.0
        ),
        "hard_intent_fulfillment_ratio": (
            float(sum(hard_intent_fulfilled_values) / float(len(hard_intent_fulfilled_values)))
            if len(hard_intent_fulfilled_values) > 0
            else 1.0
        ),
        "intent_loss_definition_version": INTENT_LOSS_DEFINITION_VERSION,
        "intent_loss_epoch_count": int(len(normalized_values)),
        **switching_summary,
        **stability_summary,
        **safety_summary,
        "kpi_summary": kpi_summary,
        "recovery_time_after_environment_change_slots": None,
        "recovery_time_definition_version": "pending_environment_change_hook",
    }


def compute_cross_scenario_generalization_gap(
    case_records: list[dict[str, Any]],
    *,
    reference_scenario_name: str = "benchmark_nominal_cluster",
    metric_field: str = "summary_experimental_metrics_intent_loss_mean_normalized",
) -> dict[str, Any]:
    def _scenario_key(record: dict[str, Any]) -> str:
        return str(record.get("scenario_case_id", "") or record.get("scenario_name", "") or "")

    reference_record = next(
        (
            record
            for record in list(case_records or [])
            if _scenario_key(record) == str(reference_scenario_name)
        ),
        None,
    )
    if reference_record is None or metric_field not in reference_record:
        return {
            "reference_scenario_name": str(reference_scenario_name),
            "scenario_key_field_used": "scenario_case_id_or_scenario_name",
            "metric_field": str(metric_field),
            "definition_version": GENERALIZATION_GAP_DEFINITION_VERSION,
            "reference_metric_value": None,
            "mean_gap": None,
            "per_scenario_gap": {},
        }
    reference_value = _safe_float(reference_record.get(metric_field), 0.0)
    per_scenario_gap: dict[str, float] = {}
    gaps: list[float] = []
    for record in list(case_records or []):
        scenario_key = _scenario_key(record)
        if scenario_key == "" or metric_field not in record:
            continue
        gap = _safe_float(record.get(metric_field), 0.0) - reference_value
        per_scenario_gap[scenario_key] = float(gap)
        if scenario_key != str(reference_scenario_name):
            gaps.append(float(gap))
    return {
        "reference_scenario_name": str(reference_scenario_name),
        "scenario_key_field_used": "scenario_case_id_or_scenario_name",
        "metric_field": str(metric_field),
        "definition_version": GENERALIZATION_GAP_DEFINITION_VERSION,
        "reference_metric_value": float(reference_value),
        "mean_gap": float(sum(gaps) / float(len(gaps))) if len(gaps) > 0 else 0.0,
        "per_scenario_gap": per_scenario_gap,
    }


def effective_control_profile_to_json(profile: dict[str, Any]) -> str:
    return json.dumps(_normalize_control_value(profile), ensure_ascii=False, sort_keys=True)
