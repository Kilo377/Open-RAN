"""Helpers shared by intent-driven orchestration rApps."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from typing import Any

from models.intent_orchestration import (
    ExplorationHealthGate,
    ForbiddenFieldCombination,
    GoalKpiTarget,
    RewardParameterRange,
    StructuredIntentGoal,
    ValidatedIntentTemplate,
    XAppCandidateScore,
)
from models.observation import Observation
from ric.near_rt.xapp_variant_catalog import (
    BASE_XAPP_CAPABILITIES,
    DEFAULT_XAPP_VARIANTS,
    build_variant_capability_catalog,
    canonical_xapp_category,
)


XAPP_CAPABILITY_CATALOG: dict[str, dict[str, object]] = build_variant_capability_catalog()


def _variant_params_catalog() -> dict[str, dict[str, object]]:
    catalog: dict[str, dict[str, object]] = {}
    for variants in DEFAULT_XAPP_VARIANTS.values():
        for variant in variants:
            variant_name = str(variant.get("name", "")).strip()
            if variant_name == "":
                continue
            params = variant.get("params", {})
            if not isinstance(params, dict):
                params = {}
            catalog[variant_name] = dict(params)
    return catalog


XAPP_VARIANT_PARAMS_CATALOG: dict[str, dict[str, object]] = _variant_params_catalog()
INTENT_LOSS_DEFINITION_VERSION = "v2_paper_intent_loss"


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def stable_intent_id(intent_text: str) -> str:
    normalized_intent = str(intent_text or "").strip().lower()
    digest = hashlib.sha1(normalized_intent.encode("utf-8")).hexdigest()[:12]
    return f"intent:{digest}"


def to_llm_serializable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return to_llm_serializable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_llm_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_llm_serializable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def llm_json_dumps(value: object) -> str:
    return json.dumps(
        to_llm_serializable(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


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


def observation_kpi_values(observation: Observation) -> dict[str, float]:
    cell_observations = list(getattr(observation, "cell_observations", []) or [])
    ue_observations = list(getattr(observation, "ue_observations", []) or [])
    queue_bytes = float(getattr(observation, "total_queue_bytes", 0) or 0.0)
    throughput_mbps = float(getattr(observation, "total_throughput_mbps", 0.0) or 0.0)
    avg_bler = float(getattr(observation, "avg_bler", 0.0) or 0.0)
    avg_sinr_db = float(getattr(observation, "avg_sinr_db", 0.0) or 0.0)
    connected_ue_count = max(1.0, float(getattr(observation, "connected_ue_count", 1) or 1.0))
    pending_ratio = float(getattr(observation, "handover_pending_ue_count", 0) or 0.0) / connected_ue_count
    power_values = [
        float(
            getattr(cell, "power_cap_dbm", getattr(cell, "tx_power_dbm", 30.0)) or 30.0
        )
        for cell in cell_observations
    ]
    power_cost = float(sum(power_values) / float(len(power_values))) if len(power_values) > 0 else 30.0
    hol_delays = [
        _safe_float(getattr(ue, "demand_hol_delay_ms", None), 0.0)
        for ue in ue_observations
        if getattr(ue, "demand_hol_delay_ms", None) is not None
    ]
    scheduler_hol_delay_ms = (
        float(sum(hol_delays) / len(hol_delays))
        if len(hol_delays) > 0
        else 0.0
    )
    return {
        "queue_bytes": queue_bytes,
        "total_queue_bytes": queue_bytes,
        "total_throughput_mbps": throughput_mbps,
        "avg_bler": avg_bler,
        "sinr": avg_sinr_db,
        "avg_sinr_db": avg_sinr_db,
        "handover_risk": pending_ratio,
        "power_cost": power_cost,
        "scheduler_hol_delay_ms": scheduler_hol_delay_ms,
        "demand_hol_delay_ms": scheduler_hol_delay_ms,
        "delay_ms": scheduler_hol_delay_ms,
    }


def _goal_target_penalty(
    *,
    kpi_name: str,
    semantic_type: str,
    reference_value: float,
    normalization_value: float,
    actual_value: float,
) -> float:
    denominator = max(abs(normalization_value), 1e-6)
    if semantic_type == "min":
        gap = max(0.0, actual_value - reference_value)
    elif semantic_type == "target":
        gap = actual_value - reference_value
    else:
        gap = max(0.0, reference_value - actual_value)
    normalized_gap = float(gap) / denominator
    return float(normalized_gap * normalized_gap)


def goal_gap_breakdown(
    goal: StructuredIntentGoal,
    observation: Observation,
    *,
    validated_intent_template: ValidatedIntentTemplate | None = None,
) -> dict[str, float]:
    values_by_kpi = observation_kpi_values(observation)
    breakdown: dict[str, float] = {}
    if validated_intent_template is not None:
        for item in list(getattr(validated_intent_template, "kpi_template", []) or []):
            kpi_name = str(getattr(item, "kpi_name", "") or "").strip().lower()
            if kpi_name == "":
                continue
            semantic_type = str(getattr(item, "semantic_type", "max") or "max").strip().lower()
            reference_value = (
                getattr(item, "target_value", None)
                if semantic_type == "target"
                else getattr(item, "threshold_value", None)
            )
            if reference_value is None:
                continue
            normalization = getattr(item, "normalization", None)
            normalization_value = _safe_float(
                getattr(normalization, "normalization_coefficient", None),
                0.0,
            )
            if normalization_value <= 0.0:
                normalization_value = _safe_float(
                    getattr(normalization, "reference_value", None),
                    0.0,
                )
            if normalization_value <= 0.0:
                normalization_value = _safe_float(getattr(item, "tolerance", None), 0.0)
            if normalization_value <= 0.0:
                normalization_value = max(abs(_safe_float(reference_value, 0.0)), 1.0)
            breakdown[kpi_name] = _goal_target_penalty(
                kpi_name=kpi_name,
                semantic_type=semantic_type,
                reference_value=_safe_float(reference_value, 0.0),
                normalization_value=float(normalization_value),
                actual_value=_safe_float(values_by_kpi.get(kpi_name), 0.0),
            )
        if len(breakdown) > 0:
            return breakdown

    for target in list(getattr(goal, "kpi_targets", []) or []):
        kpi_name = str(getattr(target, "kpi_name", "")).strip().lower()
        target_value = getattr(target, "target_value", None)
        if kpi_name == "" or target_value is None:
            continue
        objective = str(getattr(target, "objective", "maximize")).strip().lower()
        semantic_type = "target"
        if objective in {"maximize", "max"}:
            semantic_type = "max"
        elif objective in {"minimize", "min"}:
            semantic_type = "min"
        normalization_value = _safe_float(getattr(target, "tolerance", None), 0.0)
        if normalization_value <= 0.0:
            normalization_value = max(abs(_safe_float(target_value, 0.0)), 1.0)
        breakdown[kpi_name] = _goal_target_penalty(
            kpi_name=kpi_name,
            semantic_type=semantic_type,
            reference_value=_safe_float(target_value, 0.0),
            normalization_value=float(normalization_value),
            actual_value=_safe_float(values_by_kpi.get(kpi_name), 0.0),
        )
    return breakdown


def goal_gap_score(
    goal: StructuredIntentGoal,
    observation: Observation,
    *,
    validated_intent_template: ValidatedIntentTemplate | None = None,
) -> float:
    weighted_sum = 0.0
    breakdown = goal_gap_breakdown(
        goal,
        observation,
        validated_intent_template=validated_intent_template,
    )
    if validated_intent_template is not None and len(
        list(getattr(validated_intent_template, "kpi_template", []) or [])
    ) > 0:
        for item in list(getattr(validated_intent_template, "kpi_template", []) or []):
            kpi_name = str(getattr(item, "kpi_name", "")).strip().lower()
            if kpi_name not in breakdown:
                continue
            weight = max(0.0, _safe_float(getattr(item, "weight", 1.0), 1.0))
            weighted_sum += float(breakdown[kpi_name]) * weight
        return float(weighted_sum)
    for target in list(getattr(goal, "kpi_targets", []) or []):
        kpi_name = str(getattr(target, "kpi_name", "")).strip().lower()
        if kpi_name not in breakdown:
            continue
        weight = max(0.0, float(getattr(target, "weight", 1.0) or 1.0))
        weighted_sum += float(breakdown[kpi_name]) * weight
    return float(weighted_sum)


def observation_summary_for_llm(observation: Observation) -> dict[str, Any]:
    ue_observations = list(getattr(observation, "ue_observations", []) or [])
    cell_observations = list(getattr(observation, "cell_observations", []) or [])
    top_ues = sorted(
        ue_observations,
        key=lambda ue: float(getattr(ue, "queue_bytes", 0) or 0.0),
        reverse=True,
    )[:4]
    return {
        "slot_id": int(getattr(observation, "slot_id", -1)),
        "runtime_mode": str(getattr(observation, "runtime_mode", "single_active_cell")),
        "avg_sinr_db": float(getattr(observation, "avg_sinr_db", 0.0) or 0.0),
        "avg_bler": float(getattr(observation, "avg_bler", 0.0) or 0.0),
        "total_queue_bytes": int(getattr(observation, "total_queue_bytes", 0) or 0),
        "total_throughput_mbps": float(getattr(observation, "total_throughput_mbps", 0.0) or 0.0),
        "handover_pending_ue_count": int(getattr(observation, "handover_pending_ue_count", 0) or 0),
        "cells": [
            {
                "cell_id": str(getattr(cell, "cell_id", "cell_0")),
                "queue_bytes": int(getattr(cell, "total_queue_bytes", 0) or 0),
                "utilization": float(getattr(cell, "resource_grid_utilization", 0.0) or 0.0),
                "interference_db": float(
                    getattr(cell, "inter_cell_interference_penalty_db_mean", 0.0) or 0.0
                ),
                "scheduler_mode": str(getattr(cell, "scheduler_mode", "round_robin")),
            }
            for cell in cell_observations[:4]
        ],
        "top_ues": [
            {
                "ue_id": str(getattr(ue, "ue_id", "ue")),
                "slice_id": str(getattr(ue, "slice_id", "")),
                "traffic_type": str(getattr(ue, "traffic_type", "")),
                "queue_bytes": int(getattr(ue, "queue_bytes", 0) or 0),
                "sinr_db": float(getattr(ue, "sinr_db", 0.0) or 0.0),
                "bler": float(getattr(ue, "bler", 0.0) or 0.0),
                "serving_cell_id": str(getattr(ue, "serving_cell_id", "")),
            }
            for ue in top_ues
        ],
    }


def _normalize_available_xapp_filter(
    available_xapp_names: list[str] | tuple[str, ...] | set[str] | None,
) -> set[str] | None:
    if available_xapp_names is None:
        return None
    normalized = {
        str(item).strip()
        for item in list(available_xapp_names)
        if str(item).strip() != ""
    }
    return normalized


def _xapp_allowed_for_filter(
    xapp_name: str,
    *,
    available_filter: set[str] | None,
) -> bool:
    if available_filter is None:
        return True
    normalized_name = str(xapp_name).strip()
    if normalized_name == "":
        return False
    category_name = canonical_xapp_category(normalized_name)
    return normalized_name in available_filter or category_name in available_filter


def available_xapp_catalog_for_llm(
    available_xapp_names: list[str] | tuple[str, ...] | set[str] | None = None,
) -> list[dict[str, object]]:
    available_filter = _normalize_available_xapp_filter(available_xapp_names)
    return [
        {
            "xapp_name": name,
            "xapp_category": canonical_xapp_category(name),
            "description": str(spec.get("description", "")),
            "kpis": _string_list(spec.get("kpis", [])),
            "tunable_fields": _string_list(spec.get("tunable_fields", [])),
            "tuning_mode": str(spec.get("tuning_mode", "heuristic_parameter")),
            "is_variant": bool(canonical_xapp_category(name) != name),
            "variant_params": to_llm_serializable(XAPP_VARIANT_PARAMS_CATALOG.get(name, {})),
        }
        for name, spec in sorted(XAPP_CAPABILITY_CATALOG.items())
        if _xapp_allowed_for_filter(name, available_filter=available_filter)
    ]


def available_xapp_category_catalog_for_llm(
    available_xapp_names: list[str] | tuple[str, ...] | set[str] | None = None,
) -> list[dict[str, object]]:
    available_filter = _normalize_available_xapp_filter(available_xapp_names)
    catalog: list[dict[str, object]] = []
    for name, spec in sorted(BASE_XAPP_CAPABILITIES.items()):
        variant_names = [
            str(variant.get("name", ""))
            for variant in DEFAULT_XAPP_VARIANTS.get(name, [])
            if str(variant.get("name", "")).strip() != ""
            and _xapp_allowed_for_filter(
                str(variant.get("name", "")),
                available_filter=available_filter,
            )
        ]
        if available_filter is not None and name not in available_filter and len(variant_names) == 0:
            continue
        catalog.append(
            {
                "xapp_category": name,
                "description": str(spec.get("description", "")),
                "kpis": _string_list(spec.get("kpis", [])),
                "tunable_fields": _string_list(spec.get("tunable_fields", [])),
                "variant_count": int(len(variant_names)),
                "example_variants": variant_names[:4],
            }
        )
    return catalog


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
    text = str(raw_text or "").strip()
    if text == "":
        return None
    candidates: list[dict[str, Any]] = []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    text_length = len(text)
    for start in range(text_length):
        if text[start] != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for end in range(start, text_length):
            char = text[end]
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == "\"":
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate_text = text[start : end + 1]
                    try:
                        parsed = json.loads(candidate_text)
                        if isinstance(parsed, dict):
                            candidates.append(parsed)
                    except Exception:
                        pass
                    break

    if len(candidates) == 0:
        return None
    candidates.sort(
        key=lambda item: (len(item.keys()), len(json.dumps(item, ensure_ascii=False))),
        reverse=True,
    )
    return candidates[0]


def heuristic_goal_from_intent(intent_text: str, observation: Observation) -> StructuredIntentGoal:
    normalized_intent = str(intent_text or "").strip()
    lower_intent = normalized_intent.lower()
    intent_id = stable_intent_id(normalized_intent)
    kpis: list[GoalKpiTarget] = [
        GoalKpiTarget(
            kpi_name="queue_bytes",
            objective="minimize",
            target_value=max(0.0, float(getattr(observation, "total_queue_bytes", 0) or 0.0) * 0.7),
            weight=1.0,
            unit="bytes",
            priority=1,
        )
    ]
    risk_posture = "balanced"

    if any(token in lower_intent for token in ["urllc", "latency", "delay", "control"]):
        kpis.append(
            GoalKpiTarget(
                kpi_name="avg_bler",
                objective="minimize",
                target_value=0.12,
                weight=1.4,
                unit="ratio",
                priority=3,
            )
        )
        kpis.append(
            GoalKpiTarget(
                kpi_name="total_queue_bytes",
                objective="minimize",
                target_value=max(0.0, float(getattr(observation, "total_queue_bytes", 0) or 0.0) * 0.5),
                weight=1.2,
                unit="bytes",
                priority=3,
            )
        )
        risk_posture = "reliability_first"

    if any(token in lower_intent for token in ["throughput", "embb", "video", "capacity"]):
        kpis.append(
            GoalKpiTarget(
                kpi_name="total_throughput_mbps",
                objective="maximize",
                target_value=max(10.0, float(getattr(observation, "total_throughput_mbps", 0.0) or 0.0) * 1.2),
                weight=1.3,
                unit="mbps",
                priority=2,
            )
        )

    if any(token in lower_intent for token in ["energy", "power", "saving", "green"]):
        kpis.append(
            GoalKpiTarget(
                kpi_name="power_cost",
                objective="minimize",
                target_value=28.0,
                weight=0.9,
                unit="dbm_like",
                priority=1,
            )
        )

    if any(token in lower_intent for token in ["mobility", "handover", "steering", "cell-edge", "edge"]):
        kpis.append(
            GoalKpiTarget(
                kpi_name="handover_risk",
                objective="minimize",
                target_value=0.25,
                weight=1.1,
                unit="score",
                priority=2,
            )
        )

    return StructuredIntentGoal(
        goal_id=f"goal:{intent_id.split(':', 1)[-1]}",
        intent_id=intent_id,
        intent_text=normalized_intent,
        summary=f"Intent-derived goal for slot {int(getattr(observation, 'slot_id', 0) or 0)}",
        horizon_slots=12,
        risk_posture=risk_posture,
        target_entities=["cluster"],
        kpi_targets=kpis,
        constraints={"source": "heuristic_intent_parser"},
        generated_by="heuristic",
    )


def heuristic_xapp_candidates_for_goal(goal: StructuredIntentGoal) -> list[XAppCandidateScore]:
    lower_intent = str(goal.intent_text or "").lower()
    requested_kpis = {str(kpi.kpi_name).lower() for kpi in list(goal.kpi_targets or [])}
    candidates: list[XAppCandidateScore] = []
    for xapp_name, spec in sorted(XAPP_CAPABILITY_CATALOG.items()):
        spec_kpis = _string_list(spec.get("kpis", []))
        kpi_hits = sum(
            1
            for kpi_name in spec_kpis
            if any(token in kpi_name for token in requested_kpis)
        )
        intent_hits = sum(
            1
            for token in [
                "latency",
                "throughput",
                "power",
                "energy",
                "handover",
                "steering",
                "slice",
                "reliability",
                "sinr",
            ]
            if token in lower_intent and token in str(spec.get("description", "")).lower()
        )
        confidence = min(1.0, 0.20 + 0.18 * kpi_hits + 0.10 * intent_hits)
        candidates.append(
            XAppCandidateScore(
                xapp_name=xapp_name,
                rationale=str(spec.get("description", "")),
                expected_kpi_impact={
                    str(kpi_name): confidence
                    for kpi_name in spec_kpis[:3]
                },
                confidence=confidence,
                selected=False,
            )
        )
    return candidates


def _goal_targets_kpi(goal: StructuredIntentGoal | None, kpi_name: str) -> GoalKpiTarget | None:
    if goal is None:
        return None
    normalized_kpi_name = str(kpi_name or "").strip().lower()
    for target in list(getattr(goal, "kpi_targets", []) or []):
        if str(getattr(target, "kpi_name", "")).strip().lower() == normalized_kpi_name:
            return target
    return None


def _goal_prefers_throughput(goal: StructuredIntentGoal | None) -> bool:
    throughput_target = _goal_targets_kpi(goal, "total_throughput_mbps")
    if throughput_target is not None and str(getattr(throughput_target, "objective", "maximize")).strip().lower() == "maximize":
        return True
    intent_text = str(getattr(goal, "intent_text", "") or "").lower()
    summary_text = str(getattr(goal, "summary", "") or "").lower()
    combined = f"{intent_text} {summary_text}"
    return any(token in combined for token in ("throughput", "capacity", "spectral efficiency", "吞吐"))


def _goal_prefers_reliability(goal: StructuredIntentGoal | None) -> bool:
    bler_target = _goal_targets_kpi(goal, "avg_bler")
    if bler_target is not None and str(getattr(bler_target, "objective", "minimize")).strip().lower() == "minimize":
        return True
    intent_text = str(getattr(goal, "intent_text", "") or "").lower()
    summary_text = str(getattr(goal, "summary", "") or "").lower()
    combined = f"{intent_text} {summary_text}"
    return any(token in combined for token in ("urllc", "reliability", "latency", "稳定", "可靠"))


def _range_scope_for_xapp(xapp_name: str) -> str:
    category = canonical_xapp_category(xapp_name)
    if category in {"mobility_predictor_xapp", "traffic_steering_xapp", "ran_slicing_manager_xapp"}:
        return "cluster"
    return "cell"


def default_reward_parameter_ranges(selected_xapps: list[str]) -> list[RewardParameterRange]:
    ranges: list[RewardParameterRange] = []
    for xapp_name in list(selected_xapps or []):
        spec = XAPP_CAPABILITY_CATALOG.get(str(xapp_name), {})
        for field_name in _string_list(spec.get("tunable_fields", [])):
            if field_name == "latency_bias":
                ranges.append(
                    RewardParameterRange(
                        field_name=field_name,
                        target_xapp_name=xapp_name,
                        lower_bound=0.8,
                        upper_bound=2.0,
                        step=0.1,
                        tuning_mode="rl_parameter",
                    )
                )
            elif field_name == "power_cap_dbm":
                ranges.append(
                    RewardParameterRange(
                        field_name=field_name,
                        target_xapp_name=xapp_name,
                        lower_bound=24.0,
                        upper_bound=32.0,
                        step=0.5,
                        tuning_mode="rl_parameter",
                    )
                )
            elif field_name == "mobility_bias_db":
                ranges.append(
                    RewardParameterRange(
                        field_name=field_name,
                        target_xapp_name=xapp_name,
                        lower_bound=-1.0,
                        upper_bound=6.0,
                        step=0.5,
                        tuning_mode="rl_parameter",
                    )
                )
            elif field_name == "measurement_event_offset_db":
                ranges.append(
                    RewardParameterRange(
                        field_name=field_name,
                        target_xapp_name=xapp_name,
                        lower_bound=0.0,
                        upper_bound=5.0,
                        step=0.5,
                        tuning_mode="rl_parameter",
                    )
                )
            elif field_name == "steering_hysteresis_db":
                ranges.append(
                    RewardParameterRange(
                        field_name=field_name,
                        target_xapp_name=xapp_name,
                        lower_bound=0.5,
                        upper_bound=4.0,
                        step=0.5,
                        tuning_mode="rl_parameter",
                    )
                )
            elif field_name == "slice_weight_adjustment":
                ranges.append(
                    RewardParameterRange(
                        field_name=field_name,
                        target_xapp_name=xapp_name,
                        lower_bound=0.70,
                        upper_bound=1.60,
                        step=0.05,
                        tuning_mode="rl_parameter",
                    )
                )
            elif field_name in {"scheduler_policy_hint", "beam_candidate_preference", "traffic_steering_preference"}:
                discrete_values = {
                    "scheduler_policy_hint": ["balanced", "latency_bias", "slice_bias"],
                    "beam_candidate_preference": ["balanced", "coverage", "capacity"],
                    "traffic_steering_preference": ["balanced", "edge_offload", "stay_put"],
                }[field_name]
                ranges.append(
                    RewardParameterRange(
                        field_name=field_name,
                        target_xapp_name=xapp_name,
                        discrete_values=list(discrete_values),
                        tuning_mode="rl_parameter",
                    )
                )
            elif field_name == "cell_sleep_mode":
                ranges.append(
                    RewardParameterRange(
                        field_name=field_name,
                        target_xapp_name=xapp_name,
                        discrete_values=["active", "micro_sleep", "light_sleep", "deep_sleep"],
                        tuning_mode="rl_parameter",
                    )
                )
    return ranges


def default_hard_safety_envelope(
    selected_xapps: list[str],
    goal: StructuredIntentGoal | None = None,
) -> list[RewardParameterRange]:
    throughput_focused = _goal_prefers_throughput(goal)
    reliability_focused = _goal_prefers_reliability(goal)
    risk_posture = str(getattr(goal, "risk_posture", "balanced") or "balanced").strip().lower()
    hard_ranges: list[RewardParameterRange] = []
    for soft_range in default_reward_parameter_ranges(selected_xapps):
        field_name = str(getattr(soft_range, "field_name", "")).strip()
        xapp_name = str(getattr(soft_range, "target_xapp_name", "")).strip()
        if field_name == "" or xapp_name == "":
            continue
        category = canonical_xapp_category(xapp_name)
        scope = _range_scope_for_xapp(xapp_name)
        if field_name == "latency_bias":
            lower_bound = 1.00 if reliability_focused else 0.95
            upper_bound = 1.85 if category in {"latency_scheduler_xapp", "urllc_guard_xapp"} else 1.55
            if throughput_focused and category in {"latency_scheduler_xapp", "urllc_guard_xapp"}:
                upper_bound = min(1.70, upper_bound)
            hard_ranges.append(
                RewardParameterRange(
                    field_name=field_name,
                    target_xapp_name=xapp_name,
                    scope=scope,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    step=0.05,
                    max_delta=0.20,
                    tuning_mode="safety_envelope",
                )
            )
        elif field_name == "power_cap_dbm":
            lower_bound = 28.0 if reliability_focused else 27.0
            upper_bound = 30.0 if risk_posture != "aggressive" else 30.5
            if throughput_focused and category == "massive_mimo_beamformer_xapp":
                upper_bound = 30.5
            hard_ranges.append(
                RewardParameterRange(
                    field_name=field_name,
                    target_xapp_name=xapp_name,
                    scope=scope,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    step=0.25,
                    max_delta=0.75,
                    tuning_mode="safety_envelope",
                )
            )
        elif field_name == "mobility_bias_db":
            hard_ranges.append(
                RewardParameterRange(
                    field_name=field_name,
                    target_xapp_name=xapp_name,
                    scope=scope,
                    lower_bound=0.0,
                    upper_bound=4.5 if throughput_focused else 3.5,
                    step=0.25,
                    max_delta=0.75,
                    tuning_mode="safety_envelope",
                )
            )
        elif field_name == "measurement_event_offset_db":
            hard_ranges.append(
                RewardParameterRange(
                    field_name=field_name,
                    target_xapp_name=xapp_name,
                    scope=scope,
                    lower_bound=0.5,
                    upper_bound=2.5,
                    step=0.25,
                    max_delta=0.40,
                    tuning_mode="safety_envelope",
                )
            )
        elif field_name == "steering_hysteresis_db":
            hard_ranges.append(
                RewardParameterRange(
                    field_name=field_name,
                    target_xapp_name=xapp_name,
                    scope=scope,
                    lower_bound=1.0,
                    upper_bound=3.0,
                    step=0.25,
                    max_delta=0.50,
                    tuning_mode="safety_envelope",
                )
            )
        elif field_name == "slice_weight_adjustment":
            hard_ranges.append(
                RewardParameterRange(
                    field_name=field_name,
                    target_xapp_name=xapp_name,
                    scope=scope,
                    lower_bound=0.85,
                    upper_bound=1.35 if throughput_focused else 1.25,
                    step=0.05,
                    max_delta=0.10,
                    tuning_mode="safety_envelope",
                )
            )
        elif field_name == "scheduler_policy_hint":
            allowed_values = ["balanced"]
            if category in {"latency_scheduler_xapp", "urllc_guard_xapp"}:
                allowed_values = ["balanced", "latency_bias"]
            elif category in {"spectrum_optimizer_xapp", "ran_slicing_manager_xapp"}:
                allowed_values = ["balanced", "slice_bias"]
            hard_ranges.append(
                RewardParameterRange(
                    field_name=field_name,
                    target_xapp_name=xapp_name,
                    scope=scope,
                    discrete_values=allowed_values,
                    blocked_discrete_values=[
                        value
                        for value in ["balanced", "latency_bias", "slice_bias"]
                        if value not in allowed_values
                    ],
                    tuning_mode="safety_envelope",
                )
            )
        elif field_name == "beam_candidate_preference":
            allowed_values = ["balanced", "coverage"]
            if throughput_focused and category == "massive_mimo_beamformer_xapp":
                allowed_values = ["balanced", "capacity", "coverage"]
            elif category == "power_saving_xapp":
                allowed_values = ["balanced", "coverage"]
            hard_ranges.append(
                RewardParameterRange(
                    field_name=field_name,
                    target_xapp_name=xapp_name,
                    scope=scope,
                    discrete_values=allowed_values,
                    blocked_discrete_values=[
                        value
                        for value in ["balanced", "coverage", "capacity"]
                        if value not in allowed_values
                    ],
                    tuning_mode="safety_envelope",
                )
            )
        elif field_name == "traffic_steering_preference":
            allowed_values = ["balanced", "edge_offload"] if throughput_focused else ["balanced", "stay_put"]
            hard_ranges.append(
                RewardParameterRange(
                    field_name=field_name,
                    target_xapp_name=xapp_name,
                    scope=scope,
                    discrete_values=allowed_values,
                    blocked_discrete_values=[
                        value
                        for value in ["balanced", "edge_offload", "stay_put"]
                        if value not in allowed_values
                    ],
                    tuning_mode="safety_envelope",
                )
            )
        elif field_name == "cell_sleep_mode":
            hard_ranges.append(
                RewardParameterRange(
                    field_name=field_name,
                    target_xapp_name=xapp_name,
                    scope=scope,
                    discrete_values=["active", "micro_sleep"],
                    blocked_discrete_values=["light_sleep", "deep_sleep"],
                    tuning_mode="safety_envelope",
                )
            )
    return hard_ranges


def default_forbidden_combinations(selected_xapps: list[str]) -> list[ForbiddenFieldCombination]:
    field_space = {
        str(parameter_range.field_name)
        for parameter_range in default_reward_parameter_ranges(selected_xapps)
    }
    forbidden_combinations: list[ForbiddenFieldCombination] = []
    if {"scheduler_policy_hint", "beam_candidate_preference"}.issubset(field_space):
        forbidden_combinations.append(
            ForbiddenFieldCombination(
                field_values={
                    "scheduler_policy_hint": "latency_bias",
                    "beam_candidate_preference": "capacity",
                },
                scope="cell",
                reason="Avoid combining aggressive latency bias with capacity-only beams during online exploration.",
            )
        )
    if {"traffic_steering_preference", "beam_candidate_preference"}.issubset(field_space):
        forbidden_combinations.append(
            ForbiddenFieldCombination(
                field_values={
                    "traffic_steering_preference": "edge_offload",
                    "beam_candidate_preference": "capacity",
                },
                scope="cell",
                reason="Avoid aggressive offload steering and capacity-only beams in the same step.",
            )
        )
    if {"scheduler_policy_hint", "cell_sleep_mode"}.issubset(field_space):
        forbidden_combinations.extend(
            [
                ForbiddenFieldCombination(
                    field_values={
                        "scheduler_policy_hint": "latency_bias",
                        "cell_sleep_mode": sleep_mode,
                    },
                    scope="cell",
                    reason="Do not combine low-latency scheduling with deep cell sleep transitions.",
                )
                for sleep_mode in ("light_sleep", "deep_sleep")
            ]
        )
    return forbidden_combinations


def default_exploration_health_gate(goal: StructuredIntentGoal | None = None) -> ExplorationHealthGate:
    risk_posture = str(getattr(goal, "risk_posture", "balanced") or "balanced").strip().lower()
    throughput_target = _goal_targets_kpi(goal, "total_throughput_mbps")
    min_throughput = None
    if throughput_target is not None and getattr(throughput_target, "target_value", None) is not None:
        min_throughput = float(getattr(throughput_target, "target_value")) * 0.55
    max_queue = None
    queue_target = _goal_targets_kpi(goal, "total_queue_bytes") or _goal_targets_kpi(goal, "queue_bytes")
    if (
        risk_posture == "conservative"
        and queue_target is not None
        and getattr(queue_target, "target_value", None) is not None
    ):
        queue_target_value = max(0.0, float(getattr(queue_target, "target_value") or 0.0))
        max_queue = max(1_000_000, int(queue_target_value * 3.00))
    presets = {
        "conservative": {
            "max_avg_bler": 0.08,
            "max_handover_pending_ue_count": 32,
            "freeze_slots_on_violation": 6,
        },
        "aggressive": {
            "max_avg_bler": 0.16,
            "max_handover_pending_ue_count": 96,
            "freeze_slots_on_violation": 2,
        },
    }
    defaults = {
        "max_avg_bler": 0.12,
        "max_handover_pending_ue_count": 64,
        "freeze_slots_on_violation": 4,
    }
    defaults.update(presets.get(risk_posture, {}))
    return ExplorationHealthGate(
        max_avg_bler=float(defaults["max_avg_bler"]),
        max_total_queue_bytes=max_queue,
        min_total_throughput_mbps=min_throughput,
        max_handover_pending_ue_count=int(defaults["max_handover_pending_ue_count"]),
        freeze_slots_on_violation=int(defaults["freeze_slots_on_violation"]),
    )
