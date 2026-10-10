"""Explicit fusion/arbitration subsystem for Near-RT multi-xApp composition."""

from __future__ import annotations

from dataclasses import dataclass, field

from models.action import ConfigAction


@dataclass
class ArbitrationSignal:
    field_name: str
    arbitration_mode: str = "weighted_merge"
    omega: dict[str, float] = field(default_factory=dict)
    selected_source: str = ""
    candidate_values: list[dict[str, object]] = field(default_factory=list)


@dataclass
class FusionResult:
    merged_action: ConfigAction
    signals: list[ArbitrationSignal] = field(default_factory=list)


def _ordered_head_options(head_options: list[object], arbitration_profile) -> list[object]:
    if arbitration_profile is None:
        return list(head_options)
    override_order = [
        str(item).strip()
        for item in list(getattr(arbitration_profile, "override_order", []) or [])
        if str(item).strip() != ""
    ]
    if len(override_order) == 0:
        return list(head_options)
    rank_by_category = {
        str(category_name): index
        for index, category_name in enumerate(override_order)
    }
    return sorted(
        list(head_options),
        key=lambda head_option: (
            rank_by_category.get(str(getattr(head_option, "category", "")), len(rank_by_category) + 1),
            -float(getattr(head_option, "merge_weight", 1.0) or 1.0),
            str(getattr(head_option, "option_id", "") or ""),
        ),
    )


def _normalized_omega(weighted_options: list[tuple[object, float]]) -> dict[str, float]:
    positive = [
        (str(getattr(head_option, "option_id", "") or ""), max(0.0, float(weight)))
        for head_option, weight in list(weighted_options or [])
        if str(getattr(head_option, "option_id", "") or "").strip() != ""
    ]
    total_weight = float(sum(weight for _, weight in positive))
    if total_weight <= 1e-9:
        return {}
    return {
        option_id: float(weight) / total_weight
        for option_id, weight in positive
    }


def merge_scope_group(
    *,
    head_options: list[object],
    controlled_fields: list[str],
    numeric_fields: set[str],
    discrete_fields: set[str],
    dict_fields: set[str],
    source_name: str,
    arbitration_profile=None,
    guardrail_categories: set[str] | None = None,
) -> FusionResult:
    if len(head_options) == 0:
        return FusionResult(merged_action=ConfigAction(source_name=source_name), signals=[])

    seed = getattr(head_options[0], "config_action")
    merged = ConfigAction(
        apply_scope=str(getattr(seed, "apply_scope", "cell") or "cell"),
        target_cell_id=getattr(seed, "target_cell_id", None),
        source_name=source_name,
    )
    arbitration_mode = str(
        getattr(arbitration_profile, "arbitration_mode", "weighted_merge") or "weighted_merge"
    )
    guardrail_categories = set(guardrail_categories or set())
    ordered_head_options = _ordered_head_options(head_options, arbitration_profile)
    signals: list[ArbitrationSignal] = []

    for field_name in list(controlled_fields or []):
        candidate_values: list[dict[str, object]] = []
        for head_option in ordered_head_options:
            config_action = getattr(head_option, "config_action", None)
            if config_action is None:
                continue
            value = getattr(config_action, field_name, None)
            if field_name in dict_fields:
                if not isinstance(value, dict) or len(value) == 0:
                    continue
            elif value is None:
                continue
            candidate_values.append(
                {
                    "option_id": str(getattr(head_option, "option_id", "") or ""),
                    "xapp_name": str(getattr(head_option, "xapp_name", "") or ""),
                    "category": str(getattr(head_option, "category", "") or ""),
                    "merge_weight": float(getattr(head_option, "merge_weight", 1.0) or 1.0),
                    "value": dict(value) if isinstance(value, dict) else value,
                }
            )
        if len(candidate_values) == 0:
            continue

        selected_source = ""
        omega = _normalized_omega(
            [
                (head_option, float(getattr(head_option, "merge_weight", 1.0) or 1.0))
                for head_option in ordered_head_options
                if any(
                    str(candidate_value.get("option_id", "") or "")
                    == str(getattr(head_option, "option_id", "") or "")
                    for candidate_value in candidate_values
                )
            ]
        )
        if field_name in numeric_fields:
            numeric_items: list[tuple[float, float, object]] = []
            for head_option in ordered_head_options:
                config_action = getattr(head_option, "config_action", None)
                if config_action is None:
                    continue
                value = getattr(config_action, field_name, None)
                if isinstance(value, (int, float)):
                    numeric_items.append((float(value), float(getattr(head_option, "merge_weight", 1.0) or 1.0), head_option))
            if len(numeric_items) == 0:
                continue
            if arbitration_mode == "priority_override":
                selected_value = float(numeric_items[0][0])
                selected_source = str(getattr(numeric_items[0][2], "option_id", "") or "")
            elif arbitration_mode == "guardrail_first":
                guard_items = [
                    item
                    for item in numeric_items
                    if str(getattr(item[2], "category", "")) in guardrail_categories
                ]
                if len(guard_items) > 0:
                    selected_value = float(guard_items[0][0])
                    selected_source = str(getattr(guard_items[0][2], "option_id", "") or "")
                else:
                    total_weight = max(1e-6, sum(weight for _, weight, _ in numeric_items))
                    selected_value = float(sum(value * weight for value, weight, _ in numeric_items) / total_weight)
                    selected_source = "weighted_merge"
            else:
                total_weight = max(1e-6, sum(weight for _, weight, _ in numeric_items))
                selected_value = float(sum(value * weight for value, weight, _ in numeric_items) / total_weight)
                selected_source = "weighted_merge"
            setattr(merged, field_name, selected_value)
        elif field_name in discrete_fields:
            selected_value = None
            if arbitration_mode in {"priority_override", "guardrail_first"}:
                fallback_value = None
                fallback_source = ""
                for head_option in ordered_head_options:
                    config_action = getattr(head_option, "config_action", None)
                    if config_action is None:
                        continue
                    candidate_value = getattr(config_action, field_name, None)
                    if candidate_value is None:
                        continue
                    if arbitration_mode == "guardrail_first":
                        if str(getattr(head_option, "category", "")) in guardrail_categories:
                            selected_value = candidate_value
                            selected_source = str(getattr(head_option, "option_id", "") or "")
                            break
                        if fallback_value is None:
                            fallback_value = candidate_value
                            fallback_source = str(getattr(head_option, "option_id", "") or "")
                    else:
                        selected_value = candidate_value
                        selected_source = str(getattr(head_option, "option_id", "") or "")
                        break
                if selected_value is None:
                    selected_value = fallback_value
                    selected_source = fallback_source
            if selected_value is None:
                votes: dict[object, float] = {}
                source_by_value: dict[object, str] = {}
                for candidate_value in candidate_values:
                    value = candidate_value["value"]
                    votes[value] = votes.get(value, 0.0) + float(candidate_value["merge_weight"] or 0.0)
                    source_by_value.setdefault(value, str(candidate_value["option_id"] or ""))
                selected_value = max(votes.items(), key=lambda item: float(item[1]))[0]
                selected_source = source_by_value.get(selected_value, "")
            setattr(merged, field_name, selected_value)
        elif field_name in dict_fields:
            if arbitration_mode == "priority_override":
                selected_dict = None
                for head_option in ordered_head_options:
                    config_action = getattr(head_option, "config_action", None)
                    if config_action is None:
                        continue
                    candidate_value = getattr(config_action, field_name, None)
                    if isinstance(candidate_value, dict) and len(candidate_value) > 0:
                        selected_dict = dict(candidate_value)
                        selected_source = str(getattr(head_option, "option_id", "") or "")
                        break
                if selected_dict is not None:
                    setattr(merged, field_name, selected_dict)
                else:
                    continue
            else:
                key_space = {
                    str(key)
                    for candidate_value in candidate_values
                    if isinstance(candidate_value.get("value"), dict)
                    for key in dict(candidate_value["value"]).keys()
                }
                merged_dict: dict[str, float] = {}
                for dict_key in key_space:
                    weighted_sum = 0.0
                    total_weight = 0.0
                    for candidate_value in candidate_values:
                        value = candidate_value.get("value")
                        if not isinstance(value, dict) or dict_key not in value:
                            continue
                        weight = float(candidate_value.get("merge_weight", 1.0) or 1.0)
                        weighted_sum += float(value[dict_key]) * weight
                        total_weight += weight
                    if total_weight > 0.0:
                        merged_dict[dict_key] = float(weighted_sum / total_weight)
                setattr(merged, field_name, merged_dict)
                selected_source = "weighted_merge"
        signals.append(
            ArbitrationSignal(
                field_name=str(field_name),
                arbitration_mode=str(arbitration_mode),
                omega=dict(omega),
                selected_source=str(selected_source),
                candidate_values=list(candidate_values),
            )
        )
    return FusionResult(merged_action=merged, signals=signals)
