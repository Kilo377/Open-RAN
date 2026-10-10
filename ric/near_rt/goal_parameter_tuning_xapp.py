"""Near-RT RL tuning xApp with per-xApp parameter heads and joint actions."""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field

from models.action import ConfigAction, ControlDecision
from models.intent_orchestration import (
    NearRtRuntimeState,
    OmegaRuntimeFieldState,
    OmegaRuntimeState,
    ThetaRuntimeState,
    near_rt_runtime_state_payload,
    omega_runtime_state_payload,
    theta_runtime_state_payload,
)
from models.observation import Observation
from models.policy import PolicySnapshot
from ric.near_rt.fusion_arbitration import merge_scope_group
from ric.near_rt.xapp_base import XAppBase
from ric.near_rt.xapp_helpers import (
    clamp,
    dominant_slice_id,
    make_analysis_constraint,
    select_beam_stressed_cell,
    select_busiest_cell,
    select_least_loaded_cell,
    select_urlcc_focus_cell,
)
from ric.near_rt.xapp_variant_catalog import canonical_xapp_category
from ric.non_rt.intent_orchestration_helpers import goal_gap_score, observation_kpi_values


@dataclass
class _ParameterHead:
    xapp_name: str
    category: str
    tuning_mode: str
    field_ranges: dict[str, object] = field(default_factory=dict)


@dataclass
class _ResolvedFieldRange:
    field_name: str
    target_xapp_name: str
    scope: str = "cluster"
    lower_bound: float | None = None
    upper_bound: float | None = None
    step: float | None = None
    discrete_values: list[object] = field(default_factory=list)
    blocked_discrete_values: list[object] = field(default_factory=list)
    max_delta: float | None = None
    tuning_mode: str = "rl_parameter"


@dataclass
class _HeadOption:
    option_id: str
    xapp_name: str
    category: str
    tuning_mode: str
    merge_weight: float
    config_action: ConfigAction
    summary: dict[str, object] = field(default_factory=dict)


@dataclass
class _JointAction:
    joint_action_id: str
    config_actions: list[ConfigAction]
    head_options: list[_HeadOption]
    merged_summary: dict[str, object] = field(default_factory=dict)


class GoalParameterTuningXApp(XAppBase):
    """Online RL tuner with per-xApp parameter heads and joint action space."""

    _NUMERIC_FIELDS = {
        "latency_bias",
        "power_cap_dbm",
        "mobility_bias_db",
        "handover_cost_db",
        "steering_hysteresis_db",
        "measurement_event_offset_db",
    }
    _DISCRETE_FIELDS = {
        "scheduler_policy_hint",
        "beam_candidate_preference",
        "cell_sleep_mode",
        "traffic_steering_preference",
        "measurement_event_trigger",
    }
    _DICT_FIELDS = {
        "slice_weight_adjustment",
    }

    def __init__(
        self,
        *,
        name: str = "goal_parameter_tuning_xapp",
        priority: int = 22,
        learning_rate: float = 0.18,
        discount_factor: float = 0.82,
        epsilon: float = 0.28,
        min_epsilon: float = 0.05,
        epsilon_decay: float = 0.985,
        max_joint_actions: int = 64,
        random_seed: int = 20260315,
        safety_shield_enabled: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            priority=priority,
            controlled_fields=[
                "scheduler_policy_hint",
                "latency_bias",
                "slice_weight_adjustment",
                "beam_candidate_preference",
                "power_cap_dbm",
                "cell_sleep_mode",
                "traffic_steering_preference",
                "mobility_bias_db",
                "steering_hysteresis_db",
                "measurement_event_offset_db",
            ],
        )
        self.learning_rate = clamp(float(learning_rate), 0.01, 1.0)
        self.discount_factor = clamp(float(discount_factor), 0.0, 0.99)
        self.epsilon = clamp(float(epsilon), 0.0, 1.0)
        self.min_epsilon = clamp(float(min_epsilon), 0.0, 1.0)
        self.epsilon_decay = clamp(float(epsilon_decay), 0.80, 1.0)
        self.max_joint_actions = max(4, int(max_joint_actions))
        self._rng = random.Random(int(random_seed))
        self.safety_shield_enabled = bool(safety_shield_enabled)
        self._q_table: dict[str, dict[str, float]] = {}
        self._action_visit_counts: dict[str, int] = {}
        self._last_state_key: str | None = None
        self._last_joint_action_id: str | None = None
        self._last_plan_id: str | None = None
        self._exploration_freeze_remaining = 0

    def _bucket(self, value: float, thresholds: tuple[float, float], labels: tuple[str, str, str]) -> str:
        low_threshold, high_threshold = thresholds
        if value < low_threshold:
            return labels[0]
        if value < high_threshold:
            return labels[1]
        return labels[2]

    def _state_key(self, observation: Observation, policy_snapshot: PolicySnapshot, plan_id: str) -> str:
        goal = getattr(policy_snapshot, "structured_goal", None)
        plan = getattr(policy_snapshot, "xapp_selection_plan", None)
        dominant_slice, _ = dominant_slice_id(observation)
        queue_bucket = self._bucket(
            float(getattr(observation, "total_queue_bytes", 0) or 0.0),
            (20000.0, 90000.0),
            ("queue_low", "queue_med", "queue_high"),
        )
        bler_bucket = self._bucket(
            float(getattr(observation, "avg_bler", 0.0) or 0.0),
            (0.10, 0.28),
            ("bler_good", "bler_warn", "bler_bad"),
        )
        sinr_bucket = self._bucket(
            float(getattr(observation, "avg_sinr_db", 0.0) or 0.0),
            (4.0, 11.0),
            ("sinr_poor", "sinr_ok", "sinr_good"),
        )
        handover_bucket = self._bucket(
            float(getattr(observation, "handover_pending_ue_count", 0) or 0.0),
            (0.5, 2.5),
            ("handover_none", "handover_some", "handover_many"),
        )
        risk_posture = str(getattr(goal, "risk_posture", "balanced") or "balanced").strip().lower()
        selected = ",".join(sorted(str(item) for item in list(getattr(plan, "selected_xapps", []) or [])))
        return "|".join(
            [
                f"plan={plan_id}",
                f"risk={risk_posture}",
                f"slice={str(dominant_slice)}",
                f"selected={selected}",
                queue_bucket,
                bler_bucket,
                sinr_bucket,
                handover_bucket,
            ]
        )

    def _avg_power_cost(self, observation: Observation) -> float:
        cell_observations = list(getattr(observation, "cell_observations", []) or [])
        if len(cell_observations) == 0:
            return 30.0
        power_values = [
            float(
                getattr(cell, "power_cap_dbm", getattr(cell, "tx_power_dbm", 30.0)) or 30.0
            )
            for cell in cell_observations
        ]
        return float(sum(power_values) / max(1, len(power_values)))

    def _active_rl_guidance(self, policy_snapshot: PolicySnapshot):
        guidance = getattr(policy_snapshot, "rl_guidance", None)
        if guidance is None:
            return None
        guided_action_region = list(getattr(guidance, "guided_action_region", []) or [])
        reward_coefficients = dict(getattr(guidance, "reward_coefficients", {}) or {})
        if len(guided_action_region) == 0 and len(reward_coefficients) == 0:
            return None
        return guidance

    def _effective_reward_terms(self, policy_snapshot: PolicySnapshot) -> dict[str, float]:
        guidance = self._active_rl_guidance(policy_snapshot)
        if guidance is not None:
            return {
                str(key): float(value)
                for key, value in dict(getattr(guidance, "reward_coefficients", {}) or {}).items()
            }
        reward_spec = getattr(policy_snapshot, "reward_spec", None)
        return {
            str(key): float(value)
            for key, value in dict(getattr(reward_spec, "reward_terms", {}) or {}).items()
        }

    def _effective_parameter_ranges(self, policy_snapshot: PolicySnapshot) -> list[object]:
        guidance = self._active_rl_guidance(policy_snapshot)
        if guidance is not None:
            return list(getattr(guidance, "guided_action_region", []) or [])
        reward_spec = getattr(policy_snapshot, "reward_spec", None)
        return list(getattr(reward_spec, "parameter_ranges", []) or [])

    def _effective_black_box_xapps(self, policy_snapshot: PolicySnapshot) -> list[str]:
        guidance = self._active_rl_guidance(policy_snapshot)
        if guidance is not None:
            return [str(item) for item in list(getattr(guidance, "black_box_xapps", []) or [])]
        reward_spec = getattr(policy_snapshot, "reward_spec", None)
        return [str(item) for item in list(getattr(reward_spec, "black_box_xapps", []) or [])]

    def _effective_parameter_priority_vector(self, policy_snapshot: PolicySnapshot) -> dict[str, float]:
        guidance = self._active_rl_guidance(policy_snapshot)
        if guidance is None:
            return {}
        priority_vector = {
            str(field_name): max(0.0, float(priority))
            for field_name, priority in dict(getattr(guidance, "parameter_priority_vector", {}) or {}).items()
            if str(field_name).strip() != ""
        }
        total_priority = float(sum(priority_vector.values()))
        if total_priority <= 1e-9:
            return {}
        return {
            field_name: float(priority) / total_priority
            for field_name, priority in priority_vector.items()
        }

    def _effective_arbitration_profile(self, policy_snapshot: PolicySnapshot):
        orchestration_plan_view = getattr(policy_snapshot, "orchestration_plan_view", None)
        return getattr(orchestration_plan_view, "chi", None)

    def _effective_safety_constraint_set(self, policy_snapshot: PolicySnapshot, plan):
        if not self.safety_shield_enabled:
            return None
        guidance = self._active_rl_guidance(policy_snapshot)
        if guidance is not None and getattr(guidance, "selected_safety_rule_set", None) is not None:
            return getattr(guidance, "selected_safety_rule_set", None)
        return getattr(policy_snapshot, "safety_constraint_set", None)

    def _effective_hard_safety_envelope(self, policy_snapshot: PolicySnapshot, plan) -> list[object]:
        if not self.safety_shield_enabled:
            return []
        safety_constraint_set = self._effective_safety_constraint_set(policy_snapshot, plan)
        if safety_constraint_set is not None:
            return list(getattr(safety_constraint_set, "safety_envelope", []) or [])
        return list(getattr(plan, "hard_safety_envelope", []) or [])

    def _effective_forbidden_combinations(self, policy_snapshot: PolicySnapshot, plan) -> list[object]:
        if not self.safety_shield_enabled:
            return []
        safety_constraint_set = self._effective_safety_constraint_set(policy_snapshot, plan)
        if safety_constraint_set is not None:
            return list(getattr(safety_constraint_set, "forbidden_combinations", []) or [])
        return list(getattr(plan, "forbidden_combinations", []) or [])

    def _reward(self, observation: Observation, policy_snapshot: PolicySnapshot) -> float:
        goal = getattr(policy_snapshot, "structured_goal", None)
        validated_intent_template = getattr(policy_snapshot, "validated_intent_template", None)
        reward_terms = self._effective_reward_terms(policy_snapshot)
        if len(reward_terms) == 0:
            return 0.0
        throughput_norm = clamp(
            float(getattr(observation, "total_throughput_mbps", 0.0) or 0.0) / 25.0,
            0.0,
            1.5,
        )
        queue_penalty = clamp(
            float(getattr(observation, "total_queue_bytes", 0) or 0.0) / 140000.0,
            0.0,
            2.0,
        )
        bler_penalty = clamp(float(getattr(observation, "avg_bler", 0.0) or 0.0), 0.0, 1.0)
        handover_penalty = clamp(
            float(getattr(observation, "handover_pending_ue_count", 0) or 0.0)
            / max(1.0, float(getattr(observation, "connected_ue_count", 1) or 1.0)),
            0.0,
            1.0,
        )
        power_penalty = clamp(self._avg_power_cost(observation) / 32.0, 0.0, 1.5)
        reward_value = 0.0
        reward_value += float(reward_terms.get("throughput_gain", 0.0)) * throughput_norm
        reward_value += float(reward_terms.get("queue_reduction", 0.0)) * (1.0 - queue_penalty)
        reward_value += float(reward_terms.get("bler_reduction", 0.0)) * (1.0 - bler_penalty)
        reward_value += float(reward_terms.get("handover_risk_reduction", 0.0)) * (1.0 - handover_penalty)
        reward_value -= float(reward_terms.get("power_cost_penalty", 0.0)) * power_penalty
        if goal is not None:
            reward_value -= 1.25 * float(
                goal_gap_score(
                    goal,
                    observation,
                    validated_intent_template=validated_intent_template,
                )
            )
        return float(reward_value)

    def _qos_outage_status(self, observation: Observation, plan) -> tuple[bool, list[str]]:
        qos_profile = getattr(plan, "qos_protection_profile", None)
        if qos_profile is None:
            return True, []
        values_by_kpi = observation_kpi_values(observation)
        reasons: list[str] = []
        for constraint in list(getattr(qos_profile, "constraints", []) or []):
            metric_name = str(getattr(constraint, "metric_name", "") or "").strip().lower()
            semantic_type = str(getattr(constraint, "semantic_type", "max") or "max").strip().lower()
            threshold_value = getattr(constraint, "threshold_value", None)
            if metric_name == "" or threshold_value is None:
                continue
            actual_value = float(values_by_kpi.get(metric_name, 0.0) or 0.0)
            threshold_float = float(threshold_value)
            if semantic_type == "min" and actual_value > threshold_float:
                reasons.append(f"qos_{metric_name}_exceeded")
            elif semantic_type == "max" and actual_value < threshold_float:
                reasons.append(f"qos_{metric_name}_below_target")
            elif semantic_type == "target" and abs(actual_value - threshold_float) > max(abs(threshold_float), 1.0):
                reasons.append(f"qos_{metric_name}_off_target")
        return len(reasons) == 0, reasons

    def _resolve_field_range(self, soft_range, hard_range) -> _ResolvedFieldRange | None:
        field_name = str(getattr(soft_range, "field_name", "")).strip()
        target_xapp_name = str(getattr(soft_range, "target_xapp_name", "")).strip()
        if field_name == "" or target_xapp_name == "":
            return None

        soft_lower = getattr(soft_range, "lower_bound", None)
        soft_upper = getattr(soft_range, "upper_bound", None)
        hard_lower = getattr(hard_range, "lower_bound", None) if hard_range is not None else None
        hard_upper = getattr(hard_range, "upper_bound", None) if hard_range is not None else None

        lower_bound = soft_lower
        upper_bound = soft_upper
        if hard_lower is not None:
            lower_bound = float(hard_lower) if lower_bound is None else max(float(lower_bound), float(hard_lower))
        if hard_upper is not None:
            upper_bound = float(hard_upper) if upper_bound is None else min(float(upper_bound), float(hard_upper))
        if lower_bound is not None:
            lower_bound = float(lower_bound)
        if upper_bound is not None:
            upper_bound = float(upper_bound)
        if lower_bound is not None and upper_bound is not None and lower_bound > upper_bound:
            return None

        soft_discrete = [item for item in list(getattr(soft_range, "discrete_values", []) or [])]
        hard_discrete = [item for item in list(getattr(hard_range, "discrete_values", []) or [])] if hard_range is not None else []
        blocked_values = list(getattr(soft_range, "blocked_discrete_values", []) or [])
        if hard_range is not None:
            blocked_values.extend(list(getattr(hard_range, "blocked_discrete_values", []) or []))

        if len(soft_discrete) > 0 and len(hard_discrete) > 0:
            discrete_values = [item for item in soft_discrete if item in hard_discrete]
        elif len(soft_discrete) > 0:
            discrete_values = list(soft_discrete)
        else:
            discrete_values = list(hard_discrete)

        normalized_blocked: list[object] = []
        for item in blocked_values:
            if item not in normalized_blocked:
                normalized_blocked.append(item)
        if len(discrete_values) > 0:
            discrete_values = [item for item in discrete_values if item not in normalized_blocked]
            if len(discrete_values) == 0:
                return None

        step_candidates = [
            float(value)
            for value in (
                getattr(soft_range, "step", None),
                getattr(hard_range, "step", None) if hard_range is not None else None,
            )
            if value not in (None, 0, 0.0)
        ]
        max_delta = getattr(hard_range, "max_delta", None) if hard_range is not None else None
        if max_delta is None:
            max_delta = getattr(soft_range, "max_delta", None)

        return _ResolvedFieldRange(
            field_name=field_name,
            target_xapp_name=target_xapp_name,
            scope=str(
                getattr(
                    hard_range,
                    "scope",
                    getattr(soft_range, "scope", "cluster"),
                )
                or "cluster"
            ),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            step=min(step_candidates) if len(step_candidates) > 0 else None,
            discrete_values=discrete_values,
            blocked_discrete_values=normalized_blocked,
            max_delta=float(max_delta) if max_delta is not None else None,
            tuning_mode=str(getattr(soft_range, "tuning_mode", "rl_parameter") or "rl_parameter"),
        )

    def _range_groups(self, control_spec, plan, *, hard_safety_envelope=None) -> list[_ParameterHead]:
        selected_xapps = [str(item) for item in list(getattr(plan, "selected_xapps", []) or []) if str(item).strip() != ""]
        black_box_xapps = {
            str(item)
            for item in list(getattr(control_spec, "black_box_xapps", []) or [])
            if str(item).strip() != ""
        }
        soft_groups: dict[str, dict[str, object]] = {xapp_name: {} for xapp_name in selected_xapps}
        hard_groups: dict[str, dict[str, object]] = {xapp_name: {} for xapp_name in selected_xapps}
        parameter_ranges = list(
            getattr(control_spec, "guided_action_region", None)
            or getattr(control_spec, "parameter_ranges", [])
            or []
        )
        for parameter_range in parameter_ranges:
            field_name = str(getattr(parameter_range, "field_name", "")).strip()
            target_xapp_name = str(getattr(parameter_range, "target_xapp_name", "")).strip()
            if field_name == "" or target_xapp_name == "" or target_xapp_name not in soft_groups:
                continue
            soft_groups[target_xapp_name][field_name] = parameter_range
        hard_envelope_items = (
            list(hard_safety_envelope)
            if hard_safety_envelope is not None
            else list(getattr(plan, "hard_safety_envelope", []) or [])
        )
        for parameter_range in hard_envelope_items:
            field_name = str(getattr(parameter_range, "field_name", "")).strip()
            target_xapp_name = str(getattr(parameter_range, "target_xapp_name", "")).strip()
            if field_name == "" or target_xapp_name == "" or target_xapp_name not in hard_groups:
                continue
            hard_groups[target_xapp_name][field_name] = parameter_range
        heads: list[_ParameterHead] = []
        for xapp_name in selected_xapps:
            resolved_ranges: dict[str, _ResolvedFieldRange] = {}
            for field_name, soft_range in soft_groups.get(xapp_name, {}).items():
                resolved_range = self._resolve_field_range(
                    soft_range,
                    hard_groups.get(xapp_name, {}).get(field_name),
                )
                if resolved_range is not None:
                    resolved_ranges[field_name] = resolved_range
            if len(resolved_ranges) == 0:
                continue
            tuning_mode = "output_shaping" if xapp_name in black_box_xapps else "parameter_head"
            heads.append(
                _ParameterHead(
                    xapp_name=xapp_name,
                    category=canonical_xapp_category(xapp_name),
                    tuning_mode=tuning_mode,
                    field_ranges=dict(resolved_ranges),
                )
            )
        return heads

    def _interpolate_numeric(self, parameter_range, level: float) -> float:
        lower = float(getattr(parameter_range, "lower_bound", 0.0) or 0.0)
        upper = float(getattr(parameter_range, "upper_bound", lower) or lower)
        return clamp(lower + (upper - lower) * clamp(level, 0.0, 1.0), lower, upper)

    def _discrete_value(self, parameter_range, preferred_values: list[str]) -> str | None:
        values = [str(item) for item in list(getattr(parameter_range, "discrete_values", []) or [])]
        if len(values) == 0:
            return None
        for preferred in preferred_values:
            if preferred in values:
                return preferred
        return values[0]

    def _policy_snapshot_field_name(self, field_name: str) -> str:
        return {
            "scheduler_policy_hint": "preferred_scheduler_policy_hint",
            "latency_bias": "preferred_latency_bias",
            "slice_weight_adjustment": "preferred_slice_weight_adjustment",
            "beam_candidate_preference": "preferred_beam_candidate_preference",
            "power_cap_dbm": "preferred_power_cap_dbm",
            "cell_sleep_mode": "preferred_cell_sleep_mode",
            "traffic_steering_preference": "preferred_traffic_steering_preference",
            "mobility_bias_db": "preferred_mobility_bias_db",
            "handover_cost_db": "preferred_handover_cost_db",
            "steering_hysteresis_db": "preferred_steering_hysteresis_db",
            "measurement_event_trigger": "preferred_measurement_event_trigger",
            "measurement_event_offset_db": "preferred_measurement_event_offset_db",
        }.get(field_name, "")

    def _target_cell_observation(self, observation: Observation, action: ConfigAction):
        cell_observations = list(getattr(observation, "cell_observations", []) or [])
        target_cell_id = getattr(action, "target_cell_id", None)
        if target_cell_id is not None:
            for cell in cell_observations:
                if str(getattr(cell, "cell_id", "")) == str(target_cell_id):
                    return cell
        return cell_observations[0] if len(cell_observations) > 0 else None

    def _baseline_value_for_field(
        self,
        *,
        field_name: str,
        action: ConfigAction,
        observation: Observation,
        policy_snapshot: PolicySnapshot,
    ):
        scope = str(getattr(action, "apply_scope", "cell") or "cell")
        policy_attr = self._policy_snapshot_field_name(field_name)
        if field_name == "slice_weight_adjustment":
            if scope == "cell":
                cell = self._target_cell_observation(observation, action)
                cell_value = dict(getattr(cell, field_name, {}) or {}) if cell is not None else {}
                if len(cell_value) > 0:
                    return cell_value
            for cell in list(getattr(observation, "cell_observations", []) or []):
                cell_value = dict(getattr(cell, field_name, {}) or {})
                if len(cell_value) > 0:
                    return cell_value
            if policy_attr != "":
                policy_value = dict(getattr(policy_snapshot, policy_attr, {}) or {})
                if len(policy_value) > 0:
                    return policy_value
            return {}

        candidate_values: list[object] = []
        if scope == "cell":
            cell = self._target_cell_observation(observation, action)
            if cell is not None:
                cell_value = getattr(cell, field_name, None)
                if cell_value is not None:
                    candidate_values.append(cell_value)
        else:
            for cell in list(getattr(observation, "cell_observations", []) or []):
                cell_value = getattr(cell, field_name, None)
                if cell_value is not None:
                    candidate_values.append(cell_value)
            if len(candidate_values) > 1 and field_name in self._NUMERIC_FIELDS:
                numeric_values = [float(value) for value in candidate_values if isinstance(value, (int, float))]
                if len(numeric_values) > 0:
                    return float(sum(numeric_values) / len(numeric_values))
        if len(candidate_values) > 0:
            return candidate_values[0]
        if policy_attr != "":
            return getattr(policy_snapshot, policy_attr, None)
        return None

    def _step_round(self, value: float, step: float | None, anchor: float) -> float:
        if step is None or step <= 0.0:
            return float(value)
        return float(anchor + round((value - anchor) / step) * step)

    def _clip_numeric_candidate(
        self,
        *,
        raw_value: float,
        baseline_value,
        parameter_range: _ResolvedFieldRange,
        allow_missing_baseline: bool,
    ) -> float | None:
        lower_bound = getattr(parameter_range, "lower_bound", None)
        upper_bound = getattr(parameter_range, "upper_bound", None)
        step = getattr(parameter_range, "step", None)
        max_delta = getattr(parameter_range, "max_delta", None)

        if baseline_value is None:
            if not allow_missing_baseline:
                return None
            candidate = float(raw_value)
        else:
            baseline = float(baseline_value)
            if lower_bound is not None and upper_bound is not None:
                baseline = clamp(baseline, float(lower_bound), float(upper_bound))
            candidate = float(raw_value)
            if max_delta is not None:
                candidate = clamp(candidate, baseline - float(max_delta), baseline + float(max_delta))
            candidate = self._step_round(candidate, step, baseline)
        if lower_bound is not None and upper_bound is not None:
            candidate = clamp(candidate, float(lower_bound), float(upper_bound))
        elif lower_bound is not None:
            candidate = max(candidate, float(lower_bound))
        elif upper_bound is not None:
            candidate = min(candidate, float(upper_bound))
        if baseline_value is None and step is not None and step > 0.0:
            anchor = float(lower_bound) if lower_bound is not None else 0.0
            candidate = self._step_round(candidate, step, anchor)
        return float(candidate)

    def _clip_discrete_candidate(
        self,
        *,
        raw_value,
        baseline_value,
        parameter_range: _ResolvedFieldRange,
        allow_missing_baseline: bool,
    ):
        allowed_values = [item for item in list(getattr(parameter_range, "discrete_values", []) or [])]
        if len(allowed_values) == 0:
            return None
        if raw_value in allowed_values:
            return raw_value
        if baseline_value in allowed_values:
            return baseline_value
        if not allow_missing_baseline and baseline_value is None:
            return None
        return allowed_values[0]

    def _clip_slice_weights(
        self,
        *,
        raw_weights: dict[str, float],
        baseline_weights: dict[str, float] | None,
        parameter_range: _ResolvedFieldRange,
        allow_missing_baseline: bool,
    ) -> dict[str, float]:
        if baseline_weights is None:
            baseline_weights = {}
        if len(baseline_weights) == 0 and not allow_missing_baseline:
            return {}
        if len(baseline_weights) == 0:
            baseline_weights = {"urllc": 1.0, "embb": 1.0, "mmtc": 1.0}
        lower_bound = float(getattr(parameter_range, "lower_bound", 0.7) or 0.7)
        upper_bound = float(getattr(parameter_range, "upper_bound", 1.6) or 1.6)
        max_delta = getattr(parameter_range, "max_delta", None)
        step = getattr(parameter_range, "step", None)
        candidate_weights: dict[str, float] = {}
        key_space = set(baseline_weights.keys()) | set(raw_weights.keys())
        for dict_key in key_space:
            baseline_value = float(baseline_weights.get(dict_key, 1.0))
            raw_value = float(raw_weights.get(dict_key, baseline_value))
            if max_delta is not None:
                raw_value = clamp(raw_value, baseline_value - float(max_delta), baseline_value + float(max_delta))
            raw_value = clamp(raw_value, lower_bound, upper_bound)
            raw_value = self._step_round(raw_value, step, baseline_value)
            candidate_weights[str(dict_key)] = clamp(raw_value, lower_bound, upper_bound)
        return candidate_weights

    def _base_slice_weights(
        self,
        policy_snapshot: PolicySnapshot,
        *,
        action: ConfigAction | None = None,
        observation: Observation | None = None,
    ) -> dict[str, float]:
        if action is not None and observation is not None:
            baseline_value = self._baseline_value_for_field(
                field_name="slice_weight_adjustment",
                action=action,
                observation=observation,
                policy_snapshot=policy_snapshot,
            )
            if isinstance(baseline_value, dict) and len(baseline_value) > 0:
                return {str(key): float(value) for key, value in baseline_value.items()}
        base_weights = dict(getattr(policy_snapshot, "preferred_slice_weight_adjustment", {}) or {})
        if len(base_weights) == 0:
            base_weights = {"urllc": 1.0, "embb": 1.0, "mmtc": 1.0}
        return {str(key): float(value) for key, value in base_weights.items()}

    def _head_profiles(self, head: _ParameterHead) -> list[tuple[str, float, float]]:
        category = head.category
        baseline_profile = [("baseline", 0.0, 1.00)]
        if category == "mobility_predictor_xapp":
            return baseline_profile + [("stable_watch", 0.25, 0.95), ("predictive_guard", 0.55, 1.05), ("handover_push", 0.92, 1.18)]
        if category == "traffic_steering_xapp":
            return baseline_profile + [("sticky", 0.25, 0.95), ("balanced_shift", 0.55, 1.05), ("edge_offload_push", 0.92, 1.18)]
        if category == "power_saving_xapp":
            return baseline_profile + [("sla_safe", 0.25, 0.95), ("balanced_saving", 0.55, 1.00), ("deep_saving", 0.92, 1.10)]
        if category == "spectrum_optimizer_xapp":
            return baseline_profile + [("interference_safe", 0.25, 1.00), ("balanced_efficiency", 0.55, 1.08), ("throughput_dense", 0.92, 1.15)]
        if category == "latency_scheduler_xapp":
            return baseline_profile + [("mild_latency", 0.25, 1.05), ("balanced_latency", 0.55, 1.12), ("hard_latency", 0.92, 1.22)]
        if category == "massive_mimo_beamformer_xapp":
            return baseline_profile + [("coverage_safe", 0.25, 1.00), ("balanced_mimo", 0.55, 1.08), ("capacity_push", 0.92, 1.16)]
        if category == "ran_slicing_manager_xapp":
            return baseline_profile + [("fair_slicing", 0.25, 1.00), ("balanced_sla", 0.55, 1.08), ("dominant_slice_push", 0.92, 1.15)]
        if category == "urllc_guard_xapp":
            return baseline_profile + [("soft_guard", 0.25, 1.12), ("balanced_guard", 0.55, 1.22), ("hard_guard", 0.92, 1.35)]
        if category == "cell_sleep_xapp":
            return baseline_profile + [("active_guard", 0.25, 0.95), ("micro_sleep", 0.55, 1.02), ("sleep_push", 0.92, 1.10)]
        return baseline_profile + [("conservative", 0.25, 0.95), ("balanced", 0.55, 1.05), ("aggressive", 0.92, 1.15)]

    def _head_target_action(self, head: _ParameterHead, observation: Observation) -> ConfigAction:
        category = head.category
        default_cell_id = str(getattr(observation, "cell_id", "cell_0"))
        busiest_cell = select_busiest_cell(observation)
        least_loaded_cell = select_least_loaded_cell(observation)
        urllc_cell = select_urlcc_focus_cell(observation)
        beam_stressed_cell = select_beam_stressed_cell(observation)

        if category in {"mobility_predictor_xapp", "traffic_steering_xapp", "ran_slicing_manager_xapp"}:
            return ConfigAction(apply_scope="cluster", source_name=self.name)
        if category == "power_saving_xapp":
            return ConfigAction(
                target_cell_id=str(getattr(least_loaded_cell, "cell_id", default_cell_id)),
                source_name=self.name,
            )
        if category == "massive_mimo_beamformer_xapp":
            return ConfigAction(
                target_cell_id=str(getattr(beam_stressed_cell or busiest_cell, "cell_id", default_cell_id)),
                source_name=self.name,
            )
        if category == "urllc_guard_xapp":
            return ConfigAction(
                target_cell_id=str(getattr(urllc_cell or busiest_cell, "cell_id", default_cell_id)),
                source_name=self.name,
            )
        if category == "cell_sleep_xapp":
            return ConfigAction(
                target_cell_id=str(getattr(least_loaded_cell, "cell_id", default_cell_id)),
                source_name=self.name,
            )
        return ConfigAction(
            target_cell_id=str(getattr(busiest_cell, "cell_id", default_cell_id)),
            source_name=self.name,
        )

    def _head_priority_weight(
        self,
        *,
        head: _ParameterHead,
        parameter_priority_vector: dict[str, float],
    ) -> float:
        active_fields = [
            str(field_name).strip()
            for field_name in dict(getattr(head, "field_ranges", {}) or {}).keys()
            if str(field_name).strip() != ""
        ]
        if len(active_fields) == 0:
            return 0.0
        return float(
            sum(float(parameter_priority_vector.get(field_name, 0.0)) for field_name in active_fields)
            / float(len(active_fields))
        )

    def _build_head_option(
        self,
        *,
        head: _ParameterHead,
        profile_name: str,
        intensity: float,
        merge_weight: float,
        priority_weight: float,
        observation: Observation,
        policy_snapshot: PolicySnapshot,
    ) -> _HeadOption:
        dominant_slice, dominant_share = dominant_slice_id(observation)
        action = self._head_target_action(head, observation)
        category = head.category
        field_ranges = head.field_ranges
        is_baseline_profile = profile_name == "baseline"
        priority_scaled_intensity = (
            float(intensity)
            if is_baseline_profile
            else clamp(float(intensity) * (0.65 + 1.75 * float(priority_weight)), 0.0, 1.0)
        )

        if "scheduler_policy_hint" in field_ranges:
            preferences = {
                "latency_scheduler_xapp": ["latency_bias", "balanced", "slice_bias"],
                "spectrum_optimizer_xapp": ["balanced", "slice_bias", "latency_bias"],
                "ran_slicing_manager_xapp": ["slice_bias", "balanced", "latency_bias"],
                "urllc_guard_xapp": ["latency_bias", "slice_bias", "balanced"],
            }
            baseline_value = self._baseline_value_for_field(
                field_name="scheduler_policy_hint",
                action=action,
                observation=observation,
                policy_snapshot=policy_snapshot,
            )
            raw_value = baseline_value if is_baseline_profile else self._discrete_value(
                field_ranges["scheduler_policy_hint"],
                preferences.get(category, ["balanced", "slice_bias", "latency_bias"]),
            )
            action.scheduler_policy_hint = self._clip_discrete_candidate(
                raw_value=raw_value,
                baseline_value=baseline_value,
                parameter_range=field_ranges["scheduler_policy_hint"],
                allow_missing_baseline=not is_baseline_profile,
            )

        if "latency_bias" in field_ranges:
            intensity_level = {
                "latency_scheduler_xapp": 0.55 + 0.35 * priority_scaled_intensity,
                "urllc_guard_xapp": 0.65 + 0.30 * priority_scaled_intensity,
            }.get(category, 0.35 + 0.25 * priority_scaled_intensity)
            baseline_value = self._baseline_value_for_field(
                field_name="latency_bias",
                action=action,
                observation=observation,
                policy_snapshot=policy_snapshot,
            )
            raw_value = baseline_value if is_baseline_profile else self._interpolate_numeric(
                field_ranges["latency_bias"],
                clamp(intensity_level, 0.0, 1.0),
            )
            action.latency_bias = self._clip_numeric_candidate(
                raw_value=float(raw_value) if raw_value is not None else 0.0,
                baseline_value=baseline_value,
                parameter_range=field_ranges["latency_bias"],
                allow_missing_baseline=not is_baseline_profile,
            )

        if "slice_weight_adjustment" in field_ranges:
            base_weights = self._base_slice_weights(
                policy_snapshot,
                action=action,
                observation=observation,
            )
            target_slice = dominant_slice
            if category == "latency_scheduler_xapp":
                target_slice = "urllc" if "urllc" in base_weights else dominant_slice
            elif category == "spectrum_optimizer_xapp":
                target_slice = "embb" if "embb" in base_weights else dominant_slice
            raw_weights = dict(base_weights)
            if not is_baseline_profile and target_slice in raw_weights:
                raw_weights[target_slice] = float(raw_weights[target_slice]) + (
                    0.15 + 0.35 * priority_scaled_intensity
                ) * max(0.25, float(dominant_share))
            action.slice_weight_adjustment = self._clip_slice_weights(
                raw_weights=raw_weights,
                baseline_weights=base_weights,
                parameter_range=field_ranges["slice_weight_adjustment"],
                allow_missing_baseline=not is_baseline_profile,
            )

        if "beam_candidate_preference" in field_ranges:
            preferences = {
                "power_saving_xapp": ["balanced", "coverage", "capacity"],
                "massive_mimo_beamformer_xapp": ["capacity", "balanced", "coverage"] if priority_scaled_intensity > 0.6 else ["coverage", "balanced", "capacity"],
            }
            baseline_value = self._baseline_value_for_field(
                field_name="beam_candidate_preference",
                action=action,
                observation=observation,
                policy_snapshot=policy_snapshot,
            )
            raw_value = baseline_value if is_baseline_profile else self._discrete_value(
                field_ranges["beam_candidate_preference"],
                preferences.get(category, ["balanced", "coverage", "capacity"]),
            )
            action.beam_candidate_preference = self._clip_discrete_candidate(
                raw_value=raw_value,
                baseline_value=baseline_value,
                parameter_range=field_ranges["beam_candidate_preference"],
                allow_missing_baseline=not is_baseline_profile,
            )

        if "power_cap_dbm" in field_ranges:
            level = {
                "power_saving_xapp": 0.18 + 0.35 * priority_scaled_intensity,
                "massive_mimo_beamformer_xapp": 0.55 + 0.25 * priority_scaled_intensity,
                "urllc_guard_xapp": 0.68 + 0.22 * priority_scaled_intensity,
                "cell_sleep_xapp": 0.10 + 0.28 * priority_scaled_intensity,
            }.get(category, 0.45 + 0.20 * priority_scaled_intensity)
            baseline_value = self._baseline_value_for_field(
                field_name="power_cap_dbm",
                action=action,
                observation=observation,
                policy_snapshot=policy_snapshot,
            )
            raw_value = baseline_value if is_baseline_profile else self._interpolate_numeric(
                field_ranges["power_cap_dbm"],
                clamp(level, 0.0, 1.0),
            )
            action.power_cap_dbm = self._clip_numeric_candidate(
                raw_value=float(raw_value) if raw_value is not None else 0.0,
                baseline_value=baseline_value,
                parameter_range=field_ranges["power_cap_dbm"],
                allow_missing_baseline=not is_baseline_profile,
            )

        if "cell_sleep_mode" in field_ranges:
            preferences = (
                ["active", "micro_sleep", "light_sleep", "deep_sleep"]
                if priority_scaled_intensity < 0.35
                else ["micro_sleep", "light_sleep", "deep_sleep", "active"]
                if priority_scaled_intensity < 0.75
                else ["light_sleep", "deep_sleep", "micro_sleep", "active"]
            )
            baseline_value = self._baseline_value_for_field(
                field_name="cell_sleep_mode",
                action=action,
                observation=observation,
                policy_snapshot=policy_snapshot,
            )
            raw_value = baseline_value if is_baseline_profile else self._discrete_value(
                field_ranges["cell_sleep_mode"],
                preferences,
            )
            action.cell_sleep_mode = self._clip_discrete_candidate(
                raw_value=raw_value,
                baseline_value=baseline_value,
                parameter_range=field_ranges["cell_sleep_mode"],
                allow_missing_baseline=not is_baseline_profile,
            )

        if "traffic_steering_preference" in field_ranges:
            preferences = ["balanced", "edge_offload", "stay_put"] if priority_scaled_intensity < 0.65 else ["edge_offload", "balanced", "stay_put"]
            baseline_value = self._baseline_value_for_field(
                field_name="traffic_steering_preference",
                action=action,
                observation=observation,
                policy_snapshot=policy_snapshot,
            )
            raw_value = baseline_value if is_baseline_profile else self._discrete_value(
                field_ranges["traffic_steering_preference"],
                preferences,
            )
            action.traffic_steering_preference = self._clip_discrete_candidate(
                raw_value=raw_value,
                baseline_value=baseline_value,
                parameter_range=field_ranges["traffic_steering_preference"],
                allow_missing_baseline=not is_baseline_profile,
            )

        if "mobility_bias_db" in field_ranges:
            level = {
                "mobility_predictor_xapp": 0.40 + 0.55 * priority_scaled_intensity,
                "traffic_steering_xapp": 0.30 + 0.50 * priority_scaled_intensity,
            }.get(category, 0.35 + 0.30 * priority_scaled_intensity)
            baseline_value = self._baseline_value_for_field(
                field_name="mobility_bias_db",
                action=action,
                observation=observation,
                policy_snapshot=policy_snapshot,
            )
            raw_value = baseline_value if is_baseline_profile else self._interpolate_numeric(
                field_ranges["mobility_bias_db"],
                clamp(level, 0.0, 1.0),
            )
            action.mobility_bias_db = self._clip_numeric_candidate(
                raw_value=float(raw_value) if raw_value is not None else 0.0,
                baseline_value=baseline_value,
                parameter_range=field_ranges["mobility_bias_db"],
                allow_missing_baseline=not is_baseline_profile,
            )

        if "steering_hysteresis_db" in field_ranges:
            level = {
                "mobility_predictor_xapp": 0.75 - 0.55 * priority_scaled_intensity,
                "traffic_steering_xapp": 0.65 - 0.45 * priority_scaled_intensity,
            }.get(category, 0.50)
            baseline_value = self._baseline_value_for_field(
                field_name="steering_hysteresis_db",
                action=action,
                observation=observation,
                policy_snapshot=policy_snapshot,
            )
            raw_value = baseline_value if is_baseline_profile else self._interpolate_numeric(
                field_ranges["steering_hysteresis_db"],
                clamp(level, 0.0, 1.0),
            )
            action.steering_hysteresis_db = self._clip_numeric_candidate(
                raw_value=float(raw_value) if raw_value is not None else 0.0,
                baseline_value=baseline_value,
                parameter_range=field_ranges["steering_hysteresis_db"],
                allow_missing_baseline=not is_baseline_profile,
            )

        if "measurement_event_offset_db" in field_ranges:
            level = {
                "mobility_predictor_xapp": 0.65 - 0.45 * priority_scaled_intensity,
                "traffic_steering_xapp": 0.55 - 0.40 * priority_scaled_intensity,
            }.get(category, 0.45)
            baseline_value = self._baseline_value_for_field(
                field_name="measurement_event_offset_db",
                action=action,
                observation=observation,
                policy_snapshot=policy_snapshot,
            )
            raw_value = baseline_value if is_baseline_profile else self._interpolate_numeric(
                field_ranges["measurement_event_offset_db"],
                clamp(level, 0.0, 1.0),
            )
            action.measurement_event_offset_db = self._clip_numeric_candidate(
                raw_value=float(raw_value) if raw_value is not None else 0.0,
                baseline_value=baseline_value,
                parameter_range=field_ranges["measurement_event_offset_db"],
                allow_missing_baseline=not is_baseline_profile,
            )

        return _HeadOption(
            option_id=f"{head.xapp_name}:{profile_name}",
            xapp_name=head.xapp_name,
            category=head.category,
            tuning_mode=head.tuning_mode,
            merge_weight=(
                (merge_weight + 0.08 if head.tuning_mode == "output_shaping" else merge_weight)
                * (0.70 + 1.60 * float(priority_weight))
            ),
            config_action=action,
            summary={
                "profile_name": profile_name,
                "intensity": float(priority_scaled_intensity),
                "priority_weight": float(priority_weight),
                "apply_scope": str(getattr(action, "apply_scope", "cell")),
                "target_cell_id": getattr(action, "target_cell_id", None),
                "tuning_mode": head.tuning_mode,
            },
        )

    def _head_option_library(
        self,
        *,
        observation: Observation,
        policy_snapshot: PolicySnapshot,
        reward_spec,
        plan,
        hard_safety_envelope=None,
    ) -> list[list[_HeadOption]]:
        parameter_priority_vector = self._effective_parameter_priority_vector(policy_snapshot)
        heads = self._range_groups(
            reward_spec,
            plan,
            hard_safety_envelope=hard_safety_envelope,
        )
        libraries: list[list[_HeadOption]] = []
        for head in heads:
            priority_weight = self._head_priority_weight(
                head=head,
                parameter_priority_vector=parameter_priority_vector,
            )
            head_options = [
                self._build_head_option(
                    head=head,
                    profile_name=profile_name,
                    intensity=intensity,
                    merge_weight=merge_weight,
                    priority_weight=priority_weight,
                    observation=observation,
                    policy_snapshot=policy_snapshot,
                )
                for profile_name, intensity, merge_weight in self._head_profiles(head)
            ]
            if len(head_options) > 0:
                libraries.append(head_options)
        return libraries

    def _ordered_head_options_for_field(
        self,
        *,
        head_options: list[_HeadOption],
        arbitration_profile,
    ) -> list[_HeadOption]:
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
                rank_by_category.get(str(head_option.category), len(rank_by_category) + 1),
                -float(head_option.merge_weight),
                str(head_option.option_id),
            ),
        )

    def _merge_scope_group(self, head_options: list[_HeadOption], *, arbitration_profile=None) -> ConfigAction:
        return self._merge_scope_group_result(
            head_options=head_options,
            arbitration_profile=arbitration_profile,
        ).merged_action

    def _merge_scope_group_result(self, head_options: list[_HeadOption], *, arbitration_profile=None):
        return merge_scope_group(
            head_options=list(head_options or []),
            controlled_fields=list(self.controlled_fields),
            numeric_fields=set(self._NUMERIC_FIELDS),
            discrete_fields=set(self._DISCRETE_FIELDS),
            dict_fields=set(self._DICT_FIELDS),
            source_name=self.name,
            arbitration_profile=arbitration_profile,
            guardrail_categories={"urllc_guard_xapp", "latency_scheduler_xapp", "ran_slicing_manager_xapp"},
        )

    def _matches_forbidden_combination(self, config_action: ConfigAction, forbidden_combination) -> bool:
        expected_scope = str(getattr(forbidden_combination, "scope", "any") or "any")
        actual_scope = str(getattr(config_action, "apply_scope", "cell") or "cell")
        if expected_scope not in {"", "any"} and expected_scope != actual_scope:
            return False
        field_values = dict(getattr(forbidden_combination, "field_values", {}) or {})
        if len(field_values) == 0:
            return False
        for field_name, expected_value in field_values.items():
            actual_value = getattr(config_action, str(field_name), None)
            if isinstance(expected_value, list):
                if actual_value not in expected_value:
                    return False
            elif actual_value != expected_value:
                return False
        return True

    def _violates_forbidden_combinations(
        self,
        merged_actions: list[ConfigAction],
        plan,
        *,
        forbidden_combinations=None,
    ) -> list[str]:
        reasons: list[str] = []
        active_forbidden_combinations = (
            list(forbidden_combinations)
            if forbidden_combinations is not None
            else list(getattr(plan, "forbidden_combinations", []) or [])
        )
        for forbidden_combination in active_forbidden_combinations:
            for config_action in merged_actions:
                if self._matches_forbidden_combination(config_action, forbidden_combination):
                    reasons.append(str(getattr(forbidden_combination, "reason", "forbidden_combination") or "forbidden_combination"))
                    break
        return reasons

    def _joint_action_distance_from_baseline(
        self,
        *,
        joint_action: _JointAction,
        observation: Observation,
        policy_snapshot: PolicySnapshot,
    ) -> float:
        distance = 0.0
        for config_action in list(joint_action.config_actions):
            for field_name in self.controlled_fields:
                candidate_value = getattr(config_action, field_name, None)
                if field_name in self._DICT_FIELDS:
                    if not isinstance(candidate_value, dict) or len(candidate_value) == 0:
                        continue
                elif candidate_value is None:
                    continue
                baseline_value = self._baseline_value_for_field(
                    field_name=field_name,
                    action=config_action,
                    observation=observation,
                    policy_snapshot=policy_snapshot,
                )
                if field_name in self._NUMERIC_FIELDS:
                    if baseline_value is None:
                        continue
                    distance += abs(float(candidate_value) - float(baseline_value))
                elif field_name in self._DICT_FIELDS:
                    baseline_dict = dict(baseline_value or {})
                    key_space = set(candidate_value.keys()) | set(baseline_dict.keys())
                    for dict_key in key_space:
                        distance += abs(
                            float(candidate_value.get(dict_key, 1.0))
                            - float(baseline_dict.get(dict_key, 1.0))
                        )
                else:
                    distance += 0.0 if candidate_value == baseline_value else 1.0
        return float(distance)

    def _safe_baseline_joint_action(
        self,
        *,
        joint_actions: list[_JointAction],
        observation: Observation,
        policy_snapshot: PolicySnapshot,
    ) -> _JointAction:
        return min(
            joint_actions,
            key=lambda joint_action: (
                self._joint_action_distance_from_baseline(
                    joint_action=joint_action,
                    observation=observation,
                    policy_snapshot=policy_snapshot,
                ),
                joint_action.joint_action_id,
            ),
        )

    def _exploration_health_status(
        self,
        observation: Observation,
        plan,
        *,
        safety_constraint_set=None,
    ) -> tuple[bool, list[str], object]:
        if not self.safety_shield_enabled:
            return True, [], None
        health_gate = None
        if safety_constraint_set is not None:
            health_gate = getattr(safety_constraint_set, "exploration_health_gate", None)
        if health_gate is None:
            health_gate = getattr(plan, "exploration_health_gate", None)
        if health_gate is None:
            return True, [], None
        reasons: list[str] = []
        max_avg_bler = getattr(health_gate, "max_avg_bler", None)
        if max_avg_bler is not None and float(getattr(observation, "avg_bler", 0.0) or 0.0) > float(max_avg_bler):
            reasons.append("avg_bler_exceeded")
        max_total_queue_bytes = getattr(health_gate, "max_total_queue_bytes", None)
        if max_total_queue_bytes is not None and int(getattr(observation, "total_queue_bytes", 0) or 0) > int(max_total_queue_bytes):
            reasons.append("queue_exceeded")
        min_total_throughput_mbps = getattr(health_gate, "min_total_throughput_mbps", None)
        if min_total_throughput_mbps is not None and float(getattr(observation, "total_throughput_mbps", 0.0) or 0.0) < float(min_total_throughput_mbps):
            reasons.append("throughput_too_low")
        max_handover_pending_ue_count = getattr(health_gate, "max_handover_pending_ue_count", None)
        if max_handover_pending_ue_count is not None and int(getattr(observation, "handover_pending_ue_count", 0) or 0) > int(max_handover_pending_ue_count):
            reasons.append("handover_pressure_high")
        qos_ok, qos_reasons = self._qos_outage_status(observation, plan)
        if not qos_ok:
            reasons.extend(list(qos_reasons))
        return len(reasons) == 0, reasons, health_gate

    def _joint_action_library(
        self,
        *,
        observation: Observation,
        policy_snapshot: PolicySnapshot,
        reward_spec=None,
        control_spec=None,
        plan,
        safety_constraint_set=None,
    ) -> list[_JointAction]:
        effective_control_spec = control_spec if control_spec is not None else reward_spec
        head_libraries = self._head_option_library(
            observation=observation,
            policy_snapshot=policy_snapshot,
            reward_spec=effective_control_spec,
            plan=plan,
            hard_safety_envelope=self._effective_hard_safety_envelope(policy_snapshot, plan),
        )
        if len(head_libraries) == 0:
            return []
        arbitration_profile = self._effective_arbitration_profile(policy_snapshot)

        joint_actions: list[_JointAction] = []
        for index, combination in enumerate(itertools.product(*head_libraries), start=1):
            selected_head_options = list(combination)
            grouped_by_scope: dict[tuple[str, str | None], list[_HeadOption]] = {}
            for head_option in selected_head_options:
                action = head_option.config_action
                key = (
                    str(getattr(action, "apply_scope", "cell") or "cell"),
                    getattr(action, "target_cell_id", None),
                )
                grouped_by_scope.setdefault(key, []).append(head_option)
            merged_actions: list[ConfigAction] = []
            arbitration_signals: list[dict[str, object]] = []
            for _, group_options in sorted(grouped_by_scope.items(), key=lambda item: (item[0][0], str(item[0][1]))):
                fusion_result = self._merge_scope_group_result(
                    group_options,
                    arbitration_profile=arbitration_profile,
                )
                merged_actions.append(fusion_result.merged_action)
                arbitration_signals.extend(
                    [
                        {
                            "field_name": str(getattr(signal, "field_name", "") or ""),
                            "arbitration_mode": str(getattr(signal, "arbitration_mode", "") or ""),
                            "omega": dict(getattr(signal, "omega", {}) or {}),
                            "selected_source": str(getattr(signal, "selected_source", "") or ""),
                            "candidate_values": list(getattr(signal, "candidate_values", []) or []),
                        }
                        for signal in list(getattr(fusion_result, "signals", []) or [])
                    ]
                )
            forbidden_reasons = self._violates_forbidden_combinations(
                merged_actions,
                plan,
                forbidden_combinations=self._effective_forbidden_combinations(
                    policy_snapshot,
                    plan,
                ),
            )
            if len(forbidden_reasons) > 0:
                continue
            joint_action_id = "||".join(head_option.option_id for head_option in selected_head_options)
            joint_actions.append(
                _JointAction(
                    joint_action_id=joint_action_id,
                    config_actions=merged_actions,
                    head_options=selected_head_options,
                    merged_summary={
                        "head_count": len(selected_head_options),
                        "scope_target_count": len(merged_actions),
                        "head_option_ids": [head_option.option_id for head_option in selected_head_options],
                        "composition_mode": str(getattr(arbitration_profile, "composition_mode", "") or ""),
                        "arbitration_mode": str(getattr(arbitration_profile, "arbitration_mode", "") or ""),
                        "arbitration_signals": arbitration_signals,
                    },
                )
            )
            if index >= self.max_joint_actions:
                break
        return joint_actions

    def _ensure_state_row(self, state_key: str, action_ids: list[str]) -> dict[str, float]:
        row = self._q_table.setdefault(state_key, {})
        for action_id in action_ids:
            row.setdefault(action_id, 0.0)
        return row

    def _select_joint_action(
        self,
        *,
        state_key: str,
        joint_actions: list[_JointAction],
    ) -> tuple[_JointAction, bool]:
        action_ids = [joint_action.joint_action_id for joint_action in joint_actions]
        q_row = self._ensure_state_row(state_key, action_ids)
        if self._rng.random() < self.epsilon:
            return self._rng.choice(joint_actions), True
        best_action_id = max(
            action_ids,
            key=lambda candidate_id: (
                float(q_row.get(candidate_id, 0.0)),
                -int(self._action_visit_counts.get(candidate_id, 0)),
            ),
        )
        for joint_action in joint_actions:
            if joint_action.joint_action_id == best_action_id:
                return joint_action, False
        return joint_actions[0], False

    def _update_q_value(
        self,
        *,
        previous_state_key: str,
        previous_action_id: str,
        reward_value: float,
        next_state_key: str,
        next_action_ids: list[str],
    ) -> None:
        previous_row = self._ensure_state_row(previous_state_key, [previous_action_id])
        next_row = self._ensure_state_row(next_state_key, next_action_ids)
        next_best = max([float(next_row.get(action_id, 0.0)) for action_id in next_action_ids], default=0.0)
        current_q = float(previous_row.get(previous_action_id, 0.0))
        td_target = float(reward_value) + self.discount_factor * float(next_best)
        previous_row[previous_action_id] = current_q + self.learning_rate * (td_target - current_q)

    def _active_parameter_mask(self, control_spec) -> dict[str, int]:
        ambient_parameter_space = getattr(control_spec, "ambient_parameter_space", None)
        ambient_mask = dict(getattr(ambient_parameter_space, "active_parameter_mask", {}) or {})
        if len(ambient_mask) > 0:
            return {
                str(field_name): int(flag)
                for field_name, flag in ambient_mask.items()
                if str(field_name).strip() != "" and int(flag) > 0
            }
        active_mask: dict[str, int] = {}
        parameter_ranges = list(
            getattr(control_spec, "guided_action_region", None)
            or getattr(control_spec, "parameter_ranges", [])
            or []
        )
        for parameter_range in parameter_ranges:
            field_name = str(getattr(parameter_range, "field_name", "") or "").strip()
            if field_name == "":
                continue
            active_mask[field_name] = 1
        return active_mask

    def _theta_runtime_state(self, joint_action: _JointAction) -> ThetaRuntimeState:
        theta_values: dict[str, object] = {}
        for config_action in list(getattr(joint_action, "config_actions", []) or []):
            for field_name in self.controlled_fields:
                value = getattr(config_action, field_name, None)
                if value is None:
                    continue
                theta_values[str(field_name)] = (
                    dict(value) if isinstance(value, dict) else value
                )
        return ThetaRuntimeState(
            state_id=f"theta:{str(getattr(joint_action, 'joint_action_id', '') or '')}",
            values=theta_values,
            dimension_count=int(len(theta_values)),
        )

    def _omega_runtime_state(self, joint_action: _JointAction) -> OmegaRuntimeState:
        field_states: list[OmegaRuntimeFieldState] = []
        for signal in list(joint_action.merged_summary.get("arbitration_signals", []) or []):
            field_name = str(signal.get("field_name", "") or "").strip()
            if field_name == "":
                continue
            field_states.append(
                OmegaRuntimeFieldState(
                    field_name=field_name,
                    arbitration_mode=str(signal.get("arbitration_mode", "weighted_merge") or "weighted_merge"),
                    weights={
                        str(option_id): float(weight)
                        for option_id, weight in dict(signal.get("omega", {}) or {}).items()
                    },
                )
            )
        return OmegaRuntimeState(
            state_id=f"omega:{str(getattr(joint_action, 'joint_action_id', '') or '')}",
            field_states=field_states,
            field_count=int(len(field_states)),
        )

    def _near_rt_runtime_state(self, control_spec, joint_action: _JointAction) -> NearRtRuntimeState:
        active_parameter_mask = self._active_parameter_mask(control_spec)
        theta_state = self._theta_runtime_state(joint_action)
        omega_state = self._omega_runtime_state(joint_action)
        return NearRtRuntimeState(
            runtime_id=f"runtime:{str(getattr(joint_action, 'joint_action_id', '') or '')}",
            active_parameter_mask=dict(active_parameter_mask),
            active_parameter_count=int(sum(active_parameter_mask.values())),
            theta_state=theta_state,
            omega_state=omega_state,
        )

    def run(
        self,
        observation: Observation,
        policy_snapshot: PolicySnapshot,
    ) -> ControlDecision:
        guidance = self._active_rl_guidance(policy_snapshot)
        reward_spec = getattr(policy_snapshot, "reward_spec", None)
        goal = getattr(policy_snapshot, "structured_goal", None)
        plan = getattr(policy_snapshot, "xapp_selection_plan", None)
        control_spec = guidance if guidance is not None else reward_spec
        if control_spec is None or goal is None or plan is None:
            return ControlDecision(config_action=ConfigAction(source_name=self.name))

        current_plan_id = str(getattr(plan, "plan_id", "") or "")
        if current_plan_id == "":
            return ControlDecision(config_action=ConfigAction(source_name=self.name))
        if self._last_plan_id is not None and self._last_plan_id != current_plan_id:
            self._exploration_freeze_remaining = 0

        safety_constraint_set = self._effective_safety_constraint_set(policy_snapshot, plan)
        hard_safety_envelope = self._effective_hard_safety_envelope(policy_snapshot, plan)
        active_forbidden_combinations = self._effective_forbidden_combinations(
            policy_snapshot,
            plan,
        )
        active_black_box_xapps = self._effective_black_box_xapps(policy_snapshot)
        parameter_priority_vector = self._effective_parameter_priority_vector(policy_snapshot)
        arbitration_profile = self._effective_arbitration_profile(policy_snapshot)

        joint_actions = self._joint_action_library(
            observation=observation,
            policy_snapshot=policy_snapshot,
            control_spec=control_spec,
            plan=plan,
            safety_constraint_set=safety_constraint_set,
        )
        if len(joint_actions) == 0:
            return ControlDecision(config_action=ConfigAction(source_name=self.name))

        state_key = self._state_key(observation, policy_snapshot, current_plan_id)
        reward_value = self._reward(observation, policy_snapshot)
        joint_action_ids = [joint_action.joint_action_id for joint_action in joint_actions]
        health_is_good, health_reasons, health_gate = self._exploration_health_status(
            observation,
            plan,
            safety_constraint_set=safety_constraint_set,
        )

        if (
            self._last_state_key is not None
            and self._last_joint_action_id is not None
            and self._last_plan_id == current_plan_id
        ):
            self._update_q_value(
                previous_state_key=self._last_state_key,
                previous_action_id=self._last_joint_action_id,
                reward_value=reward_value,
                next_state_key=state_key,
                next_action_ids=joint_action_ids,
            )

        if self.safety_shield_enabled and not health_is_good:
            freeze_slots = int(getattr(health_gate, "freeze_slots_on_violation", 4) or 4) if health_gate is not None else 4
            self._exploration_freeze_remaining = max(self._exploration_freeze_remaining, freeze_slots)
        elif self.safety_shield_enabled and self._exploration_freeze_remaining > 0:
            self._exploration_freeze_remaining = max(0, self._exploration_freeze_remaining - 1)
        else:
            self._exploration_freeze_remaining = 0

        exploration_allowed = health_is_good and self._exploration_freeze_remaining == 0
        selection_mode = "epsilon_greedy"
        if exploration_allowed:
            selected_joint_action, explored = self._select_joint_action(
                state_key=state_key,
                joint_actions=joint_actions,
            )
        else:
            selected_joint_action = self._safe_baseline_joint_action(
                joint_actions=joint_actions,
                observation=observation,
                policy_snapshot=policy_snapshot,
            )
            explored = False
            selection_mode = "safe_baseline_fallback"
        self._action_visit_counts[selected_joint_action.joint_action_id] = int(
            self._action_visit_counts.get(selected_joint_action.joint_action_id, 0)
        ) + 1
        self._last_state_key = state_key
        self._last_joint_action_id = selected_joint_action.joint_action_id
        self._last_plan_id = current_plan_id
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        q_row = self._ensure_state_row(state_key, joint_action_ids)
        near_rt_runtime_state = self._near_rt_runtime_state(control_spec, selected_joint_action)
        active_parameter_mask = dict(getattr(near_rt_runtime_state, "active_parameter_mask", {}) or {})
        theta_t = theta_runtime_state_payload(getattr(near_rt_runtime_state, "theta_state", None)).get(
            "values",
            {},
        )
        omega_t = {
            str(field_state.get("field_name", "") or ""): {
                str(option_id): float(weight)
                for option_id, weight in dict(field_state.get("weights", {}) or {}).items()
            }
            for field_state in list(
                omega_runtime_state_payload(getattr(near_rt_runtime_state, "omega_state", None)).get(
                    "field_states",
                    [],
                )
                or []
            )
            if str(field_state.get("field_name", "") or "").strip() != ""
        }
        dominant_slice, dominant_share = dominant_slice_id(observation)
        analysis_constraint = make_analysis_constraint(
            constraint_type="goal_parameter_tuning_rl_joint_heads",
            source_name=self.name,
            scope="cluster",
            target_id=None,
            params={
                "goal_id": str(getattr(goal, "goal_id", "")),
                "selection_plan_id": current_plan_id,
                "selected_xapps": list(getattr(plan, "selected_xapps", []) or []),
                "guidance_mode": "rl_guidance_contract" if guidance is not None else "legacy_reward_spec",
                "guidance_id": str(getattr(guidance, "guidance_id", "") or ""),
                "guidance_source_epoch": getattr(guidance, "source_epoch", None) if guidance is not None else None,
                "guidance_fallback_used": bool(getattr(guidance, "fallback_used", False)) if guidance is not None else False,
                "parameter_priority_vector": dict(parameter_priority_vector),
                "composition_profile_name": str(getattr(arbitration_profile, "profile_name", "") or ""),
                "composition_mode": str(getattr(arbitration_profile, "composition_mode", "") or ""),
                "arbitration_mode": str(getattr(arbitration_profile, "arbitration_mode", "") or ""),
                "override_order": list(getattr(arbitration_profile, "override_order", []) or []),
                "black_box_xapps": list(active_black_box_xapps),
                "safety_shield_enabled": bool(self.safety_shield_enabled),
                "hard_safety_envelope_count": len(list(hard_safety_envelope or [])),
                "forbidden_combination_count": len(list(active_forbidden_combinations or [])),
                "selected_safety_rule_ids": [
                    str(getattr(rule_ref, "rule_id", "") or "")
                    for rule_ref in list(
                        getattr(safety_constraint_set, "selected_rule_refs", []) or []
                    )
                ]
                if safety_constraint_set is not None
                else [],
                "qos_profile_id": str(
                    getattr(getattr(plan, "qos_protection_profile", None), "profile_id", "")
                    or ""
                ),
                "qos_constraint_count": int(
                    len(
                        list(
                            getattr(
                                getattr(plan, "qos_protection_profile", None),
                                "constraints",
                                [],
                            )
                            or []
                        )
                    )
                ),
                "state_key": state_key,
                "joint_action_id": selected_joint_action.joint_action_id,
                "explored": bool(explored),
                "exploration_allowed": bool(exploration_allowed),
                "selection_mode": selection_mode,
                "health_gate_block_reasons": list(health_reasons),
                "freeze_remaining_slots": int(self._exploration_freeze_remaining),
                "reward_value": float(reward_value),
                "epsilon": float(self.epsilon),
                "theta_t": dict(theta_t),
                "active_parameter_mask": dict(active_parameter_mask),
                "active_parameter_count": int(sum(active_parameter_mask.values())),
                "omega_t": dict(omega_t),
                "near_rt_runtime_state": near_rt_runtime_state_payload(near_rt_runtime_state),
                "dominant_slice_id": str(dominant_slice),
                "dominant_slice_share": float(dominant_share),
                "head_options": [
                    {
                        "xapp_name": head_option.xapp_name,
                        "category": head_option.category,
                        "tuning_mode": head_option.tuning_mode,
                        "option_id": head_option.option_id,
                        "summary": dict(head_option.summary),
                    }
                    for head_option in selected_joint_action.head_options
                ],
                "merged_action_targets": [
                    {
                        "apply_scope": str(getattr(config_action, "apply_scope", "cell") or "cell"),
                        "target_cell_id": getattr(config_action, "target_cell_id", None),
                        "scheduler_policy_hint": getattr(config_action, "scheduler_policy_hint", None),
                        "latency_bias": getattr(config_action, "latency_bias", None),
                        "slice_weight_adjustment": dict(getattr(config_action, "slice_weight_adjustment", {}) or {}),
                        "beam_candidate_preference": getattr(config_action, "beam_candidate_preference", None),
                        "power_cap_dbm": getattr(config_action, "power_cap_dbm", None),
                        "cell_sleep_mode": getattr(config_action, "cell_sleep_mode", None),
                        "traffic_steering_preference": getattr(config_action, "traffic_steering_preference", None),
                        "mobility_bias_db": getattr(config_action, "mobility_bias_db", None),
                        "steering_hysteresis_db": getattr(config_action, "steering_hysteresis_db", None),
                        "measurement_event_offset_db": getattr(config_action, "measurement_event_offset_db", None),
                    }
                    for config_action in selected_joint_action.config_actions
                ],
                "arbitration_signals": list(
                    selected_joint_action.merged_summary.get("arbitration_signals", [])
                ),
                "q_values": {
                    action_id: float(q_row.get(action_id, 0.0))
                    for action_id in joint_action_ids
                },
            },
        )
        primary_action = (
            selected_joint_action.config_actions[0]
            if len(selected_joint_action.config_actions) > 0
            else ConfigAction(source_name=self.name)
        )
        return ControlDecision(
            config_action=primary_action,
            config_actions=list(selected_joint_action.config_actions),
            policy_constraints=[analysis_constraint],
        )
